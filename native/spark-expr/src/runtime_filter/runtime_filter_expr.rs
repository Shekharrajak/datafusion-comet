// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Unified Runtime Filter Expression
//!
//! This module provides a unified expression that can hold any type of runtime filter
//! (IN, MinMax, or Bloom) and apply it during query execution.

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::common::{Result as DFResult, ScalarValue};
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::PhysicalExpr;

use super::filter_type::RuntimeFilterType;
use super::in_filter::{InFilter, InFilterExpr};
use super::min_max_filter::{MinMaxFilter, MinMaxFilterExpr};

/// A unified runtime filter that can hold any filter type
#[derive(Debug, Clone)]
pub enum RuntimeFilter {
    /// IN filter for exact matches
    In(InFilter),
    /// Min/Max filter for range filtering
    MinMax(MinMaxFilter),
    /// Bloom filter (uses existing BloomFilterMightContain)
    Bloom(Vec<u8>),
}

impl RuntimeFilter {
    /// Get the filter type
    pub fn filter_type(&self) -> RuntimeFilterType {
        match self {
            RuntimeFilter::In(_) => RuntimeFilterType::In,
            RuntimeFilter::MinMax(_) => RuntimeFilterType::MinMax,
            RuntimeFilter::Bloom(_) => RuntimeFilterType::Bloom,
        }
    }

    /// Check if the filter can have false positives
    pub fn can_have_false_positives(&self) -> bool {
        matches!(self, RuntimeFilter::Bloom(_))
    }

    /// Serialize the filter to bytes
    pub fn to_bytes(&self) -> super::error::RuntimeFilterResult<Vec<u8>> {
        let mut bytes = Vec::new();

        match self {
            RuntimeFilter::In(filter) => {
                bytes.push(0u8); // type marker
                bytes.extend(filter.to_bytes()?);
            }
            RuntimeFilter::MinMax(filter) => {
                bytes.push(1u8); // type marker
                bytes.extend(filter.to_bytes()?);
            }
            RuntimeFilter::Bloom(bloom_bytes) => {
                bytes.push(2u8); // type marker
                let len = bloom_bytes.len() as u32;
                bytes.extend_from_slice(&len.to_le_bytes());
                bytes.extend(bloom_bytes);
            }
        }

        Ok(bytes)
    }

    /// Deserialize the filter from bytes
    pub fn from_bytes(bytes: &[u8], max_capacity: usize) -> super::error::RuntimeFilterResult<Self> {
        if bytes.is_empty() {
            return Err(super::error::RuntimeFilterError::SerializationError {
                message: "Empty byte array".to_string(),
            });
        }

        let filter_type = bytes[0];

        match filter_type {
            0 => {
                let filter = InFilter::from_bytes(&bytes[1..], max_capacity)?;
                Ok(RuntimeFilter::In(filter))
            }
            1 => {
                let filter = MinMaxFilter::from_bytes(&bytes[1..])?;
                Ok(RuntimeFilter::MinMax(filter))
            }
            2 => {
                let len = u32::from_le_bytes(bytes[1..5].try_into().map_err(|_| {
                    super::error::RuntimeFilterError::SerializationError {
                        message: "Failed to read bloom filter length".to_string(),
                    }
                })?) as usize;
                let bloom_bytes = bytes[5..5 + len].to_vec();
                Ok(RuntimeFilter::Bloom(bloom_bytes))
            }
            _ => Err(super::error::RuntimeFilterError::SerializationError {
                message: format!("Unknown filter type: {}", filter_type),
            }),
        }
    }
}

/// A physical expression that applies a runtime filter to a column
#[derive(Debug)]
pub struct RuntimeFilterExpr {
    /// The column expression to filter
    child: Arc<dyn PhysicalExpr>,
    /// The runtime filter to apply
    filter: RuntimeFilter,
    /// Unique identifier for this filter (for metrics/debugging)
    filter_id: String,
}

impl RuntimeFilterExpr {
    /// Create a new runtime filter expression
    pub fn new(child: Arc<dyn PhysicalExpr>, filter: RuntimeFilter, filter_id: String) -> Self {
        Self {
            child,
            filter,
            filter_id,
        }
    }

    /// Create an IN filter expression
    pub fn new_in_filter(
        child: Arc<dyn PhysicalExpr>,
        filter: InFilter,
        filter_id: String,
    ) -> Self {
        Self::new(child, RuntimeFilter::In(filter), filter_id)
    }

    /// Create a MinMax filter expression
    pub fn new_min_max_filter(
        child: Arc<dyn PhysicalExpr>,
        filter: MinMaxFilter,
        filter_id: String,
    ) -> Self {
        Self::new(child, RuntimeFilter::MinMax(filter), filter_id)
    }

    /// Create a Bloom filter expression
    pub fn new_bloom_filter(
        child: Arc<dyn PhysicalExpr>,
        bloom_bytes: Vec<u8>,
        filter_id: String,
    ) -> Self {
        Self::new(child, RuntimeFilter::Bloom(bloom_bytes), filter_id)
    }

    /// Get the filter type
    pub fn filter_type(&self) -> RuntimeFilterType {
        self.filter.filter_type()
    }

    /// Get the filter ID
    pub fn filter_id(&self) -> &str {
        &self.filter_id
    }
}

impl Clone for RuntimeFilterExpr {
    fn clone(&self) -> Self {
        Self {
            child: Arc::clone(&self.child),
            filter: self.filter.clone(),
            filter_id: self.filter_id.clone(),
        }
    }
}

impl Display for RuntimeFilterExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RuntimeFilter[{}](type={}, child={})",
            self.filter_id,
            self.filter.filter_type(),
            self.child
        )
    }
}

impl PartialEq for RuntimeFilterExpr {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child) && self.filter_id == other.filter_id
    }
}

impl Eq for RuntimeFilterExpr {}

impl PartialEq<dyn Any> for RuntimeFilterExpr {
    fn eq(&self, other: &dyn Any) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.child.eq(&other.child) && self.filter_id == other.filter_id
        } else {
            false
        }
    }
}

impl Hash for RuntimeFilterExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.filter_id.hash(state);
    }
}

impl PhysicalExpr for RuntimeFilterExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> DFResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> DFResult<bool> {
        Ok(true)
    }

    fn evaluate(&self, batch: &RecordBatch) -> DFResult<ColumnarValue> {
        match &self.filter {
            RuntimeFilter::In(in_filter) => {
                let expr = InFilterExpr::new(Arc::clone(&self.child), in_filter.clone());
                expr.evaluate(batch)
            }
            RuntimeFilter::MinMax(mm_filter) => {
                let expr = MinMaxFilterExpr::new(Arc::clone(&self.child), mm_filter.clone());
                expr.evaluate(batch)
            }
            RuntimeFilter::Bloom(_bloom_bytes) => {
                // For Bloom filter, delegate to the existing BloomFilterMightContain
                // This is a placeholder - in practice, we'd use the existing implementation
                Err(datafusion::common::DataFusionError::NotImplemented(
                    "Bloom filter evaluation should use BloomFilterMightContain".to_string(),
                ))
            }
        }
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.child]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DFResult<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(RuntimeFilterExpr::new(
            Arc::clone(&children[0]),
            self.filter.clone(),
            self.filter_id.clone(),
        )))
    }

    fn fmt_sql(&self, _: &mut Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

/// Builder for creating runtime filters with optimal type selection
pub struct RuntimeFilterBuilder {
    /// Configuration for filter creation
    in_filter_threshold: usize,
    /// Whether to prefer MinMax for numeric types
    prefer_min_max_for_numeric: bool,
}

impl RuntimeFilterBuilder {
    pub fn new() -> Self {
        Self {
            in_filter_threshold: super::DEFAULT_IN_FILTER_THRESHOLD,
            prefer_min_max_for_numeric: true,
        }
    }

    pub fn with_in_filter_threshold(mut self, threshold: usize) -> Self {
        self.in_filter_threshold = threshold;
        self
    }

    pub fn with_prefer_min_max(mut self, prefer: bool) -> Self {
        self.prefer_min_max_for_numeric = prefer;
        self
    }

    /// Build an IN filter from Int64 values
    pub fn build_in_filter_int64(
        &self,
        values: Vec<i64>,
    ) -> super::error::RuntimeFilterResult<RuntimeFilter> {
        let filter = InFilter::from_int64_values(values, self.in_filter_threshold)?;
        Ok(RuntimeFilter::In(filter))
    }

    /// Build a MinMax filter from Int64 bounds
    pub fn build_min_max_filter_int64(
        &self,
        min: i64,
        max: i64,
    ) -> super::error::RuntimeFilterResult<RuntimeFilter> {
        let filter = MinMaxFilter::with_bounds(
            ScalarValue::Int64(Some(min)),
            ScalarValue::Int64(Some(max)),
        )?;
        Ok(RuntimeFilter::MinMax(filter))
    }

    /// Automatically select and build the optimal filter type
    pub fn build_optimal_filter_int64(
        &self,
        values: Vec<i64>,
    ) -> super::error::RuntimeFilterResult<RuntimeFilter> {
        let cardinality = values.len();

        // For numeric types with bounded range, MinMax is often better
        if self.prefer_min_max_for_numeric && cardinality > 1 {
            let min = *values.iter().min().unwrap();
            let max = *values.iter().max().unwrap();

            // If the range is reasonable relative to cardinality, use MinMax
            let range = (max - min) as f64;
            let density = cardinality as f64 / range.max(1.0);

            // High density (values close together) -> MinMax is good
            if density > 0.1 || cardinality > self.in_filter_threshold {
                return self.build_min_max_filter_int64(min, max);
            }
        }

        // Otherwise use IN filter for exact matches
        if cardinality <= self.in_filter_threshold {
            self.build_in_filter_int64(values)
        } else {
            // Fall back to Bloom filter for large cardinality
            // This would integrate with the existing Bloom filter implementation
            Err(super::error::RuntimeFilterError::Internal {
                message: "Bloom filter creation should use BloomFilterAgg".to_string(),
            })
        }
    }
}

impl Default for RuntimeFilterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Int64Array};
    use datafusion::physical_expr::expressions::Column;

    #[test]
    fn test_runtime_filter_in() {
        let filter = InFilter::from_int64_values(vec![1, 2, 3], 1000).unwrap();
        let rf = RuntimeFilter::In(filter);

        assert_eq!(rf.filter_type(), RuntimeFilterType::In);
        assert!(!rf.can_have_false_positives());
    }

    #[test]
    fn test_runtime_filter_min_max() {
        let filter = MinMaxFilter::with_bounds(
            ScalarValue::Int64(Some(10)),
            ScalarValue::Int64(Some(100)),
        )
        .unwrap();
        let rf = RuntimeFilter::MinMax(filter);

        assert_eq!(rf.filter_type(), RuntimeFilterType::MinMax);
        assert!(!rf.can_have_false_positives());
    }

    #[test]
    fn test_runtime_filter_bloom() {
        let rf = RuntimeFilter::Bloom(vec![0u8; 100]);

        assert_eq!(rf.filter_type(), RuntimeFilterType::Bloom);
        assert!(rf.can_have_false_positives());
    }

    #[test]
    fn test_runtime_filter_serialization() {
        let filter = InFilter::from_int64_values(vec![1, 2, 3], 1000).unwrap();
        let rf = RuntimeFilter::In(filter);

        let bytes = rf.to_bytes().unwrap();
        let restored = RuntimeFilter::from_bytes(&bytes, 1000).unwrap();

        assert_eq!(rf.filter_type(), restored.filter_type());
    }

    #[test]
    fn test_runtime_filter_builder() {
        let builder = RuntimeFilterBuilder::new();

        // Small cardinality -> IN filter
        let filter = builder.build_in_filter_int64(vec![1, 2, 3]).unwrap();
        assert_eq!(filter.filter_type(), RuntimeFilterType::In);

        // MinMax filter
        let filter = builder.build_min_max_filter_int64(10, 100).unwrap();
        assert_eq!(filter.filter_type(), RuntimeFilterType::MinMax);
    }

    #[test]
    fn test_runtime_filter_expr() {
        let schema = Schema::new(vec![arrow::datatypes::Field::new(
            "col",
            DataType::Int64,
            false,
        )]);

        let col_expr = Arc::new(Column::new("col", 0)) as Arc<dyn PhysicalExpr>;
        let filter = InFilter::from_int64_values(vec![1, 2, 3], 1000).unwrap();
        let expr = RuntimeFilterExpr::new_in_filter(col_expr, filter, "test_filter".to_string());

        assert_eq!(expr.filter_type(), RuntimeFilterType::In);
        assert_eq!(expr.filter_id(), "test_filter");

        // Test evaluation
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int64Array::from(vec![1, 4, 2, 5, 3])) as ArrayRef],
        )
        .unwrap();

        let result = expr.evaluate(&batch).unwrap();
        match result {
            ColumnarValue::Array(arr) => {
                let bool_arr = arr
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .unwrap();
                assert_eq!(bool_arr.value(0), true); // 1 in filter
                assert_eq!(bool_arr.value(1), false); // 4 not in filter
                assert_eq!(bool_arr.value(2), true); // 2 in filter
                assert_eq!(bool_arr.value(3), false); // 5 not in filter
                assert_eq!(bool_arr.value(4), true); // 3 in filter
            }
            _ => panic!("Expected array result"),
        }
    }
}
