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

//! Min/Max Filter implementation for runtime filtering
//!
//! The Min/Max filter is optimal for numeric and date types where the join key
//! has a bounded range. It provides zero false positives and extremely fast
//! two-comparison lookups.

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Float64Array, Int32Array, Int64Array,
};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::common::{Result as DFResult, ScalarValue};
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::PhysicalExpr;

use super::error::{RuntimeFilterError, RuntimeFilterResult};

/// A runtime Min/Max filter that holds minimum and maximum bounds
#[derive(Debug, Clone, PartialEq)]
pub struct MinMaxFilter {
    /// Minimum value (inclusive)
    min: ScalarValue,
    /// Maximum value (inclusive)
    max: ScalarValue,
    /// Data type of the filter
    data_type: DataType,
    /// Whether the filter has been initialized with at least one value
    initialized: bool,
}

impl Eq for MinMaxFilter {}

impl Hash for MinMaxFilter {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data_type.hash(state);
        self.initialized.hash(state);
        // Hash string representations of min/max as proxy
        format!("{:?}", self.min).hash(state);
        format!("{:?}", self.max).hash(state);
    }
}

impl MinMaxFilter {
    /// Create a new uninitialized Min/Max filter for the given data type
    pub fn new(data_type: DataType) -> RuntimeFilterResult<Self> {
        let (min, max) = Self::initial_bounds(&data_type)?;
        Ok(Self {
            min,
            max,
            data_type,
            initialized: false,
        })
    }

    /// Create a Min/Max filter with explicit bounds
    pub fn with_bounds(min: ScalarValue, max: ScalarValue) -> RuntimeFilterResult<Self> {
        let data_type = min.data_type();
        if data_type != max.data_type() {
            return Err(RuntimeFilterError::TypeMismatch {
                expected: format!("{:?}", data_type),
                actual: format!("{:?}", max.data_type()),
            });
        }

        Ok(Self {
            min,
            max,
            data_type,
            initialized: true,
        })
    }

    /// Get initial bounds for a data type (inverted for update logic)
    fn initial_bounds(data_type: &DataType) -> RuntimeFilterResult<(ScalarValue, ScalarValue)> {
        match data_type {
            DataType::Int32 => Ok((
                ScalarValue::Int32(Some(i32::MAX)),
                ScalarValue::Int32(Some(i32::MIN)),
            )),
            DataType::Int64 => Ok((
                ScalarValue::Int64(Some(i64::MAX)),
                ScalarValue::Int64(Some(i64::MIN)),
            )),
            DataType::Float64 => Ok((
                ScalarValue::Float64(Some(f64::INFINITY)),
                ScalarValue::Float64(Some(f64::NEG_INFINITY)),
            )),
            DataType::Date32 => Ok((
                ScalarValue::Date32(Some(i32::MAX)),
                ScalarValue::Date32(Some(i32::MIN)),
            )),
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, tz) => Ok((
                ScalarValue::TimestampMicrosecond(Some(i64::MAX), tz.clone()),
                ScalarValue::TimestampMicrosecond(Some(i64::MIN), tz.clone()),
            )),
            _ => Err(RuntimeFilterError::TypeMismatch {
                expected: "Numeric or temporal type".to_string(),
                actual: format!("{:?}", data_type),
            }),
        }
    }

    /// Update the filter with a new Int64 value
    pub fn update_int64(&mut self, value: i64) -> RuntimeFilterResult<()> {
        match (&mut self.min, &mut self.max) {
            (ScalarValue::Int64(Some(min)), ScalarValue::Int64(Some(max))) => {
                if value < *min {
                    *min = value;
                }
                if value > *max {
                    *max = value;
                }
                self.initialized = true;
                Ok(())
            }
            _ => Err(RuntimeFilterError::TypeMismatch {
                expected: "Int64".to_string(),
                actual: format!("{:?}", self.data_type),
            }),
        }
    }

    /// Update the filter with a new Int32 value
    pub fn update_int32(&mut self, value: i32) -> RuntimeFilterResult<()> {
        match (&mut self.min, &mut self.max) {
            (ScalarValue::Int32(Some(min)), ScalarValue::Int32(Some(max))) => {
                if value < *min {
                    *min = value;
                }
                if value > *max {
                    *max = value;
                }
                self.initialized = true;
                Ok(())
            }
            _ => Err(RuntimeFilterError::TypeMismatch {
                expected: "Int32".to_string(),
                actual: format!("{:?}", self.data_type),
            }),
        }
    }

    /// Update the filter with a new Float64 value
    pub fn update_float64(&mut self, value: f64) -> RuntimeFilterResult<()> {
        match (&mut self.min, &mut self.max) {
            (ScalarValue::Float64(Some(min)), ScalarValue::Float64(Some(max))) => {
                if value < *min {
                    *min = value;
                }
                if value > *max {
                    *max = value;
                }
                self.initialized = true;
                Ok(())
            }
            _ => Err(RuntimeFilterError::TypeMismatch {
                expected: "Float64".to_string(),
                actual: format!("{:?}", self.data_type),
            }),
        }
    }

    /// Update the filter with a new Date32 value
    pub fn update_date32(&mut self, value: i32) -> RuntimeFilterResult<()> {
        match (&mut self.min, &mut self.max) {
            (ScalarValue::Date32(Some(min)), ScalarValue::Date32(Some(max))) => {
                if value < *min {
                    *min = value;
                }
                if value > *max {
                    *max = value;
                }
                self.initialized = true;
                Ok(())
            }
            _ => Err(RuntimeFilterError::TypeMismatch {
                expected: "Date32".to_string(),
                actual: format!("{:?}", self.data_type),
            }),
        }
    }

    /// Check if a value is within the filter's bounds
    pub fn contains_int64(&self, value: i64) -> bool {
        if !self.initialized {
            return true; // Uninitialized filter passes all values
        }
        match (&self.min, &self.max) {
            (ScalarValue::Int64(Some(min)), ScalarValue::Int64(Some(max))) => {
                value >= *min && value <= *max
            }
            _ => true,
        }
    }

    /// Check if a value is within the filter's bounds
    pub fn contains_int32(&self, value: i32) -> bool {
        if !self.initialized {
            return true;
        }
        match (&self.min, &self.max) {
            (ScalarValue::Int32(Some(min)), ScalarValue::Int32(Some(max))) => {
                value >= *min && value <= *max
            }
            _ => true,
        }
    }

    /// Check if a value is within the filter's bounds
    pub fn contains_float64(&self, value: f64) -> bool {
        if !self.initialized {
            return true;
        }
        match (&self.min, &self.max) {
            (ScalarValue::Float64(Some(min)), ScalarValue::Float64(Some(max))) => {
                value >= *min && value <= *max
            }
            _ => true,
        }
    }

    /// Check if a value is within the filter's bounds
    pub fn contains_date32(&self, value: i32) -> bool {
        if !self.initialized {
            return true;
        }
        match (&self.min, &self.max) {
            (ScalarValue::Date32(Some(min)), ScalarValue::Date32(Some(max))) => {
                value >= *min && value <= *max
            }
            _ => true,
        }
    }

    /// Get the minimum value
    pub fn min(&self) -> &ScalarValue {
        &self.min
    }

    /// Get the maximum value
    pub fn max(&self) -> &ScalarValue {
        &self.max
    }

    /// Get the data type
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Check if the filter is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Apply the filter to an Int64 array, returning a boolean mask
    pub fn filter_int64_array(&self, array: &Int64Array) -> BooleanArray {
        if !self.initialized {
            return BooleanArray::from(vec![true; array.len()]);
        }

        match (&self.min, &self.max) {
            (ScalarValue::Int64(Some(min)), ScalarValue::Int64(Some(max))) => array
                .iter()
                .map(|opt_val| opt_val.map(|v| v >= *min && v <= *max))
                .collect(),
            _ => BooleanArray::from(vec![true; array.len()]),
        }
    }

    /// Apply the filter to an Int32 array, returning a boolean mask
    pub fn filter_int32_array(&self, array: &Int32Array) -> BooleanArray {
        if !self.initialized {
            return BooleanArray::from(vec![true; array.len()]);
        }

        match (&self.min, &self.max) {
            (ScalarValue::Int32(Some(min)), ScalarValue::Int32(Some(max))) => array
                .iter()
                .map(|opt_val| opt_val.map(|v| v >= *min && v <= *max))
                .collect(),
            _ => BooleanArray::from(vec![true; array.len()]),
        }
    }

    /// Apply the filter to a Float64 array, returning a boolean mask
    pub fn filter_float64_array(&self, array: &Float64Array) -> BooleanArray {
        if !self.initialized {
            return BooleanArray::from(vec![true; array.len()]);
        }

        match (&self.min, &self.max) {
            (ScalarValue::Float64(Some(min)), ScalarValue::Float64(Some(max))) => array
                .iter()
                .map(|opt_val| opt_val.map(|v| v >= *min && v <= *max))
                .collect(),
            _ => BooleanArray::from(vec![true; array.len()]),
        }
    }

    /// Apply the filter to a Date32 array, returning a boolean mask
    pub fn filter_date32_array(&self, array: &Date32Array) -> BooleanArray {
        if !self.initialized {
            return BooleanArray::from(vec![true; array.len()]);
        }

        match (&self.min, &self.max) {
            (ScalarValue::Date32(Some(min)), ScalarValue::Date32(Some(max))) => array
                .iter()
                .map(|opt_val| opt_val.map(|v| v >= *min && v <= *max))
                .collect(),
            _ => BooleanArray::from(vec![true; array.len()]),
        }
    }

    /// Merge another MinMaxFilter into this one
    pub fn merge(&mut self, other: &MinMaxFilter) -> RuntimeFilterResult<()> {
        if self.data_type != other.data_type {
            return Err(RuntimeFilterError::TypeMismatch {
                expected: format!("{:?}", self.data_type),
                actual: format!("{:?}", other.data_type),
            });
        }

        if !other.initialized {
            return Ok(());
        }

        match (&mut self.min, &mut self.max, &other.min, &other.max) {
            (
                ScalarValue::Int64(Some(self_min)),
                ScalarValue::Int64(Some(self_max)),
                ScalarValue::Int64(Some(other_min)),
                ScalarValue::Int64(Some(other_max)),
            ) => {
                if *other_min < *self_min || !self.initialized {
                    *self_min = *other_min;
                }
                if *other_max > *self_max || !self.initialized {
                    *self_max = *other_max;
                }
                self.initialized = true;
            }
            (
                ScalarValue::Int32(Some(self_min)),
                ScalarValue::Int32(Some(self_max)),
                ScalarValue::Int32(Some(other_min)),
                ScalarValue::Int32(Some(other_max)),
            ) => {
                if *other_min < *self_min || !self.initialized {
                    *self_min = *other_min;
                }
                if *other_max > *self_max || !self.initialized {
                    *self_max = *other_max;
                }
                self.initialized = true;
            }
            _ => {}
        }

        Ok(())
    }

    /// Serialize the filter to bytes for network transfer
    pub fn to_bytes(&self) -> RuntimeFilterResult<Vec<u8>> {
        if !self.initialized {
            return Err(RuntimeFilterError::NotInitialized {
                filter_type: "MinMax".to_string(),
            });
        }

        let mut bytes = Vec::new();

        // Serialize based on data type
        match (&self.min, &self.max) {
            (ScalarValue::Int64(Some(min)), ScalarValue::Int64(Some(max))) => {
                bytes.push(0u8); // type = Int64
                bytes.extend_from_slice(&min.to_le_bytes());
                bytes.extend_from_slice(&max.to_le_bytes());
            }
            (ScalarValue::Int32(Some(min)), ScalarValue::Int32(Some(max))) => {
                bytes.push(1u8); // type = Int32
                bytes.extend_from_slice(&min.to_le_bytes());
                bytes.extend_from_slice(&max.to_le_bytes());
            }
            (ScalarValue::Float64(Some(min)), ScalarValue::Float64(Some(max))) => {
                bytes.push(2u8); // type = Float64
                bytes.extend_from_slice(&min.to_le_bytes());
                bytes.extend_from_slice(&max.to_le_bytes());
            }
            (ScalarValue::Date32(Some(min)), ScalarValue::Date32(Some(max))) => {
                bytes.push(3u8); // type = Date32
                bytes.extend_from_slice(&min.to_le_bytes());
                bytes.extend_from_slice(&max.to_le_bytes());
            }
            _ => {
                return Err(RuntimeFilterError::SerializationError {
                    message: format!("Unsupported data type: {:?}", self.data_type),
                });
            }
        }

        Ok(bytes)
    }

    /// Deserialize the filter from bytes
    pub fn from_bytes(bytes: &[u8]) -> RuntimeFilterResult<Self> {
        if bytes.is_empty() {
            return Err(RuntimeFilterError::SerializationError {
                message: "Empty byte array".to_string(),
            });
        }

        let filter_type = bytes[0];

        match filter_type {
            0 => {
                // Int64
                let min = i64::from_le_bytes(bytes[1..9].try_into().map_err(|_| {
                    RuntimeFilterError::SerializationError {
                        message: "Failed to read min".to_string(),
                    }
                })?);
                let max = i64::from_le_bytes(bytes[9..17].try_into().map_err(|_| {
                    RuntimeFilterError::SerializationError {
                        message: "Failed to read max".to_string(),
                    }
                })?);
                Ok(Self {
                    min: ScalarValue::Int64(Some(min)),
                    max: ScalarValue::Int64(Some(max)),
                    data_type: DataType::Int64,
                    initialized: true,
                })
            }
            1 => {
                // Int32
                let min = i32::from_le_bytes(bytes[1..5].try_into().map_err(|_| {
                    RuntimeFilterError::SerializationError {
                        message: "Failed to read min".to_string(),
                    }
                })?);
                let max = i32::from_le_bytes(bytes[5..9].try_into().map_err(|_| {
                    RuntimeFilterError::SerializationError {
                        message: "Failed to read max".to_string(),
                    }
                })?);
                Ok(Self {
                    min: ScalarValue::Int32(Some(min)),
                    max: ScalarValue::Int32(Some(max)),
                    data_type: DataType::Int32,
                    initialized: true,
                })
            }
            2 => {
                // Float64
                let min = f64::from_le_bytes(bytes[1..9].try_into().map_err(|_| {
                    RuntimeFilterError::SerializationError {
                        message: "Failed to read min".to_string(),
                    }
                })?);
                let max = f64::from_le_bytes(bytes[9..17].try_into().map_err(|_| {
                    RuntimeFilterError::SerializationError {
                        message: "Failed to read max".to_string(),
                    }
                })?);
                Ok(Self {
                    min: ScalarValue::Float64(Some(min)),
                    max: ScalarValue::Float64(Some(max)),
                    data_type: DataType::Float64,
                    initialized: true,
                })
            }
            3 => {
                // Date32
                let min = i32::from_le_bytes(bytes[1..5].try_into().map_err(|_| {
                    RuntimeFilterError::SerializationError {
                        message: "Failed to read min".to_string(),
                    }
                })?);
                let max = i32::from_le_bytes(bytes[5..9].try_into().map_err(|_| {
                    RuntimeFilterError::SerializationError {
                        message: "Failed to read max".to_string(),
                    }
                })?);
                Ok(Self {
                    min: ScalarValue::Date32(Some(min)),
                    max: ScalarValue::Date32(Some(max)),
                    data_type: DataType::Date32,
                    initialized: true,
                })
            }
            _ => Err(RuntimeFilterError::SerializationError {
                message: format!("Unknown filter type: {}", filter_type),
            }),
        }
    }
}

/// Physical expression wrapper for Min/Max filter
#[derive(Debug, Clone)]
pub struct MinMaxFilterExpr {
    /// The column to filter
    child: Arc<dyn PhysicalExpr>,
    /// The Min/Max filter
    filter: MinMaxFilter,
}

impl MinMaxFilterExpr {
    pub fn new(child: Arc<dyn PhysicalExpr>, filter: MinMaxFilter) -> Self {
        Self { child, filter }
    }
}

impl Display for MinMaxFilterExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MinMaxFilter(min={:?}, max={:?})", self.filter.min(), self.filter.max())
    }
}

impl PartialEq for MinMaxFilterExpr {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child) && self.filter == other.filter
    }
}

impl Eq for MinMaxFilterExpr {}

impl Hash for MinMaxFilterExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.child.hash(state);
        self.filter.hash(state);
    }
}

impl PartialEq<dyn Any> for MinMaxFilterExpr {
    fn eq(&self, other: &dyn Any) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.child.eq(&other.child) && self.filter == other.filter
        } else {
            false
        }
    }
}

impl PhysicalExpr for MinMaxFilterExpr {
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
        let child_value = self.child.evaluate(batch)?;

        match child_value {
            ColumnarValue::Array(array) => {
                let result: BooleanArray = match self.filter.data_type() {
                    DataType::Int64 => {
                        let int_array = array
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or_else(|| {
                                datafusion::common::DataFusionError::Internal(
                                    "Expected Int64Array".to_string(),
                                )
                            })?;
                        self.filter.filter_int64_array(int_array)
                    }
                    DataType::Int32 => {
                        let int_array = array
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .ok_or_else(|| {
                                datafusion::common::DataFusionError::Internal(
                                    "Expected Int32Array".to_string(),
                                )
                            })?;
                        self.filter.filter_int32_array(int_array)
                    }
                    DataType::Float64 => {
                        let float_array = array
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| {
                                datafusion::common::DataFusionError::Internal(
                                    "Expected Float64Array".to_string(),
                                )
                            })?;
                        self.filter.filter_float64_array(float_array)
                    }
                    DataType::Date32 => {
                        let date_array = array
                            .as_any()
                            .downcast_ref::<Date32Array>()
                            .ok_or_else(|| {
                                datafusion::common::DataFusionError::Internal(
                                    "Expected Date32Array".to_string(),
                                )
                            })?;
                        self.filter.filter_date32_array(date_array)
                    }
                    dt => {
                        return Err(datafusion::common::DataFusionError::Internal(format!(
                            "Unsupported data type for MinMax filter: {:?}",
                            dt
                        )));
                    }
                };
                Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
            }
            ColumnarValue::Scalar(scalar) => {
                let result = match scalar {
                    ScalarValue::Int64(Some(v)) => self.filter.contains_int64(v),
                    ScalarValue::Int32(Some(v)) => self.filter.contains_int32(v),
                    ScalarValue::Float64(Some(v)) => self.filter.contains_float64(v),
                    ScalarValue::Date32(Some(v)) => self.filter.contains_date32(v),
                    _ => true,
                };
                Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(result))))
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
        Ok(Arc::new(MinMaxFilterExpr::new(
            Arc::clone(&children[0]),
            self.filter.clone(),
        )))
    }

    fn fmt_sql(&self, _: &mut Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_max_filter_int64() {
        let mut filter = MinMaxFilter::new(DataType::Int64).unwrap();

        filter.update_int64(10).unwrap();
        filter.update_int64(20).unwrap();
        filter.update_int64(5).unwrap();

        assert!(filter.is_initialized());
        assert!(filter.contains_int64(5));
        assert!(filter.contains_int64(15));
        assert!(filter.contains_int64(20));
        assert!(!filter.contains_int64(4));
        assert!(!filter.contains_int64(21));
    }

    #[test]
    fn test_min_max_filter_with_bounds() {
        let filter = MinMaxFilter::with_bounds(
            ScalarValue::Int64(Some(10)),
            ScalarValue::Int64(Some(100)),
        )
        .unwrap();

        assert!(filter.contains_int64(10));
        assert!(filter.contains_int64(50));
        assert!(filter.contains_int64(100));
        assert!(!filter.contains_int64(9));
        assert!(!filter.contains_int64(101));
    }

    #[test]
    fn test_min_max_filter_array() {
        let filter = MinMaxFilter::with_bounds(
            ScalarValue::Int64(Some(10)),
            ScalarValue::Int64(Some(20)),
        )
        .unwrap();

        let array = Int64Array::from(vec![Some(5), Some(10), Some(15), None, Some(25)]);
        let result = filter.filter_int64_array(&array);

        assert_eq!(result.value(0), false); // 5 < 10
        assert_eq!(result.value(1), true); // 10 in range
        assert_eq!(result.value(2), true); // 15 in range
        assert!(result.is_null(3)); // NULL
        assert_eq!(result.value(4), false); // 25 > 20
    }

    #[test]
    fn test_min_max_filter_merge() {
        let mut filter1 = MinMaxFilter::with_bounds(
            ScalarValue::Int64(Some(10)),
            ScalarValue::Int64(Some(20)),
        )
        .unwrap();

        let filter2 = MinMaxFilter::with_bounds(
            ScalarValue::Int64(Some(5)),
            ScalarValue::Int64(Some(25)),
        )
        .unwrap();

        filter1.merge(&filter2).unwrap();

        assert!(filter1.contains_int64(5));
        assert!(filter1.contains_int64(25));
    }

    #[test]
    fn test_min_max_filter_serialization() {
        let filter = MinMaxFilter::with_bounds(
            ScalarValue::Int64(Some(10)),
            ScalarValue::Int64(Some(100)),
        )
        .unwrap();

        let bytes = filter.to_bytes().unwrap();
        let restored = MinMaxFilter::from_bytes(&bytes).unwrap();

        assert_eq!(filter.min(), restored.min());
        assert_eq!(filter.max(), restored.max());
    }
}
