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

//! IN Filter implementation for runtime filtering
//!
//! The IN filter is optimal for small cardinality (<1000 values) joins.
//! It provides exact matches with zero false positives and O(1) lookup time.

use std::any::Any;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, Int64Array, StringArray};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::common::Result as DFResult;
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::PhysicalExpr;

use super::error::{RuntimeFilterError, RuntimeFilterResult};
use super::filter_type::RuntimeFilterType;

/// A runtime IN filter that holds a set of values for exact matching
#[derive(Clone)]
pub struct InFilter {
    /// The set of Int64 values (most common case)
    int_values: Option<HashSet<i64>>,
    /// The set of String values
    string_values: Option<HashSet<String>>,
    /// Maximum capacity before rejecting new values
    max_capacity: usize,
    /// Data type of the filter
    data_type: DataType,
}

impl Debug for InFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let count = self.cardinality();
        f.debug_struct("InFilter")
            .field("cardinality", &count)
            .field("max_capacity", &self.max_capacity)
            .field("data_type", &self.data_type)
            .finish()
    }
}

impl PartialEq for InFilter {
    fn eq(&self, other: &Self) -> bool {
        self.int_values == other.int_values
            && self.string_values == other.string_values
            && self.max_capacity == other.max_capacity
            && self.data_type == other.data_type
    }
}

impl Eq for InFilter {}

impl Hash for InFilter {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.max_capacity.hash(state);
        self.data_type.hash(state);
        // Hash cardinality as a proxy for the set contents
        self.cardinality().hash(state);
    }
}

impl InFilter {
    /// Create a new empty IN filter for Int64 values
    pub fn new_int64(max_capacity: usize) -> Self {
        Self {
            int_values: Some(HashSet::with_capacity(max_capacity.min(1000))),
            string_values: None,
            max_capacity,
            data_type: DataType::Int64,
        }
    }

    /// Create a new empty IN filter for String values
    pub fn new_string(max_capacity: usize) -> Self {
        Self {
            int_values: None,
            string_values: Some(HashSet::with_capacity(max_capacity.min(1000))),
            max_capacity,
            data_type: DataType::Utf8,
        }
    }

    /// Create an IN filter from a vector of Int64 values
    pub fn from_int64_values(values: Vec<i64>, max_capacity: usize) -> RuntimeFilterResult<Self> {
        if values.len() > max_capacity {
            return Err(RuntimeFilterError::CapacityExceeded {
                filter_type: RuntimeFilterType::In.to_string(),
                max_capacity,
                requested: values.len(),
            });
        }

        Ok(Self {
            int_values: Some(values.into_iter().collect()),
            string_values: None,
            max_capacity,
            data_type: DataType::Int64,
        })
    }

    /// Create an IN filter from a vector of String values
    pub fn from_string_values(
        values: Vec<String>,
        max_capacity: usize,
    ) -> RuntimeFilterResult<Self> {
        if values.len() > max_capacity {
            return Err(RuntimeFilterError::CapacityExceeded {
                filter_type: RuntimeFilterType::In.to_string(),
                max_capacity,
                requested: values.len(),
            });
        }

        Ok(Self {
            int_values: None,
            string_values: Some(values.into_iter().collect()),
            max_capacity,
            data_type: DataType::Utf8,
        })
    }

    /// Add an Int64 value to the filter
    pub fn add_int64(&mut self, value: i64) -> RuntimeFilterResult<()> {
        let set = self.int_values.as_mut().ok_or_else(|| {
            RuntimeFilterError::TypeMismatch {
                expected: "Int64".to_string(),
                actual: "String".to_string(),
            }
        })?;

        if set.len() >= self.max_capacity {
            return Err(RuntimeFilterError::CapacityExceeded {
                filter_type: RuntimeFilterType::In.to_string(),
                max_capacity: self.max_capacity,
                requested: set.len() + 1,
            });
        }

        set.insert(value);
        Ok(())
    }

    /// Add a String value to the filter
    pub fn add_string(&mut self, value: String) -> RuntimeFilterResult<()> {
        let set = self.string_values.as_mut().ok_or_else(|| {
            RuntimeFilterError::TypeMismatch {
                expected: "String".to_string(),
                actual: "Int64".to_string(),
            }
        })?;

        if set.len() >= self.max_capacity {
            return Err(RuntimeFilterError::CapacityExceeded {
                filter_type: RuntimeFilterType::In.to_string(),
                max_capacity: self.max_capacity,
                requested: set.len() + 1,
            });
        }

        set.insert(value);
        Ok(())
    }

    /// Check if the filter contains an Int64 value
    pub fn contains_int64(&self, value: i64) -> bool {
        self.int_values
            .as_ref()
            .map(|set| set.contains(&value))
            .unwrap_or(false)
    }

    /// Check if the filter contains a String value
    pub fn contains_string(&self, value: &str) -> bool {
        self.string_values
            .as_ref()
            .map(|set| set.contains(value))
            .unwrap_or(false)
    }

    /// Get the number of values in the filter
    pub fn cardinality(&self) -> usize {
        self.int_values
            .as_ref()
            .map(|s| s.len())
            .or_else(|| self.string_values.as_ref().map(|s| s.len()))
            .unwrap_or(0)
    }

    /// Check if the filter is empty
    pub fn is_empty(&self) -> bool {
        self.cardinality() == 0
    }

    /// Get the data type of the filter
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Apply the filter to an Int64 array, returning a boolean mask
    pub fn filter_int64_array(&self, array: &Int64Array) -> RuntimeFilterResult<BooleanArray> {
        let set = self.int_values.as_ref().ok_or_else(|| {
            RuntimeFilterError::TypeMismatch {
                expected: "Int64".to_string(),
                actual: format!("{:?}", self.data_type),
            }
        })?;

        let result: BooleanArray = array
            .iter()
            .map(|opt_val| opt_val.map(|v| set.contains(&v)))
            .collect();

        Ok(result)
    }

    /// Apply the filter to a String array, returning a boolean mask
    pub fn filter_string_array(&self, array: &StringArray) -> RuntimeFilterResult<BooleanArray> {
        let set = self.string_values.as_ref().ok_or_else(|| {
            RuntimeFilterError::TypeMismatch {
                expected: "String".to_string(),
                actual: format!("{:?}", self.data_type),
            }
        })?;

        let result: BooleanArray = array
            .iter()
            .map(|opt_val| opt_val.map(|v| set.contains(v)))
            .collect();

        Ok(result)
    }

    /// Serialize the filter to bytes for network transfer
    pub fn to_bytes(&self) -> RuntimeFilterResult<Vec<u8>> {
        // Simple serialization format:
        // [1 byte: type (0=int64, 1=string)]
        // [4 bytes: count]
        // [values...]
        let mut bytes = Vec::new();

        if let Some(int_values) = &self.int_values {
            bytes.push(0u8); // type = int64
            let count = int_values.len() as u32;
            bytes.extend_from_slice(&count.to_le_bytes());
            for v in int_values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        } else if let Some(string_values) = &self.string_values {
            bytes.push(1u8); // type = string
            let count = string_values.len() as u32;
            bytes.extend_from_slice(&count.to_le_bytes());
            for s in string_values {
                let len = s.len() as u32;
                bytes.extend_from_slice(&len.to_le_bytes());
                bytes.extend_from_slice(s.as_bytes());
            }
        } else {
            return Err(RuntimeFilterError::NotInitialized {
                filter_type: "IN".to_string(),
            });
        }

        Ok(bytes)
    }

    /// Deserialize the filter from bytes
    pub fn from_bytes(bytes: &[u8], max_capacity: usize) -> RuntimeFilterResult<Self> {
        if bytes.is_empty() {
            return Err(RuntimeFilterError::SerializationError {
                message: "Empty byte array".to_string(),
            });
        }

        let filter_type = bytes[0];
        let count = u32::from_le_bytes(bytes[1..5].try_into().map_err(|_| {
            RuntimeFilterError::SerializationError {
                message: "Failed to read count".to_string(),
            }
        })?) as usize;

        if count > max_capacity {
            return Err(RuntimeFilterError::CapacityExceeded {
                filter_type: RuntimeFilterType::In.to_string(),
                max_capacity,
                requested: count,
            });
        }

        match filter_type {
            0 => {
                // Int64
                let mut values = HashSet::with_capacity(count);
                let mut offset = 5;
                for _ in 0..count {
                    let v = i64::from_le_bytes(bytes[offset..offset + 8].try_into().map_err(
                        |_| RuntimeFilterError::SerializationError {
                            message: "Failed to read int64 value".to_string(),
                        },
                    )?);
                    values.insert(v);
                    offset += 8;
                }
                Ok(Self {
                    int_values: Some(values),
                    string_values: None,
                    max_capacity,
                    data_type: DataType::Int64,
                })
            }
            1 => {
                // String
                let mut values = HashSet::with_capacity(count);
                let mut offset = 5;
                for _ in 0..count {
                    let len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().map_err(
                        |_| RuntimeFilterError::SerializationError {
                            message: "Failed to read string length".to_string(),
                        },
                    )?) as usize;
                    offset += 4;
                    let s = String::from_utf8(bytes[offset..offset + len].to_vec()).map_err(
                        |_| RuntimeFilterError::SerializationError {
                            message: "Invalid UTF-8 string".to_string(),
                        },
                    )?;
                    values.insert(s);
                    offset += len;
                }
                Ok(Self {
                    int_values: None,
                    string_values: Some(values),
                    max_capacity,
                    data_type: DataType::Utf8,
                })
            }
            _ => Err(RuntimeFilterError::SerializationError {
                message: format!("Unknown filter type: {}", filter_type),
            }),
        }
    }
}

/// Physical expression wrapper for IN filter
#[derive(Debug, Clone)]
pub struct InFilterExpr {
    /// The column to filter
    child: Arc<dyn PhysicalExpr>,
    /// The IN filter
    filter: InFilter,
}

impl InFilterExpr {
    pub fn new(child: Arc<dyn PhysicalExpr>, filter: InFilter) -> Self {
        Self { child, filter }
    }
}

impl Display for InFilterExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "InFilter(cardinality={})", self.filter.cardinality())
    }
}

impl PartialEq for InFilterExpr {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child) && self.filter == other.filter
    }
}

impl Eq for InFilterExpr {}

impl Hash for InFilterExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.child.hash(state);
        self.filter.hash(state);
    }
}

impl PartialEq<dyn Any> for InFilterExpr {
    fn eq(&self, other: &dyn Any) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.child.eq(&other.child) && self.filter == other.filter
        } else {
            false
        }
    }
}

impl PhysicalExpr for InFilterExpr {
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
                let result = match self.filter.data_type() {
                    DataType::Int64 => {
                        let int_array = array
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or_else(|| {
                                datafusion::common::DataFusionError::Internal(
                                    "Expected Int64Array".to_string(),
                                )
                            })?;
                        self.filter.filter_int64_array(int_array)?
                    }
                    DataType::Utf8 => {
                        let string_array = array
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| {
                                datafusion::common::DataFusionError::Internal(
                                    "Expected StringArray".to_string(),
                                )
                            })?;
                        self.filter.filter_string_array(string_array)?
                    }
                    dt => {
                        return Err(datafusion::common::DataFusionError::Internal(format!(
                            "Unsupported data type for IN filter: {:?}",
                            dt
                        )));
                    }
                };
                Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
            }
            ColumnarValue::Scalar(scalar) => {
                // For scalar values, evaluate and return scalar boolean
                let result = match (scalar, self.filter.data_type()) {
                    (datafusion::common::ScalarValue::Int64(Some(v)), DataType::Int64) => {
                        self.filter.contains_int64(v)
                    }
                    (datafusion::common::ScalarValue::Utf8(Some(v)), DataType::Utf8) => {
                        self.filter.contains_string(&v)
                    }
                    _ => false,
                };
                Ok(ColumnarValue::Scalar(
                    datafusion::common::ScalarValue::Boolean(Some(result)),
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
        Ok(Arc::new(InFilterExpr::new(
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
    use crate::runtime_filter::DEFAULT_IN_FILTER_THRESHOLD;

    #[test]
    fn test_in_filter_int64() {
        let mut filter = InFilter::new_int64(DEFAULT_IN_FILTER_THRESHOLD);
        filter.add_int64(1).unwrap();
        filter.add_int64(2).unwrap();
        filter.add_int64(3).unwrap();

        assert!(filter.contains_int64(1));
        assert!(filter.contains_int64(2));
        assert!(!filter.contains_int64(4));
        assert_eq!(filter.cardinality(), 3);
    }

    #[test]
    fn test_in_filter_string() {
        let mut filter = InFilter::new_string(DEFAULT_IN_FILTER_THRESHOLD);
        filter.add_string("a".to_string()).unwrap();
        filter.add_string("b".to_string()).unwrap();

        assert!(filter.contains_string("a"));
        assert!(filter.contains_string("b"));
        assert!(!filter.contains_string("c"));
    }

    #[test]
    fn test_in_filter_capacity() {
        let mut filter = InFilter::new_int64(2);
        filter.add_int64(1).unwrap();
        filter.add_int64(2).unwrap();

        let result = filter.add_int64(3);
        assert!(result.is_err());
    }

    #[test]
    fn test_in_filter_serialization() {
        let filter =
            InFilter::from_int64_values(vec![1, 2, 3], DEFAULT_IN_FILTER_THRESHOLD).unwrap();
        let bytes = filter.to_bytes().unwrap();
        let restored = InFilter::from_bytes(&bytes, DEFAULT_IN_FILTER_THRESHOLD).unwrap();

        assert_eq!(filter.cardinality(), restored.cardinality());
        assert!(restored.contains_int64(1));
        assert!(restored.contains_int64(2));
        assert!(restored.contains_int64(3));
    }

    #[test]
    fn test_in_filter_array() {
        let filter =
            InFilter::from_int64_values(vec![1, 3, 5], DEFAULT_IN_FILTER_THRESHOLD).unwrap();
        let array = Int64Array::from(vec![Some(1), Some(2), Some(3), None, Some(5)]);

        let result = filter.filter_int64_array(&array).unwrap();

        assert_eq!(result.value(0), true); // 1 is in filter
        assert_eq!(result.value(1), false); // 2 is not in filter
        assert_eq!(result.value(2), true); // 3 is in filter
        assert!(result.is_null(3)); // NULL
        assert_eq!(result.value(4), true); // 5 is in filter
    }
}
