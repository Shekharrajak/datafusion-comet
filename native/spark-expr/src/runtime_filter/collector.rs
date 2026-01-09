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

//! Runtime Filter Collector
//!
//! This module provides the `RuntimeFilterCollector` which collects values from
//! the build side of a hash join and produces runtime filters that can be pushed
//! down to scan operators for I/O reduction.

use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use arrow::array::{Array, ArrayRef};
use arrow::datatypes::DataType;
use datafusion::common::ScalarValue;

use super::error::{RuntimeFilterError, RuntimeFilterResult};
use super::filter_type::RuntimeFilterType;
use super::in_filter::InFilter;
use super::min_max_filter::MinMaxFilter;
use super::RuntimeFilterConfig;

/// Collected runtime filter data that can be serialized and broadcast
#[derive(Debug, Clone)]
pub struct CollectedFilter {
    /// Unique identifier for this filter
    pub filter_id: String,
    /// Type of filter collected
    pub filter_type: RuntimeFilterType,
    /// Column index this filter applies to
    pub column_index: usize,
    /// Data type of the filtered column
    pub data_type: DataType,
    /// Min value (for MinMax filter)
    pub min_value: Option<ScalarValue>,
    /// Max value (for MinMax filter)
    pub max_value: Option<ScalarValue>,
    /// Distinct values (for IN filter, when cardinality is small)
    pub in_values: Option<Vec<ScalarValue>>,
    /// Number of rows in build side
    pub build_row_count: usize,
    /// Estimated selectivity
    pub estimated_selectivity: f64,
}

impl CollectedFilter {
    /// Create a MinMax filter from collected bounds
    pub fn new_min_max(
        filter_id: String,
        column_index: usize,
        data_type: DataType,
        min_value: ScalarValue,
        max_value: ScalarValue,
        build_row_count: usize,
    ) -> Self {
        Self {
            filter_id,
            filter_type: RuntimeFilterType::MinMax,
            column_index,
            data_type,
            min_value: Some(min_value),
            max_value: Some(max_value),
            in_values: None,
            build_row_count,
            estimated_selectivity: 0.0, // Will be updated when applied
        }
    }

    /// Create an IN filter from collected values
    pub fn new_in(
        filter_id: String,
        column_index: usize,
        data_type: DataType,
        values: Vec<ScalarValue>,
        build_row_count: usize,
    ) -> Self {
        Self {
            filter_id,
            filter_type: RuntimeFilterType::In,
            column_index,
            data_type,
            min_value: None,
            max_value: None,
            in_values: Some(values),
            build_row_count,
            estimated_selectivity: 0.0,
        }
    }

    /// Check if a scalar value passes this filter
    pub fn contains(&self, value: &ScalarValue) -> bool {
        match self.filter_type {
            RuntimeFilterType::MinMax => {
                if let (Some(min), Some(max)) = (&self.min_value, &self.max_value) {
                    // Check if value is within [min, max] range
                    value >= min && value <= max
                } else {
                    true // No bounds, pass everything
                }
            }
            RuntimeFilterType::In => {
                if let Some(values) = &self.in_values {
                    values.contains(value)
                } else {
                    true
                }
            }
            RuntimeFilterType::Bloom => {
                // Bloom filter not yet implemented in collector
                true
            }
        }
    }

    /// Create an InFilter from this collected filter
    pub fn to_in_filter(&self) -> RuntimeFilterResult<InFilter> {
        if let Some(values) = &self.in_values {
            // Convert ScalarValues to appropriate InFilter based on data type
            match &self.data_type {
                DataType::Int64 => {
                    let int_values: Vec<i64> = values
                        .iter()
                        .filter_map(|v| match v {
                            ScalarValue::Int64(Some(i)) => Some(*i),
                            _ => None,
                        })
                        .collect();
                    InFilter::from_int64_values(int_values, values.len())
                }
                DataType::Utf8 => {
                    let str_values: Vec<String> = values
                        .iter()
                        .filter_map(|v| match v {
                            ScalarValue::Utf8(Some(s)) => Some(s.clone()),
                            _ => None,
                        })
                        .collect();
                    InFilter::from_string_values(str_values, values.len())
                }
                _ => Err(RuntimeFilterError::InvalidFilterData {
                    message: format!(
                        "Unsupported data type for IN filter: {:?}",
                        self.data_type
                    ),
                }),
            }
        } else {
            Err(RuntimeFilterError::InvalidState(
                "Cannot create InFilter from non-IN collected filter".to_string(),
            ))
        }
    }

    /// Create a MinMaxFilter from this collected filter
    pub fn to_min_max_filter(&self) -> RuntimeFilterResult<MinMaxFilter> {
        if let (Some(min), Some(max)) = (&self.min_value, &self.max_value) {
            MinMaxFilter::with_bounds(min.clone(), max.clone())
        } else {
            Err(RuntimeFilterError::InvalidState(
                "Cannot create MinMaxFilter without min/max bounds".to_string(),
            ))
        }
    }
}

/// Collector for runtime filter values during hash join build phase
///
/// This collector accumulates values from the build side of a hash join
/// and produces runtime filters that can be used to prune data on the probe side.
pub struct RuntimeFilterCollector {
    /// Configuration for filter behavior
    config: RuntimeFilterConfig,
    /// Column index being collected
    column_index: usize,
    /// Data type of the column
    data_type: DataType,
    /// Collected min value
    min_value: RwLock<Option<ScalarValue>>,
    /// Collected max value
    max_value: RwLock<Option<ScalarValue>>,
    /// Collected distinct values (for IN filter)
    distinct_values: RwLock<HashSet<ScalarValue>>,
    /// Total rows processed
    row_count: RwLock<usize>,
    /// Whether collection is still active
    is_collecting: RwLock<bool>,
}

impl RuntimeFilterCollector {
    /// Create a new collector for the given column
    pub fn new(config: RuntimeFilterConfig, column_index: usize, data_type: DataType) -> Self {
        Self {
            config,
            column_index,
            data_type,
            min_value: RwLock::new(None),
            max_value: RwLock::new(None),
            distinct_values: RwLock::new(HashSet::new()),
            row_count: RwLock::new(0),
            is_collecting: RwLock::new(true),
        }
    }

    /// Collect values from an array batch
    pub fn collect_batch(&self, array: &ArrayRef) -> RuntimeFilterResult<()> {
        let is_collecting = *self.is_collecting.read().unwrap();
        if !is_collecting {
            return Ok(());
        }

        let mut row_count = self.row_count.write().unwrap();
        *row_count += array.len();

        // Collect min/max values
        self.update_min_max(array)?;

        // Collect distinct values if cardinality is within threshold
        let distinct_count = {
            let distinct = self.distinct_values.read().unwrap();
            distinct.len()
        };

        if distinct_count <= self.config.in_filter_threshold {
            self.collect_distinct_values(array)?;
        }

        Ok(())
    }

    /// Update min/max bounds from array
    fn update_min_max(&self, array: &ArrayRef) -> RuntimeFilterResult<()> {
        // Iterate through array to find min/max values
        for i in 0..array.len() {
            if array.is_null(i) {
                continue;
            }

            let value = ScalarValue::try_from_array(array, i)?;

            // Update min
            {
                let mut min_val = self.min_value.write().unwrap();
                match &*min_val {
                    Some(existing) if &value < existing => *min_val = Some(value.clone()),
                    None => *min_val = Some(value.clone()),
                    _ => {}
                }
            }

            // Update max
            {
                let mut max_val = self.max_value.write().unwrap();
                match &*max_val {
                    Some(existing) if &value > existing => *max_val = Some(value.clone()),
                    None => *max_val = Some(value.clone()),
                    _ => {}
                }
            }
        }

        Ok(())
    }

    /// Collect distinct values from array
    fn collect_distinct_values(&self, array: &ArrayRef) -> RuntimeFilterResult<()> {
        let mut distinct = self.distinct_values.write().unwrap();

        // Stop collecting if we exceed threshold
        if distinct.len() > self.config.in_filter_threshold {
            return Ok(());
        }

        for i in 0..array.len() {
            if !array.is_null(i) {
                let value = ScalarValue::try_from_array(array, i)?;
                distinct.insert(value);

                // Stop if we exceed threshold
                if distinct.len() > self.config.in_filter_threshold {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Finalize collection and produce the runtime filter
    pub fn finalize(&self, filter_id: String) -> RuntimeFilterResult<CollectedFilter> {
        // Mark collection as complete
        *self.is_collecting.write().unwrap() = false;

        let row_count = *self.row_count.read().unwrap();
        let distinct_values = self.distinct_values.read().unwrap();
        let min_value = self.min_value.read().unwrap();
        let max_value = self.max_value.read().unwrap();

        let is_numeric = matches!(
            self.data_type,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64
                | DataType::Date32
                | DataType::Date64
                | DataType::Timestamp(_, _)
        );

        // Determine filter type based on collected statistics
        let filter_type =
            self.config
                .select_filter_type(distinct_values.len(), is_numeric);

        match filter_type {
            RuntimeFilterType::In if distinct_values.len() <= self.config.in_filter_threshold => {
                Ok(CollectedFilter::new_in(
                    filter_id,
                    self.column_index,
                    self.data_type.clone(),
                    distinct_values.iter().cloned().collect(),
                    row_count,
                ))
            }
            RuntimeFilterType::MinMax | RuntimeFilterType::In => {
                // Use MinMax if we have bounds
                if let (Some(min), Some(max)) = (min_value.clone(), max_value.clone()) {
                    Ok(CollectedFilter::new_min_max(
                        filter_id,
                        self.column_index,
                        self.data_type.clone(),
                        min,
                        max,
                        row_count,
                    ))
                } else {
                    Err(RuntimeFilterError::InvalidState(
                        "No min/max values collected".to_string(),
                    ))
                }
            }
            RuntimeFilterType::Bloom => {
                // Fall back to MinMax for now (Bloom filter support to be added)
                if let (Some(min), Some(max)) = (min_value.clone(), max_value.clone()) {
                    Ok(CollectedFilter::new_min_max(
                        filter_id,
                        self.column_index,
                        self.data_type.clone(),
                        min,
                        max,
                        row_count,
                    ))
                } else {
                    Err(RuntimeFilterError::InvalidState(
                        "No min/max values collected".to_string(),
                    ))
                }
            }
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> (usize, usize, bool, bool) {
        let row_count = *self.row_count.read().unwrap();
        let distinct_count = self.distinct_values.read().unwrap().len();
        let has_min = self.min_value.read().unwrap().is_some();
        let has_max = self.max_value.read().unwrap().is_some();
        (row_count, distinct_count, has_min, has_max)
    }
}

/// Manager for multiple runtime filter collectors
pub struct RuntimeFilterManager {
    /// Active collectors indexed by filter ID
    collectors: RwLock<Vec<Arc<RuntimeFilterCollector>>>,
    /// Finalized filters ready for application
    filters: RwLock<Vec<CollectedFilter>>,
    /// Configuration
    config: RuntimeFilterConfig,
}

impl RuntimeFilterManager {
    /// Create a new filter manager
    pub fn new(config: RuntimeFilterConfig) -> Self {
        Self {
            collectors: RwLock::new(Vec::new()),
            filters: RwLock::new(Vec::new()),
            config,
        }
    }

    /// Create a new collector for a column
    pub fn create_collector(
        &self,
        column_index: usize,
        data_type: DataType,
    ) -> Arc<RuntimeFilterCollector> {
        let collector = Arc::new(RuntimeFilterCollector::new(
            self.config.clone(),
            column_index,
            data_type,
        ));
        self.collectors.write().unwrap().push(Arc::clone(&collector));
        collector
    }

    /// Finalize all collectors and produce filters
    pub fn finalize_all(&self) -> RuntimeFilterResult<Vec<CollectedFilter>> {
        let collectors = self.collectors.read().unwrap();
        let mut filters = self.filters.write().unwrap();

        for (idx, collector) in collectors.iter().enumerate() {
            let filter_id = format!("rf_{}", idx);
            match collector.finalize(filter_id) {
                Ok(filter) => filters.push(filter),
                Err(e) => {
                    // Log but don't fail - some filters may not be applicable
                    eprintln!("Failed to finalize filter {}: {:?}", idx, e);
                }
            }
        }

        Ok(filters.clone())
    }

    /// Get all finalized filters
    pub fn get_filters(&self) -> Vec<CollectedFilter> {
        self.filters.read().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};

    #[test]
    fn test_collector_min_max() {
        let config = RuntimeFilterConfig::default();
        let collector = RuntimeFilterCollector::new(config, 0, DataType::Int64);

        // Create test array
        let array: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 5, 30, 15]));
        collector.collect_batch(&array).unwrap();

        let filter = collector.finalize("test".to_string()).unwrap();

        assert_eq!(filter.min_value, Some(ScalarValue::Int64(Some(5))));
        assert_eq!(filter.max_value, Some(ScalarValue::Int64(Some(30))));
    }

    #[test]
    fn test_collector_distinct_values() {
        let mut config = RuntimeFilterConfig::default();
        config.in_filter_threshold = 100; // Ensure IN filter is used

        let collector = RuntimeFilterCollector::new(config, 0, DataType::Utf8);

        // Create test array with few distinct values
        let array: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c", "a", "b"]));
        collector.collect_batch(&array).unwrap();

        let filter = collector.finalize("test".to_string()).unwrap();

        assert_eq!(filter.filter_type, RuntimeFilterType::In);
        assert!(filter.in_values.is_some());
        assert_eq!(filter.in_values.unwrap().len(), 3);
    }

    #[test]
    fn test_filter_contains() {
        let filter = CollectedFilter::new_min_max(
            "test".to_string(),
            0,
            DataType::Int64,
            ScalarValue::Int64(Some(10)),
            ScalarValue::Int64(Some(100)),
            1000,
        );

        assert!(filter.contains(&ScalarValue::Int64(Some(50))));
        assert!(filter.contains(&ScalarValue::Int64(Some(10))));
        assert!(filter.contains(&ScalarValue::Int64(Some(100))));
        assert!(!filter.contains(&ScalarValue::Int64(Some(5))));
        assert!(!filter.contains(&ScalarValue::Int64(Some(150))));
    }
}
