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

//! Runtime filter type definitions

use std::fmt;

/// Types of runtime filters supported by Comet
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeFilterType {
    /// IN filter for small cardinality exact matches
    /// Best for: cardinality < 1000, any data type
    /// Characteristics: Zero false positives, O(1) lookup with HashSet
    In,

    /// Min/Max range filter for numeric and date types
    /// Best for: numeric/date types with bounded ranges
    /// Characteristics: Zero false positives within range, very fast comparison
    MinMax,

    /// Bloom filter for large cardinality
    /// Best for: cardinality >= 1000
    /// Characteristics: Probabilistic (false positives possible), memory efficient
    Bloom,
}

impl RuntimeFilterType {
    /// Returns whether this filter type can have false positives
    pub fn can_have_false_positives(&self) -> bool {
        matches!(self, RuntimeFilterType::Bloom)
    }

    /// Returns the memory efficiency rating (1-10, 10 being most efficient)
    pub fn memory_efficiency(&self) -> u8 {
        match self {
            RuntimeFilterType::In => 3,      // Stores all values
            RuntimeFilterType::MinMax => 10, // Only stores 2 values
            RuntimeFilterType::Bloom => 8,   // Compact bit array
        }
    }

    /// Returns the lookup speed rating (1-10, 10 being fastest)
    pub fn lookup_speed(&self) -> u8 {
        match self {
            RuntimeFilterType::In => 9,      // HashSet O(1)
            RuntimeFilterType::MinMax => 10, // Two comparisons
            RuntimeFilterType::Bloom => 7,   // Multiple hash computations
        }
    }

    /// Check if this filter type is suitable for the given data type
    pub fn is_suitable_for_type(&self, is_numeric: bool, is_temporal: bool) -> bool {
        match self {
            RuntimeFilterType::In => true, // Works for all types
            RuntimeFilterType::MinMax => is_numeric || is_temporal,
            RuntimeFilterType::Bloom => true, // Works for all types
        }
    }
}

impl fmt::Display for RuntimeFilterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeFilterType::In => write!(f, "IN"),
            RuntimeFilterType::MinMax => write!(f, "MinMax"),
            RuntimeFilterType::Bloom => write!(f, "Bloom"),
        }
    }
}

/// Statistics used for filter type selection
#[derive(Debug, Clone)]
pub struct FilterSelectionStats {
    /// Number of distinct values (cardinality)
    pub cardinality: usize,
    /// Whether the data type is numeric
    pub is_numeric: bool,
    /// Whether the data type is temporal (date/timestamp)
    pub is_temporal: bool,
    /// Estimated selectivity (0.0 - 1.0)
    pub selectivity: f64,
    /// Size of the build side in bytes
    pub build_side_bytes: Option<u64>,
}

impl FilterSelectionStats {
    pub fn new(cardinality: usize, is_numeric: bool, is_temporal: bool) -> Self {
        Self {
            cardinality,
            is_numeric,
            is_temporal,
            selectivity: 1.0,
            build_side_bytes: None,
        }
    }

    pub fn with_selectivity(mut self, selectivity: f64) -> Self {
        self.selectivity = selectivity;
        self
    }

    pub fn with_build_size(mut self, bytes: u64) -> Self {
        self.build_side_bytes = Some(bytes);
        self
    }
}

/// Filter type selector that chooses the optimal filter based on statistics
pub struct FilterTypeSelector {
    in_filter_threshold: usize,
    min_selectivity: f64,
}

impl FilterTypeSelector {
    pub fn new(in_filter_threshold: usize, min_selectivity: f64) -> Self {
        Self {
            in_filter_threshold,
            min_selectivity,
        }
    }

    /// Select the optimal filter type based on statistics
    pub fn select(&self, stats: &FilterSelectionStats) -> Option<RuntimeFilterType> {
        // Don't create filter if selectivity is too low (would filter too few rows)
        if stats.selectivity < self.min_selectivity {
            return None;
        }

        // For numeric/temporal types, prefer MinMax if cardinality is reasonable
        if (stats.is_numeric || stats.is_temporal) && stats.cardinality > 1 {
            // MinMax is almost always best for numeric types
            // It has zero false positives and is extremely fast
            if stats.cardinality <= self.in_filter_threshold {
                return Some(RuntimeFilterType::MinMax);
            }
        }

        // For small cardinality, use IN filter
        if stats.cardinality <= self.in_filter_threshold {
            return Some(RuntimeFilterType::In);
        }

        // For large cardinality, use Bloom filter
        Some(RuntimeFilterType::Bloom)
    }
}

impl Default for FilterTypeSelector {
    fn default() -> Self {
        Self::new(1000, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_type_properties() {
        assert!(!RuntimeFilterType::In.can_have_false_positives());
        assert!(!RuntimeFilterType::MinMax.can_have_false_positives());
        assert!(RuntimeFilterType::Bloom.can_have_false_positives());
    }

    #[test]
    fn test_filter_type_suitability() {
        assert!(RuntimeFilterType::In.is_suitable_for_type(false, false));
        assert!(RuntimeFilterType::MinMax.is_suitable_for_type(true, false));
        assert!(!RuntimeFilterType::MinMax.is_suitable_for_type(false, false));
        assert!(RuntimeFilterType::MinMax.is_suitable_for_type(false, true));
    }

    #[test]
    fn test_filter_type_selection() {
        let selector = FilterTypeSelector::default();

        // Small cardinality numeric -> MinMax
        let stats = FilterSelectionStats::new(100, true, false);
        assert_eq!(selector.select(&stats), Some(RuntimeFilterType::MinMax));

        // Small cardinality string -> IN
        let stats = FilterSelectionStats::new(100, false, false);
        assert_eq!(selector.select(&stats), Some(RuntimeFilterType::In));

        // Large cardinality -> Bloom
        let stats = FilterSelectionStats::new(5000, true, false);
        assert_eq!(selector.select(&stats), Some(RuntimeFilterType::Bloom));

        // Low selectivity -> None
        let stats = FilterSelectionStats::new(100, true, false).with_selectivity(0.1);
        assert_eq!(selector.select(&stats), None);
    }
}
