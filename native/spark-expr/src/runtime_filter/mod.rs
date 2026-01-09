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

//! Runtime Filter Module for Comet
//!
//! This module provides native runtime filter implementations that can significantly
//! reduce I/O during join operations by filtering data at scan time.
//!
//! # Filter Types
//!
//! - **InFilter**: Exact match filter for small cardinality (<1000 values)
//! - **MinMaxFilter**: Range-based filter for numeric/date types
//! - **BloomFilter**: Probabilistic filter for large cardinality (existing implementation)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    RuntimeFilterManager                      │
//! │  - Coordinates filter creation and application               │
//! │  - Selects optimal filter type based on statistics           │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!          ┌───────────────────┼───────────────────┐
//!          ▼                   ▼                   ▼
//!    ┌──────────┐       ┌──────────┐       ┌──────────┐
//!    │ InFilter │       │ MinMax   │       │  Bloom   │
//!    │          │       │ Filter   │       │  Filter  │
//!    └──────────┘       └──────────┘       └──────────┘
//! ```

mod collector;
mod error;
mod filter_type;
mod in_filter;
mod min_max_filter;
mod predicate;
mod runtime_filter_expr;

pub use collector::{CollectedFilter, RuntimeFilterCollector, RuntimeFilterManager};
pub use error::{RuntimeFilterError, RuntimeFilterResult};
pub use filter_type::RuntimeFilterType;
pub use in_filter::InFilter;
pub use min_max_filter::MinMaxFilter;
pub use predicate::{RuntimeFilterPredicateBuilder, RuntimeFilterStats};
pub use runtime_filter_expr::RuntimeFilterExpr;

/// Default threshold for switching from IN filter to Bloom filter
pub const DEFAULT_IN_FILTER_THRESHOLD: usize = 1000;

/// Default false positive probability for Bloom filters
pub const DEFAULT_BLOOM_FILTER_FPP: f64 = 0.01;

/// Configuration for runtime filter behavior
#[derive(Debug, Clone)]
pub struct RuntimeFilterConfig {
    /// Whether runtime filters are enabled
    pub enabled: bool,
    /// Maximum number of values for IN filter before switching to Bloom
    pub in_filter_threshold: usize,
    /// False positive probability for Bloom filters
    pub bloom_filter_fpp: f64,
    /// Minimum selectivity to apply runtime filter (0.0 - 1.0)
    pub min_selectivity: f64,
}

impl Default for RuntimeFilterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            in_filter_threshold: DEFAULT_IN_FILTER_THRESHOLD,
            bloom_filter_fpp: DEFAULT_BLOOM_FILTER_FPP,
            min_selectivity: 0.5,
        }
    }
}

impl RuntimeFilterConfig {
    /// Create a new configuration with custom settings
    pub fn new(
        enabled: bool,
        in_filter_threshold: usize,
        bloom_filter_fpp: f64,
        min_selectivity: f64,
    ) -> Self {
        Self {
            enabled,
            in_filter_threshold,
            bloom_filter_fpp,
            min_selectivity,
        }
    }

    /// Determine the optimal filter type based on cardinality
    pub fn select_filter_type(&self, cardinality: usize, is_numeric: bool) -> RuntimeFilterType {
        if cardinality <= self.in_filter_threshold {
            if is_numeric && cardinality > 1 {
                // For numeric types with reasonable cardinality, prefer MinMax
                RuntimeFilterType::MinMax
            } else {
                RuntimeFilterType::In
            }
        } else {
            RuntimeFilterType::Bloom
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_type_selection() {
        let config = RuntimeFilterConfig::default();

        // Small cardinality numeric -> MinMax
        assert_eq!(
            config.select_filter_type(100, true),
            RuntimeFilterType::MinMax
        );

        // Small cardinality non-numeric -> In
        assert_eq!(config.select_filter_type(100, false), RuntimeFilterType::In);

        // Large cardinality -> Bloom
        assert_eq!(
            config.select_filter_type(5000, true),
            RuntimeFilterType::Bloom
        );
    }
}
