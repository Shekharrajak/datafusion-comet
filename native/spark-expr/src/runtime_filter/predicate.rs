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

//! Runtime Filter Predicate Builder
//!
//! This module provides utilities for converting collected runtime filters into
//! DataFusion physical expressions that can be pushed down to scan operators
//! for row-group level pruning in Parquet files.

use std::sync::Arc;

use arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion::common::ScalarValue;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion::physical_expr::PhysicalExpr;

use super::collector::CollectedFilter;
use super::error::RuntimeFilterResult;
use super::filter_type::RuntimeFilterType;

/// Builds DataFusion physical expressions from collected runtime filters.
///
/// These expressions can be used for:
/// - Row-group level pruning using Parquet statistics
/// - Row-level filtering during scan
pub struct RuntimeFilterPredicateBuilder {
    /// Schema of the table being scanned
    schema: SchemaRef,
}

impl RuntimeFilterPredicateBuilder {
    /// Create a new predicate builder for the given schema
    pub fn new(schema: SchemaRef) -> Self {
        Self { schema }
    }

    /// Build a physical expression from a collected filter
    ///
    /// For MinMax filters, creates: column >= min AND column <= max
    /// For IN filters with small cardinality, creates: column IN (v1, v2, ...)
    pub fn build_predicate(
        &self,
        filter: &CollectedFilter,
        column_name: &str,
    ) -> RuntimeFilterResult<Option<Arc<dyn PhysicalExpr>>> {
        // Find the column in the schema
        let column_idx = match self.schema.index_of(column_name) {
            Ok(idx) => idx,
            Err(_) => return Ok(None), // Column not found, skip filter
        };

        let column_expr: Arc<dyn PhysicalExpr> =
            Arc::new(Column::new(column_name, column_idx));

        match filter.filter_type {
            RuntimeFilterType::MinMax => {
                self.build_min_max_predicate(column_expr, filter)
            }
            RuntimeFilterType::In => {
                self.build_in_predicate(column_expr, filter)
            }
            RuntimeFilterType::Bloom => {
                // Bloom filter predicates are handled separately
                Ok(None)
            }
        }
    }

    /// Build a MinMax range predicate: column >= min AND column <= max
    fn build_min_max_predicate(
        &self,
        column_expr: Arc<dyn PhysicalExpr>,
        filter: &CollectedFilter,
    ) -> RuntimeFilterResult<Option<Arc<dyn PhysicalExpr>>> {
        let (min_value, max_value) = match (&filter.min_value, &filter.max_value) {
            (Some(min), Some(max)) => (min.clone(), max.clone()),
            _ => return Ok(None),
        };

        // Create: column >= min
        let min_literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(min_value));
        let ge_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::clone(&column_expr),
            Operator::GtEq,
            min_literal,
        ));

        // Create: column <= max
        let max_literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(max_value));
        let le_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::clone(&column_expr),
            Operator::LtEq,
            max_literal,
        ));

        // Combine: column >= min AND column <= max
        let and_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            ge_expr,
            Operator::And,
            le_expr,
        ));

        Ok(Some(and_expr))
    }

    /// Build an IN predicate for small cardinality filters
    fn build_in_predicate(
        &self,
        column_expr: Arc<dyn PhysicalExpr>,
        filter: &CollectedFilter,
    ) -> RuntimeFilterResult<Option<Arc<dyn PhysicalExpr>>> {
        let values = match &filter.in_values {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(None),
        };

        // For small IN lists, convert to OR of equality checks
        // column = v1 OR column = v2 OR ...
        if values.len() <= 10 {
            let mut or_expr: Option<Arc<dyn PhysicalExpr>> = None;

            for value in values {
                let literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(value.clone()));
                let eq_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                    Arc::clone(&column_expr),
                    Operator::Eq,
                    literal,
                ));

                or_expr = Some(match or_expr {
                    Some(existing) => Arc::new(BinaryExpr::new(
                        existing,
                        Operator::Or,
                        eq_expr,
                    )),
                    None => eq_expr,
                });
            }

            return Ok(or_expr);
        }

        // For larger IN lists, fall back to MinMax bounds if available
        if let (Some(min), Some(max)) = (&filter.min_value, &filter.max_value) {
            let min_literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(min.clone()));
            let max_literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(max.clone()));

            let ge_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                Arc::clone(&column_expr),
                Operator::GtEq,
                min_literal,
            ));

            let le_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                Arc::clone(&column_expr),
                Operator::LtEq,
                max_literal,
            ));

            let and_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                ge_expr,
                Operator::And,
                le_expr,
            ));

            return Ok(Some(and_expr));
        }

        Ok(None)
    }

    /// Combine multiple filter predicates with AND
    pub fn combine_predicates(
        predicates: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Option<Arc<dyn PhysicalExpr>> {
        if predicates.is_empty() {
            return None;
        }

        let mut result = predicates[0].clone();
        for pred in predicates.into_iter().skip(1) {
            result = Arc::new(BinaryExpr::new(result, Operator::And, pred));
        }

        Some(result)
    }
}

/// Statistics for runtime filter application
#[derive(Debug, Default, Clone)]
pub struct RuntimeFilterStats {
    /// Number of row groups that matched the filter
    pub row_groups_matched: usize,
    /// Number of row groups pruned by the filter
    pub row_groups_pruned: usize,
    /// Number of rows that passed the filter
    pub rows_passed: usize,
    /// Number of rows pruned by the filter
    pub rows_pruned: usize,
    /// Estimated bytes saved by pruning
    pub bytes_saved: usize,
    /// Time spent evaluating filters (nanoseconds)
    pub eval_time_nanos: u64,
}

impl RuntimeFilterStats {
    /// Calculate selectivity (ratio of rows passed to total rows)
    pub fn selectivity(&self) -> f64 {
        let total = self.rows_passed + self.rows_pruned;
        if total == 0 {
            1.0
        } else {
            self.rows_passed as f64 / total as f64
        }
    }

    /// Calculate row group pruning ratio
    pub fn row_group_pruning_ratio(&self) -> f64 {
        let total = self.row_groups_matched + self.row_groups_pruned;
        if total == 0 {
            0.0
        } else {
            self.row_groups_pruned as f64 / total as f64
        }
    }

    /// Merge statistics from another instance
    pub fn merge(&mut self, other: &RuntimeFilterStats) {
        self.row_groups_matched += other.row_groups_matched;
        self.row_groups_pruned += other.row_groups_pruned;
        self.rows_passed += other.rows_passed;
        self.rows_pruned += other.rows_pruned;
        self.bytes_saved += other.bytes_saved;
        self.eval_time_nanos += other.eval_time_nanos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("value", DataType::Float64, true),
        ]))
    }

    #[test]
    fn test_build_min_max_predicate() {
        let schema = test_schema();
        let builder = RuntimeFilterPredicateBuilder::new(schema);

        let filter = CollectedFilter::new_min_max(
            "test".to_string(),
            0,
            DataType::Int64,
            ScalarValue::Int64(Some(10)),
            ScalarValue::Int64(Some(100)),
            1000,
        );

        let predicate = builder.build_predicate(&filter, "id").unwrap();
        assert!(predicate.is_some());

        // The predicate should be: id >= 10 AND id <= 100
        let pred_str = format!("{:?}", predicate.unwrap());
        assert!(pred_str.contains("GtEq") || pred_str.contains(">="));
    }

    #[test]
    fn test_build_in_predicate_small() {
        let schema = test_schema();
        let builder = RuntimeFilterPredicateBuilder::new(schema);

        let filter = CollectedFilter::new_in(
            "test".to_string(),
            0,
            DataType::Int64,
            vec![
                ScalarValue::Int64(Some(1)),
                ScalarValue::Int64(Some(2)),
                ScalarValue::Int64(Some(3)),
            ],
            3,
        );

        let predicate = builder.build_predicate(&filter, "id").unwrap();
        assert!(predicate.is_some());
    }

    #[test]
    fn test_combine_predicates() {
        let schema = test_schema();
        let builder = RuntimeFilterPredicateBuilder::new(Arc::clone(&schema));

        let filter1 = CollectedFilter::new_min_max(
            "f1".to_string(),
            0,
            DataType::Int64,
            ScalarValue::Int64(Some(10)),
            ScalarValue::Int64(Some(100)),
            1000,
        );

        let filter2 = CollectedFilter::new_min_max(
            "f2".to_string(),
            2,
            DataType::Float64,
            ScalarValue::Float64(Some(0.0)),
            ScalarValue::Float64(Some(1.0)),
            1000,
        );

        let pred1 = builder.build_predicate(&filter1, "id").unwrap().unwrap();
        let pred2 = builder.build_predicate(&filter2, "value").unwrap().unwrap();

        let combined = RuntimeFilterPredicateBuilder::combine_predicates(vec![pred1, pred2]);
        assert!(combined.is_some());
    }

    #[test]
    fn test_stats_selectivity() {
        let mut stats = RuntimeFilterStats::default();
        stats.rows_passed = 100;
        stats.rows_pruned = 900;

        assert!((stats.selectivity() - 0.1).abs() < 0.001);
    }
}
