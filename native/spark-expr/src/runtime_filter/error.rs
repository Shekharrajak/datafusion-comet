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

//! Error types for Runtime Filter operations

use arrow::error::ArrowError;
use datafusion::common::DataFusionError;
use std::fmt;

/// Errors that can occur during runtime filter operations
#[derive(Debug)]
pub enum RuntimeFilterError {
    /// Error when building a filter from invalid data
    InvalidFilterData { message: String },

    /// Error when filter type doesn't match the data type
    TypeMismatch {
        expected: String,
        actual: String,
    },

    /// Error when filter capacity is exceeded
    CapacityExceeded {
        filter_type: String,
        max_capacity: usize,
        requested: usize,
    },

    /// Error during filter serialization/deserialization
    SerializationError { message: String },

    /// Error when filter is not initialized
    NotInitialized { filter_type: String },

    /// Arrow error during filter operations
    ArrowError(ArrowError),

    /// DataFusion error
    DataFusionError(DataFusionError),

    /// Internal error
    Internal { message: String },

    /// Invalid state error
    InvalidState(String),
}

impl fmt::Display for RuntimeFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeFilterError::InvalidFilterData { message } => {
                write!(f, "Invalid filter data: {}", message)
            }
            RuntimeFilterError::TypeMismatch { expected, actual } => {
                write!(
                    f,
                    "Type mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            RuntimeFilterError::CapacityExceeded {
                filter_type,
                max_capacity,
                requested,
            } => {
                write!(
                    f,
                    "{} filter capacity exceeded: max {}, requested {}",
                    filter_type, max_capacity, requested
                )
            }
            RuntimeFilterError::SerializationError { message } => {
                write!(f, "Serialization error: {}", message)
            }
            RuntimeFilterError::NotInitialized { filter_type } => {
                write!(f, "{} filter not initialized", filter_type)
            }
            RuntimeFilterError::ArrowError(e) => {
                write!(f, "Arrow error: {}", e)
            }
            RuntimeFilterError::DataFusionError(e) => {
                write!(f, "DataFusion error: {}", e)
            }
            RuntimeFilterError::Internal { message } => {
                write!(f, "Internal error: {}", message)
            }
            RuntimeFilterError::InvalidState(message) => {
                write!(f, "Invalid state: {}", message)
            }
        }
    }
}

impl std::error::Error for RuntimeFilterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RuntimeFilterError::ArrowError(e) => Some(e),
            RuntimeFilterError::DataFusionError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ArrowError> for RuntimeFilterError {
    fn from(error: ArrowError) -> Self {
        RuntimeFilterError::ArrowError(error)
    }
}

impl From<DataFusionError> for RuntimeFilterError {
    fn from(error: DataFusionError) -> Self {
        RuntimeFilterError::DataFusionError(error)
    }
}

impl From<RuntimeFilterError> for DataFusionError {
    fn from(error: RuntimeFilterError) -> Self {
        DataFusionError::External(Box::new(error))
    }
}

/// Result type for runtime filter operations
pub type RuntimeFilterResult<T> = Result<T, RuntimeFilterError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RuntimeFilterError::InvalidFilterData {
            message: "test error".to_string(),
        };
        assert!(err.to_string().contains("Invalid filter data"));

        let err = RuntimeFilterError::TypeMismatch {
            expected: "Int64".to_string(),
            actual: "String".to_string(),
        };
        assert!(err.to_string().contains("Type mismatch"));

        let err = RuntimeFilterError::CapacityExceeded {
            filter_type: "IN".to_string(),
            max_capacity: 1000,
            requested: 2000,
        };
        assert!(err.to_string().contains("capacity exceeded"));
    }

    #[test]
    fn test_error_conversion() {
        let arrow_err = ArrowError::InvalidArgumentError("test".to_string());
        let runtime_err: RuntimeFilterError = arrow_err.into();
        assert!(matches!(runtime_err, RuntimeFilterError::ArrowError(_)));

        let df_err: DataFusionError = runtime_err.into();
        assert!(matches!(df_err, DataFusionError::External(_)));
    }
}
