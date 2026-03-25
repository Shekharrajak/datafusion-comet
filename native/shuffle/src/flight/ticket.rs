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

//! Ticket encoding/decoding for Arrow Flight shuffle partition requests.
//!
//! A ticket identifies a specific shuffle partition on a remote executor.
//! Format: JSON-encoded struct containing the shuffle file path and partition id.

use arrow_flight::Ticket;
use std::fmt;

/// Identifies a shuffle partition on a remote executor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShufflePartitionTicket {
    /// Path to the shuffle data file on the remote executor's local disk.
    pub path: String,
    /// Partition index within the shuffle output.
    pub partition_id: usize,
}

impl ShufflePartitionTicket {
    pub fn new(path: String, partition_id: usize) -> Self {
        Self { path, partition_id }
    }

    /// Encode this ticket into an Arrow Flight Ticket.
    pub fn to_ticket(&self) -> Ticket {
        let bytes = serde_json::to_vec(self).expect("ticket serialization cannot fail");
        Ticket::new(bytes)
    }

    /// Decode an Arrow Flight Ticket back into a ShufflePartitionTicket.
    pub fn from_ticket(ticket: &Ticket) -> Result<Self, TicketError> {
        serde_json::from_slice(&ticket.ticket).map_err(TicketError::Deserialize)
    }
}

/// Errors that can occur during ticket operations.
#[derive(Debug)]
pub enum TicketError {
    Deserialize(serde_json::Error),
}

impl fmt::Display for TicketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TicketError::Deserialize(e) => write!(f, "failed to decode shuffle ticket: {e}"),
        }
    }
}

impl std::error::Error for TicketError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ticket_roundtrip() {
        let original = ShufflePartitionTicket::new(
            "/tmp/spark/shuffle/job1/stage2/data.arrow".to_string(),
            42,
        );
        let flight_ticket = original.to_ticket();
        let decoded = ShufflePartitionTicket::from_ticket(&flight_ticket).unwrap();
        assert_eq!(decoded.path, original.path);
        assert_eq!(decoded.partition_id, original.partition_id);
    }

    #[test]
    fn test_invalid_ticket() {
        let bad_ticket = Ticket::new(b"not json".to_vec());
        assert!(ShufflePartitionTicket::from_ticket(&bad_ticket).is_err());
    }
}
