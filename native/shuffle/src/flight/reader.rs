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

//! Arrow Flight client for fetching shuffle partitions from remote executors.

use std::collections::HashMap;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use arrow_flight::decode::FlightRecordBatchStream as ArrowFlightStream;
use arrow_flight::flight_service_client::FlightServiceClient;
use datafusion::error::{DataFusionError, Result};
use futures::TryStreamExt;
use log::debug;

use crate::flight::ticket::ShufflePartitionTicket;

/// Fetch a shuffle partition from a remote executor via Arrow Flight.
///
/// Connects to the Flight server at `host:port`, sends a DoGet request
/// with the shuffle file path and partition id, and collects the result
/// as a vector of Arrow RecordBatches.
pub async fn fetch_partition_via_flight(
    host: &str,
    port: u16,
    path: &str,
    partition_id: usize,
) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    let addr = format!("http://{host}:{port}");
    debug!("Flight fetch: partition {partition_id} from {addr} path={path}");

    let channel = tonic::transport::Channel::from_shared(addr.clone())
        .map_err(|e| DataFusionError::External(Box::new(e)))?
        .connect()
        .await
        .map_err(|e| {
            DataFusionError::External(format!("Flight connect to {addr} failed: {e}").into())
        })?;

    let mut client = FlightServiceClient::new(channel);

    let ticket = ShufflePartitionTicket::new(path.to_string(), partition_id).to_ticket();
    let response = client.do_get(ticket).await.map_err(|e| {
        DataFusionError::External(format!("Flight DoGet failed: {e}").into())
    })?;

    let flight_stream = ArrowFlightStream::new_from_flight_data(
        response
            .into_inner()
            .map_err(|e| arrow_flight::error::FlightError::Tonic(Box::new(e))),
    );

    let schema = flight_stream
        .schema()
        .cloned()
        .unwrap_or_else(|| std::sync::Arc::new(arrow::datatypes::Schema::empty()));
    let batches: Vec<RecordBatch> = flight_stream
        .try_collect()
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    Ok((schema, batches))
}
