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

//! Embedded Arrow Flight server for serving shuffle partitions.
//!
//! Each Comet executor starts one instance of this server. Remote executors
//! use Arrow Flight `DoGet` to fetch shuffle partitions directly, bypassing
//! Spark's Netty-based BlockTransferService.

use std::io::BufReader;
use std::path::Path;
use std::pin::Pin;

use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::IpcWriteOptions;
use arrow::ipc::CompressionType;
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use futures::{Stream, StreamExt};
use log::{error, info};
use tokio::sync::oneshot;
use tonic::transport::Server;

use crate::flight::ticket::ShufflePartitionTicket;

type BoxedFlightStream<T> =
    Pin<Box<dyn Stream<Item = Result<T, tonic::Status>> + Send + 'static>>;

/// Handle to a running Flight server, used for graceful shutdown.
pub struct FlightServerHandle {
    shutdown_tx: oneshot::Sender<()>,
    port: u16,
}

impl FlightServerHandle {
    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
    }
}

/// Arrow Flight service that serves Comet shuffle partitions from local disk.
#[derive(Clone)]
pub struct CometFlightService {}

impl CometFlightService {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for CometFlightService {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl FlightService for CometFlightService {
    type DoGetStream = BoxedFlightStream<FlightData>;
    type DoPutStream = BoxedFlightStream<PutResult>;
    type DoExchangeStream = BoxedFlightStream<FlightData>;
    type DoActionStream = BoxedFlightStream<arrow_flight::Result>;
    type ListActionsStream = BoxedFlightStream<ActionType>;
    type ListFlightsStream = BoxedFlightStream<FlightInfo>;
    type HandshakeStream = BoxedFlightStream<HandshakeResponse>;

    async fn do_get(
        &self,
        request: tonic::Request<Ticket>,
    ) -> Result<tonic::Response<Self::DoGetStream>, tonic::Status> {
        let ticket = request.into_inner();
        let partition_ticket = ShufflePartitionTicket::from_ticket(&ticket)
            .map_err(|e| tonic::Status::invalid_argument(e.to_string()))?;

        let data_path = partition_ticket.path.clone();
        let path = Path::new(&data_path);
        if !path.exists() {
            return Err(tonic::Status::not_found(format!(
                "Shuffle file not found: {data_path}"
            )));
        }

        let file = std::fs::File::open(path).map_err(|e| {
            tonic::Status::internal(format!("Failed to open shuffle file {data_path}: {e}"))
        })?;
        let reader = StreamReader::try_new(BufReader::new(file), None)
            .map_err(|e| tonic::Status::internal(format!("Failed to read Arrow IPC: {e}")))?;

        let schema = reader.schema();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<RecordBatch, arrow::error::ArrowError>>(2);

        tokio::task::spawn_blocking(move || {
            for batch_result in reader {
                if tx.blocking_send(batch_result).is_err() {
                    break;
                }
            }
        });

        let batch_stream = tokio_stream::wrappers::ReceiverStream::new(rx)
            .map(|r| r.map_err(arrow_flight::error::FlightError::from));

        let write_options = IpcWriteOptions::default()
            .try_with_compression(Some(CompressionType::LZ4_FRAME))
            .map_err(|e| tonic::Status::internal(format!("IPC options error: {e}")))?;

        let flight_stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .with_options(write_options)
            .build(batch_stream)
            .map(|r| r.map_err(|e| tonic::Status::from_error(Box::new(e))));

        Ok(tonic::Response::new(Box::pin(flight_stream) as Self::DoGetStream))
    }

    async fn handshake(
        &self,
        _request: tonic::Request<tonic::Streaming<HandshakeRequest>>,
    ) -> Result<tonic::Response<Self::HandshakeStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("handshake"))
    }

    async fn list_flights(
        &self,
        _request: tonic::Request<Criteria>,
    ) -> Result<tonic::Response<Self::ListFlightsStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("list_flights"))
    }

    async fn get_flight_info(
        &self,
        _request: tonic::Request<FlightDescriptor>,
    ) -> Result<tonic::Response<FlightInfo>, tonic::Status> {
        Err(tonic::Status::unimplemented("get_flight_info"))
    }

    async fn poll_flight_info(
        &self,
        _request: tonic::Request<FlightDescriptor>,
    ) -> Result<tonic::Response<PollInfo>, tonic::Status> {
        Err(tonic::Status::unimplemented("poll_flight_info"))
    }

    async fn get_schema(
        &self,
        _request: tonic::Request<FlightDescriptor>,
    ) -> Result<tonic::Response<SchemaResult>, tonic::Status> {
        Err(tonic::Status::unimplemented("get_schema"))
    }

    async fn do_put(
        &self,
        _request: tonic::Request<tonic::Streaming<FlightData>>,
    ) -> Result<tonic::Response<Self::DoPutStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("do_put"))
    }

    async fn do_exchange(
        &self,
        _request: tonic::Request<tonic::Streaming<FlightData>>,
    ) -> Result<tonic::Response<Self::DoExchangeStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("do_exchange"))
    }

    async fn do_action(
        &self,
        _request: tonic::Request<Action>,
    ) -> Result<tonic::Response<Self::DoActionStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("do_action"))
    }

    async fn list_actions(
        &self,
        _request: tonic::Request<Empty>,
    ) -> Result<tonic::Response<Self::ListActionsStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("list_actions"))
    }
}

/// Start the Comet Flight server on the given port (0 = auto-assign).
pub async fn start_flight_server(port: u16) -> Result<FlightServerHandle, Box<dyn std::error::Error>> {
    let service = CometFlightService::new();
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    let actual_port = listener.local_addr()?.port();

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    info!("Comet Flight server starting on port {actual_port}");
    tokio::spawn(async move {
        let result = Server::builder()
            .add_service(FlightServiceServer::new(service))
            .serve_with_incoming_shutdown(incoming, async {
                let _ = shutdown_rx.await;
            })
            .await;
        if let Err(e) = result {
            error!("Comet Flight server error: {e}");
        }
    });

    Ok(FlightServerHandle {
        shutdown_tx,
        port: actual_port,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_starts_and_stops() {
        let handle = start_flight_server(0).await.expect("server should start");
        assert!(handle.port() > 0);
        handle.shutdown();
    }
}
