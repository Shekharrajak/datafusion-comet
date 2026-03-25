/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.spark.sql.comet.execution.shuffle

import org.apache.spark.{InterruptibleIterator, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.shuffle.{BaseShuffleHandle, ShuffleReader, ShuffleReadMetricsReporter}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.{BlockId, BlockManagerId}

/**
 * Shuffle reader that fetches data via Arrow Flight from remote Comet executors.
 *
 * Instead of using Spark's Netty-based BlockTransferService, this reader calls
 * the native Arrow Flight client to fetch shuffle partitions directly from
 * remote executors' embedded Flight servers.
 *
 * This keeps the entire shuffle read path in native (Rust) code, avoiding
 * JVM boundary crossings for data transfer.
 *
 * NOTE: This is a scaffolding implementation. The actual native Flight client
 * integration requires JNI methods that are not yet wired.
 */
class CometFlightShuffleReader[K, C](
    handle: BaseShuffleHandle[K, _, C],
    blocksByAddress: Iterator[(BlockManagerId, scala.collection.Seq[(BlockId, Long, Int)])],
    context: TaskContext,
    readMetrics: ShuffleReadMetricsReporter)
    extends ShuffleReader[K, C]
    with Logging {

  private val dep = handle.dependency.asInstanceOf[CometShuffleDependency[_, _, _]]

  override def read(): Iterator[Product2[K, C]] = {
    // TODO: Phase 2 implementation
    // 1. Resolve partition locations from blocksByAddress
    //    - Extract host:flightPort for each BlockManagerId
    //    - Extract shuffle file path from BlockId
    // 2. For each remote partition:
    //    a. Call native fetchPartitionViaFlight(host, port, path, partitionId)
    //    b. Native code returns Arrow RecordBatches via FFI
    //    c. Wrap as ColumnarBatch
    // 3. For local partitions:
    //    a. Read directly from local disk (same as current path)
    // 4. Merge all partition streams

    logWarning(
      "CometFlightShuffleReader is not yet fully implemented. " +
        "Falling back to CometBlockStoreShuffleReader.")

    // Fallback: delegate to the standard block store reader.
    // This preserves correctness while the Flight path is being built.
    val fallbackReader = new CometBlockStoreShuffleReader[K, C](
      handle,
      blocksByAddress,
      context,
      readMetrics)
    fallbackReader.read()
  }
}
