// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"

namespace comms::pipes::benchmark {

// Helper to compute global thread ID across all blocks
__device__ inline unsigned int getGlobalThreadId() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void p2pSend(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);
  p2p.send(group, srcBuff, nBytes, 0, timeout);
}

__global__ void p2pRecv(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);
  p2p.recv(group, dstBuff, nBytes, 0, timeout);
}

__global__ void p2pSendTimed(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    TimingStats* stats,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  unsigned int globalThreadId = getGlobalThreadId();

  // Only first thread globally records start time
  if (globalThreadId == 0) {
    stats->startCycle = clock64();
  }

  p2p.send(group, srcBuff, nBytes);

  // Only first thread globally records end time
  if (globalThreadId == 0) {
    unsigned long long end = clock64();
    stats->endCycle = end;
    stats->totalCycles = end - stats->startCycle;
  }
}

__global__ void p2pRecvTimed(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    TimingStats* stats,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  unsigned int globalThreadId = getGlobalThreadId();

  // Only first thread globally records start time
  if (globalThreadId == 0) {
    stats->startCycle = clock64();
  }

  p2p.recv(group, dstBuff, nBytes);

  // Only first thread globally records end time
  if (globalThreadId == 0) {
    unsigned long long end = clock64();
    stats->endCycle = end;
    stats->totalCycles = end - stats->startCycle;
  }
}

__global__ void p2pBidirectional(
    P2pNvlTransportDevice p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);

  // Partition groups into 2: half for send, half for recv
  auto [partition_id, subgroup] = group.partition_interleaved(2);
  if (partition_id == 0) {
    p2p.send(subgroup, sendBuff, nBytes, 0, timeout);
  } else {
    p2p.recv(subgroup, recvBuff, nBytes, 0, timeout);
  }
}

/**
 * asymmetric_recv_impl - Batched recv that coalesces fine-grained ChunkStates
 *
 * Replicates the structure of P2pNvlTransportDevice::recv() but processes
 * chunks in "macro chunks" of size recvChunkSize. For each macro chunk:
 *   1. Wait for all constituent fine-grained ChunkStates (spin loops)
 *   2. Single large memcpy_vectorized for the full macro chunk
 *   3. Release all constituent ChunkStates
 *
 * @param p2p Transport device (configured at fine chunk granularity)
 * @param group ThreadGroup for cooperative processing
 * @param dstBuff Destination buffer
 * @param nBytes Total bytes to receive
 * @param recvChunkSize Coarse recv chunk size (must be multiple of
 *        p2p.chunk_size())
 * @param timeout Timeout configuration
 */
__device__ __forceinline__ void asymmetric_recv_impl(
    P2pNvlTransportDevice& p2p,
    ThreadGroup& group,
    void* dstBuff,
    std::size_t nBytes,
    std::size_t recvChunkSize,
    const Timeout& timeout) {
#ifdef __CUDA_ARCH__
  char* dst = reinterpret_cast<char*>(dstBuff);
  char* recvBuffer = p2p.local_data_buffer();
  ChunkState* recvStates = p2p.local_state_buffer();

  const std::size_t transportChunkSize = p2p.chunk_size();
  const std::size_t dataBufferSize = p2p.data_buffer_size();
  const std::size_t pipelineDepth = p2p.pipeline_depth();

  // Validate: recvChunkSize must be an exact multiple of transport chunk size
  if (recvChunkSize % transportChunkSize != 0) {
    printf(
        "asymmetric_recv: recvChunkSize (%llu) must be a multiple of "
        "transportChunkSize (%llu)\n",
        (unsigned long long)recvChunkSize,
        (unsigned long long)transportChunkSize);
    __trap();
  }

  // Validate: dataBufferSize must be an exact multiple of recvChunkSize
  // to ensure macro-chunk indexing doesn't overshoot the state buffer
  if (dataBufferSize % recvChunkSize != 0) {
    printf(
        "asymmetric_recv: dataBufferSize (%llu) must be a multiple of "
        "recvChunkSize (%llu)\n",
        (unsigned long long)dataBufferSize,
        (unsigned long long)recvChunkSize);
    __trap();
  }

  const std::size_t chunksPerStep =
      (dataBufferSize + transportChunkSize - 1) / transportChunkSize;
  const std::size_t totalSteps = (nBytes + dataBufferSize - 1) / dataBufferSize;
  const std::size_t batchFactor = recvChunkSize / transportChunkSize;

  for (std::size_t stepId = 0; stepId < totalSteps; ++stepId) {
    const std::size_t pipelineIdx = stepId % pipelineDepth;
    const std::size_t dataBufferOffset = pipelineIdx * dataBufferSize;
    const std::size_t stateOffset = pipelineIdx * chunksPerStep;
    const std::size_t stepOffset = stepId * dataBufferSize;
    const std::size_t stepBytes = (stepOffset + dataBufferSize <= nBytes)
        ? dataBufferSize
        : nBytes - stepOffset;
    const std::size_t numMacroChunks =
        (stepBytes + recvChunkSize - 1) / recvChunkSize;

    group.for_each_item_contiguous(numMacroChunks, [&](uint32_t macroIdx) {
      const std::size_t macroOffset = macroIdx * recvChunkSize;
      const std::size_t macroBytes = (macroOffset + recvChunkSize <= stepBytes)
          ? recvChunkSize
          : stepBytes - macroOffset;
      if (macroBytes == 0) {
        return;
      }

      const std::size_t firstChunk = macroIdx * batchFactor;
      const std::size_t numFineChunks =
          (macroBytes + transportChunkSize - 1) / transportChunkSize;

      // Phase 1: Wait for all fine-grained chunks (spin loops, no sync
      // overhead)
      for (std::size_t i = 0; i < numFineChunks; ++i) {
        recvStates[stateOffset + firstChunk + i].wait_ready_to_recv(
            group, stepId, 0, timeout);
      }

      // Phase 2: Single large memcpy for the entire macro chunk
      memcpy_vectorized(
          dst + stepOffset + macroOffset,
          recvBuffer + dataBufferOffset + macroOffset,
          macroBytes,
          group);

      // Phase 3: Batched release of all fine-grained chunks.
      // A single group.sync() fences the Phase 2 memcpy, then the leader
      // performs N release-stores. This replaces N ready_to_send() calls
      // (each with its own group.sync()) with 1 sync + N stores,
      // eliminating (N-1) redundant barriers.
      group.sync();
      if (group.is_leader()) {
        for (std::size_t i = 0; i < numFineChunks; ++i) {
          recvStates[stateOffset + firstChunk + i].release_to_send();
        }
      }
    });
  }
#endif
}

__global__ void p2pAsymmetricRecv(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    std::size_t recvChunkSize,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);
  asymmetric_recv_impl(p2p, group, dstBuff, nBytes, recvChunkSize, timeout);
}

__global__ void p2pAsymmetricBidirectional(
    P2pNvlTransportDevice p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    std::size_t recvChunkSize,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);

  // Partition groups into 2: half for send, half for recv
  auto [partition_id, subgroup] = group.partition_interleaved(2);
  if (partition_id == 0) {
    p2p.send(subgroup, sendBuff, nBytes, 0, timeout);
  } else {
    asymmetric_recv_impl(
        p2p, subgroup, recvBuff, nBytes, recvChunkSize, timeout);
  }
}

/**
 * asymmetric_send_impl - Batched send that coalesces fine-grained ChunkStates
 *
 * Replicates the structure of P2pNvlTransportDevice::send() but processes
 * chunks in "macro chunks" of size sendChunkSize. For each macro chunk:
 *   1. Wait for all constituent fine-grained ChunkStates (spin loops)
 *   2. Single large memcpy_vectorized for the full macro chunk (over NVLink)
 *   3. Release all constituent ChunkStates via batched release_to_recv
 *
 * @param p2p Transport device (configured at fine chunk granularity)
 * @param group ThreadGroup for cooperative processing
 * @param srcBuff Source buffer
 * @param nBytes Total bytes to send
 * @param sendChunkSize Coarse send chunk size (must be multiple of
 *        p2p.chunk_size())
 * @param timeout Timeout configuration
 */
__device__ __forceinline__ void asymmetric_send_impl(
    P2pNvlTransportDevice& p2p,
    ThreadGroup& group,
    void* srcBuff,
    std::size_t nBytes,
    std::size_t sendChunkSize,
    const Timeout& timeout) {
#ifdef __CUDA_ARCH__
  char* src = reinterpret_cast<char*>(srcBuff);
  char* sendBuffer = p2p.remote_data_buffer();
  ChunkState* sendStates = p2p.remote_state_buffer();

  const std::size_t transportChunkSize = p2p.chunk_size();
  const std::size_t dataBufferSize = p2p.data_buffer_size();
  const std::size_t pipelineDepth = p2p.pipeline_depth();

  // Validate: sendChunkSize must be an exact multiple of transport chunk size
  if (sendChunkSize % transportChunkSize != 0) {
    printf(
        "asymmetric_send: sendChunkSize (%llu) must be a multiple of "
        "transportChunkSize (%llu)\n",
        (unsigned long long)sendChunkSize,
        (unsigned long long)transportChunkSize);
    __trap();
  }

  // Validate: dataBufferSize must be an exact multiple of sendChunkSize
  // to ensure macro-chunk indexing doesn't overshoot the state buffer
  if (dataBufferSize % sendChunkSize != 0) {
    printf(
        "asymmetric_send: dataBufferSize (%llu) must be a multiple of "
        "sendChunkSize (%llu)\n",
        (unsigned long long)dataBufferSize,
        (unsigned long long)sendChunkSize);
    __trap();
  }

  const std::size_t chunksPerStep =
      (dataBufferSize + transportChunkSize - 1) / transportChunkSize;
  const std::size_t totalSteps = (nBytes + dataBufferSize - 1) / dataBufferSize;
  const std::size_t batchFactor = sendChunkSize / transportChunkSize;

  for (std::size_t stepId = 0; stepId < totalSteps; ++stepId) {
    const std::size_t pipelineIdx = stepId % pipelineDepth;
    const std::size_t dataBufferOffset = pipelineIdx * dataBufferSize;
    const std::size_t stateOffset = pipelineIdx * chunksPerStep;
    const std::size_t stepOffset = stepId * dataBufferSize;
    const std::size_t stepBytes = (stepOffset + dataBufferSize <= nBytes)
        ? dataBufferSize
        : nBytes - stepOffset;
    const std::size_t numMacroChunks =
        (stepBytes + sendChunkSize - 1) / sendChunkSize;

    group.for_each_item_contiguous(numMacroChunks, [&](uint32_t macroIdx) {
      const std::size_t macroOffset = macroIdx * sendChunkSize;
      const std::size_t macroBytes = (macroOffset + sendChunkSize <= stepBytes)
          ? sendChunkSize
          : stepBytes - macroOffset;
      if (macroBytes == 0) {
        return;
      }

      const std::size_t firstChunk = macroIdx * batchFactor;
      const std::size_t numFineChunks =
          (macroBytes + transportChunkSize - 1) / transportChunkSize;

      // Phase 1: Wait for all fine-grained chunks to be ready for send
      // (spin loops, no sync overhead). Poll sequentially since receiver
      // releases them in order.
      for (std::size_t i = 0; i < numFineChunks; ++i) {
        sendStates[stateOffset + firstChunk + i].wait_ready_to_send(
            group, timeout);
      }

      // Phase 2: Single large memcpy for the entire macro chunk (over NVLink)
      memcpy_vectorized(
          sendBuffer + dataBufferOffset + macroOffset,
          src + stepOffset + macroOffset,
          macroBytes,
          group);

      // Phase 3: Batched release of all fine-grained chunks.
      // A single group.sync() fences the Phase 2 memcpy, then the leader
      // performs N release-stores. This replaces N ready_to_recv() calls
      // (each with its own group.sync()) with 1 sync + N stores,
      // eliminating (N-1) redundant barriers.
      group.sync();
      if (group.is_leader()) {
        for (std::size_t i = 0; i < numFineChunks; ++i) {
          sendStates[stateOffset + firstChunk + i].release_to_recv(stepId, 0);
        }
      }
    });
  }
#endif
}

__global__ void p2pAsymmetricSend(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    std::size_t sendChunkSize,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);
  asymmetric_send_impl(p2p, group, srcBuff, nBytes, sendChunkSize, timeout);
}

__global__ void p2pAsymmetricSendBidirectional(
    P2pNvlTransportDevice p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    std::size_t sendChunkSize,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);

  // Partition groups into 2: half for send, half for recv
  auto [partition_id, subgroup] = group.partition_interleaved(2);
  if (partition_id == 0) {
    asymmetric_send_impl(
        p2p, subgroup, sendBuff, nBytes, sendChunkSize, timeout);
  } else {
    p2p.recv(subgroup, recvBuff, nBytes, 0, timeout);
  }
}

__global__ void p2pAsymmetricBothBidirectional(
    P2pNvlTransportDevice p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    std::size_t sendChunkSize,
    std::size_t recvChunkSize,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);

  // Partition groups into 2: half for send, half for recv
  auto [partition_id, subgroup] = group.partition_interleaved(2);
  if (partition_id == 0) {
    asymmetric_send_impl(
        p2p, subgroup, sendBuff, nBytes, sendChunkSize, timeout);
  } else {
    asymmetric_recv_impl(
        p2p, subgroup, recvBuff, nBytes, recvChunkSize, timeout);
  }
}

__global__ void p2pSignalBenchKernel(
    P2pNvlTransportDevice p2p,
    int nSteps,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);

  // Each group operates on its own signal slot for parallelism
  uint64_t signal_id = group.group_id;

  // Ping-pong signaling pattern:
  // - Signal peer's signal buffer (remote write)
  // - Wait on local signal buffer (local read)
  // multiple times before the other reads.
  for (int step = 1; step <= nSteps; ++step) {
    p2p.signal_threadgroup(group, signal_id, SignalOp::SIGNAL_ADD, 1);
    p2p.wait_signal_until_threadgroup(
        group, signal_id, CmpOp::CMP_EQ, static_cast<uint64_t>(step));
  }
}

__global__ void p2pSendOne(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  p2p.send_one(group, srcBuff, nBytes);
}

__global__ void
p2pRecvOne(P2pNvlTransportDevice p2p, void* dstBuff, SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  std::size_t nbytes;
  p2p.recv_one(group, dstBuff, &nbytes);
}

__global__ void p2pSendMultiple(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    DeviceSpan<const std::size_t> chunkSizes,
    DeviceSpan<const std::size_t> chunkIndices,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  p2p.send_multiple(group, srcBuff, chunkSizes, chunkIndices);
}

__global__ void p2pRecvMultiple(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    DeviceSpan<std::size_t> chunkSizes,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  p2p.recv_multiple(group, dstBuff, chunkSizes);
}

__global__ void p2pStreamSend(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);
  char* src = reinterpret_cast<char*>(srcBuff);
  auto send = p2p.send_stream(nBytes, 0, timeout);
  send.for_each_slot(group, [&](auto slot) {
    memcpy_vectorized(slot.data, src + slot.offset, slot.size, group);
  });
}

__global__ void p2pStreamRecv(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);
  char* dst = reinterpret_cast<char*>(dstBuff);
  auto recv = p2p.recv_stream(nBytes, 0, timeout);
  recv.for_each_ready_chunk(group, [&](auto chunk) {
    memcpy_vectorized(dst + chunk.offset, chunk.data, chunk.size, group);
  });
}

} // namespace comms::pipes::benchmark
