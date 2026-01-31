// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"

namespace comms::pipes::benchmark {

// Helper to compute global thread ID across all blocks
__device__ inline unsigned int getGlobalThreadId() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void p2pSend(
    P2pNvlTransportDevice* p2p,
    void* srcBuff,
    std::size_t nBytes,
    uint32_t call_index,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);
  p2p->send(group, srcBuff, nBytes, call_index, timeout);
}

__global__ void p2pRecv(
    P2pNvlTransportDevice* p2p,
    void* dstBuff,
    std::size_t nBytes,
    uint32_t call_index,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);
  p2p->recv(group, dstBuff, nBytes, call_index, timeout);
}

__global__ void p2pSendTimed(
    P2pNvlTransportDevice* p2p,
    void* srcBuff,
    std::size_t nBytes,
    uint32_t call_index,
    TimingStats* stats,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  unsigned int globalThreadId = getGlobalThreadId();

  // Only first thread globally records start time
  if (globalThreadId == 0) {
    stats->startCycle = clock64();
  }

  p2p->send(group, srcBuff, nBytes, call_index);

  // Only first thread globally records end time
  if (globalThreadId == 0) {
    unsigned long long end = clock64();
    stats->endCycle = end;
    stats->totalCycles = end - stats->startCycle;
  }
}

__global__ void p2pRecvTimed(
    P2pNvlTransportDevice* p2p,
    void* dstBuff,
    std::size_t nBytes,
    uint32_t call_index,
    TimingStats* stats,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  unsigned int globalThreadId = getGlobalThreadId();

  // Only first thread globally records start time
  if (globalThreadId == 0) {
    stats->startCycle = clock64();
  }

  p2p->recv(group, dstBuff, nBytes, call_index);

  // Only first thread globally records end time
  if (globalThreadId == 0) {
    unsigned long long end = clock64();
    stats->endCycle = end;
    stats->totalCycles = end - stats->startCycle;
  }
}

// Need to add launch_bounds to avoid "too many resources requested for launch"
// error
__global__ __launch_bounds__(512, 1) void p2pBidirectional(
    P2pNvlTransportDevice* p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    uint32_t call_index,
    SyncScope groupScope,
    Timeout timeout) {
  timeout.start();
  auto group = make_thread_group(groupScope);

  // Partition groups into 2: half for send, half for recv
  auto [partition_id, subgroup] = group.partition_interleaved(2);
  if (partition_id == 0) {
    p2p->send(subgroup, sendBuff, nBytes, call_index, timeout);
  } else {
    p2p->recv(subgroup, recvBuff, nBytes, call_index, timeout);
  }
}

__global__ void p2pSignalBenchKernel(
    P2pNvlTransportDevice* p2p,
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
    p2p->signal_threadgroup(group, signal_id, SignalOp::SIGNAL_ADD, 1);
    p2p->wait_signal_until_threadgroup(
        group, signal_id, CmpOp::CMP_EQ, static_cast<uint64_t>(step));
  }
}

} // namespace comms::pipes::benchmark
