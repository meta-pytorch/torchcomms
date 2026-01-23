// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"

namespace comms::pipes::benchmark {

// Helper to compute global thread ID across all blocks
__device__ inline unsigned int getGlobalThreadId() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

// Helper to get the appropriate thread group
__device__ inline ThreadGroup getThreadGroup(bool useBlockGroups) {
  return useBlockGroups ? make_block_group() : make_warp_group();
}

__global__ void p2pSend(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    bool useBlockGroups) {
  auto group = getThreadGroup(useBlockGroups);
  p2p.send(group, srcBuff, nBytes);
}

__global__ void p2pRecv(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    bool useBlockGroups) {
  auto group = getThreadGroup(useBlockGroups);
  p2p.recv(group, dstBuff, nBytes);
}

__global__ void p2pSendTimed(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    TimingStats* stats,
    bool useBlockGroups) {
  auto group = getThreadGroup(useBlockGroups);
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
    bool useBlockGroups) {
  auto group = getThreadGroup(useBlockGroups);
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
    bool useBlockGroups) {
  auto group = getThreadGroup(useBlockGroups);

  // Partition groups into 2: half for send, half for recv
  auto [partition_id, subgroup] = group.partition_interleaved(2);
  if (partition_id == 0) {
    p2p.send(subgroup, sendBuff, nBytes);
  } else {
    p2p.recv(subgroup, recvBuff, nBytes);
  }
}

__global__ void p2pSignalBenchKernel(
    P2pNvlTransportDevice p2p,
    int nSteps,
    bool useBlockGroups) {
  auto group = getThreadGroup(useBlockGroups);

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

__global__ void allToAllvKernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos) {
  allToAllv(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports_per_rank,
      send_chunk_infos,
      recv_chunk_infos);
}

} // namespace comms::pipes::benchmark
