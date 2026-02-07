// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/benchmarks/MultiPeerBenchmark.cuh"

namespace comms::pipes::benchmark {

// =============================================================================
// Thread Group Helpers
// =============================================================================

template <SyncScope S>
__device__ __forceinline__ auto makeGroup() {
  return make_thread_group(S);
}

// Compute unique slot index for current thread group
// Returns int to match barrier/signal slot API expectations
template <SyncScope S>
__device__ __forceinline__ int computeSlotIndex() {
  if constexpr (S == SyncScope::BLOCK) {
    // One slot per block
    return static_cast<int>(blockIdx.x);
  } else {
    // One slot per warp - use comms::device::kWarpSize
    auto warpsPerBlock = blockDim.x / comms::device::kWarpSize;
    auto warpIdInBlock = threadIdx.x / comms::device::kWarpSize;
    return static_cast<int>(blockIdx.x * warpsPerBlock + warpIdInBlock);
  }
}

// =============================================================================
// Explicit Template Instantiations for Helper Functions
// =============================================================================

// Note: These helpers are used by benchmark kernels added in later diffs.
// The explicit instantiations ensure the templates are available for linking.

template __device__ auto makeGroup<SyncScope::WARP>();
template __device__ auto makeGroup<SyncScope::BLOCK>();

template __device__ int computeSlotIndex<SyncScope::WARP>();
template __device__ int computeSlotIndex<SyncScope::BLOCK>();

// =============================================================================
// Signal Ping-Pong Benchmark Kernel Implementation
// =============================================================================

template <SyncScope S>
__global__ void multiPeerSignalPingPongKernel(
    MultiPeerDeviceTransport transport,
    int peerIndex,
    int nSteps) {
  auto group = makeGroup<S>();
  int slotId = computeSlotIndex<S>();
  int myRank = transport.rank();

  // For 2-rank ping-pong, there's exactly one peer at index 0
  assert(peerIndex == 0);

  // Ping-pong pattern:
  // - Even steps: Rank 0 signals, Rank 1 waits
  // - Odd steps: Rank 1 signals, Rank 0 waits
  //
  // Uses SIGNAL_ADD: each signal adds 1, wait for cumulative value.
  // After N signals from peer, peer's cumulative value = N.

  for (int step = 0; step < nSteps; ++step) {
    bool myTurnToSignal = ((step % 2) == myRank);

    if (myTurnToSignal) {
      transport.signal_peer(peerIndex, group, slotId, SignalOp::SIGNAL_ADD, 1);
    } else {
      // Wait for (step/2 + 1) signals from peer
      uint64_t expectedValue = (step / 2) + 1;
      transport.wait_signal(group, slotId, CmpOp::CMP_GE, expectedValue);
    }
  }
}

// =============================================================================
// Signal-All Benchmark Kernel Implementation
// =============================================================================

template <SyncScope S>
__global__ void multiPeerSignalAllKernel(
    MultiPeerDeviceTransport transport,
    int nSteps) {
  auto group = makeGroup<S>();
  int slotId = computeSlotIndex<S>();

  // In the per-signal model, all (nRanks-1) peers write to the same slot.
  // After step+1 iterations, accumulated value = (nRanks-1) * (step+1).
  int nPeers = transport.n_ranks() - 1;

  for (int step = 0; step < nSteps; ++step) {
    // Signal all peers
    transport.signal_all(group, slotId, SignalOp::SIGNAL_ADD, 1);

    // Wait for accumulated signals from all peers
    transport.wait_signal(
        group,
        slotId,
        CmpOp::CMP_GE,
        static_cast<uint64_t>(nPeers * (step + 1)));
  }
}

// =============================================================================
// Explicit Template Instantiations for Signal Kernels
// =============================================================================

// Signal ping-pong kernels
template __global__ void multiPeerSignalPingPongKernel<SyncScope::WARP>(
    MultiPeerDeviceTransport,
    int);
template __global__ void multiPeerSignalPingPongKernel<SyncScope::BLOCK>(
    MultiPeerDeviceTransport,
    int);

// Signal-all kernels
template __global__ void multiPeerSignalAllKernel<SyncScope::WARP>(
    MultiPeerDeviceTransport,
    int);
template __global__ void multiPeerSignalAllKernel<SyncScope::BLOCK>(
    MultiPeerDeviceTransport,
    int);

} // namespace comms::pipes::benchmark
