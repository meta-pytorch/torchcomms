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
// Barrier Benchmark Kernel Implementation
// =============================================================================

template <SyncScope G>
__global__ void multiPeerBarrierKernel(DeviceWindow dw, int nSteps) {
  auto group = makeGroup<G>();
  int slotId = computeSlotIndex<G>();

  for (int step = 0; step < nSteps; ++step) {
    dw.barrier(group, slotId);
  }
}

// =============================================================================
// Signal Ping-Pong Benchmark Kernel Implementation
// =============================================================================

template <SyncScope S>
__global__ void
multiPeerSignalPingPongKernel(DeviceWindow dw, int targetRank, int nSteps) {
  auto group = makeGroup<S>();
  int slotId = computeSlotIndex<S>();
  int myRank = dw.rank();

  // Ping-pong pattern:
  // - Even steps: Rank 0 signals, Rank 1 waits
  // - Odd steps: Rank 1 signals, Rank 0 waits
  //
  // Uses SIGNAL_ADD: each signal adds 1, wait for cumulative value.
  // After N signals from peer, peer's cumulative value = N.

  for (int step = 0; step < nSteps; ++step) {
    bool myTurnToSignal = ((step % 2) == myRank);

    if (myTurnToSignal) {
      dw.signal_peer(group, targetRank, slotId, SignalOp::SIGNAL_ADD, 1);
    } else {
      // Wait for (step/2 + 1) signals from peer
      uint64_t expectedValue = (step / 2) + 1;
      dw.wait_signal(group, slotId, CmpOp::CMP_GE, expectedValue);
    }
  }
}

// =============================================================================
// Signal-All Benchmark Kernel Implementation
// =============================================================================

template <SyncScope S>
__global__ void multiPeerSignalAllKernel(DeviceWindow dw, int nSteps) {
  auto group = makeGroup<S>();
  int slotId = computeSlotIndex<S>();

  // In the per-signal model, all (nRanks-1) peers write to the same slot.
  // After step+1 iterations, accumulated value = (nRanks-1) * (step+1).
  int nPeers = dw.n_ranks() - 1;

  for (int step = 0; step < nSteps; ++step) {
    // Signal all peers
    dw.signal_all(group, slotId, SignalOp::SIGNAL_ADD, 1);

    // Wait for accumulated signals from all peers
    dw.wait_signal(
        group,
        slotId,
        CmpOp::CMP_GE,
        static_cast<uint64_t>(nPeers * (step + 1)));
  }
}

// =============================================================================
// Explicit Template Instantiations for Barrier Kernels
// =============================================================================

// Barrier kernels
template __global__ void multiPeerBarrierKernel<SyncScope::WARP>(
    DeviceWindow,
    int);
template __global__ void multiPeerBarrierKernel<SyncScope::BLOCK>(
    DeviceWindow,
    int);

// =============================================================================
// Explicit Template Instantiations for Signal Kernels
// =============================================================================

// Signal ping-pong kernels
template __global__ void
multiPeerSignalPingPongKernel<SyncScope::WARP>(DeviceWindow, int, int);
template __global__ void
multiPeerSignalPingPongKernel<SyncScope::BLOCK>(DeviceWindow, int, int);

// Signal-all kernels
template __global__ void multiPeerSignalAllKernel<SyncScope::WARP>(
    DeviceWindow,
    int);
template __global__ void multiPeerSignalAllKernel<SyncScope::BLOCK>(
    DeviceWindow,
    int);

// =============================================================================
// Put Ping-Pong Benchmark Kernel Implementation
// =============================================================================

template <SyncScope S>
__global__ void multiPeerPutPingPongKernel(
    DeviceWindow dw,
    int targetRank,
    const void* localSrc,
    void* remoteDst,
    std::size_t nbytes,
    int nSteps) {
  auto group = makeGroup<S>();
  int slotId = computeSlotIndex<S>();
  int myRank = dw.rank();

  // Ping-pong pattern using put() + signal_peer():
  // - Even steps: Rank 0 puts data, Rank 1 waits
  // - Odd steps: Rank 1 puts data, Rank 0 waits
  //
  // Uses SIGNAL_ADD: each signal adds 1, wait for cumulative value.

  for (int step = 0; step < nSteps; ++step) {
    bool myTurnToPut = ((step % 2) == myRank);

    if (myTurnToPut) {
      // Put data to peer's buffer
      dw.put(targetRank, group, remoteDst, localSrc, nbytes);
      // Sync to ensure put completes before signaling
      group.sync();
      // Signal peer that data is ready
      dw.signal_peer(group, targetRank, slotId, SignalOp::SIGNAL_ADD, 1);
    } else {
      // Wait for (step/2 + 1) signals from peer
      uint64_t expectedValue = (step / 2) + 1;
      dw.wait_signal(group, slotId, CmpOp::CMP_GE, expectedValue);
    }
  }
}

// =============================================================================
// Put+Signal Ping-Pong Benchmark Kernel Implementation
// =============================================================================

template <SyncScope S>
__global__ void multiPeerPutSignalPingPongKernel(
    DeviceWindow dw,
    int targetRank,
    const void* localSrc,
    void* remoteDst,
    std::size_t nbytes,
    int nSteps) {
  auto group = makeGroup<S>();
  int slotId = computeSlotIndex<S>();
  int myRank = dw.rank();

  // Ping-pong pattern using put_signal():
  // - Even steps: Rank 0 puts+signals, Rank 1 waits
  // - Odd steps: Rank 1 puts+signals, Rank 0 waits
  //
  // Uses put_signal() convenience API which combines put + signal_peer.

  for (int step = 0; step < nSteps; ++step) {
    bool myTurnToPut = ((step % 2) == myRank);

    if (myTurnToPut) {
      // Combined put + signal
      dw.put_signal(targetRank, group, remoteDst, localSrc, nbytes, slotId, 1);
    } else {
      // Wait for (step/2 + 1) signals from peer
      uint64_t expectedValue = (step / 2) + 1;
      dw.wait_signal(group, slotId, CmpOp::CMP_GE, expectedValue);
    }
  }
}

// =============================================================================
// Explicit Template Instantiations for Put Kernels
// =============================================================================

// Put ping-pong kernels
template __global__ void multiPeerPutPingPongKernel<SyncScope::WARP>(
    DeviceWindow,
    int,
    const void*,
    void*,
    std::size_t,
    int);
template __global__ void multiPeerPutPingPongKernel<SyncScope::BLOCK>(
    DeviceWindow,
    int,
    const void*,
    void*,
    std::size_t,
    int);

// Put+signal ping-pong kernels
template __global__ void multiPeerPutSignalPingPongKernel<SyncScope::WARP>(
    DeviceWindow,
    int,
    const void*,
    void*,
    std::size_t,
    int);
template __global__ void multiPeerPutSignalPingPongKernel<SyncScope::BLOCK>(
    DeviceWindow,
    int,
    const void*,
    void*,
    std::size_t,
    int);

} // namespace comms::pipes::benchmark
