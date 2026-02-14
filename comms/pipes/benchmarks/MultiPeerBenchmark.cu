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

} // namespace comms::pipes::benchmark
