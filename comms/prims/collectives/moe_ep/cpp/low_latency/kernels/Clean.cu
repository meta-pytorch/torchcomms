// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/moe_ep/cpp/low_latency/kernels/Clean.cuh"

#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Exception.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Launch.cuh"

// `clean_low_latency_buffer` zeroes the send/recv signal regions between
// iterations. The multi-rank barrier is provided by the IBGDA device
// transport once it is wired into `LowLatencyRuntime`; the single-rank fast
// path skips it (a cross-rank barrier is a no-op when numRanks == 1).
//
// The host wrapper passes through to a single launch; the kernel does
// `kNumThreads`-strided memsets across the two clean regions.

namespace comms::prims::moe_ep::kernels {

namespace {

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__
    void clean_low_latency_buffer_kernel(
        std::int64_t* clean_0,
        int num_clean_int_0,
        std::int64_t* clean_1,
        int num_clean_int_1) {
  // Pre-barrier: stable cross-rank fence before clearing. With a single
  // rank this degenerates to __syncthreads().
  __syncthreads();

  const int thread_id = static_cast<int>(threadIdx.x);

  // Clean region 0
  for (int i = thread_id; i < num_clean_int_0; i += kNumThreads) {
    clean_0[i] = 0;
  }

  // Clean region 1
  if (clean_1 != nullptr) {
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads) {
      clean_1[i] = 0;
    }
  }

  // Post-barrier
  __syncthreads();
}

} // namespace

void clean_low_latency_buffer(
    std::int64_t* clean_0,
    int num_clean_int_0,
    std::int64_t* clean_1,
    int num_clean_int_1,
    int /*rank*/,
    int /*num_ranks*/,
    int* /*mask_buffer_ptr*/,
    int* /*sync_buffer_ptr*/,
    cudaStream_t stream) {
  constexpr int kNumThreads = 256;
  if (num_clean_int_0 <= 0 && num_clean_int_1 <= 0) {
    return;
  }

  SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
  LAUNCH_KERNEL_NON_COOPERATIVE(
      &cfg,
      clean_low_latency_buffer_kernel<kNumThreads>,
      clean_0,
      num_clean_int_0,
      clean_1,
      num_clean_int_1);
}

} // namespace comms::prims::moe_ep::kernels
