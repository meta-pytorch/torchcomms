// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Dummy translation unit so `:kernel_headers` qualifies as a
// `comms_gpu_cpp_library` (which requires at least one src file). The
// library exists to expose Exception.cuh / Launch.cuh / KernelConfigs.cuh /
// KernelUtils.cuh / EpBuffer.cuh through the GPU build pipeline so
// downstream `.cu` consumers see them HIPified on AMD.

#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Exception.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Launch.cuh"

namespace comms::prims::moe_ep::kernels::detail {
// No-op anchor symbol so the .a isn't completely empty.
inline int kernel_headers_anchor() {
  return 0;
}
} // namespace comms::prims::moe_ep::kernels::detail
