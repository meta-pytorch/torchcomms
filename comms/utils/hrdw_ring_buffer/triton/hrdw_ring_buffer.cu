// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// HRDWRingBuffer Triton Device API — bitcode implementation
//
// Compiled to LLVM bitcode (libhrdw_ring_buffer.bc) for linking into Triton
// kernels. Casts void* handles back to typed HRDWRingBufferDeviceHandle and
// forwards to handle.write(...) (which dispatches to
// HrdwRingBufferWriter<DataT, Device>::write) — single source of truth for
// the ring-write protocol.

#include <cstdint>

#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"
#include "comms/utils/hrdw_ring_buffer/triton/hrdw_ring_buffer.h"

using namespace hrdw_ring_buffer;

extern "C" {

__device__ void device_hrdw_ring_write(
    void* ring_ptr,
    void* write_index_ptr,
    int mask,
    int shift,
    long long data) {
  if (ring_ptr == nullptr) {
    return;
  }
  // Single-lane the ring write. Triton compiles a uniform `extern_elementwise`
  // call into one invocation per thread in the CTA, so without this guard a
  // single `if pid == 0: emit(...)` block in the kernel produces `num_warps *
  // 32` ring writes per program. Gating on `threadIdx == 0` makes the helper
  // idempotent at the CTA level, which is the only sensible call pattern for
  // a coarse start/end marker.
  if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
    return;
  }
  using DataT = uint64_t;
  constexpr auto Scope = MemoryCoherenceScope::Device;

  HRDWRingBufferDeviceHandle<DataT, Scope> handle{
      static_cast<HRDWEntry<DataT, Scope>*>(ring_ptr),
      static_cast<uint64_t*>(write_index_ptr),
      static_cast<uint32_t>(mask),
      static_cast<uint32_t>(shift)};
  handle.write(static_cast<uint64_t>(data));
}

} // extern "C"
