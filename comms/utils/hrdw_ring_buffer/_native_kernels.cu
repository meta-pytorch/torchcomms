// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Explicit template instantiations for launchRingBufferWrite<uint64_t, *>.
// Linked by the pybinding .cpp so both Device and System scope rings can
// expose a host-side write() helper for testing. Compiled for both CUDA
// and HIP via the BUCK target's `cuda_srcs` + `hip_srcs` listing — the
// device-side `HrdwRingBufferWriter<DataT, C>::write` in HRDWRingBuffer.h
// has an explicit `#if defined(__HIPCC__)` branch that prints + aborts
// instead of executing the `atom.exch.b128` PTX (no HIP equivalent), so
// the HIP-side instantiations link cleanly even though the host helper
// is functionally unsupported under AMD.

#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"

namespace hrdw_ring_buffer {

template cudaError_t
launchRingBufferWrite<uint64_t, MemoryCoherenceScope::Device>(
    cudaStream_t,
    HRDWEntry<uint64_t, MemoryCoherenceScope::Device>*,
    uint64_t*,
    uint32_t,
    uint32_t,
    uint64_t);

template cudaError_t
launchRingBufferWrite<uint64_t, MemoryCoherenceScope::System>(
    cudaStream_t,
    HRDWEntry<uint64_t, MemoryCoherenceScope::System>*,
    uint64_t*,
    uint32_t,
    uint32_t,
    uint64_t);

} // namespace hrdw_ring_buffer
