// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// HRDWRingBuffer Triton Device API — C-style declarations for LLVM bitcode
//
// Compiled to LLVM bitcode with clang for linking with Triton kernels (same
// pattern as comms/torchcomms/triton/colltrace.h). Triton kernels declare
// these via @core.extern and call via tl.extern_elementwise to write
// (timestamp, tag) entries into an HRDWRingBuffer<uint64_t, Device>.
//
// All functions use void* opaque handles + flat primitive args to avoid C++
// type dependencies. The implementation in hrdw_ring_buffer.cu casts back
// to typed handles and forwards to hrdwRingBufferWrite.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Write one entry into a Device-scope HRDWRingBuffer<uint64_t>.
//
// The ring handle (ring_ptr, write_index_ptr, mask, shift) comes from
// RingBuffer.device_handle() on the Python side. The data field is a
// caller-supplied uint64 tag — pack application-specific fields (kernel ID,
// phase, op type, etc.) into 64 bits on the caller side.
//
// The underlying write atomically claims a slot (device-scope atomicAdd)
// and stores {globaltimer_ns, data}. No internal thread/block gating —
// the caller decides which threads write (typically program_id == 0).
//
// No-op when ring_ptr is null.
__device__ void device_hrdw_ring_write(
    void* ring_ptr, // HRDWEntry<uint64_t, Device>*
    void* write_index_ptr, // uint64_t*
    int mask,
    int shift,
    long long data); // uint64_t tag

#ifdef __cplusplus
}
#endif
