// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_RING_H_
#define CTRAN_GPE_RING_H_

#include <cstdint>

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/utils/colltrace/ColltraceDeviceHandle.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"

namespace ctran::gpe {

// The GPE dispatch ring is System-scope (host worker polls concurrently) and
// Blocking (lossless): a captured kernel that publishes its cmd id must never
// have that publish silently overwritten before the worker consumes it, or the
// cmd would never fire and its kernel would hang on KERNEL_TERMINATE. Declaring
// the ring, reader, and handle from these shared constants makes "the worker
// sees every publish" a structural property — the Blocking ring allocates the
// consumer cursor and its reader publishes it, inseparably.
inline constexpr auto kGpeRingScope =
    hrdw_ring_buffer::MemoryCoherenceScope::System;
inline constexpr auto kGpeRingPolicy = hrdw_ring_buffer::WritePolicy::Blocking;

// Comm-local command id assigned at submit() time. 64-bit so a long-lived comm
// can never wrap it back through the reserved 0 ("unassigned") sentinel or onto
// a still-live id, even at millions of captured cmds/sec for the process
// lifetime. It fits the ring's 16-byte entry either way, so the width is free.
using GpeCmdId = uint64_t;

// Ring entries carry a comm-local command id assigned at submit() time. The
// GPE worker maps the id back to its CtranGpeCmd via the per-comm registry.
// (GpeRingReader is host-only and lives in CtranGpeImpl.h — this header is
// device-included and must not pull in the host reader.)
using GpeRing =
    hrdw_ring_buffer::HRDWRingBuffer<GpeCmdId, kGpeRingScope, kGpeRingPolicy>;
using GpeRingHandle = hrdw_ring_buffer::
    HRDWRingBufferDeviceHandle<GpeCmdId, kGpeRingScope, kGpeRingPolicy>;

// Per-cmd ring header. The GPE fills it host-side at submit() on the
// device-ring path (per-cmd, so no concurrent-kernel race); the kernel's
// KernelStartGpe prologue reads it via KernelFlagDev::gpeHdr and publishes
// cmdId to the ring.
struct GpeKernelFlagHeader {
  GpeRingHandle ring{};
  GpeCmdId cmdId{0};
  uint32_t enabled{0};
};

// Device-facing view of a per-cmd kernel flag object: the per-block start/stop
// flags plus the ring headers, laid out in pinned host memory. Every GPE kernel
// takes a KernelFlagDev* as its first argument and reads gpeHdr/colltraceHdr
// directly — no pointer-offset recovery. KernelFlagItem (host) embeds this as
// its first member, so &kernelFlag->dev is the pointer handed to the kernel.
// colltraceHdr is armed host-side at submit() for the kernels that bound a
// logical collective; a default (null-ring) handle means unarmed, so the
// in-kernel emit no-ops. See meta::comms::colltrace::ColltraceDeviceHandle.
struct KernelFlagDev {
  volatile int flag_[CTRAN_ALGO_MAX_THREAD_BLOCKS];
  GpeKernelFlagHeader gpeHdr{};
  meta::comms::colltrace::ColltraceDeviceHandle colltraceHdr{};
};

} // namespace ctran::gpe

#endif // CTRAN_GPE_RING_H_
