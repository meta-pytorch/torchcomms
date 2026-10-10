/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "registered_alltoall_kernels.h"

namespace rcclx::relay {
namespace {

constexpr int kActive = kRegisteredAllToAllActiveRanks;
constexpr int kThreads = kRegisteredAllToAllThreads;
constexpr size_t kVectorBytes = 16;

__device__ __forceinline__ bool epochReached(uint32_t got, uint32_t want) {
  return (got - want) <= (uint32_t(-1) >> 1);
}

__device__ __forceinline__ void storeFlag(uint32_t* address, uint32_t value) {
  __scoped_atomic_store_n(
      address, value, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
}

// Polls with relaxed loads and pays one system-scope acquire (cache
// invalidate) once the flag arrives, not one per poll.
__device__ __forceinline__ void waitFlag(
    const uint32_t* address,
    uint32_t want) {
  while (!epochReached(
      __scoped_atomic_load_n(address, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM),
      want)) {
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

__device__ __forceinline__ void drainVectorMemory() {
  constexpr unsigned kGfx9VmcntZero = 0x0F70;
  __builtin_amdgcn_s_waitcnt(kGfx9VmcntZero);
}

// Copies rows of rowBytes between two strided row layouts; part/parts split
// the 16-byte vectors across cooperating CTAs.
__device__ __forceinline__ void copyRows(
    unsigned char* __restrict__ dst,
    size_t dstRowStride,
    const unsigned char* __restrict__ src,
    size_t srcRowStride,
    int rows,
    size_t rowBytes,
    int part,
    int parts) {
  const size_t vectorsPerRow = rowBytes / kVectorBytes;
  const size_t total = static_cast<size_t>(rows) * vectorsPerRow;
  for (size_t i = static_cast<size_t>(part) * kThreads + threadIdx.x; i < total;
       i += static_cast<size_t>(parts) * kThreads) {
    const size_t row = i / vectorsPerRow;
    const size_t vector = i % vectorsPerRow;
    *reinterpret_cast<uint4*>(
        dst + row * dstRowStride + vector * kVectorBytes) =
        *reinterpret_cast<const uint4*>(
            src + row * srcRowStride + vector * kVectorBytes);
  }
}

// Every CTA drains its memory traffic and takes a ticket; the last one runs
// the closing step. Returns true in the last CTA's thread 0.
__device__ __forceinline__ bool lastCta(uint32_t* ticket) {
  drainVectorMemory();
  __syncthreads();
  __shared__ bool last;
  if (threadIdx.x == 0) {
    const uint32_t previous = __scoped_atomic_fetch_add(
        ticket, 1u, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE);
    last = previous + 1 == gridDim.x;
    if (last) {
      __scoped_atomic_store_n(
          ticket, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
  }
  __syncthreads();
  return last && threadIdx.x == 0;
}

__device__ __forceinline__ uint32_t
readEpoch(const RegisteredAllToAllFlags* flags) {
  __shared__ uint32_t epoch;
  if (threadIdx.x == 0) {
    epoch = flags->calls + 1u;
  }
  __syncthreads();
  return epoch;
}

// Grid: kActive * directCtasPerPeer direct CTAs (source s = (me + k) % kActive,
// k = 0 is the local copy), then (kActive - 1) * helpers * relayCtasPerGroup
// relay pullers, then as many relay pushers. A pusher copies this rank's
// relayed rows for one destination into its own staging on one helper GPU and
// flags the destination per chunk; the destination's puller copies them into
// its receive buffer. Nothing runs on the helper GPUs. CTA 0 publishes the
// start epoch; the last CTA publishes "done reading" to every source and waits
// until every peer is done reading this rank's send buffer and staging, so the
// caller may overwrite the send buffer after the kernel returns and the next
// call may overwrite the staging. Every value derives from the device-resident
// call counter, so the kernel is safe to capture and replay.
__global__ void __launch_bounds__(kThreads)
    registeredAllToAllActiveKernel(RegisteredAllToAllArgs args) {
  const RegisteredAllToAllGeometry& g = args.geometry;
  const int me = args.index;
  RegisteredAllToAllFlags* mine = args.activeFlags[me];
  const uint32_t epoch = readEpoch(mine);

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int peer = 0; peer < kActive; ++peer) {
      if (peer != me) {
        storeFlag(&args.activeFlags[peer]->start[me], epoch);
      }
    }
  }

  const int directCtas = kActive * g.directCtasPerPeer;
  const int relayCtas = (kActive - 1) * g.helpers * g.relayCtasPerGroup;
  const int chunks =
      g.relayRows == 0 ? 0 : (g.relayRows + g.chunkRows - 1) / g.chunkRows;
  const auto block = static_cast<int>(blockIdx.x);
  if (block < directCtas) {
    const int offset = block / g.directCtasPerPeer;
    const int part = block % g.directCtasPerPeer;
    const int source = (me + offset) % kActive;
    if (source != me && threadIdx.x == 0) {
      waitFlag(&mine->start[source], epoch);
    }
    __syncthreads();
    copyRows(
        args.recv + source * g.recvPeerStride,
        g.recvRowStride,
        args.send[source] + me * g.sendPeerStride,
        g.sendRowStride,
        source == me ? g.rows : g.directRows,
        g.rowBytes,
        part,
        g.directCtasPerPeer);
  } else if (block < directCtas + relayCtas) {
    const int relay = block - directCtas;
    const int group = relay / g.relayCtasPerGroup;
    const int lane = relay % g.relayCtasPerGroup;
    const int source = (me + 1 + group / g.helpers) % kActive;
    const int h = group % g.helpers;
    const unsigned char* staging = args.staging[source][h] +
        static_cast<size_t>(me) * g.relayRows * g.rowBytes;
    unsigned char* recv = args.recv + source * g.recvPeerStride +
        static_cast<size_t>(g.directRows + h * g.relayRows) * g.recvRowStride;
    for (int c = lane; c < chunks; c += g.relayCtasPerGroup) {
      if (threadIdx.x == 0) {
        waitFlag(&mine->chunk[source][h][c], epoch);
      }
      __syncthreads();
      const int row = c * g.chunkRows;
      copyRows(
          recv + static_cast<size_t>(row) * g.recvRowStride,
          g.recvRowStride,
          staging + static_cast<size_t>(row) * g.rowBytes,
          g.rowBytes,
          min(g.chunkRows, g.relayRows - row),
          g.rowBytes,
          0,
          1);
      __syncthreads();
    }
  } else {
    const int push = block - directCtas - relayCtas;
    const int group = push / g.relayCtasPerGroup;
    const int lane = push % g.relayCtasPerGroup;
    const int dest = (me + 1 + group / g.helpers) % kActive;
    const int h = group % g.helpers;
    const unsigned char* from = args.send[me] + dest * g.sendPeerStride +
        static_cast<size_t>(g.directRows + h * g.relayRows) * g.sendRowStride;
    unsigned char* staging = args.staging[me][h] +
        static_cast<size_t>(dest) * g.relayRows * g.rowBytes;
    for (int c = lane; c < chunks; c += g.relayCtasPerGroup) {
      const int row = c * g.chunkRows;
      copyRows(
          staging + static_cast<size_t>(row) * g.rowBytes,
          g.rowBytes,
          from + static_cast<size_t>(row) * g.sendRowStride,
          g.sendRowStride,
          min(g.chunkRows, g.relayRows - row),
          g.rowBytes,
          0,
          1);
      drainVectorMemory();
      __syncthreads();
      if (threadIdx.x == 0) {
        storeFlag(&args.activeFlags[dest]->chunk[me][h][c], epoch);
      }
    }
  }

  if (!lastCta(&mine->ticket)) {
    return;
  }
  for (int peer = 0; peer < kActive; ++peer) {
    if (peer != me) {
      storeFlag(&args.activeFlags[peer]->done[me], epoch);
    }
  }
  for (int peer = 0; peer < kActive; ++peer) {
    if (peer != me) {
      waitFlag(&mine->done[peer], epoch);
    }
  }
  mine->calls = epoch;
}

__global__ void registeredAllToAllProbeKernel(
    const RegisteredAllToAllProbe* probes,
    int count,
    uint64_t* out) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) {
    return;
  }
  const RegisteredAllToAllProbe probe = probes[i];
  uint4 value;
  if (probe.atomic != 0) {
    const auto* words = reinterpret_cast<const uint32_t*>(probe.address);
    value.x = __scoped_atomic_load_n(
        words + 0, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    value.y = __scoped_atomic_load_n(
        words + 1, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    value.z = __scoped_atomic_load_n(
        words + 2, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    value.w = __scoped_atomic_load_n(
        words + 3, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
  } else {
    value = *reinterpret_cast<const uint4*>(probe.address);
  }
  out[2 * i] = (static_cast<uint64_t>(value.y) << 32) | value.x;
  out[2 * i + 1] = (static_cast<uint64_t>(value.w) << 32) | value.z;
}

} // namespace

size_t registeredAllToAllStagingBytes(
    const RegisteredAllToAllGeometry& geometry) {
  return static_cast<size_t>(kActive) * geometry.relayRows * geometry.rowBytes;
}

hipError_t launchRegisteredAllToAllActive(
    const RegisteredAllToAllArgs& args,
    hipStream_t stream) {
  const RegisteredAllToAllGeometry& g = args.geometry;
  const int blocks = kActive * g.directCtasPerPeer +
      2 * (kActive - 1) * g.helpers * g.relayCtasPerGroup;
  hipLaunchKernelGGL(
      registeredAllToAllActiveKernel,
      dim3(blocks),
      dim3(kThreads),
      0,
      stream,
      args);
  return hipPeekAtLastError();
}

hipError_t launchRegisteredAllToAllProbe(
    const RegisteredAllToAllProbe* probes,
    int count,
    uint64_t* out,
    hipStream_t stream) {
  if (count == 0) {
    return hipSuccess;
  }
  constexpr int kProbeThreads = 64;
  hipLaunchKernelGGL(
      registeredAllToAllProbeKernel,
      dim3((count + kProbeThreads - 1) / kProbeThreads),
      dim3(kProbeThreads),
      0,
      stream,
      probes,
      count,
      out);
  return hipPeekAtLastError();
}

} // namespace rcclx::relay
