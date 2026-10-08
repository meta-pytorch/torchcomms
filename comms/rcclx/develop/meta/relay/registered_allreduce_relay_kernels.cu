/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "registered_allreduce_relay_kernels.h"

namespace rcclx::relay {
namespace {

constexpr int kThreads = kRegisteredRelayThreads;
constexpr unsigned kGfx9VmcntZero = 0x0F70;
constexpr int kPerVector = 8;

typedef unsigned int v4u __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf16ToFloat(uint16_t v) {
  return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

__device__ __forceinline__ uint16_t floatToBf16(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    return 0x7fc0u;
  }
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

__device__ __forceinline__ bool reached(uint32_t got, uint32_t want) {
  return (got - want) <= (uint32_t(-1) >> 1);
}

__device__ __forceinline__ void waitFlag(const uint32_t* flag, uint32_t want) {
  while (!reached(
      __scoped_atomic_load_n(flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM),
      want)) {
  }
  __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void storeFlag(uint32_t* flag, uint32_t value) {
  __scoped_atomic_store_n(flag, value, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
}

// Rank 0's value first, so both ranks produce the same bits.
__device__ __forceinline__ v4u combine(v4u mine, v4u theirs, int rank) {
  uint16_t a[kPerVector];
  uint16_t b[kPerVector];
  uint16_t r[kPerVector];
  __builtin_memcpy(a, rank == 0 ? &mine : &theirs, 16);
  __builtin_memcpy(b, rank == 0 ? &theirs : &mine, 16);
#pragma unroll
  for (int e = 0; e < kPerVector; ++e) {
    r[e] = floatToBf16(bf16ToFloat(a[e]) + bf16ToFloat(b[e]));
  }
  v4u out;
  __builtin_memcpy(&out, r, 16);
  return out;
}

__device__ void reduceSlice(
    const v4u* mine,
    const v4u* __restrict__ theirs,
    v4u* out,
    size_t vectors,
    int rank) {
  for (size_t v = threadIdx.x; v < vectors; v += kThreads) {
    out[v] = combine(mine[v], __builtin_nontemporal_load(theirs + v), rank);
  }
}

__device__ void
pushSlice(const v4u* __restrict__ src, v4u* __restrict__ dst, size_t vectors) {
  for (size_t v = threadIdx.x; v < vectors; v += kThreads) {
    __builtin_nontemporal_store(src[v], dst + v);
  }
}

__device__ __forceinline__ size_t
directSlice(const RegisteredRelayArgs& a, size_t k) {
  const size_t cycle = static_cast<size_t>(a.directWeight) + a.helpers;
  return (k / a.directWeight) * cycle + k % a.directWeight;
}

__device__ __forceinline__ size_t
helperSlice(const RegisteredRelayArgs& a, int h, size_t k) {
  const size_t cycle = static_cast<size_t>(a.directWeight) + a.helpers;
  return k * cycle + a.directWeight + h;
}

__device__ __forceinline__ bool lastCta(uint32_t* ticket) {
  __builtin_amdgcn_s_waitcnt(kGfx9VmcntZero);
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

__global__ void __launch_bounds__(kThreads)
    registeredRelayAllReduceKernel(RegisteredRelayArgs a) {
  RegisteredRelayFlags* mine = a.myFlags;
  RegisteredRelayFlags* peer = a.peerFlags;
  __shared__ uint32_t epochShared;
  if (threadIdx.x == 0) {
    epochShared = mine->calls.value + 1u;
  }
  __syncthreads();
  const uint32_t epoch = epochShared;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    storeFlag(&peer->start.value, epoch);
  }

  const v4u* input = static_cast<const v4u*>(a.mine);
  v4u* output = static_cast<v4u*>(a.output);
  const size_t nSlices = (a.vectors + a.sliceVectors - 1) / a.sliceVectors;
  const int helperCtas = a.helpers * a.lanes;
  const int block = static_cast<int>(blockIdx.x);

  if (block < a.directLanes) {
    // The peer's input is ready once its kernel has started.
    if (threadIdx.x == 0) {
      waitFlag(&mine->start.value, epoch);
    }
    __syncthreads();
    const v4u* theirs = static_cast<const v4u*>(a.peer);
    for (size_t k = block;; k += a.directLanes) {
      const size_t s = directSlice(a, k);
      if (s >= nSlices) {
        break;
      }
      const size_t off = s * a.sliceVectors;
      reduceSlice(
          input + off,
          theirs + off,
          output + off,
          min(a.sliceVectors, a.vectors - off),
          a.rank);
    }
  } else {
    const bool push = block < a.directLanes + helperCtas;
    const int id =
        push ? block - a.directLanes : block - a.directLanes - helperCtas;
    const int h = id / a.lanes;
    const int lane = id % a.lanes;
    uint32_t* counter =
        push ? &mine->pushSeq[h][lane] : &mine->reduceSeq[h][lane];
    const uint32_t base = *counter;
    uint32_t n = 0;
    for (size_t k = lane;; k += a.lanes, ++n) {
      const size_t s = helperSlice(a, h, k);
      if (s >= nSlices) {
        break;
      }
      const uint32_t seq = base + n + 1u;
      const int slot = static_cast<int>(seq % static_cast<uint32_t>(a.slots));
      const size_t off = s * a.sliceVectors;
      const size_t vectors = min(a.sliceVectors, a.vectors - off);
      const size_t slotOff =
          (static_cast<size_t>(lane) * a.slots + slot) * a.slotVectors;
      if (push) {
        // The slot's previous occupant must have been consumed by the peer.
        // Unconditional and wrap-safe: for the first `slots` sequences the
        // zeroed credit already satisfies it, and slots is a power of two, so
        // seq % slots stays continuous across the 32-bit wrap.
        if (threadIdx.x == 0) {
          waitFlag(&mine->freed[h][lane][slot], seq - a.slots);
        }
        __syncthreads();
        pushSlice(input + off, static_cast<v4u*>(a.push[h]) + slotOff, vectors);
        __builtin_amdgcn_s_waitcnt(kGfx9VmcntZero);
        __syncthreads();
        if (threadIdx.x == 0) {
          storeFlag(&peer->full[h][lane][slot], seq);
        }
      } else {
        if (threadIdx.x == 0) {
          waitFlag(&mine->full[h][lane][slot], seq);
        }
        __syncthreads();
        reduceSlice(
            input + off,
            static_cast<const v4u*>(a.pull[h]) + slotOff,
            output + off,
            vectors,
            a.rank);
        __builtin_amdgcn_s_waitcnt(kGfx9VmcntZero);
        __syncthreads();
        if (threadIdx.x == 0) {
          storeFlag(&peer->freed[h][lane][slot], seq);
        }
      }
    }
    if (threadIdx.x == 0) {
      *counter = base + n;
    }
  }

  // Tell the peer this rank is done reading its input, and hold the stream
  // until the peer is done reading this rank's, so the producer may overwrite
  // the registered input after the kernel.
  if (!lastCta(&mine->ticket.value)) {
    return;
  }
  storeFlag(&peer->done.value, epoch);
  waitFlag(&mine->done.value, epoch);
  mine->calls.value = epoch;
}

} // namespace

hipError_t launchRegisteredRelayAllReduce(
    const RegisteredRelayArgs& args,
    hipStream_t stream) {
  const int blocks = args.directLanes + 2 * args.helpers * args.lanes;
  hipLaunchKernelGGL(
      registeredRelayAllReduceKernel,
      dim3(blocks),
      dim3(kThreads),
      0,
      stream,
      args);
  return hipPeekAtLastError();
}

} // namespace rcclx::relay
