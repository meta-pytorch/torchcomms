/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "registered_allreduce_kernels.h"

namespace rcclx::relay {
namespace {

constexpr int kElementsPerVector = 16 / sizeof(__nv_bfloat16);
constexpr int kPeerGatherLanes =
    kRegisteredAllReduceThreads / kRegisteredAllReduceRanks;
static_assert(kElementsPerVector == 8);
static_assert(kPeerGatherLanes == 128);

// BF16 vectors are loaded and stored; sums accumulate in FP32 and round to
// BF16 (nearest even) once, as AITER's custom all-reduce does.
using Vec = __bf16 __attribute__((ext_vector_type(kElementsPerVector)));
using Acc = float __attribute__((ext_vector_type(kElementsPerVector)));

// AITER reduces a four-rank payload of at least this size in two stages: the
// vectors are split into four contiguous quarters and quarter q is summed
// starting at rank q. Smaller payloads, and all two-rank ones, start at rank 0.
// Matching that order makes the sums bit-identical to AITER's.
constexpr size_t kAiterTwoStageMinBytes = 160 * 1024;

enum class Phase : int {
  Start = 0,
  Midpoint = 1,
  Relay = 2,
};

__device__ __forceinline__ bool epochReached(uint32_t got, uint32_t want) {
  return (got - want) <= (uint32_t(-1) >> 1);
}

__device__ __forceinline__ Acc widen(const Vec& value) {
  Acc result;
#pragma unroll
  for (int element = 0; element < kElementsPerVector; ++element) {
    result[element] = static_cast<float>(value[element]);
  }
  return result;
}

__device__ __forceinline__ Vec narrow(const Acc& value) {
  Vec result;
#pragma unroll
  for (int element = 0; element < kElementsPerVector; ++element) {
    result[element] = static_cast<__bf16>(value[element]);
  }
  return result;
}

__device__ __forceinline__ Vec loadPeer(
    const RegisteredAllReduceInputTable& inputs,
    int rank,
    int distance,
    size_t shardOffset,
    size_t vectorIndex) {
  const int source = (rank + distance) % kRegisteredAllReduceRanks;
  return reinterpret_cast<const Vec*>(
      inputs.input[source] + shardOffset)[vectorIndex];
}

// Reduces one vector of the quarter this rank owns (quarter `rank` of a 0.5 or
// 1 MiB payload, or of the 64 x 8192 epilogue rows), in AITER's two-stage
// order for that quarter: the owner's own contribution, then ranks rank + 1,
// rank + 2 and rank + 3. Named values keep the operands in registers.
__device__ __forceinline__ Vec reduceVector(
    const RegisteredAllReduceInputTable& inputs,
    int rank,
    const Vec& own,
    size_t shardOffset,
    size_t vectorIndex) {
  const Vec next = loadPeer(inputs, rank, 1, shardOffset, vectorIndex);
  const Vec opposite = loadPeer(inputs, rank, 2, shardOffset, vectorIndex);
  const Vec previous = loadPeer(inputs, rank, 3, shardOffset, vectorIndex);
  Acc sum = widen(own);
  sum += widen(next);
  sum += widen(opposite);
  sum += widen(previous);
  return narrow(sum);
}

template <bool ScopedAtomics, int Order>
__device__ __forceinline__ void peerStore(uint32_t* address, uint32_t value) {
  if constexpr (ScopedAtomics) {
    __scoped_atomic_store_n(address, value, Order, __MEMORY_SCOPE_SYSTEM);
  } else {
    __atomic_store_n(address, value, Order);
  }
}

template <bool ScopedAtomics, int Order>
__device__ __forceinline__ uint32_t peerLoad(const uint32_t* address) {
  if constexpr (ScopedAtomics) {
    return __scoped_atomic_load_n(address, Order, __MEMORY_SCOPE_DEVICE);
  } else {
    return __atomic_load_n(address, Order);
  }
}

__device__ __forceinline__ RegisteredAllReduceBlockState& blockState(
    const RegisteredAllReduceStateTable& states,
    int rank) {
  return states.state[rank]->block[blockIdx.x];
}

template <bool ScopedAtomics, bool SystemFence>
__device__ __forceinline__ void phaseBarrier(
    const RegisteredAllReduceStateTable& states,
    int rank,
    Phase phase,
    const uint32_t* targetEpochs,
    const uint32_t* sourceEpochs) {
  if constexpr (SystemFence) {
    if (threadIdx.x == 0) {
      __threadfence_system();
    }
  }
  __syncthreads();

  if (threadIdx.x < kRegisteredAllReduceRanks && threadIdx.x != rank) {
    const auto peer = threadIdx.x;
    uint32_t* remoteSignal =
        &blockState(states, peer).phase[static_cast<int>(phase)][rank];
    uint32_t* localSignal =
        &blockState(states, rank).phase[static_cast<int>(phase)][peer];
    peerStore<ScopedAtomics, __ATOMIC_RELEASE>(
        remoteSignal, targetEpochs[peer]);
    while (!epochReached(
        peerLoad<ScopedAtomics, __ATOMIC_ACQUIRE>(localSignal),
        sourceEpochs[peer])) {
    }
  }
  __syncthreads();
}

template <bool ScopedAtomics>
__device__ __forceinline__ void startHandshake(
    int rank,
    const RegisteredAllReduceStateTable& states,
    uint32_t* targetEpochs,
    uint32_t* sourceEpochs) {
  if (threadIdx.x < kRegisteredAllReduceRanks && threadIdx.x != rank) {
    const auto peer = threadIdx.x;
    RegisteredAllReduceBlockState& local = blockState(states, rank);
    uint32_t* remoteStart =
        &blockState(states, peer).phase[static_cast<int>(Phase::Start)][rank];
    uint32_t* localStart = &local.phase[static_cast<int>(Phase::Start)][peer];

    const uint32_t epoch = local.seq[peer] + 1u;
    local.seq[peer] = epoch;
    targetEpochs[peer] = epoch;
    peerStore<ScopedAtomics, __ATOMIC_RELAXED>(remoteStart, epoch);

    const uint32_t want = local.seen[peer] + 1u;
    uint32_t got = 0;
    do {
      got = peerLoad<ScopedAtomics, __ATOMIC_RELAXED>(localStart);
    } while (!epochReached(got, want));
    local.seen[peer] = got;
    sourceEpochs[peer] = got;
  }
  __syncthreads();
}

// Waits until this wave's outstanding vector-memory stores have completed.
// The midpoint flag is published by wave 0, so every wave that wrote scratch
// drains its own stores first.
__device__ __forceinline__ void drainWaveStores() {
  constexpr unsigned kGfx9VmcntZero = 0x0F70;
  __builtin_amdgcn_s_waitcnt(kGfx9VmcntZero);
}

// With uncached state, drained scratch stores are already visible in memory and
// peer scratch loads cannot hit stale cache lines, so the midpoint needs no
// cache writeback or invalidate (the same condition RCCL's gfx9 cheap fence
// relies on).
#if defined(HIP_UNCACHED_MEMORY)
constexpr bool kUncachedState = true;
#else
constexpr bool kUncachedState = false;
#endif

template <bool ScopedAtomics>
__device__ __forceinline__ void midpointBarrier(
    const RegisteredAllReduceStateTable& states,
    int rank,
    const uint32_t* targetEpochs,
    const uint32_t* sourceEpochs) {
  __syncthreads();
  if constexpr (!ScopedAtomics) {
    phaseBarrier<false, true>(
        states, rank, Phase::Midpoint, targetEpochs, sourceEpochs);
    return;
  }
  if (threadIdx.x < kRegisteredAllReduceRanks && threadIdx.x != rank) {
    const auto peer = threadIdx.x;
    uint32_t* remoteSignal =
        &blockState(states, peer)
             .phase[static_cast<int>(Phase::Midpoint)][rank];
    const uint32_t* localSignal =
        &blockState(states, rank)
             .phase[static_cast<int>(Phase::Midpoint)][peer];
    if constexpr (kUncachedState) {
      peerStore<true, __ATOMIC_RELAXED>(remoteSignal, targetEpochs[peer]);
    } else {
      peerStore<true, __ATOMIC_RELEASE>(remoteSignal, targetEpochs[peer]);
    }
    while (!epochReached(
        peerLoad<true, __ATOMIC_RELAXED>(localSignal), sourceEpochs[peer])) {
    }
    if constexpr (!kUncachedState) {
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);
    }
  }
  __syncthreads();
}

// One 128-vector chunk of every shard per CTA. Group 0 reduces the chunk of the
// locally owned shard; after the midpoint it publishes that chunk from
// registers while groups 1-3 each gather the same chunk of one peer's shard.
template <size_t PayloadBytes, int Blocks, bool ScopedAtomics>
__device__ __forceinline__ void runTwoStage(
    __nv_bfloat16* __restrict__ output,
    int rank,
    const RegisteredAllReduceInputTable& inputs,
    const RegisteredAllReduceStateTable& states,
    uint32_t* targetEpochs,
    uint32_t* sourceEpochs) {
  constexpr size_t kShardCount =
      PayloadBytes / (sizeof(__nv_bfloat16) * kRegisteredAllReduceRanks);
  constexpr size_t kShardVectorCount = kShardCount / kElementsPerVector;
  static_assert(PayloadBytes % kRegisteredAllReduceRanks == 0);
  static_assert(kShardVectorCount == Blocks * kPeerGatherLanes);
  static_assert(Blocks <= kRegisteredAllReduceMaxBlocks);

  const auto group = threadIdx.x / kPeerGatherLanes;
  const auto lane = threadIdx.x % kPeerGatherLanes;
  const size_t vector = blockIdx.x * kPeerGatherLanes + lane;
  const size_t shardOffset = static_cast<size_t>(rank) * kShardCount;
  // The local input is ready by stream order, so it is loaded before the start
  // handshake and its latency overlaps the wait for peers.
  Vec own;
  if (group == 0) {
    own =
        reinterpret_cast<const Vec*>(inputs.input[rank] + shardOffset)[vector];
  }

  startHandshake<ScopedAtomics>(rank, states, targetEpochs, sourceEpochs);

  Vec reduced;
  if (group == 0) {
    reduced = reduceVector(inputs, rank, own, shardOffset, vector);
    reinterpret_cast<Vec*>(
        registeredAllReduceScratch(states.state[rank]))[vector] = reduced;
    drainWaveStores();
  }

  midpointBarrier<ScopedAtomics>(states, rank, targetEpochs, sourceEpochs);

  const int shard = (group + rank) % kRegisteredAllReduceRanks;
  Vec* destination =
      reinterpret_cast<Vec*>(output + static_cast<size_t>(shard) * kShardCount);
  if (group == 0) {
    destination[vector] = reduced;
  } else {
    destination[vector] = reinterpret_cast<const Vec*>(
        registeredAllReduceScratch(states.state[shard]))[vector];
  }
}

#if defined(ROCM_VERSION) && ROCM_VERSION >= 70200
constexpr bool kHalfMiBScopedAtomics = true;
#else
constexpr bool kHalfMiBScopedAtomics = false;
#endif

__global__ void registeredAllReduceHalfMiBKernel(
    __nv_bfloat16* __restrict__ output,
    int rank,
    RegisteredAllReduceInputTable inputs,
    RegisteredAllReduceStateTable states) {
  __shared__ uint32_t targetEpochs[kRegisteredAllReduceRanks];
  __shared__ uint32_t sourceEpochs[kRegisteredAllReduceRanks];
  runTwoStage<
      kRegisteredAllReduceHalfMiBBytes,
      kRegisteredAllReduceHalfMiBBlocks,
      kHalfMiBScopedAtomics>(
      output, rank, inputs, states, targetEpochs, sourceEpochs);
}

__global__ void registeredAllReduceOneMiBKernel(
    __nv_bfloat16* __restrict__ output,
    int rank,
    RegisteredAllReduceInputTable inputs,
    RegisteredAllReduceStateTable states) {
  __shared__ uint32_t targetEpochs[kRegisteredAllReduceRanks];
  __shared__ uint32_t sourceEpochs[kRegisteredAllReduceRanks];
  runTwoStage<
      kRegisteredAllReduceOneMiBBytes,
      kRegisteredAllReduceOneMiBBlocks,
      true>(output, rank, inputs, states, targetEpochs, sourceEpochs);
}

// Two-stage all-reduce of any other four-rank payload from
// kAiterTwoStageMinBytes, in AITER's split and order: quarter q holds vectors
// [q * part, (q + 1) * part) with part = vectors / 4 (the last quarter takes
// the remainder), and rank q sums quarter q starting with its own contribution.
// Like AITER's kernel, 512 threads form four 128-lane groups: in stage 1 group
// g loads source (rank + g) % 4 of a 128-vector chunk of the own quarter into
// shared memory, so each wave reads one peer, and group 0 sums the four and
// stores the result to its scratch and output; in stage 2 group g copies the
// same chunk of owner (rank + g) % 4's quarter from that owner's scratch.
// Chunk c runs on CTA c % gridDim.x on every rank, so a CTA reads only scratch
// the same-index peer CTA wrote, and the per-CTA midpoint barrier orders it.
constexpr int kTwoStageLanes = 128;
constexpr int kTwoStageThreads = kTwoStageLanes * kRegisteredAllReduceRanks;

__global__ void __launch_bounds__(kTwoStageThreads)
    registeredAllReduceTwoStageKernel(
        __nv_bfloat16* __restrict__ output,
        int rank,
        RegisteredAllReduceInputTable inputs,
        RegisteredAllReduceStateTable states,
        size_t vectors) {
  __shared__ uint32_t targetEpochs[kRegisteredAllReduceRanks];
  __shared__ uint32_t sourceEpochs[kRegisteredAllReduceRanks];
  __shared__ Vec staged[kRegisteredAllReduceRanks][kTwoStageLanes];
  constexpr int kRanks = kRegisteredAllReduceRanks;
  const size_t part = vectors / kRanks;
  const auto quarterCount = [part, vectors](int q) {
    return q == kRanks - 1 ? vectors - (kRanks - 1) * part : part;
  };
  const auto group = threadIdx.x / kTwoStageLanes;
  const auto lane = threadIdx.x % kTwoStageLanes;
  Vec* destination = reinterpret_cast<Vec*>(output);

  const size_t ownBegin = rank * part;
  const size_t ownCount = quarterCount(rank);
  const size_t shardOffset = ownBegin * kElementsPerVector;
  // The local input is ready by stream order: the first chunk's own vector is
  // loaded before the handshake so its latency overlaps the wait for peers.
  const size_t firstVector =
      static_cast<size_t>(blockIdx.x) * kTwoStageLanes + lane;
  Vec early;
  if (group == 0 && firstVector < ownCount) {
    early = loadPeer(inputs, rank, 0, shardOffset, firstVector);
  }

  startHandshake<true>(rank, states, targetEpochs, sourceEpochs);

  Vec* scratch =
      reinterpret_cast<Vec*>(registeredAllReduceScratch(states.state[rank]));
  for (size_t chunk = blockIdx.x; chunk * kTwoStageLanes < ownCount;
       chunk += gridDim.x) {
    const size_t i = chunk * kTwoStageLanes + lane;
    if (i < ownCount) {
      staged[group][lane] = group == 0 && chunk == blockIdx.x
          ? early
          : loadPeer(inputs, rank, group, shardOffset, i);
    }
    __syncthreads();
    if (group == 0 && i < ownCount) {
      Acc sum = widen(staged[0][lane]);
      sum += widen(staged[1][lane]);
      sum += widen(staged[2][lane]);
      sum += widen(staged[3][lane]);
      const Vec reduced = narrow(sum);
      scratch[i] = reduced;
      destination[ownBegin + i] = reduced;
    }
    __syncthreads();
  }
  drainWaveStores();

  midpointBarrier<true>(states, rank, targetEpochs, sourceEpochs);

  if (group != 0) {
    const int owner = (rank + group) % kRanks;
    const Vec* ownerScratch = reinterpret_cast<const Vec*>(
        registeredAllReduceScratch(states.state[owner]));
    const size_t begin = owner * part;
    const size_t count = quarterCount(owner);
    for (size_t chunk = blockIdx.x; chunk * kTwoStageLanes < count;
         chunk += gridDim.x) {
      const size_t i = chunk * kTwoStageLanes + lane;
      if (i < count) {
        destination[begin + i] = ownerScratch[i];
      }
    }
  }
}

int twoStageBlocks(size_t vectors) {
  const size_t largestQuarter = vectors -
      (kRegisteredAllReduceRanks - 1) * (vectors / kRegisteredAllReduceRanks);
  const size_t blocks = (largestQuarter + kTwoStageLanes - 1) / kTwoStageLanes;
  return static_cast<int>(
      blocks < kRegisteredAllReduceMaxBlocks ? blocks
                                             : kRegisteredAllReduceMaxBlocks);
}

// Per-CTA all-rank exchange of the relay route. Every wave drains its stores
// (staged chunks in helper HBM) before the signal; with uncached state and
// staging those stores are then visible to any GPU. Otherwise the stores and
// the signal are fenced at system scope, since staging lives on a third GPU.
__device__ __forceinline__ void relayBarrier(
    const RegisteredAllReduceStateTable& states,
    int rank,
    Phase phase,
    const uint32_t* targetEpochs,
    const uint32_t* sourceEpochs) {
  drainWaveStores();
  if constexpr (!kUncachedState) {
    __threadfence_system();
  }
  __syncthreads();
  if (threadIdx.x < kRegisteredAllReduceRanks && threadIdx.x != rank) {
    const auto peer = threadIdx.x;
    uint32_t* remoteSignal =
        &blockState(states, peer).phase[static_cast<int>(phase)][rank];
    const uint32_t* localSignal =
        &blockState(states, rank).phase[static_cast<int>(phase)][peer];
    if constexpr (kUncachedState) {
      peerStore<true, __ATOMIC_RELAXED>(remoteSignal, targetEpochs[peer]);
    } else {
      peerStore<true, __ATOMIC_RELEASE>(remoteSignal, targetEpochs[peer]);
    }
    while (!epochReached(
        peerLoad<true, __ATOMIC_RELAXED>(localSignal), sourceEpochs[peer])) {
    }
    if constexpr (!kUncachedState) {
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    }
  }
  __syncthreads();
}

// The relay route of a four-rank two-stage all-reduce (see
// RegisteredAllReduceRelayRoute). The first gridDim.x - helperBlocks CTAs run
// the two-stage kernel over the direct chunks. The other CTAs serve helper
// h = id / lanes: each rank stages its contribution to every other owner's
// relayed chunks in its buffer on h; after an exchange each owner sums the
// three staged contributions and its own, in AITER's order, into its output
// and its buffer's reduced slots; after a second exchange every rank copies the
// other owners' reduced chunks. Chunk-to-CTA assignment is the same on every
// rank, so the per-CTA handshakes order every staging and scratch access, and
// the next call's start handshake fences their reuse.
constexpr int kRelayLanes = 128;
constexpr int kRelayThreads = kRelayLanes * kRegisteredAllReduceRanks;
static_assert(kRelayLanes == kRegisteredAllReduceRelayChunkVectors);

__global__ void __launch_bounds__(kRelayThreads) registeredAllReduceRelayKernel(
    __nv_bfloat16* __restrict__ output,
    int rank,
    RegisteredAllReduceInputTable inputs,
    RegisteredAllReduceStateTable states,
    size_t vectors,
    RegisteredAllReduceRelayRoute route) {
  __shared__ uint32_t targetEpochs[kRegisteredAllReduceRanks];
  __shared__ uint32_t sourceEpochs[kRegisteredAllReduceRanks];
  __shared__ uint32_t relayTarget[kRegisteredAllReduceRanks];
  __shared__ uint32_t relaySource[kRegisteredAllReduceRanks];
  __shared__ Vec staged[kRegisteredAllReduceRanks][kRelayLanes];
  constexpr int kRanks = kRegisteredAllReduceRanks;
  const size_t part = vectors / kRanks;
  const auto quarterCount = [part, vectors](int q) {
    return q == kRanks - 1 ? vectors - (kRanks - 1) * part : part;
  };
  const size_t cycle = static_cast<size_t>(route.directWeight) + route.helpers;
  const int directBlocks = static_cast<int>(gridDim.x) - route.helperBlocks;
  const auto group = threadIdx.x / kRelayLanes;
  const auto lane = threadIdx.x % kRelayLanes;
  const auto input = [&inputs](int r) {
    return reinterpret_cast<const Vec*>(inputs.input[r]);
  };
  Vec* destination = reinterpret_cast<Vec*>(output);
  const size_t ownBegin = rank * part;
  const size_t ownCount = quarterCount(rank);

  startHandshake<true>(rank, states, targetEpochs, sourceEpochs);

  if (static_cast<int>(blockIdx.x) < directBlocks) {
    const auto directChunk = [&](size_t k) {
      return (k / route.directWeight) * cycle + k % route.directWeight;
    };
    Vec* scratch =
        reinterpret_cast<Vec*>(registeredAllReduceScratch(states.state[rank]));
    const Vec* source = input((rank + group) % kRanks) + ownBegin;
    for (size_t k = blockIdx.x;; k += directBlocks) {
      const size_t i = directChunk(k) * kRelayLanes + lane;
      if (i - lane >= ownCount) {
        break;
      }
      if (i < ownCount) {
        staged[group][lane] = source[i];
      }
      __syncthreads();
      if (group == 0 && i < ownCount) {
        Acc sum = widen(staged[0][lane]);
        sum += widen(staged[1][lane]);
        sum += widen(staged[2][lane]);
        sum += widen(staged[3][lane]);
        const Vec reduced = narrow(sum);
        scratch[i] = reduced;
        destination[ownBegin + i] = reduced;
      }
      __syncthreads();
    }
    drainWaveStores();
    midpointBarrier<true>(states, rank, targetEpochs, sourceEpochs);
    if (group != 0) {
      const int owner = (rank + group) % kRanks;
      const Vec* ownerScratch = reinterpret_cast<const Vec*>(
          registeredAllReduceScratch(states.state[owner]));
      const size_t begin = owner * part;
      const size_t count = quarterCount(owner);
      for (size_t k = blockIdx.x;; k += directBlocks) {
        const size_t i = directChunk(k) * kRelayLanes + lane;
        if (i - lane >= count) {
          break;
        }
        if (i < count) {
          destination[begin + i] = ownerScratch[i];
        }
      }
    }
    return;
  }

  const int id = static_cast<int>(blockIdx.x) - directBlocks;
  const int lanesPerHelper = route.helperBlocks / route.helpers;
  const int h = id / lanesPerHelper;
  const int first = id % lanesPerHelper;
  const auto helperChunk = [&](size_t k) {
    return k * cycle + route.directWeight + h;
  };
  const auto slot = [&](int r, size_t s, size_t k) {
    return static_cast<Vec*>(route.staging[r][h]) +
        (s * route.slotChunks + k) * kRelayLanes + lane;
  };

  // Stage this rank's contribution to the other owners' relayed chunks.
  if (group != 0) {
    const int owner = (rank + group) % kRanks;
    const Vec* mine = input(rank) + owner * part;
    const size_t count = quarterCount(owner);
    for (size_t k = first;; k += lanesPerHelper) {
      const size_t i = helperChunk(k) * kRelayLanes + lane;
      if (i - lane >= count) {
        break;
      }
      if (i < count) {
        *slot(rank, owner, k) = mine[i];
      }
    }
  }
  relayBarrier(states, rank, Phase::Midpoint, targetEpochs, sourceEpochs);

  // Sum this rank's relayed chunks: group g holds source rank + g.
  const int from = (rank + group) % kRanks;
  for (size_t k = first;; k += lanesPerHelper) {
    const size_t i = helperChunk(k) * kRelayLanes + lane;
    if (i - lane >= ownCount) {
      break;
    }
    if (i < ownCount) {
      staged[group][lane] =
          group == 0 ? input(rank)[ownBegin + i] : *slot(from, rank, k);
    }
    __syncthreads();
    if (group == 0 && i < ownCount) {
      Acc sum = widen(staged[0][lane]);
      sum += widen(staged[1][lane]);
      sum += widen(staged[2][lane]);
      sum += widen(staged[3][lane]);
      const Vec reduced = narrow(sum);
      destination[ownBegin + i] = reduced;
      *slot(rank, kRanks, k) = reduced;
    }
    __syncthreads();
  }
  // The second exchange counts helper calls only (see relaySeq).
  RegisteredAllReduceBlockState& local = blockState(states, rank);
  if (threadIdx.x < kRanks && threadIdx.x != rank) {
    relayTarget[threadIdx.x] = local.relaySeq[threadIdx.x] + 1u;
    relaySource[threadIdx.x] = local.relaySeen[threadIdx.x] + 1u;
  }
  relayBarrier(states, rank, Phase::Relay, relayTarget, relaySource);
  if (threadIdx.x < kRanks && threadIdx.x != rank) {
    local.relaySeq[threadIdx.x] = relayTarget[threadIdx.x];
    local.relaySeen[threadIdx.x] = relaySource[threadIdx.x];
  }

  if (group != 0) {
    const int owner = (rank + group) % kRanks;
    const size_t begin = owner * part;
    const size_t count = quarterCount(owner);
    for (size_t k = first;; k += lanesPerHelper) {
      const size_t i = helperChunk(k) * kRelayLanes + lane;
      if (i - lane >= count) {
        break;
      }
      if (i < count) {
        destination[begin + i] = *slot(owner, kRanks, k);
      }
    }
  }
}

int relayDirectBlocks(size_t vectors, const RegisteredAllReduceRelayRoute& r) {
  const size_t largestQuarter = vectors -
      (kRegisteredAllReduceRanks - 1) * (vectors / kRegisteredAllReduceRanks);
  const size_t chunks = (largestQuarter + kRelayLanes - 1) / kRelayLanes;
  const size_t cycle = static_cast<size_t>(r.directWeight) + r.helpers;
  const size_t tail = chunks % cycle;
  const size_t direct = (chunks / cycle) * r.directWeight +
      (tail < static_cast<size_t>(r.directWeight) ? tail : r.directWeight);
  const size_t room =
      static_cast<size_t>(kRegisteredAllReduceMaxBlocks - r.helperBlocks);
  const size_t blocks = direct < room ? direct : room;
  return static_cast<int>(blocks > 0 ? blocks : 1);
}

// Gated-residual-norm epilogue with a fixed arithmetic order, matching the
// reference Triton post-norm/gated-residual/pre-norm kernel bit for bit: thread
// t owns elements [8t, 8t + 8) of the row, row sums use the per-thread order,
// DPP wave tree, and 16-wave butterfly of that kernel, and multiply-adds are
// fused exactly where it fuses them.
constexpr int kNormWaves = kRegisteredAllReduceNormThreads / 64;
static_assert(
    kRegisteredAllReduceNormHidden ==
    kRegisteredAllReduceNormThreads * kElementsPerVector);
static_assert(kNormWaves == 16);

template <int DppControl, int RowMask = 0xf>
__device__ __forceinline__ float dppUpdate(float old, float source) {
  return __builtin_bit_cast(
      float,
      __builtin_amdgcn_update_dpp(
          __builtin_bit_cast(int, old),
          __builtin_bit_cast(int, source),
          DppControl,
          RowMask,
          0xf,
          true));
}

// Every thread redoes the order-identical cross-wave butterfly, so one barrier
// suffices; each reduction uses its own LDS slots.
__device__ __forceinline__ float normRowSum(float value, float* waveSums) {
  constexpr int kRowShr = 0x110;
  constexpr int kRowBcast15 = 0x142;
  constexpr int kRowBcast31 = 0x143;
  value = value + dppUpdate<kRowShr + 8>(0.0f, value);
  value = value + dppUpdate<kRowShr + 4>(0.0f, value);
  value = value + dppUpdate<kRowShr + 2>(0.0f, value);
  value = value + dppUpdate<kRowShr + 1>(0.0f, value);
  value = dppUpdate<kRowBcast15, 0xa>(value, value) + value;
  value = value + dppUpdate<kRowBcast31>(0.0f, value);
  const float waveTotal = __builtin_bit_cast(
      float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, value), 63));
  if (threadIdx.x % 64 == 0) {
    waveSums[threadIdx.x / 64] = waveTotal;
  }
  __syncthreads();
  float sums[kNormWaves];
#pragma unroll
  for (int wave = 0; wave < kNormWaves; ++wave) {
    sums[wave] = waveSums[wave];
  }
#pragma unroll
  for (int step = kNormWaves / 2; step >= 1; step /= 2) {
#pragma unroll
    for (int wave = 0; wave < step; ++wave) {
      sums[wave] = sums[wave] + sums[wave + step];
    }
  }
  return sums[0];
}

__device__ __forceinline__ float normSquareSum(
    const float (&values)[kElementsPerVector]) {
#pragma clang fp contract(off)
  float sum = values[1] * values[1];
  sum = __builtin_fmaf(values[0], values[0], sum);
#pragma unroll
  for (int element = 2; element < kElementsPerVector; ++element) {
    sum = __builtin_fmaf(values[element], values[element], sum);
  }
  return sum;
}

__device__ __forceinline__ float normScale(float sum, float epsilon) {
  constexpr float kInverseHidden = 1.0f / kRegisteredAllReduceNormHidden;
  return __builtin_amdgcn_rsqf(__builtin_fmaf(sum, kInverseHidden, epsilon));
}

__device__ __forceinline__ float roundToBf16(float value) {
  return static_cast<float>(static_cast<__bf16>(value));
}

// Operands that do not depend on the reduction; loaded before the handshake.
struct NormOperands {
  Vec postNormWeight;
  Vec preNormWeight;
  float residual[kElementsPerVector];
  float gateAlpha[kElementsPerVector];
  float gateBeta[kElementsPerVector];
};

__device__ __forceinline__ void loadFloats(
    float (&destination)[kElementsPerVector],
    const float* source) {
  *reinterpret_cast<float4*>(&destination[0]) =
      *reinterpret_cast<const float4*>(source);
  *reinterpret_cast<float4*>(&destination[4]) =
      *reinterpret_cast<const float4*>(source + 4);
}

__device__ __forceinline__ void storeFloats(
    float* destination,
    const float (&source)[kElementsPerVector]) {
  *reinterpret_cast<float4*>(destination) =
      *reinterpret_cast<const float4*>(&source[0]);
  *reinterpret_cast<float4*>(destination + 4) =
      *reinterpret_cast<const float4*>(&source[4]);
}

__device__ __forceinline__ NormOperands loadNormOperands(
    size_t elementOffset,
    const RegisteredAllReduceGatedResidualNormArgs& norm) {
  const auto channel = threadIdx.x * kElementsPerVector;
  NormOperands operands;
  operands.postNormWeight =
      *reinterpret_cast<const Vec*>(norm.postNormWeight + channel);
  operands.preNormWeight =
      *reinterpret_cast<const Vec*>(norm.preNormWeight + channel);
  loadFloats(operands.residual, norm.residualIn + elementOffset);
  loadFloats(operands.gateAlpha, norm.gateAlpha + channel);
  loadFloats(operands.gateBeta, norm.gateBeta + channel);
  return operands;
}

__device__ __forceinline__ void gatedResidualNorm(
    const Vec& reduced,
    const NormOperands& operands,
    size_t elementOffset,
    __nv_bfloat16* __restrict__ output,
    const RegisteredAllReduceGatedResidualNormArgs& norm) {
#pragma clang fp contract(off)
  __shared__ float postSums[kNormWaves];
  __shared__ float preSums[kNormWaves];
  float branch[kElementsPerVector];
#pragma unroll
  for (int element = 0; element < kElementsPerVector; ++element) {
    branch[element] = static_cast<float>(reduced[element]);
  }
  const float postScale = normScale(
      normRowSum(normSquareSum(branch), postSums), norm.postNormEpsilon);
  float residual[kElementsPerVector];
  float preInput[kElementsPerVector];
#pragma unroll
  for (int element = 0; element < kElementsPerVector; ++element) {
    const float normedBranch = roundToBf16(
        branch[element] * postScale *
        static_cast<float>(operands.postNormWeight[element]));
    residual[element] = __builtin_fmaf(
        operands.gateBeta[element],
        normedBranch,
        operands.gateAlpha[element] * operands.residual[element]);
    preInput[element] = roundToBf16(residual[element]);
  }
  storeFloats(norm.residualOut + elementOffset, residual);
  const float preScale = normScale(
      normRowSum(normSquareSum(preInput), preSums), norm.preNormEpsilon);
  Vec normed;
  float normedFloat[kElementsPerVector];
#pragma unroll
  for (int element = 0; element < kElementsPerVector; ++element) {
    normedFloat[element] = roundToBf16(
        preInput[element] * preScale *
        static_cast<float>(operands.preNormWeight[element]));
    normed[element] = static_cast<__bf16>(normedFloat[element]);
  }
  *reinterpret_cast<Vec*>(output + elementOffset) = normed;
  if (norm.routerOut != nullptr) {
    storeFloats(norm.routerOut + elementOffset, normedFloat);
  }
}

// One CTA per row. The row owner (row / 16, the AITER quarter) reduces it into
// its scratch and publishes one midpoint epoch per peer; every rank then runs
// the epilogue on the full row. The plain reduced row is never written.
__global__ void __launch_bounds__(kRegisteredAllReduceNormThreads)
    registeredAllReduceGatedResidualNormKernel(
        __nv_bfloat16* __restrict__ output,
        int rank,
        RegisteredAllReduceInputTable inputs,
        RegisteredAllReduceStateTable states,
        RegisteredAllReduceGatedResidualNormArgs norm) {
  constexpr int kRowVectors =
      kRegisteredAllReduceNormHidden / kElementsPerVector;
  __shared__ uint32_t callsShared;
  const auto row = blockIdx.x;
  const int owner = row / kRegisteredAllReduceRowsPerRank;
  const auto thread = threadIdx.x;
  const size_t rowOffset =
      static_cast<size_t>(row) * kRegisteredAllReduceNormHidden;
  const size_t elementOffset = rowOffset + thread * kElementsPerVector;
  RegisteredAllReduceRowState& local = states.state[rank]->row[row];
  const NormOperands operands = loadNormOperands(elementOffset, norm);
  if (thread == 0) {
    const uint32_t calls = local.calls + 1u;
    local.calls = calls;
    callsShared = calls;
    if (rank != owner) {
      peerStore<true, __ATOMIC_RELAXED>(
          &states.state[owner]->row[row].start[rank], calls);
    }
  }
  __syncthreads();
  const uint32_t calls = callsShared;
  Vec* ownerScratch =
      reinterpret_cast<Vec*>(registeredAllReduceScratch(states.state[owner])) +
      static_cast<size_t>(row - owner * kRegisteredAllReduceRowsPerRank) *
          kRowVectors;

  Vec reduced;
  if (rank == owner) {
    const Vec own =
        reinterpret_cast<const Vec*>(inputs.input[rank] + rowOffset)[thread];
    if (thread < kRegisteredAllReduceRanks && thread != rank) {
      while (!epochReached(
          peerLoad<true, __ATOMIC_RELAXED>(&local.start[thread]), calls)) {
      }
    }
    __syncthreads();
    reduced = reduceVector(inputs, rank, own, rowOffset, thread);
    ownerScratch[thread] = reduced;
    drainWaveStores();
    __syncthreads();
    if (thread < kRegisteredAllReduceRanks && thread != rank) {
      uint32_t* remote = &states.state[thread]->row[row].midpoint;
      if constexpr (kUncachedState) {
        peerStore<true, __ATOMIC_RELAXED>(remote, calls);
      } else {
        peerStore<true, __ATOMIC_RELEASE>(remote, calls);
      }
    }
  } else {
    if (thread == 0) {
      while (!epochReached(
          peerLoad<true, __ATOMIC_RELAXED>(&local.midpoint), calls)) {
      }
      if constexpr (!kUncachedState) {
        __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);
      }
    }
    __syncthreads();
    reduced = ownerScratch[thread];
  }
  gatedResidualNorm(reduced, operands, elementOffset, output, norm);
}

// Generic one-shot path: every rank reduces the whole payload, each vector in
// AITER's order for its position (firstSummand). Per CTA, a start handshake
// (inputs ready) precedes the peer reads and a done handshake (inputs no longer
// read) follows them, so the caller may overwrite its input once the call
// completes. It uses the same per-CTA slots and epochs as the four-rank
// kernels, so all payload kinds may be interleaved on one request.
template <int NRanks>
__device__ __forceinline__ void genericStartHandshake(
    int rank,
    const RegisteredAllReduceStateTable& states,
    uint32_t* targetEpochs,
    uint32_t* sourceEpochs) {
  if (threadIdx.x < NRanks && threadIdx.x != rank) {
    const auto peer = threadIdx.x;
    RegisteredAllReduceBlockState& local = blockState(states, rank);
    uint32_t* remoteStart =
        &blockState(states, peer).phase[static_cast<int>(Phase::Start)][rank];
    uint32_t* localStart = &local.phase[static_cast<int>(Phase::Start)][peer];
    const uint32_t epoch = local.seq[peer] + 1u;
    local.seq[peer] = epoch;
    targetEpochs[peer] = epoch;
    peerStore<true, __ATOMIC_RELAXED>(remoteStart, epoch);
    const uint32_t want = local.seen[peer] + 1u;
    uint32_t got = 0;
    do {
      got = peerLoad<true, __ATOMIC_RELAXED>(localStart);
    } while (!epochReached(got, want));
    local.seen[peer] = got;
    sourceEpochs[peer] = got;
  }
  __syncthreads();
}

template <int NRanks>
__device__ __forceinline__ void genericDoneHandshake(
    int rank,
    const RegisteredAllReduceStateTable& states,
    const uint32_t* targetEpochs,
    const uint32_t* sourceEpochs) {
  // Every peer-input load of this CTA has returned here: each loaded vector
  // feeds its thread's output store before the barrier, so no release is
  // needed to tell peers their inputs are free.
  __syncthreads();
  if (threadIdx.x < NRanks && threadIdx.x != rank) {
    const auto peer = threadIdx.x;
    peerStore<true, __ATOMIC_RELAXED>(
        &blockState(states, peer)
             .phase[static_cast<int>(Phase::Midpoint)][rank],
        targetEpochs[peer]);
    const uint32_t* localSignal =
        &blockState(states, rank)
             .phase[static_cast<int>(Phase::Midpoint)][peer];
    while (!epochReached(
        peerLoad<true, __ATOMIC_RELAXED>(localSignal), sourceEpochs[peer])) {
    }
  }
}

// The rank whose contribution AITER adds first for vector `vector` of a
// `vectors`-vector payload (see kAiterTwoStageMinBytes).
template <int NRanks>
__device__ __forceinline__ int firstSummand(size_t vector, size_t vectors) {
  if constexpr (NRanks == 2) {
    return 0;
  } else {
    if (vectors * sizeof(Vec) < kAiterTwoStageMinBytes) {
      return 0;
    }
    const size_t part = vectors / NRanks;
    return vector < (NRanks - 1) * part ? static_cast<int>(vector / part)
                                        : NRanks - 1;
  }
}

// Contributions are loaded by compile-time source index (a runtime-indexed
// pointer table would leave registers) and summed in rotated order by select.
template <int NRanks>
__device__ __forceinline__ Vec
rotatedContribution(const Vec (&contributions)[NRanks], int source) {
  Vec value = contributions[0];
#pragma unroll
  for (int candidate = 1; candidate < NRanks; ++candidate) {
    value = source == candidate ? contributions[candidate] : value;
  }
  return value;
}

template <int NRanks>
__device__ __forceinline__ Vec reduceVectorGeneric(
    const RegisteredAllReduceInputTable& inputs,
    int rank,
    const Vec& own,
    size_t vectorIndex,
    size_t vectors) {
  Vec contributions[NRanks];
#pragma unroll
  for (int source = 0; source < NRanks; ++source) {
    contributions[source] = source == rank
        ? own
        : reinterpret_cast<const Vec*>(inputs.input[source])[vectorIndex];
  }
  const int first = firstSummand<NRanks>(vectorIndex, vectors);
  Acc sum = widen(rotatedContribution<NRanks>(contributions, first));
#pragma unroll
  for (int step = 1; step < NRanks; ++step) {
    sum += widen(
        rotatedContribution<NRanks>(contributions, (first + step) % NRanks));
  }
  return narrow(sum);
}

// Each CTA owns a contiguous range of vectors, strided over its threads.
template <int NRanks>
__global__ void __launch_bounds__(kRegisteredAllReduceThreads)
    registeredAllReduceGenericKernel(
        __nv_bfloat16* __restrict__ output,
        int rank,
        RegisteredAllReduceInputTable inputs,
        RegisteredAllReduceStateTable states,
        size_t vectors) {
  __shared__ uint32_t targetEpochs[kRegisteredAllReduceRanks];
  __shared__ uint32_t sourceEpochs[kRegisteredAllReduceRanks];
  const size_t perBlock = (vectors + gridDim.x - 1) / gridDim.x;
  const size_t begin = blockIdx.x * perBlock;
  const size_t end = begin + perBlock < vectors ? begin + perBlock : vectors;
  const Vec* own = reinterpret_cast<const Vec*>(inputs.input[rank]);
  Vec* destination = reinterpret_cast<Vec*>(output);
  // The first vector's local input overlaps the start handshake.
  const size_t first = begin + threadIdx.x;
  Vec firstOwn;
  if (first < end) {
    firstOwn = own[first];
  }
  genericStartHandshake<NRanks>(rank, states, targetEpochs, sourceEpochs);
  for (size_t vector = first; vector < end;
       vector += kRegisteredAllReduceThreads) {
    const Vec local = vector == first ? firstOwn : own[vector];
    destination[vector] =
        reduceVectorGeneric<NRanks>(inputs, rank, local, vector, vectors);
  }
  genericDoneHandshake<NRanks>(rank, states, targetEpochs, sourceEpochs);
}

// One vector per thread until the grid reaches its cap; more CTAs finish the
// small decode payloads sooner than fewer, fuller ones.
int genericBlocks(size_t vectors) {
  constexpr size_t kVectorsPerBlock = kRegisteredAllReduceThreads;
  const size_t blocks = (vectors + kVectorsPerBlock - 1) / kVectorsPerBlock;
  return static_cast<int>(
      blocks < kRegisteredAllReduceGenericMaxBlocks
          ? blocks
          : kRegisteredAllReduceGenericMaxBlocks);
}

template <int NRanks>
void launchGeneric(
    __nv_bfloat16* output,
    int rank,
    const RegisteredAllReduceInputTable& inputs,
    const RegisteredAllReduceStateTable& states,
    size_t vectors,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      registeredAllReduceGenericKernel<NRanks>,
      dim3(genericBlocks(vectors)),
      dim3(kRegisteredAllReduceThreads),
      0,
      stream,
      output,
      rank,
      inputs,
      states,
      vectors);
}

__global__ void registeredAllReduceProbeKernel(
    const unsigned char* const* addresses,
    int count,
    uint64_t* out) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) {
    return;
  }
  const uint4 value = *reinterpret_cast<const uint4*>(addresses[i]);
  out[2 * i] = (static_cast<uint64_t>(value.y) << 32) | value.x;
  out[2 * i + 1] = (static_cast<uint64_t>(value.w) << 32) | value.z;
}

// Wide-row gated-residual-norm epilogue. Arithmetic order matches the reference
// Triton post-norm/gated-residual/pre-norm kernel at its 8-warp configuration
// for 4608-element rows: each thread sums its first eight squares with the
// same multiply-add chain as the 8192 epilogue, then adds its (masked) second
// eight squares one by one; the DPP wave tree is unchanged and the cross-wave
// butterfly spans 8 waves. The mean is a correctly rounded division by the
// hidden size, then the epsilon is added. Every rank reduces its rows itself
// (one-shot), so no scratch and no row owner are involved.
constexpr int kWideNormWaves = kRegisteredAllReduceWideNormThreads / 64;
constexpr int kWideNormHeadElements =
    kRegisteredAllReduceWideNormThreads * kElementsPerVector;
constexpr int kWideNormTailThreads =
    (kRegisteredAllReduceWideNormHidden - kWideNormHeadElements) /
    kElementsPerVector;
static_assert(kWideNormWaves == 8);
static_assert(
    kWideNormTailThreads > 0 &&
    kWideNormTailThreads <= kRegisteredAllReduceWideNormThreads);

__device__ __forceinline__ float wideRowSum(float value, float* waveSums) {
  constexpr int kRowShr = 0x110;
  constexpr int kRowBcast15 = 0x142;
  constexpr int kRowBcast31 = 0x143;
  value = value + dppUpdate<kRowShr + 8>(0.0f, value);
  value = value + dppUpdate<kRowShr + 4>(0.0f, value);
  value = value + dppUpdate<kRowShr + 2>(0.0f, value);
  value = value + dppUpdate<kRowShr + 1>(0.0f, value);
  value = dppUpdate<kRowBcast15, 0xa>(value, value) + value;
  value = value + dppUpdate<kRowBcast31>(0.0f, value);
  const float waveTotal = __builtin_bit_cast(
      float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, value), 63));
  if (threadIdx.x % 64 == 0) {
    waveSums[threadIdx.x / 64] = waveTotal;
  }
  __syncthreads();
  float sums[kWideNormWaves];
#pragma unroll
  for (int wave = 0; wave < kWideNormWaves; ++wave) {
    sums[wave] = waveSums[wave];
  }
#pragma unroll
  for (int step = kWideNormWaves / 2; step >= 1; step /= 2) {
#pragma unroll
    for (int wave = 0; wave < step; ++wave) {
      sums[wave] = sums[wave] + sums[wave + step];
    }
  }
  return sums[0];
}

__device__ __forceinline__ float wideSquareSum(
    const float (&head)[kElementsPerVector],
    const float (&tail)[kElementsPerVector],
    bool hasTail) {
#pragma clang fp contract(off)
  float sum = normSquareSum(head);
#pragma unroll
  for (int element = 0; element < kElementsPerVector; ++element) {
    sum = sum + (hasTail ? tail[element] * tail[element] : 0.0f);
  }
  return sum;
}

__device__ __forceinline__ float wideNormScale(float sum, float epsilon) {
#pragma clang fp contract(off)
  constexpr float kHidden = kRegisteredAllReduceWideNormHidden;
  return __builtin_amdgcn_rsqf(__fdiv_rn(sum, kHidden) + epsilon);
}

// One eight-element slice of a row: its reduction, operands, and results.
struct WideSlice {
  size_t offset;
  int channel;
  Vec reduced;
  float branch[kElementsPerVector];
  float residual[kElementsPerVector];
  float preInput[kElementsPerVector];
  float gateAlpha[kElementsPerVector];
  float gateBeta[kElementsPerVector];
  float residualIn[kElementsPerVector];
  Vec postNormWeight;
  Vec preNormWeight;
};

template <bool PostWeight>
__device__ __forceinline__ void loadWideOperands(
    WideSlice& slice,
    const RegisteredAllReduceGatedResidualNormArgs& norm) {
  loadFloats(slice.residualIn, norm.residualIn + slice.offset);
  loadFloats(slice.gateAlpha, norm.gateAlpha + slice.channel);
  loadFloats(slice.gateBeta, norm.gateBeta + slice.channel);
  slice.preNormWeight =
      *reinterpret_cast<const Vec*>(norm.preNormWeight + slice.channel);
  if constexpr (PostWeight) {
    slice.postNormWeight =
        *reinterpret_cast<const Vec*>(norm.postNormWeight + slice.channel);
  }
}

template <bool PostWeight>
__device__ __forceinline__ void wideResidual(
    WideSlice& slice,
    float postScale) {
#pragma clang fp contract(off)
#pragma unroll
  for (int element = 0; element < kElementsPerVector; ++element) {
    float scaled = slice.branch[element] * postScale;
    if constexpr (PostWeight) {
      scaled = scaled * static_cast<float>(slice.postNormWeight[element]);
    }
    const float normedBranch = roundToBf16(scaled);
    slice.residual[element] = __builtin_fmaf(
        slice.gateBeta[element],
        normedBranch,
        slice.gateAlpha[element] * slice.residualIn[element]);
    slice.preInput[element] = roundToBf16(slice.residual[element]);
  }
}

__device__ __forceinline__ void wideStore(
    const WideSlice& slice,
    float preScale,
    __nv_bfloat16* __restrict__ output,
    const RegisteredAllReduceGatedResidualNormArgs& norm) {
#pragma clang fp contract(off)
  storeFloats(norm.residualOut + slice.offset, slice.residual);
  Vec normed;
  float normedFloat[kElementsPerVector];
#pragma unroll
  for (int element = 0; element < kElementsPerVector; ++element) {
    normedFloat[element] = roundToBf16(
        slice.preInput[element] * preScale *
        static_cast<float>(slice.preNormWeight[element]));
    normed[element] = static_cast<__bf16>(normedFloat[element]);
  }
  *reinterpret_cast<Vec*>(output + slice.offset) = normed;
  if (norm.routerOut != nullptr) {
    storeFloats(norm.routerOut + slice.offset, normedFloat);
  }
}

template <int NRanks, bool PostWeight>
__global__ void __launch_bounds__(kRegisteredAllReduceWideNormThreads)
    registeredAllReduceWideGatedResidualNormKernel(
        __nv_bfloat16* __restrict__ output,
        int rank,
        RegisteredAllReduceInputTable inputs,
        RegisteredAllReduceStateTable states,
        RegisteredAllReduceGatedResidualNormArgs norm) {
  __shared__ uint32_t targetEpochs[kRegisteredAllReduceRanks];
  __shared__ uint32_t sourceEpochs[kRegisteredAllReduceRanks];
  __shared__ float postSums[kWideNormWaves];
  __shared__ float preSums[kWideNormWaves];
  const bool hasTail = threadIdx.x < kWideNormTailThreads;
  const size_t rowOffset =
      static_cast<size_t>(blockIdx.x) * kRegisteredAllReduceWideNormHidden;
  WideSlice head;
  WideSlice tail;
  head.channel = threadIdx.x * kElementsPerVector;
  head.offset = rowOffset + head.channel;
  tail.channel = kWideNormHeadElements + head.channel;
  tail.offset = rowOffset + tail.channel;
  const Vec* own = reinterpret_cast<const Vec*>(inputs.input[rank]);
  const Vec headOwn = own[head.offset / kElementsPerVector];
  Vec tailOwn;
  loadWideOperands<PostWeight>(head, norm);
  if (hasTail) {
    tailOwn = own[tail.offset / kElementsPerVector];
    loadWideOperands<PostWeight>(tail, norm);
  }

  genericStartHandshake<NRanks>(rank, states, targetEpochs, sourceEpochs);
  const size_t vectors = static_cast<size_t>(gridDim.x) *
      (kRegisteredAllReduceWideNormHidden / kElementsPerVector);
  head.reduced = reduceVectorGeneric<NRanks>(
      inputs, rank, headOwn, head.offset / kElementsPerVector, vectors);
  if (hasTail) {
    tail.reduced = reduceVectorGeneric<NRanks>(
        inputs, rank, tailOwn, tail.offset / kElementsPerVector, vectors);
  }
#pragma unroll
  for (int element = 0; element < kElementsPerVector; ++element) {
    head.branch[element] = static_cast<float>(head.reduced[element]);
    tail.branch[element] =
        hasTail ? static_cast<float>(tail.reduced[element]) : 0.0f;
  }

  const float postScale = wideNormScale(
      wideRowSum(wideSquareSum(head.branch, tail.branch, hasTail), postSums),
      norm.postNormEpsilon);
  wideResidual<PostWeight>(head, postScale);
  if (hasTail) {
    wideResidual<PostWeight>(tail, postScale);
  }
  const float preScale = wideNormScale(
      wideRowSum(wideSquareSum(head.preInput, tail.preInput, hasTail), preSums),
      norm.preNormEpsilon);
  wideStore(head, preScale, output, norm);
  if (hasTail) {
    wideStore(tail, preScale, output, norm);
  }
  genericDoneHandshake<NRanks>(rank, states, targetEpochs, sourceEpochs);
}

template <int NRanks, bool PostWeight>
void launchWideNorm(
    __nv_bfloat16* output,
    int rank,
    const RegisteredAllReduceInputTable& inputs,
    const RegisteredAllReduceStateTable& states,
    const RegisteredAllReduceGatedResidualNormArgs& norm,
    int rows,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      (registeredAllReduceWideGatedResidualNormKernel<NRanks, PostWeight>),
      dim3(rows),
      dim3(kRegisteredAllReduceWideNormThreads),
      0,
      stream,
      output,
      rank,
      inputs,
      states,
      norm);
}

template <int NRanks>
void launchWideNormRanks(
    __nv_bfloat16* output,
    int rank,
    const RegisteredAllReduceInputTable& inputs,
    const RegisteredAllReduceStateTable& states,
    const RegisteredAllReduceGatedResidualNormArgs& norm,
    int rows,
    hipStream_t stream) {
  if (norm.postNormWeight != nullptr) {
    launchWideNorm<NRanks, true>(
        output, rank, inputs, states, norm, rows, stream);
  } else {
    launchWideNorm<NRanks, false>(
        output, rank, inputs, states, norm, rows, stream);
  }
}

// Row epilogue for widths and Triton launch configurations [redacted] decodes
// with at the served row counts: 8192-element rows at 16 warps and
// 4096-element rows at 8 warps. Threads is
// 64 x num_warps, and thread t owns elements [8t + 8 * Threads * j, +8) for
// j < Vectors (Triton's blocked layout). Each thread accumulates its squares in
// element order (the first two contracted as fma(e0, e0, e1 * e1)), then the
// DPP wave tree and a cross-wave butterfly over num_warps waves; the mean is
// fma(sum, 1 / Hidden, epsilon) (Hidden is a power of two). The post-norm
// weight is optional.
template <int Hidden, int Threads>
struct RowNormShape {
  static constexpr int kWaves = Threads / 64;
  static constexpr int kVectors = Hidden / (kElementsPerVector * Threads);
  static constexpr int kRowVectors = Hidden / kElementsPerVector;
  static_assert(kVectors >= 1);
  static_assert(kVectors * kElementsPerVector * Threads == Hidden);
};

template <int Waves>
__device__ __forceinline__ float rowTreeSum(float value, float* waveSums) {
  constexpr int kRowShr = 0x110;
  constexpr int kRowBcast15 = 0x142;
  constexpr int kRowBcast31 = 0x143;
  value = value + dppUpdate<kRowShr + 8>(0.0f, value);
  value = value + dppUpdate<kRowShr + 4>(0.0f, value);
  value = value + dppUpdate<kRowShr + 2>(0.0f, value);
  value = value + dppUpdate<kRowShr + 1>(0.0f, value);
  value = dppUpdate<kRowBcast15, 0xa>(value, value) + value;
  value = value + dppUpdate<kRowBcast31>(0.0f, value);
  const float waveTotal = __builtin_bit_cast(
      float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, value), 63));
  if (threadIdx.x % 64 == 0) {
    waveSums[threadIdx.x / 64] = waveTotal;
  }
  __syncthreads();
  float sums[Waves];
#pragma unroll
  for (int wave = 0; wave < Waves; ++wave) {
    sums[wave] = waveSums[wave];
  }
#pragma unroll
  for (int step = Waves / 2; step >= 1; step /= 2) {
#pragma unroll
    for (int wave = 0; wave < step; ++wave) {
      sums[wave] = sums[wave] + sums[wave + step];
    }
  }
  return sums[0];
}

template <int Vectors>
__device__ __forceinline__ float rowSquareSum(
    const float (&values)[Vectors][kElementsPerVector]) {
#pragma clang fp contract(off)
  float sum = values[0][1] * values[0][1];
  sum = __builtin_fmaf(values[0][0], values[0][0], sum);
#pragma unroll
  for (int element = 2; element < kElementsPerVector; ++element) {
    sum = __builtin_fmaf(values[0][element], values[0][element], sum);
  }
#pragma unroll
  for (int j = 1; j < Vectors; ++j) {
#pragma unroll
    for (int element = 0; element < kElementsPerVector; ++element) {
      sum = __builtin_fmaf(values[j][element], values[j][element], sum);
    }
  }
  return sum;
}

template <int Hidden>
__device__ __forceinline__ float rowNormScale(float sum, float epsilon) {
  constexpr float kInverseHidden = 1.0f / Hidden;
  return __builtin_amdgcn_rsqf(__builtin_fmaf(sum, kInverseHidden, epsilon));
}

// Operands of one thread's slices that do not depend on the reduction.
template <int Vectors>
struct RowNormOperands {
  Vec postNormWeight[Vectors];
  Vec preNormWeight[Vectors];
  float residual[Vectors][kElementsPerVector];
  float gateAlpha[Vectors][kElementsPerVector];
  float gateBeta[Vectors][kElementsPerVector];
};

template <int Hidden, int Threads, bool PostWeight>
__device__ __forceinline__
    RowNormOperands<RowNormShape<Hidden, Threads>::kVectors>
    loadRowNormOperands(
        size_t rowOffset,
        const RegisteredAllReduceGatedResidualNormArgs& norm) {
  constexpr int kVectors = RowNormShape<Hidden, Threads>::kVectors;
  RowNormOperands<kVectors> operands;
#pragma unroll
  for (int j = 0; j < kVectors; ++j) {
    const int channel = (j * Threads + threadIdx.x) * kElementsPerVector;
    if constexpr (PostWeight) {
      operands.postNormWeight[j] =
          *reinterpret_cast<const Vec*>(norm.postNormWeight + channel);
    }
    operands.preNormWeight[j] =
        *reinterpret_cast<const Vec*>(norm.preNormWeight + channel);
    loadFloats(operands.residual[j], norm.residualIn + rowOffset + channel);
    loadFloats(operands.gateAlpha[j], norm.gateAlpha + channel);
    loadFloats(operands.gateBeta[j], norm.gateBeta + channel);
  }
  return operands;
}

template <int Hidden, int Threads, bool PostWeight>
__device__ __forceinline__ void rowNormEpilogue(
    const Vec (&reduced)[RowNormShape<Hidden, Threads>::kVectors],
    const RowNormOperands<RowNormShape<Hidden, Threads>::kVectors>& operands,
    size_t rowOffset,
    __nv_bfloat16* __restrict__ output,
    const RegisteredAllReduceGatedResidualNormArgs& norm) {
#pragma clang fp contract(off)
  using Shape = RowNormShape<Hidden, Threads>;
  constexpr int kVectors = Shape::kVectors;
  __shared__ float postSums[Shape::kWaves];
  __shared__ float preSums[Shape::kWaves];
  float branch[kVectors][kElementsPerVector];
#pragma unroll
  for (int j = 0; j < kVectors; ++j) {
#pragma unroll
    for (int element = 0; element < kElementsPerVector; ++element) {
      branch[j][element] = static_cast<float>(reduced[j][element]);
    }
  }
  const float postScale = rowNormScale<Hidden>(
      rowTreeSum<Shape::kWaves>(rowSquareSum<kVectors>(branch), postSums),
      norm.postNormEpsilon);
  float residual[kVectors][kElementsPerVector];
  float preInput[kVectors][kElementsPerVector];
#pragma unroll
  for (int j = 0; j < kVectors; ++j) {
#pragma unroll
    for (int element = 0; element < kElementsPerVector; ++element) {
      float scaled = branch[j][element] * postScale;
      if constexpr (PostWeight) {
        scaled =
            scaled * static_cast<float>(operands.postNormWeight[j][element]);
      }
      const float normedBranch = roundToBf16(scaled);
      residual[j][element] = __builtin_fmaf(
          operands.gateBeta[j][element],
          normedBranch,
          operands.gateAlpha[j][element] * operands.residual[j][element]);
      preInput[j][element] = roundToBf16(residual[j][element]);
    }
  }
#pragma unroll
  for (int j = 0; j < kVectors; ++j) {
    const size_t offset =
        rowOffset + (j * Threads + threadIdx.x) * kElementsPerVector;
    storeFloats(norm.residualOut + offset, residual[j]);
  }
  const float preScale = rowNormScale<Hidden>(
      rowTreeSum<Shape::kWaves>(rowSquareSum<kVectors>(preInput), preSums),
      norm.preNormEpsilon);
#pragma unroll
  for (int j = 0; j < kVectors; ++j) {
    const size_t offset =
        rowOffset + (j * Threads + threadIdx.x) * kElementsPerVector;
    Vec normed;
    float normedFloat[kElementsPerVector];
#pragma unroll
    for (int element = 0; element < kElementsPerVector; ++element) {
      normedFloat[element] = roundToBf16(
          preInput[j][element] * preScale *
          static_cast<float>(operands.preNormWeight[j][element]));
      normed[element] = static_cast<__bf16>(normedFloat[element]);
    }
    *reinterpret_cast<Vec*>(output + offset) = normed;
    if (norm.routerOut != nullptr) {
      storeFloats(norm.routerOut + offset, normedFloat);
    }
  }
}

// One CTA per row on every rank. Rank q reduces quarter q of the row's
// 16-byte vectors (each in AITER's order for its position in the payload) into
// its scratch and publishes it to every peer; every rank then gathers the four
// quarters and runs the epilogue on the full row. A rank reads its peers'
// inputs only after each has published its start epoch for the row, and
// overwrites its scratch quarter only after every peer has started the next
// call (so has finished reading this one). The plain reduced row is never
// written.
template <int Hidden, int Threads, bool PostWeight>
__global__ void __launch_bounds__(Threads) registeredAllReduceRowNormKernel(
    __nv_bfloat16* __restrict__ output,
    int rank,
    RegisteredAllReduceInputTable inputs,
    RegisteredAllReduceStateTable states,
    RegisteredAllReduceGatedResidualNormArgs norm,
    int rows) {
  using Shape = RowNormShape<Hidden, Threads>;
  constexpr int kVectors = Shape::kVectors;
  constexpr int kQuarterVectors =
      Shape::kRowVectors / kRegisteredAllReduceRanks;
  constexpr int kLoads =
      (kRegisteredAllReduceRanks * kQuarterVectors + Threads - 1) / Threads;
  static_assert(kQuarterVectors <= Threads);
  // Every thread loads part of the four contributions to this rank's quarter,
  // so all of the CTA's loads are in flight at once.
  __shared__ Vec staged[kRegisteredAllReduceRanks][kQuarterVectors];
  __shared__ uint32_t callsShared;
  const auto row = blockIdx.x;
  const size_t rowOffset = static_cast<size_t>(row) * Hidden;
  const size_t rowVector = static_cast<size_t>(row) * Shape::kRowVectors;
  const size_t vectors = static_cast<size_t>(rows) * Shape::kRowVectors;
  RegisteredAllReduceRowState& local = states.state[rank]->row[row];
  const size_t quarterBegin = rowVector + rank * kQuarterVectors;
  const auto operands =
      loadRowNormOperands<Hidden, Threads, PostWeight>(rowOffset, norm);
  if (threadIdx.x == 0) {
    const uint32_t calls = local.rowCalls + 1u;
    local.rowCalls = calls;
    callsShared = calls;
  }
  __syncthreads();
  const uint32_t calls = callsShared;
  if (threadIdx.x < kRegisteredAllReduceRanks && threadIdx.x != rank) {
    peerStore<true, __ATOMIC_RELAXED>(
        &states.state[threadIdx.x]->row[row].rowStart[rank], calls);
    while (!epochReached(
        peerLoad<true, __ATOMIC_RELAXED>(&local.rowStart[threadIdx.x]),
        calls)) {
    }
  }
  __syncthreads();

  Vec* const ownScratch =
      reinterpret_cast<Vec*>(registeredAllReduceScratch(states.state[rank])) +
      static_cast<size_t>(row) * kQuarterVectors;
#pragma unroll
  for (int load = 0; load < kLoads; ++load) {
    const auto index = load * Threads + threadIdx.x;
    if (index < kRegisteredAllReduceRanks * kQuarterVectors) {
      const int source = index / kQuarterVectors;
      const int vector = index - source * kQuarterVectors;
      staged[source][vector] = reinterpret_cast<const Vec*>(
          inputs.input[source])[quarterBegin + vector];
    }
  }
  __syncthreads();
  if (threadIdx.x < kQuarterVectors) {
    const int first = firstSummand<kRegisteredAllReduceRanks>(
        quarterBegin + threadIdx.x, vectors);
    Acc sum = widen(staged[first][threadIdx.x]);
#pragma unroll
    for (int step = 1; step < kRegisteredAllReduceRanks; ++step) {
      sum += widen(
          staged[(first + step) % kRegisteredAllReduceRanks][threadIdx.x]);
    }
    ownScratch[threadIdx.x] = narrow(sum);
  }
  drainWaveStores();
  __syncthreads();
  if (threadIdx.x < kRegisteredAllReduceRanks && threadIdx.x != rank) {
    uint32_t* remote = &states.state[threadIdx.x]->row[row].quarter[rank];
    if constexpr (kUncachedState) {
      peerStore<true, __ATOMIC_RELAXED>(remote, calls);
    } else {
      peerStore<true, __ATOMIC_RELEASE>(remote, calls);
    }
    while (!epochReached(
        peerLoad<true, __ATOMIC_RELAXED>(&local.quarter[threadIdx.x]), calls)) {
    }
    if constexpr (!kUncachedState) {
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);
    }
  }
  __syncthreads();

  Vec reduced[kVectors];
#pragma unroll
  for (int j = 0; j < kVectors; ++j) {
    const auto column = j * Threads + threadIdx.x;
    const int quarter = column / kQuarterVectors;
    reduced[j] =
        (reinterpret_cast<const Vec*>(
             registeredAllReduceScratch(states.state[quarter])) +
         static_cast<size_t>(row) *
             kQuarterVectors)[column - quarter * kQuarterVectors];
  }
  rowNormEpilogue<Hidden, Threads, PostWeight>(
      reduced, operands, rowOffset, output, norm);
}

// The epilogue alone on already-reduced rows, for checking its arithmetic
// against the reference kernel.
template <int Hidden, int Threads, bool PostWeight>
__global__ void __launch_bounds__(Threads) registeredAllReduceRowNormOnlyKernel(
    const __nv_bfloat16* __restrict__ reducedRows,
    __nv_bfloat16* __restrict__ output,
    RegisteredAllReduceGatedResidualNormArgs norm) {
  using Shape = RowNormShape<Hidden, Threads>;
  const size_t rowOffset = static_cast<size_t>(blockIdx.x) * Hidden;
  const auto operands =
      loadRowNormOperands<Hidden, Threads, PostWeight>(rowOffset, norm);
  Vec reduced[Shape::kVectors];
#pragma unroll
  for (int j = 0; j < Shape::kVectors; ++j) {
    reduced[j] = reinterpret_cast<const Vec*>(
        reducedRows + rowOffset)[j * Threads + threadIdx.x];
  }
  rowNormEpilogue<Hidden, Threads, PostWeight>(
      reduced, operands, rowOffset, output, norm);
}

// Served row counts per width; the Triton kernel runs 16 warps for 8192 and 8
// for 4096 there.
constexpr int rowNormMaxRows(size_t hidden) {
  return hidden == kRegisteredAllReduceNormHidden
      ? kRegisteredAllReduceRowNormMaxRows
      : hidden == kRegisteredAllReduceNarrowNormHidden
      ? kRegisteredAllReduceNarrowRowNormMaxRows
      : 0;
}

template <template <int, int, bool> class Launch, typename... Args>
hipError_t
dispatchRowNorm(size_t hidden, int rows, bool postWeight, Args&&... args) {
  if (rows < 1 || rows > rowNormMaxRows(hidden)) {
    return hipErrorInvalidValue;
  }
  if (hidden == kRegisteredAllReduceNormHidden) {
    return postWeight ? Launch<8192, 1024, true>::run(args...)
                      : Launch<8192, 1024, false>::run(args...);
  }
  return postWeight ? Launch<4096, 512, true>::run(args...)
                    : Launch<4096, 512, false>::run(args...);
}

template <int Hidden, int Threads, bool PostWeight>
struct LaunchRowNorm {
  static hipError_t run(
      void* output,
      const RegisteredAllReduceInputTable& inputs,
      const RegisteredAllReduceStateTable& states,
      int rank,
      const RegisteredAllReduceGatedResidualNormArgs& norm,
      int rows,
      hipStream_t stream) {
    hipLaunchKernelGGL(
        (registeredAllReduceRowNormKernel<Hidden, Threads, PostWeight>),
        dim3(rows),
        dim3(Threads),
        0,
        stream,
        reinterpret_cast<__nv_bfloat16*>(output),
        rank,
        inputs,
        states,
        norm,
        rows);
    return hipPeekAtLastError();
  }
};

template <int Hidden, int Threads, bool PostWeight>
struct LaunchRowNormOnly {
  static hipError_t run(
      const void* reducedRows,
      void* output,
      const RegisteredAllReduceGatedResidualNormArgs& norm,
      int rows,
      hipStream_t stream) {
    hipLaunchKernelGGL(
        (registeredAllReduceRowNormOnlyKernel<Hidden, Threads, PostWeight>),
        dim3(rows),
        dim3(Threads),
        0,
        stream,
        reinterpret_cast<const __nv_bfloat16*>(reducedRows),
        reinterpret_cast<__nv_bfloat16*>(output),
        norm);
    return hipPeekAtLastError();
  }
};

} // namespace

hipError_t launchRegisteredAllReduceRowNormForTest(
    const void* reducedRows,
    void* output,
    size_t hiddenSize,
    int rows,
    const RegisteredAllReduceGatedResidualNormArgs& norm,
    hipStream_t stream) {
  if (rows < 1 || norm.preNormWeight == nullptr) {
    return hipErrorInvalidValue;
  }
  return dispatchRowNorm<LaunchRowNormOnly>(
      hiddenSize,
      rows,
      norm.postNormWeight != nullptr,
      reducedRows,
      output,
      norm,
      rows,
      stream);
}

hipError_t launchRegisteredAllReduceProbe(
    const unsigned char* const* addresses,
    int count,
    uint64_t* out,
    hipStream_t stream) {
  constexpr int kProbeThreads = 64;
  hipLaunchKernelGGL(
      registeredAllReduceProbeKernel,
      dim3((count + kProbeThreads - 1) / kProbeThreads),
      dim3(kProbeThreads),
      0,
      stream,
      addresses,
      count,
      out);
  return hipPeekAtLastError();
}

hipError_t launchRegisteredAllReduceRelayKernel(
    void* output,
    RegisteredAllReduceInputTable inputs,
    RegisteredAllReduceStateTable states,
    int rank,
    size_t count,
    const RegisteredAllReduceRelayRoute& route,
    hipStream_t stream) {
  const size_t bytes = count * sizeof(__nv_bfloat16);
  if (output == nullptr || rank < 0 || rank >= kRegisteredAllReduceRanks ||
      bytes < kAiterTwoStageMinBytes || bytes % sizeof(Vec) != 0 ||
      route.helpers < 1 ||
      route.helpers > kRegisteredAllReduceRelayMaxHelpers ||
      route.directWeight < 1 || route.helperBlocks < route.helpers ||
      route.helperBlocks % route.helpers != 0 ||
      route.helperBlocks >= kRegisteredAllReduceMaxBlocks) {
    return hipErrorInvalidValue;
  }
  const size_t vectors = bytes / sizeof(Vec);
  hipLaunchKernelGGL(
      registeredAllReduceRelayKernel,
      dim3(relayDirectBlocks(vectors, route) + route.helperBlocks),
      dim3(kRelayThreads),
      0,
      stream,
      reinterpret_cast<__nv_bfloat16*>(output),
      rank,
      inputs,
      states,
      vectors,
      route);
  return hipPeekAtLastError();
}

hipError_t launchRegisteredAllReduceKernel(
    void* output,
    RegisteredAllReduceInputTable inputs,
    RegisteredAllReduceStateTable states,
    int rank,
    size_t count,
    const RegisteredAllReduceGatedResidualNormArgs* norm,
    hipStream_t stream,
    int nRanks,
    size_t hiddenSize) {
  const size_t bytes = count * sizeof(__nv_bfloat16);
  if (!isRegisteredAllReduceRankCount(nRanks) || rank < 0 || rank >= nRanks) {
    return hipErrorInvalidValue;
  }
  const bool fourRanks = nRanks == kRegisteredAllReduceRanks;
  if (norm != nullptr && hiddenSize == kRegisteredAllReduceWideNormHidden) {
    const size_t rows = count / kRegisteredAllReduceWideNormHidden;
    if (count % kRegisteredAllReduceWideNormHidden != 0 || rows == 0 ||
        rows > kRegisteredAllReduceWideNormMaxRows ||
        norm->preNormWeight == nullptr) {
      return hipErrorInvalidValue;
    }
    auto* const typedOutput = reinterpret_cast<__nv_bfloat16*>(output);
    if (fourRanks) {
      launchWideNormRanks<kRegisteredAllReduceRanks>(
          typedOutput,
          rank,
          inputs,
          states,
          *norm,
          static_cast<int>(rows),
          stream);
    } else {
      launchWideNormRanks<2>(
          typedOutput,
          rank,
          inputs,
          states,
          *norm,
          static_cast<int>(rows),
          stream);
    }
  } else if (
      norm != nullptr &&
      !(hiddenSize == kRegisteredAllReduceNormHidden &&
        bytes == kRegisteredAllReduceOneMiBBytes &&
        norm->postNormWeight != nullptr)) {
    const size_t rows = hiddenSize == 0 ? 0 : count / hiddenSize;
    if (!fourRanks || rows == 0 || count % hiddenSize != 0 ||
        rows > static_cast<size_t>(rowNormMaxRows(hiddenSize)) ||
        norm->preNormWeight == nullptr) {
      return hipErrorInvalidValue;
    }
    return dispatchRowNorm<LaunchRowNorm>(
        hiddenSize,
        static_cast<int>(rows),
        norm->postNormWeight != nullptr,
        output,
        inputs,
        states,
        rank,
        *norm,
        static_cast<int>(rows),
        stream);
  } else if (norm != nullptr) {
    if (!fourRanks || bytes != kRegisteredAllReduceOneMiBBytes ||
        hiddenSize != kRegisteredAllReduceNormHidden) {
      return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        registeredAllReduceGatedResidualNormKernel,
        dim3(kRegisteredAllReduceRows),
        dim3(kRegisteredAllReduceNormThreads),
        0,
        stream,
        reinterpret_cast<__nv_bfloat16*>(output),
        rank,
        inputs,
        states,
        *norm);
  } else if (fourRanks && bytes == kRegisteredAllReduceHalfMiBBytes) {
    hipLaunchKernelGGL(
        registeredAllReduceHalfMiBKernel,
        dim3(kRegisteredAllReduceHalfMiBBlocks),
        dim3(kRegisteredAllReduceThreads),
        0,
        stream,
        reinterpret_cast<__nv_bfloat16*>(output),
        rank,
        inputs,
        states);
  } else if (fourRanks && bytes == kRegisteredAllReduceOneMiBBytes) {
    hipLaunchKernelGGL(
        registeredAllReduceOneMiBKernel,
        dim3(kRegisteredAllReduceOneMiBBlocks),
        dim3(kRegisteredAllReduceThreads),
        0,
        stream,
        reinterpret_cast<__nv_bfloat16*>(output),
        rank,
        inputs,
        states);
  } else if (bytes == 0 || bytes % sizeof(Vec) != 0) {
    return hipErrorInvalidValue;
  } else if (fourRanks && bytes >= kAiterTwoStageMinBytes) {
    const size_t vectors = bytes / sizeof(Vec);
    hipLaunchKernelGGL(
        registeredAllReduceTwoStageKernel,
        dim3(twoStageBlocks(vectors)),
        dim3(kTwoStageThreads),
        0,
        stream,
        reinterpret_cast<__nv_bfloat16*>(output),
        rank,
        inputs,
        states,
        vectors);
  } else if (fourRanks) {
    launchGeneric<kRegisteredAllReduceRanks>(
        reinterpret_cast<__nv_bfloat16*>(output),
        rank,
        inputs,
        states,
        bytes / sizeof(Vec),
        stream);
  } else {
    launchGeneric<2>(
        reinterpret_cast<__nv_bfloat16*>(output),
        rank,
        inputs,
        states,
        bytes / sizeof(Vec),
        stream);
  }
  return hipPeekAtLastError();
}

} // namespace rcclx::relay
