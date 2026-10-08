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
static_assert(
    kRegisteredAllReduceMaxScratchBytes == 256 * 1024,
    "A four-rank 1 MiB all-reduce stages one 256 KiB shard per rank");

// Native bf16 vector: each add is an f32 add plus one round-to-nearest-even
// conversion, matching canonical stepwise BF16 at lower instruction cost.
using Vec = __bf16 __attribute__((ext_vector_type(kElementsPerVector)));

enum class Phase : int {
  Start = 0,
  Midpoint = 1,
};

__device__ __forceinline__ bool epochReached(uint32_t got, uint32_t want) {
  return (got - want) <= (uint32_t(-1) >> 1);
}

__device__ __forceinline__ void addContribution(
    Vec& accumulator,
    const Vec& contribution) {
#pragma unroll
  for (int element = 0; element < kElementsPerVector; ++element) {
    accumulator[element] = accumulator[element] + contribution[element];
  }
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

__device__ __forceinline__ Vec contributionOf(
    int source,
    int rank,
    Vec own,
    Vec next,
    Vec opposite,
    Vec previous) {
  const int distance =
      (source - rank + kRegisteredAllReduceRanks) % kRegisteredAllReduceRanks;
  Vec contribution = own;
  contribution = distance == 1 ? next : contribution;
  contribution = distance == 2 ? opposite : contribution;
  contribution = distance == 3 ? previous : contribution;
  return contribution;
}

// Canonical rank 0 -> rank 3 stepwise BF16 reduction; the caller supplies its
// own (already loaded) contribution. Peers are loaded by rotated index and
// placed into canonical order with selects, so a runtime rank adds no
// branches. Named values (not an array) keep the operands in registers.
__device__ __forceinline__ Vec reduceVector(
    const RegisteredAllReduceInputTable& inputs,
    int rank,
    const Vec& own,
    size_t shardOffset,
    size_t vectorIndex) {
  const Vec next = loadPeer(inputs, rank, 1, shardOffset, vectorIndex);
  const Vec opposite = loadPeer(inputs, rank, 2, shardOffset, vectorIndex);
  const Vec previous = loadPeer(inputs, rank, 3, shardOffset, vectorIndex);
  Vec accumulator = contributionOf(0, rank, own, next, opposite, previous);
  addContribution(
      accumulator, contributionOf(1, rank, own, next, opposite, previous));
  addContribution(
      accumulator, contributionOf(2, rank, own, next, opposite, previous));
  addContribution(
      accumulator, contributionOf(3, rank, own, next, opposite, previous));
  return accumulator;
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
    reinterpret_cast<Vec*>(states.state[rank]->scratch)[vector] = reduced;
    drainWaveStores();
  }

  midpointBarrier<ScopedAtomics>(states, rank, targetEpochs, sourceEpochs);

  const int shard = (group + rank) % kRegisteredAllReduceRanks;
  Vec* destination =
      reinterpret_cast<Vec*>(output + static_cast<size_t>(shard) * kShardCount);
  if (group == 0) {
    destination[vector] = reduced;
  } else {
    destination[vector] =
        reinterpret_cast<const Vec*>(states.state[shard]->scratch)[vector];
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

// Generic one-shot path: every rank reduces the whole payload in canonical
// rank 0 -> N-1 stepwise BF16 order. Per CTA, a start handshake (inputs ready)
// precedes the peer reads and a done handshake (inputs no longer read)
// follows them, so the caller may overwrite its input once the call completes.
// It uses the same per-CTA slots and epochs as the four-rank kernels, so all
// payload kinds may be interleaved on one request.
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

template <int NRanks>
__device__ __forceinline__ Vec reduceVectorGeneric(
    const RegisteredAllReduceInputTable& inputs,
    int rank,
    const Vec& own,
    size_t vectorIndex) {
  Vec contributions[NRanks];
#pragma unroll
  for (int source = 0; source < NRanks; ++source) {
    contributions[source] = source == rank
        ? own
        : reinterpret_cast<const Vec*>(inputs.input[source])[vectorIndex];
  }
  Vec accumulator = contributions[0];
#pragma unroll
  for (int source = 1; source < NRanks; ++source) {
    addContribution(accumulator, contributions[source]);
  }
  return accumulator;
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
        reduceVectorGeneric<NRanks>(inputs, rank, local, vector);
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

} // namespace

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

hipError_t launchRegisteredAllReduceKernel(
    void* output,
    RegisteredAllReduceInputTable inputs,
    RegisteredAllReduceStateTable states,
    int rank,
    size_t count,
    hipStream_t stream,
    int nRanks) {
  const size_t bytes = count * sizeof(__nv_bfloat16);
  if (!isRegisteredAllReduceRankCount(nRanks) || rank < 0 || rank >= nRanks) {
    return hipErrorInvalidValue;
  }
  const bool fourRanks = nRanks == kRegisteredAllReduceRanks;
  if (fourRanks && bytes == kRegisteredAllReduceHalfMiBBytes) {
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
