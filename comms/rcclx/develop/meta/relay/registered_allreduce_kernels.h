/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace rcclx::relay {

constexpr int kRegisteredAllReduceRanks = 4;
constexpr int kRegisteredAllReduceThreads = 512;
constexpr int kRegisteredAllReduceHalfMiBBlocks = 64;
constexpr int kRegisteredAllReduceOneMiBBlocks = 128;
// Every kernel's grid is capped here; each CTA owns one protocol slot.
constexpr int kRegisteredAllReduceMaxBlocks = 256;
static_assert(
    kRegisteredAllReduceOneMiBBlocks <= kRegisteredAllReduceMaxBlocks);
constexpr size_t kRegisteredAllReduceHalfMiBBytes = 512 * 1024;
constexpr size_t kRegisteredAllReduceOneMiBBytes = 1024 * 1024;
constexpr size_t kRegisteredAllReduceMaxScratchBytes =
    kRegisteredAllReduceOneMiBBytes / kRegisteredAllReduceRanks;
// Any other 16-byte-aligned payload, and every two-rank payload, runs a generic
// one-shot kernel with at most this many CTAs.
constexpr int kRegisteredAllReduceGenericMaxBlocks = 64;
static_assert(
    kRegisteredAllReduceGenericMaxBlocks <= kRegisteredAllReduceOneMiBBlocks);

constexpr bool isRegisteredAllReduceRankCount(int nRanks) {
  return nRanks == 2 || nRanks == kRegisteredAllReduceRanks;
}

// Protocol words of one CTA. Each CTA owns a separate 256-byte slot so remote
// flag writes, polls, and epoch updates from different CTAs never share a line.
struct alignas(256) RegisteredAllReduceBlockState {
  // Start, midpoint, and the relay's second exchange.
  uint32_t phase[3][kRegisteredAllReduceRanks];
  uint32_t seq[kRegisteredAllReduceRanks];
  uint32_t seen[kRegisteredAllReduceRanks];
  // The relay's second exchange has its own epochs: only the CTAs serving a
  // helper write it, so the shared epochs above also advance on calls that
  // never write it. Every exchange's epochs advance only on the calls that
  // write it, so each word trails its waiter by at most one call.
  uint32_t relaySeq[kRegisteredAllReduceRanks];
  uint32_t relaySeen[kRegisteredAllReduceRanks];
};

// The gated-residual-norm epilogue runs on 1 MiB payloads viewed as 64 rows of
// 8192 elements, one 1024-thread CTA per row (one 16-byte vector per thread).
constexpr int kRegisteredAllReduceRows = 64;
constexpr int kRegisteredAllReduceRowsPerRank =
    kRegisteredAllReduceRows / kRegisteredAllReduceRanks;
constexpr int kRegisteredAllReduceNormHidden = 8192;
constexpr int kRegisteredAllReduceNormThreads = 1024;

// The wide-row epilogue serves 4608-element rows on two or four ranks, 1 to 64
// rows, one 512-thread CTA per row: thread t owns elements [8t, 8t + 8) and,
// for the first 64 threads, [4096 + 8t, 4096 + 8t + 8). The post-norm weight
// is optional.
constexpr int kRegisteredAllReduceWideNormHidden = 4608;
constexpr int kRegisteredAllReduceWideNormThreads = 512;
constexpr int kRegisteredAllReduceWideNormMaxRows = 64;
static_assert(
    kRegisteredAllReduceWideNormMaxRows <= kRegisteredAllReduceMaxBlocks);

// Epilogue rows with their own protocol line (the SBD verify forward carries up
// to 448 rows).
constexpr int kRegisteredAllReduceMaxRows = 512;
static_assert(kRegisteredAllReduceRows <= kRegisteredAllReduceMaxRows);

// Protocol words of one epilogue row CTA, on their own 256-byte line. The row
// owner collects start epochs from the other ranks and publishes the midpoint
// epoch once the reduced row is in its scratch.
struct alignas(256) RegisteredAllReduceRowState {
  uint32_t start[kRegisteredAllReduceRanks];
  uint32_t midpoint;
  uint32_t calls;
};

// The fixed protocol header of a state region. A four-rank request's scratch
// (one shard, capacity / 4) follows it in the same allocation, so the region
// size follows the request's capacity.
struct alignas(256) RegisteredAllReduceStateRegion {
  RegisteredAllReduceBlockState block[kRegisteredAllReduceMaxBlocks];
  RegisteredAllReduceRowState row[kRegisteredAllReduceMaxRows];
  uint64_t sentinel;
};

// The largest shard of a four-rank request: a quarter of its 16-byte vectors,
// plus the up to three vectors the last quarter takes when the count does not
// divide by four (AITER's split).
constexpr size_t registeredAllReduceScratchBytes(
    size_t capacityBytes,
    int nRanks) {
  return nRanks == kRegisteredAllReduceRanks
      ? (capacityBytes / 16 / kRegisteredAllReduceRanks +
         kRegisteredAllReduceRanks - 1) *
          16
      : 0;
}

__host__ __device__ inline unsigned char* registeredAllReduceScratch(
    RegisteredAllReduceStateRegion* state) {
  return reinterpret_cast<unsigned char*>(state + 1);
}

struct RegisteredAllReduceInputTable {
  const __nv_bfloat16* input[kRegisteredAllReduceRanks];
};

struct RegisteredAllReduceStateTable {
  RegisteredAllReduceStateRegion* state[kRegisteredAllReduceRanks];
};

// Rank-local operands of the gated-residual-norm epilogue (see
// ncclRegisteredAllReduceGatedResidualNorm). residualOut may alias residualIn;
// routerOut may be null.
struct RegisteredAllReduceGatedResidualNormArgs {
  const float* residualIn;
  float* residualOut;
  float* routerOut;
  const __nv_bfloat16* postNormWeight;
  const __nv_bfloat16* preNormWeight;
  const float* gateAlpha;
  const float* gateBeta;
  float postNormEpsilon;
  float preNormEpsilon;
};

// Helper-GPU route of a four-rank two-stage all-reduce. Each quarter is cut
// into kRegisteredAllReduceRelayChunkVectors-vector chunks; of every
// directWeight + helpers consecutive chunks the first directWeight run the
// two-stage kernel over the direct links and chunk directWeight + h is relayed
// through helper GPU h. staging[r][h] is rank r's buffer in helper h's HBM
// (mapped here): slots [q * slotChunks, (q + 1) * slotChunks) hold r's
// contribution to owner q's relayed chunks, and slots [4 * slotChunks,
// 5 * slotChunks) hold r's reduced relayed chunks of its own quarter.
constexpr int kRegisteredAllReduceRelayMaxHelpers = 4;
constexpr size_t kRegisteredAllReduceRelayChunkVectors = 128;

struct RegisteredAllReduceRelayRoute {
  void* staging[kRegisteredAllReduceRanks][kRegisteredAllReduceRelayMaxHelpers];
  size_t slotChunks;
  int helpers;
  int directWeight;
  int helperBlocks;
};

// Chunks of helper h's staging slots per owner for a request of this capacity.
constexpr size_t registeredAllReduceRelaySlotChunks(
    size_t capacityBytes,
    int directWeight,
    int helpers) {
  const size_t quarterVectors = capacityBytes / 16 / kRegisteredAllReduceRanks +
      kRegisteredAllReduceRanks;
  const size_t chunks =
      (quarterVectors + kRegisteredAllReduceRelayChunkVectors - 1) /
      kRegisteredAllReduceRelayChunkVectors;
  const size_t cycle = static_cast<size_t>(directWeight) + helpers;
  return (chunks + cycle - 1) / cycle;
}

constexpr size_t registeredAllReduceRelayStagingBytes(size_t slotChunks) {
  return (kRegisteredAllReduceRanks + 1) * slotChunks *
      kRegisteredAllReduceRelayChunkVectors * 16;
}

// Four-rank BF16 SUM of `count` elements over the direct links and the
// route's helper GPUs, bitwise equal to the two-stage kernel.
hipError_t launchRegisteredAllReduceRelayKernel(
    void* output,
    RegisteredAllReduceInputTable inputs,
    RegisteredAllReduceStateTable states,
    int rank,
    size_t count,
    const RegisteredAllReduceRelayRoute& route,
    hipStream_t stream);

// Enqueues exactly one payload-specific kernel and never synchronizes stream.
// The request lifecycle owns cross-call ordering and final stream quiescence.
// With norm, output receives the pre-norm output of the epilogue instead of
// the plain sum.
// nRanks is 2 or 4. Four-rank 0.5/1 MiB payloads and the epilogue keep their
// dedicated kernels; the epilogue requires four ranks.
hipError_t launchRegisteredAllReduceKernel(
    void* output,
    RegisteredAllReduceInputTable inputs,
    RegisteredAllReduceStateTable states,
    int rank,
    size_t count,
    const RegisteredAllReduceGatedResidualNormArgs* norm,
    hipStream_t stream,
    int nRanks = kRegisteredAllReduceRanks,
    size_t hiddenSize = kRegisteredAllReduceNormHidden);

// Reads 16 bytes at each of `count` addresses (a device array) into out[2i],
// out[2i + 1] with plain 16-byte loads, the way the kernels read peer inputs.
hipError_t launchRegisteredAllReduceProbe(
    const unsigned char* const* addresses,
    int count,
    uint64_t* out,
    hipStream_t stream);

} // namespace rcclx::relay
