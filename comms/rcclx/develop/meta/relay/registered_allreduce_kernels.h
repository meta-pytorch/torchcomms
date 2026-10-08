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
constexpr int kRegisteredAllReduceMaxBlocks = kRegisteredAllReduceOneMiBBlocks;
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
  uint32_t phase[2][kRegisteredAllReduceRanks];
  uint32_t seq[kRegisteredAllReduceRanks];
  uint32_t seen[kRegisteredAllReduceRanks];
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
    kRegisteredAllReduceWideNormMaxRows <= kRegisteredAllReduceOneMiBBlocks);

// Protocol words of one epilogue row CTA, on their own 256-byte line. The row
// owner collects start epochs from the other ranks and publishes the midpoint
// epoch once the reduced row is in its scratch.
struct alignas(256) RegisteredAllReduceRowState {
  uint32_t start[kRegisteredAllReduceRanks];
  uint32_t midpoint;
  uint32_t calls;
};

struct alignas(256) RegisteredAllReduceStateRegion {
  alignas(16) unsigned char scratch[kRegisteredAllReduceMaxScratchBytes];
  RegisteredAllReduceBlockState block[kRegisteredAllReduceMaxBlocks];
  RegisteredAllReduceRowState row[kRegisteredAllReduceRows];
  uint64_t sentinel;
};

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
