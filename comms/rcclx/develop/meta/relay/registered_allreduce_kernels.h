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

struct alignas(256) RegisteredAllReduceStateRegion {
  alignas(16) unsigned char scratch[kRegisteredAllReduceMaxScratchBytes];
  RegisteredAllReduceBlockState block[kRegisteredAllReduceMaxBlocks];
  uint64_t sentinel;
};

struct RegisteredAllReduceInputTable {
  const __nv_bfloat16* input[kRegisteredAllReduceRanks];
};

struct RegisteredAllReduceStateTable {
  RegisteredAllReduceStateRegion* state[kRegisteredAllReduceRanks];
};

// Enqueues exactly one payload-specific kernel and never synchronizes stream.
// The request lifecycle owns cross-call ordering and final stream quiescence.
// nRanks is 2 or 4. Four-rank 0.5/1 MiB payloads keep their dedicated
// two-stage kernels.
hipError_t launchRegisteredAllReduceKernel(
    void* output,
    RegisteredAllReduceInputTable inputs,
    RegisteredAllReduceStateTable states,
    int rank,
    size_t count,
    hipStream_t stream,
    int nRanks = kRegisteredAllReduceRanks);

// Reads 16 bytes at each of `count` addresses (a device array) into out[2i],
// out[2i + 1] with plain 16-byte loads, the way the kernels read peer inputs.
hipError_t launchRegisteredAllReduceProbe(
    const unsigned char* const* addresses,
    int count,
    uint64_t* out,
    hipStream_t stream);

} // namespace rcclx::relay
