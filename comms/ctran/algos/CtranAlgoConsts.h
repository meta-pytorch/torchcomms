// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>

#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::allreduce::ring {

// Maximum BDP (bandwidth-delay product) per GPU architecture.
constexpr size_t kDefaultMaxBDP =
    128ULL * 1024 * 1024; // 128MB (GB200/Blackwell)
constexpr size_t kHopperMaxBDP = 32ULL * 1024 * 1024; // 32MB (H100)

// Per-arch BiDir AllGather thresholds (used by auto-tune, CVAR value -2).
// BiDir AG sends data in both directions during AllGather; beneficial for
// messages up to approximately the BDP where ring latency dominates.
constexpr size_t kDefaultBidirAgMaxSize =
    128ULL * 1024 * 1024; // 128MB (GB200/Blackwell)
constexpr size_t kHopperBidirAgMaxSize =
    4ULL * 1024 * 1024; // 4MB (H100/Hopper)

// Maximum BDP across all architectures — used for buffer pre-allocation
// when the arch is not yet known.
constexpr size_t kMaxBDP = kDefaultMaxBDP;

static_assert(kMaxBDP >= kDefaultMaxBDP);
static_assert(kMaxBDP >= kHopperMaxBDP);

// --- Staging Buffer Size helpers ---

constexpr size_t kStagingBufSize = 32ULL * 1024 * 1024;

// Returns the chunk override product (or 0 if CVARs not set).
inline size_t getStagingBufSizeFromChunkOverride() {
  if (NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE > 0 &&
      NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS > 0) {
    return static_cast<size_t>(NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE) *
        NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS;
  }
  return 0;
}

// Derives the physical staging buffer size (for allocation).
// Priority: (1) chunk override, (2) STAGING_BUF_SIZE CVAR, (3) kStagingBufSize.
inline size_t getStagingBufSize() {
  size_t result = getStagingBufSizeFromChunkOverride();
  if (result > 0) {
    return result;
  }
  if (NCCL_CTRAN_ALLREDUCE_RING_STAGING_BUF_SIZE > 0) {
    return static_cast<size_t>(NCCL_CTRAN_ALLREDUCE_RING_STAGING_BUF_SIZE);
  }
  return kStagingBufSize;
}

} // namespace ctran::allreduce::ring
