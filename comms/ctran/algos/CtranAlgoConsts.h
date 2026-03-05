// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>

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

} // namespace ctran::allreduce::ring
