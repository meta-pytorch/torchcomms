// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>

namespace ctran::allreduce::ring {

// Maximum BDP (bandwidth-delay product) per GPU architecture.
constexpr size_t kDefaultMaxBDP =
    128ULL * 1024 * 1024; // 128MB (GB200/Blackwell)
constexpr size_t kHopperMaxBDP = 32ULL * 1024 * 1024; // 32MB (H100)

// Maximum BDP across all architectures — used for buffer pre-allocation
// when the arch is not yet known.
constexpr size_t kMaxBDP = kDefaultMaxBDP;

static_assert(kMaxBDP >= kDefaultMaxBDP);
static_assert(kMaxBDP >= kHopperMaxBDP);

} // namespace ctran::allreduce::ring
