// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <string_view>

namespace ctran {

// Maps an algorithm display name (KernelConfig::algoName) to its NCCL_*_ALGO
// cvar token, e.g. "CtranAllReduceRing" -> "ctring". Unmapped names are
// returned unchanged.
std::string_view algoShortName(std::string_view displayName);

} // namespace ctran
