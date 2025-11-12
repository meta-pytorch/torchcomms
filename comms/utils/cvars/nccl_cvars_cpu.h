// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <string>
#include <vector>

#include <folly/String.h>

namespace ncclx::cvar::detail {

// Wraps a call to folly::split s.t. the call is compiled for the cpu and not
// for the gpu. Relevant since folly::split calls AVX instructions.
std::vector<std::string>
split(char delim, std::string_view in, const folly::SplitOptions& opts);

// Wraps a call to folly::split s.t. the call is compiled for the cpu and not
// for the gpu. Relevant since folly::split calls AVX instructions.
std::array<std::string, 2> split_noexact_2(char delim, std::string_view in);

} // namespace ncclx::cvar::detail
