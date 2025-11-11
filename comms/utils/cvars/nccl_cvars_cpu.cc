// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/cvars/nccl_cvars_cpu.h"

namespace ncclx::cvar::detail {

std::vector<std::string>
split(char delim, std::string_view in, const folly::SplitOptions& opts) {
  std::vector<std::string> ret;
  folly::split(delim, in, ret, opts);
  return ret;
}

std::array<std::string, 2> split_noexact_2(char delim, std::string_view in) {
  std::array<std::string, 2> ret;
  folly::split<false>(delim, in, ret[0], ret[1]);
  return ret;
}

} // namespace ncclx::cvar::detail
