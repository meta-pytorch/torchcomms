// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <unordered_map>

namespace ncclx {

// getNcclxInfo is defined in nccl.h

// Used for testing only. Do not use in production.
std::unordered_map<std::string, std::string> testOnlyGatherNcclxInfo();

} // namespace ncclx
