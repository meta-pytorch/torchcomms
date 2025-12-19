// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

namespace torch {
namespace comms {

bool string_to_bool(const std::string& str);

// Convert environment variable to specified type, with default value if not set
template <typename T>
T env_to_value(const std::string& env_key, const T& default_value);

// Query rank and size based on TORCHCOMM_BOOTSTRAP_RANKSIZE_QUERY_METHOD
std::pair<int, int> query_ranksize();

} // namespace comms
} // namespace torch
