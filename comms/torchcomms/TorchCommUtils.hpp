// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

namespace torch {
namespace comms {

bool string_to_bool(const std::string& str);

// Convert environment variable to specified type, with default value if not set
template <typename T>
T env_to_value(const std::string& env_key, const T& default_value);

// Counts the number of lines in a file
int count_file_lines(
    const std::string& filepath,
    bool ignore_empty_lines = true);

std::pair<int, int> query_pals_ranksize();

// Query rank and size based on TORCHCOMM_BOOTSTRAP_RANKSIZE_QUERY_METHOD
std::pair<int, int> query_ranksize();

} // namespace comms
} // namespace torch
