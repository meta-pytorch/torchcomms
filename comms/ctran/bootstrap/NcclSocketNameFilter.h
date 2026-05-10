// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <vector>

namespace ctran::bootstrap {

struct IfNameFilter {
  std::vector<std::string> entries;
  bool negate{false};
  bool exactMatch{false};
};

// Parse NCCL_SOCKET_IFNAME spec string into an IfNameFilter.
// Handles ^, =, and comma-separated lists.
// TODO: Have callers pass IfNameFilter directly to getInterfaceAddress instead
// of parsing inside it. The current coupling to NCCL_SOCKET_IFNAME format makes
// it fragile when input ifname strings vary. Deferred because it impacts mccl
// callsites too.
IfNameFilter parseIfNameSpec(const std::string& spec);

// Check if a system interface name matches the filter.
// Empty filter (no entries) matches everything.
bool matchesIfName(const IfNameFilter& filter, const char* ifaName);

} // namespace ctran::bootstrap
