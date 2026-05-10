// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "NcclSocketNameFilter.h"

#include <cstring>

namespace ctran::bootstrap {

IfNameFilter parseIfNameSpec(const std::string& spec) {
  IfNameFilter filter;
  if (spec.empty()) {
    return filter;
  }

  const char* ptr = spec.c_str();

  // Check for ^ (negate) prefix
  if (*ptr == '^') {
    filter.negate = true;
    ptr++;
  }

  // Check for = (exact match) prefix
  if (*ptr == '=') {
    filter.exactMatch = true;
    ptr++;
  }

  // Split remaining string by commas
  std::string remaining(ptr);
  size_t start = 0;
  while (start < remaining.size()) {
    size_t comma = remaining.find(',', start);
    std::string token;
    if (comma == std::string::npos) {
      token = remaining.substr(start);
      start = remaining.size();
    } else {
      token = remaining.substr(start, comma - start);
      start = comma + 1;
    }
    if (!token.empty()) {
      filter.entries.push_back(std::move(token));
    }
  }

  return filter;
}

bool matchesIfName(const IfNameFilter& filter, const char* ifaName) {
  // Empty filter matches everything
  if (filter.entries.empty()) {
    return true;
  }

  bool matched = false;
  for (const auto& entry : filter.entries) {
    if (filter.exactMatch) {
      if (std::strcmp(ifaName, entry.c_str()) == 0) {
        matched = true;
        break;
      }
    } else {
      if (std::strncmp(ifaName, entry.c_str(), entry.size()) == 0) {
        matched = true;
        break;
      }
    }
  }

  return filter.negate ? !matched : matched;
}

} // namespace ctran::bootstrap
