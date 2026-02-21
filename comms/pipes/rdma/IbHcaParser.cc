// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/rdma/IbHcaParser.h"

#include <algorithm>
#include <stdexcept>

namespace comms::pipes {

void IbHcaParser::parse_prefix(const std::string& hcaStr, size_t& pos) {
  // Check ^= before ^ before = to handle combined prefix correctly
  if (hcaStr.compare(pos, 2, "^=") == 0) {
    matchMode_ = HcaMatchMode::EXACT_EXCLUDE;
    pos += 2;
  } else if (hcaStr[pos] == '^') {
    matchMode_ = HcaMatchMode::PREFIX_EXCLUDE;
    pos += 1;
  } else if (hcaStr[pos] == '=') {
    matchMode_ = HcaMatchMode::EXACT_INCLUDE;
    pos += 1;
  } else {
    matchMode_ = HcaMatchMode::PREFIX_INCLUDE;
  }
}

HcaEntry IbHcaParser::parse_entry(const std::string& token) const {
  HcaEntry entry;
  auto colonPos = token.find(':');
  if (colonPos != std::string::npos) {
    entry.name = token.substr(0, colonPos);
    entry.port = std::stoi(token.substr(colonPos + 1));
  } else {
    entry.name = token;
    entry.port = -1;
  }
  return entry;
}

IbHcaParser::IbHcaParser(const std::string& hcaStr) {
  if (hcaStr.empty()) {
    return;
  }

  // Parse mode prefix
  size_t pos = 0;
  parse_prefix(hcaStr, pos);

  // Split by ',' and parse each entry
  std::string remaining = hcaStr.substr(pos);
  size_t start = 0;
  while (start < remaining.size()) {
    auto commaPos = remaining.find(',', start);
    std::string token;
    if (commaPos != std::string::npos) {
      token = remaining.substr(start, commaPos - start);
      start = commaPos + 1;
    } else {
      token = remaining.substr(start);
      start = remaining.size();
    }

    // Trim whitespace
    auto trimStart = token.find_first_not_of(" \t");
    auto trimEnd = token.find_last_not_of(" \t");
    if (trimStart == std::string::npos) {
      continue; // Skip empty tokens
    }
    token = token.substr(trimStart, trimEnd - trimStart + 1);

    entries_.push_back(parse_entry(token));
  }

  if (entries_.size() > kMaxHcaDevices) {
    throw std::runtime_error(
        "Too many IB HCA entries (" + std::to_string(entries_.size()) +
        "), maximum is " + std::to_string(kMaxHcaDevices));
  }
}

bool IbHcaParser::matches(const std::string& devName, int port) const {
  // Empty entries means no filter - match everything
  if (entries_.empty()) {
    return true;
  }

  bool matchFound = false;
  for (const auto& entry : entries_) {
    bool nameMatch = false;
    if (matchMode_ == HcaMatchMode::PREFIX_INCLUDE ||
        matchMode_ == HcaMatchMode::PREFIX_EXCLUDE) {
      nameMatch = devName.compare(0, entry.name.size(), entry.name) == 0;
    } else {
      // EXACT modes
      nameMatch = (devName == entry.name);
    }

    if (nameMatch) {
      // If both entry port and query port are specified, they must match
      if (entry.port >= 0 && port >= 0 && entry.port != port) {
        continue;
      }
      matchFound = true;
      break;
    }
  }

  if (matchMode_ == HcaMatchMode::PREFIX_INCLUDE ||
      matchMode_ == HcaMatchMode::EXACT_INCLUDE) {
    return matchFound;
  }
  // EXCLUDE modes: return the inverse
  return !matchFound;
}

} // namespace comms::pipes
