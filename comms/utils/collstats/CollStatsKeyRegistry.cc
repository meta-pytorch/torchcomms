// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/CollStatsKeyRegistry.h"

namespace meta::comms::collstats {

uint32_t CollStatsKeyRegistry::resolve(const CollStatKey& key) {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = ids_.find(key);
  if (it != ids_.end()) {
    return it->second;
  }
  if (byId_.size() >= capacity_) {
    ++catchAllCount_;
    return capacity_;
  }
  const uint32_t id = static_cast<uint32_t>(byId_.size());
  byId_.push_back(key);
  ids_.emplace(key, id);
  return id;
}

uint32_t CollStatsKeyRegistry::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<uint32_t>(byId_.size());
}

std::vector<CollStatKey> CollStatsKeyRegistry::keys() const {
  std::lock_guard<std::mutex> lock(mu_);
  return byId_;
}

uint64_t CollStatsKeyRegistry::catchAllCount() const {
  std::lock_guard<std::mutex> lock(mu_);
  return catchAllCount_;
}

} // namespace meta::comms::collstats
