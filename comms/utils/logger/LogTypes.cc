// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/LogTypes.h"

#include <atomic>

namespace meta::comms::logger {
namespace {

std::atomic<uint64_t>& getSubSystemMask() {
  static std::atomic<uint64_t> subSystemMask{0};
  return subSystemMask;
}

} // namespace

void setSubSystemMask(uint64_t subSystemMaskValue) {
  getSubSystemMask().store(subSystemMaskValue, std::memory_order_relaxed);
}

bool isEnabledSubSystemBitwise(uint64_t subSystem) {
  return (getSubSystemMask().load(std::memory_order_relaxed) & subSystem) != 0;
}

} // namespace meta::comms::logger
