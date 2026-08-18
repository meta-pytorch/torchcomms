// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/LogTypes.h"

namespace meta::comms::logger {
namespace {

uint64_t& getSubSystemMask() {
  static uint64_t subSystemMask = 0;
  return subSystemMask;
}

} // namespace

void setSubSystemMask(uint64_t subSystemMaskValue) {
  getSubSystemMask() = subSystemMaskValue;
}

bool isEnabledSubSystemBitwise(uint64_t subSystem) {
  return getSubSystemMask() & subSystem;
}

} // namespace meta::comms::logger
