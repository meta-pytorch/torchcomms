// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>

namespace meta::comms::logger {

enum class LogLevel {
  NONE = 0,
  VERSION = 1,
  ERROR = 2,
  WARN = 3,
  INFO = 4,
  ABORT = 5,
  TRACE = 6
};

#pragma push_macro("NET")
#pragma push_macro("INIT")
#undef NET
#undef INIT
enum SubSystem {
  INIT = 0x1,
  COLL = 0x2,
  P2P = 0x4,
  SHM = 0x8,
  NET = 0x10,
  GRAPH = 0x20,
  TUNING = 0x40,
  ENV = 0x80,
  ALLOC = 0x100,
  CALL = 0x200,
  PROXY = 0x400,
  NVLS = 0x800,
  BOOTSTRAP = 0x1000,
  REG = 0x2000,
  PROFILE = 0x4000,
  RAS = 0x8000,
  ALL = ~0
};
#pragma pop_macro("INIT")
#pragma pop_macro("NET")

void setSubSystemMask(uint64_t subSystemMask);

bool isEnabledSubSystemBitwise(uint64_t subSystem);

} // namespace meta::comms::logger
