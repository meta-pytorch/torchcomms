// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <string>

namespace uniflow::benchmark {

struct BootstrapConfig {
  std::string masterAddr;
  int32_t masterPort{29500};
  int32_t rank{0};
  int32_t worldSize{1};
  int32_t localRank{0};

  bool isRank0() const {
    return rank == 0;
  }

  /// Reads MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK.
  /// Throws if any required var is missing or invalid.
  static BootstrapConfig fromEnv();
};

} // namespace uniflow::benchmark
