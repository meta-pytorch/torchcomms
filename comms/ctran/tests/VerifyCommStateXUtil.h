// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <vector>

namespace ncclx {
class CommStateX;
} // namespace ncclx

namespace ctran::testing {

class VerifyCommStateXHelper {
 public:
  struct RankIdentity {
    static constexpr int kMaxHostLen = 256;
    char hostname[kMaxHostLen]{};
    int pid{-1};

    RankIdentity() = default;

    // Factory: populate with current process's hostname and pid
    static RankIdentity local();
  };

  // Verify hostname matches for all ranks
  void verifyAllHosts(
      const ncclx::CommStateX* statex,
      const std::vector<RankIdentity>& allRankIds) const;

  // Verify gPid is correct and unique for all ranks
  void verifyAllGPids(
      const ncclx::CommStateX* statex,
      const std::vector<RankIdentity>& allRankIds) const;
};

} // namespace ctran::testing
