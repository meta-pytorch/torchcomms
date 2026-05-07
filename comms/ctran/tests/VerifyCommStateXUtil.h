// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
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

  // Per-rank verification (no RankIdentity needed)
  static void verifyHost(
      const ncclx::CommStateX* statex,
      int rank,
      const std::string& expectedHostname);
  static void verifyGPid(
      const ncclx::CommStateX* statex,
      int rank,
      const std::string& expectedHostname,
      int expectedPid);
  static void verifyGPidUniqueness(const ncclx::CommStateX* statex, int nRanks);

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
