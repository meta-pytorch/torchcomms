// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "VerifyCommStateXUtil.h"

#include <unistd.h>
#include <set>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "comms/ctran/commstate/CommStateX.h"

namespace ctran::testing {

VerifyCommStateXHelper::RankIdentity
VerifyCommStateXHelper::RankIdentity::local() {
  RankIdentity id;
  gethostname(id.hostname, kMaxHostLen);
  id.pid = getpid();
  return id;
}

void VerifyCommStateXHelper::verifyHost(
    const ncclx::CommStateX* statex,
    int rank,
    const std::string& expectedHostname) {
  EXPECT_EQ(statex->host(rank), expectedHostname)
      << "rank " << rank << " host mismatch";
}

void VerifyCommStateXHelper::verifyGPid(
    const ncclx::CommStateX* statex,
    int rank,
    const std::string& expectedHostname,
    int expectedPid) {
  const std::string expectedGPid =
      fmt::format("{}:{}", expectedHostname, expectedPid);
  EXPECT_EQ(statex->gPid(rank), expectedGPid)
      << "rank " << rank << " gPid mismatch";
}

void VerifyCommStateXHelper::verifyGPidUniqueness(
    const ncclx::CommStateX* statex,
    int nRanks) {
  std::set<std::string> gPids;
  for (int r = 0; r < nRanks; r++) {
    auto [it, inserted] = gPids.insert(statex->gPid(r));
    EXPECT_TRUE(inserted) << "gPid collision at rank " << r << ": "
                          << statex->gPid(r);
  }
}

void VerifyCommStateXHelper::verifyAllHosts(
    const ncclx::CommStateX* statex,
    const std::vector<RankIdentity>& allRankIds) const {
  for (int r = 0; r < static_cast<int>(allRankIds.size()); r++) {
    verifyHost(statex, r, allRankIds[r].hostname);
  }
}

void VerifyCommStateXHelper::verifyAllGPids(
    const ncclx::CommStateX* statex,
    const std::vector<RankIdentity>& allRankIds) const {
  for (int r = 0; r < static_cast<int>(allRankIds.size()); r++) {
    verifyGPid(statex, r, allRankIds[r].hostname, allRankIds[r].pid);
  }
  verifyGPidUniqueness(statex, static_cast<int>(allRankIds.size()));
}

} // namespace ctran::testing
