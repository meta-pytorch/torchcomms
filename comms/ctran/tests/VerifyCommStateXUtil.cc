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

void VerifyCommStateXHelper::verifyAllHosts(
    const ncclx::CommStateX* statex,
    const std::vector<RankIdentity>& allRankIds) const {
  const int numRanks = static_cast<int>(allRankIds.size());
  for (int r = 0; r < numRanks; r++) {
    EXPECT_EQ(statex->host(r), std::string(allRankIds[r].hostname))
        << "rank " << r << " host mismatch";
  }
}

void VerifyCommStateXHelper::verifyAllGPids(
    const ncclx::CommStateX* statex,
    const std::vector<RankIdentity>& allRankIds) const {
  const int numRanks = static_cast<int>(allRankIds.size());
  std::set<std::string> gPids;
  for (int r = 0; r < numRanks; r++) {
    const std::string expectedGPid =
        fmt::format("{}:{}", allRankIds[r].hostname, allRankIds[r].pid);
    EXPECT_EQ(statex->gPid(r), expectedGPid)
        << "rank " << r << " gPid mismatch";

    auto [it, inserted] = gPids.insert(statex->gPid(r));
    EXPECT_TRUE(inserted) << "gPid collision at rank " << r << ": "
                          << statex->gPid(r);
  }
}

} // namespace ctran::testing
