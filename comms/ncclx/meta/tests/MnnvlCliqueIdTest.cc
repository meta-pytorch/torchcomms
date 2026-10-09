// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <string>

#include <gtest/gtest.h>

#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/MnnvlCliqueId.h"

namespace {

class MnnvlCliqueIdTest : public ::testing::Test {
 protected:
  // cvars are zero until ncclCvarInit(), and the helper treats any
  // NCCL_MNNVL_CLIQUE_ID other than -1 as a conflict.
  EnvRAII<int64_t> cliqueId_{NCCL_MNNVL_CLIQUE_ID, -1};
  EnvRAII<int> cliqueSize_{NCCL_MNNVL_CLIQUE_SIZE, 4};
  SysEnvRAII worldSize_{"WORLD_SIZE", "16"};
};

TEST_F(MnnvlCliqueIdTest, GlobalRankDividedByCliqueSize) {
  for (int rank : {0, 3, 4, 15}) {
    SysEnvRAII rankEnv("RANK", std::to_string(rank));
    int cliqueId = -1;
    ASSERT_EQ(
        ncclx::assignMnnvlCliqueIdBasedOnCliqueSize(&cliqueId), ncclSuccess);
    EXPECT_EQ(cliqueId, rank / 4);
  }
}

TEST_F(MnnvlCliqueIdTest, FatalOnNonPositiveCliqueSize) {
  SysEnvRAII rankEnv("RANK", "0");
  EnvRAII<int> cliqueSize(NCCL_MNNVL_CLIQUE_SIZE, 0);
  int cliqueId = -1;
  EXPECT_DEATH(
      ncclx::assignMnnvlCliqueIdBasedOnCliqueSize(&cliqueId),
      "must be positive");
}

TEST_F(MnnvlCliqueIdTest, FatalWhenCliqueIdAlsoSet) {
  SysEnvRAII rankEnv("RANK", "0");
  EnvRAII<int64_t> cliqueIdEnv(NCCL_MNNVL_CLIQUE_ID, 2);
  int cliqueId = -1;
  EXPECT_DEATH(
      ncclx::assignMnnvlCliqueIdBasedOnCliqueSize(&cliqueId),
      "can NOT be set at the same time");
}

TEST_F(MnnvlCliqueIdTest, FatalWhenRankUnset) {
  int cliqueId = -1;
  EXPECT_DEATH(
      {
        unsetenv("RANK");
        ncclx::assignMnnvlCliqueIdBasedOnCliqueSize(&cliqueId);
      },
      "RANK is not set");
}

TEST_F(MnnvlCliqueIdTest, FatalWhenWorldSizeUnset) {
  SysEnvRAII rankEnv("RANK", "0");
  int cliqueId = -1;
  EXPECT_DEATH(
      {
        unsetenv("WORLD_SIZE");
        ncclx::assignMnnvlCliqueIdBasedOnCliqueSize(&cliqueId);
      },
      "WORLD_SIZE is not set");
}

TEST_F(MnnvlCliqueIdTest, FatalWhenWorldSizeNotMultiple) {
  SysEnvRAII rankEnv("RANK", "0");
  SysEnvRAII worldSize("WORLD_SIZE", "18");
  int cliqueId = -1;
  EXPECT_DEATH(
      ncclx::assignMnnvlCliqueIdBasedOnCliqueSize(&cliqueId), "not a multiple");
}

} // namespace
