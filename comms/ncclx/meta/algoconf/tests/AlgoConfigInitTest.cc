// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "meta/hints/GlobalHints.h" // @manual
#include "nccl.h"

// Skip initEnv()/ncclCvarInit() to verify global hints are inaccessible
// before NCCL initialization.
class AlgoConfigInitTest : public NcclxBaseTestFixture {
 protected:
  AlgoConfigInitTest() {
    initEnvAtSetup = false;
  }
};

TEST_F(AlgoConfigInitTest, SetHintBeforeCommCreation) {
  // Expect invalid access to AlgoConfig global hints before comm creation
  ASSERT_FALSE(ncclx::setGlobalHint("algo_sendrecv", "orig"));
  auto res = ncclx::getGlobalHint("algo_sendrecv");
  ASSERT_FALSE(res.has_value());

  ASSERT_FALSE(ncclx::resetGlobalHint("algo_sendrecv"));

  ncclComm_t comm = ncclx::test::createNcclComm(
      globalRank, numRanks, localRank, bootstrap_.get());

  // Expect valid access to AlgoConfig global hints after comm creation
  ASSERT_TRUE(ncclx::setGlobalHint("algo_sendrecv", "orig"));
  res = ncclx::getGlobalHint("algo_sendrecv");
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), "orig");
  ASSERT_TRUE(ncclx::resetGlobalHint("algo_sendrecv"));

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
