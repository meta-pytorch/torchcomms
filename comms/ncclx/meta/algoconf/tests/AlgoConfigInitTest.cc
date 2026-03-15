// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/hints/GlobalHints.h" // @manual
#include "nccl.h"

TEST(AlgoConfigInitTest, SetHintBeforeCommCreation) {
  // Expect invalid access to AlgoConfig global hints before comm creation
  ASSERT_FALSE(ncclx::setGlobalHint("algo_sendrecv", "orig"));
  auto res = ncclx::getGlobalHint("algo_sendrecv");
  ASSERT_FALSE(res.has_value());

  ASSERT_FALSE(ncclx::resetGlobalHint("algo_sendrecv"));

  auto [localRank, globalRank, numRanks, localSize] = getTcpStoreOrMpiInfo();

  ASSERT_EQ(cudaSetDevice(localRank), cudaSuccess)
      << "cudaSetDevice failed with device: " << localRank;

  ncclComm_t comm __attribute__((unused)) =
      createNcclComm(globalRank, numRanks, localRank);

  // Expect valid access to AlgoConfig global hints after comm creation
  ASSERT_TRUE(ncclx::setGlobalHint("algo_sendrecv", "orig"));
  res = ncclx::getGlobalHint("algo_sendrecv");
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), "orig");
  ASSERT_TRUE(ncclx::resetGlobalHint("algo_sendrecv"));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
