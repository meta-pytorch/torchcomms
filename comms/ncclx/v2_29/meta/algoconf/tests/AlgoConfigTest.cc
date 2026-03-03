// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <gtest/gtest.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual
#include "meta/algoconf/AlgoConfig.h" // @manual
#include "meta/hints/GlobalHints.h" // @manual

using ncclx::algoconf::getAllGatherAlgo;
using ncclx::algoconf::getAllReduceAlgo;
using ncclx::algoconf::getAllToAllVAlgo;
using ncclx::algoconf::getRmaAlgo;
using ncclx::algoconf::getSendRecvAlgo;
using ncclx::algoconf::testOnlyResetAlgoConfig;

class AlgoConfigUT : public ::testing::Test {
 public:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }
};

TEST_F(AlgoConfigUT, AlgoDefaultCtran) {
  setenv("NCCL_SENDRECV_ALGO", "ctran", 1);
  ncclx::testOnlyResetGlobalHints();
  testOnlyResetAlgoConfig();

  ASSERT_EQ(getSendRecvAlgo(), NCCL_SENDRECV_ALGO::ctran);
}

TEST_F(AlgoConfigUT, AlgoDefaultOrig) {
  setenv("NCCL_SENDRECV_ALGO", "orig", 1);
  ncclx::testOnlyResetGlobalHints();
  testOnlyResetAlgoConfig();

  ASSERT_EQ(getSendRecvAlgo(), NCCL_SENDRECV_ALGO::orig);
}

TEST_F(AlgoConfigUT, SendRecvAlgoHintOverride) {
  setenv("NCCL_SENDRECV_ALGO", "orig", 1);
  ncclx::testOnlyResetGlobalHints();
  testOnlyResetAlgoConfig();

  const char* hintName = "algo_sendrecv";
  ASSERT_EQ(getSendRecvAlgo(), NCCL_SENDRECV_ALGO::orig);

  // set/reset multiple times
  const int iter = 10;
  for (int i = 0; i < iter; i++) {
    ASSERT_TRUE(ncclx::setGlobalHint(hintName, "ctran"));

    auto getHintVal = ncclx::getGlobalHint(hintName);
    ASSERT_TRUE(getHintVal.has_value());
    ASSERT_EQ(getHintVal.value(), "ctran");

    ASSERT_EQ(getSendRecvAlgo(), NCCL_SENDRECV_ALGO::ctran);

    ASSERT_TRUE(ncclx::resetGlobalHint(hintName));
    getHintVal = ncclx::getGlobalHint(hintName);
    ASSERT_FALSE(getHintVal.has_value());

    ASSERT_EQ(getSendRecvAlgo(), NCCL_SENDRECV_ALGO::orig);
  }
}

TEST_F(AlgoConfigUT, AllGatherAlgoHintOverride) {
  setenv("NCCL_ALLGATHER_ALGO", "orig", 1);
  CLOGF(WARN, "before testOnlyResetGlobalHints");
  ncclx::testOnlyResetGlobalHints();
  CLOGF(WARN, "after testOnlyResetGlobalHints");
  testOnlyResetAlgoConfig();
  CLOGF(WARN, "after testOnlyResetAlgoConfig");

  const char* hintName = "algo_allgather";
  ASSERT_EQ(getAllGatherAlgo(), NCCL_ALLGATHER_ALGO::orig);
  CLOGF(WARN, "after getAllGatherAlgo");

  // set/reset multiple times
  std::unordered_map<enum NCCL_ALLGATHER_ALGO, const char*> overrideAlgos = {
      {NCCL_ALLGATHER_ALGO::ctran, "ctran"},
      {NCCL_ALLGATHER_ALGO::ctring, "ctring"},
      {NCCL_ALLGATHER_ALGO::ctrd, "ctrd"},
      {NCCL_ALLGATHER_ALGO::ctbrucks, "ctbrucks"},
      {NCCL_ALLGATHER_ALGO::ctdirect, "ctdirect"},
  };

  const int iter = 10;
  for (int i = 0; i < iter; i++) {
    for (auto& [algo, hintVal] : overrideAlgos) {
      ASSERT_TRUE(ncclx::setGlobalHint(hintName, hintVal));

      auto getHintVal = ncclx::getGlobalHint(hintName);
      ASSERT_TRUE(getHintVal.has_value());
      ASSERT_EQ(getHintVal.value(), hintVal);

      ASSERT_EQ(getAllGatherAlgo(), algo);

      ASSERT_TRUE(ncclx::resetGlobalHint(hintName));
      getHintVal = ncclx::getGlobalHint(hintName);
      ASSERT_FALSE(getHintVal.has_value());

      ASSERT_EQ(getAllGatherAlgo(), NCCL_ALLGATHER_ALGO::orig);
    }
  }
}

TEST_F(AlgoConfigUT, AllReduceAlgoHintOverride) {
  setenv("NCCL_ALLREDUCE_ALGO", "orig", 1);
  ncclx::testOnlyResetGlobalHints();
  testOnlyResetAlgoConfig();

  const char* hintName = "algo_allreduce";
  ASSERT_EQ(getAllReduceAlgo(), NCCL_ALLREDUCE_ALGO::orig);

  std::unordered_map<enum NCCL_ALLREDUCE_ALGO, const char*> overrideAlgos = {
      {NCCL_ALLREDUCE_ALGO::ctran, "ctran"},
      {NCCL_ALLREDUCE_ALGO::ctdirect, "ctdirect"},
      {NCCL_ALLREDUCE_ALGO::ctring, "ctring"},
  };

  const int iter = 10;
  for (int i = 0; i < iter; i++) {
    for (auto& [algo, hintVal] : overrideAlgos) {
      ASSERT_TRUE(ncclx::setGlobalHint(hintName, hintVal));

      auto getHintVal = ncclx::getGlobalHint(hintName);
      ASSERT_TRUE(getHintVal.has_value());
      ASSERT_EQ(getHintVal.value(), hintVal);

      ASSERT_EQ(getAllReduceAlgo(), algo);

      ASSERT_TRUE(ncclx::resetGlobalHint(hintName));
      getHintVal = ncclx::getGlobalHint(hintName);
      ASSERT_FALSE(getHintVal.has_value());

      ASSERT_EQ(getAllReduceAlgo(), NCCL_ALLREDUCE_ALGO::orig);
    }
  }
}

TEST_F(AlgoConfigUT, AlltoAllVAlgoHintOverride) {
  setenv("NCCL_ALLTOALLV_ALGO", "orig", 1);
  CLOGF(WARN, "before testOnlyResetGlobalHints");
  ncclx::testOnlyResetGlobalHints();
  CLOGF(WARN, "after testOnlyResetGlobalHints");
  testOnlyResetAlgoConfig();
  CLOGF(WARN, "after testOnlyResetAlgoConfig");

  const char* hintName = "algo_alltoallv";
  ASSERT_EQ(getAllToAllVAlgo(), NCCL_ALLTOALLV_ALGO::orig);
  CLOGF(WARN, "after getAllToAllVAlgo");

  // set/reset multiple times
  std::unordered_map<enum NCCL_ALLTOALLV_ALGO, const char*> overrideAlgos = {
      {NCCL_ALLTOALLV_ALGO::ctran, "ctran"},
      {NCCL_ALLTOALLV_ALGO::compCtran, "compCtran"},
      {NCCL_ALLTOALLV_ALGO::bsCompCtran, "bsCompCtran"},
  };

  const int iter = 10;
  for (int i = 0; i < iter; i++) {
    for (auto& [algo, hintVal] : overrideAlgos) {
      ASSERT_TRUE(ncclx::setGlobalHint(hintName, hintVal));

      auto getHintVal = ncclx::getGlobalHint(hintName);
      ASSERT_TRUE(getHintVal.has_value());
      ASSERT_EQ(getHintVal.value(), hintVal);

      ASSERT_EQ(getAllToAllVAlgo(), algo);

      ASSERT_TRUE(ncclx::resetGlobalHint(hintName));
      getHintVal = ncclx::getGlobalHint(hintName);
      ASSERT_FALSE(getHintVal.has_value());

      ASSERT_EQ(getAllToAllVAlgo(), NCCL_ALLTOALLV_ALGO::orig);
    }
  }
}

TEST_F(AlgoConfigUT, InvalidAlgoHint) {
  setenv("NCCL_SENDRECV_ALGO", "orig", 1);
  ncclx::testOnlyResetGlobalHints();
  testOnlyResetAlgoConfig();

  ASSERT_EQ(getSendRecvAlgo(), NCCL_SENDRECV_ALGO::orig);

  // reset with invalid algo hint name will return error
  ASSERT_FALSE(ncclx::resetGlobalHint("algo_dummy_hint_name"));

  // set with invalid algo value is ignored
  ASSERT_TRUE(ncclx::setGlobalHint("algo_sendrecv", "dummy_val"));
  ASSERT_EQ(getSendRecvAlgo(), NCCL_SENDRECV_ALGO::orig);

  // can query the value even if it is invalid for AlgoConfig
  auto getHintVal = ncclx::getGlobalHint("algo_sendrecv");
  ASSERT_TRUE(getHintVal.has_value());
  ASSERT_EQ(getHintVal.value(), "dummy_val");
}

TEST_F(AlgoConfigUT, RmaAlgoHintOverride) {
  setenv("NCCL_RMA_ALGO", "ctran", 1);
  ncclx::testOnlyResetGlobalHints();
  testOnlyResetAlgoConfig();

  const char* hintName = "algo_rma";
  ASSERT_EQ(getRmaAlgo(), NCCL_RMA_ALGO::ctran);

  // set/reset multiple times
  std::unordered_map<enum NCCL_RMA_ALGO, const char*> overrideAlgos = {
      {NCCL_RMA_ALGO::orig, "orig"},
  };

  const int iter = 10;
  for (int i = 0; i < iter; i++) {
    for (auto& [algo, hintVal] : overrideAlgos) {
      ASSERT_TRUE(ncclx::setGlobalHint(hintName, hintVal));

      auto getHintVal = ncclx::getGlobalHint(hintName);
      ASSERT_TRUE(getHintVal.has_value());
      ASSERT_EQ(getHintVal.value(), hintVal);

      ASSERT_EQ(getRmaAlgo(), algo);

      ASSERT_TRUE(ncclx::resetGlobalHint(hintName));
      getHintVal = ncclx::getGlobalHint(hintName);
      ASSERT_FALSE(getHintVal.has_value());

      ASSERT_EQ(getRmaAlgo(), NCCL_RMA_ALGO::ctran);
    }
  }
}
