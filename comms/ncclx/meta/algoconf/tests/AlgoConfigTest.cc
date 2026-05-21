// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <gtest/gtest.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual
#include "meta/algoconf/AlgoConfig.h" // @manual
#include "meta/hints/GlobalHints.h" // @manual

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
  std::unordered_map<enum NCCL_SENDRECV_ALGO, const char*> overrideAlgos = {
      {NCCL_SENDRECV_ALGO::ctran, "ctran"},
      {NCCL_SENDRECV_ALGO::ctzcopy, "ctzcopy"},
      {NCCL_SENDRECV_ALGO::ctp2p, "ctp2p"},
  };

  const int iter = 10;
  for (int i = 0; i < iter; i++) {
    for (auto& [algo, hintVal] : overrideAlgos) {
      ASSERT_TRUE(ncclx::setGlobalHint(hintName, hintVal));

      auto getHintVal = ncclx::getGlobalHint(hintName);
      ASSERT_TRUE(getHintVal.has_value());
      ASSERT_EQ(getHintVal.value(), hintVal);

      ASSERT_EQ(getSendRecvAlgo(), algo);

      ASSERT_TRUE(ncclx::resetGlobalHint(hintName));
      getHintVal = ncclx::getGlobalHint(hintName);
      ASSERT_FALSE(getHintVal.has_value());

      ASSERT_EQ(getSendRecvAlgo(), NCCL_SENDRECV_ALGO::orig);
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
