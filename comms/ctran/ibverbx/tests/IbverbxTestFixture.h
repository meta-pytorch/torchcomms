// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <random>

#define IBVERBX_TEST_FRIENDS                                            \
  friend class IbverbxTestFixture;                                      \
  FRIEND_TEST(IbverbxTestFixture, IbvVirtualCqRegisterPhysicalQp);      \
  FRIEND_TEST(                                                          \
      IbverbxTestFixture, IbvVirtualCqRegisterPhysicalQpMoveSemantics); \
  FRIEND_TEST(                                                          \
      IbverbxTestFixture, IbvVirtualQpRegisterUnregisterWithVirtualCq); \
  FRIEND_TEST(IbverbxTestFixture, IbvVirtualQpUpdateWrState);           \
  FRIEND_TEST(IbverbxTestFixture, IbvVirtualQpBuildVirtualWc);          \
  FRIEND_TEST(IbverbxTestFixture, IbvVirtualQpIsSendOpcode);            \
  FRIEND_TEST(IbverbxTestFixture, IbvVirtualQpHasQpCapacity);           \
  FRIEND_TEST(IbverbxTestFixture, IbvVirtualQpBuildPhysicalSendWr);

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/utils/checks.h"

namespace ibverbx {

// use broadcom nic for AMD platform, use mellanox nic for NV platform
#if defined(__HIP_PLATFORM_AMD__) && !defined(USE_FE_NIC)
const std::string kNicPrefix("bnxt_re");
#else
const std::string kNicPrefix("mlx5_");
#endif

constexpr uint8_t kPortNum = 1;

#if defined(USE_FE_NIC)
constexpr int kGidIndex = 1;
#else
constexpr int kGidIndex = 3;
#endif

// helper functions
inline ibv_qp_init_attr makeIbvQpInitAttr(ibv_cq* cq) {
  ibv_qp_init_attr initAttr{};
  memset(&initAttr, 0, sizeof(ibv_qp_init_attr));
  initAttr.send_cq = cq;
  initAttr.recv_cq = cq;
  initAttr.qp_type = IBV_QPT_RC; // Reliable Connection
  initAttr.sq_sig_all = 0;
  initAttr.cap.max_send_wr = 256; // maximum outstanding send WRs
  initAttr.cap.max_recv_wr = 256; // maximum outstanding recv WRs
  initAttr.cap.max_send_sge = 1;
  initAttr.cap.max_recv_sge = 1;
  initAttr.cap.max_inline_data = 0;
  return initAttr;
}

class IbverbxTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(ibvInit());
  }
};

} // namespace ibverbx
