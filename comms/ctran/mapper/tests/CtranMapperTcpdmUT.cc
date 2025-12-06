// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperImpl.h"
#include "comms/ctran/mapper/CtranMapperRegMem.h"
#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/logger/LogUtils.h"

using ctran::CtranTcpDm;

class CtranMapperTcpdmTest : public ::testing::Test {
 public:
  std::unique_ptr<TestCtranCommRAII> commRAII_;
  CtranComm* dummyComm_{nullptr};
  std::unique_ptr<CtranMapper> mapper;

 protected:
  void SetUp() override {
    setenv("TCP_DEVMEM_SKIP_AGENT", "1", 1);
    setenv("TCP_DEVMEM_RECONFIGURE_DEVICES", "0", 1);
    // TCPDM only available with certain kernel version. Skip test if with
    // incompatible kernel
    try {
      setenv("NCCL_CTRAN_BACKENDS", "tcpdm", 1);
      ncclCvarInit();
      auto commRAII = createDummyCtranComm();
      commRAII.reset();
    } catch (const std::runtime_error& e) {
      GTEST_SKIP() << "TCPDM backend not enabled. Skip test";
    } catch (const ctran::utils::Exception& e) {
      GTEST_SKIP() << "TCPDM backend not enabled. Skip test";
    }
  }
  void TearDown() override {
    unsetenv("NCCL_CTRAN_BACKENDS");
    unsetenv("MCCL_CTRAN_BACKENDS");
    commRAII_.reset();
  }
  void createComm() {
    ncclCvarInit();
    commRAII_ = createDummyCtranComm();
    dummyComm_ = commRAII_->ctranComm;
  }
};

TEST_F(CtranMapperTcpdmTest, EnableTCPDMBackendThroughCVARs) {
  setenv("NCCL_CTRAN_BACKENDS", "tcpdm", 1);
  ASSERT_STREQ(getenv("NCCL_CTRAN_BACKENDS"), "tcpdm");
  this->createComm();
  auto mapper = std::make_unique<CtranMapper>(this->dummyComm_);
  auto rank = this->dummyComm_->statex_->rank();
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::IB));
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::NVL));
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::SOCKET));
  EXPECT_TRUE(mapper->hasBackend(rank, CtranMapperBackend::TCPDM));
}
TEST_F(CtranMapperTcpdmTest, OverrideBackendThroughHints) {
  setenv("NCCL_CTRAN_BACKENDS", "nvl,ib,socket", 1);
  ASSERT_STREQ(getenv("NCCL_CTRAN_BACKENDS"), "nvl,ib,socket");
  setenv("MCCL_CTRAN_BACKENDS", "tcpdm", 1);
  ASSERT_STREQ(getenv("MCCL_CTRAN_BACKENDS"), "tcpdm");
  this->createComm();
  auto mapper = std::make_unique<CtranMapper>(this->dummyComm_);
  auto rank = this->dummyComm_->statex_->rank();
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::IB));
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::NVL));
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::SOCKET));
  EXPECT_TRUE(mapper->hasBackend(rank, CtranMapperBackend::TCPDM));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  return RUN_ALL_TESTS();
}
