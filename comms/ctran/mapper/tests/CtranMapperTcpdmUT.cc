// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>

#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranTestUtils.h"

using ctran::CtranTcpDm;

class CtranMapperTcpdmTest : public ::testing::Test {
 public:
  std::unique_ptr<ctran::TestCtranCommRAII> commRAII_;
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
      auto commRAII = ctran::createDummyCtranComm();
      commRAII.reset();
    } catch (const ctran::utils::Exception&) {
      GTEST_SKIP() << "TCPDM backend not enabled. Skip test";
    }
  }
  void TearDown() override {
    unsetenv("NCCL_CTRAN_BACKENDS");
    if (timer_.joinable()) {
      timer_.join();
    }
    commRAII_.reset();
  }
  void createComm(bool abortEnabled = false) {
    ncclCvarInit();
    commRAII_ = ctran::createDummyCtranComm(0, abortEnabled);
    dummyComm_ = commRAII_->ctranComm.get();
  }
  void setAbortAfter(std::chrono::milliseconds delay) {
    timer_ = std::thread([this, delay]() {
      std::this_thread::sleep_for(delay);
      dummyComm_->setAbort();
    });
  }

  std::thread timer_;
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

TEST_F(CtranMapperTcpdmTest, WaitNotifyReturnsWhenCommAborts) {
  setenv("NCCL_CTRAN_BACKENDS", "tcpdm", 1);
  this->createComm(/*abortEnabled=*/true);

  auto mapper = std::make_unique<CtranMapper>(this->dummyComm_);
  CtranMapperNotify notify;
  notify.peer = 0;
  notify.backend = CtranMapperBackend::TCPDM;

  this->setAbortAfter(std::chrono::milliseconds(100));

  EXPECT_THROW(mapper->waitNotify(&notify), ::ctran::utils::Exception);
}

TEST_F(CtranMapperTcpdmTest, CheckNotifyReturnsWhenCommAborts) {
  setenv("NCCL_CTRAN_BACKENDS", "tcpdm", 1);
  this->createComm(/*abortEnabled=*/true);

  auto mapper = std::make_unique<CtranMapper>(this->dummyComm_);
  CtranMapperNotify notify;
  notify.peer = 0;
  notify.backend = CtranMapperBackend::TCPDM;
  bool done = false;

  this->dummyComm_->setAbort();

  EXPECT_THROW(mapper->checkNotify(&notify, &done), ::ctran::utils::Exception);
}

TEST_F(CtranMapperTcpdmTest, TestSomeRequestsReturnsWhenCommAborts) {
  setenv("NCCL_CTRAN_BACKENDS", "tcpdm", 1);
  this->createComm(/*abortEnabled=*/true);

  auto mapper = std::make_unique<CtranMapper>(this->dummyComm_);
  auto req = std::make_unique<CtranMapperRequest>();
  req->peer = 0;
  req->backend = CtranMapperBackend::TCPDM;
  std::vector<std::unique_ptr<CtranMapperRequest>> reqs;
  reqs.push_back(std::move(req));

  this->dummyComm_->setAbort();

  EXPECT_THROW(mapper->testSomeRequests(reqs), ::ctran::utils::Exception);
}
TEST_F(CtranMapperTcpdmTest, OverrideBackendThroughHints) {
  // Test that config_.backends overrides NCCL_CTRAN_BACKENDS CVAR.
  setenv("NCCL_CTRAN_BACKENDS", "nvl,ib,socket", 1);
  ASSERT_STREQ(getenv("NCCL_CTRAN_BACKENDS"), "nvl,ib,socket");
  this->createComm();
  // Directly set config_.backends to override CVAR-based backend selection
  this->dummyComm_->config_.backends = {CommBackend::TCPDM};
  auto mapper = std::make_unique<CtranMapper>(this->dummyComm_);
  auto rank = this->dummyComm_->statex_->rank();
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::IB));
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::NVL));
  EXPECT_FALSE(mapper->hasBackend(rank, CtranMapperBackend::SOCKET));
  EXPECT_TRUE(mapper->hasBackend(rank, CtranMapperBackend::TCPDM));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
