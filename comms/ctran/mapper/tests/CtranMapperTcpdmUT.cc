// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
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
  void TearDown() override {
    if (timer_.joinable()) {
      timer_.join();
    }
    commRAII_.reset();
    unsetenv("NCCL_CTRAN_BACKENDS");
    unsetenv("TCP_DEVMEM_SKIP_AGENT");
    unsetenv("TCP_DEVMEM_RECONFIGURE_DEVICES");
    ncclCvarInit();
  }
  std::optional<std::string> tryCreateComm(bool abortEnabled = false) {
    setenv("TCP_DEVMEM_SKIP_AGENT", "1", 1);
    setenv("TCP_DEVMEM_RECONFIGURE_DEVICES", "0", 1);
    try {
      ncclCvarInit();
      commRAII_ = ctran::createDummyCtranComm(0, abortEnabled);
      dummyComm_ = commRAII_->ctranComm.get();
      return std::nullopt;
    } catch (const ctran::utils::Exception& e) {
      return e.what();
    }
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
  ncclCvarInit();
  EXPECT_THAT(
      CtranMapper::getToEnableBackends({}),
      ::testing::ElementsAre(CtranMapperBackend::TCPDM));
}

TEST_F(CtranMapperTcpdmTest, WaitNotifyReturnsWhenCommAborts) {
  setenv("NCCL_CTRAN_BACKENDS", "tcpdm", 1);
  if (auto error = this->tryCreateComm(/*abortEnabled=*/true)) {
    GTEST_SKIP() << "TCPDM communicator unavailable: " << *error;
  }

  auto mapper = std::make_unique<CtranMapper>(this->dummyComm_);
  CtranMapperNotify notify;
  notify.peer = 0;
  notify.backend = CtranMapperBackend::TCPDM;

  this->setAbortAfter(std::chrono::milliseconds(100));

  EXPECT_THROW(mapper->waitNotify(&notify), ::ctran::utils::Exception);
}

TEST_F(CtranMapperTcpdmTest, CheckNotifyReturnsWhenCommAborts) {
  setenv("NCCL_CTRAN_BACKENDS", "tcpdm", 1);
  if (auto error = this->tryCreateComm(/*abortEnabled=*/true)) {
    GTEST_SKIP() << "TCPDM communicator unavailable: " << *error;
  }

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
  if (auto error = this->tryCreateComm(/*abortEnabled=*/true)) {
    GTEST_SKIP() << "TCPDM communicator unavailable: " << *error;
  }

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
  setenv("NCCL_CTRAN_BACKENDS", "nvl,ib,socket", 1);
  ASSERT_STREQ(getenv("NCCL_CTRAN_BACKENDS"), "nvl,ib,socket");
  ncclCvarInit();
  EXPECT_THAT(
      CtranMapper::getToEnableBackends({}),
      ::testing::UnorderedElementsAre(
          CtranMapperBackend::NVL,
          CtranMapperBackend::IB,
          CtranMapperBackend::SOCKET));
  EXPECT_THAT(
      CtranMapper::getToEnableBackends({CommBackend::TCPDM}),
      ::testing::ElementsAre(CtranMapperBackend::TCPDM));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
