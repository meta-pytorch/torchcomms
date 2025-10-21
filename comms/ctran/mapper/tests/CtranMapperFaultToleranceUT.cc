// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/tests/CtranStandaloneUTUtils.h"

namespace ctran::testing {

class CtranMapperFaultToleranceTest : public CtranStandaloneBaseTest {
 protected:
  void SetUp() override {
    setupBase();
    initCtranComm(::ctran::utils::createAbort(/*enabled=*/true));
  }
  void startTimer(int delaySeconds, std::function<void()> routine) {
    timer_ = std::thread([delaySeconds, routine = std::move(routine)]() {
      std::this_thread::sleep_for(std::chrono::seconds(delaySeconds));
      routine();
    });
  }

  void TearDown() override {
    CtranStandaloneBaseTest::TearDown();
    timer_.join();
  }

  std::thread timer_;

  void runTestWaitNotify(CtranMapperBackend backend);
};

TEST_F(CtranMapperFaultToleranceTest, WaitRequestNotCompleteAndAbort) {
  // pick any reqType to test not complete request and abort
  static constexpr auto reqType = CtranMapperRequest::ReqType::RECV_CTRL;

  auto mapper = std::make_unique<CtranMapper>(ctranComm.get());
  auto req = std::make_unique<CtranMapperRequest>(reqType, rank);

  startTimer(/*delaySeconds=*/2, [&]() {
    bool complete = true;
    EXPECT_EQ(commSuccess, mapper->testRequest(req.get(), &complete));
    EXPECT_FALSE(complete);
    ctranComm->setAbort();
  });
  EXPECT_THROW(mapper->waitRequest(req.get()), ::ctran::utils::Exception);
}

TEST_F(CtranMapperFaultToleranceTest, WaitNotifyNotCompleteAndAbortIB) {
  this->runTestWaitNotify(CtranMapperBackend::IB);
}
TEST_F(CtranMapperFaultToleranceTest, WaitNotifyNotCompleteAndAbortNVL) {
  this->runTestWaitNotify(CtranMapperBackend::NVL);
}

void CtranMapperFaultToleranceTest::runTestWaitNotify(
    CtranMapperBackend backend) {
  auto mapper = std::make_unique<CtranMapper>(ctranComm.get());

  auto kernElem = std::make_unique<KernelElem>();
  kernElem->ngroups = 1;
  kernElem->setStatus(KernelElem::ElemStatus::POSTED);

  auto notify = std::make_unique<CtranMapperNotify>();
  notify->peer = 0;
  notify->notifyCnt = 1;
  notify->backend = backend;
  notify->kernElem = kernElem.get();

  startTimer(/*delaySeconds=*/2, [&]() {
    bool complete = false;
    EXPECT_EQ(commSuccess, mapper->checkNotify(notify.get(), &complete));
    EXPECT_FALSE(complete);
    ctranComm->setAbort();
  });
  EXPECT_THROW(mapper->waitNotify(notify.get()), ::ctran::utils::Exception);
}

} // namespace ctran::testing
