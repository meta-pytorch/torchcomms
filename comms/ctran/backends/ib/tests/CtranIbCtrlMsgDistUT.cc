// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"

using namespace ctran::ibvwrap;

commResult_t waitIbReq(CtranIbRequest& req, std::unique_ptr<CtranIb>& ctranIb) {
  do {
    COMMCHECK_TEST(ctranIb->progress());
  } while (!req.isComplete());
  return commSuccess;
}

class CtranIbCtrlMsgTest : public ctran::CtranDistTestFixture {
 public:
  CtranIbCtrlMsgTest() = default;
  void SetUp() override {
    CtranDistTestFixture::SetUp();
    this->comm_ = makeCtranComm();
    this->comm = this->comm_.get();
    this->ctrlMgr = std::make_unique<CtranCtrlManager>();
  }

  void TearDown() override {
    this->ctrlMgr.reset();
    this->comm_.reset();
    CtranDistTestFixture::TearDown();
  }

  void printTestDesc(const std::string& testName, const std::string& testDesc) {
    XLOG_IF(WARN, this->globalRank == 0)
        << testName << " numRanks " << this->numRanks
        << ". Description: " << testDesc << std::endl;
  }

 protected:
  std::unique_ptr<CtranComm> comm_{nullptr};
  CtranComm* comm{nullptr};
  std::unique_ptr<CtranCtrlManager> ctrlMgr{nullptr};
};

TEST_F(CtranIbCtrlMsgTest, CtrlMsg) {
  this->printTestDesc(
      "SendRecvCtrlMsg",
      "Expect rank 2 can issue multiple send control msgs to ranks 0 and 1, and match to the corresponding recvs");

  try {
    auto ctranIb = std::make_unique<CtranIb>(this->comm, this->ctrlMgr.get());
    std::vector<CtranIbRequest> reqs;
    std::vector<ControlMsg> smsgs;
    ControlMsg rmsg0(ControlMsgType::IB_EXPORT_MEM);
    ControlMsg rmsg1(ControlMsgType::IB_EXPORT_MEM);

    if (this->numRanks < 3) {
      GTEST_SKIP() << "Need at least 3 ranks to run this test";
    }

    CtranIbEpochRAII epochRAII(ctranIb.get());
    // Choose largest rank as sender to test bootstrap + pendingOps logic;
    // The larger one will be connected via ListenThread and has to put
    // ctrlMsg into pendingOps
    const int sendRank = 2, recvRank0 = 0, recvRank1 = 1;

    if (this->globalRank == sendRank) {
      reqs.resize(3, CtranIbRequest());
      smsgs.resize(3, ControlMsg(ControlMsgType::IB_EXPORT_MEM));
      // send two msgs to rank 1
      smsgs[0].ibExp.remoteAddr = 99;
      smsgs[0].ibExp.rkeys[0] = recvRank0;
      smsgs[0].ibExp.rkeys[1] = recvRank0;
      smsgs[0].ibExp.nKeys = 2;
      COMMCHECK_TEST(ctranIb->isendCtrlMsg(
          smsgs[0].type, &smsgs[0], sizeof(smsgs[0]), recvRank0, reqs[0]));

      // let recvRank0 connected via ListenThread first; thus the next
      // isendCtrlMsg shall be directly posted. Expect the two msgs are
      // arrived in order
      sleep(2);

      smsgs[1].ibExp.remoteAddr = 100;
      smsgs[1].ibExp.rkeys[0] = recvRank0;
      smsgs[1].ibExp.rkeys[1] = recvRank0;
      smsgs[1].ibExp.nKeys = 2;

      COMMCHECK_TEST(ctranIb->isendCtrlMsg(
          smsgs[1].type, &smsgs[1], sizeof(smsgs[1]), recvRank0, reqs[1]));

      // send one msg to rank 2
      smsgs[2].ibExp.remoteAddr = 101;
      smsgs[2].ibExp.rkeys[0] = recvRank1;
      smsgs[2].ibExp.rkeys[1] = recvRank1;
      smsgs[2].ibExp.nKeys = 2;
      COMMCHECK_TEST(ctranIb->isendCtrlMsg(
          smsgs[2].type, &smsgs[2], sizeof(smsgs[2]), recvRank1, reqs[2]));
    } else if (this->globalRank == recvRank0) {
      reqs.resize(2, CtranIbRequest());
      sleep(1); // let sendRank put msgs into pendingOps first

      // receive two msgs from rank 0; assuming receive in order
      COMMCHECK_TEST(
          ctranIb->irecvCtrlMsg(&rmsg0, sizeof(rmsg0), sendRank, reqs[0]));
      COMMCHECK_TEST(
          ctranIb->irecvCtrlMsg(&rmsg1, sizeof(rmsg1), sendRank, reqs[1]));
    } else if (this->globalRank == recvRank1) {
      reqs.resize(1, CtranIbRequest());

      // receive one msg from rank 0
      COMMCHECK_TEST(
          ctranIb->irecvCtrlMsg(&rmsg0, sizeof(rmsg0), sendRank, reqs[0]));
    }

    for (auto& req : reqs) {
      waitIbReq(req, ctranIb);
    }

    if (this->globalRank == recvRank0) {
      EXPECT_EQ(rmsg0.ibExp.rkeys[0], recvRank0);
      EXPECT_EQ(rmsg0.ibExp.rkeys[1], recvRank0);
      EXPECT_EQ(rmsg0.ibExp.nKeys, 2);
      EXPECT_EQ(rmsg0.ibExp.remoteAddr, 99);
      EXPECT_EQ(rmsg1.ibExp.rkeys[0], recvRank0);
      EXPECT_EQ(rmsg1.ibExp.rkeys[1], recvRank0);
      EXPECT_EQ(rmsg1.ibExp.nKeys, 2);
      EXPECT_EQ(rmsg1.ibExp.remoteAddr, 100);
    } else if (this->globalRank == recvRank1) {
      EXPECT_EQ(rmsg0.ibExp.rkeys[0], recvRank1);
      EXPECT_EQ(rmsg0.ibExp.rkeys[1], recvRank1);
      EXPECT_EQ(rmsg0.ibExp.nKeys, 2);
      EXPECT_EQ(rmsg0.ibExp.remoteAddr, 101);
    }
  } catch (const std::bad_alloc& _) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
