// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <iostream>
#include <memory>

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>

#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/socket/CtranSocket.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"

commResult_t waitSocketReq(
    CtranSocketRequest& req,
    std::unique_ptr<CtranSocket>& ctranSock) {
  do {
    COMMCHECK_TEST(ctranSock->progress());
  } while (!req.isComplete());
  return commSuccess;
}

class CtranSocketTest : public ctran::CtranDistTestFixture {
 public:
  CtranSocketTest() = default;
  void SetUp() override {
    ctran::CtranDistTestFixture::SetUp();
    comm_ = makeCtranComm();
    comm = comm_.get();
    ctrlMgr = std::make_unique<CtranCtrlManager>();
  }

  void printTestDesc(const std::string& testName, const std::string& testDesc) {
    if (globalRank == 0) {
      std::cout << testName << " numRanks " << numRanks << "." << std::endl
                << testDesc << std::endl;
    }
  }

 protected:
  std::unique_ptr<CtranComm> comm_{nullptr};
  CtranComm* comm{nullptr};
  std::unique_ptr<CtranCtrlManager> ctrlMgr{nullptr};
};

TEST_F(CtranSocketTest, NormalInitialize) {
  printTestDesc(
      "NormalInitialize",
      "Expect CtranSocket to be initialized without internal error.");

  auto ctranSock = std::make_unique<CtranSocket>(comm, ctrlMgr.get());
  EXPECT_NE(ctranSock, nullptr);
}

TEST_F(CtranSocketTest, InitializeWithoutComm) {
  const std::string eth = "eth1";
  EnvRAII env1(NCCL_SOCKET_IFNAME, eth);
  printTestDesc(
      "InitializeWithoutComm",
      "Expect CtranSocket to be initialized without internal error.");

  const auto& rank = comm->statex_->rank();
  const auto& cudaDev = comm->statex_->cudaDev();
  const auto& commHash = comm->statex_->commHash();
  const auto& commDesc = comm->config_.commDesc;

  auto maybeAddr = ctran::bootstrap::getInterfaceAddress(
      NCCL_SOCKET_IFNAME, NCCL_SOCKET_IPADDR_PREFIX);
  ASSERT_FALSE(maybeAddr.hasError());

  // Create server socket, and bind it to reserve the port. The subsequent test
  // can use that port. The socket object will be destroyed (& port released)
  // when it goes out of scope.
  ctran::bootstrap::ServerSocket serverSocket(1);
  serverSocket.bind(
      folly::SocketAddress(*maybeAddr, 0), NCCL_SOCKET_IFNAME, true);
  int port = serverSocket.getListenAddress()->getPort();
  SocketServerAddr serverAddr{.port = port, .ipv6 = maybeAddr->str()};

  auto ctranSock = std::make_unique<CtranSocket>(
      rank, cudaDev, commHash, commDesc, ctrlMgr.get(), serverAddr);
  EXPECT_NE(ctranSock, nullptr);

  // test send/recv control message
  // i.e. bootstrap connect/accept with provided ip/port
  struct SocketServerAddrTmp {
    int port{-1};
    char ipv6[1024]{};
  };
  SocketServerAddrTmp serverAddrTmp{.port = serverAddr.port};
  strcpy(serverAddrTmp.ipv6, serverAddr.ipv6.c_str());
  std::vector<SocketServerAddrTmp> serverAddrs(numRanks);
  serverAddrs[globalRank] = serverAddrTmp;
  auto resFuture = comm->bootstrap_->allGather(
      serverAddrs.data(),
      sizeof(SocketServerAddrTmp),
      comm->statex_->rank(),
      comm->statex_->nRanks());
  COMMCHECK_TEST(static_cast<commResult_t>(std::move(resFuture).get()));

  ControlMsg smsg(ControlMsgType::IB_EXPORT_MEM);
  ControlMsg rmsg(ControlMsgType::IB_EXPORT_MEM);
  CtranSocketRequest req;

  smsg.ibExp.remoteAddr = 99;
  smsg.ibExp.rkeys[0] = 1;
  smsg.ibExp.nKeys = 1;
  SocketServerAddr remoteAddr;
  if (globalRank == 0) {
    remoteAddr.port = serverAddrs[1].port;
    remoteAddr.ipv6.assign(serverAddrs[1].ipv6);
    COMMCHECK_TEST(ctranSock->isendCtrlMsg(smsg, 1, remoteAddr, req));
  } else if (globalRank == 1) {
    remoteAddr.port = serverAddrs[0].port;
    remoteAddr.ipv6.assign(serverAddrs[0].ipv6);
    COMMCHECK_TEST(ctranSock->irecvCtrlMsg(rmsg, 0, remoteAddr, req));
  } else {
    // no-op for non-communicating ranks
    COMMCHECK_TEST(req.complete());
  }
  waitSocketReq(req, ctranSock);

  if (globalRank == 1) {
    EXPECT_EQ(rmsg.ibExp.rkeys[0], smsg.ibExp.rkeys[0]);
    EXPECT_EQ(rmsg.ibExp.remoteAddr, smsg.ibExp.remoteAddr);
  }
}

namespace {
constexpr int testRkey = 9;
constexpr uint64_t testRemoteAddr = 100;
bool testCbFlag = false;
commResult_t testCtrlMsgCb(int peer, void* msgPtr, void* ctx) {
  bool* testCbFlagPtr = reinterpret_cast<bool*>(ctx);
  *testCbFlagPtr = true;
  EXPECT_EQ(peer, 0);
  auto msg = reinterpret_cast<ControlMsg*>(msgPtr);
  EXPECT_EQ(msg->type, ControlMsgType::IB_EXPORT_MEM);
  EXPECT_EQ(msg->ibExp.rkeys[0], testRkey);
  EXPECT_EQ(msg->ibExp.remoteAddr, testRemoteAddr);
  return commSuccess;
}
} // namespace

TEST_F(CtranSocketTest, CbCtrlMsg) {
  this->printTestDesc(
      "CbCtrlMsg",
      "Expect rank 0 can issue a send control msg that triggers corresponding callback on rank 1");

  // Register callback
  this->ctrlMgr->regCb(
      ControlMsgType::IB_EXPORT_MEM, testCtrlMsgCb, &testCbFlag);

  auto ctranSock =
      std::make_unique<CtranSocket>(this->comm, this->ctrlMgr.get());
  CtranSocketRequest req;
  ControlMsg smsg(ControlMsgType::IB_EXPORT_MEM);

  smsg.ibExp.remoteAddr = testRemoteAddr;
  smsg.ibExp.rkeys[0] = testRkey;
  smsg.ibExp.nKeys = 1;
  if (this->globalRank == 0) {
    COMMCHECK_TEST(ctranSock->isendCtrlMsg(smsg, 1, req));

    // Wait until send finishes
    waitSocketReq(req, ctranSock);
  } else if (this->globalRank == 1) {
    // Wait until callback is triggered
    do {
      COMMCHECK_TEST(ctranSock->progress());
    } while (!testCbFlag);
  } else {
    // no-op for non-communicating ranks
    COMMCHECK_TEST(req.complete());
  }
}
TEST_F(CtranSocketTest, CtrlMsg) {
  printTestDesc(
      "SendRecvCtrlMsg",
      "Expect rank 2 can issue multiple send control msgs to ranks 0 and 1, and match to the corresponding recvs");

  auto ctranSock = std::make_unique<CtranSocket>(comm, ctrlMgr.get());
  std::vector<CtranSocketRequest> reqs;
  std::vector<ControlMsg> smsgs;
  ControlMsg rmsg0(ControlMsgType::IB_EXPORT_MEM);
  ControlMsg rmsg1(ControlMsgType::IB_EXPORT_MEM);

  if (numRanks < 3) {
    GTEST_SKIP() << "Need at least 3 ranks to run this test";
  }

  // Choose largest rank as sender to test bootstrap + pendingOps logic;
  // The larger one will be connected via ListenThread and has to put ctrlMsg
  // into pendingOps
  const int sendRank = 2, recvRank0 = 0, recvRank1 = 1;

  if (globalRank == sendRank) {
    reqs.resize(3, CtranSocketRequest());
    smsgs.resize(3, ControlMsg(ControlMsgType::IB_EXPORT_MEM));
    // send two msgs to rank 1
    smsgs[0].ibExp.remoteAddr = 99;
    smsgs[0].ibExp.rkeys[0] = recvRank0;
    smsgs[0].ibExp.nKeys = 1;
    COMMCHECK_TEST(ctranSock->isendCtrlMsg(smsgs[0], recvRank0, reqs[0]));

    // let recvRank0 connected via ListenThread first; thus the next
    // isendCtrlMsg shall be directly posted. Expect the two msgs are arrived
    // in order
    sleep(2);

    smsgs[1].ibExp.remoteAddr = 100;
    smsgs[1].ibExp.rkeys[0] = recvRank0;
    smsgs[1].ibExp.nKeys = 1;

    COMMCHECK_TEST(ctranSock->isendCtrlMsg(smsgs[1], recvRank0, reqs[1]));

    // send one msg to rank 2
    smsgs[2].ibExp.remoteAddr = 101;
    smsgs[2].ibExp.rkeys[0] = recvRank1;
    smsgs[2].ibExp.nKeys = 1;
    COMMCHECK_TEST(ctranSock->isendCtrlMsg(smsgs[2], recvRank1, reqs[2]));
  } else if (globalRank == recvRank0) {
    reqs.resize(2, CtranSocketRequest());
    sleep(1); // let sendRank put msgs into pendingOps first

    // receive two msgs from rank 0; assuming receive in order
    COMMCHECK_TEST(ctranSock->irecvCtrlMsg(rmsg0, sendRank, reqs[0]));
    COMMCHECK_TEST(ctranSock->irecvCtrlMsg(rmsg1, sendRank, reqs[1]));
  } else if (globalRank == recvRank1) {
    reqs.resize(1, CtranSocketRequest());
    // receive one msg from rank 0
    COMMCHECK_TEST(ctranSock->irecvCtrlMsg(rmsg0, sendRank, reqs[0]));
  }

  for (auto& req : reqs) {
    waitSocketReq(req, ctranSock);
  }

  if (globalRank == recvRank0) {
    EXPECT_EQ(rmsg0.ibExp.rkeys[0], recvRank0);
    EXPECT_EQ(rmsg0.ibExp.remoteAddr, 99);
    EXPECT_EQ(rmsg1.ibExp.rkeys[0], recvRank0);
    EXPECT_EQ(rmsg1.ibExp.remoteAddr, 100);
  } else if (globalRank == recvRank1) {
    EXPECT_EQ(rmsg0.ibExp.rkeys[0], recvRank1);
    EXPECT_EQ(rmsg0.ibExp.remoteAddr, 101);
  }
}

TEST_F(CtranSocketTest, AllGather) {
  printTestDesc(
      "AllGather",
      "Expect every rank to recv&send a control msg to all other ranks");
  auto ctranSock = std::make_unique<CtranSocket>(comm, ctrlMgr.get());
  std::vector<CtranSocketRequest> sreqs(numRanks);
  std::vector<CtranSocketRequest> rreqs(numRanks);
  std::vector<ControlMsg> smsgs(numRanks);
  std::vector<ControlMsg> rmsgs(numRanks);
  for (int i = 0; i < numRanks; i++) {
    if (i == globalRank) {
      continue;
    }
    // post recv request
    auto& rreq = rreqs[i];
    auto& rmsg = rmsgs[i];
    rmsg.setType(ControlMsgType::IB_EXPORT_MEM);
    rmsg.ibExp.remoteAddr = 0;
    rmsg.ibExp.rkeys[0] = 0;
    rmsg.ibExp.nKeys = 1;
    COMMCHECK_TEST(ctranSock->irecvCtrlMsg(rmsg, i, rreq));
    // post send request
    auto& sreq = sreqs[i];
    auto& smsg = smsgs[i];
    smsg.setType(ControlMsgType::IB_EXPORT_MEM);
    smsg.ibExp.remoteAddr = 99;
    smsg.ibExp.rkeys[0] = globalRank;
    smsg.ibExp.nKeys = 1;
    COMMCHECK_TEST(ctranSock->isendCtrlMsg(smsg, i, sreq));
  }
  for (int i = 0; i < numRanks; i++) {
    if (i == globalRank) {
      continue;
    }
    auto& rreq = rreqs[i];
    waitSocketReq(rreq, ctranSock);
    auto& sreq = sreqs[i];
    waitSocketReq(sreq, ctranSock);
  }
  for (int i = 0; i < numRanks; i++) {
    if (i == globalRank) {
      continue;
    }
    auto& rmsg = rmsgs[i];
    EXPECT_EQ(rmsg.ibExp.rkeys[0], i);
    EXPECT_EQ(rmsg.ibExp.remoteAddr, 99);
  }
}
TEST_F(CtranSocketTest, MatchAnyCtrlMsg) {
  printTestDesc(
      "MatchAnyCtrlMsg",
      "Expect rank 0 can issue a send control msg to rank 1 and matches to the UNSPECIFIED recv on rank1");

  auto ctranSock = std::make_unique<CtranSocket>(comm, ctrlMgr.get());
  const int nCtrl = 300; // exceed MAX_CONTROL_MSGS
  std::vector<CtranSocketRequest> reqs(nCtrl);
  std::vector<ControlMsg> smsgs(nCtrl);
  std::vector<ControlMsg> rmsgs(nCtrl);

  for (int i = 0; i < nCtrl; i++) {
    auto& smsg = smsgs[i];
    auto& rmsg = rmsgs[i];
    auto& req = reqs[i];
    smsg.setType(ControlMsgType::IB_EXPORT_MEM);
    smsg.ibExp.remoteAddr = 99;
    smsg.ibExp.rkeys[0] = i + 1;
    smsg.ibExp.nKeys = 1;
    rmsg.setType(ControlMsgType::UNSPECIFIED);
    rmsg.ibExp.remoteAddr = 0;
    rmsg.ibExp.rkeys[0] = 0;
    rmsg.ibExp.nKeys = 1;

    if (globalRank == 0) {
      COMMCHECK_TEST(ctranSock->isendCtrlMsg(smsg, 1, req));
    } else if (globalRank == 1) {
      COMMCHECK_TEST(ctranSock->irecvCtrlMsg(rmsg, 0, req));
    } else {
      // no-op for non-communicating ranks
      COMMCHECK_TEST(req.complete());
    }
  }
  for (int i = 0; i < nCtrl; i++) {
    waitSocketReq(reqs[i], ctranSock);

    if (globalRank == 1) {
      auto& rmsg = rmsgs[i];
      EXPECT_EQ(rmsg.type, ControlMsgType::IB_EXPORT_MEM);
      EXPECT_EQ(rmsg.ibExp.rkeys[0], i + 1);
      EXPECT_EQ(rmsg.ibExp.remoteAddr, 99);
    }
  }
}

TEST_F(CtranSocketTest, CtrlMsgAndPreConnect) {
  printTestDesc(
      "CtrlMsgAndPreConnect",
      "Expect rank 0 can issue a send control msg, followed by preConnect"
      "the preConnect is expected to be a no-op");

  auto ctranSock = std::make_unique<CtranSocket>(comm, ctrlMgr.get());
  CtranSocketRequest req;
  ControlMsg smsg(ControlMsgType::IB_EXPORT_MEM);
  ControlMsg rmsg(ControlMsgType::IB_EXPORT_MEM);
  constexpr int sendRank = 0, recvRank = 1;

  // pre-connect the peer
  std::unordered_set<int> peerRanks;
  if (globalRank == recvRank) {
    peerRanks.insert(sendRank);
    COMMCHECK_TEST(ctranSock->preConnect(peerRanks));
  } else if (globalRank == sendRank) {
    peerRanks.insert(recvRank);
    COMMCHECK_TEST(ctranSock->preConnect(peerRanks));
  }

  smsg.ibExp.remoteAddr = 99;
  smsg.ibExp.rkeys[0] = 1;
  smsg.ibExp.nKeys = 1;
  if (globalRank == sendRank) {
    COMMCHECK_TEST(ctranSock->isendCtrlMsg(smsg, recvRank, req));
  } else if (globalRank == recvRank) {
    COMMCHECK_TEST(ctranSock->irecvCtrlMsg(rmsg, sendRank, req));
  } else {
    // no-op for non-communicating ranks
    COMMCHECK_TEST(req.complete());
  }
  waitSocketReq(req, ctranSock);

  if (globalRank == recvRank) {
    EXPECT_EQ(rmsg.ibExp.rkeys[0], smsg.ibExp.rkeys[0]);
    EXPECT_EQ(rmsg.ibExp.remoteAddr, smsg.ibExp.remoteAddr);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
