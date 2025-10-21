// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <ifaddrs.h>
#include <nccl.h>
#include <net/if.h>
#include <stdlib.h>
#include "comms/ctran/CtranEx.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

using namespace ctran;

class CtranExBethTest : public CtranExBaseTest {
  void SetUp() override {
    const std::string beth{"beth4"};
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_CLIENT_SOCKET_IFNAME", beth.c_str(), 1);
    setenv("NCCL_SOCKET_IFNAME", beth.c_str(), 1);

    // Ensure quick failure
    setenv("NCCL_SOCKET_RETRY_CNT", "2", 1);
    setenv("NCCL_SOCKET_RETRY_SLEEP_MSEC", "1", 1);
    if (!checkIfNameExist(beth)) {
      GTEST_SKIP() << fmt::format("Skip: No BE NIC ({}) exists", beth);
    }

    // create ncclComm_ with beth4
    CtranExBaseTest::SetUp();
  }

 protected:
  const std::string defaultDesc_{"CtranExBethTest"};
};

TEST_F(CtranExBethTest, InitializedWithClientIfName) {
  // CtranEx will use the specified ifName to connect to remote peers.
  // fastinit w/ BE NCCL_CLIENT_SOCKET_IFNAME=eth4 won't affect ctranEx w/ FE
  const std::string exIfname = "eth0";
  if (!checkIfNameExist(exIfname)) {
    GTEST_SKIP() << fmt::format("Skip: No NIC ({}) exists", exIfname);
  }

  std::string ipv6;
  {
    // Temporarily overwrite NCCL_SOCKET_IFNAME to get its ipv6
    EnvRAII socketIfname(NCCL_SOCKET_IFNAME, exIfname);
    ipv6 = getIPv6();
    if (ipv6.empty()) {
      GTEST_SKIP() << "CTRAN-IB: No socket interfaces found. Skip test";
    }
  }

  // Use CTRAN ServerSocket to get a free port
  ctran::bootstrap::ServerSocket serverSocket(1);
  serverSocket.bind(folly::SocketAddress("::0", 0), NCCL_SOCKET_IFNAME, true);

  // Use different ifName to create CtranEx
  std::unique_ptr<CtranEx> ctranEx = nullptr;
  const CtranExHostInfo hostInfo = {
      .port = serverSocket.getListenAddress()->getPort(),
      .ipv6 = ipv6,
      .ifName = exIfname,
  };
  createCtranEx(hostInfo, ctranEx);
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  for (auto i = 0; i < numRanks; i++) {
    ASSERT_EQ(allHostInfos.at(i).ifName, exIfname)
        << "Invalid ifname received from rank " << i << ", expect " << exIfname
        << std::endl;
  }

  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;

  std::cout << "Rank " << globalRank << " connectiong to rightPeer "
            << rightPeer << " using " << allHostInfos[rightPeer].toString()
            << std::endl;
  std::cout << "Rank " << globalRank << " connectiong to leftPeer " << leftPeer
            << " using " << allHostInfos[leftPeer].toString() << std::endl;

  // QP connection with FE
  CtranExRequest *sreq = nullptr, *rreq = nullptr;
  ASSERT_EQ(
      ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq), ncclSuccess);
  ASSERT_EQ(
      ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
      ncclSuccess);
  ASSERT_NE(sreq, nullptr);
  ASSERT_NE(rreq, nullptr);
  ASSERT_EQ(sreq->wait(), ncclSuccess);
  ASSERT_EQ(rreq->wait(), ncclSuccess);

  // NCCL Comm fastinit with BE
  ncclComm_t comm_be = createNcclComm(globalRank, numRanks, localRank);
  ASSERT_EQ(comm_be->initState, ncclSuccess);
  ASSERT_EQ(ncclCommDestroy(comm_be), ncclSuccess);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
