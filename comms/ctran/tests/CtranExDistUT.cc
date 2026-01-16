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

class CtranExTest : public CtranExBaseTest {
 protected:
  void SetUp() override {
    CtranExBaseTest::SetUp();
    // Reserved a port for CTRAN server
    std::string socketIfName;
    if (NCCL_SOCKET_IFNAME.empty()) {
      socketIfName = "";
    } else {
      socketIfName = NCCL_SOCKET_IFNAME[0];
    }
    serverSocket_.bind(folly::SocketAddress("::0", 0), socketIfName, true);
  }

  int getReservedPort() {
    return serverSocket_.getListenAddress()->getPort();
  }

  // Use CTRAN ServerSocket to get a free port
  ctran::bootstrap::ServerSocket serverSocket_{1};
  const std::string defaultDesc_{"CtranExTest"};
};

TEST_F(CtranExTest, Initialized) {
  std::unique_ptr<CtranEx> ctranEx = nullptr;

  const auto& ipv6 = getIPv6();
  if (ipv6.empty()) {
    GTEST_SKIP() << "CTRAN-IB: No socket interfaces found. Skip test";
  }

  const auto& hostname = getHostname();
  if (hostname.empty()) {
    GTEST_SKIP() << "CTRAN-IB: No hostname found. Skip test";
  }

  const CtranExHostInfo hostInfo = {
      .port = getReservedPort(),
      .ipv6 = ipv6,
      .hostname = hostname,
  };

  try {
    ctranEx = std::make_unique<CtranEx>(
        globalRank, localRank, hostInfo, defaultBackends_, defaultDesc_);
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "CTRAN-IB: IB backend not enabled. Skip test";
  }
  ASSERT_NE(ctranEx, nullptr);
  EXPECT_TRUE(ctranEx->isInitialized());
}

TEST_F(CtranExTest, InitializedWithIpv6) {
  std::unique_ptr<CtranEx> ctranEx = nullptr;

  const auto& ipv6 = getIPv6();
  if (ipv6.empty()) {
    GTEST_SKIP() << "CTRAN-IB: No socket interfaces found. Skip test";
  }

  const CtranExHostInfo hostInfo = {
      .port = getReservedPort(),
      .ipv6 = ipv6,
  };

  try {
    ctranEx = std::make_unique<CtranEx>(
        globalRank, localRank, hostInfo, defaultBackends_, defaultDesc_);
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "CTRAN-IB: IB backend not enabled. Skip test";
  }
  ASSERT_NE(ctranEx, nullptr);
  EXPECT_TRUE(ctranEx->isInitialized());
}

TEST_F(CtranExTest, InitializedWithInvalidPort) {
  std::unique_ptr<CtranEx> ctranEx = nullptr;

  const auto& ipv6 = getIPv6();
  if (ipv6.empty()) {
    GTEST_SKIP() << "CTRAN-IB: No socket interfaces found. Skip test";
  }

  const CtranExHostInfo hostInfo = {
      .port = -1,
      .ipv6 = ipv6,
  };

  try {
    ctranEx = std::make_unique<CtranEx>(
        globalRank, localRank, hostInfo, defaultBackends_, defaultDesc_);
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "CTRAN-IB: IB backend not enabled. Skip test";
  } catch (const std::runtime_error& e) {
    EXPECT_THAT(e.what(), ::testing::HasSubstr("Invalid port number -1"));
    ASSERT_EQ(ctranEx, nullptr);
    return;
  }

  GTEST_FAIL() << "Expected exception to be thrown, should not reach here";
}

TEST_F(CtranExTest, InitializedWithInvalidIPv6) {
  std::unique_ptr<CtranEx> ctranEx = nullptr;

  const CtranExHostInfo hostInfo = {
      .port = getReservedPort(),
      .ipv6 = "dummy",
  };

  try {
    ctranEx = std::make_unique<CtranEx>(
        globalRank, localRank, hostInfo, defaultBackends_, defaultDesc_);
  } catch (const std::bad_alloc& e) {
    GTEST_SKIP() << "CTRAN-IB: IB backend not enabled. Skip test";
  } catch (const std::runtime_error& e) {
    // FIXME: we don't have the proper error message when user gives an invalid
    // ipv6. Thus, skip checking error message for now. We need adjust the error
    // reporting first.
    ASSERT_EQ(ctranEx, nullptr);
    return;
  }

  GTEST_FAIL() << "Expected exception to be thrown, should not reach here";
}

// Skip failure path testing with invalid hostname, because socket seems happy
// with a dummy hostname.

TEST_F(CtranExTest, GpuMemReg) {
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  constexpr size_t kBufSize = 8192;
  void* regHdl = nullptr;
  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, kBufSize));
  ASSERT_EQ(ctranEx->regMem(buf, kBufSize, &regHdl), ncclSuccess);
  ASSERT_NE(regHdl, nullptr);

  ASSERT_EQ(ctranEx->deregMem(regHdl), ncclSuccess);
  CUDACHECK_TEST(cudaFree(buf));
}

TEST_F(CtranExTest, CpuMemReg) {
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  constexpr size_t kBufSize = 8192;
  void* regHdl = nullptr;
  void* buf = malloc(kBufSize);
  ASSERT_EQ(ctranEx->regMem(buf, kBufSize, &regHdl), ncclSuccess);
  ASSERT_NE(regHdl, nullptr);

  ASSERT_EQ(ctranEx->deregMem(regHdl), ncclSuccess);
  free(buf);
}

TEST_F(CtranExTest, SmallGpuMemReg) {
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  // By default, we support small buffer registation sizes
  constexpr size_t kBufSize = 2;
  void* regHdl = nullptr;
  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, kBufSize));
  ASSERT_EQ(ctranEx->regMem(buf, kBufSize, &regHdl), commSuccess);
  ASSERT_NE(regHdl, nullptr);

  ASSERT_EQ(ctranEx->deregMem(regHdl), commSuccess);
  CUDACHECK_TEST(cudaFree(buf));
}

TEST_F(CtranExTest, ConnectAndDataTransfer) {
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  constexpr size_t kBufSize = 8192;
  void* regHdl = nullptr;
  int* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, kBufSize));

  // Use first half as send buffer, and second half as recv buffer
  size_t recvOffset = kBufSize / sizeof(int) / 2;
  size_t sendBytes = kBufSize / 2;
  assignChunkValue(buf, kBufSize / sizeof(int) / 2, globalRank, 1);
  assignChunkValue(buf + recvOffset, kBufSize / sizeof(int) / 2, 0, 0);
  ASSERT_EQ(ctranEx->regMem(buf, kBufSize, &regHdl), ncclSuccess);
  ASSERT_NE(regHdl, nullptr);

  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  for (auto i = 0; i < numRanks; i++) {
    ASSERT_EQ(allHostInfos.at(i).ifName, hostInfo.ifName)
        << "Invalid ifname received from rank " << i << ", expect "
        << hostInfo.ifName << std::endl;
  }

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer.
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;
  CtranExRequest* req = nullptr;
  std::unique_ptr<CtranExRequest> sreq, rreq, preq, freq;

  // Send local buffer info to left peer to enable remote access from it
  ASSERT_EQ(
      ctranEx->isendCtrl(
          buf, kBufSize, regHdl, leftPeer, allHostInfos[leftPeer], &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  sreq = std::unique_ptr<CtranExRequest>(req);

  // Receive remote buffer info from right peer to enable remote access to it
  void* peerRemoteBuf = nullptr;
  uint32_t peerRemoteKey = 0;
  ASSERT_EQ(
      ctranEx->irecvCtrl(
          rightPeer,
          allHostInfos[rightPeer],
          &peerRemoteBuf,
          &peerRemoteKey,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  rreq = std::unique_ptr<CtranExRequest>(req);

  // Wait receive ctrl msg to finish and check the received buffer info
  ASSERT_EQ(rreq->wait(), ncclSuccess);
  EXPECT_NE(peerRemoteBuf, nullptr);
  EXPECT_NE(peerRemoteKey, 0);

  // Send data to right peer
  ASSERT_EQ(
      ctranEx->iput(
          buf,
          sendBytes,
          regHdl,
          rightPeer,
          reinterpret_cast<int*>(peerRemoteBuf) + recvOffset,
          peerRemoteKey,
          true /* notify */,
          &req),
      ncclSuccess);
  ASSERT_NE(req, nullptr);
  preq = std::unique_ptr<CtranExRequest>(req);

  // Wait left peer to transfer data to my buf, and check the received data
  ASSERT_EQ(ctranEx->waitNotify(leftPeer), ncclSuccess);

  // Perform PCI flush before checking data
  ASSERT_EQ(ctranEx->iflush(buf, regHdl, &req), ncclSuccess);
  ASSERT_NE(req, nullptr);
  freq = std::unique_ptr<CtranExRequest>(req);
  ASSERT_EQ(freq->wait(), ncclSuccess);

  auto errs =
      checkChunkValue(buf + recvOffset, sendBytes / sizeof(int), leftPeer, 1);
  EXPECT_EQ(errs, 0) << "Found received data from leftPeer " << leftPeer
                     << " is not correct at globalRank " << globalRank << ", "
                     << errs << " errors";

  // Wait send ctrl msg and put to finish and release resource
  ASSERT_EQ(sreq->wait(), ncclSuccess);
  ASSERT_EQ(preq->wait(), ncclSuccess);

  ASSERT_EQ(ctranEx->deregMem(regHdl), ncclSuccess);
  CUDACHECK_TEST(cudaFree(buf));
}

TEST_F(CtranExTest, CtrlSync) {
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer.
  // The first send/recv will do QP connection
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;

  // Use first exchange to connect to remote peers
  exchangeHostInfo(hostInfo, allHostInfos);

  const int kNumSync = 10;
  std::vector<std::unique_ptr<CtranExRequest>> sreqs(kNumSync);
  std::vector<std::unique_ptr<CtranExRequest>> rreqs(kNumSync);
  CtranExRequest *sreq = nullptr, *rreq = nullptr;

  for (int i = 0; i < kNumSync; i++) {
    if (i == 0) {
      ASSERT_EQ(
          ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq),
          ncclSuccess);
      ASSERT_EQ(
          ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
          ncclSuccess);
    } else {
      ASSERT_EQ(ctranEx->isendCtrl(leftPeer, &sreq), ncclSuccess);
      ASSERT_EQ(ctranEx->irecvCtrl(rightPeer, &rreq), ncclSuccess);
    }
    ASSERT_NE(sreq, nullptr);
    sreqs[i] = std::unique_ptr<CtranExRequest>(sreq);
    ASSERT_NE(rreq, nullptr);
    rreqs[i] = std::unique_ptr<CtranExRequest>(rreq);

    // Randomly wait on some recvs to ensure it doesn't cause hang or any
    // mismatch
    if (i % 3 == 0) {
      ASSERT_EQ(rreqs[i]->wait(), ncclSuccess);
      rreqs[i].reset();
    }
  }

  for (int i = 0; i < kNumSync; i++) {
    ASSERT_EQ(sreqs[i]->wait(), ncclSuccess);
    if (rreqs[i]) {
      // Some rreqs have already been released
      ASSERT_EQ(rreqs[i]->wait(), ncclSuccess);
    }
  }
}

TEST_F(CtranExTest, InvalidPeer) {
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  constexpr size_t kBufSize = 8192;
  void* regHdl = nullptr;
  void* buf = malloc(kBufSize);
  ASSERT_EQ(ctranEx->regMem(buf, kBufSize, &regHdl), ncclSuccess);
  ASSERT_NE(regHdl, nullptr);

  CtranExRequest* req = nullptr;
  ASSERT_EQ(ctranEx->isendCtrl(globalRank, hostInfo, &req), ncclInvalidUsage);

  ASSERT_EQ(ctranEx->irecvCtrl(globalRank, hostInfo, &req), ncclInvalidUsage);

  ASSERT_EQ(
      ctranEx->isendCtrl(buf, kBufSize, regHdl, globalRank, hostInfo, &req),
      ncclInvalidUsage);

  void* peerRemoteBuf = nullptr;
  uint32_t peerRemoteKey = 0;
  ASSERT_EQ(
      ctranEx->irecvCtrl(
          globalRank, hostInfo, &peerRemoteBuf, &peerRemoteKey, &req),
      ncclInvalidUsage);

  ASSERT_EQ(
      ctranEx->iput(
          buf,
          kBufSize,
          regHdl,
          globalRank,
          buf,
          peerRemoteKey,
          true /* notify */,
          &req),
      ncclInvalidUsage);

  ASSERT_EQ(ctranEx->waitNotify(globalRank), ncclInvalidUsage);
  bool done = false;
  ASSERT_EQ(ctranEx->checkNotify(globalRank, done), ncclInvalidUsage);

  ASSERT_EQ(ctranEx->deregMem(regHdl), ncclSuccess);
  free(buf);
}

TEST_F(CtranExTest, ReconfigAndShrink) {
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::unique_ptr<CtranEx> ctranEx = nullptr;
  createCtranEx(hostInfo, ctranEx);
  ASSERT_NE(ctranEx, nullptr);

  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);
  int rightPeer = (globalRank + 1) % numRanks;
  int leftPeer = (globalRank - 1 + numRanks) % numRanks;

  // Exchange ctrl msg as a Ring where local rank sends data to right peer, and
  // receives data from left peer. The first send/recv will do QP connection
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

  // Kill rank 3 and all rest ranks reset remote states
  if (globalRank != 3) {
    ASSERT_EQ(ctranEx->releaseRemoteTransStates(), ncclSuccess);

    // Recreate cq
    ASSERT_EQ(ctranEx->initRemoteTransStates(), ncclSuccess);
  }

  barrier(); // barrier to make sure all ranks have reset ctranEx and are
             // ready to reconfig
             // Note: we only have global barrier in UT so kill rank 3 after
             // this barrier, but rank 3 does nothing to CtranEx

  if (globalRank == 3) {
    // Rank 3 drops out of the ring
    return;
  }

  // update left/right peer in the shrinked ring
  int shrinkedSize = numRanks - 1;
  rightPeer = (globalRank + 1) % shrinkedSize;
  leftPeer = (globalRank - 1 + shrinkedSize) % shrinkedSize;

  // Exchange ctrl msg in the shrinked ring which will do qp connection
  ASSERT_EQ(
      ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq), ncclSuccess);
  ASSERT_EQ(
      ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
      ncclSuccess);
  ASSERT_NE(sreq, nullptr);
  ASSERT_NE(rreq, nullptr);
  ASSERT_EQ(sreq->wait(), ncclSuccess);
  ASSERT_EQ(rreq->wait(), ncclSuccess);

  // Rank 0 try connecting to rank 3: expect a ncclRemoteError since rank 3 has
  // dead
  if (globalRank == 0) {
    ASSERT_NE(ctranEx->isendCtrl(3, allHostInfos[3], &sreq), ncclSuccess);
  }
}

TEST_F(CtranExTest, ReconfigAndGrow) {
  const CtranExHostInfo hostInfo = getDefaultHostInfo();
  std::vector<CtranExHostInfo> allHostInfos;
  exchangeHostInfo(hostInfo, allHostInfos);

  std::unique_ptr<CtranEx> ctranEx = nullptr;
  int leftPeer = -1, rightPeer = -1;
  CtranExRequest *sreq = nullptr, *rreq = nullptr;

  // Assume rank 3 is offline at the beginning.
  // We will create CtranEx on rank 0, 1, 2
  if (globalRank != 3) {
    createCtranEx(hostInfo, ctranEx);
    ASSERT_NE(ctranEx, nullptr);

    // Exchange ctrl msg as a Ring where local rank sends data to right peer,
    // and receives data from left peer. The first send/recv will do QP
    // connection
    int startSize = numRanks - 1;
    rightPeer = (globalRank + 1) % startSize;
    leftPeer = (globalRank - 1 + startSize) % startSize;
    ASSERT_EQ(
        ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq),
        ncclSuccess);
    ASSERT_EQ(
        ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
        ncclSuccess);
    ASSERT_NE(sreq, nullptr);
    ASSERT_NE(rreq, nullptr);
    ASSERT_EQ(sreq->wait(), ncclSuccess);
    ASSERT_EQ(rreq->wait(), ncclSuccess);

    ASSERT_EQ(ctranEx->releaseRemoteTransStates(), ncclSuccess);
  }

  // Make sure all ranks reach here before rank 3 becomes online
  barrier();

  // Rank 3 is now back online, and we need to create a new CtranEx for it
  // Rank 0, 1, 2 will re-init CtranEx
  if (globalRank == 3) {
    createCtranEx(hostInfo, ctranEx);
    ASSERT_NE(ctranEx, nullptr);
  } else {
    ASSERT_EQ(ctranEx->initRemoteTransStates(), ncclSuccess);
  }

  // do a barrier to make sure all ranks have reset ctranEx and are ready
  // to reconfig
  barrier();

  // update left/right peer in the growed ring
  rightPeer = (globalRank + 1) % numRanks;
  leftPeer = (globalRank - 1 + numRanks) % numRanks;
  sreq = nullptr;
  rreq = nullptr;

  // Exchange ctrl msg in the shrinked ring which will do qp connection
  ASSERT_EQ(
      ctranEx->isendCtrl(leftPeer, allHostInfos[leftPeer], &sreq), ncclSuccess);
  ASSERT_EQ(
      ctranEx->irecvCtrl(rightPeer, allHostInfos[rightPeer], &rreq),
      ncclSuccess);
  ASSERT_NE(sreq, nullptr);
  ASSERT_NE(rreq, nullptr);
  ASSERT_EQ(sreq->wait(), ncclSuccess);
  ASSERT_EQ(rreq->wait(), ncclSuccess);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
