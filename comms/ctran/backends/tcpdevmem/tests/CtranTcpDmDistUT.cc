// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdlib.h>
#include <chrono>
#include <cstddef>
#include <memory>

#include <gtest/gtest.h>

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/tcpdevmem/CtranTcpDm.h"
#include "comms/ctran/backends/tcpdevmem/CtranTcpDmSingleton.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

using ctran::CtranTcpDm;
using ctran::CtranTcpDmRequest;
using ctran::CtranTcpDmSingleton;

commResult_t waitTcpReq(
    CtranTcpDmRequest& req,
    std::unique_ptr<CtranTcpDm>& ctranTcpDm) {
  while (!req.isComplete()) {
    ctranTcpDm->progress();
  }
  return commSuccess;
}

class CtranTcpTest : public NcclxBaseTest {
 public:
  CtranTcpTest() = default;
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    // Use TCP Devmem plugin in regular TCP mode until we have proper
    // kernel and FW installed on the hosts.
    setenv("TCP_DEVMEM_ENABLE", "0", 0);
    NcclxBaseTest::SetUp();
    ncclCvarInit(); // initialize cvars explicitly to take effect
    this->commDeprecated = createNcclComm(
        globalRank, numRanks, localRank, false, nullptr, server.get());
    this->comm = this->commDeprecated->ctranComm_.get();

    this->ctrlMgr = std::make_unique<CtranCtrlManager>();

    try {
      this->ctranTcpDm =
          std::make_unique<CtranTcpDm>(this->comm, this->ctrlMgr.get());
    } catch (const ctran::utils::Exception& e) {
      GTEST_SKIP() << "TCPDM backend not enabled. Skip test";
    } catch (const std::runtime_error& e) {
      GTEST_SKIP() << "TCPDM backend not enabled. Skip test";
    }
  }

  void TearDown() override {
    finalizeNcclComm(globalRank, server.get());
    NCCLCHECK_TEST(ncclCommDestroy(this->commDeprecated));
    this->ctrlMgr.reset();
    this->ctranTcpDm.reset();
    NcclxBaseTest::TearDown();
  }

  void printTestDesc(const std::string& testName, const std::string& testDesc) {
    // NOTE: Printing it as WARN to make this log visible as our default setting
    // is to only print WARN and above logs.
    CLOGF_IF(
        WARN,
        this->globalRank == 0,
        "{} numRanks {}. Description: {}",
        testName,
        this->numRanks,
        testDesc);
  }

 protected:
  CtranComm* comm{nullptr};
  std::unique_ptr<CtranTcpDm> ctranTcpDm{nullptr};
  // TODO: remove this once refatoring is finished
  // !!! DO NOT USE !!!
  ncclComm_t commDeprecated{nullptr};
  std::unique_ptr<CtranCtrlManager> ctrlMgr{nullptr};
  const int sendRank{0}, recvRank{1};
};

TEST_F(CtranTcpTest, RegMrHost) {
  this->printTestDesc(
      "RegMr", "Expect CtranTcpDm to be able to register host memory.");

  char buf[256];

  const auto& cudaDev = this->comm->statex_->cudaDev();
  void* memHandle = nullptr;

  COMMCHECK_TEST(CtranTcpDm::regMem(buf, sizeof(buf), cudaDev, &memHandle));

  COMMCHECK_TEST(CtranTcpDm::deregMem(memHandle));
}

TEST_F(CtranTcpTest, RegMr) {
  this->printTestDesc(
      "RegMr", "Expect CtranTcpDm to be able to register device memory.");

  const auto& cudaDev = this->comm->statex_->cudaDev();

  size_t len = 8192;
  void* buf{nullptr};
  CUDACHECK_TEST(cudaMalloc(&buf, len));

  void* memHandle = nullptr;
  COMMCHECK_TEST(CtranTcpDm::regMem(buf, sizeof(buf), cudaDev, &memHandle));

  COMMCHECK_TEST(CtranTcpDm::deregMem(memHandle));

  CUDACHECK_TEST(cudaFree(buf));
}

TEST_F(CtranTcpTest, PreConnect) {
  this->printTestDesc(
      "PreConnect", "Expect CtranTcpDm to be able to connect to peers.");

  std::unordered_set<int> peerRanks;
  if (this->globalRank == recvRank) {
    peerRanks.insert(sendRank);
    COMMCHECK_TEST(ctranTcpDm->preConnect(peerRanks));
  } else if (this->globalRank == sendRank) {
    peerRanks.insert(recvRank);
    COMMCHECK_TEST(ctranTcpDm->preConnect(peerRanks));
  }
}

TEST_F(CtranTcpTest, SendRecv) {
  this->printTestDesc(
      "SendRecv", "Expect CtranTcpDm to be able to send and receive data.");

  const auto& cudaDev = this->comm->statex_->cudaDev();
  void* memHandle = nullptr;
  uint32_t send = 0xcafebeef;
  uint32_t recv = 0;

  CtranTcpDmRequest req{};

  std::unordered_set<int> peerRanks;
  if (this->globalRank == recvRank) {
    peerRanks.insert(sendRank);
    COMMCHECK_TEST(
        CtranTcpDm::regMem(&recv, sizeof(recv), cudaDev, &memHandle));

    COMMCHECK_TEST(
        ctranTcpDm->irecv(sendRank, memHandle, &recv, sizeof(recv), req, 0));
    COMMCHECK_TEST(waitTcpReq(req, ctranTcpDm));
    EXPECT_EQ(send, recv);
  } else if (this->globalRank == sendRank) {
    peerRanks.insert(recvRank);
    COMMCHECK_TEST(
        CtranTcpDm::regMem(&send, sizeof(send), cudaDev, &memHandle));

    COMMCHECK_TEST(
        ctranTcpDm->isend(recvRank, memHandle, &send, sizeof(send), req));
    COMMCHECK_TEST(waitTcpReq(req, ctranTcpDm));
  }
  COMMCHECK_TEST(CtranTcpDm::deregMem(memHandle));
}

TEST_F(CtranTcpTest, getIfNames) {
  if (!CtranTcpDmSingleton::supportBondTransport()) {
    GTEST_SKIP() << "Skip test specific for BondTransport";
  }

  this->printTestDesc(
      "getIfNames", "Expect CtranTcpDm to be able to get interface names.");
  std::vector<std::string> ilist = {};
  int n = 1;
  std::vector<std::vector<std::string>> expected = {};
  auto res = CtranTcpDmSingleton::getIfNames(ilist, n);
  EXPECT_EQ(res == expected, 1);

  ilist = {"mlx5_0:1", "mlx5_1:1", "mlx5_3:1", "mlx5_4:1"};
  n = 2;
  expected = {{"beth0", "beth1"}, {"beth2", "beth3"}};
  res = CtranTcpDmSingleton::getIfNames(ilist, n);
  EXPECT_EQ(res == expected, 1);

  n = 1;
  expected = {{"beth0"}, {"beth1"}, {"beth2"}, {"beth3"}};
  res = CtranTcpDmSingleton::getIfNames(ilist, n);
  EXPECT_EQ(res == expected, 1);

  n = 0;
  expected = {};
  res = CtranTcpDmSingleton::getIfNames(ilist, n);
  EXPECT_EQ(res == expected, 1);

  n = 10;
  res = CtranTcpDmSingleton::getIfNames(ilist, n);
  EXPECT_EQ(res == expected, 1);

  ilist = {"mlx5_0:1", "mlx5_1:1", "mlx5_3:1"};
  n = 2;
  expected = {{"beth0", "beth1"}};
  res = CtranTcpDmSingleton::getIfNames(ilist, n);
  EXPECT_EQ(res == expected, 1);

  ilist = {
      "mlx5_2:1",
      "mlx5_0",
      "xyz:",
      "mlx5_1:",
      "mlx5_6:1",
      "mlx5_3:1",
      "mlx5_4:1"};
  expected = {{"beth0", "beth1"}, {"beth2", "beth3"}};
  res = CtranTcpDmSingleton::getIfNames(ilist, n);
  EXPECT_EQ(res == expected, 1);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
