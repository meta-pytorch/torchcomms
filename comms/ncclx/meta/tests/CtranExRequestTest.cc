// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <cstring>

#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "meta/wrapper/CtranExComm.h"

class CtranExRequestTest : public NcclxBaseTestFixture {
 public:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp();
  }

  void TearDown() override {
    NcclxBaseTestFixture::TearDown();
  }
};

TEST_F(CtranExRequestTest, WaitBlocksUntilComplete) {
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  auto ctranExComm =
      std::make_unique<::ctran::CtranExComm>(comm, "ctranExRequestTest");
  ASSERT_NE(ctranExComm, nullptr);

  if (!ctranExComm->isInitialized() || !ctranExComm->supportBroadcast()) {
    GTEST_SKIP() << "CtranEx not initialized or broadcast not supported.";
  }

  constexpr int count = 1024;
  constexpr int root = 0;
  constexpr int fillValue = 42;

  int* sendbuf = reinterpret_cast<int*>(malloc(count * sizeof(int)));
  int* recvbuf = reinterpret_cast<int*>(malloc(count * sizeof(int)));
  ASSERT_NE(sendbuf, nullptr);
  ASSERT_NE(recvbuf, nullptr);

  void* sendHandle = nullptr;
  void* recvHandle = nullptr;
  NCCLCHECK_TEST(
      ctranExComm->regMem(sendbuf, count * sizeof(int), &sendHandle, true));
  NCCLCHECK_TEST(
      ctranExComm->regMem(recvbuf, count * sizeof(int), &recvHandle, true));

  if (globalRank == root) {
    for (int i = 0; i < count; i++) {
      sendbuf[i] = fillValue;
    }
  }
  memset(recvbuf, 0, count * sizeof(int));

  ::ctran::CtranExRequest* reqPtr = nullptr;
  NCCLCHECK_TEST(
      ctranExComm->broadcast(sendbuf, recvbuf, count, ncclInt, root, &reqPtr));
  ASSERT_NE(reqPtr, nullptr);
  auto req = std::unique_ptr<::ctran::CtranExRequest>(reqPtr);

  ASSERT_EQ(req->wait(), ncclSuccess);

  for (int i = 0; i < count; i++) {
    EXPECT_EQ(recvbuf[i], fillValue)
        << "Data mismatch at index " << i << " on rank " << globalRank
        << ": wait() may have returned before broadcast completed.";
    if (recvbuf[i] != fillValue) {
      break;
    }
  }

  NCCLCHECK_TEST(ctranExComm->deregMem(sendHandle));
  NCCLCHECK_TEST(ctranExComm->deregMem(recvHandle));
  free(sendbuf);
  free(recvbuf);
}

TEST_F(CtranExRequestTest, WaitIsIdempotent) {
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  auto ctranExComm =
      std::make_unique<::ctran::CtranExComm>(comm, "ctranExRequestTest");
  ASSERT_NE(ctranExComm, nullptr);

  if (!ctranExComm->isInitialized() || !ctranExComm->supportBroadcast()) {
    GTEST_SKIP() << "CtranEx not initialized or broadcast not supported.";
  }

  constexpr int count = 256;

  int* buf = reinterpret_cast<int*>(malloc(count * sizeof(int)));
  ASSERT_NE(buf, nullptr);
  void* handle = nullptr;
  NCCLCHECK_TEST(ctranExComm->regMem(buf, count * sizeof(int), &handle, true));

  ::ctran::CtranExRequest* reqPtr = nullptr;
  NCCLCHECK_TEST(ctranExComm->broadcast(buf, buf, count, ncclInt, 0, &reqPtr));
  ASSERT_NE(reqPtr, nullptr);
  auto req = std::unique_ptr<::ctran::CtranExRequest>(reqPtr);

  ASSERT_EQ(req->wait(), ncclSuccess);
  ASSERT_EQ(req->wait(), ncclSuccess);

  NCCLCHECK_TEST(ctranExComm->deregMem(handle));
  free(buf);
}

TEST_F(CtranExRequestTest, MultipleBroadcastsWaitAll) {
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  auto ctranExComm =
      std::make_unique<::ctran::CtranExComm>(comm, "ctranExRequestTest");
  ASSERT_NE(ctranExComm, nullptr);

  if (!ctranExComm->isInitialized() || !ctranExComm->supportBroadcast()) {
    GTEST_SKIP() << "CtranEx not initialized or broadcast not supported.";
  }

  constexpr int count = 512;
  constexpr int nColl = 5;

  int* sendbuf = reinterpret_cast<int*>(malloc(count * sizeof(int)));
  int* recvbuf = reinterpret_cast<int*>(malloc(count * sizeof(int)));
  ASSERT_NE(sendbuf, nullptr);
  ASSERT_NE(recvbuf, nullptr);

  void* sendHandle = nullptr;
  void* recvHandle = nullptr;
  NCCLCHECK_TEST(
      ctranExComm->regMem(sendbuf, count * sizeof(int), &sendHandle, true));
  NCCLCHECK_TEST(
      ctranExComm->regMem(recvbuf, count * sizeof(int), &recvHandle, true));

  std::vector<std::unique_ptr<::ctran::CtranExRequest>> reqs;

  for (int i = 0; i < nColl; i++) {
    if (globalRank == 0) {
      for (int j = 0; j < count; j++) {
        sendbuf[j] = i + 1;
      }
    }
    memset(recvbuf, 0, count * sizeof(int));

    ::ctran::CtranExRequest* reqPtr = nullptr;
    NCCLCHECK_TEST(
        ctranExComm->broadcast(sendbuf, recvbuf, count, ncclInt, 0, &reqPtr));
    ASSERT_NE(reqPtr, nullptr);
    reqs.push_back(std::unique_ptr<::ctran::CtranExRequest>(reqPtr));
  }

  for (auto& req : reqs) {
    ASSERT_EQ(req->wait(), ncclSuccess);
  }

  EXPECT_EQ(recvbuf[0], nColl)
      << "Last broadcast data not received on rank " << globalRank
      << ": wait() may have returned before broadcast completed.";

  NCCLCHECK_TEST(ctranExComm->deregMem(sendHandle));
  NCCLCHECK_TEST(ctranExComm->deregMem(recvHandle));
  free(sendbuf);
  free(recvbuf);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
