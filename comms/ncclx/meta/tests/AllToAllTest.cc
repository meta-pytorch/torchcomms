// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include "checks.h"

#include "comms/ctran/Ctran.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/commDump.h"

static const int kTotalColls = 5;

class AllToAllTest : public NcclxBaseTestFixture {
 public:
  AllToAllTest() = default;
  void SetUp() override {
#ifdef TEST_ENABLE_CTRAN
    setenv("NCCL_COLLTRACE", "trace", 0);
#endif
    NcclxBaseTestFixture::SetUp();

    this->comm = ncclx::test::createNcclComm(
        globalRank, numRanks, localRank, bootstrap_.get());

    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    NcclxBaseTestFixture::TearDown();
  }

  void run(bool registFlag = false) {
#ifdef NCCL_ALLTOALL_SUPPORTED

    // create and register buffers
    constexpr int count = 1048576;
    int *sendBuf = nullptr, *recvBuf = nullptr;

    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * numRanks * sizeof(int)));

    for (int r = 0; r < numRanks; r++) {
      int expectedVal = globalRank * 100 + r + 1;
      assignChunkValue(sendBuf + r * count, count, expectedVal);
      assignChunkValue(recvBuf + r * count, count, -1);
    }

#ifdef NCCL_REGISTRATION_SUPPORTED
    if (registFlag) {
      NCCLCHECK_TEST(ncclCommRegister(
          comm, sendBuf, count * numRanks * sizeof(int), &sendHandle));
      NCCLCHECK_TEST(ncclCommRegister(
          comm, recvBuf, count * numRanks * sizeof(int), &recvHandle));
    }
#endif

    // run alltoall
    for (int i = 0; i < kTotalColls; i++) {
      auto res = ncclAllToAll(sendBuf, recvBuf, count, ncclInt, comm, stream);
      ASSERT_EQ(res, ncclSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    for (int r = 0; r < numRanks; r++) {
      int expectedVal = r * 100 + globalRank + 1;
      int errs = checkChunkValue(recvBuf + r * count, count, expectedVal);
      EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << r
                         << " at " << recvBuf + r * count << " with " << errs
                         << " errors";
    }

#ifdef NCCL_REGISTRATION_SUPPORTED
    if (registFlag) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
      NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    }
#endif

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));

#ifdef TEST_ENABLE_CTRAN

    if (comm->newCollTrace) {
      EXPECT_TRUE(
          meta::comms::ncclx::waitForCollTraceDrain(*comm->newCollTrace));

      auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);
      if (dumpMap.count("CT_pastColls")) {
        auto ctPastColls = folly::parseJson(dumpMap["CT_pastColls"]);
        int totalColls = kTotalColls;
        if (count == 0) {
          totalColls = 0;
        }
        EXPECT_EQ(ctPastColls.size(), totalColls);
        for (int i = 0; i < static_cast<int>(ctPastColls.size()); i++) {
          EXPECT_EQ(ctPastColls[i]["collId"].asInt(), i);
          EXPECT_EQ(ctPastColls[i]["opCount"].asInt(), i);
        }
      }
    }
#endif
#endif
  }

 protected:
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_F(AllToAllTest, OutOfPlace) {
  run();
}

#ifdef TEST_ENABLE_CTRAN
TEST_F(AllToAllTest, Ctran) {
  auto envGuard = EnvRAII(NCCL_ALLTOALL_ALGO, NCCL_ALLTOALL_ALGO::ctran);
  run();
}
#endif

TEST_F(AllToAllTest, InvalidSendbuf) {
#ifdef NCCL_ALLTOALL_SUPPORTED
  constexpr int count = 1048576;
  int* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, count * numRanks * sizeof(int)));

  // run alltoall
  auto res = ncclAllToAll(nullptr, buf, count, ncclInt, comm, stream);
  ASSERT_EQ(res, ncclInvalidArgument);
  CUDACHECK_TEST(cudaFree(buf));
#endif
}

TEST_F(AllToAllTest, InvalidRecvbuf) {
#ifdef NCCL_ALLTOALL_SUPPORTED
  constexpr int count = 1048576;
  int* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, count * numRanks * sizeof(int)));

  // run alltoall
  auto res = ncclAllToAll(buf, nullptr, count, ncclInt, comm, stream);
  ASSERT_EQ(res, ncclInvalidArgument);
  CUDACHECK_TEST(cudaFree(buf));
#endif
}

TEST_F(AllToAllTest, InvalidInPlace) {
#ifdef NCCL_ALLTOALL_SUPPORTED
  constexpr int count = 1048576;
  int* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, count * numRanks * sizeof(int)));

  // run alltoall
  auto res = ncclAllToAll(buf, buf, count, ncclInt, comm, stream);
  ASSERT_EQ(res, ncclInvalidArgument);
  CUDACHECK_TEST(cudaFree(buf));
#endif
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
