// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include "comms/ctran/tests/CtranNcclTestUtils.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestUtils.h"

class CtranStressQpConnTest : public ctran::CtranDistTestFixture {
 public:
  // Times to repeat the test
  int repeat{5};
  // Number of comms to create in each iteration
  int numComms{10};
  std::unique_ptr<CtranComm> commWorld_;
  CtranComm* commWorld{nullptr};

  CtranStressQpConnTest() = default;

  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    ctran::CtranDistTestFixture::SetUp();

    // Allow overriding the number of comms and repeat count
    char* repeatStr = getenv("NUM_REPEAT");
    if (repeatStr) {
      repeat = atoi(repeatStr);
    }

    char* numCommsStr = getenv("NUM_COMMS");
    if (numCommsStr) {
      numComms = atoi(numCommsStr);
    }

    commWorld_ = makeCtranComm();
    commWorld = commWorld_.get();
  }

  void TearDown() override {
    commWorld = nullptr;
    commWorld_.reset();
    ctran::CtranDistTestFixture::TearDown();
  }

  void* allocBuf(size_t nbytes, void** handle, CtranComm* comm) {
    std::vector<TestMemSegment> segments;
    void* buf = ctran::CtranNcclTestHelpers::prepareBuf(
        nbytes, kMemNcclMemAlloc, segments);
    COMMCHECK_TEST(comm->ctran_->commRegister(buf, nbytes, handle));
    return buf;
  }

  void releaseBuf(void* buf, size_t nbytes, void* handle, CtranComm* comm) {
    COMMCHECK_TEST(comm->ctran_->commDeregister(handle));
    ctran::CtranNcclTestHelpers::releaseBuf(buf, nbytes, kMemNcclMemAlloc);
  }
};

TEST_F(CtranStressQpConnTest, AllToAll) {
  // Create comm and run 1 collective to ensure QP connection
  // Repeat it multiple times to catch potential race in QP connection
  const int count = 65536;

  if (!commWorld->ctran_->mapper->hasBackend()) {
    GTEST_SKIP() << "No backend available. Skip test";
  }

  for (int iter = 0; iter < repeat; iter++) {
    if (globalRank == 0) {
      std::cout
          << "StressRun "
          << ::testing::UnitTest::GetInstance()->current_test_info()->name()
          << " with " << numComms << " comms in iteration " << iter
          << " of total " << repeat << std::endl;
    }

    size_t bufCount = count * numRanks;

    std::vector<std::unique_ptr<CtranComm>> comms;
    comms.reserve(numComms);
    std::vector<cudaStream_t> streams(numComms, 0);

    std::vector<void*> sendBufs(numComms, nullptr);
    std::vector<void*> sendHdls(numComms, nullptr);
    std::vector<void*> recvBufs(numComms, nullptr);
    std::vector<void*> recvHdls(numComms, nullptr);

    // Create all communicators and streams
    for (int i = 0; i < numComms; ++i) {
      comms.push_back(makeCtranComm());
      CUDACHECK_TEST(cudaStreamCreate(&streams[i]));

      sendBufs[i] =
          allocBuf(bufCount * sizeof(int), &sendHdls[i], comms[i].get());
      ASSERT_NE(sendBufs[i], nullptr);
      recvBufs[i] =
          allocBuf(bufCount * sizeof(int), &recvHdls[i], comms[i].get());
      ASSERT_NE(recvBufs[i], nullptr);
    }

    // Launch collective on each communicator concurrently
    for (int i = 0; i < numComms; ++i) {
      auto res = ctranAllToAll(
          sendBufs[i],
          recvBufs[i],
          count,
          commInt,
          comms[i].get(),
          streams[i],
          NCCL_ALLTOALL_ALGO::ctran);
      ASSERT_EQ(res, commSuccess);
    }

    // Let all collectives complete
    CUDACHECK_TEST(cudaDeviceSynchronize());

    for (int i = 0; i < numComms; ++i) {
      const size_t bufSize = bufCount * sizeof(int);
      releaseBuf(sendBufs[i], bufSize, sendHdls[i], comms[i].get());
      releaseBuf(recvBufs[i], bufSize, recvHdls[i], comms[i].get());
      CUDACHECK_TEST(cudaStreamDestroy(streams[i]));
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
