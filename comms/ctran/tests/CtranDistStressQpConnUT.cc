// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include "CtranUtUtils.h"

#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

class CtranStressQpConnTest : public NcclxBaseTest, public CtranBaseTest {
 public:
  // Times to repeat the test
  int repeat{5};
  // Number of comms to create in each iteration
  int numComms{10};
  ncclComm_t commWorld;

  CtranStressQpConnTest() = default;

  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    NcclxBaseTest::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));

    // Allow overriding the number of comms and repeat count
    char* repeatStr = getenv("NUM_REPEAT");
    if (repeatStr) {
      repeat = atoi(repeatStr);
    }

    char* numCommsStr = getenv("NUM_COMMS");
    if (numCommsStr) {
      numComms = atoi(numCommsStr);
    }

    commWorld = createNcclComm(globalRank, numRanks, localRank);
  }

  void TearDown() override {
    NcclxBaseTest::TearDown();
    NCCLCHECK_TEST(ncclCommDestroy(commWorld));
  }

  void* allocBuf(size_t nbytes, void** handle, ncclComm_t comm) {
    void* buf = nullptr;
    NCCLCHECK_TEST(ncclMemAlloc(&buf, nbytes));
    NCCLCHECK_TEST(ncclCommRegister(comm, buf, nbytes, handle));
    return buf;
  }

  void releaseBuf(void* buf, void* handle, ncclComm_t comm) {
    NCCLCHECK_TEST(ncclCommDeregister(comm, handle));
    NCCLCHECK_TEST(ncclMemFree(buf));
  }
};

TEST_F(CtranStressQpConnTest, AllToAll) {
  // Create comm and run 1 collective to ensure QP connection
  // Repeat it multiple times to catch potential race in QP connection
  const int count = 65536;

  if (!commWorld->ctranComm_->ctran_->mapper->hasBackend()) {
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

    const int groupSize = commWorld->ctranComm_->statex_.get()->nRanks();
    std::vector<int> groupRanks(groupSize);
    for (int i = 0; i < groupSize; ++i) {
      groupRanks[i] = i;
    }

    size_t bufCount = count * numRanks;

    // One stream per communicator
    std::vector<ncclComm_t> comms(numComms, NCCL_COMM_NULL);
    std::vector<cudaStream_t> streams(numComms, 0);

    // Separate buffers for each communicator to allow concurrent collectives
    std::vector<void*> sendBufs(numComms, nullptr);
    std::vector<void*> sendHdls(numComms, nullptr);
    std::vector<void*> recvBufs(numComms, nullptr);
    std::vector<void*> recvHdls(numComms, nullptr);

    // Create all communicators and streams
    for (int i = 0; i < numComms; ++i) {
      ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
      const std::string commDest = std::string("test_comm") + std::to_string(i);
      config.commDesc = commDest.c_str();
      config.splitGroupRanks = groupRanks.data();
      config.splitGroupSize = groupSize;

      NCCLCHECK_TEST(
          ncclCommSplit(commWorld, 1, globalRank, &comms[i], &config));
      CUDACHECK_TEST(cudaStreamCreate(&streams[i]));

      sendBufs[i] = allocBuf(bufCount * sizeof(int), &sendHdls[i], comms[i]);
      ASSERT_NE(sendBufs[i], nullptr);
      recvBufs[i] = allocBuf(bufCount * sizeof(int), &recvHdls[i], comms[i]);
      ASSERT_NE(recvBufs[i], nullptr);
    }

    // Launch collective on each communicator concurrently
    for (int i = 0; i < numComms; ++i) {
      auto res = ctranAllToAll(
          sendBufs[i],
          recvBufs[i],
          count,
          commInt,
          comms[i]->ctranComm_.get(),
          streams[i]);
      ASSERT_EQ(res, commSuccess);
    }

    // Let all collectives complete
    CUDACHECK_TEST(cudaDeviceSynchronize());

    for (int i = 0; i < numComms; ++i) {
      releaseBuf(sendBufs[i], sendHdls[i], comms[i]);
      releaseBuf(recvBufs[i], recvHdls[i], comms[i]);
      NCCLCHECK_TEST(ncclCommDestroy(comms[i]));
      CUDACHECK_TEST(cudaStreamDestroy(streams[i]));
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
