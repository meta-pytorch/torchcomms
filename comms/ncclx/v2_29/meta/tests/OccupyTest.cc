#include <gtest/gtest.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "comm.h"
#include "comms/testinfra/TestUtils.h"
#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"

#include <folly/init/Init.h>

class MPIEnvironment : public DistEnvironmentBase {
 public:
  void SetUp() override {
    DistEnvironmentBase::SetUp();
    // Initialize CVAR so that we can overwrite global variable in each test
    initEnv();
  }
};

class OccupyTest : public ::testing::Test {
 public:
  OccupyTest() = default;
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
  }

  void runOccupy(const int nColl, ncclComm_t comm, int cycle) {
    for (int i = 0; i < nColl; i++) {
      NCCLCHECK_TEST(ncclOccupy(
          sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream, 0, cycle));
    }
  }

 protected:
  int count{32 * 1024 * 1024};
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  cudaStream_t stream;
};

TEST_F(OccupyTest, OccupySleep) {
  auto comm = createNcclComm(this->globalRank, this->numRanks, this->localRank);

  cudaEvent_t start_event, end_event;
  CUDACHECK_TEST(cudaEventCreate(&start_event));
  CUDACHECK_TEST(cudaEventCreate(&end_event));

  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  CUDACHECK_TEST(cudaEventRecord(start_event, stream));
  runOccupy(1, comm, 1000000);
  CUDACHECK_TEST(cudaEventRecord(end_event, stream));
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  float elapsed_time_ms = -1.0;
  CUDACHECK_TEST(
      cudaEventElapsedTime(&elapsed_time_ms, start_event, end_event));
  printf("ncclOccupy() took time %.2f ms\n", elapsed_time_ms);
  EXPECT_GT(elapsed_time_ms, 0.1); // 0.76ms with A100

  CUDACHECK_TEST(cudaEventDestroy(start_event));
  CUDACHECK_TEST(cudaEventDestroy(end_event));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
