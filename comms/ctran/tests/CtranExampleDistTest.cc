// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <folly/init/Init.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

class CtranExampleEnvironment : public ctran::CtranEnvironmentBase {
 public:
  void SetUp() override {
    CtranEnvironmentBase::SetUp();

    // set logging level to WARN
    setenv("NCCL_DEBUG", "WARN", 1);
  }
};

class CtranExampleDistTest : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    CtranDistTestFixture::SetUp();
    // add extra setup here
  }

  void TearDown() override {
    // add extra teardown here
    CtranDistTestFixture::TearDown();
  }
};

TEST_F(CtranExampleDistTest, Basic) {
  auto comm = makeCtranComm();
  auto algo = comm->ctran_->algo.get();
  auto devStated_d = algo->getDevState();
  ASSERT_NE(devStated_d, nullptr);

  XLOG(WARN) << "Rank " << globalRank << ": ctran comm created";
  size_t numElements = 10;

  // Create input/output device buffers
  DeviceBuffer input_d(numElements * sizeof(int32_t));
  DeviceBuffer output_d(numElements * sizeof(int32_t) * numRanks);

  // Prepare host input data
  std::vector<int32_t> input_h(numElements);
  for (int i = 0; i < numElements; i++) {
    input_h[i] = globalRank;
  }

  // Copy input data to device
  cudaMemcpy(
      input_d.get(),
      input_h.data(),
      numElements * sizeof(int),
      cudaMemcpyHostToDevice);
  cudaMemset(output_d.get(), 0, numElements * sizeof(int) * numRanks);

  auto ret = ctranAllGather(
      input_d.get(),
      output_d.get(),
      numElements,
      commInt32,
      comm.get(),
      0,
      NCCL_ALLGATHER_ALGO::ctdirect);
  ASSERT_EQ(ret, commSuccess);

  // Wait for kernel completion
  cudaDeviceSynchronize();

  // Copy output data back to host
  std::vector<int32_t> output_h(numElements * numRanks);
  cudaMemcpy(
      output_h.data(),
      output_d.get(),
      numElements * sizeof(int) * numRanks,
      cudaMemcpyDeviceToHost);

  // Verify output
  for (int i = 0; i < numElements * numRanks; i++) {
    int expected = i / numElements;
    EXPECT_EQ(output_h[i], expected)
        << "Rank " << globalRank << ": mismatch at index " << i << ", expected "
        << expected << ", got " << output_h[i];
  }

  XLOG(WARN) << "Rank " << globalRank
             << ": AllGather Direct verification passed";
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranExampleEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
