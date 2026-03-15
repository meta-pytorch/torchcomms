// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>

#include <numeric>
#include <vector>

#include <folly/init/Init.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"

using namespace meta::comms;

class DeviceAllToAllvEnvironment : public ctran::CtranEnvironmentBase {
 public:
  void SetUp() override {
    ctran::CtranEnvironmentBase::SetUp();
    setenv("NCCL_CTRAN_USE_PIPES", "1", 1);
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
  }
};

class DeviceAllToAllvTest : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    CtranDistTestFixture::SetUp();
    CUDACHECK_TEST(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream_));
    CtranDistTestFixture::TearDown();
  }

 protected:
  cudaStream_t stream_;
};

// Uniform split: each rank sends/receives chunkSize elements to/from every peer
TEST_F(DeviceAllToAllvTest, UniformSplit) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  // Check support — skip if not all NVLink peers
  if (!ctranDeviceAllToAllvSupport(comm.get())) {
    GTEST_SKIP() << "deviceAllToAllv not supported (requires all NVLink peers)";
  }

  const int nRanks = numRanks;
  const size_t chunkSize = 1024; // CTRAN minimum
  const size_t totalSize = chunkSize * nRanks;

  // Allocate GPU buffers
  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, totalSize * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, totalSize * sizeof(float)));

  // Fill send buffer with rank value
  std::vector<float> h_send(totalSize, static_cast<float>(globalRank));
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf,
      h_send.data(),
      totalSize * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, totalSize * sizeof(float)));

  // Create device count/offset arrays (uniform split)
  std::vector<int64_t> h_counts(nRanks, static_cast<int64_t>(chunkSize));
  std::vector<int64_t> h_offsets(nRanks);
  for (int i = 0; i < nRanks; i++) {
    h_offsets[i] = static_cast<int64_t>(i * chunkSize);
  }

  int64_t* d_sendcounts = nullptr;
  int64_t* d_recvcounts = nullptr;
  int64_t* d_senddispls = nullptr;
  int64_t* d_recvdispls = nullptr;
  CUDACHECK_TEST(cudaMalloc(&d_sendcounts, nRanks * sizeof(int64_t)));
  CUDACHECK_TEST(cudaMalloc(&d_recvcounts, nRanks * sizeof(int64_t)));
  CUDACHECK_TEST(cudaMalloc(&d_senddispls, nRanks * sizeof(int64_t)));
  CUDACHECK_TEST(cudaMalloc(&d_recvdispls, nRanks * sizeof(int64_t)));

  CUDACHECK_TEST(cudaMemcpy(
      d_sendcounts,
      h_counts.data(),
      nRanks * sizeof(int64_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recvcounts,
      h_counts.data(),
      nRanks * sizeof(int64_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_senddispls,
      h_offsets.data(),
      nRanks * sizeof(int64_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recvdispls,
      h_offsets.data(),
      nRanks * sizeof(int64_t),
      cudaMemcpyHostToDevice));

  // Run deviceAllToAllv
  auto result = ctranDeviceAllToAllv(
      sendBuf,
      recvBuf,
      d_sendcounts,
      d_recvcounts,
      d_senddispls,
      d_recvdispls,
      commFloat,
      comm.get(),
      stream_);
  ASSERT_EQ(result, commSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(stream_));

  // Verify: segment j should contain value j (sent from rank j)
  std::vector<float> h_recv(totalSize);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv.data(),
      recvBuf,
      totalSize * sizeof(float),
      cudaMemcpyDeviceToHost));

  for (int j = 0; j < nRanks; j++) {
    for (size_t k = 0; k < chunkSize; k++) {
      EXPECT_EQ(h_recv[j * chunkSize + k], static_cast<float>(j))
          << "Rank " << globalRank << ": segment " << j << " element " << k
          << " expected " << j << " got " << h_recv[j * chunkSize + k];
    }
  }

  // Cleanup
  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(d_sendcounts));
  CUDACHECK_TEST(cudaFree(d_recvcounts));
  CUDACHECK_TEST(cudaFree(d_senddispls));
  CUDACHECK_TEST(cudaFree(d_recvdispls));
}

// Verify support check passes when pipes is initialized
TEST_F(DeviceAllToAllvTest, SupportedWithPipes) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  // Should be supported with MultiPeerTransport and NVLink peers
  EXPECT_TRUE(ctranDeviceAllToAllvSupport(comm.get()));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DeviceAllToAllvEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
