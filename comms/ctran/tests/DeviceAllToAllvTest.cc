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

// CUDA graph tests: capture ctranDeviceAllToAllv into a graph and replay.
// DeviceAllToAllvPipes passes empty opGroup to GPE (flag=nullptr), so the
// kernel launch is inherently graph-capturable with no host nodes.
#if defined(TEST_CUDA_GRAPH_MODE)

TEST_F(DeviceAllToAllvTest, UniformSplitCudaGraph) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  if (!ctranDeviceAllToAllvSupport(comm.get())) {
    GTEST_SKIP() << "deviceAllToAllv not supported (requires all NVLink peers)";
  }

  const int nRanks = numRanks;
  const size_t chunkSize = 1024;
  const size_t totalSize = chunkSize * nRanks;

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, totalSize * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, totalSize * sizeof(float)));

  std::vector<float> h_send(totalSize, static_cast<float>(globalRank));
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf,
      h_send.data(),
      totalSize * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, totalSize * sizeof(float)));

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

  // Use a separate stream for graph capture (not the fixture stream)
  cudaStream_t cudagraph_stream;
  CUDACHECK_TEST(cudaStreamCreate(&cudagraph_stream));

  // Capture
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  CUDACHECK_TEST(
      cudaStreamBeginCapture(cudagraph_stream, cudaStreamCaptureModeGlobal));
  auto result = ctranDeviceAllToAllv(
      sendBuf,
      recvBuf,
      d_sendcounts,
      d_recvcounts,
      d_senddispls,
      d_recvdispls,
      commFloat,
      comm.get(),
      cudagraph_stream);
  ASSERT_EQ(result, commSuccess);
  CUDACHECK_TEST(cudaStreamEndCapture(cudagraph_stream, &graph));
  CUDACHECK_TEST(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

  // Replay
  CUDACHECK_TEST(cudaGraphLaunch(instance, cudagraph_stream));
  CUDACHECK_TEST(cudaStreamSynchronize(cudagraph_stream));

  // Verify
  std::vector<float> h_recv(totalSize);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv.data(),
      recvBuf,
      totalSize * sizeof(float),
      cudaMemcpyDeviceToHost));
  for (int j = 0; j < nRanks; j++) {
    for (size_t k = 0; k < chunkSize; k++) {
      EXPECT_EQ(h_recv[j * chunkSize + k], static_cast<float>(j))
          << "Rank " << globalRank << ": segment " << j << " element " << k;
    }
  }

  CUDACHECK_TEST(cudaGraphExecDestroy(instance));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaStreamDestroy(cudagraph_stream));
  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(d_sendcounts));
  CUDACHECK_TEST(cudaFree(d_recvcounts));
  CUDACHECK_TEST(cudaFree(d_senddispls));
  CUDACHECK_TEST(cudaFree(d_recvdispls));
}

TEST_F(DeviceAllToAllvTest, UniformSplitCudaGraphMultiReplay) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  if (!ctranDeviceAllToAllvSupport(comm.get())) {
    GTEST_SKIP() << "deviceAllToAllv not supported (requires all NVLink peers)";
  }

  const int nRanks = numRanks;
  const size_t chunkSize = 1024;
  const size_t totalSize = chunkSize * nRanks;

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, totalSize * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, totalSize * sizeof(float)));

  std::vector<float> h_send(totalSize, static_cast<float>(globalRank));
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf,
      h_send.data(),
      totalSize * sizeof(float),
      cudaMemcpyHostToDevice));

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

  cudaStream_t cudagraph_stream;
  CUDACHECK_TEST(cudaStreamCreate(&cudagraph_stream));

  cudaGraph_t graph;
  cudaGraphExec_t instance;
  CUDACHECK_TEST(
      cudaStreamBeginCapture(cudagraph_stream, cudaStreamCaptureModeGlobal));
  auto result = ctranDeviceAllToAllv(
      sendBuf,
      recvBuf,
      d_sendcounts,
      d_recvcounts,
      d_senddispls,
      d_recvdispls,
      commFloat,
      comm.get(),
      cudagraph_stream);
  ASSERT_EQ(result, commSuccess);
  CUDACHECK_TEST(cudaStreamEndCapture(cudagraph_stream, &graph));
  CUDACHECK_TEST(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

  // Replay 5 times, verify each time
  constexpr int numIters = 5;
  for (int iter = 0; iter < numIters; iter++) {
    CUDACHECK_TEST(cudaMemsetAsync(
        recvBuf, 0, totalSize * sizeof(float), cudagraph_stream));
    CUDACHECK_TEST(cudaGraphLaunch(instance, cudagraph_stream));
    CUDACHECK_TEST(cudaStreamSynchronize(cudagraph_stream));

    std::vector<float> h_recv(totalSize);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(),
        recvBuf,
        totalSize * sizeof(float),
        cudaMemcpyDeviceToHost));
    for (int j = 0; j < nRanks; j++) {
      for (size_t k = 0; k < chunkSize; k++) {
        EXPECT_EQ(h_recv[j * chunkSize + k], static_cast<float>(j))
            << "Iter " << iter << " Rank " << globalRank << ": segment " << j
            << " element " << k;
      }
    }
  }

  CUDACHECK_TEST(cudaGraphExecDestroy(instance));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaStreamDestroy(cudagraph_stream));
  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(d_sendcounts));
  CUDACHECK_TEST(cudaFree(d_recvcounts));
  CUDACHECK_TEST(cudaFree(d_senddispls));
  CUDACHECK_TEST(cudaFree(d_recvdispls));
}

TEST_F(DeviceAllToAllvTest, UniformSplitCudaGraphChangedData) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  if (!ctranDeviceAllToAllvSupport(comm.get())) {
    GTEST_SKIP() << "deviceAllToAllv not supported (requires all NVLink peers)";
  }

  const int nRanks = numRanks;
  const size_t chunkSize = 1024;
  const size_t totalSize = chunkSize * nRanks;

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, totalSize * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, totalSize * sizeof(float)));

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

  // Fill initial data for capture
  std::vector<float> h_send(totalSize, static_cast<float>(globalRank));
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf,
      h_send.data(),
      totalSize * sizeof(float),
      cudaMemcpyHostToDevice));

  cudaStream_t cudagraph_stream;
  CUDACHECK_TEST(cudaStreamCreate(&cudagraph_stream));

  cudaGraph_t graph;
  cudaGraphExec_t instance;
  CUDACHECK_TEST(
      cudaStreamBeginCapture(cudagraph_stream, cudaStreamCaptureModeGlobal));
  auto result = ctranDeviceAllToAllv(
      sendBuf,
      recvBuf,
      d_sendcounts,
      d_recvcounts,
      d_senddispls,
      d_recvdispls,
      commFloat,
      comm.get(),
      cudagraph_stream);
  ASSERT_EQ(result, commSuccess);
  CUDACHECK_TEST(cudaStreamEndCapture(cudagraph_stream, &graph));
  CUDACHECK_TEST(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

  // Replay with different send data each iteration
  constexpr int numIters = 3;
  for (int iter = 0; iter < numIters; iter++) {
    float fillVal = static_cast<float>(globalRank * 100 + iter);
    std::vector<float> h_data(totalSize, fillVal);
    // Use async memcpy on the graph stream to avoid ordering issues
    CUDACHECK_TEST(cudaMemcpyAsync(
        sendBuf,
        h_data.data(),
        totalSize * sizeof(float),
        cudaMemcpyHostToDevice,
        cudagraph_stream));
    CUDACHECK_TEST(cudaMemsetAsync(
        recvBuf, 0, totalSize * sizeof(float), cudagraph_stream));
    CUDACHECK_TEST(cudaGraphLaunch(instance, cudagraph_stream));
    CUDACHECK_TEST(cudaStreamSynchronize(cudagraph_stream));

    std::vector<float> h_recv(totalSize);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(),
        recvBuf,
        totalSize * sizeof(float),
        cudaMemcpyDeviceToHost));
    for (int j = 0; j < nRanks; j++) {
      float expected = static_cast<float>(j * 100 + iter);
      for (size_t k = 0; k < chunkSize; k++) {
        EXPECT_EQ(h_recv[j * chunkSize + k], expected)
            << "Iter " << iter << " Rank " << globalRank << ": segment " << j
            << " element " << k << " expected " << expected;
      }
    }
  }

  CUDACHECK_TEST(cudaGraphExecDestroy(instance));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaStreamDestroy(cudagraph_stream));
  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(d_sendcounts));
  CUDACHECK_TEST(cudaFree(d_recvcounts));
  CUDACHECK_TEST(cudaFree(d_senddispls));
  CUDACHECK_TEST(cudaFree(d_recvdispls));
}

#endif // TEST_CUDA_GRAPH_MODE

// Verify support check passes when pipes is initialized
TEST_F(DeviceAllToAllvTest, SupportedWithPipes) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  if (!comm->multiPeerTransport_) {
    GTEST_SKIP() << "MultiPeerTransport not available (requires NVLink peers)";
  }

  // Should be supported with MultiPeerTransport and NVLink peers
  EXPECT_TRUE(ctranDeviceAllToAllvSupport(comm.get()));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DeviceAllToAllvEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
