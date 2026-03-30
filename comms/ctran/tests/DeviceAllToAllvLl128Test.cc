// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <folly/init/Init.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace meta::comms;

class DeviceAllToAllvLl128Environment : public ctran::CtranEnvironmentBase {
 public:
  void SetUp() override {
    ctran::CtranEnvironmentBase::SetUp();
    setenv("NCCL_CTRAN_USE_PIPES", "1", 1);
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    setenv("NCCL_CTRAN_DA2A_LL128_THRESHOLD", "262144", 1);
    setenv("NCCL_CTRAN_PIPES_DISABLE_IB", "1", 1);
    setenv("NCCL_CTRAN_ALLOW_CUDA_GRAPH", "1", 1);
  }

  void TearDown() override {
    unsetenv("NCCL_CTRAN_USE_PIPES");
    unsetenv("NCCL_CTRAN_ENABLE");
    unsetenv("NCCL_CTRAN_DA2A_LL128_THRESHOLD");
    unsetenv("NCCL_CTRAN_PIPES_DISABLE_IB");
    unsetenv("NCCL_CTRAN_ALLOW_CUDA_GRAPH");
    ctran::CtranEnvironmentBase::TearDown();
  }
};

// RAII wrapper for paired device count arrays.
// Note: CUDACHECK_TEST (ASSERT_*) in ctor records failure but doesn't throw;
// members default to nullptr so destructor is safe on failure.
struct DeviceCounts {
  int64_t* send_d = nullptr;
  int64_t* recv_d = nullptr;

  DeviceCounts(
      const std::vector<int64_t>& h_send,
      const std::vector<int64_t>& h_recv) {
    size_t n = h_send.size();
    CUDACHECK_TEST(cudaMalloc(&send_d, n * sizeof(int64_t)));
    CUDACHECK_TEST(cudaMalloc(&recv_d, n * sizeof(int64_t)));
    CUDACHECK_TEST(cudaMemcpy(
        send_d, h_send.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        recv_d, h_recv.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
  }

  ~DeviceCounts() {
    cudaFree(send_d);
    cudaFree(recv_d);
  }

  DeviceCounts(const DeviceCounts&) = delete;
  DeviceCounts& operator=(const DeviceCounts&) = delete;
};

class DeviceAllToAllvLl128Test : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    CtranDistTestFixture::SetUp();
    CUDACHECK_TEST(cudaStreamCreate(&stream_));
    comm_ = makeCtranComm();
    ASSERT_NE(comm_, nullptr);
    ASSERT_NE(comm_->multiPeerTransport_, nullptr);
    if (!ctranDeviceAllToAllvSupport(comm_.get())) {
      GTEST_SKIP()
          << "deviceAllToAllv not supported (requires all NVLink peers)";
    }
  }

  void TearDown() override {
    comm_.reset();
    CUDACHECK_TEST(cudaStreamDestroy(stream_));
    CtranDistTestFixture::TearDown();
  }

 protected:
  void verify_uniform_recv(
      float* recvBuf,
      size_t totalSize,
      size_t elementsPerPeer,
      const std::string& prefix = "") {
    std::vector<float> h_recv(totalSize);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(),
        recvBuf,
        totalSize * sizeof(float),
        cudaMemcpyDeviceToHost));
    for (int j = 0; j < numRanks; j++) {
      for (size_t k = 0; k < elementsPerPeer; k++) {
        EXPECT_EQ(h_recv[j * elementsPerPeer + k], static_cast<float>(j))
            << prefix << "Rank " << globalRank << ": segment " << j
            << " element " << k;
      }
    }
  }

  cudaStream_t stream_;
  std::unique_ptr<CtranComm> comm_;
};

// Parameterized fixture: bool param = true for CUDA graph mode, false for
// eager.
class DeviceAllToAllvLl128ParamTest
    : public DeviceAllToAllvLl128Test,
      public ::testing::WithParamInterface<bool> {
 protected:
  bool useCudaGraph() const {
    return GetParam();
  }

  // Run ctranDeviceAllToAllv either eagerly or via CUDA graph capture.
  commResult_t runAllToAllv(
      const void* sendBuf,
      void* recvBuf,
      const int64_t* sendcounts_d,
      const int64_t* recvcounts_d,
      commDataType_t datatype,
      int64_t sendcountsMultiplier = 1,
      int64_t recvcountsMultiplier = 1) {
    if (!useCudaGraph()) {
      auto result = ctranDeviceAllToAllv(
          sendBuf,
          recvBuf,
          sendcounts_d,
          recvcounts_d,
          datatype,
          comm_.get(),
          stream_,
          sendcountsMultiplier,
          recvcountsMultiplier);
      if (result != commSuccess) {
        return result;
      }
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
      return commSuccess;
    }

    // CUDA graph capture flow
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    CUDACHECK_TEST(
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    auto result = ctranDeviceAllToAllv(
        sendBuf,
        recvBuf,
        sendcounts_d,
        recvcounts_d,
        datatype,
        comm_.get(),
        stream_,
        sendcountsMultiplier,
        recvcountsMultiplier);
    if (result != commSuccess) {
      cudaGraph_t abandoned;
      cudaStreamEndCapture(stream_, &abandoned);
      return result;
    }
    CUDACHECK_TEST(cudaStreamEndCapture(stream_, &graph));
    CUDACHECK_TEST(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
    CUDACHECK_TEST(cudaGraphLaunch(instance, stream_));
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    CUDACHECK_TEST(cudaGraphExecDestroy(instance));
    CUDACHECK_TEST(cudaGraphDestroy(graph));
    return commSuccess;
  }
};

// Uniform split: each rank sends/receives chunkSize elements to/from every peer
TEST_P(DeviceAllToAllvLl128ParamTest, UniformSplitLl128) {
  const int nRanks = numRanks;
  // chunkSize must be multiple of 4 for float32 to satisfy LL128 16-byte
  // alignment
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
  DeviceCounts counts(h_counts, h_counts);

  auto result =
      runAllToAllv(sendBuf, recvBuf, counts.send_d, counts.recv_d, commFloat);
  ASSERT_EQ(result, commSuccess);

  verify_uniform_recv(recvBuf, totalSize, chunkSize);

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

// Variable split: each rank sends different amounts per peer
TEST_P(DeviceAllToAllvLl128ParamTest, VariableSplitLl128) {
  const int nRanks = numRanks;
  // Base chunk + rank * 64 — all counts are multiples of 4 (float32 alignment)
  const size_t baseChunk = 256;
  const size_t step = 64;

  // Compute per-peer counts
  std::vector<int64_t> h_sendcounts(nRanks);
  std::vector<int64_t> h_recvcounts(nRanks);

  size_t sendTotal = 0;
  size_t recvTotal = 0;
  for (int i = 0; i < nRanks; i++) {
    // Each rank sends baseChunk + i * step elements to peer i
    h_sendcounts[i] = static_cast<int64_t>(baseChunk + i * step);
    sendTotal += h_sendcounts[i];

    // This rank receives baseChunk + globalRank * step from peer i
    // (peer i sends baseChunk + globalRank * step to us)
    h_recvcounts[i] = static_cast<int64_t>(baseChunk + globalRank * step);
    recvTotal += h_recvcounts[i];
  }

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendTotal * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvTotal * sizeof(float)));

  // Fill send buffer: each element = globalRank
  std::vector<float> h_send(sendTotal, static_cast<float>(globalRank));
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf,
      h_send.data(),
      sendTotal * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvTotal * sizeof(float)));

  DeviceCounts counts(h_sendcounts, h_recvcounts);

  auto result =
      runAllToAllv(sendBuf, recvBuf, counts.send_d, counts.recv_d, commFloat);
  ASSERT_EQ(result, commSuccess);

  std::vector<float> h_recv(recvTotal);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv.data(),
      recvBuf,
      recvTotal * sizeof(float),
      cudaMemcpyDeviceToHost));

  // Verify: segment from peer j should contain value j
  // Compute recv displacements for verification
  size_t recvOffset = 0;
  for (int j = 0; j < nRanks; j++) {
    size_t count = h_recvcounts[j];
    for (size_t k = 0; k < count; k++) {
      EXPECT_EQ(h_recv[recvOffset + k], static_cast<float>(j))
          << "Rank " << globalRank << ": from peer " << j << " element " << k
          << " expected " << j << " got " << h_recv[recvOffset + k];
    }
    recvOffset += count;
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

// Zero count peer: all cross-rank traffic is zero (self-copy only)
TEST_P(DeviceAllToAllvLl128ParamTest, ZeroCountPeerLl128) {
  const int nRanks = numRanks;
  const size_t chunkSize = 1024;

  // Each rank self-copies chunkSize; all cross-rank counts are zero
  std::vector<int64_t> h_sendcounts(nRanks);
  std::vector<int64_t> h_recvcounts(nRanks);

  size_t sendTotal = 0;
  size_t recvTotal = 0;
  for (int i = 0; i < nRanks; i++) {
    if (i == globalRank) {
      h_sendcounts[i] = static_cast<int64_t>(chunkSize);
      h_recvcounts[i] = static_cast<int64_t>(chunkSize);
    } else {
      h_sendcounts[i] = 0;
      h_recvcounts[i] = 0;
    }
    sendTotal += h_sendcounts[i];
    recvTotal += h_recvcounts[i];
  }

  // Need at least 1 byte allocated even if total is 0
  size_t sendAllocSize = std::max(sendTotal, static_cast<size_t>(1));
  size_t recvAllocSize = std::max(recvTotal, static_cast<size_t>(1));

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendAllocSize * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvAllocSize * sizeof(float)));

  std::vector<float> h_send(sendAllocSize, static_cast<float>(globalRank));
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf,
      h_send.data(),
      sendAllocSize * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvAllocSize * sizeof(float)));

  DeviceCounts counts(h_sendcounts, h_recvcounts);

  auto result =
      runAllToAllv(sendBuf, recvBuf, counts.send_d, counts.recv_d, commFloat);
  ASSERT_EQ(result, commSuccess);

  std::vector<float> h_recv(recvAllocSize);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv.data(),
      recvBuf,
      recvAllocSize * sizeof(float),
      cudaMemcpyDeviceToHost));

  // Verify: segment from peer j (j != 0) should contain value j
  // Compute recv displacements for verification
  size_t recvOffset = 0;
  for (int j = 0; j < nRanks; j++) {
    size_t count = h_recvcounts[j];
    for (size_t k = 0; k < count; k++) {
      EXPECT_EQ(h_recv[recvOffset + k], static_cast<float>(j))
          << "Rank " << globalRank << ": from peer " << j << " element " << k
          << " expected " << j << " got " << h_recv[recvOffset + k];
    }
    recvOffset += count;
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

// Multi-dimensional uniform split: split sizes are "row counts" (dim-0 slices),
// and each row has numCols elements. The kernel multiplies counts by
// sendcountsMultiplier/recvcountsMultiplier to get actual element counts.
TEST_P(DeviceAllToAllvLl128ParamTest, UniformSplitLl128MultiDim) {
  const int nRanks = numRanks;
  const size_t chunkRows = 1024; // rows per peer
  const size_t numCols = 4; // elements per row (must keep 16-byte alignment)
  const size_t totalRows = chunkRows * nRanks;
  const size_t totalElements = totalRows * numCols;

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, totalElements * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, totalElements * sizeof(float)));

  std::vector<float> h_send(totalElements, static_cast<float>(globalRank));
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf,
      h_send.data(),
      totalElements * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, totalElements * sizeof(float)));

  // Split sizes are ROW counts, not element counts
  std::vector<int64_t> h_counts(nRanks, static_cast<int64_t>(chunkRows));
  DeviceCounts counts(h_counts, h_counts);

  // Pass scalingFactor = numCols to convert row counts to element counts
  auto result = runAllToAllv(
      sendBuf,
      recvBuf,
      counts.send_d,
      counts.recv_d,
      commFloat,
      static_cast<int64_t>(numCols),
      static_cast<int64_t>(numCols));
  ASSERT_EQ(result, commSuccess);

  verify_uniform_recv(recvBuf, totalElements, chunkRows * numCols);

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

INSTANTIATE_TEST_SUITE_P(
    DeviceAllToAllvLl128,
    DeviceAllToAllvLl128ParamTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "CudaGraph" : "Eager";
    });

// Mixed protocol: set a low threshold so small per-peer transfers use LL128
// and large transfers fall back to Simple within the same collective.
TEST_F(DeviceAllToAllvLl128Test, MixedProtocolLl128Simple) {
  const int nRanks = numRanks;
  if (nRanks < 2) {
    GTEST_SKIP() << "Mixed protocol test requires at least 2 ranks";
  }

  // Set threshold to 2KB — peers with <= 2KB use LL128, others use Simple
  EnvRAII thresholdOverride(NCCL_CTRAN_DA2A_LL128_THRESHOLD, (int64_t)2048);

  // Variable sizes: peer 0 gets 256 elements (1KB), others get 4096 (16KB)
  // This ensures at least one peer is below threshold and at least one is
  // above.
  std::vector<int64_t> h_sendcounts(nRanks);
  std::vector<int64_t> h_recvcounts(nRanks);
  size_t sendTotal = 0, recvTotal = 0;

  for (int i = 0; i < nRanks; i++) {
    // Counts must be multiples of 4 (float32 × 4 = 16 bytes alignment)
    h_sendcounts[i] = (i == 0) ? 256 : 4096;
    h_recvcounts[i] = (globalRank == 0) ? 256 : 4096;
    sendTotal += h_sendcounts[i];
    recvTotal += h_recvcounts[i];
  }

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendTotal * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvTotal * sizeof(float)));

  std::vector<float> h_send(sendTotal, static_cast<float>(globalRank));
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf,
      h_send.data(),
      sendTotal * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvTotal * sizeof(float)));

  DeviceCounts counts(h_sendcounts, h_recvcounts);

  auto result = ctranDeviceAllToAllv(
      sendBuf,
      recvBuf,
      counts.send_d,
      counts.recv_d,
      commFloat,
      comm_.get(),
      stream_);
  ASSERT_EQ(result, commSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(stream_));

  // Verify correctness
  std::vector<float> h_recv(recvTotal);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv.data(),
      recvBuf,
      recvTotal * sizeof(float),
      cudaMemcpyDeviceToHost));

  size_t recvOffset = 0;
  for (int j = 0; j < nRanks; j++) {
    size_t count = h_recvcounts[j];
    for (size_t k = 0; k < count; k++) {
      EXPECT_EQ(h_recv[recvOffset + k], static_cast<float>(j))
          << "Rank " << globalRank << ": from peer " << j << " element " << k;
    }
    recvOffset += count;
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

// Threshold=0: forces Simple protocol for all peers (LL128 disabled).
TEST_F(DeviceAllToAllvLl128Test, ThresholdZeroForcesSimple) {
  EnvRAII thresholdOverride(NCCL_CTRAN_DA2A_LL128_THRESHOLD, (int64_t)0);

  const int nRanks = numRanks;
  const size_t chunkSize = 256;
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
  DeviceCounts counts(h_counts, h_counts);

  auto result = ctranDeviceAllToAllv(
      sendBuf,
      recvBuf,
      counts.send_d,
      counts.recv_d,
      commFloat,
      comm_.get(),
      stream_);
  ASSERT_EQ(result, commSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(stream_));

  verify_uniform_recv(recvBuf, totalSize, chunkSize);

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

// Advanced graph tests: multi-replay and changed-data verify graph correctness
// beyond basic capture/launch. Kept as TEST_F since they test specific graph
// behaviors that don't map to the eager/graph parameterization.
TEST_F(DeviceAllToAllvLl128Test, CudaGraphMultiReplay) {
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
  DeviceCounts counts(h_counts, h_counts);

  cudaStream_t cudagraph_stream;
  CUDACHECK_TEST(cudaStreamCreate(&cudagraph_stream));

  cudaGraph_t graph;
  cudaGraphExec_t instance;
  CUDACHECK_TEST(
      cudaStreamBeginCapture(cudagraph_stream, cudaStreamCaptureModeGlobal));
  auto result = ctranDeviceAllToAllv(
      sendBuf,
      recvBuf,
      counts.send_d,
      counts.recv_d,
      commFloat,
      comm_.get(),
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

    std::string prefix = "Iter " + std::to_string(iter) + " ";
    verify_uniform_recv(recvBuf, totalSize, chunkSize, prefix);
  }

  CUDACHECK_TEST(cudaGraphExecDestroy(instance));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaStreamDestroy(cudagraph_stream));
  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

TEST_F(DeviceAllToAllvLl128Test, CudaGraphChangedData) {
  const int nRanks = numRanks;
  const size_t chunkSize = 1024;
  const size_t totalSize = chunkSize * nRanks;

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, totalSize * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, totalSize * sizeof(float)));

  std::vector<int64_t> h_counts(nRanks, static_cast<int64_t>(chunkSize));
  DeviceCounts counts(h_counts, h_counts);

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
      counts.send_d,
      counts.recv_d,
      commFloat,
      comm_.get(),
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
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DeviceAllToAllvLl128Environment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
