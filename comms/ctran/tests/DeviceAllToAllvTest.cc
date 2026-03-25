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

// Parameterized fixture: bool param controls CUDA graph mode.
// When true, the alltoallv is captured into a CUDA graph and replayed.
class DeviceAllToAllvParamTest : public ctran::CtranDistTestFixture,
                                 public ::testing::WithParamInterface<bool> {
 public:
  bool useCudaGraph() const {
    return GetParam();
  }

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

  // Run ctranDeviceAllToAllv either eagerly or via CUDA graph capture+replay.
  commResult_t runAllToAllv(
      const void* sendbuff,
      void* recvbuff,
      const int64_t* sendcounts_d,
      const int64_t* recvcounts_d,
      commDataType_t datatype,
      CtranComm* comm,
      int64_t sendMultiplier = 1,
      int64_t recvMultiplier = 1) {
    if (!useCudaGraph()) {
      auto result = ctranDeviceAllToAllv(
          sendbuff,
          recvbuff,
          sendcounts_d,
          recvcounts_d,
          datatype,
          comm,
          stream_,
          sendMultiplier,
          recvMultiplier);
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
      return result;
    }

    // CUDA graph path: capture, instantiate, replay, destroy
    cudaStream_t graphStream;
    CUDACHECK_TEST(cudaStreamCreate(&graphStream));

    CUDACHECK_TEST(
        cudaStreamBeginCapture(graphStream, cudaStreamCaptureModeGlobal));
    auto result = ctranDeviceAllToAllv(
        sendbuff,
        recvbuff,
        sendcounts_d,
        recvcounts_d,
        datatype,
        comm,
        graphStream,
        sendMultiplier,
        recvMultiplier);
    if (result != commSuccess) {
      cudaStreamEndCapture(graphStream, nullptr);
      cudaStreamDestroy(graphStream);
      return result;
    }

    cudaGraph_t graph;
    CUDACHECK_TEST(cudaStreamEndCapture(graphStream, &graph));

    cudaGraphExec_t instance;
    CUDACHECK_TEST(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    CUDACHECK_TEST(cudaGraphLaunch(instance, graphStream));
    CUDACHECK_TEST(cudaStreamSynchronize(graphStream));

    CUDACHECK_TEST(cudaGraphExecDestroy(instance));
    CUDACHECK_TEST(cudaGraphDestroy(graph));
    CUDACHECK_TEST(cudaStreamDestroy(graphStream));
    return result;
  }
};

// ---------------------------------------------------------------------------
// Balanced: every peer sends/receives the same chunk size
// ---------------------------------------------------------------------------
TEST_P(DeviceAllToAllvParamTest, Balanced) {
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

  int64_t* d_sendcounts = nullptr;
  int64_t* d_recvcounts = nullptr;
  CUDACHECK_TEST(cudaMalloc(&d_sendcounts, nRanks * sizeof(int64_t)));
  CUDACHECK_TEST(cudaMalloc(&d_recvcounts, nRanks * sizeof(int64_t)));
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

  auto result = runAllToAllv(
      sendBuf, recvBuf, d_sendcounts, d_recvcounts, commFloat, comm.get());
  ASSERT_EQ(result, commSuccess);

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
          << "Rank " << globalRank << ": segment " << j << " element " << k;
    }
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(d_sendcounts));
  CUDACHECK_TEST(cudaFree(d_recvcounts));
}

// ---------------------------------------------------------------------------
// Imbalanced: variable send/recv sizes per peer
// sendcounts[j] = (globalRank * nRanks + j + 1) * baseChunk
// recvcounts[j] = (j * nRanks + globalRank + 1) * baseChunk
// This is consistent: rank r's sendcounts[j] == rank j's recvcounts[r].
// ---------------------------------------------------------------------------
TEST_P(DeviceAllToAllvParamTest, Imbalanced) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  if (!ctranDeviceAllToAllvSupport(comm.get())) {
    GTEST_SKIP() << "deviceAllToAllv not supported (requires all NVLink peers)";
  }

  const int nRanks = numRanks;
  const size_t baseChunk = 1024;

  // Compute per-peer send/recv counts
  std::vector<int64_t> h_sendcounts(nRanks);
  std::vector<int64_t> h_recvcounts(nRanks);
  size_t totalSend = 0;
  size_t totalRecv = 0;
  for (int j = 0; j < nRanks; j++) {
    h_sendcounts[j] =
        static_cast<int64_t>((globalRank * nRanks + j + 1) * baseChunk);
    h_recvcounts[j] =
        static_cast<int64_t>((j * nRanks + globalRank + 1) * baseChunk);
    totalSend += h_sendcounts[j];
    totalRecv += h_recvcounts[j];
  }

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, totalSend * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, totalRecv * sizeof(float)));

  // Fill send buffer: all elements = globalRank
  std::vector<float> h_send(totalSend, static_cast<float>(globalRank));
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf,
      h_send.data(),
      totalSend * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, totalRecv * sizeof(float)));

  int64_t* d_sendcounts = nullptr;
  int64_t* d_recvcounts = nullptr;
  CUDACHECK_TEST(cudaMalloc(&d_sendcounts, nRanks * sizeof(int64_t)));
  CUDACHECK_TEST(cudaMalloc(&d_recvcounts, nRanks * sizeof(int64_t)));
  CUDACHECK_TEST(cudaMemcpy(
      d_sendcounts,
      h_sendcounts.data(),
      nRanks * sizeof(int64_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recvcounts,
      h_recvcounts.data(),
      nRanks * sizeof(int64_t),
      cudaMemcpyHostToDevice));

  auto result = runAllToAllv(
      sendBuf, recvBuf, d_sendcounts, d_recvcounts, commFloat, comm.get());
  ASSERT_EQ(result, commSuccess);

  // Verify: chunk from rank j should contain all j's
  std::vector<float> h_recv(totalRecv);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv.data(),
      recvBuf,
      totalRecv * sizeof(float),
      cudaMemcpyDeviceToHost));

  size_t offset = 0;
  for (int j = 0; j < nRanks; j++) {
    for (int64_t k = 0; k < h_recvcounts[j]; k++) {
      EXPECT_EQ(h_recv[offset + k], static_cast<float>(j))
          << "Rank " << globalRank << ": chunk from rank " << j << " element "
          << k << " (recvcount=" << h_recvcounts[j] << ")";
    }
    offset += h_recvcounts[j];
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(d_sendcounts));
  CUDACHECK_TEST(cudaFree(d_recvcounts));
}

// ---------------------------------------------------------------------------
// BalancedMultiDim: uniform row counts with multiplier (multi-dim tensors)
// ---------------------------------------------------------------------------
TEST_P(DeviceAllToAllvParamTest, BalancedMultiDim) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  if (!ctranDeviceAllToAllvSupport(comm.get())) {
    GTEST_SKIP() << "deviceAllToAllv not supported (requires all NVLink peers)";
  }

  const int nRanks = numRanks;
  const size_t chunkRows = 1024;
  const size_t numCols = 4;
  const size_t totalElements = chunkRows * nRanks * numCols;

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

  std::vector<int64_t> h_counts(nRanks, static_cast<int64_t>(chunkRows));

  int64_t* d_sendcounts = nullptr;
  int64_t* d_recvcounts = nullptr;
  CUDACHECK_TEST(cudaMalloc(&d_sendcounts, nRanks * sizeof(int64_t)));
  CUDACHECK_TEST(cudaMalloc(&d_recvcounts, nRanks * sizeof(int64_t)));
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

  auto result = runAllToAllv(
      sendBuf,
      recvBuf,
      d_sendcounts,
      d_recvcounts,
      commFloat,
      comm.get(),
      static_cast<int64_t>(numCols),
      static_cast<int64_t>(numCols));
  ASSERT_EQ(result, commSuccess);

  std::vector<float> h_recv(totalElements);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv.data(),
      recvBuf,
      totalElements * sizeof(float),
      cudaMemcpyDeviceToHost));

  const size_t elementsPerPeer = chunkRows * numCols;
  for (int j = 0; j < nRanks; j++) {
    for (size_t k = 0; k < elementsPerPeer; k++) {
      EXPECT_EQ(h_recv[j * elementsPerPeer + k], static_cast<float>(j))
          << "Rank " << globalRank << ": segment " << j << " element " << k;
    }
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(d_sendcounts));
  CUDACHECK_TEST(cudaFree(d_recvcounts));
}

// ---------------------------------------------------------------------------
// ImbalancedMultiDim: variable row counts with multiplier
// ---------------------------------------------------------------------------
TEST_P(DeviceAllToAllvParamTest, ImbalancedMultiDim) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  if (!ctranDeviceAllToAllvSupport(comm.get())) {
    GTEST_SKIP() << "deviceAllToAllv not supported (requires all NVLink peers)";
  }

  const int nRanks = numRanks;
  const size_t baseRows = 1024;
  const size_t numCols = 4;

  // Row counts per peer (variable)
  std::vector<int64_t> h_sendRowCounts(nRanks);
  std::vector<int64_t> h_recvRowCounts(nRanks);
  size_t totalSendElems = 0;
  size_t totalRecvElems = 0;
  for (int j = 0; j < nRanks; j++) {
    h_sendRowCounts[j] =
        static_cast<int64_t>((globalRank * nRanks + j + 1) * baseRows);
    h_recvRowCounts[j] =
        static_cast<int64_t>((j * nRanks + globalRank + 1) * baseRows);
    totalSendElems += h_sendRowCounts[j] * numCols;
    totalRecvElems += h_recvRowCounts[j] * numCols;
  }

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, totalSendElems * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, totalRecvElems * sizeof(float)));

  std::vector<float> h_send(totalSendElems, static_cast<float>(globalRank));
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf,
      h_send.data(),
      totalSendElems * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, totalRecvElems * sizeof(float)));

  int64_t* d_sendcounts = nullptr;
  int64_t* d_recvcounts = nullptr;
  CUDACHECK_TEST(cudaMalloc(&d_sendcounts, nRanks * sizeof(int64_t)));
  CUDACHECK_TEST(cudaMalloc(&d_recvcounts, nRanks * sizeof(int64_t)));
  CUDACHECK_TEST(cudaMemcpy(
      d_sendcounts,
      h_sendRowCounts.data(),
      nRanks * sizeof(int64_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recvcounts,
      h_recvRowCounts.data(),
      nRanks * sizeof(int64_t),
      cudaMemcpyHostToDevice));

  auto result = runAllToAllv(
      sendBuf,
      recvBuf,
      d_sendcounts,
      d_recvcounts,
      commFloat,
      comm.get(),
      static_cast<int64_t>(numCols),
      static_cast<int64_t>(numCols));
  ASSERT_EQ(result, commSuccess);

  std::vector<float> h_recv(totalRecvElems);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv.data(),
      recvBuf,
      totalRecvElems * sizeof(float),
      cudaMemcpyDeviceToHost));

  size_t offset = 0;
  for (int j = 0; j < nRanks; j++) {
    size_t chunkElems = h_recvRowCounts[j] * numCols;
    for (size_t k = 0; k < chunkElems; k++) {
      EXPECT_EQ(h_recv[offset + k], static_cast<float>(j))
          << "Rank " << globalRank << ": chunk from rank " << j << " element "
          << k << " (recvRows=" << h_recvRowCounts[j] << ")";
    }
    offset += chunkElems;
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(d_sendcounts));
  CUDACHECK_TEST(cudaFree(d_recvcounts));
}

INSTANTIATE_TEST_SUITE_P(
    DeviceAllToAllv,
    DeviceAllToAllvParamTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "CudaGraph" : "Eager";
    });

// ---------------------------------------------------------------------------
// Non-parameterized: support check
// ---------------------------------------------------------------------------
class DeviceAllToAllvTest : public ctran::CtranDistTestFixture {};

TEST_F(DeviceAllToAllvTest, SupportedWithPipes) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  if (!comm->multiPeerTransport_) {
    GTEST_SKIP() << "MultiPeerTransport not available (requires NVLink peers)";
  }

  EXPECT_TRUE(ctranDeviceAllToAllvSupport(comm.get()));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DeviceAllToAllvEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
