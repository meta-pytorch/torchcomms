// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Iterated stress tests for P2pNvlTransportDevice APIs.

#include "PipesTransportIteratedTest.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "IteratedTestHelpers.hpp"
#include "PipesTransportIteratedTestKernels.cuh"
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"

using namespace torchcomms::device::test;

// =============================================================================
// Setup / Teardown
// =============================================================================

void PipesTransportIteratedTest::SetUp() {
  if (!shouldRunIteratedTest()) {
    GTEST_SKIP()
        << "Skipping iterated tests (RUN_DEVICE_ITERATED_TEST not set)";
  }
  const char* pipes_env = getenv("RUN_PIPES_DEVICE_API_TEST");
  if (!pipes_env) {
    GTEST_SKIP()
        << "Skipping Pipes iterated tests (RUN_PIPES_DEVICE_API_TEST not set)";
  }

  config_ = parseIteratedTestConfig();
  wrapper_ = std::make_unique<TorchCommTestWrapper>();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_index_ = rank_ % at::cuda::device_count();

  ASSERT_GE(num_ranks_, 2) << "Need at least 2 ranks for transport tests";

  // Get device transport handle
  auto ncclx = std::dynamic_pointer_cast<torch::comms::TorchCommNCCLX>(
      torchcomm_->getBackendImpl());
  ASSERT_NE(ncclx, nullptr) << "Backend is not TorchCommNCCLX";

  auto handle_ptr = ncclx->get_device_transport();
  ASSERT_NE(handle_ptr, 0) << "get_device_transport returned null";

  auto copy_err = cudaMemcpy(
      &handle_,
      reinterpret_cast<void*>(handle_ptr),
      sizeof(comms::pipes::MultiPeerDeviceHandle),
      cudaMemcpyDeviceToHost);
  ASSERT_EQ(copy_err, cudaSuccess) << "Failed to copy transport handle";
  ASSERT_EQ(handle_.myRank, rank_);

  if (handle_.numNvlPeers == 0) {
    GTEST_SKIP() << "No NVL peers available — transport tests require NVLink";
  }

  // Ring pattern: peer is the other rank
  peer_ = (rank_ == 0) ? 1 : 0;
}

void PipesTransportIteratedTest::TearDown() {
  torchcomm_.reset();
  wrapper_.reset();
}

// =============================================================================
// Helpers
// =============================================================================

namespace {

void checkKernelResults(
    int* d_results,
    int iterations,
    const std::string& tag) {
  std::vector<int> h_results(iterations);
  cudaMemcpy(
      h_results.data(),
      d_results,
      iterations * sizeof(int),
      cudaMemcpyDeviceToHost);
  for (int i = 0; i < iterations; i++) {
    ASSERT_EQ(h_results[i], 1)
        << tag << ": verification failed at iteration " << i;
  }
}

const char* scopeNameForThreads(int num_threads) {
  return (num_threads >= 256) ? "BLOCK" : "WARP";
}

} // namespace

// =============================================================================
// Test Implementations
// =============================================================================

void PipesTransportIteratedTest::testIteratedSendRecv(
    size_t msg_bytes,
    int num_threads) {
  size_t count = msg_bytes / sizeof(float);
  if (count == 0) {
    count = 1;
  }
  int iterations = config_.num_iterations;

  SCOPED_TRACE(
      ::testing::Message() << "TransportIteratedSendRecv msg="
                           << formatBytes(msg_bytes)
                           << " scope=" << scopeNameForThreads(num_threads)
                           << " iters=" << iterations);

  float* d_buf = nullptr;
  ASSERT_EQ(cudaMalloc(&d_buf, count * sizeof(float)), cudaSuccess);

  int* d_results = nullptr;
  ASSERT_EQ(cudaMalloc(&d_results, iterations * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_results, 0, iterations * sizeof(int)), cudaSuccess);

  torchcomm_->barrier(false);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchTransportIteratedSendRecvKernel(
        handle_,
        d_buf,
        count,
        peer_,
        iterations,
        num_threads,
        d_results,
        stream.stream());
  }
  stream.synchronize();

  checkKernelResults(
      d_results,
      iterations,
      "TransportSendRecv(" + formatBytes(msg_bytes) + "," +
          scopeNameForThreads(num_threads) + ")");

  cudaFree(d_results);
  cudaFree(d_buf);
  torchcomm_->barrier(false);
}

void PipesTransportIteratedTest::testIteratedSignal(int num_threads) {
  int iterations = config_.num_iterations;

  SCOPED_TRACE(
      ::testing::Message() << "TransportIteratedSignal scope="
                           << scopeNameForThreads(num_threads));

  torchcomm_->barrier(false);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchTransportIteratedSignalKernel(
        handle_, peer_, iterations, num_threads, stream.stream());
  }
  stream.synchronize();
  torchcomm_->barrier(false);
}

void PipesTransportIteratedTest::testIteratedCombined(
    size_t msg_bytes,
    int num_threads) {
  size_t count = msg_bytes / sizeof(float);
  if (count == 0) {
    count = 1;
  }
  int iterations = config_.num_iterations;

  SCOPED_TRACE(
      ::testing::Message() << "TransportIteratedCombined msg="
                           << formatBytes(msg_bytes)
                           << " scope=" << scopeNameForThreads(num_threads));

  float* d_buf = nullptr;
  ASSERT_EQ(cudaMalloc(&d_buf, count * sizeof(float)), cudaSuccess);

  int* d_results = nullptr;
  ASSERT_EQ(cudaMalloc(&d_results, iterations * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_results, 0, iterations * sizeof(int)), cudaSuccess);

  torchcomm_->barrier(false);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchTransportIteratedCombinedKernel(
        handle_,
        d_buf,
        count,
        peer_,
        iterations,
        num_threads,
        d_results,
        stream.stream());
  }
  stream.synchronize();

  checkKernelResults(
      d_results,
      iterations,
      "TransportCombined(" + formatBytes(msg_bytes) + "," +
          scopeNameForThreads(num_threads) + ")");

  cudaFree(d_results);
  cudaFree(d_buf);
  torchcomm_->barrier(false);
}

void PipesTransportIteratedTest::testIteratedLl128(size_t nbytes) {
  int iterations = config_.num_iterations;

  // LL128 requires 16-byte alignment and multiple-of-16 size
  ASSERT_EQ(nbytes % 16, 0u) << "LL128 requires nbytes multiple of 16";

  SCOPED_TRACE(
      ::testing::Message() << "TransportIteratedLl128 nbytes="
                           << formatBytes(nbytes));

  char* d_buf = nullptr;
  ASSERT_EQ(cudaMalloc(&d_buf, nbytes), cudaSuccess);

  int* d_results = nullptr;
  ASSERT_EQ(cudaMalloc(&d_results, iterations * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_results, 0, iterations * sizeof(int)), cudaSuccess);

  torchcomm_->barrier(false);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchTransportIteratedLl128Kernel(
        handle_, d_buf, nbytes, peer_, iterations, d_results, stream.stream());
  }
  stream.synchronize();

  checkKernelResults(
      d_results, iterations, "TransportLl128(" + formatBytes(nbytes) + ")");

  cudaFree(d_results);
  cudaFree(d_buf);
  torchcomm_->barrier(false);
}

// =============================================================================
// Parameterized Test Registrations
// =============================================================================

// --- SendRecv: parameterized by (msg_bytes, num_threads) ---

struct TransportSendRecvParam {
  size_t msg_bytes;
  int num_threads; // 32 = WARP, 256 = BLOCK
};

class TransportIteratedSendRecvTest
    : public PipesTransportIteratedTest,
      public ::testing::WithParamInterface<TransportSendRecvParam> {};

TEST_P(TransportIteratedSendRecvTest, SendRecv) {
  testIteratedSendRecv(GetParam().msg_bytes, GetParam().num_threads);
}

INSTANTIATE_TEST_SUITE_P(
    IteratedSendRecv,
    TransportIteratedSendRecvTest,
    ::testing::Values(
        // WARP scope (32 threads)
        TransportSendRecvParam{1024, 32}, // 1KB
        TransportSendRecvParam{1048576, 32}, // 1MB
        TransportSendRecvParam{16777216, 32}, // 16MB
        TransportSendRecvParam{536870912, 32}, // 512MB
        // BLOCK scope (256 threads)
        TransportSendRecvParam{1024, 256}, // 1KB
        TransportSendRecvParam{1048576, 256}, // 1MB
        TransportSendRecvParam{16777216, 256}, // 16MB
        TransportSendRecvParam{536870912, 256} // 512MB
        ),
    [](const ::testing::TestParamInfo<TransportSendRecvParam>& info) {
      return std::to_string(info.param.msg_bytes) + "B_" +
          std::string(info.param.num_threads >= 256 ? "BLOCK" : "WARP");
    });

// --- Signal: parameterized by num_threads ---

class TransportIteratedSignalTest : public PipesTransportIteratedTest,
                                    public ::testing::WithParamInterface<int> {
};

TEST_P(TransportIteratedSignalTest, Signal) {
  testIteratedSignal(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    IteratedSignal,
    TransportIteratedSignalTest,
    ::testing::Values(32, 256),
    [](const ::testing::TestParamInfo<int>& info) {
      return std::string(info.param >= 256 ? "BLOCK" : "WARP");
    });

// --- Combined: parameterized by (msg_bytes, num_threads) ---

struct TransportCombinedParam {
  size_t msg_bytes;
  int num_threads;
};

class TransportIteratedCombinedTest
    : public PipesTransportIteratedTest,
      public ::testing::WithParamInterface<TransportCombinedParam> {};

TEST_P(TransportIteratedCombinedTest, Combined) {
  testIteratedCombined(GetParam().msg_bytes, GetParam().num_threads);
}

INSTANTIATE_TEST_SUITE_P(
    IteratedCombined,
    TransportIteratedCombinedTest,
    ::testing::Values(
        // WARP scope
        TransportCombinedParam{1024, 32},
        TransportCombinedParam{1048576, 32},
        TransportCombinedParam{536870912, 32},
        // BLOCK scope
        TransportCombinedParam{1024, 256},
        TransportCombinedParam{1048576, 256},
        TransportCombinedParam{536870912, 256}),
    [](const ::testing::TestParamInfo<TransportCombinedParam>& info) {
      return std::to_string(info.param.msg_bytes) + "B_" +
          std::string(info.param.num_threads >= 256 ? "BLOCK" : "WARP");
    });

// --- LL128: parameterized by nbytes (warp-only) ---

class TransportIteratedLl128Test
    : public PipesTransportIteratedTest,
      public ::testing::WithParamInterface<size_t> {};

TEST_P(TransportIteratedLl128Test, Ll128) {
  testIteratedLl128(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    IteratedLl128,
    TransportIteratedLl128Test,
    ::testing::Values(
        static_cast<size_t>(1024), // 1KB
        static_cast<size_t>(65536)), // 64KB
    [](const ::testing::TestParamInfo<size_t>& info) {
      return std::to_string(info.param) + "B";
    });
