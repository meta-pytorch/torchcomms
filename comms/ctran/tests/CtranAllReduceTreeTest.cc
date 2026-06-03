// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/tests/AllReduceTreeTestKernels.cuh"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

/**
 * Distributed CTRAN AllReduce correctness fixture.
 *
 * The fixture is algorithm-parameterized and validates execution status plus
 * the numerical reduction result. It intentionally avoids colltrace assertions
 * so the same test can cover multiple algorithm implementations.
 */
class CtranAllReduceTest : public ctran::CtranDistTestFixture,
                           public CtranBaseTest {
 public:
  CtranAllReduceTest() = default;

  /** Identical-input repeats used to check bitwise deterministic output. */
  static constexpr int kDeterminismRepeats = 4;

  /**
   * Configure CTRAN AllReduce and Pipes before constructing the communicator.
   */
  void SetUp() override {
#ifdef CTRAN_TEST_SOCKET_ONLY_BACKEND
    setenv("NCCL_CTRAN_BACKENDS", "socket, nvl", 1);
#endif
    setenv("NCCL_ALLREDUCE_ALGO", "ctree", 1);
    setenv("NCCL_CTRAN_USE_PIPES", "1", 1);
    setenv("NCCL_CTRAN_IBGDA_SENDRECV_ENABLE", "1", 1);
    // Re-read CVAR-backed environment overrides before fixture setup.
    ncclCvarInit();
    ctran::CtranDistTestFixture::SetUp();
    ctranComm = makeCtranComm();
  }

  void TearDown() override {
    ctran::CtranDistTestFixture::TearDown();
  }

  /**
   * Run one FP32 `commSum` AllReduce case and validate max absolute error.
   */
  void runCorrectnessTest(
      enum NCCL_ALLREDUCE_ALGO algo,
      size_t count,
      bool inPlace) {
    // Check support before the zero-count no-op so disabled algorithms skip
    // consistently with non-empty cases.
    if (!ctranAllReduceSupport(ctranComm.get(), algo)) {
      GTEST_SKIP() << "ctranAllReduceSupport returns false for algo "
                   << allReduceAlgoName(algo) << ", skip test";
    }

    // Zero-element case: no-op path
    if (count == 0) {
      auto res = runAllReduce(
          algo,
          nullptr,
          nullptr,
          0,
          commFloat,
          commSum,
          ctranComm.get(),
          testStream,
          /*timeout=*/std::nullopt);
      EXPECT_EQ(res, commSuccess);
      return;
    }

    const size_t bytes = count * sizeof(float);
    const size_t allocBytes = bytes < CTRAN_MIN_REGISTRATION_SIZE
        ? CTRAN_MIN_REGISTRATION_SIZE
        : bytes;

    std::vector<TestMemSegment> segments;
    void* sendbuf = prepareBuf(allocBytes, kMemNcclMemAlloc, segments);
    void* recvbuf =
        inPlace ? sendbuf : prepareBuf(allocBytes, kMemNcclMemAlloc, segments);
    void* expectedBuf = prepareBuf(allocBytes, kMemNcclMemAlloc, segments);

    for (auto& segment : segments) {
      COMMCHECK_TEST(ctran::globalRegisterWithPtr(segment.ptr, segment.size));
    }

    // Initialize expected: sum of initData across all ranks
    launchInitExpectedKernel(
        static_cast<float*>(expectedBuf),
        count,
        numRanks,
        /*rep=*/0,
        testStream);

    std::array<uint64_t, kDeterminismRepeats> checksums{};
    for (int repeat = 0; repeat < kDeterminismRepeats; ++repeat) {
      // Reinitialize identical inputs for each repeat. This keeps in-place
      // cases from feeding the previous reduction result into the next run.
      launchInitDataKernel(
          static_cast<float*>(sendbuf),
          count,
          globalRank,
          /*rep=*/0,
          testStream);
      CUDACHECK_TEST(cudaStreamSynchronize(testStream));

      commResult_t res = runAllReduce(
          algo,
          sendbuf,
          recvbuf,
          count,
          commFloat,
          commSum,
          ctranComm.get(),
          testStream,
          /*timeout=*/std::nullopt);
      EXPECT_EQ(res, commSuccess);

      CUDACHECK_TEST(cudaStreamSynchronize(testStream));

      // Validate: max absolute error across all elements
      const float* resultBuf =
          inPlace ? static_cast<float*>(sendbuf) : static_cast<float*>(recvbuf);
      double maxDelta = computeMaxDelta(
          resultBuf, static_cast<float*>(expectedBuf), count, testStream);

      // Threshold: 1e-5 * max(1, numRanks - 1) accounts for FP32 accumulation
      // error. For single-rank (numRanks=1), use 1e-5 as minimum threshold.
      double threshold = 1e-5 * std::max(1, numRanks - 1);
      ASSERT_LT(maxDelta, threshold)
          << "maxDelta=" << maxDelta << " threshold=" << threshold
          << " count=" << count << " rank=" << globalRank
          << " inPlace=" << inPlace << " repeat=" << repeat;

      checksums[repeat] = computeRawBitsChecksum(resultBuf, count, testStream);
    }

    for (int repeat = 1; repeat < kDeterminismRepeats; ++repeat) {
      ASSERT_EQ(checksums[0], checksums[repeat])
          << "non-deterministic AllReduce output checksum for count=" << count
          << " rank=" << globalRank << " inPlace=" << inPlace
          << " repeat=" << repeat;
    }

    verifyGpeLeak(ctranComm->ctran_.get());

    for (auto& segment : segments) {
      COMMCHECK_TEST(ctran::globalDeregisterWithPtr(segment.ptr, segment.size));
    }

    if (!inPlace) {
      releaseBuf(recvbuf, allocBytes, kMemNcclMemAlloc);
    }
    releaseBuf(expectedBuf, allocBytes, kMemNcclMemAlloc);
    releaseBuf(sendbuf, allocBytes, kMemNcclMemAlloc);
  }

 protected:
  /** Dispatch one AllReduce call for the algorithm under test. */
  commResult_t runAllReduce(
      enum NCCL_ALLREDUCE_ALGO algo,
      const void* sendbuf,
      void* recvbuf,
      size_t count,
      commDataType_t datatype,
      commRedOp_t redOp,
      CtranComm* comm,
      cudaStream_t stream,
      std::optional<std::chrono::milliseconds> timeout) {
    switch (algo) {
      case NCCL_ALLREDUCE_ALGO::ctree:
        return ctranAllReduceTree(
            sendbuf, recvbuf, count, datatype, redOp, comm, stream, timeout);
      case NCCL_ALLREDUCE_ALGO::ctring:
        return ctranAllReduceRing(
            sendbuf, recvbuf, count, datatype, redOp, comm, stream, timeout);
      case NCCL_ALLREDUCE_ALGO::ctdirect:
        return ctranAllReduceDirect(
            sendbuf, recvbuf, count, datatype, redOp, comm, stream, timeout);
      case NCCL_ALLREDUCE_ALGO::pipesflatring:
        return ctranAllReduce(
            sendbuf,
            recvbuf,
            count,
            datatype,
            redOp,
            comm,
            stream,
            algo,
            timeout);
      default:
        return commInvalidArgument;
    }
  }

  /** CUDA stream used by buffer initialization, AllReduce, and validation. */
  cudaStream_t testStream{0};
  /** Communicator under test for this distributed rank. */
  std::unique_ptr<CtranComm> ctranComm{nullptr};
};

#if defined(CTRAN_ALLREDUCE_TEST_NVL_ONLY)
/**
 * NVL-only topology test using the default local topology.
 */
class NVL_ONLY : public CtranAllReduceTest,
                 public ::testing::WithParamInterface<
                     std::tuple<enum NCCL_ALLREDUCE_ALGO, size_t, bool>> {};

TEST_P(NVL_ONLY, Correctness) {
  auto [algo, count, inPlace] = GetParam();
  runCorrectnessTest(algo, count, inPlace);
}

inline std::string nvlTestName(
    const testing::TestParamInfo<NVL_ONLY::ParamType>& info) {
  auto [algo, count, inPlace] = info.param;
  return allReduceAlgoName(algo) + "_" + std::to_string(count) + "elements_" +
      (inPlace ? "InPlace" : "OutOfPlace");
}

INSTANTIATE_TEST_SUITE_P(
    CtranAllReduce,
    NVL_ONLY,
    ::testing::Combine(
        ::testing::Values(NCCL_ALLREDUCE_ALGO::ctree),
        ::testing::Values(
            // Zero
            (size_t)0,
            // Tiny
            (size_t)1,
            (size_t)7,
            (size_t)16,
            // Small
            (size_t)64,
            (size_t)256,
            (size_t)1024,
            // Medium
            (size_t)4096,
            (size_t)16384,
            (size_t)65536,
            // Large
            (size_t)262144,
            (size_t)1048576,
            (size_t)4194304,
            // Large + tail
            (size_t)(1048576 - 7),
            (size_t)(1048576 + 7),
            (size_t)(4194304 - 13),
            (size_t)(4194304 + 13)),
        ::testing::Bool()),
    nvlTestName);
#endif

#if defined(CTRAN_ALLREDUCE_TEST_IB_ONLY)
/**
 * IB-only topology test that disables local P2P and forces a no-local topology.
 */
class IB_ONLY : public CtranAllReduceTest,
                public ::testing::WithParamInterface<
                    std::tuple<enum NCCL_ALLREDUCE_ALGO, size_t, bool>> {
 public:
  /**
   * Force traffic through the inter-node path before communicator creation.
   */
  void SetUp() override {
    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal", 1);
    setenv("NCCL_IGNORE_TOPO_LOAD_FAILURE", "1", 1);
    setenv("NCCL_P2P_DISABLE", "1", 1);
    CtranAllReduceTest::SetUp();
  }

  void TearDown() override {
    unsetenv("NCCL_COMM_STATE_DEBUG_TOPO");
    unsetenv("NCCL_IGNORE_TOPO_LOAD_FAILURE");
    unsetenv("NCCL_P2P_DISABLE");
    CtranAllReduceTest::TearDown();
  }
};

TEST_P(IB_ONLY, Correctness) {
  auto [algo, count, inPlace] = GetParam();
  runCorrectnessTest(algo, count, inPlace);
}

inline std::string ibTestName(
    const testing::TestParamInfo<IB_ONLY::ParamType>& info) {
  auto [algo, count, inPlace] = info.param;
  return allReduceAlgoName(algo) + "_" + std::to_string(count) + "elements_" +
      (inPlace ? "InPlace" : "OutOfPlace");
}

INSTANTIATE_TEST_SUITE_P(
    CtranAllReduce,
    IB_ONLY,
    ::testing::Combine(
        ::testing::Values(
            NCCL_ALLREDUCE_ALGO::ctree,
            NCCL_ALLREDUCE_ALGO::pipesflatring),
        ::testing::Values(
            // Zero
            (size_t)0,
            // Tiny
            (size_t)1,
            (size_t)7,
            (size_t)16,
            // Small
            (size_t)64,
            (size_t)256,
            (size_t)1024,
            // Medium
            (size_t)4096,
            (size_t)16384,
            (size_t)65536,
            // Large
            (size_t)262144,
            (size_t)1048576,
            (size_t)4194304,
            // Large + tail
            (size_t)(1048576 - 7),
            (size_t)(1048576 + 7),
            (size_t)(4194304 - 13),
            (size_t)(4194304 + 13)),
        ::testing::Bool()),
    ibTestName);
#endif

#if defined(CTRAN_ALLREDUCE_TEST_HYBRID)
/**
 * Hybrid topology test that exercises NVL and IB paths together.
 */
class HYBRID : public CtranAllReduceTest,
               public ::testing::WithParamInterface<
                   std::tuple<enum NCCL_ALLREDUCE_ALGO, size_t, bool>> {};

TEST_P(HYBRID, Correctness) {
  auto [algo, count, inPlace] = GetParam();
  runCorrectnessTest(algo, count, inPlace);
}

inline std::string hybridTestName(
    const testing::TestParamInfo<HYBRID::ParamType>& info) {
  auto [algo, count, inPlace] = info.param;
  return allReduceAlgoName(algo) + "_" + std::to_string(count) + "elements_" +
      (inPlace ? "InPlace" : "OutOfPlace");
}

INSTANTIATE_TEST_SUITE_P(
    CtranAllReduce,
    HYBRID,
    ::testing::Combine(
        ::testing::Values(NCCL_ALLREDUCE_ALGO::ctree),
        ::testing::Values(
            (size_t)1,
            (size_t)4096,
            (size_t)1048576,
            (size_t)(1048576 - 7),
            (size_t)(1048576 + 7)),
        ::testing::Bool()),
    hybridTestName);
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
