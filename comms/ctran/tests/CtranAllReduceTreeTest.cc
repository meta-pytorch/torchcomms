// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "CtranUtUtils.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/tests/AllReduceTreeTestKernels.cuh"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

/**
 * Distributed CTRAN AllReduce correctness fixture.
 *
 * The fixture is algorithm-parameterized and validates execution status plus
 * the numerical reduction result. It intentionally avoids colltrace assertions
 * so the same test can cover CTREE and future AllReduce algorithm variants.
 */
class CtranAllReduceTest : public ctran::CtranDistTestFixture,
                           public CtranBaseTest {
 public:
  CtranAllReduceTest() = default;

  /** Identical-input repeats used to check bitwise deterministic output. */
  static constexpr int kDeterminismRepeats = 4;

  static void* allocateDeviceBuffer(size_t bytes) {
    void* ptr = nullptr;
    COMMCHECK_TEST(
        ctran::utils::commCuMemAlloc(
            &ptr,
            nullptr,
            ctran::utils::getCuMemAllocHandleType(),
            bytes,
            nullptr,
            "CtranAllReduceTest"));
    return ptr;
  }

  static void releaseDeviceBuffer(void* ptr) {
    COMMCHECK_TEST(ctran::utils::commCuMemFree(ptr));
  }

  static void launchInitData(
      void* buf,
      commDataType_t datatype,
      size_t count,
      int rank,
      int rep,
      cudaStream_t stream) {
    switch (datatype) {
      case commFloat32:
        launchInitDataKernel(
            static_cast<float*>(buf), count, rank, rep, stream);
        return;
      case commFloat16:
        launchInitDataKernel(
            static_cast<__half*>(buf), count, rank, rep, stream);
        return;
      default:
        ADD_FAILURE() << "unsupported datatype " << datatype;
        return;
    }
  }

  static void launchInitExpected(
      void* buf,
      commDataType_t datatype,
      size_t count,
      int nranks,
      int rep,
      cudaStream_t stream) {
    switch (datatype) {
      case commFloat32:
        launchInitExpectedKernel(
            static_cast<float*>(buf), count, nranks, rep, stream);
        return;
      case commFloat16:
        launchInitExpectedKernel(
            static_cast<__half*>(buf), count, nranks, rep, stream);
        return;
      default:
        ADD_FAILURE() << "unsupported datatype " << datatype;
        return;
    }
  }

  static double computeMaxDeltaForType(
      const void* actual,
      const void* expected,
      commDataType_t datatype,
      size_t count,
      cudaStream_t stream) {
    switch (datatype) {
      case commFloat32:
        return computeMaxDelta(
            static_cast<const float*>(actual),
            static_cast<const float*>(expected),
            count,
            stream);
      case commFloat16:
        return computeMaxDelta(
            static_cast<const __half*>(actual),
            static_cast<const __half*>(expected),
            count,
            stream);
      default:
        ADD_FAILURE() << "unsupported datatype " << datatype;
        return DBL_MAX;
    }
  }

  static uint64_t computeRawBitsChecksumForType(
      const void* data,
      commDataType_t datatype,
      size_t count,
      cudaStream_t stream) {
    switch (datatype) {
      case commFloat32:
        return computeRawBitsChecksum(
            static_cast<const float*>(data), count, stream);
      case commFloat16:
        return computeRawBitsChecksum(
            static_cast<const __half*>(data), count, stream);
      default:
        ADD_FAILURE() << "unsupported datatype " << datatype;
        return 0;
    }
  }

  static double thresholdForDatatype(commDataType_t datatype, int nranks) {
    const int rankScale = std::max(1, nranks - 1);
    switch (datatype) {
      case commFloat32:
        return 1e-5 * rankScale;
      case commFloat16:
        return 5e-3 * rankScale;
      default:
        return 0.0;
    }
  }

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
   * Run one `commSum` AllReduce case and validate max absolute error.
   */
  void runCorrectnessTest(
      enum NCCL_ALLREDUCE_ALGO algo,
      commDataType_t datatype,
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
          datatype,
          commSum,
          ctranComm.get(),
          testStream,
          /*timeout=*/std::nullopt);
      EXPECT_EQ(res, commSuccess);
      return;
    }

    const size_t bytes = count * commTypeSize(datatype);
    void* sendbuf = allocateDeviceBuffer(bytes);
    void* recvbuf = inPlace ? sendbuf : allocateDeviceBuffer(bytes);
    void* expectedBuf = allocateDeviceBuffer(bytes);

    // Initialize expected: sum of initData across all ranks
    launchInitExpected(
        expectedBuf,
        datatype,
        count,
        numRanks,
        /*rep=*/0,
        testStream);

    std::array<uint64_t, kDeterminismRepeats> checksums{};
    for (int repeat = 0; repeat < kDeterminismRepeats; ++repeat) {
      // Reinitialize identical inputs for each repeat. This keeps in-place
      // cases from feeding the previous reduction result into the next run.
      launchInitData(
          sendbuf,
          datatype,
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
          datatype,
          commSum,
          ctranComm.get(),
          testStream,
          /*timeout=*/std::nullopt);
      EXPECT_EQ(res, commSuccess);

      CUDACHECK_TEST(cudaStreamSynchronize(testStream));

      // Validate: max absolute error across all elements
      const void* resultBuf = inPlace ? sendbuf : recvbuf;
      double maxDelta = computeMaxDeltaForType(
          resultBuf, expectedBuf, datatype, count, testStream);

      double threshold = thresholdForDatatype(datatype, numRanks);
      ASSERT_LT(maxDelta, threshold)
          << "maxDelta=" << maxDelta << " threshold=" << threshold
          << " count=" << count << " rank=" << globalRank
          << " datatype=" << commDataTypeToString(datatype)
          << " inPlace=" << inPlace << " repeat=" << repeat;

      checksums[repeat] =
          computeRawBitsChecksumForType(resultBuf, datatype, count, testStream);
    }

    for (int repeat = 1; repeat < kDeterminismRepeats; ++repeat) {
      ASSERT_EQ(checksums[0], checksums[repeat])
          << "non-deterministic AllReduce output checksum for count=" << count
          << " rank=" << globalRank
          << " datatype=" << commDataTypeToString(datatype)
          << " inPlace=" << inPlace << " repeat=" << repeat;
    }

    verifyGpeLeak(ctranComm->ctran_.get());

    if (!inPlace) {
      releaseDeviceBuffer(recvbuf);
    }
    releaseDeviceBuffer(expectedBuf);
    releaseDeviceBuffer(sendbuf);
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
      default:
        return commInvalidArgument;
    }
  }

  /** CUDA stream used by buffer initialization, AllReduce, and validation. */
  cudaStream_t testStream{0};
  /** Communicator under test for this distributed rank. */
  std::unique_ptr<CtranComm> ctranComm{nullptr};
};

using AllReduceParam =
    std::tuple<enum NCCL_ALLREDUCE_ALGO, commDataType_t, size_t, bool>;

const std::vector<size_t>& allReduceElementCounts() {
  static const std::vector<size_t> counts{
      // Zero
      0,
      // Tiny
      1,
      7,
      16,
      // Small
      64,
      256,
      1024,
      // Medium
      4096,
      16384,
      65536,
      // Large
      262144,
      1048576,
      4194304,
      // Large + tail
      1048576 - 7,
      1048576 + 7,
      4194304 - 13,
      4194304 + 13,
  };
  return counts;
}

const std::vector<commDataType_t>& allReduceDataTypes() {
  static const std::vector<commDataType_t> datatypes{
      commFloat32,
      commFloat16,
  };
  return datatypes;
}

std::string dataTypeTestName(commDataType_t datatype) {
  switch (datatype) {
    case commFloat32:
      return "FP32";
    case commFloat16:
      return "FP16";
    default:
      return "UnsupportedType";
  }
}

std::string allReduceTestName(
    enum NCCL_ALLREDUCE_ALGO algo,
    commDataType_t datatype,
    size_t count,
    bool inPlace) {
  return allReduceAlgoName(algo) + "_" + dataTypeTestName(datatype) + "_" +
      std::to_string(count) + "elements_" +
      (inPlace ? "InPlace" : "OutOfPlace");
}

#if defined(CTRAN_ALLREDUCE_TEST_NVL_ONLY)
/**
 * NVL-only topology test using the default local topology.
 */
class NVL_ONLY : public CtranAllReduceTest,
                 public ::testing::WithParamInterface<AllReduceParam> {};

TEST_P(NVL_ONLY, Correctness) {
  auto [algo, datatype, count, inPlace] = GetParam();
  runCorrectnessTest(algo, datatype, count, inPlace);
}

inline std::string nvlTestName(
    const testing::TestParamInfo<NVL_ONLY::ParamType>& info) {
  auto [algo, datatype, count, inPlace] = info.param;
  return allReduceTestName(algo, datatype, count, inPlace);
}

INSTANTIATE_TEST_SUITE_P(
    CtranAllReduce,
    NVL_ONLY,
    ::testing::Combine(
        ::testing::Values(NCCL_ALLREDUCE_ALGO::ctree),
        ::testing::ValuesIn(allReduceDataTypes()),
        ::testing::ValuesIn(allReduceElementCounts()),
        ::testing::Bool()),
    nvlTestName);
#endif

#if defined(CTRAN_ALLREDUCE_TEST_IB_ONLY)
/**
 * IB-only topology test that disables local P2P and forces a no-local topology.
 */
class IB_ONLY : public CtranAllReduceTest,
                public ::testing::WithParamInterface<AllReduceParam> {
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
  auto [algo, datatype, count, inPlace] = GetParam();
  runCorrectnessTest(algo, datatype, count, inPlace);
}

inline std::string ibTestName(
    const testing::TestParamInfo<IB_ONLY::ParamType>& info) {
  auto [algo, datatype, count, inPlace] = info.param;
  return allReduceTestName(algo, datatype, count, inPlace);
}

INSTANTIATE_TEST_SUITE_P(
    CtranAllReduce,
    IB_ONLY,
    ::testing::Combine(
        ::testing::Values(NCCL_ALLREDUCE_ALGO::ctree),
        ::testing::ValuesIn(allReduceDataTypes()),
        ::testing::ValuesIn(allReduceElementCounts()),
        ::testing::Bool()),
    ibTestName);
#endif

#if defined(CTRAN_ALLREDUCE_TEST_HYBRID)
/**
 * Hybrid topology test that exercises NVL and IB paths together.
 */
class HYBRID : public CtranAllReduceTest,
               public ::testing::WithParamInterface<AllReduceParam> {};

TEST_P(HYBRID, Correctness) {
  auto [algo, datatype, count, inPlace] = GetParam();
  runCorrectnessTest(algo, datatype, count, inPlace);
}

inline std::string hybridTestName(
    const testing::TestParamInfo<HYBRID::ParamType>& info) {
  auto [algo, datatype, count, inPlace] = info.param;
  return allReduceTestName(algo, datatype, count, inPlace);
}

INSTANTIATE_TEST_SUITE_P(
    CtranAllReduce,
    HYBRID,
    ::testing::Combine(
        ::testing::Values(NCCL_ALLREDUCE_ALGO::ctree),
        ::testing::ValuesIn(allReduceDataTypes()),
        ::testing::ValuesIn(allReduceElementCounts()),
        ::testing::Bool()),
    hybridTestName);
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
