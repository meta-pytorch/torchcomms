// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <memory>
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
class AllReduceTestSuite : public ctran::CtranDistTestFixture,
                           public CtranBaseTest {
 public:
  AllReduceTestSuite() = default;

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
            "AllReduceTestSuite"));
    return ptr;
  }

  struct DeviceBufferDeleter {
    void operator()(void* ptr) const {
      if (ptr != nullptr) {
        releaseDeviceBuffer(ptr);
      }
    }
  };

  using DeviceBuffer = std::unique_ptr<void, DeviceBufferDeleter>;

  static DeviceBuffer makeDeviceBuffer(size_t bytes) {
    return DeviceBuffer(allocateDeviceBuffer(bytes));
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
        // FP16 accumulation error grows with rank count, but keep the bound
        // tight enough to catch small bit-corruption regressions.
        return 1e-3 * rankScale;
      default:
        return 0.0;
    }
  }

  static void
  expectDeviceBytePattern(const void* ptr, size_t bytes, uint8_t expected) {
    if (bytes == 0) {
      return;
    }
    std::vector<uint8_t> host(bytes);
    CUDACHECK_TEST(cudaMemcpy(host.data(), ptr, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < bytes; ++i) {
      ASSERT_EQ(host[i], expected) << "guard byte mismatch at offset " << i;
    }
  }

  /**
   * Configure CTRAN AllReduce and Pipes before constructing the communicator.
   */
  void SetUp() override {
    ctran::CtranDistTestFixture::SetUp(envOverrides());
    ctranComm = makeCtranComm();
  }

  void TearDown() override {
    ctran::CtranDistTestFixture::TearDown();
    ncclCvarInit();
  }

  /**
   * Run one `commSum` AllReduce case and validate max absolute error.
   */
  void runCorrectnessTest(
      enum NCCL_ALLREDUCE_ALGO algo,
      commDataType_t datatype,
      size_t count,
      bool inPlace,
      size_t bufferOffsetBytes = 0) {
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

    constexpr uint8_t kGuardPattern = 0xA5;
    const size_t bytes = count * commTypeSize(datatype);
    const size_t allocBytes = bytes + bufferOffsetBytes;
    DeviceBuffer sendbuf = makeDeviceBuffer(allocBytes);
    DeviceBuffer recvbuf = inPlace ? nullptr : makeDeviceBuffer(allocBytes);
    DeviceBuffer expectedBuf = makeDeviceBuffer(bytes);
    CUDACHECK_TEST(cudaMemset(sendbuf.get(), kGuardPattern, allocBytes));
    if (!inPlace) {
      CUDACHECK_TEST(cudaMemset(recvbuf.get(), kGuardPattern, allocBytes));
    }
    void* const sendPtr = static_cast<char*>(sendbuf.get()) + bufferOffsetBytes;
    void* const recvPtr = inPlace
        ? sendPtr
        : static_cast<char*>(recvbuf.get()) + bufferOffsetBytes;

    // Initialize expected: sum of initData across all ranks
    launchInitExpected(
        expectedBuf.get(),
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
          sendPtr,
          datatype,
          count,
          globalRank,
          /*rep=*/0,
          testStream);
      CUDACHECK_TEST(cudaStreamSynchronize(testStream));

      commResult_t res = runAllReduce(
          algo,
          sendPtr,
          recvPtr,
          count,
          datatype,
          commSum,
          ctranComm.get(),
          testStream,
          /*timeout=*/std::nullopt);
      EXPECT_EQ(res, commSuccess);

      CUDACHECK_TEST(cudaStreamSynchronize(testStream));

      // Validate: max absolute error across all elements
      const void* resultBuf = inPlace ? sendPtr : recvPtr;
      double maxDelta = computeMaxDeltaForType(
          resultBuf, expectedBuf.get(), datatype, count, testStream);

      double threshold = thresholdForDatatype(datatype, numRanks);
      ASSERT_LT(maxDelta, threshold)
          << "maxDelta=" << maxDelta << " threshold=" << threshold
          << " count=" << count << " rank=" << globalRank
          << " datatype=" << commDataTypeToString(datatype)
          << " inPlace=" << inPlace << " repeat=" << repeat;

      checksums[repeat] =
          computeRawBitsChecksumForType(resultBuf, datatype, count, testStream);
    }

    CUDACHECK_TEST(cudaStreamSynchronize(testStream));
    expectDeviceBytePattern(sendbuf.get(), bufferOffsetBytes, kGuardPattern);
    if (!inPlace) {
      expectDeviceBytePattern(recvbuf.get(), bufferOffsetBytes, kGuardPattern);
    }

    for (int repeat = 1; repeat < kDeterminismRepeats; ++repeat) {
      ASSERT_EQ(checksums[0], checksums[repeat])
          << "non-deterministic AllReduce output checksum for count=" << count
          << " rank=" << globalRank
          << " datatype=" << commDataTypeToString(datatype)
          << " inPlace=" << inPlace << " repeat=" << repeat;
    }

    verifyGpeLeak(ctranComm->ctran_.get());
  }

  void runCorrectnessSweep(
      enum NCCL_ALLREDUCE_ALGO algo,
      commDataType_t datatype,
      const std::vector<size_t>& counts,
      bool inPlace,
      size_t bufferOffsetBytes = 0) {
    for (size_t count : counts) {
      runCorrectnessTest(algo, datatype, count, inPlace, bufferOffsetBytes);
    }
  }

  void runOffsetCorrectnessTest(
      enum NCCL_ALLREDUCE_ALGO algo,
      commDataType_t datatype,
      size_t offsetElements,
      bool inPlace) {
    const size_t typeBytes = commTypeSize(datatype);
    const size_t count = 100 / typeBytes;
    const size_t offsetBytes = offsetElements * typeBytes;
    runCorrectnessTest(algo, datatype, count, inPlace, offsetBytes);
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

  /**
   * Return per-test environment overrides that must be visible before
   * communicator construction and restored after fixture teardown.
   */
  virtual ctran::CtranEnvs envOverrides() const {
    ctran::CtranEnvs envs{
        {"NCCL_ALLREDUCE_ALGO", "ctree"},
        {"NCCL_CTRAN_USE_PIPES", "1"},
        {"NCCL_CTRAN_IBGDA_SENDRECV_ENABLE", "1"},
        {"NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE", "33554432"},
        // Keep setup cost bounded across the parameterized tree sweep.
        {"NCCL_CTRAN_MAX_NBLOCKS", "8"},
        {"NCCL_CTRAN_IB_MAX_GROUPS", "16"},
    };
#ifdef CTRAN_TEST_SOCKET_ONLY_BACKEND
    envs.emplace_back("NCCL_CTRAN_BACKENDS", "socket, nvl");
#endif
    return envs;
  }
};

using AllReduceParam =
    std::tuple<enum NCCL_ALLREDUCE_ALGO, commDataType_t, size_t, bool>;
using AllReduceMixedOrderParam =
    std::tuple<enum NCCL_ALLREDUCE_ALGO, commDataType_t, bool>;
using AllReduceOffsetParam =
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

const std::vector<size_t>& allReduceMixedOrderElementCounts() {
  static const std::vector<size_t> counts{
      1,
      1048576 + 7,
      7,
      65536,
      16,
      1048576 - 7,
  };
  return counts;
}

const std::vector<size_t>& allReduceOffsetElementOffsets() {
  static const std::vector<size_t> offsets{
      1,
  };
  return offsets;
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

std::string allReduceMixedOrderTestName(
    enum NCCL_ALLREDUCE_ALGO algo,
    commDataType_t datatype,
    bool inPlace) {
  return allReduceAlgoName(algo) + "_" + dataTypeTestName(datatype) +
      "_MixedOrder_" + (inPlace ? "InPlace" : "OutOfPlace");
}

std::string allReduceOffsetTestName(
    enum NCCL_ALLREDUCE_ALGO algo,
    commDataType_t datatype,
    size_t offsetElements,
    bool inPlace) {
  return allReduceAlgoName(algo) + "_" + dataTypeTestName(datatype) +
      "_100B_Offset" + std::to_string(offsetElements) + "Element_" +
      (inPlace ? "InPlace" : "OutOfPlace");
}

inline std::string mixedOrderTestName(
    const testing::TestParamInfo<AllReduceMixedOrderParam>& info) {
  auto [algo, datatype, inPlace] = info.param;
  return allReduceMixedOrderTestName(algo, datatype, inPlace);
}

inline std::string offsetTestName(
    const testing::TestParamInfo<AllReduceOffsetParam>& info) {
  auto [algo, datatype, offsetElements, inPlace] = info.param;
  return allReduceOffsetTestName(algo, datatype, offsetElements, inPlace);
}

inline std::string correctnessTestName(
    const testing::TestParamInfo<AllReduceParam>& info) {
  auto [algo, datatype, count, inPlace] = info.param;
  return allReduceTestName(algo, datatype, count, inPlace);
}

#if defined(CTRAN_ALLREDUCE_TEST_NVL_ONLY) || \
    defined(CTRAN_ALLREDUCE_TEST_IB_ONLY) ||  \
    defined(CTRAN_ALLREDUCE_TEST_HYBRID)
class CtranAllReduceTopologyTest : public AllReduceTestSuite {
 public:
  ctran::CtranEnvs envOverrides() const override {
    auto envs = AllReduceTestSuite::envOverrides();
#if defined(CTRAN_ALLREDUCE_TEST_IB_ONLY)
    envs.emplace_back("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal");
    envs.emplace_back("NCCL_IGNORE_TOPO_LOAD_FAILURE", "1");
    envs.emplace_back("NCCL_P2P_DISABLE", "1");
#endif
    return envs;
  }
};

class CtranAllReduceCorrectnessTest
    : public CtranAllReduceTopologyTest,
      public ::testing::WithParamInterface<AllReduceParam> {};
class CtranAllReduceMixedOrderTest
    : public CtranAllReduceTopologyTest,
      public ::testing::WithParamInterface<AllReduceMixedOrderParam> {};
class CtranAllReduceOffsetTest
    : public CtranAllReduceTopologyTest,
      public ::testing::WithParamInterface<AllReduceOffsetParam> {};

TEST_P(CtranAllReduceCorrectnessTest, Correctness) {
  auto [algo, datatype, count, inPlace] = GetParam();
  runCorrectnessTest(algo, datatype, count, inPlace);
}

TEST_P(CtranAllReduceMixedOrderTest, Correctness) {
  auto [algo, datatype, inPlace] = GetParam();
  runCorrectnessSweep(
      algo, datatype, allReduceMixedOrderElementCounts(), inPlace);
}

TEST_P(CtranAllReduceOffsetTest, Correctness) {
  auto [algo, datatype, offsetElements, inPlace] = GetParam();
  runOffsetCorrectnessTest(algo, datatype, offsetElements, inPlace);
}

INSTANTIATE_TEST_SUITE_P(
    CtranAllReduce,
    CtranAllReduceCorrectnessTest,
    ::testing::Combine(
        ::testing::Values(NCCL_ALLREDUCE_ALGO::ctree),
        ::testing::ValuesIn(allReduceDataTypes()),
        ::testing::ValuesIn(allReduceElementCounts()),
        ::testing::Bool()),
    correctnessTestName);

INSTANTIATE_TEST_SUITE_P(
    CtranAllReduce,
    CtranAllReduceMixedOrderTest,
    ::testing::Combine(
        ::testing::Values(NCCL_ALLREDUCE_ALGO::ctree),
        ::testing::ValuesIn(allReduceDataTypes()),
        ::testing::Bool()),
    mixedOrderTestName);

INSTANTIATE_TEST_SUITE_P(
    CtranAllReduce,
    CtranAllReduceOffsetTest,
    ::testing::Combine(
        ::testing::Values(NCCL_ALLREDUCE_ALGO::ctree),
        ::testing::ValuesIn(allReduceDataTypes()),
        ::testing::ValuesIn(allReduceOffsetElementOffsets()),
        ::testing::Bool()),
    offsetTestName);
#else
#error "Define one CTRAN AllReduce topology: NVL_ONLY, IB_ONLY, or HYBRID"
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
