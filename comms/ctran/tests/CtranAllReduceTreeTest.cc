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
    DeviceBuffer sendbuf = makeDeviceBuffer(bytes);
    DeviceBuffer recvbuf = inPlace ? nullptr : makeDeviceBuffer(bytes);
    DeviceBuffer expectedBuf = makeDeviceBuffer(bytes);
    void* const sendPtr = sendbuf.get();
    void* const recvPtr = inPlace ? sendPtr : recvbuf.get();

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

    for (int repeat = 1; repeat < kDeterminismRepeats; ++repeat) {
      ASSERT_EQ(checksums[0], checksums[repeat])
          << "non-deterministic AllReduce output checksum for count=" << count
          << " rank=" << globalRank
          << " datatype=" << commDataTypeToString(datatype)
          << " inPlace=" << inPlace << " repeat=" << repeat;
    }

    verifyGpeLeak(ctranComm->ctran_.get());
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
      case NCCL_ALLREDUCE_ALGO::cthierarchical_ring:
        return ctranAllReduceHierarchicalRing(
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
    };
#ifdef CTRAN_TEST_SOCKET_ONLY_BACKEND
    envs.emplace_back("NCCL_CTRAN_BACKENDS", "socket, nvl");
#endif
    return envs;
  }
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

INSTANTIATE_TEST_SUITE_P(
    HierRing,
    NVL_ONLY,
    ::testing::Combine(
        ::testing::Values(NCCL_ALLREDUCE_ALGO::cthierarchical_ring),
        ::testing::Values(commFloat32),
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
   * Extend the base CTREE overrides to force the inter-node path without
   * leaking topology overrides into later tests.
   */
  ctran::CtranEnvs envOverrides() const override {
    auto envs = CtranAllReduceTest::envOverrides();
    envs.emplace_back("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal");
    envs.emplace_back("NCCL_IGNORE_TOPO_LOAD_FAILURE", "1");
    envs.emplace_back("NCCL_P2P_DISABLE", "1");
    return envs;
  }
};

TEST_P(IB_ONLY, Correctness) {
  auto [algo, datatype, count, inPlace] = GetParam();
  runCorrectnessTest(algo, datatype, count, inPlace);
}

// Locks in the intentional no-fallback contract: cthierarchical_ring rejects
// unsupported reduction ops and datatypes with commInvalidArgument rather than
// silently mis-reducing or falling back (see AllReduceIbRing.cc). Runs its
// assertions exactly once: on the hierring fp32 count==1 out-of-place param at
// numRanks > 1. globalRank/numRanks are fixture members (see
// runCorrectnessTest).
TEST_P(IB_ONLY, HierRingRejectsUnsupported) {
  const auto algo = std::get<0>(GetParam());
  const auto datatype = std::get<1>(GetParam());
  const auto count = std::get<2>(GetParam());
  const auto inPlace = std::get<3>(GetParam());
  if (algo != NCCL_ALLREDUCE_ALGO::cthierarchical_ring || count != 1 ||
      datatype != commFloat32 || inPlace) {
    GTEST_SKIP()
        << "negative-path check runs once on hierring fp32 count==1 OOP";
  }
  // The (datatype, redOp) guards are only reachable at numRanks > 1; at
  // numRanks==1 the function returns commSuccess via the memcpy short-circuit
  // before the guards (e.g. on the IB_ONLY_1x1 target). numRanks is a fixture
  // member.
  if (numRanks < 2) {
    GTEST_SKIP() << "negative-path check requires numRanks > 1";
  }

  // Guards return commInvalidArgument before dereferencing buffers, so null
  // buffers are safe here (mirrors the zero-element no-op case above) -- no raw
  // CUDA allocation needed.

  // Unsupported reduction op (fp32 + commProd) -> explicit invalid argument.
  EXPECT_EQ(
      runAllReduce(
          NCCL_ALLREDUCE_ALGO::cthierarchical_ring,
          /*sendbuf=*/nullptr,
          /*recvbuf=*/nullptr,
          count,
          commFloat32,
          commProd,
          ctranComm.get(),
          testStream,
          /*timeout=*/std::nullopt),
      commInvalidArgument)
      << "rank=" << globalRank << " expected commProd to be rejected";

  // Unsupported datatype (commFloat64 + commSum) -> explicit invalid argument.
  // commFloat64 stays unsupported even after fp16 lands in Part 2.
  EXPECT_EQ(
      runAllReduce(
          NCCL_ALLREDUCE_ALGO::cthierarchical_ring,
          /*sendbuf=*/nullptr,
          /*recvbuf=*/nullptr,
          count,
          commFloat64,
          commSum,
          ctranComm.get(),
          testStream,
          /*timeout=*/std::nullopt),
      commInvalidArgument)
      << "rank=" << globalRank << " expected commFloat64 to be rejected";
}

// NOTE: the host-side IB-transport guards in ctranAllReduceHierarchicalRing
// (null multiPeerTransport_, lazy-connect mode, and non-P2P_IBGDA ring
// neighbor) are intentionally NOT exercised here -- they are defensive guards
// against misconfiguration. None is cleanly reachable from this distributed
// fixture: a lazy-connect comm (NCCL_CTRAN_IBGDA_LAZY_CONNECT=1) crashes during
// CtranComm teardown with unmaterialized peers (a separate comm-lifecycle issue
// unrelated to this algo), and forcing an IBRC/socket-only or null-transport
// ring neighbor is impractical in this harness. The guards are simple
// host-side type/null checks that return commInvalidArgument before the kernel
// dereferences transports[...].p2p_ib.ibgda; see AllReduceIbRing.cc.

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

INSTANTIATE_TEST_SUITE_P(
    HierRing,
    IB_ONLY,
    ::testing::Combine(
        ::testing::Values(NCCL_ALLREDUCE_ALGO::cthierarchical_ring),
        ::testing::Values(commFloat32),
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

INSTANTIATE_TEST_SUITE_P(
    HierRing,
    HYBRID,
    ::testing::Combine(
        ::testing::Values(NCCL_ALLREDUCE_ALGO::cthierarchical_ring),
        ::testing::Values(commFloat32),
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
