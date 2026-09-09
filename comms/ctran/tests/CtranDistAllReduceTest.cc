// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <thread>
#include <type_traits>
#include "comms/ctran/utils/CtranLogger.h"

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

// Reduce the value range to avoid integer overflow when running large count
constexpr size_t VAL_RANGE = 1024;
// Reduce the value range for commProd to avoid accumulated precision loss or
// numerical difference between CPU and GPU for floating points
constexpr size_t VAL_RANGE_PROD = 8;

namespace {

template <typename T>
float roundToLowPrecision(float value) {
  return static_cast<float>(static_cast<T>(value));
}

template <typename T>
inline constexpr bool kAcceptWidenedPreMulResult = false;

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
inline constexpr bool kAcceptWidenedPreMulResult<__nv_bfloat16> = true;
#endif

template <typename T>
std::vector<float> preMulSumCyclicResults(const std::vector<float>& inputs) {
  const float preMul =
      roundToLowPrecision<T>(static_cast<float>(1.0 / inputs.size()));
  std::vector<float> results;
  results.reserve((kAcceptWidenedPreMulResult<T> ? 2 : 1) * inputs.size());
  const auto addResult = [&](float result) {
    if (std::find(results.begin(), results.end(), result) == results.end()) {
      results.push_back(result);
    }
  };
  for (size_t start = 0; start < inputs.size(); ++start) {
    float roundedProductSum = 0.0f;
    float widenedProductSum = 0.0f;
    for (size_t offset = 0; offset < inputs.size(); ++offset) {
      const float input =
          roundToLowPrecision<T>(inputs[(start + offset) % inputs.size()]);
      const float scaled = roundToLowPrecision<T>(input * preMul);
      roundedProductSum = roundToLowPrecision<T>(roundedProductSum + scaled);
      widenedProductSum =
          roundToLowPrecision<T>(widenedProductSum + input * preMul);
    }
    addResult(roundedProductSum);
    if constexpr (kAcceptWidenedPreMulResult<T>) {
      addResult(widenedProductSum);
    }
  }
  return results;
}

} // namespace

template <typename TYPE>
class CtranAllReduceTest : public ctran::CtranDistTestFixture,
                           public CtranBaseTest {
 public:
  struct AllReduceTestOptions {
    std::vector<CtranMapperBackend> excludedBackends{CtranMapperBackend::NVL};
    std::optional<float> constantInput;
    std::vector<float> expectedConstantResults;
    double constantResultTolerance{0.05};
  };

  CtranAllReduceTest() = default;
  commDataType_t dt = ctran::getCommDataType<TYPE>();
  size_t bytes;
  size_t bufSize;
  void *sendbuf, *recvbuf;
  std::vector<TestMemSegment> segments;
  TYPE* hostbuf;

  void SetUp() override {
#ifdef CTRAN_TEST_SOCKET_ONLY_BACKEND
    setenv("NCCL_CTRAN_BACKENDS", "socket, nvl", 1);
#endif
    ctran::CtranDistTestFixture::SetUp();
    ctranComm = makeCtranComm();
    segments.clear();
  }

  void TearDown() override {
    ctran::CtranDistTestFixture::TearDown();
  }

  void memorySetUp(
      size_t count,
      TestInPlaceType inplace,
      commRedOp_t op,
      MemAllocType memType,
      std::optional<float> constantInput = std::nullopt) {
    sendbuf = recvbuf = nullptr;
    bytes = count * commTypeSize(dt);
    if (bytes < CTRAN_MIN_REGISTRATION_SIZE) {
      bytes = CTRAN_MIN_REGISTRATION_SIZE;
    }
    bufSize = bytes;

    CUDACHECK_TEST(cudaHostAlloc(&hostbuf, bytes, 0));
    for (size_t i = 0; i < count; i++) {
      auto val = i % VAL_RANGE + globalRank;
      if (constantInput.has_value()) {
        hostbuf[i] = static_cast<TYPE>(*constantInput);
      } else if (op == commProd) {
        // use smaller value range to avoid overflow or accumulated precision
        // loss for floating points
        hostbuf[i] = (TYPE)(val % VAL_RANGE_PROD);
      } else {
        hostbuf[i] = (TYPE)(val);
      }
    }

    sendbuf = prepareBuf(bytes, memType, segments);
    CUDACHECK_TEST(cudaMemcpy(sendbuf, hostbuf, bytes, cudaMemcpyDefault));

    if (inplace == kTestOutOfPlace) {
      recvbuf = prepareBuf(bytes, memType, segments);
      CUDACHECK_TEST(cudaMemcpy(recvbuf, hostbuf, bytes, cudaMemcpyDefault));
    } else {
      recvbuf = sendbuf;
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  void memoryCleanUp(MemAllocType memType) {
    CUDACHECK_TEST(cudaFreeHost(hostbuf));
    if (recvbuf != sendbuf) {
      releaseBuf(recvbuf, bytes, memType);
    }
    releaseBuf(sendbuf, bytes, memType);
  }

  void verifyResult(size_t count, commRedOp_t op) {
    std::vector<TYPE> observedVals(count, 117);
    FB_CUDACHECKIGNORE(cudaMemcpy(
        observedVals.data(),
        recvbuf,
        count * commTypeSize(dt),
        cudaMemcpyDefault));
    int error_count = 0;
    for (size_t i = 0; i < count; i++) {
      TYPE exp = (TYPE)0;
      size_t baseVal = i % VAL_RANGE;
      if (op == commSum) {
        exp = (TYPE)(baseVal * this->numRanks +
                     this->numRanks * (this->numRanks - 1) / 2);
      } else if (op == commProd) {
        exp = (TYPE)1;
        for (size_t j = 0; j < this->numRanks; j++) {
          exp *= TYPE((baseVal + j) % VAL_RANGE_PROD);
        }
      } else if (op == commMax) {
        exp = (TYPE)(baseVal + this->numRanks - 1);
      } else if (op == commMin) {
        exp = (TYPE)(baseVal);
      } else if (op == commAvg) {
        exp = static_cast<TYPE>(
            static_cast<float>(baseVal) +
            static_cast<float>(this->numRanks - 1) / 2.0f);
      }
      bool matches = observedVals[i] == exp;
      float expectedForLog = static_cast<float>(exp);
      if constexpr (
          std::is_same_v<TYPE, half>
#if defined(__CUDA_BF16_TYPES_EXIST__)
          || std::is_same_v<TYPE, __nv_bfloat16>
#endif
      ) {
        if (op == commAvg) {
          std::vector<float> inputs(this->numRanks);
          for (int rank = 0; rank < this->numRanks; ++rank) {
            inputs[rank] = static_cast<float>(baseVal + rank);
          }
          const auto expectedValues = preMulSumCyclicResults<TYPE>(inputs);
          expectedForLog = expectedValues.front();
          matches =
              std::find(
                  expectedValues.begin(),
                  expectedValues.end(),
                  static_cast<float>(observedVals[i])) != expectedValues.end();
        }
      }
      // log the first 3 errors
      if (error_count < 3) {
        EXPECT_TRUE(matches)
            << "  i=" << i << std::endl
            << "  observed=" << static_cast<float>(observedVals[i])
            << " expected=" << expectedForLog << std::endl
            << "  count=" << count << " on rank " << this->globalRank
            << std::endl;
      }
      // count errors
      if (!matches) {
        if (error_count < 20) {
          CTRAN_LOG_STREAM(WARN)
              << "error[" << error_count << "]: " << " data[" << i << "] "
              << static_cast<float>(observedVals[i]) << " vs exp "
              << expectedForLog;
        }
        error_count++;
      }
    }
    ASSERT_EQ(error_count, 0) << "  error count=" << count << " on rank "
                              << this->globalRank << std::endl;
  }

  void verifyConstantResult(
      size_t count,
      const std::vector<float>& expectedValues,
      double relativeTolerance = 0.05) {
    ASSERT_FALSE(expectedValues.empty());
    std::vector<TYPE> observedVals(count, 117);
    FB_CUDACHECKIGNORE(cudaMemcpy(
        observedVals.data(),
        recvbuf,
        count * commTypeSize(dt),
        cudaMemcpyDefault));

    std::vector<double> roundedExpectedValues;
    roundedExpectedValues.reserve(expectedValues.size());
    for (const float expectedValue : expectedValues) {
      roundedExpectedValues.push_back(
          static_cast<float>(static_cast<TYPE>(expectedValue)));
    }
    int errorCount = 0;
    for (size_t i = 0; i < count; i++) {
      const double observed = static_cast<float>(observedVals[i]);
      const bool matches = std::isfinite(observed) &&
          std::any_of(roundedExpectedValues.begin(),
                      roundedExpectedValues.end(),
                      [&](double expected) {
                        const double tolerance =
                            std::max(std::abs(expected), 1.0) *
                            relativeTolerance;
                        return relativeTolerance == 0.0
                            ? observed == expected
                            : std::abs(observed - expected) <= tolerance;
                      });
      if (!matches) {
        if (errorCount < 3) {
          EXPECT_TRUE(std::isfinite(observed)) << "  i=" << i;
          EXPECT_TRUE(matches)
              << "  i=" << i << " observed=" << observed
              << " first expected=" << roundedExpectedValues.front();
        }
        if (errorCount < 20) {
          CTRAN_LOG_STREAM(WARN)
              << "error[" << errorCount << "]: " << " data[" << i << "] "
              << observed << " vs first exp " << roundedExpectedValues.front();
        }
        errorCount++;
      }
    }
    ASSERT_EQ(errorCount, 0) << "  error count=" << errorCount << " on rank "
                             << this->globalRank << std::endl;
  }

  /* test given Allreduce function */
  void beginTest(
      commResult_t allreduceFunc(
          const void* sendbuff,
          void* recvbuff,
          size_t count,
          commDataType_t datatype,
          commRedOp_t redOp,
          CtranComm* comm,
          cudaStream_t stream,
          std::optional<std::chrono::milliseconds> timeout),
      enum NCCL_ALLREDUCE_ALGO algo,
      size_t count,
      TestInPlaceType inplace,
      commRedOp_t op,
      MemAllocType memType,
      const AllReduceTestOptions& options = {}) {
    if (memType == kCuMemAllocDisjoint && !NCCL_CTRAN_IB_DMABUF_ENABLE) {
      GTEST_SKIP() << "dmabuf is not supported, skip disjoint test";
    }

    memorySetUp(count, inplace, op, memType, options.constantInput);

    if (!ctranAllReduceSupport(ctranComm.get(), algo)) {
      GTEST_SKIP() << "ctranAllReduceSupport returns fails, skip test";
    }

    for (auto& segment : segments) {
      COMMCHECK_TEST(ctran::globalRegisterWithPtr(segment.ptr, segment.size));
    }

    ASSERT_TRUE(
        meta::comms::colltrace::testOnlyClearCollTraceRecords(ctranComm.get()));

    if (inplace == kTestInPlace) {
      auto res = allreduceFunc(
          recvbuf,
          recvbuf,
          count,
          dt,
          op,
          ctranComm.get(),
          testStream,
          /*timeout=*/std::nullopt);
      EXPECT_EQ(res, commSuccess);
    } else {
      auto res = allreduceFunc(
          sendbuf,
          recvbuf,
          count,
          dt,
          op,
          ctranComm.get(),
          testStream,
          /*timeout=*/std::nullopt);
      EXPECT_EQ(res, commSuccess);
    }

    FB_CUDACHECKIGNORE(cudaStreamSynchronize(testStream));

    if (options.constantInput.has_value()) {
      const auto expectedValues = options.expectedConstantResults.empty()
          ? std::vector<float>{*options.constantInput}
          : options.expectedConstantResults;
      verifyConstantResult(
          count, expectedValues, options.constantResultTolerance);
    } else {
      verifyResult(count, op);
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());
    // Sleep for a while to make sure all the colls are finished
    std::this_thread::sleep_for(std::chrono::seconds(2));

    ASSERT_NE(ctranComm->colltraceNew_, nullptr);
    auto dumpMap = ctran::dumpCollTrace(ctranComm.get());

    EXPECT_NE(dumpMap["CT_pastColls"], "[]");
    EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
    EXPECT_EQ(dumpMap["CT_currentColls"], "[]");

    auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
    EXPECT_EQ(pastCollsJson.size(), 1);

    auto lastColl = pastCollsJson[0];
    EXPECT_EQ(lastColl["opName"].asString(), "AllReduce");
    EXPECT_EQ(lastColl["count"].asInt(), count);
    EXPECT_THAT(
        lastColl["algoName"].asString(),
        testing::HasSubstr(allReduceAlgoName(algo)));

    // AllReduce uses kernel reduce not NVL iput
    verifyBackendsUsed(
        ctranComm->ctran_.get(),
        ctranComm->statex_.get(),
        kMemNcclMemAlloc,
        options.excludedBackends);
    verifyGpeLeak(ctranComm->ctran_.get());

    for (auto& segment : segments) {
      COMMCHECK_TEST(ctran::globalDeregisterWithPtr(segment.ptr, segment.size));
    }

    memoryCleanUp(memType);
  }

 protected:
  cudaStream_t testStream{0};
  std::unique_ptr<CtranComm> ctranComm{nullptr};
};

class CtranAllReduceTestParamUInt64
    : public CtranAllReduceTest<uint64_t>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {};

TEST_P(CtranAllReduceTestParamUInt64, AllReduceDirectUInt64) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceDirect,
      NCCL_ALLREDUCE_ALGO::ctdirect,
      count,
      inplace,
      op,
      memType);
}

class CtranAllReduceTestParamFp32
    : public CtranAllReduceTest<float>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {};

TEST_P(CtranAllReduceTestParamFp32, AllReduceDirectFp32) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceDirect,
      NCCL_ALLREDUCE_ALGO::ctdirect,
      count,
      inplace,
      op,
      memType);
}

// common parameters for all tests
auto testingValues = ::testing::Values(
    std::make_tuple(1, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(2, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(3, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(9, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(16, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(17, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(32, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8192, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024 * 1024, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 + 17,
        kTestOutOfPlace,
        commSum,
        kMemNcclMemAlloc),
    std::make_tuple(1, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(2, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(3, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(9, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(17, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(32, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8192, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024 * 1024, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024 * 1024 + 17, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commProd, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commProd, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commAvg, kMemNcclMemAlloc));

// common function to get test name from test parameter
inline std::string getTestName(
    const testing::TestParamInfo<CtranAllReduceTestParamUInt64::ParamType>&
        info) {
  return std::to_string(std::get<0>(info.param)) + "elements_" +
      testInPlaceTypeToStr(std::get<1>(info.param)) + "_" +
      commOpToString(std::get<2>(info.param)) + "_" +
      testMemAllocTypeToStr(std::get<3>(info.param));
}

// Tests for UInt64
INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllReduceTestParamUInt64,
    testingValues,
    getTestName);
// Tests for Float32
INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllReduceTestParamFp32,
    testingValues,
    getTestName);

// TODO: enable ctring test for nLocalRanks > 1 case, currently CtranIB connect
// to localRanks does not seem to work.
// TODO: enable tiny message sizes and ops other than commSum. This is a
// separate class because Ring does not support some sizes & ops yet
class CtranAllReduceRingTestParamUInt64
    : public CtranAllReduceTest<uint64_t>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {
 public:
  void SetUp() override {
    if (!ctran::isNolocalTopo()) {
      GTEST_SKIP() << "Ring AllReduce tests require nolocal topology; skip.";
    }
    CtranAllReduceTest::SetUp();
  }
};
TEST_P(CtranAllReduceRingTestParamUInt64, AllReduceRingUInt64) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      count,
      inplace,
      op,
      memType);
}

// TODO: enable tiny message sizes and ops other than commSum. This is a
// separate class because Ring does not support some sizes & ops yet
class CtranAllReduceRingTestParamFp32
    : public CtranAllReduceTest<float>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {
 public:
  void SetUp() override {
    if (!ctran::isNolocalTopo()) {
      GTEST_SKIP() << "Ring AllReduce tests require nolocal topology; skip.";
    }
    CtranAllReduceTest::SetUp();
  }
};
TEST_P(CtranAllReduceRingTestParamFp32, AllReduceRingFp32) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      count,
      inplace,
      op,
      memType);
}

class CtranAllReduceRingTestParamFloat16
    : public CtranAllReduceTest<half>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {
 public:
  void SetUp() override {
    if (!ctran::isNolocalTopo()) {
      GTEST_SKIP() << "Ring AllReduce tests require nolocal topology; skip.";
    }
    CtranAllReduceTest::SetUp();
  }
};

TEST_P(CtranAllReduceRingTestParamFloat16, AllReduceRingFloat16) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      count,
      inplace,
      op,
      memType);
}

class CtranAllReduceRingFloat16AvgOverflowTest
    : public CtranAllReduceTest<half>,
      public ::testing::WithParamInterface<TestInPlaceType> {
 public:
  void SetUp() override {
    if (!ctran::isNolocalTopo()) {
      GTEST_SKIP() << "Ring AllReduce tests require nolocal topology; skip.";
    }
    CtranAllReduceTest::SetUp();
  }
};

TEST_P(CtranAllReduceRingFloat16AvgOverflowTest, PreMultiplyStaysFinite) {
  ASSERT_GT(this->numRanks, 1);
  constexpr float kFloat16Max = 65504.0f;
  std::vector<float> inputs(this->numRanks);
  for (int rank = 0; rank < this->numRanks; ++rank) {
    const float fraction = 0.5f +
        0.25f * static_cast<float>(rank) /
            static_cast<float>(this->numRanks - 1);
    inputs[rank] = fraction * kFloat16Max;
  }
  const auto expectedValues = preMulSumCyclicResults<half>(inputs);
  ASSERT_TRUE(
      std::all_of(
          expectedValues.begin(), expectedValues.end(), [](float value) {
            return std::isfinite(value);
          }));
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      1024 * 1024 + 17,
      GetParam(),
      commAvg,
      kMemNcclMemAlloc,
      {
          .constantInput = inputs.at(this->globalRank),
          .expectedConstantResults = expectedValues,
          .constantResultTolerance = 0.0,
      });
}

TEST(CtranAllReduceRingFloat16NcclReferenceTest, TenRankMaximumCanOverflow) {
  constexpr float kFloat16Max = 65504.0f;
  const auto expectedValues =
      preMulSumCyclicResults<half>(std::vector<float>(10, kFloat16Max));
  ASSERT_EQ(expectedValues.size(), 1);
  EXPECT_TRUE(std::isinf(expectedValues.front()));
  EXPECT_GT(expectedValues.front(), 0.0f);
}

class CtranAllReduceRingFloat16NcclSemanticsTest
    : public CtranAllReduceTest<half> {
 public:
  void SetUp() override {
    if (!ctran::isNolocalTopo()) {
      GTEST_SKIP() << "Ring AllReduce tests require nolocal topology; skip.";
    }
    CtranAllReduceTest::SetUp();
  }
};

TEST_F(
    CtranAllReduceRingFloat16NcclSemanticsTest,
    UsesFloat16RoundedPreMulSemantics) {
  if (this->numRanks != 3) {
    GTEST_SKIP() << "This test requires exactly three ranks";
  }
  const std::vector<float> inputs = {0.0625f, 0.125f, 0.4375f};
  const auto expectedValues = preMulSumCyclicResults<half>(inputs);
  ASSERT_EQ(expectedValues.size(), 1);
  float inputSum = 0.0f;
  for (const float input : inputs) {
    inputSum += input;
  }
  const float postDivideResult =
      roundToLowPrecision<half>(inputSum / this->numRanks);
  ASSERT_NE(expectedValues.front(), postDivideResult);
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      16,
      kTestOutOfPlace,
      commAvg,
      kMemNcclMemAlloc,
      {
          .constantInput = inputs.at(this->globalRank),
          .expectedConstantResults = expectedValues,
          .constantResultTolerance = 0.0,
      });
}

#if defined(__CUDA_BF16_TYPES_EXIST__)
class CtranAllReduceRingTestParamBfloat16
    : public CtranAllReduceTest<__nv_bfloat16>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {
 public:
  void SetUp() override {
    if (!ctran::isNolocalTopo()) {
      GTEST_SKIP() << "Ring AllReduce tests require nolocal topology; skip.";
    }
    CtranAllReduceTest::SetUp();
  }
};

TEST_P(CtranAllReduceRingTestParamBfloat16, AllReduceRingBfloat16) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      count,
      inplace,
      op,
      memType);
}

class CtranAllReduceRingBfloat16AvgOverflowTest
    : public CtranAllReduceTest<__nv_bfloat16>,
      public ::testing::WithParamInterface<TestInPlaceType> {
 public:
  void SetUp() override {
    if (!ctran::isNolocalTopo()) {
      GTEST_SKIP() << "Ring AllReduce tests require nolocal topology; skip.";
    }
    CtranAllReduceTest::SetUp();
  }
};

TEST_P(CtranAllReduceRingBfloat16AvgOverflowTest, PreMultiplyStaysFinite) {
  ASSERT_GT(this->numRanks, 1);
  constexpr float kBfloat16Max = 3.38953139e38f;
  std::vector<float> inputs(this->numRanks);
  for (int rank = 0; rank < this->numRanks; ++rank) {
    const float fraction = 0.5f +
        0.25f * static_cast<float>(rank) /
            static_cast<float>(this->numRanks - 1);
    inputs[rank] = fraction * kBfloat16Max;
  }
  const auto expectedValues = preMulSumCyclicResults<__nv_bfloat16>(inputs);
  ASSERT_TRUE(
      std::all_of(
          expectedValues.begin(), expectedValues.end(), [](float value) {
            return std::isfinite(value);
          }));
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      1024 * 1024 + 17,
      GetParam(),
      commAvg,
      kMemNcclMemAlloc,
      {
          .constantInput = inputs.at(this->globalRank),
          .expectedConstantResults = expectedValues,
          .constantResultTolerance = 0.0,
      });
}

TEST(CtranAllReduceRingBfloat16NcclReferenceTest, TenRankMaximumCanOverflow) {
  constexpr float kBfloat16Max = 3.38953139e38f;
  const auto expectedValues = preMulSumCyclicResults<__nv_bfloat16>(
      std::vector<float>(10, kBfloat16Max));
  ASSERT_EQ(expectedValues.size(), 1);
  EXPECT_TRUE(std::isinf(expectedValues.front()));
  EXPECT_GT(expectedValues.front(), 0.0f);
}

class CtranAllReduceRingBfloat16NcclSemanticsTest
    : public CtranAllReduceTest<__nv_bfloat16> {
 public:
  void SetUp() override {
    if (!ctran::isNolocalTopo()) {
      GTEST_SKIP() << "Ring AllReduce tests require nolocal topology; skip.";
    }
    CtranAllReduceTest::SetUp();
  }
};

TEST_F(
    CtranAllReduceRingBfloat16NcclSemanticsTest,
    UsesBfloat16RoundedReciprocal) {
  if (this->numRanks != 3) {
    GTEST_SKIP() << "This test requires exactly three ranks";
  }
  constexpr float kInput = 0.4375f;
  const std::vector<float> inputs(this->numRanks, kInput);
  const auto expectedValues = preMulSumCyclicResults<__nv_bfloat16>(inputs);
  ASSERT_EQ(expectedValues.size(), 1);
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      16,
      kTestOutOfPlace,
      commAvg,
      kMemNcclMemAlloc,
      {
          .constantInput = kInput,
          .expectedConstantResults = expectedValues,
          .constantResultTolerance = 0.0,
      });
}
#endif

auto testingValuesRing = ::testing::Values(
    std::make_tuple(16, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(17, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(32, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8192, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024 * 1024, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 + 17,
        kTestOutOfPlace,
        commSum,
        kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(17, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(32, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8192, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024 * 1024, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024 * 1024 + 17, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commProd, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commProd, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(16, kTestOutOfPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestOutOfPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 + 15,
        kTestOutOfPlace,
        commAvg,
        kMemNcclMemAlloc),
    std::make_tuple(
        1024 * 1024 + 17,
        kTestOutOfPlace,
        commAvg,
        kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(8195, kTestInPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(1024 * 1024 + 15, kTestInPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(1024 * 1024 + 17, kTestInPlace, commAvg, kMemNcclMemAlloc));

// Tests for UInt64
INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllReduceRingTestParamUInt64,
    testingValuesRing,
    getTestName);
// Tests for Float32
INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllReduceRingTestParamFp32,
    testingValuesRing,
    getTestName);

auto testingValuesRingFloat16 = ::testing::Values(
    std::make_tuple(16, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(16, kTestOutOfPlace, commProd, kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commProd, kMemNcclMemAlloc),
    std::make_tuple(16, kTestOutOfPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(16, kTestOutOfPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(16, kTestOutOfPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commAvg, kMemNcclMemAlloc));

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllReduceRingTestParamFloat16,
    testingValuesRingFloat16,
    getTestName);

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllReduceRingFloat16AvgOverflowTest,
    ::testing::Values(kTestOutOfPlace, kTestInPlace));

#if defined(__CUDA_BF16_TYPES_EXIST__)
auto testingValuesRingBfloat16 = ::testing::Values(
    std::make_tuple(16, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(16, kTestOutOfPlace, commProd, kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commProd, kMemNcclMemAlloc),
    std::make_tuple(16, kTestOutOfPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(16, kTestOutOfPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(16, kTestOutOfPlace, commAvg, kMemNcclMemAlloc),
    std::make_tuple(16, kTestInPlace, commAvg, kMemNcclMemAlloc));

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllReduceRingTestParamBfloat16,
    testingValuesRingBfloat16,
    getTestName);

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllReduceRingBfloat16AvgOverflowTest,
    ::testing::Values(kTestOutOfPlace, kTestInPlace));
#endif

// =============================================================================
// Bi-directional AllGather tests for Ring algorithm
// These tests explicitly enable/disable bi-directional AG optimization to
// ensure both code paths are exercised separately.
// =============================================================================

// Test fixture with bi-directional AG explicitly disabled (simple kernel path)
class CtranAllReduceRingBidirAgDisabledTestFp32
    : public CtranAllReduceTest<float>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {
 public:
  void SetUp() override {
    if (!ctran::isNolocalTopo()) {
      GTEST_SKIP() << "Ring AllReduce tests require nolocal topology; skip.";
    }
    // Disable bi-directional AG optimization
    setenv("NCCL_CTRAN_ALLREDUCE_RING_BIDIR_AG_MAX_SIZE", "0", 1);
    ncclCvarInit();
    CtranAllReduceTest::SetUp();
  }
};

TEST_P(CtranAllReduceRingBidirAgDisabledTestFp32, AllReduceRingNoBidirFp32) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      count,
      inplace,
      op,
      memType);
}

// Test fixture with bi-directional AG explicitly enabled for all sizes
class CtranAllReduceRingBidirAgEnabledTestFp32
    : public CtranAllReduceTest<float>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {
 public:
  void SetUp() override {
    if (!ctran::isNolocalTopo()) {
      GTEST_SKIP() << "Ring AllReduce tests require nolocal topology; skip.";
    }
    // Enable bi-directional AG optimization for all message sizes
    setenv("NCCL_CTRAN_ALLREDUCE_RING_BIDIR_AG_MAX_SIZE", "-1", 1);
    ncclCvarInit();
    CtranAllReduceTest::SetUp();
  }
};

TEST_P(CtranAllReduceRingBidirAgEnabledTestFp32, AllReduceRingBidirFp32) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      count,
      inplace,
      op,
      memType);
}

// Test values covering various message sizes to exercise bidir AG code paths
auto testingValuesBidirAg = ::testing::Values(
    // Small messages (within default 4MB threshold)
    std::make_tuple(1024, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8192, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024 * 1024, kTestOutOfPlace, commSum, kMemNcclMemAlloc),
    // In-place variants
    std::make_tuple(1024, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(8192, kTestInPlace, commSum, kMemNcclMemAlloc),
    std::make_tuple(1024 * 1024, kTestInPlace, commSum, kMemNcclMemAlloc),
    // Different reduction operations
    std::make_tuple(8192, kTestOutOfPlace, commMax, kMemNcclMemAlloc),
    std::make_tuple(8192, kTestOutOfPlace, commMin, kMemNcclMemAlloc),
    std::make_tuple(8192, kTestOutOfPlace, commAvg, kMemNcclMemAlloc));

// Tests for bi-directional AG disabled (simple kernel)
INSTANTIATE_TEST_SUITE_P(
    CtranTestBidirAgDisabled,
    CtranAllReduceRingBidirAgDisabledTestFp32,
    testingValuesBidirAg,
    getTestName);

// Tests for bi-directional AG enabled for all sizes
INSTANTIATE_TEST_SUITE_P(
    CtranTestBidirAgEnabled,
    CtranAllReduceRingBidirAgEnabledTestFp32,
    testingValuesBidirAg,
    getTestName);

// =============================================================================
// TCPDM backend tests for AllReduceRing
// Requires multi-host setup with devmem-capable NICs.
// Run with: buck test <target> -c comms.hosts=<host1>,<host2>
// =============================================================================

#ifdef CTRAN_TEST_TCPDM_BACKEND

class CtranAllReduceRingTcpDmTestFp32
    : public CtranAllReduceTest<float>,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, commRedOp_t, MemAllocType>> {
 public:
  void SetUp() override {
    if (!getenv("ENABLE_TCPDM_TEST")) {
      GTEST_SKIP()
          << "TCPDM test requires ENABLE_TCPDM_TEST=1 and devmem-capable hosts";
    }

    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal", 1);
    // TODO: re-enable bidir AG for TCPDM once reverse recvKernElem is allocated
    setenv("NCCL_CTRAN_ALLREDUCE_RING_BIDIR_AG_MAX_SIZE", "0", 1);
    // Force numBlocks/blockSize for TCPDM unpack kernel
    setenv("NCCL_CTRAN_ALLREDUCE_RING_NUM_THREAD_BLOCKS", "16", 1);
    setenv("NCCL_CTRAN_ALLREDUCE_RING_THREAD_BLOCK_SIZE", "256", 1);

    // Detect number of local ranks (ppn) from MPI
    int worldSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    // With 2 hosts, ppn = worldSize / 2
    int numHosts = 2;
    int ppn = worldSize / numHosts;

    // PIX-optimal GPU/NIC assignment for GRANDTETON H100:
    //   beth3 → GPU 0, beth4 → GPU 1, beth0 → GPU 2, beth1 → GPU 3
    if (ppn >= 4) {
      setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3", 1);
      setenv("TCP_DEVMEM_IFNAME", "beth3,beth4,beth0,beth1", 1);
    } else if (ppn >= 2) {
      setenv("CUDA_VISIBLE_DEVICES", "2,3", 1);
      setenv("TCP_DEVMEM_IFNAME", "beth0,beth1", 1);
    } else {
      setenv("CUDA_VISIBLE_DEVICES", "2", 1);
      setenv("TCP_DEVMEM_IFNAME", "beth0", 1);
    }

    // TCPDM transport config
    setenv("TCP_DEVMEM_SKIP_AGENT", "1", 1);

    // NCCL bootstrap on beth7 (separate from data NICs)
    setenv("NCCL_SOCKET_IFNAME", "beth7", 1);
    setenv("NCCL_CLIENT_SOCKET_IFNAME", "beth7", 1);

    // NCCL ctran: select TCPDM backend
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    setenv("NCCL_CTRAN_BACKENDS", "tcpdm", 1);

    setenv("NCCL_CTRAN_UNPACK_NUM_THREAD_BLOCKS", "16", 1);
    setenv("NCCL_P2P_NET_CHUNKSIZE", "262144", 1);
    setenv("NCCL_BUFFSIZE", "4194304", 1);

    ncclCvarInit();
    CtranAllReduceTest::SetUp();
  }
};

TEST_P(CtranAllReduceRingTcpDmTestFp32, AllReduceRingTcpDmFp32) {
  const auto& [count, inplace, op, memType] = GetParam();
  beginTest(
      ctranAllReduceRing,
      NCCL_ALLREDUCE_ALGO::ctring,
      count,
      inplace,
      op,
      memType,
      {.excludedBackends = {CtranMapperBackend::NVL, CtranMapperBackend::IB}});
}
// Only add one test case as the current TCPDM backend does not support
// consecutive ctrantcpdm create and destroy in the single process.
auto testingValuesTcpDm = ::testing::Values(
    std::make_tuple(1024 * 1024, kTestOutOfPlace, commSum, kMemNcclMemAlloc));

INSTANTIATE_TEST_SUITE_P(
    CtranTestTcpDm,
    CtranAllReduceRingTcpDmTestFp32,
    testingValuesTcpDm,
    getTestName);

#endif

class CtranAllReduceIbTest : public CtranAllReduceTest<uint64_t> {
 public:
  CtranAllReduceIbTest() = default;

  void SetUp() override {
    setenv("NCCL_CTRAN_IB_QP_CONFIG_ALGO", "allreduce:131072,1,dqplb,8,192", 1);
    ncclCvarInit();
    CtranAllReduceTest::SetUp();
  }

  void TearDown() override {
    CtranAllReduceTest::TearDown();
  }
};

TEST_F(CtranAllReduceIbTest, AllReduceIbConfig) {
  ASSERT_NE(ctranComm.get(), nullptr) << "ctranComm should not be null";
  ASSERT_NE(ctranComm->ctran_, nullptr) << "ctran should not be null";

  if (ctranComm->ctran_->algo == nullptr) {
    GTEST_SKIP() << "No ctran algo found, skip test";
  }

  CtranIbConfig* ctranIbConfigPtr =
      ctranComm->ctran_->algo->getCollToVcConfig(CollType::ALLREDUCE);

  ASSERT_NE(ctranIbConfigPtr, nullptr)
      << "AllReduce IB config should not be null";

  // Verify the config values match the env var:
  // "allreduce:131072,1,dqplb,8,192"
  EXPECT_EQ(ctranIbConfigPtr->qpScalingTh, 131072);
  EXPECT_EQ(ctranIbConfigPtr->numQps, 1);
  EXPECT_EQ(ctranIbConfigPtr->vcMode, NCCL_CTRAN_IB_VC_MODE::dqplb);
  EXPECT_EQ(ctranIbConfigPtr->qpMsgs, 8);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
