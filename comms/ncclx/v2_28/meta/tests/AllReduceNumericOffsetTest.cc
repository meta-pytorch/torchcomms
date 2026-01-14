// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <stdlib.h>
#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <comm.h>
#include <nccl.h>
#include "checks.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

#include "meta/wrapper/MetaFactory.h"

constexpr size_t VAL_RANGE = 1024;

/**
 * Template test class for tensor slicing and AllReduce operations.
 *
 * This test implements the following scenario:
 *   tensorA = baseTensor.copy()[0:tensorASize]   // Full tensor copy
 *   tensorB = baseTensor.copy()[offset:tensorASize]  // Slice starting from
 * offset allreduce(tensorA, comm) allreduce(tensorB, comm) max_diff = tensorB -
 * tensorA[offset:tensorASize] // Compare overlapping sections
 *
 */
template <typename TYPE>
class AllReduceNumericOffsetTest : public NcclxBaseTest {
 public:
  AllReduceNumericOffsetTest() = default;

  void SetUp() override {
    NcclxBaseTest::SetUp();

    comm = createNcclComm(
        globalRank, numRanks, localRank, false, nullptr, server.get());

    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    finalizeNcclComm(globalRank, server.get());
    NcclxBaseTest::TearDown();
  }

  void assignChunkValue(TYPE* buf, size_t count, TYPE val) {
    std::vector<TYPE> expectedVals(count, val);
    CUDACHECKIGNORE(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(TYPE), cudaMemcpyDefault));
  }

  void verifyMaxDiff(
      TYPE* tensorA,
      TYPE* tensorB,
      size_t tensorASize,
      size_t tensorBSize,
      size_t offset,
      bool expectedEqual) {
    // Copy results back to host
    std::vector<TYPE> tensorAResult(tensorASize);
    std::vector<TYPE> tensorBResult(tensorBSize);

    CUDACHECK_TEST(cudaMemcpy(
        tensorAResult.data(),
        tensorA,
        tensorASize * sizeof(TYPE),
        cudaMemcpyDefault));

    CUDACHECK_TEST(cudaMemcpy(
        tensorBResult.data(),
        tensorB,
        tensorBSize * sizeof(TYPE),
        cudaMemcpyDefault));

    // Calculate max_diff = tensorB - tensorA[offset:tensorASize]
    TYPE max_diff = 0;
    int nDiffs = 0;
    for (int i = 0; i < tensorBSize; i++) {
      TYPE b = tensorBResult[i];
      TYPE a = tensorAResult[i + offset];
      TYPE diff =
          (a > b) ? (a - b) : (b - a); // Manual abs to avoid std::abs ambiguity
      if (diff > max_diff) {
        max_diff = diff;
        nDiffs++;
      }
    }

    EXPECT_EQ(nDiffs == 0, expectedEqual)
        << "max_diff should be " << (expectedEqual ? "0" : "non-zero")
        << " but got " << float(max_diff) << " on rank " << this->globalRank;
  }

  void run(
      ncclDataType_t dataType,
      size_t tensorASize,
      size_t tensorBSize,
      size_t offset,
      bool expectedEqual) {
    TYPE *tensorA = nullptr, *tensorB = nullptr;

    // create and register buffers
    FB_COMMCHECKTHROW(ncclToMetaComm(
        ncclMemAlloc((void**)&tensorA, tensorASize * sizeof(TYPE))));
    FB_COMMCHECKTHROW(ncclToMetaComm(
        ncclMemAlloc((void**)&tensorB, tensorBSize * sizeof(TYPE))));

    // Initialize tensorA with rank-specific values
    std::vector<TYPE> hostTensorA(tensorASize);
    for (size_t i = 0; i < tensorASize; i++) {
      auto val = i % VAL_RANGE + globalRank;
      hostTensorA[i] = (TYPE)(val);
    }
    CUDACHECK_TEST(cudaMemcpy(
        tensorA,
        hostTensorA.data(),
        tensorASize * sizeof(TYPE),
        cudaMemcpyDefault));

    // Initialize tensorB as slice of tensorA (tensorA[offset:])
    std::vector<TYPE> hostTensorB(tensorBSize);
    for (size_t i = 0; i < tensorBSize; i++) {
      hostTensorB[i] = hostTensorA[i + offset];
    }
    CUDACHECK_TEST(cudaMemcpy(
        tensorB,
        hostTensorB.data(),
        tensorBSize * sizeof(TYPE),
        cudaMemcpyDefault));

    void *tensorAHandle = nullptr, *tensorBHandle = nullptr;
    NCCLCHECK_TEST(ncclCommRegister(
        comm, tensorA, tensorASize * sizeof(TYPE), &tensorAHandle));
    NCCLCHECK_TEST(ncclCommRegister(
        comm, tensorB, tensorBSize * sizeof(TYPE), &tensorBHandle));

    // Perform allreduce on tensorA
    ncclResult_t res1 = ncclAllReduce(
        tensorA, tensorA, tensorASize, dataType, ncclSum, comm, stream);
    ASSERT_EQ(res1, ncclSuccess);

    // Perform allreduce on tensorB
    ncclResult_t res2 = ncclAllReduce(
        tensorB, tensorB, tensorBSize, dataType, ncclSum, comm, stream);
    ASSERT_EQ(res2, ncclSuccess);

    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    // Verify that overlapping sections are identical
    verifyMaxDiff(
        tensorA, tensorB, tensorASize, tensorBSize, offset, expectedEqual);

    NCCLCHECK_TEST(ncclCommDeregister(comm, tensorAHandle));
    NCCLCHECK_TEST(ncclCommDeregister(comm, tensorBHandle));
    NCCLCHECK_TEST(ncclMemFree(tensorA));
    NCCLCHECK_TEST(ncclMemFree(tensorB));
  }

 protected:
  ncclComm_t comm{};
  cudaStream_t stream{};
};

// Typed test classes for specific data types
class AllReduceNumericOffsetTestParamInt32
    : public AllReduceNumericOffsetTest<int>,
      public ::testing::WithParamInterface<std::tuple<size_t, size_t, size_t>> {
};

TEST_P(AllReduceNumericOffsetTestParamInt32, StableSliceTestInt32) {
  const auto& [tensorASize, tensorBSize, offset] = GetParam();
  run(ncclInt32, tensorASize, tensorBSize, offset, true);
}

class AllReduceNumericOffsetTestParamBF16
    : public AllReduceNumericOffsetTest<__nv_bfloat16>,
      public ::testing::WithParamInterface<std::tuple<size_t, size_t, size_t>> {
};

TEST_P(AllReduceNumericOffsetTestParamBF16, StableSliceTestBF16) {
  const auto& [tensorASize, tensorBSize, offset] = GetParam();
  run(ncclBfloat16, tensorASize, tensorBSize, offset, false);
}

// Test parameters: {tensorASize, tensorBSize, offset}
auto stableTestingValues = ::testing::Values(
    std::make_tuple(100, 90, 10), // Basic slice test
    std::make_tuple(1000, 500, 250), // Larger slice test
    std::make_tuple(64, 32, 16), // Power of 2 sizes
    std::make_tuple(1024, 512, 256), // Larger power of 2 sizes
    std::make_tuple(8195, 4000, 1000) // Non-power of 2 sizes
);

// common function to get test name from test parameter
inline std::string getStableTestName(
    const testing::TestParamInfo<
        AllReduceNumericOffsetTestParamBF16::ParamType>& info) {
  return "TensorA_" + std::to_string(std::get<0>(info.param)) + "_TensorB_" +
      std::to_string(std::get<1>(info.param)) + "_Offset_" +
      std::to_string(std::get<2>(info.param));
}

// Tests for Int32
INSTANTIATE_TEST_SUITE_P(
    AllReduceStableTests,
    AllReduceNumericOffsetTestParamInt32,
    stableTestingValues,
    getStableTestName);

// Tests for BFloat16
INSTANTIATE_TEST_SUITE_P(
    AllReduceStableTests,
    AllReduceNumericOffsetTestParamBF16,
    stableTestingValues,
    getStableTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
