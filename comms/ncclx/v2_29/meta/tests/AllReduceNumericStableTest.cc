// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <stdlib.h>
#include <cstddef>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/Random.h>
#include <folly/init/Init.h>

#include <comm.h>
#include <nccl.h>
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

#include "meta/wrapper/MetaFactory.h"

/**
 * Template test class for testing AllReduce stability with identical input
 * values. This test verifies that multiple AllReduce operations with identical
 * inputs produce identical outputs, ensuring numerical stability.
 */
template <typename TYPE>
class AllReduceStableTest : public NcclxBaseTest {
 public:
  AllReduceStableTest() = default;

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

 protected:
  ncclComm_t comm{};
  cudaStream_t stream{};
};

template <typename TYPE>
class AllReduceStableTestIdenticalInput
    : public AllReduceStableTest<TYPE>,
      public ::testing::WithParamInterface<std::tuple<size_t, ncclDataType_t>> {
 public:
  int checkChunkValue(TYPE* tensorA, TYPE* tensorB, size_t count) {
    std::vector<TYPE> observedA(count, -1);
    std::vector<TYPE> observedB(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedA.data(), tensorA, count * sizeof(TYPE), cudaMemcpyDefault));
    CUDACHECK_TEST(cudaMemcpy(
        observedB.data(), tensorB, count * sizeof(TYPE), cudaMemcpyDefault));
    int errs = 0;

    // Use manual print rather than EXPECT_THAT to print failing location
    for (auto i = 0; i < count; ++i) {
      if (observedA[i] != observedB[i]) {
        if (errs < 10) {
          printf(
              "[%d] observedA[%d] = %f, observedB = %f\n",
              this->globalRank,
              i,
              static_cast<float>(observedA[i]),
              static_cast<float>(observedB[i]));
        }
        errs++;
      }
    }
    return errs;
  }

  void runTest(size_t tensorSize, ncclDataType_t dataType) {
    TYPE *tensorA = nullptr, *tensorB = nullptr;

    NCCLCHECK_TEST(ncclMemAlloc((void**)&tensorA, tensorSize * sizeof(TYPE)));
    NCCLCHECK_TEST(ncclMemAlloc((void**)&tensorB, tensorSize * sizeof(TYPE)));

    std::vector<TYPE> randomValues(tensorSize);
    for (size_t i = 0; i < tensorSize; i++) {
      if constexpr (std::is_same_v<TYPE, float>) {
        randomValues[i] = static_cast<TYPE>(folly::Random::rand32()) /
            static_cast<TYPE>(std::numeric_limits<uint32_t>::max());
      } else {
        randomValues[i] = static_cast<TYPE>(folly::Random::rand32() % 1000);
      }
    }

    CUDACHECK_TEST(cudaMemcpy(
        tensorA,
        randomValues.data(),
        tensorSize * sizeof(TYPE),
        cudaMemcpyDefault));
    CUDACHECK_TEST(cudaMemcpy(
        tensorB,
        randomValues.data(),
        tensorSize * sizeof(TYPE),
        cudaMemcpyDefault));

    void *tensorAHandle = nullptr, *tensorBHandle = nullptr;
    NCCLCHECK_TEST(ncclCommRegister(
        this->comm, tensorA, tensorSize * sizeof(TYPE), &tensorAHandle));
    NCCLCHECK_TEST(ncclCommRegister(
        this->comm, tensorB, tensorSize * sizeof(TYPE), &tensorBHandle));

    ncclResult_t res1 = ncclAllReduce(
        tensorA,
        tensorA,
        tensorSize,
        dataType,
        ncclSum,
        this->comm,
        this->stream);
    ASSERT_EQ(res1, ncclSuccess);

    ncclResult_t res2 = ncclAllReduce(
        tensorB,
        tensorB,
        tensorSize,
        dataType,
        ncclSum,
        this->comm,
        this->stream);
    ASSERT_EQ(res2, ncclSuccess);

    CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

    // Check tensorA results
    int errs = checkChunkValue(tensorA, tensorB, tensorSize);
    EXPECT_EQ(errs, 0) << "TensorA and TensorB has " << errs
                       << " mismatched values on rank " << this->globalRank;

    NCCLCHECK_TEST(ncclCommDeregister(this->comm, tensorAHandle));
    NCCLCHECK_TEST(ncclCommDeregister(this->comm, tensorBHandle));
    NCCLCHECK_TEST(ncclMemFree(tensorA));
    NCCLCHECK_TEST(ncclMemFree(tensorB));
  }
};

class AllReduceStableTestIdenticalInputFloat
    : public AllReduceStableTestIdenticalInput<float> {};

TEST_P(AllReduceStableTestIdenticalInputFloat, IdenticalInputValuesFloat) {
  const auto& [tensorSize, dataType] = GetParam();
  runTest(tensorSize, dataType);
}

class AllReduceStableTestIdenticalInputInt32
    : public AllReduceStableTestIdenticalInput<int> {};

TEST_P(AllReduceStableTestIdenticalInputInt32, IdenticalInputValuesInt32) {
  const auto& [tensorSize, dataType] = GetParam();
  runTest(tensorSize, dataType);
}

class AllReduceStableTestIdenticalInputBF16
    : public AllReduceStableTestIdenticalInput<__nv_bfloat16> {};

TEST_P(AllReduceStableTestIdenticalInputBF16, IdenticalInputValuesBF16) {
  const auto& [tensorSize, dataType] = GetParam();
  runTest(tensorSize, dataType);
}

// Test parameters: {tensorSize, dataType}
auto identicalInputTestParamsFloat = ::testing::Values(
    std::make_tuple(64, ncclFloat32),
    std::make_tuple(256, ncclFloat32),
    std::make_tuple(1024, ncclFloat32),
    std::make_tuple(4096, ncclFloat32),
    std::make_tuple(8195, ncclFloat32));

auto identicalInputTestParamsInt32 = ::testing::Values(
    std::make_tuple(64, ncclInt32),
    std::make_tuple(256, ncclInt32),
    std::make_tuple(1024, ncclInt32),
    std::make_tuple(4096, ncclInt32),
    std::make_tuple(8195, ncclInt32));

auto identicalInputTestParamsBF16 = ::testing::Values(
    std::make_tuple(64, ncclBfloat16),
    std::make_tuple(256, ncclBfloat16),
    std::make_tuple(1024, ncclBfloat16),
    std::make_tuple(4096, ncclBfloat16),
    std::make_tuple(8195, ncclBfloat16));

inline std::string getIdenticalInputTestName(
    const testing::TestParamInfo<std::tuple<size_t, ncclDataType_t>>& info) {
  return "TensorSize_" + std::to_string(std::get<0>(info.param));
}

// Tests for Float
INSTANTIATE_TEST_SUITE_P(
    AllReduceStableTests,
    AllReduceStableTestIdenticalInputFloat,
    identicalInputTestParamsFloat,
    getIdenticalInputTestName);

// Tests for Int32
INSTANTIATE_TEST_SUITE_P(
    AllReduceStableTests,
    AllReduceStableTestIdenticalInputInt32,
    identicalInputTestParamsInt32,
    getIdenticalInputTestName);

INSTANTIATE_TEST_SUITE_P(
    AllReduceStableTests,
    AllReduceStableTestIdenticalInputBF16,
    identicalInputTestParamsBF16,
    getIdenticalInputTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
