// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdlib.h>
#include <cstddef>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <comm.h>
#include <nccl.h>
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

#include "meta/wrapper/MetaFactory.h"

/**
 * Template test class for testing AllReduce with uniform input per rank.
 *
 * This test verifies that when each rank has all identical values in their
 * input tensor (but different across ranks), the AllReduce output has all
 * identical values.
 *
 * Example:
 *   Rank 0: [a, a, a, ...]
 *   Rank 1: [b, b, b, ...]
 *   Rank 2: [c, c, c, ...]
 *   ...
 *   After AllReduce(Sum): All ranks should have [a+b+c+..., a+b+c+..., ...]
 */

enum class AllReduceAlgo { SingleRing, SingleTree };

template <typename TYPE>
class AllReduceUniformTest : public NcclxBaseTest {
 public:
  AllReduceUniformTest() = default;

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
class AllReduceUniformTestBF16
    : public AllReduceUniformTest<TYPE>,
      public ::testing::WithParamInterface<std::tuple<size_t, AllReduceAlgo>> {
 public:
  /**
   * Verify all elements in the tensor are identical
   */
  bool verifyAllElementsIdentical(TYPE* tensor, size_t count) {
    std::vector<TYPE> observed(count);
    CUDACHECK_TEST(cudaMemcpy(
        observed.data(), tensor, count * sizeof(TYPE), cudaMemcpyDefault));

    if (count == 0) {
      return true;
    }

    TYPE firstValue = observed[0];
    int diffCount = 0;
    float maxDiff = 0.0f;

    for (size_t i = 1; i < count; ++i) {
      if (observed[i] != firstValue) {
        float diff = std::abs(
            static_cast<float>(observed[i]) - static_cast<float>(firstValue));
        if (diffCount < 10) {
          printf(
              "[Rank %d] Element mismatch at index %zu: expected %f, got %f (diff: %f)\n",
              this->globalRank,
              i,
              static_cast<float>(firstValue),
              static_cast<float>(observed[i]),
              diff);
        }
        diffCount++;
        maxDiff = std::max(maxDiff, diff);
      }
    }

    if (diffCount > 0) {
      printf(
          "[Rank %d] Total mismatches: %d / %zu, max diff: %f\n",
          this->globalRank,
          diffCount,
          count,
          maxDiff);
    }

    return diffCount == 0;
  }

  void runTest(size_t tensorSize, AllReduceAlgo algoType) {
    if (algoType == AllReduceAlgo::SingleRing) {
      NCCL_ALGO = "ring";
      NCCL_MAX_NCHANNELS = 1;
    } else if (algoType == AllReduceAlgo::SingleTree) {
      NCCL_ALGO = "tree";
      NCCL_MAX_NCHANNELS = 1;
    } else {
      throw std::runtime_error("Invalid AllReduceAlgo");
    }

    if (algoType == AllReduceAlgo::SingleRing) {
      // See details in https://fburl.com/gdoc/ho8vil04
      GTEST_SKIP()
          << "SingleRing AllReduce is not position invariant due to data chunking";
    }

    TYPE* inputTensor = nullptr;

    // Allocate device memory
    NCCLCHECK_TEST(
        ncclMemAlloc((void**)&inputTensor, tensorSize * sizeof(TYPE)));

    // Generate a single random BF16 value per rank
    std::random_device rd;
    std::mt19937 gen(
        rd() +
        this->globalRank); // Seed with rank for different values per rank
    std::uniform_real_distribution<float> dis(0.1f, 10.0f);

    float rankValue = dis(gen);
    TYPE uniformValue = static_cast<TYPE>(rankValue);

    // Fill entire input tensor with the same value
    std::vector<TYPE> hostInput(tensorSize, uniformValue);

    printf(
        "[Rank %d] Filling tensor with uniform value: %f\n",
        this->globalRank,
        static_cast<float>(uniformValue));

    CUDACHECK_TEST(cudaMemcpy(
        inputTensor,
        hostInput.data(),
        tensorSize * sizeof(TYPE),
        cudaMemcpyDefault));

    // Register buffer
    void* inputHandle = nullptr;
    NCCLCHECK_TEST(ncclCommRegister(
        this->comm, inputTensor, tensorSize * sizeof(TYPE), &inputHandle));

    // Perform AllReduce (in-place)
    ncclResult_t res = ncclAllReduce(
        inputTensor,
        inputTensor,
        tensorSize,
        ncclBfloat16,
        ncclSum,
        this->comm,
        this->stream);
    ASSERT_EQ(res, ncclSuccess);

    CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

    // Verify all elements in output are identical
    bool allIdentical = verifyAllElementsIdentical(inputTensor, tensorSize);

    EXPECT_TRUE(allIdentical)
        << "Not all elements are identical after AllReduce on rank "
        << this->globalRank << " with tensor size " << tensorSize;

    // Cleanup
    NCCLCHECK_TEST(ncclCommDeregister(this->comm, inputHandle));
    NCCLCHECK_TEST(ncclMemFree(inputTensor));
  }
};

using AllReduceUniformTestBF16Inst = AllReduceUniformTestBF16<__nv_bfloat16>;

// FIXME(T254162415): Test disabled due to mpirun runtime failure.
// The test has been broken since 2025-11-06 (codemod c6494ace57c2).
TEST_P(AllReduceUniformTestBF16Inst, DISABLED_UniformInputPerRank) {
  const auto [tensorSize, algoType] = GetParam();
  runTest(tensorSize, algoType);
}

// Test parameters: different tensor sizes
auto uniformTestParams = ::testing::Values(
    // single ring
    std::make_tuple(256, AllReduceAlgo::SingleRing), // Small
    std::make_tuple(1024, AllReduceAlgo::SingleRing), // 1K
    std::make_tuple(4096, AllReduceAlgo::SingleRing), // 4K
    std::make_tuple(8195, AllReduceAlgo::SingleRing), // Non-power of 2
    std::make_tuple(16384, AllReduceAlgo::SingleRing), // 16K
    std::make_tuple(65536, AllReduceAlgo::SingleRing), // 64K
    // SingleTree
    std::make_tuple(256, AllReduceAlgo::SingleTree), // Small
    std::make_tuple(1024, AllReduceAlgo::SingleTree), // 1K
    std::make_tuple(4096, AllReduceAlgo::SingleTree), // 4K
    std::make_tuple(8195, AllReduceAlgo::SingleTree), // Non-power of 2
    std::make_tuple(16384, AllReduceAlgo::SingleTree), // 16K
    std::make_tuple(65536, AllReduceAlgo::SingleTree) // 64K
); // 64K

inline std::string getUniformTestName(
    const testing::TestParamInfo<std::tuple<size_t, AllReduceAlgo>>& info) {
  const auto& [tensorSize, algoType] = info.param;
  std::string algoName =
      (algoType == AllReduceAlgo::SingleRing) ? "SingleRing" : "SingleTree";
  return "TensorSize_" + std::to_string(tensorSize) + "_" + algoName;
}

INSTANTIATE_TEST_SUITE_P(
    AllReduceUniformTests,
    AllReduceUniformTestBF16Inst,
    uniformTestParams,
    getUniformTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
