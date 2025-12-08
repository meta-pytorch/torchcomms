// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <cuda_fp16.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/tests/CtranAlgoDevTestUtils.h"
#include "comms/ctran/algos/tests/CtranAlgoDevUTKernels.cuh"
#include "comms/ctran/tests/CtranXPlatUtUtils.h"

class CtranAlgoDevTest : public ::testing::Test {
 public:
  CtranAlgoDevTest() = default;

 protected:
  static void SetUpTestCase() {}

  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(cudaDev_));
    CUDACHECK_TEST(cudaEventCreate(&start_));
    CUDACHECK_TEST(cudaEventCreate(&stop_));
  }
  void TearDown() override {
    CUDACHECK_TEST(cudaEventDestroy(start_));
    CUDACHECK_TEST(cudaEventDestroy(stop_));
  }

 protected:
  int cudaDev_{0};
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

// Used for testing various data types
template <typename T>
class CtranAlgoDevTypedTest : public CtranAlgoDevTest {
 public:
  CtranAlgoDevTypedTest() = default;
};
// define all types going to be tested
using testingTypes = ::testing::Types<
    float,
    double,
    char,
    int,
    int64_t,
    uint32_t,
    uint64_t,
    half
#if defined(__CUDA_BF16_TYPES_EXIST__)
    ,
    __nv_bfloat16
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    ,
    __nv_fp8_e4m3,
    __nv_fp8_e5m2
#endif
    >;
TYPED_TEST_SUITE(CtranAlgoDevTypedTest, testingTypes);

TEST_F(CtranAlgoDevTest, devStateLoadToShm) {
  constexpr int localRanks = CTRAN_MAX_NVL_PEERS;
  constexpr int localRank = 1;
  constexpr int node = 2;
  constexpr int nNodes = 4;

  auto dummyDevState = CtranAlgoDeviceState();
  dummyDevState.bufSize = 65536;
  dummyDevState.statex.rank_ = localRanks * node + localRank;
  dummyDevState.statex.pid_ = getpid();
  dummyDevState.statex.localRank_ = localRank;
  dummyDevState.statex.localRanks_ = localRanks;
  dummyDevState.statex.nRanks_ = localRanks * nNodes;
  dummyDevState.statex.nNodes_ = nNodes;
  dummyDevState.statex.commHash_ = 0x12345678;

  void* devPtr = nullptr;
  CUDACHECK_TEST(cudaMalloc(&devPtr, sizeof(CtranAlgoDeviceState) * 2));

  void* devInPtr = devPtr;
  void* devOutPtr =
      reinterpret_cast<char*>(devPtr) + sizeof(CtranAlgoDeviceState);

  CUDACHECK_TEST(cudaMemcpy(
      devPtr,
      &dummyDevState,
      sizeof(CtranAlgoDeviceState),
      cudaMemcpyHostToDevice));

  dim3 grid = {2, 1, 1};
  dim3 block = {640, 1, 1};
  void* args[2] = {&devInPtr, &devOutPtr};
  void* fn = (void*)devStateLoadToShmTestKernel<true>;
  CUDA_LAUNCH_KERNEL_WITH_DYNAMIC_SHM_TEST(
      fn, grid, block, args, sizeof(CtranAlgoDeviceState), nullptr);

  // Check correctness
  auto dummyDevStateCheck = CtranAlgoDeviceState();
  CUDACHECK_TEST(cudaMemcpy(
      &dummyDevStateCheck,
      devOutPtr,
      sizeof(CtranAlgoDeviceState),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(dummyDevStateCheck.bufSize, dummyDevState.bufSize);

  const auto& checkStatex = dummyDevStateCheck.statex;
  const auto& expStatex = dummyDevState.statex;

  EXPECT_EQ(checkStatex.rank_, expStatex.rank_);
  EXPECT_EQ(checkStatex.pid_, expStatex.pid_);
  EXPECT_EQ(checkStatex.localRank_, expStatex.localRank_);
  EXPECT_EQ(checkStatex.localRanks_, expStatex.localRanks_);
  EXPECT_EQ(checkStatex.nRanks_, expStatex.nRanks_);
  EXPECT_EQ(checkStatex.nNodes_, expStatex.nNodes_);
  EXPECT_EQ(checkStatex.commHash_, expStatex.commHash_);

  // Report performance
  constexpr int iteration = 1000;
  fn = (void*)devStateLoadToShmTestKernel<false>;
  CUDACHECK_TEST(cudaEventRecord(start_));
  for (int x = 0; x < iteration; x++) {
    CUDA_LAUNCH_KERNEL_WITH_DYNAMIC_SHM_TEST(
        fn, grid, block, args, sizeof(CtranAlgoDeviceState), nullptr);
  }
  CUDACHECK_TEST(cudaEventRecord(stop_));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  float gpuTimeMs_ = 0;
  CUDACHECK_TEST(cudaEventElapsedTime(&gpuTimeMs_, start_, stop_));
  printf(
      "Load CtranAlgoDeviceState with %ld bytes latency %.2f us\n",
      sizeof(CtranAlgoDeviceState),
      gpuTimeMs_ * 1000 / iteration);

  CUDACHECK_TEST(cudaFree(devPtr));
}

/* helper function to get expected value given the datatype and op
 * each vector assisns a value for testing the reduce kernel, e.g.,
 *   - vector-1: [1, 1, ..., 1]
 *   - vector-2: [2, 2, ..., 2] and so on.
 */
template <typename T>
static T getExpectedVal(
    size_t nvectors,
    T initVal = 1.0,
    commRedOp_t op = commSum,
    int nranks = 1) {
  T expectedVal = T(float(1.0));
  if (op == commSum || op == commAvg) {
    expectedVal = T(0.0);
    for (int i = 1; i <= nvectors; ++i) {
      expectedVal = T(float(expectedVal) + float(initVal) * i);
    }
    if (op == commAvg) {
      expectedVal = T(float(expectedVal) / nranks);
    }
  } else if (op == commProd) {
    for (int i = 1; i <= nvectors; ++i) {
      expectedVal = T(float(expectedVal) * float(initVal) * i);
    }
  } else if (op == commMax) {
    expectedVal = T(float(nvectors));
  } else if (op == commMin) {
    expectedVal = initVal;
  }
  return expectedVal;
}

template <typename T>
void localReduceTest(
    size_t nsrcs,
    size_t ndsts,
    size_t count,
    commRedOp_t op = commSum,
    int nranks = 1,
    bool subsetThreadBlocks = false,
    uint numThreadBlocks = 2) {
  size_t nbytes = count * sizeof(T);
  // prepare src buffers
  T initVal = T(float(1.0));
  T expectedVal = getExpectedVal<T>(nsrcs, initVal, op, nranks);
  std::vector<T> srcHost(count, initVal);

  void** srcsDev{nullptr};
  // srcs has to be accessible to CUDA kernel, so use pinned host memory
  CUDACHECK_TEST(
      cudaHostAlloc(&srcsDev, nsrcs * sizeof(void*), cudaHostAllocDefault));
  for (int i = 0; i < nsrcs; ++i) {
    srcHost.assign(count, T(float(initVal) * (i + 1)));
    CUDACHECK_TEST(cudaMalloc(&srcsDev[i], nbytes));
    CUDACHECK_TEST(
        cudaMemcpy(srcsDev[i], srcHost.data(), nbytes, cudaMemcpyHostToDevice));
  }
  // prepare output buffers, set default value to 0
  void** dstsDev{nullptr};
  CUDACHECK_TEST(
      cudaHostAlloc(&dstsDev, ndsts * sizeof(void*), cudaHostAllocDefault));
  for (int i = 0; i < nsrcs; ++i) {
    CUDACHECK_TEST(cudaMalloc(&dstsDev[i], nbytes));
    CUDACHECK_TEST(cudaMemset(dstsDev[i], 0, nbytes));
  }

  // launch reduce kernel
  dim3 grid = {numThreadBlocks, 1, 1};
  dim3 block = {640, 1, 1};

  void* args[7] = {&nsrcs, &srcsDev, &ndsts, &dstsDev, &count, &op, &nranks};
  void* fn;
  if (subsetThreadBlocks) {
    fn = (void*)testCtranLocalReduceSubsetThreadBlocks<T>;
  } else {
    fn = (void*)testCtranLocalReduce<T>;
  }
  CUDACHECK_TEST(cudaLaunchKernel(fn, grid, block, args));

  // Copy result to host and check correctness
  std::vector<T> dstHost(count, initVal);
  for (int d = 0; d < ndsts; ++d) {
    CUDACHECK_TEST(
        cudaMemcpy(dstHost.data(), dstsDev[d], nbytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < count; ++i) {
#if CUDA_VERSION < 12200
      // host-side support of math operations, e.g., operators like ==, for fp16
      // and bf16 are introduced after CUDA 12.2. For older CUDA version, cast
      // up to fp32 for the comparison
      if (std::is_same<T, half>::value
#if defined(__CUDA_BF16_TYPES_EXIST__)
          || std::is_same<T, __nv_bfloat16>::value
#endif
      ) {
        ASSERT_EQ(float(dstHost[i]), float(expectedVal))
            << " at dstHost[" << i << "] count " << count;
      } else
#endif
      {
        ASSERT_EQ(dstHost[i], expectedVal)
            << " at dstHost[" << i << "] count " << count;
      }
    }
  }

  for (int i = 0; i < ndsts; ++i) {
    CUDACHECK_TEST(cudaFree(dstsDev[i]));
  }
  CUDACHECK_TEST(cudaFreeHost(dstsDev));
  for (int i = 0; i < nsrcs; ++i) {
    CUDACHECK_TEST(cudaFree(srcsDev[i]));
  }
  CUDACHECK_TEST(cudaFreeHost(srcsDev));
}

TYPED_TEST(CtranAlgoDevTypedTest, localReduceSum) {
  // test reduction kernel with 2 or 8 srcs, and 1 or 2 dsts, which are common
  // cases for inter-node and intra-node collectives.
  std::vector<size_t> testnSrcs{2, 8};
  std::vector<size_t> testnDsts{1, 2};
  constexpr size_t count = 8192;

  for (auto nsrcs : testnSrcs) {
    for (auto ndsts : testnDsts) {
      localReduceTest<TypeParam>(nsrcs, ndsts, count, commSum);
    }
  }
}

TYPED_TEST(CtranAlgoDevTypedTest, localReduceSumSubsetThreadBlocks) {
  // test reduction kernel with 2 or 8 srcs, and 1 or 2 dsts, which are common
  // cases for inter-node and intra-node collectives.
  std::vector<size_t> testnSrcs{2, 8};
  std::vector<size_t> testnDsts{1, 2};
  constexpr size_t count = 8192;

  for (auto nsrcs : testnSrcs) {
    for (auto ndsts : testnDsts) {
      localReduceTest<TypeParam>(
          nsrcs,
          ndsts,
          count,
          commSum,
          /*nranks=*/1,
          /*subsetThreadBlocks=*/true,
          /*numThreadBlocks=*/4);
    }
  }
}

TYPED_TEST(CtranAlgoDevTypedTest, localReduceSumUnaligned) {
  // test reduction kernel with 2 or 8 srcs, and 1 or 2 dsts, which are common
  // cases for inter-node and intra-node collectives.
  std::vector<size_t> testnSrcs{2, 8};
  std::vector<size_t> testnDsts{1, 2};
  constexpr size_t count = 1041;

  for (auto nsrcs : testnSrcs) {
    for (auto ndsts : testnDsts) {
      localReduceTest<TypeParam>(nsrcs, ndsts, count, commSum);
    }
  }
}

TYPED_TEST(CtranAlgoDevTypedTest, localReduceProd) {
  // test reduction kernel with 2 or 8 srcs, and 1 or 2 dsts, which are common
  // cases for inter-node and intra-node collectives.
  std::vector<size_t> testnSrcs{2, 8};
  std::vector<size_t> testnDsts{1, 2};
  constexpr size_t count = 8192;

  for (auto nsrcs : testnSrcs) {
    for (auto ndsts : testnDsts) {
      localReduceTest<TypeParam>(nsrcs, ndsts, count, commProd);
    }
  }
}

TYPED_TEST(CtranAlgoDevTypedTest, localReduceMax) {
  // test reduction kernel with 2 or 8 srcs, and 1 or 2 dsts, which are common
  // cases for inter-node and intra-node collectives.
  std::vector<size_t> testnSrcs{2, 8};
  std::vector<size_t> testnDsts{1, 2};
  constexpr size_t count = 8192;

  for (auto nsrcs : testnSrcs) {
    for (auto ndsts : testnDsts) {
      localReduceTest<TypeParam>(nsrcs, ndsts, count, commMax);
    }
  }
}

TYPED_TEST(CtranAlgoDevTypedTest, localReduceMin) {
  // test reduction kernel with 2 or 8 srcs, and 1 or 2 dsts, which are common
  // cases for inter-node and intra-node collectives.
  std::vector<size_t> testnSrcs{2, 8};
  std::vector<size_t> testnDsts{1, 2};
  constexpr size_t count = 8192;

  for (auto nsrcs : testnSrcs) {
    for (auto ndsts : testnDsts) {
      localReduceTest<TypeParam>(nsrcs, ndsts, count, commMin);
    }
  }
}

TYPED_TEST(CtranAlgoDevTypedTest, localReduceAvg) {
  // test reduction kernel with 2 or 8 srcs, and 1 or 2 dsts, which are common
  // cases for inter-node and intra-node collectives.
  std::vector<size_t> testnSrcs{2, 8};
  std::vector<size_t> testnDsts{1, 2};
  constexpr size_t count = 8192;

  for (auto nsrcs : testnSrcs) {
    for (auto ndsts : testnDsts) {
      localReduceTest<TypeParam>(nsrcs, ndsts, count, commAvg, /*nranks=*/16);
    }
  }
}

template <typename T, typename RedT>
void dequantizedAllToAllReduceTest(
    size_t count,
    size_t nRanks,
    commRedOp_t op = commSum) {
  size_t nbytes = count * nRanks * sizeof(T);
  // prepare src buffers
  T initVal = T(float(1.0));
  T expectedVal = getExpectedVal<T>(nRanks, initVal, op);

  void* srcDev{nullptr};
  CUDACHECK_TEST(cudaMalloc(&srcDev, nbytes));

  // Create host buffer with proper typed values
  std::vector<T> srcHost(count * nRanks);
  for (int i = 0; i < nRanks; i++) {
    for (int j = 0; j < count; j++) {
      srcHost[i * count + j] = T(float(initVal) * (i + 1));
    }
  }

  // Copy initialized host data to device
  CUDACHECK_TEST(
      cudaMemcpy(srcDev, srcHost.data(), nbytes, cudaMemcpyHostToDevice));

  // prepare output buffer, set default value to 0
  void* dstDev{nullptr};
  CUDACHECK_TEST(cudaMalloc(&dstDev, nbytes));
  CUDACHECK_TEST(cudaMemset(dstDev, 0, nbytes));

  // launch reduce kernel
  dim3 grid = {2, 1, 1};
  dim3 block = {640, 1, 1};
  // assume my rank is 0
  int myRank = 0;

  void* args[6] = {&srcDev, &dstDev, &count, &myRank, &nRanks, &op};
  void* fn = (void*)testDequantizedAllToAllLocalReduce<T, RedT>;
  CUDACHECK_TEST(cudaLaunchKernel(fn, grid, block, args));

  // Copy result to host and check correctness
  std::vector<T> dstHost(count * nRanks, initVal);
  CUDACHECK_TEST(
      cudaMemcpy(dstHost.data(), dstDev, nbytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < count; ++i) {
#if CUDA_VERSION < 12200
    // host-side support of math operations, e.g., operators like ==, for fp16
    // and bf16 are introduced after CUDA 12.2. For older CUDA version, cast
    // up to fp32 for the comparison
    if (std::is_same<T, half>::value
#if defined(__CUDA_BF16_TYPES_EXIST__)
        || std::is_same<T, __nv_bfloat16>::value
#endif
    ) {
      ASSERT_EQ(float(dstHost[i]), float(expectedVal))
          << " at dstHost[" << i << "] count " << count;
    } else
#endif
    {
      ASSERT_EQ(dstHost[i], expectedVal)
          << " at dstHost[" << i << "] count " << count;
    }
  }

  CUDACHECK_TEST(cudaFree(srcDev));
  CUDACHECK_TEST(cudaFree(dstDev));
}

TYPED_TEST(CtranAlgoDevTypedTest, dequantizedAllToAllReduceSum) {
  std::vector<size_t> ranks{8};
  constexpr size_t count = 8192;

  for (auto rank : ranks) {
    dequantizedAllToAllReduceTest<float, float>(count, rank, commSum);
  }
}

TYPED_TEST(CtranAlgoDevTypedTest, dequantizedAllToAllReduceSumUnaligned) {
  std::vector<size_t> ranks{8, 16};
  constexpr size_t count = 1041;

  for (auto rank : ranks) {
    dequantizedAllToAllReduceTest<TypeParam, TypeParam>(count, rank, commSum);
  }
}

TYPED_TEST(CtranAlgoDevTypedTest, dequantizedAllToAllReduceProd) {
  std::vector<size_t> ranks{2, 8};
  constexpr size_t count = 8192;

  for (auto rank : ranks) {
    dequantizedAllToAllReduceTest<TypeParam, TypeParam>(count, rank, commProd);
  }
}

TYPED_TEST(CtranAlgoDevTypedTest, dequantizedAllToAllReduceMax) {
  std::vector<size_t> ranks{8, 16};
  constexpr size_t count = 8192;

  for (auto rank : ranks) {
    dequantizedAllToAllReduceTest<TypeParam, TypeParam>(count, rank, commMax);
  }
}

TYPED_TEST(CtranAlgoDevTypedTest, dequantizedAllToAllReduceMin) {
  std::vector<size_t> ranks{8, 16};
  constexpr size_t count = 8192;

  for (auto rank : ranks) {
    dequantizedAllToAllReduceTest<TypeParam, TypeParam>(count, rank, commMin);
  }
}
