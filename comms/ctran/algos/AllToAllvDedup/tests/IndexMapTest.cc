#include <gtest/gtest.h>

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/testinfra/TestUtils.h"

extern __global__ void testIndexMapTransposeKernel(
    const int* idxMap,
    const int count,
    int* outMap,
    const int iter);

extern __global__ void testIndexMapTransposeSubsetKernel(
    const int* idxMap,
    const int* subIndices,
    const int subSize,
    const int size,
    int* outMap,
    const int iter);

extern __global__ void
testIndexMapCountKernel(const int* idxMap, const int size, int* outCount);

extern __global__ void testIndexMapCountMergeKernel(
    const int** idxMaps,
    const int size,
    const int numMaps,
    int* outCount);

extern __global__ void testIndexMapTransposeSubsetWithFirstLastKernel(
    const int* idxMap,
    const int* subIndices,
    const int subSize,
    const int size,
    int* outMap,
    int* firstRecvIdx,
    int* lastRecvIdx,
    int* count);

class CtranIndexMapTest : public ::testing::Test {
 public:
  CtranIndexMapTest() = default;

 protected:
  static void SetUpTestCase() {
    CUDACHECK_TEST(cudaEventCreate(&start_));
    CUDACHECK_TEST(cudaEventCreate(&stop_));
  }
  static void TearDownTestCase() {
    CUDACHECK_TEST(cudaEventDestroy(start_));
    CUDACHECK_TEST(cudaEventDestroy(stop_));
  }

  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(cudaDev_));
  }

  void TearDown() override {}

  template <typename T>
  void allocDeviceArg(std::vector<T>& argH, T*& argD) {
    const size_t nBytes = sizeof(T) * argH.size();
    ASSERT_EQ(cudaMalloc((void**)&argD, nBytes), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(argD, argH.data(), nBytes, cudaMemcpyDefault), cudaSuccess);
  }

  template <typename T>
  void copyDeviceToHost(T* argD, std::vector<T>& argH) {
    const size_t nBytes = sizeof(T) * argH.size();
    ASSERT_EQ(
        cudaMemcpy(argH.data(), argD, nBytes, cudaMemcpyDefault), cudaSuccess);
  }

 protected:
  int cudaDev_{0};
  static cudaEvent_t start_;
  static cudaEvent_t stop_;
};
cudaEvent_t CtranIndexMapTest::start_ = nullptr;
cudaEvent_t CtranIndexMapTest::stop_ = nullptr;

class CtranIndexMapTestParamFixture
    : public CtranIndexMapTest,
      // count, stride
      public ::testing::WithParamInterface<std::tuple<int, int>> {};

TEST_P(CtranIndexMapTestParamFixture, Transpose) {
  auto [count, stride] = GetParam();

  const int numIter = 50;

  std::vector<int> idxMapH(count);
  // store output of each iteration
  std::vector<int> outMapH(count * numIter, -99);
  int recvIdx = 0;
  for (int i = 0; i < count; i++) {
    if (i % (stride + 1) == 0) {
      idxMapH[i] = recvIdx++;
    } else {
      idxMapH[i] = -1;
    }
  }

  int *idxMap = nullptr, *outMap = nullptr;
  allocDeviceArg(idxMapH, idxMap);
  allocDeviceArg(outMapH, outMap);

  dim3 grid = {1, 1, 1};
  dim3 block = {256, 1, 1};
  void* args[4] = {
      (void*)&idxMap, (void*)&count, (void*)&outMap, (void*)&numIter};
  ASSERT_EQ(
      cudaLaunchKernel((void*)testIndexMapTransposeKernel, grid, block, args),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  copyDeviceToHost(outMap, outMapH);

  for (int iter = 0; iter < numIter; iter++) {
    int* curOut = outMapH.data() + iter * count;

    // chceck populated values
    int lastValidIdx = -1;
    for (int i = 0; i < count; i++) {
      if (idxMapH[i] != -1) {
        int recvIdx = idxMapH[i];
        ASSERT_EQ(curOut[recvIdx], i);
        lastValidIdx = recvIdx;
      }
    }

    // check all other values are -1
    for (int i = lastValidIdx + 1; i < count; i++) {
      ASSERT_EQ(curOut[i], -1);
    }
  }

  ASSERT_EQ(cudaFree(idxMap), cudaSuccess);
  ASSERT_EQ(cudaFree(outMap), cudaSuccess);
}

TEST_P(CtranIndexMapTestParamFixture, TransposePerf) {
  auto [count, stride] = GetParam();
  const int numIter = 1000;
  std::vector<int> idxMapH(count);
  std::vector<int> outMapH(count * numIter, -99);
  int recvIdx = 0;
  for (int i = 0; i < count; i++) {
    if (i % (stride + 1) == 0) {
      idxMapH[i] = recvIdx++;
    } else {
      idxMapH[i] = -1;
    }
  }
  int *idxMap = nullptr, *outMap = nullptr;
  allocDeviceArg(idxMapH, idxMap);
  allocDeviceArg(outMapH, outMap);
  dim3 grid = {1, 1, 1};
  dim3 block = {256, 1, 1};
  void* args[4] = {
      (void*)&idxMap, (void*)&count, (void*)&outMap, (void*)&numIter};
  // Warmup
  ASSERT_EQ(
      cudaLaunchKernel((void*)testIndexMapTransposeKernel, grid, block, args),
      cudaSuccess);
  // Actual run
  ASSERT_EQ(cudaEventRecord(start_), cudaSuccess);
  ASSERT_EQ(
      cudaLaunchKernel((void*)testIndexMapTransposeKernel, grid, block, args),
      cudaSuccess);
  ASSERT_EQ(cudaEventRecord(stop_), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  float gpuTimeMs = 0;
  ASSERT_EQ(cudaEventElapsedTime(&gpuTimeMs, start_, stop_), cudaSuccess);
  printf(
      "transpose latency with count=%d stride=%d, total latency of %d iterations: %f us\n",
      count,
      stride,
      numIter,
      gpuTimeMs * 1000);
  ASSERT_EQ(cudaFree(idxMap), cudaSuccess);
  ASSERT_EQ(cudaFree(outMap), cudaSuccess);
}

class CtranIndexMapSubsetTestParamFixture
    : public CtranIndexMapTest,
      // count, stride, subCount, subStart, subStride
      public ::testing::WithParamInterface<
          std::tuple<int, int, int, int, int>> {};

TEST_P(CtranIndexMapSubsetTestParamFixture, TransposeSubset) {
  auto [count, stride, subCount, subStart, subStride] = GetParam();

  const int numIter = 50;
  std::vector<int> idxMapH(count);
  // store output of each iteration
  std::vector<int> outMapH(count * numIter, -99);
  int recvIdx = 0;
  for (int i = 0; i < count; i++) {
    if (i % (stride + 1) == 0) {
      idxMapH[i] = recvIdx++;
    } else {
      idxMapH[i] = -1;
    }
  }
  std::vector<int> subIndicesH(subCount);
  int idx = subStart;
  for (int i = 0; i < subCount; i++) {
    subIndicesH[i] = idx;
    idx += subStride;
  }

  int *idxMap = nullptr, *outMap = nullptr, *subIndices = nullptr;
  allocDeviceArg(idxMapH, idxMap);
  allocDeviceArg(outMapH, outMap);
  allocDeviceArg(subIndicesH, subIndices);

  dim3 grid = {1, 1, 1};
  dim3 block = {256, 1, 1};
  void* args[6] = {
      (void*)&idxMap,
      (void*)&subIndices,
      (void*)&subCount,
      (void*)&count,
      (void*)&outMap,
      (void*)&numIter};

  ASSERT_EQ(
      cudaLaunchKernel(
          (void*)testIndexMapTransposeSubsetKernel, grid, block, args),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  copyDeviceToHost(outMap, outMapH);

  // get firstRecvIdx
  int firstRecvIdx = -1;
  for (int i = 0; i < subCount; i++) {
    int sendIdx = subIndicesH[i];
    if (sendIdx < count && idxMapH[sendIdx] != -1) {
      int recvIdx = idxMapH[sendIdx];
      if (firstRecvIdx == -1) {
        firstRecvIdx = recvIdx;
        break;
      }
    }
  }

  // compute mappings
  std::vector<std::pair<int, int>> expectedMappings;
  for (int i = 0; i < subCount; i++) {
    int sendIdx = subIndicesH[i];
    if (sendIdx < count && idxMapH[sendIdx] != -1) {
      int recvIdx = idxMapH[sendIdx];
      int relRecvIdx = recvIdx - firstRecvIdx;
      int relSendIdx = i;
      expectedMappings.push_back({relRecvIdx, relSendIdx});
    }
  }

  for (int iter = 0; iter < numIter; iter++) {
    int* curOut = outMapH.data() + iter * count;

    for (const auto& [relRecvIdx, relSendIdx] : expectedMappings) {
      ASSERT_EQ(curOut[relRecvIdx], relSendIdx);
    }
  }

  ASSERT_EQ(cudaFree(idxMap), cudaSuccess);
  ASSERT_EQ(cudaFree(outMap), cudaSuccess);
  ASSERT_EQ(cudaFree(subIndices), cudaSuccess);
}

TEST_P(CtranIndexMapSubsetTestParamFixture, TransposeSubsetPerf) {
  auto [count, stride, subCount, subStart, subStride] = GetParam();
  const int numIter = 1000;
  std::vector<int> idxMapH(count);
  std::vector<int> outMapH(count * numIter, -99);
  int recvIdx = 0;
  for (int i = 0; i < count; i++) {
    if (i % (stride + 1) == 0) {
      idxMapH[i] = recvIdx++;
    } else {
      idxMapH[i] = -1;
    }
  }
  std::vector<int> subIndicesH(subCount);
  int idx = subStart;
  for (int i = 0; i < subCount; i++) {
    subIndicesH[i] = idx;
    idx += subStride;
  }

  int *idxMap = nullptr, *outMap = nullptr, *subIndices = nullptr;
  allocDeviceArg(idxMapH, idxMap);
  allocDeviceArg(outMapH, outMap);
  allocDeviceArg(subIndicesH, subIndices);

  dim3 grid = {1, 1, 1};
  dim3 block = {256, 1, 1};
  void* args[6] = {
      (void*)&idxMap,
      (void*)&subIndices,
      (void*)&subCount,
      (void*)&count,
      (void*)&outMap,
      (void*)&numIter};

  // Warmup
  ASSERT_EQ(
      cudaLaunchKernel(
          (void*)testIndexMapTransposeSubsetKernel, grid, block, args),
      cudaSuccess);
  // Actual run
  ASSERT_EQ(cudaEventRecord(start_), cudaSuccess);
  ASSERT_EQ(
      cudaLaunchKernel(
          (void*)testIndexMapTransposeSubsetKernel, grid, block, args),
      cudaSuccess);
  ASSERT_EQ(cudaEventRecord(stop_), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  float gpuTimeMs = 0;
  ASSERT_EQ(cudaEventElapsedTime(&gpuTimeMs, start_, stop_), cudaSuccess);
  printf(
      "transpose subset latency with count=%d stride=%d subCount=%d, total latency of %d iterations: %f us\n",
      count,
      stride,
      subCount,
      numIter,
      gpuTimeMs * 1000);

  ASSERT_EQ(cudaFree(idxMap), cudaSuccess);
  ASSERT_EQ(cudaFree(outMap), cudaSuccess);
  ASSERT_EQ(cudaFree(subIndices), cudaSuccess);
}

TEST_F(CtranIndexMapTest, Count) {
  dim3 grid = {1, 1, 1};
  dim3 block = {256, 1, 1};

  // simple test
  std::vector<int> idxMapH1 = {0, 1, -1, 2, -1, 3};

  int* idxMap1 = nullptr;
  int* outCount1 = nullptr;
  std::vector<int> outCountH1(1);

  allocDeviceArg(idxMapH1, idxMap1);
  allocDeviceArg(outCountH1, outCount1);

  int size1 = idxMapH1.size();
  void* args1[3] = {(void*)&idxMap1, (void*)&size1, (void*)&outCount1};

  ASSERT_EQ(
      cudaLaunchKernel((void*)testIndexMapCountKernel, grid, block, args1),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  copyDeviceToHost(outCount1, outCountH1);
  ASSERT_EQ(outCountH1[0], 4);

  ASSERT_EQ(cudaFree(idxMap1), cudaSuccess);
  ASSERT_EQ(cudaFree(outCount1), cudaSuccess);

  // test all -1
  std::vector<int> idxMapH2 = {-1, -1, -1, -1};

  int *idxMap2 = nullptr, *outCount2 = nullptr;
  std::vector<int> outCountH2(1);

  allocDeviceArg(idxMapH2, idxMap2);
  allocDeviceArg(outCountH2, outCount2);

  int size2 = idxMapH2.size();
  void* args2[3] = {(void*)&idxMap2, (void*)&size2, (void*)&outCount2};

  ASSERT_EQ(
      cudaLaunchKernel((void*)testIndexMapCountKernel, grid, block, args2),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  copyDeviceToHost(outCount2, outCountH2);
  ASSERT_EQ(outCountH2[0], 0);

  ASSERT_EQ(cudaFree(idxMap2), cudaSuccess);
  ASSERT_EQ(cudaFree(outCount2), cudaSuccess);
}

TEST_F(CtranIndexMapTest, CountMerge) {
  std::vector<int> map1H = {0, -1, 1, -1, 2};
  std::vector<int> map2H = {-1, 0, -1, -1, -1};
  std::vector<int> map3H = {0, -1, -1, -1, 1};

  int expectedCount = 4;

  int *map1D = nullptr, *map2D = nullptr, *map3D = nullptr;
  allocDeviceArg(map1H, map1D);
  allocDeviceArg(map2H, map2D);
  allocDeviceArg(map3H, map3D);

  std::vector<int*> mapsH = {map1D, map2D, map3D};
  int** mapsD = nullptr;
  allocDeviceArg(mapsH, mapsD);

  std::vector<int> outCountH(1);
  int* outCountD = nullptr;
  allocDeviceArg(outCountH, outCountD);

  dim3 grid = {1, 1, 1};
  dim3 block = {256, 1, 1};

  int size = map1H.size();
  int numMaps = 3;
  void* args[4] = {
      (void*)&mapsD, (void*)&size, (void*)&numMaps, (void*)&outCountD};

  ASSERT_EQ(
      cudaLaunchKernel((void*)testIndexMapCountMergeKernel, grid, block, args),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  copyDeviceToHost(outCountD, outCountH);
  ASSERT_EQ(outCountH[0], expectedCount);

  ASSERT_EQ(cudaFree(map1D), cudaSuccess);
  ASSERT_EQ(cudaFree(map2D), cudaSuccess);
  ASSERT_EQ(cudaFree(map3D), cudaSuccess);
  ASSERT_EQ(cudaFree(mapsD), cudaSuccess);
  ASSERT_EQ(cudaFree(outCountD), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranIndexMapTestParamFixture,
    ::testing::Values(
        // count, stride
        std::make_tuple(64, 1),
        std::make_tuple(128, 2),
        std::make_tuple(256, 3),
        std::make_tuple(512, 4)),
    [&](const testing::TestParamInfo<CtranIndexMapTestParamFixture::ParamType>&
            info) {
      return std::to_string(std::get<0>(info.param)) + "count_" +
          std::to_string(std::get<1>(info.param)) + "stride";
    });

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranIndexMapSubsetTestParamFixture,
    ::testing::Values(
        // count, stride, subCount, subStart, subStride
        std::make_tuple(32, 2, 32, 0, 3),
        std::make_tuple(64, 2, 32, 0, 3),
        std::make_tuple(128, 2, 32, 0, 3),
        std::make_tuple(256, 2, 32, 0, 3),
        std::make_tuple(128, 1, 16, 0, 7),
        std::make_tuple(256, 3, 64, 1, 2)),
    [&](const testing::TestParamInfo<
        CtranIndexMapSubsetTestParamFixture::ParamType>& info) {
      return std::to_string(std::get<0>(info.param)) + "count_" +
          std::to_string(std::get<1>(info.param)) + "stride_" +
          std::to_string(std::get<2>(info.param)) + "subCount_" +
          std::to_string(std::get<3>(info.param)) + "subStart_" +
          std::to_string(std::get<4>(info.param)) + "subStride";
    });
