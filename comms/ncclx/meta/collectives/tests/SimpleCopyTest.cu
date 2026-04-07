// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <numeric>
#include <vector>

#include "meta/collectives/kernels/reduce_copy.cuh"

// Wrapper kernel for the vectorized copy __device__ function.
template <int Unroll, typename T>
__global__ void copy_kernel(T* dst, const T* src, ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::copy<Unroll, T>(
      thread, nThreads, (void*)src, (void*)dst, nElts);
}

// Wrapper kernel for reduceCopy with 2 sources.
template <int Unroll, typename T>
__global__ void
reduce_copy_2src_kernel(T* dst, const T* src0, const T* src1, ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::reduceCopy<Unroll, T>(
      thread, nThreads, (void*)dst, nElts, src0, src1);
}

// Wrapper kernel for reduceCopy with 3 sources.
template <int Unroll, typename T>
__global__ void reduce_copy_3src_kernel(
    T* dst,
    const T* src0,
    const T* src1,
    const T* src2,
    ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::reduceCopy<Unroll, T>(
      thread, nThreads, (void*)dst, nElts, src0, src1, src2);
}

// Wrapper kernel for reduceCopy with 1 source (degenerates to plain copy).
template <int Unroll, typename T>
__global__ void reduce_copy_1src_kernel(T* dst, const T* src, ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::reduceCopy<Unroll, T>(
      thread, nThreads, (void*)dst, nElts, src);
}

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

class SimpleCopyTest : public ::testing::Test {
 protected:
  static constexpr int64_t kMaxN = 4L * 1024L * 1024L + 16;
  static constexpr int kBlockSize = 256;
  static constexpr int kDefaultBlocks = 32;

  int* d_src = nullptr;
  int* d_dst = nullptr;
  // Additional source buffers for reduce-copy tests.
  int* d_src1 = nullptr;
  int* d_src2 = nullptr;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_src, kMaxN * sizeof(int)));
    CUDACHECK(cudaMalloc(&d_dst, kMaxN * sizeof(int)));
    CUDACHECK(cudaMalloc(&d_src1, kMaxN * sizeof(int)));
    CUDACHECK(cudaMalloc(&d_src2, kMaxN * sizeof(int)));
  }

  void TearDown() override {
    CUDACHECK(cudaFree(d_src));
    CUDACHECK(cudaFree(d_dst));
    CUDACHECK(cudaFree(d_src1));
    CUDACHECK(cudaFree(d_src2));
  }

  // Fill src with sequential ints, zero dst, launch kernel, verify dst == src.
  template <typename LaunchFn>
  void verifyCopy(int* src, int* dst, int64_t nElts, LaunchFn launchFn) {
    if (nElts == 0) {
      launchFn(dst, src, 0);
      CUDACHECK(cudaGetLastError());
      CUDACHECK(cudaDeviceSynchronize());
      return;
    }

    std::vector<int> h_src(nElts);
    std::iota(h_src.begin(), h_src.end(), 0);
    CUDACHECK(cudaMemcpy(
        src, h_src.data(), nElts * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(dst, 0, nElts * sizeof(int)));

    launchFn(dst, src, nElts);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<int> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(), dst, nElts * sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_src, h_dst);
  }
};

// Aligned pointers, various sizes (exercises 16-byte aligned fast path).
TEST_F(SimpleCopyTest, CopyAligned) {
  constexpr int64_t sizes[] = {
      32, 1024, 4L * 1024L * 1024L, 1000, 3000007, 4100000};

  for (int64_t nElts : sizes) {
    SCOPED_TRACE("nElts=" + std::to_string(nElts));
    verifyCopy(d_src, d_dst, nElts, [&](int* dst, int* src, ssize_t n) {
      copy_kernel<4, int><<<kDefaultBlocks, kBlockSize>>>(dst, src, n);
      CUDACHECK(cudaGetLastError());
    });
  }
}

// Misaligned pointers (exercises fallback path without 16-byte vectorization).
TEST_F(SimpleCopyTest, CopyMisaligned) {
  struct Config {
    int srcOff;
    int dstOff;
    int64_t nElts;
  };

  constexpr Config configs[] = {
      {1, 1, 4000000},
      {3, 3, 3000007},
      {1, 3, 4100000},
      {0, 1, 3999999},
  };

  for (const auto& cfg : configs) {
    SCOPED_TRACE(
        "srcOff=" + std::to_string(cfg.srcOff) + " dstOff=" +
        std::to_string(cfg.dstOff) + " nElts=" + std::to_string(cfg.nElts));
    verifyCopy(
        d_src + cfg.srcOff,
        d_dst + cfg.dstOff,
        cfg.nElts,
        [&](int* dst, int* src, ssize_t n) {
          copy_kernel<4, int><<<kDefaultBlocks, kBlockSize>>>(dst, src, n);
          CUDACHECK(cudaGetLastError());
        });
  }
}

// Edge cases with small element counts (stresses tail/partial logic).
TEST_F(SimpleCopyTest, CopySmallSizes) {
  constexpr int64_t sizes[] = {0, 1, 2, 31, 32, 33, 127, 128, 129};

  for (int64_t nElts : sizes) {
    SCOPED_TRACE("nElts=" + std::to_string(nElts));
    verifyCopy(d_src, d_dst, nElts, [&](int* dst, int* src, ssize_t n) {
      copy_kernel<4, int><<<kDefaultBlocks, kBlockSize>>>(dst, src, n);
      CUDACHECK(cudaGetLastError());
    });
  }
}

// All Unroll values with a fixed size.
TEST_F(SimpleCopyTest, CopyUnrollVariants) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  auto test = [&](auto unrollTag) {
    constexpr int U = decltype(unrollTag)::value;
    SCOPED_TRACE("Unroll=" + std::to_string(U));
    verifyCopy(d_src, d_dst, nElts, [&](int* dst, int* src, ssize_t n) {
      copy_kernel<U, int><<<kDefaultBlocks, kBlockSize>>>(dst, src, n);
      CUDACHECK(cudaGetLastError());
    });
  };

  test(std::integral_constant<int, 1>{});
  test(std::integral_constant<int, 2>{});
  test(std::integral_constant<int, 4>{});
  test(std::integral_constant<int, 8>{});
}

// =============================================================================
// Reduce-copy tests
// =============================================================================

// 2-source reduce-copy: verify dst[i] == src0[i] + src1[i].
TEST_F(SimpleCopyTest, ReduceCopy2Sources) {
  constexpr int64_t sizes[] = {
      32, 1024, 4L * 1024L * 1024L, 1000, 3000007, 4100000};

  for (int64_t nElts : sizes) {
    SCOPED_TRACE("nElts=" + std::to_string(nElts));

    std::vector<int> h_src0(nElts);
    std::vector<int> h_src1(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      h_src0[i] = static_cast<int>(i);
      h_src1[i] = static_cast<int>(i * 2 + 1);
    }
    CUDACHECK(cudaMemcpy(
        d_src, h_src0.data(), nElts * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(
        d_src1, h_src1.data(), nElts * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(d_dst, 0, nElts * sizeof(int)));

    reduce_copy_2src_kernel<4, int>
        <<<kDefaultBlocks, kBlockSize>>>(d_dst, d_src, d_src1, nElts);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<int> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(), d_dst, nElts * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> expected(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      expected[i] = h_src0[i] + h_src1[i];
    }
    EXPECT_EQ(h_dst, expected);
  }
}

// 3-source reduce-copy: verify dst[i] == src0[i] + src1[i] + src2[i].
TEST_F(SimpleCopyTest, ReduceCopy3Sources) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<int> h_src0(nElts);
  std::vector<int> h_src1(nElts);
  std::vector<int> h_src2(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_src0[i] = static_cast<int>(i);
    h_src1[i] = static_cast<int>(i * 2 + 1);
    h_src2[i] = static_cast<int>(i * 3 + 7);
  }
  CUDACHECK(cudaMemcpy(
      d_src, h_src0.data(), nElts * sizeof(int), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_src1, h_src1.data(), nElts * sizeof(int), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_src2, h_src2.data(), nElts * sizeof(int), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dst, 0, nElts * sizeof(int)));

  reduce_copy_3src_kernel<4, int>
      <<<kDefaultBlocks, kBlockSize>>>(d_dst, d_src, d_src1, d_src2, nElts);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<int> h_dst(nElts);
  CUDACHECK(cudaMemcpy(
      h_dst.data(), d_dst, nElts * sizeof(int), cudaMemcpyDeviceToHost));

  std::vector<int> expected(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    expected[i] = h_src0[i] + h_src1[i] + h_src2[i];
  }
  EXPECT_EQ(h_dst, expected);
}

// 1-source reduce-copy: degenerates to plain copy.
TEST_F(SimpleCopyTest, ReduceCopy1Source) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<int> h_src(nElts);
  std::iota(h_src.begin(), h_src.end(), 0);
  CUDACHECK(cudaMemcpy(
      d_src, h_src.data(), nElts * sizeof(int), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dst, 0, nElts * sizeof(int)));

  reduce_copy_1src_kernel<4, int>
      <<<kDefaultBlocks, kBlockSize>>>(d_dst, d_src, nElts);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<int> h_dst(nElts);
  CUDACHECK(cudaMemcpy(
      h_dst.data(), d_dst, nElts * sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(h_src, h_dst);
}

// Reduce-copy with small sizes.
TEST_F(SimpleCopyTest, ReduceCopySmallSizes) {
  constexpr int64_t sizes[] = {0, 1, 2, 31, 32, 33, 127, 128, 129};

  for (int64_t nElts : sizes) {
    SCOPED_TRACE("nElts=" + std::to_string(nElts));
    if (nElts == 0) {
      reduce_copy_2src_kernel<4, int>
          <<<kDefaultBlocks, kBlockSize>>>(d_dst, d_src, d_src1, 0);
      CUDACHECK(cudaGetLastError());
      CUDACHECK(cudaDeviceSynchronize());
      continue;
    }

    std::vector<int> h_src0(nElts);
    std::vector<int> h_src1(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      h_src0[i] = static_cast<int>(i + 1);
      h_src1[i] = static_cast<int>(i * 3);
    }
    CUDACHECK(cudaMemcpy(
        d_src, h_src0.data(), nElts * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(
        d_src1, h_src1.data(), nElts * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(d_dst, 0, nElts * sizeof(int)));

    reduce_copy_2src_kernel<4, int>
        <<<kDefaultBlocks, kBlockSize>>>(d_dst, d_src, d_src1, nElts);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<int> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(), d_dst, nElts * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> expected(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      expected[i] = h_src0[i] + h_src1[i];
    }
    EXPECT_EQ(h_dst, expected);
  }
}

// Reduce-copy with misaligned pointers.
TEST_F(SimpleCopyTest, ReduceCopyMisaligned) {
  struct Config {
    int src0Off;
    int src1Off;
    int dstOff;
    int64_t nElts;
  };

  constexpr Config configs[] = {
      {1, 1, 1, 4000000},
      {3, 3, 3, 3000007},
      {1, 3, 0, 4100000},
      {0, 1, 3, 3999999},
  };

  for (const auto& cfg : configs) {
    SCOPED_TRACE(
        "src0Off=" + std::to_string(cfg.src0Off) + " src1Off=" +
        std::to_string(cfg.src1Off) + " dstOff=" + std::to_string(cfg.dstOff) +
        " nElts=" + std::to_string(cfg.nElts));

    int* src0 = d_src + cfg.src0Off;
    int* src1 = d_src1 + cfg.src1Off;
    int* dst = d_dst + cfg.dstOff;

    std::vector<int> h_src0(cfg.nElts);
    std::vector<int> h_src1(cfg.nElts);
    for (int64_t i = 0; i < cfg.nElts; i++) {
      h_src0[i] = static_cast<int>(i);
      h_src1[i] = static_cast<int>(i * 2 + 1);
    }
    CUDACHECK(cudaMemcpy(
        src0, h_src0.data(), cfg.nElts * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(
        src1, h_src1.data(), cfg.nElts * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(dst, 0, cfg.nElts * sizeof(int)));

    reduce_copy_2src_kernel<4, int>
        <<<kDefaultBlocks, kBlockSize>>>(dst, src0, src1, cfg.nElts);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<int> h_dst(cfg.nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(), dst, cfg.nElts * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> expected(cfg.nElts);
    for (int64_t i = 0; i < cfg.nElts; i++) {
      expected[i] = h_src0[i] + h_src1[i];
    }
    EXPECT_EQ(h_dst, expected);
  }
}

// Reduce-copy with different Unroll values.
TEST_F(SimpleCopyTest, ReduceCopyUnrollVariants) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<int> h_src0(nElts);
  std::vector<int> h_src1(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_src0[i] = static_cast<int>(i);
    h_src1[i] = static_cast<int>(i * 2 + 1);
  }
  CUDACHECK(cudaMemcpy(
      d_src, h_src0.data(), nElts * sizeof(int), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_src1, h_src1.data(), nElts * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<int> expected(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    expected[i] = h_src0[i] + h_src1[i];
  }

  auto test = [&](auto unrollTag) {
    constexpr int U = decltype(unrollTag)::value;
    SCOPED_TRACE("Unroll=" + std::to_string(U));
    CUDACHECK(cudaMemset(d_dst, 0, nElts * sizeof(int)));

    reduce_copy_2src_kernel<U, int>
        <<<kDefaultBlocks, kBlockSize>>>(d_dst, d_src, d_src1, nElts);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<int> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(), d_dst, nElts * sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_dst, expected);
  };

  test(std::integral_constant<int, 1>{});
  test(std::integral_constant<int, 2>{});
  test(std::integral_constant<int, 4>{});
  test(std::integral_constant<int, 8>{});
}
