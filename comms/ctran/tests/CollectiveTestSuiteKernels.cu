// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/tests/CollectiveTestSuiteKernels.cuh"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {

void checkCuda(
    cudaError_t result,
    const char* expr,
    const char* file,
    int line) {
  if (result != cudaSuccess) {
    std::fprintf(
        stderr,
        "Failed: Cuda error %s:%d '%s' while running %s\n",
        file,
        line,
        cudaGetErrorString(result),
        expr);
    std::abort();
  }
}

#define COLLECTIVE_TEST_SUITE_CUDA_CHECK(cmd) \
  checkCuda((cmd), #cmd, __FILE__, __LINE__)

struct CudaFreeDeleter {
  void operator()(void* ptr) const {
    if (ptr != nullptr) {
      COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaFree(ptr));
    }
  }
};

template <typename T>
std::unique_ptr<T, CudaFreeDeleter> makeDeviceScratch(size_t count) {
  T* ptr = nullptr;
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
  return std::unique_ptr<T, CudaFreeDeleter>(ptr);
}

} // namespace

template <typename T>
struct TestValueOps;

template <>
struct TestValueOps<float> {
  using Word = uint32_t;
  __device__ __forceinline__ static float fromFloat(float value) {
    return value;
  }
  __device__ __forceinline__ static double toDouble(float value) {
    return static_cast<double>(value);
  }
};

template <>
struct TestValueOps<__half> {
  using Word = uint16_t;
  __device__ __forceinline__ static __half fromFloat(float value) {
    return __float2half(value);
  }
  __device__ __forceinline__ static double toDouble(__half value) {
    return static_cast<double>(__half2float(value));
  }
};

// Per-element value: 1.0f / (1.0f + (rep + rank + i) % 256)
// This produces different values per element and per rank.
__device__ static float elementValue(int rank, int rep, size_t i) {
  return 1.0f / (1.0f + static_cast<float>((rep + rank + i) % 256));
}

template <typename T>
__global__ void initDataKernel(T* buf, size_t count, int rank, int rep) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    buf[idx] = TestValueOps<T>::fromFloat(elementValue(rank, rep, idx));
  }
}

template <typename T>
__global__ void initExpectedKernel(T* buf, size_t count, int nranks, int rep) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    float sum = 0.0f;
    for (int r = 0; r < nranks; r++) {
      sum += elementValue(r, rep, idx);
    }
    buf[idx] = TestValueOps<T>::fromFloat(sum);
  }
}

template <typename T>
__global__ void initScatterExpectedKernel(
    T* buf,
    size_t recvcount,
    int nranks,
    int rep,
    size_t baseIdx) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < recvcount) {
    float sum = 0.0f;
    for (int r = 0; r < nranks; ++r) {
      sum += elementValue(r, rep, baseIdx + i);
    }
    buf[i] = TestValueOps<T>::fromFloat(sum);
  }
}

// Each block computes the max abs delta over its assigned elements, then
// writes the per-block maximum into blockMaxes[blockIdx.x].
template <typename T>
__global__ void deltaKernel(
    const T* actual,
    const T* expected,
    size_t count,
    double* blockMaxes) {
  extern __shared__ double sdata[];

  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  double localMax = 0.0;
  if (idx < count) {
    localMax = fabs(
        TestValueOps<T>::toDouble(actual[idx]) -
        TestValueOps<T>::toDouble(expected[idx]));
  }

  sdata[threadIdx.x] = localMax;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] = fmax(sdata[threadIdx.x], sdata[threadIdx.x + s]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    blockMaxes[blockIdx.x] = sdata[0];
  }
}

__device__ __forceinline__ uint64_t mixChecksum(uint64_t acc, uint64_t value) {
  acc ^= value + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
  return acc * 0x100000001b3ULL;
}

template <typename T>
__global__ void
rawBitsChecksumKernel(const T* data, size_t count, uint64_t* result) {
  constexpr int kBlockSize = 256;
  __shared__ uint64_t partials[kBlockSize];

  using Word = typename TestValueOps<T>::Word;
  const Word* words = reinterpret_cast<const Word*>(data);
  uint64_t local = 0xcbf29ce484222325ULL ^
      (static_cast<uint64_t>(threadIdx.x) << 32) ^ count ^
      (static_cast<uint64_t>(sizeof(T)) << 56);
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    const uint64_t taggedValue =
        (static_cast<uint64_t>(words[i]) << 32) ^ static_cast<uint64_t>(i);
    local = mixChecksum(local, taggedValue);
  }

  partials[threadIdx.x] = local;
  __syncthreads();

  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partials[threadIdx.x] =
          mixChecksum(partials[threadIdx.x], partials[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *result = partials[0];
  }
}

void launchInitDataKernel(
    float* buf,
    size_t count,
    int rank,
    int rep,
    cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  constexpr int kBlockSize = 256;
  int numBlocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
  initDataKernel<<<numBlocks, kBlockSize, 0, stream>>>(buf, count, rank, rep);
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaPeekAtLastError());
}

void launchInitDataKernel(
    __half* buf,
    size_t count,
    int rank,
    int rep,
    cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  constexpr int kBlockSize = 256;
  int numBlocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
  initDataKernel<<<numBlocks, kBlockSize, 0, stream>>>(buf, count, rank, rep);
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaPeekAtLastError());
}

void launchInitExpectedKernel(
    float* buf,
    size_t count,
    int nranks,
    int rep,
    cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  constexpr int kBlockSize = 256;
  int numBlocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
  initExpectedKernel<<<numBlocks, kBlockSize, 0, stream>>>(
      buf, count, nranks, rep);
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaPeekAtLastError());
}

void launchInitExpectedKernel(
    __half* buf,
    size_t count,
    int nranks,
    int rep,
    cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  constexpr int kBlockSize = 256;
  int numBlocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
  initExpectedKernel<<<numBlocks, kBlockSize, 0, stream>>>(
      buf, count, nranks, rep);
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaPeekAtLastError());
}

void launchInitScatterExpectedKernel(
    float* buf,
    size_t recvcount,
    int nranks,
    int rep,
    size_t baseIdx,
    cudaStream_t stream) {
  if (recvcount == 0) {
    return;
  }
  constexpr int kBlockSize = 256;
  int numBlocks = static_cast<int>((recvcount + kBlockSize - 1) / kBlockSize);
  initScatterExpectedKernel<<<numBlocks, kBlockSize, 0, stream>>>(
      buf, recvcount, nranks, rep, baseIdx);
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaPeekAtLastError());
}

void launchInitScatterExpectedKernel(
    __half* buf,
    size_t recvcount,
    int nranks,
    int rep,
    size_t baseIdx,
    cudaStream_t stream) {
  if (recvcount == 0) {
    return;
  }
  constexpr int kBlockSize = 256;
  int numBlocks = static_cast<int>((recvcount + kBlockSize - 1) / kBlockSize);
  initScatterExpectedKernel<<<numBlocks, kBlockSize, 0, stream>>>(
      buf, recvcount, nranks, rep, baseIdx);
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
double computeMaxDeltaTyped(
    const T* actual,
    const T* expected,
    size_t count,
    cudaStream_t stream) {
  if (count == 0) {
    return 0.0;
  }

  constexpr int kBlockSize = 256;
  int numBlocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);

  auto dBlockMaxes = makeDeviceScratch<double>(numBlocks);

  size_t sharedBytes = kBlockSize * sizeof(double);
  deltaKernel<<<numBlocks, kBlockSize, sharedBytes, stream>>>(
      actual, expected, count, dBlockMaxes.get());
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaPeekAtLastError());

  std::vector<double> hBlockMaxes(numBlocks);
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaMemcpyAsync(
      hBlockMaxes.data(),
      dBlockMaxes.get(),
      numBlocks * sizeof(double),
      cudaMemcpyDeviceToHost,
      stream));
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaStreamSynchronize(stream));

  double maxDelta = 0.0;
  for (int i = 0; i < numBlocks; i++) {
    maxDelta = std::fmax(maxDelta, hBlockMaxes[i]);
  }

  return maxDelta;
}

double computeMaxDelta(
    const float* actual,
    const float* expected,
    size_t count,
    cudaStream_t stream) {
  return computeMaxDeltaTyped(actual, expected, count, stream);
}

double computeMaxDelta(
    const __half* actual,
    const __half* expected,
    size_t count,
    cudaStream_t stream) {
  return computeMaxDeltaTyped(actual, expected, count, stream);
}

template <typename T>
uint64_t
computeRawBitsChecksumTyped(const T* data, size_t count, cudaStream_t stream) {
  if (count == 0) {
    return 0;
  }

  auto dChecksum = makeDeviceScratch<uint64_t>(1);

  constexpr int kBlockSize = 256;
  rawBitsChecksumKernel<<<1, kBlockSize, 0, stream>>>(
      data, count, dChecksum.get());
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaPeekAtLastError());

  uint64_t hChecksum = 0;
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaMemcpyAsync(
      &hChecksum,
      dChecksum.get(),
      sizeof(uint64_t),
      cudaMemcpyDeviceToHost,
      stream));
  COLLECTIVE_TEST_SUITE_CUDA_CHECK(cudaStreamSynchronize(stream));

  return hChecksum;
}

uint64_t
computeRawBitsChecksum(const float* data, size_t count, cudaStream_t stream) {
  return computeRawBitsChecksumTyped(data, count, stream);
}

uint64_t
computeRawBitsChecksum(const __half* data, size_t count, cudaStream_t stream) {
  return computeRawBitsChecksumTyped(data, count, stream);
}
