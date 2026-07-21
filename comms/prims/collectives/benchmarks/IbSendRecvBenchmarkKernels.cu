// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Benchmark-only CUDA kernel: zero-copy RDMA put with BLOCK-scope ops.
// Not part of the library — used only by IbSendRecvBenchmark.

#include "comms/prims/collectives/benchmarks/IbSendRecvBenchmarkKernels.cuh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"

namespace comms::prims::ib::benchmark {

using torchcomms::device::CmpOp;
using torchcomms::device::CoopScope;
using torchcomms::device::DeviceWindowNCCL;
using torchcomms::device::RegisteredBufferNCCL;
using torchcomms::device::SignalOp;

namespace {

constexpr int kPatternThreads = 256;
constexpr int kMaxPatternBlocks = 4096;
constexpr int kChecksumThreads = 256;

void check_cuda(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(result));
    std::abort();
  }
}

struct CudaFreeDeleter {
  void operator()(void* ptr) const {
    if (ptr != nullptr) {
      check_cuda(cudaFree(ptr), "checksum partial free failed");
    }
  }
};

template <typename T>
std::unique_ptr<T, CudaFreeDeleter> make_device_scratch(size_t count) {
  T* ptr = nullptr;
  check_cuda(
      cudaMalloc(&ptr, count * sizeof(T)),
      "checksum partial allocation failed");
  return std::unique_ptr<T, CudaFreeDeleter>(ptr);
}

__host__ __device__ __forceinline__ uint64_t
mix_checksum(uint64_t acc, uint64_t value) {
  acc ^= value + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
  return acc * 0x100000001b3ULL;
}

__host__ __device__ __forceinline__ uint64_t splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__host__ __device__ __forceinline__ uint8_t
rank_pattern_byte(int rank, size_t index) {
  const uint64_t rankSeed = static_cast<uint64_t>(static_cast<uint32_t>(rank))
      << 48;
  return static_cast<uint8_t>(
      splitmix64(rankSeed ^ static_cast<uint64_t>(index)));
}

__global__ void
init_rank_pattern_kernel(uint8_t* data, size_t nbytes, int rank) {
  const size_t stride =
      static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < nbytes;
       i += stride) {
    data[i] = rank_pattern_byte(rank, i);
  }
}

template <bool UseExpectedPattern>
__global__ void byte_checksum_kernel(
    const uint8_t* data,
    size_t nbytes,
    int rank,
    int expected_rank,
    uint64_t* partials,
    MismatchRecord* mismatch_partials) {
  __shared__ uint64_t blockPartials[kChecksumThreads];
  __shared__ uint64_t mismatchIndices[kChecksumThreads];
  __shared__ uint32_t mismatchObserved[kChecksumThreads];
  __shared__ uint32_t mismatchExpected[kChecksumThreads];

  const size_t globalThread =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride =
      static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
  uint64_t local = 0xcbf29ce484222325ULL ^
      (static_cast<uint64_t>(blockIdx.x) << 40) ^
      (static_cast<uint64_t>(threadIdx.x) << 32) ^ nbytes;
  uint64_t localMismatchIndex = kNoMismatchIndex;
  uint32_t localMismatchObserved = 0;
  uint32_t localMismatchExpected = 0;

  for (size_t i = globalThread; i < nbytes; i += stride) {
    uint8_t value = 0;
    if constexpr (UseExpectedPattern) {
      value = rank_pattern_byte(rank, i);
    } else {
      value = data[i];
    }
    const uint64_t taggedValue =
        (static_cast<uint64_t>(value) << 32) ^ static_cast<uint64_t>(i);
    local = mix_checksum(local, taggedValue);

    if (expected_rank >= 0) {
      const uint8_t expected = rank_pattern_byte(expected_rank, i);
      if (value != expected && i < localMismatchIndex) {
        localMismatchIndex = i;
        localMismatchObserved = value;
        localMismatchExpected = expected;
      }
    }
  }

  blockPartials[threadIdx.x] = local;
  mismatchIndices[threadIdx.x] = localMismatchIndex;
  mismatchObserved[threadIdx.x] = localMismatchObserved;
  mismatchExpected[threadIdx.x] = localMismatchExpected;
  __syncthreads();

  for (int strideInBlock = kChecksumThreads / 2; strideInBlock > 0;
       strideInBlock >>= 1) {
    if (threadIdx.x < strideInBlock) {
      blockPartials[threadIdx.x] = mix_checksum(
          blockPartials[threadIdx.x],
          blockPartials[threadIdx.x + strideInBlock]);
      if (mismatchIndices[threadIdx.x + strideInBlock] <
          mismatchIndices[threadIdx.x]) {
        mismatchIndices[threadIdx.x] =
            mismatchIndices[threadIdx.x + strideInBlock];
        mismatchObserved[threadIdx.x] =
            mismatchObserved[threadIdx.x + strideInBlock];
        mismatchExpected[threadIdx.x] =
            mismatchExpected[threadIdx.x + strideInBlock];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    partials[blockIdx.x] = blockPartials[0];
    if (mismatch_partials != nullptr) {
      MismatchRecord record;
      record.index = mismatchIndices[0];
      record.observed = mismatchObserved[0];
      record.expected = mismatchExpected[0];
      mismatch_partials[blockIdx.x] = record;
    }
  }
}

__global__ void checksum_reduce_kernel(
    const uint64_t* partials,
    uint64_t* checksum) {
  __shared__ uint64_t blockPartials[kChecksumPartialBlocks];

  blockPartials[threadIdx.x] = partials[threadIdx.x];
  __syncthreads();

  for (int strideInBlock = kChecksumPartialBlocks / 2; strideInBlock > 0;
       strideInBlock >>= 1) {
    if (threadIdx.x < strideInBlock) {
      blockPartials[threadIdx.x] = mix_checksum(
          blockPartials[threadIdx.x],
          blockPartials[threadIdx.x + strideInBlock]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *checksum = mix_checksum(
        0xcbf29ce484222325ULL ^ static_cast<uint64_t>(kChecksumPartialBlocks),
        blockPartials[0]);
  }
}

__global__ void put_bw_kernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    size_t total_bytes,
    int dst_rank,
    int src_rank,
    int signal_base,
    int iterations) {
  auto pid = blockIdx.x;
  auto num_blks = gridDim.x;
  size_t tile_bytes = total_bytes / num_blks;
  size_t tile_offset = pid * tile_bytes;

  int data_signal = pid;
  int ack_signal = num_blks + pid;

  for (int iter = 0; iter < iterations; iter++) {
    uint64_t expected = static_cast<uint64_t>(signal_base + iter + 1);

    // Put tile to remote, piggyback DATA_READY signal
    win->put(
        tile_offset,
        src_buf,
        tile_offset,
        dst_rank,
        tile_bytes,
        data_signal,
        -1,
        CoopScope::BLOCK);
    win->flush(CoopScope::BLOCK);

    // Wait for remote's data to arrive (their put to us)
    win->wait_signal_from(
        src_rank, data_signal, CmpOp::GE, expected, CoopScope::BLOCK);

    // ACK to remote sender: we received their data
    win->signal(src_rank, ack_signal, SignalOp::ADD, 1, CoopScope::BLOCK);

    // Wait for ACK from our receiver: they received our data
    win->wait_signal_from(
        dst_rank, ack_signal, CmpOp::GE, expected, CoopScope::BLOCK);
  }
}

} // namespace

void launch_put_bw_kernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    size_t total_bytes,
    int dst_rank,
    int src_rank,
    int num_blocks,
    int signal_base,
    int iterations,
    cudaStream_t stream) {
  put_bw_kernel<<<num_blocks, 256, 0, stream>>>(
      win, src_buf, total_bytes, dst_rank, src_rank, signal_base, iterations);
  cudaError_t err = cudaGetLastError();
  assert(err == cudaSuccess && "put_bw_kernel launch failed");
}

void launch_init_rank_pattern_kernel(
    void* data,
    size_t nbytes,
    int rank,
    cudaStream_t stream) {
  if (nbytes == 0) {
    return;
  }
  const int numBlocks = static_cast<int>(std::min<size_t>(
      kMaxPatternBlocks, (nbytes + kPatternThreads - 1) / kPatternThreads));
  init_rank_pattern_kernel<<<numBlocks, kPatternThreads, 0, stream>>>(
      static_cast<uint8_t*>(data), nbytes, rank);
  cudaError_t err = cudaGetLastError();
  assert(err == cudaSuccess && "init_rank_pattern_kernel launch failed");
}

uint8_t expected_rank_pattern_byte(int rank, size_t index) {
  return rank_pattern_byte(rank, index);
}

namespace {

template <bool UseExpectedPattern>
void launch_checksum_impl(
    const void* data,
    size_t nbytes,
    int rank,
    int expected_rank,
    uint64_t* checksum,
    uint64_t* partials,
    MismatchRecord* mismatch_partials,
    cudaStream_t stream) {
  if (nbytes == 0) {
    check_cuda(
        cudaMemsetAsync(checksum, 0, sizeof(uint64_t), stream),
        "zero checksum write failed");
    return;
  }

  byte_checksum_kernel<UseExpectedPattern>
      <<<kChecksumPartialBlocks, kChecksumThreads, 0, stream>>>(
          static_cast<const uint8_t*>(data),
          nbytes,
          rank,
          expected_rank,
          partials,
          mismatch_partials);
  check_cuda(cudaGetLastError(), "byte_checksum_kernel launch failed");

  checksum_reduce_kernel<<<1, kChecksumPartialBlocks, 0, stream>>>(
      partials, checksum);
  check_cuda(cudaGetLastError(), "checksum_reduce_kernel launch failed");
}

template <bool UseExpectedPattern>
uint64_t compute_checksum_impl(
    const void* data,
    size_t nbytes,
    int rank,
    cudaStream_t stream) {
  if (nbytes == 0) {
    return 0;
  }

  auto deviceChecksum = make_device_scratch<uint64_t>(1);
  auto devicePartials = make_device_scratch<uint64_t>(kChecksumPartialBlocks);
  launch_checksum_impl<UseExpectedPattern>(
      data,
      nbytes,
      rank,
      /*expected_rank=*/-1,
      deviceChecksum.get(),
      devicePartials.get(),
      /*mismatch_partials=*/nullptr,
      stream);

  uint64_t hostChecksum = 0;
  check_cuda(
      cudaMemcpyAsync(
          &hostChecksum,
          deviceChecksum.get(),
          sizeof(uint64_t),
          cudaMemcpyDeviceToHost,
          stream),
      "checksum copy failed");
  check_cuda(cudaStreamSynchronize(stream), "checksum stream sync failed");

  return hostChecksum;
}

} // namespace

void launch_buffer_checksum(
    const void* data,
    size_t nbytes,
    int expected_rank,
    uint64_t* checksum,
    uint64_t* partials,
    MismatchRecord* mismatch_partials,
    cudaStream_t stream) {
  launch_checksum_impl</*UseExpectedPattern=*/false>(
      data,
      nbytes,
      /*rank=*/0,
      expected_rank,
      checksum,
      partials,
      mismatch_partials,
      stream);
}

void launch_rank_pattern_checksum(
    size_t nbytes,
    int rank,
    uint64_t* checksum,
    uint64_t* partials,
    cudaStream_t stream) {
  launch_checksum_impl</*UseExpectedPattern=*/true>(
      nullptr,
      nbytes,
      rank,
      /*expected_rank=*/-1,
      checksum,
      partials,
      /*mismatch_partials=*/nullptr,
      stream);
}

uint64_t
compute_buffer_checksum(const void* data, size_t nbytes, cudaStream_t stream) {
  return compute_checksum_impl</*UseExpectedPattern=*/false>(
      data, nbytes, /*rank=*/0, stream);
}

uint64_t
compute_rank_pattern_checksum(size_t nbytes, int rank, cudaStream_t stream) {
  return compute_checksum_impl</*UseExpectedPattern=*/true>(
      nullptr, nbytes, rank, stream);
}

} // namespace comms::prims::ib::benchmark
