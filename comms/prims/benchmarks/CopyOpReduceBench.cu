// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/benchmarks/CopyOpReduceBench.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/utils/CudaRAII.h"

namespace comms::prims::benchmark {
namespace {

constexpr int kBlockSize = 384;
constexpr int kBlocks = 1;

using StagedPolicy = TileReduceStaged<float, SumOp, 24576, kBlockSize>;
using SmemPolicy = CpAsyncSmemReduce<float, SumOp, 8192, kBlockSize, 2>;

template <typename Policy>
__global__ void reduceKernel(
    float* output,
    const float* staging,
    const float* local,
    std::size_t nbytes) {
  auto group = make_block_group();
  Policy::recv(
      reinterpret_cast<char*>(output),
      reinterpret_cast<const char*>(staging),
      nbytes,
      group,
      0,
      reinterpret_cast<const char*>(local));
}

void checkCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void launchPolicy(
    CopyOpReducePolicy policy,
    float* output,
    const float* staging,
    const float* local,
    std::size_t nbytes,
    std::size_t dynamicSmem) {
  switch (policy) {
    case CopyOpReducePolicy::TileReduceStaged:
      reduceKernel<StagedPolicy>
          <<<kBlocks, kBlockSize>>>(output, staging, local, nbytes);
      return;
    case CopyOpReducePolicy::CpAsyncSmemReduce:
      reduceKernel<SmemPolicy><<<kBlocks, kBlockSize, dynamicSmem>>>(
          output, staging, local, nbytes);
      return;
  }
  throw std::invalid_argument("unknown CopyOp reduce policy");
}

std::size_t smemForPolicy(CopyOpReducePolicy policy) {
  return policy == CopyOpReducePolicy::CpAsyncSmemReduce
      ? SmemPolicy::smem_bytes()
      : 0;
}

} // namespace

CopyOpReduceTiming runCopyOpReduceBenchmark(
    CopyOpReducePolicy policy,
    std::size_t nbytes,
    int iterations) {
  const std::size_t allocationBytes = nbytes * kBlocks;
  meta::comms::DeviceBuffer staging(allocationBytes);
  meta::comms::DeviceBuffer local(allocationBytes);
  meta::comms::DeviceBuffer output(allocationBytes);
  checkCuda(
      cudaMemset(staging.get(), 1, allocationBytes), "initialize staging");
  checkCuda(cudaMemset(local.get(), 2, allocationBytes), "initialize local");

  const std::size_t dynamicSmem = smemForPolicy(policy);
  if (dynamicSmem > 0) {
    checkCuda(
        cudaFuncSetAttribute(
            reduceKernel<SmemPolicy>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(dynamicSmem)),
        "set dynamic shared memory");
  }

  auto* outputPtr = static_cast<float*>(output.get());
  const auto* stagingPtr = static_cast<const float*>(staging.get());
  const auto* localPtr = static_cast<const float*>(local.get());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  checkCuda(cudaEventCreate(&start), "create start event");
  checkCuda(cudaEventCreate(&stop), "create stop event");

  launchPolicy(policy, outputPtr, stagingPtr, localPtr, nbytes, dynamicSmem);
  checkCuda(cudaGetLastError(), "warmup launch");
  checkCuda(cudaDeviceSynchronize(), "warmup synchronize");

  checkCuda(cudaEventRecord(start), "record start");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    launchPolicy(policy, outputPtr, stagingPtr, localPtr, nbytes, dynamicSmem);
    checkCuda(cudaGetLastError(), "benchmark launch");
  }
  checkCuda(cudaEventRecord(stop), "record stop");
  checkCuda(cudaEventSynchronize(stop), "benchmark synchronize");

  float elapsedMs = 0;
  checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "measure elapsed");
  checkCuda(cudaEventDestroy(stop), "destroy stop event");
  checkCuda(cudaEventDestroy(start), "destroy start event");

  const float operations = static_cast<float>(iterations);
  const float timeUs = elapsedMs * 1000.0f / operations;
  const float payloadGBps =
      static_cast<float>(allocationBytes) / 1000.0f / timeUs;
  return CopyOpReduceTiming{
      .timeUs = timeUs,
      .payloadGBps = payloadGBps,
      .memoryGBps = 3.0f * payloadGBps,
  };
}

} // namespace comms::prims::benchmark
