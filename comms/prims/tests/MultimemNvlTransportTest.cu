// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/tests/MultimemNvlTransportTest.cuh"

#include <type_traits>

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/transport/nvl/MultimemNvlReduce.cuh"
#include "comms/prims/transport/nvl/MultimemNvlStageLayout.cuh"

namespace comms::prims::test {

namespace {

__global__ void setUserSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value) {
  auto group = make_warp_group();
  transport.signal(group, signalId, SignalOp::SIGNAL_SET, value);
}

__global__ void setInternalSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value) {
  auto group = make_warp_group();
  transport.signal_internal(group, signalId, SignalOp::SIGNAL_SET, value);
}

__global__ void addUserSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value) {
  auto group = make_warp_group();
  transport.signal(group, signalId, SignalOp::SIGNAL_ADD, value);
}

__global__ void addInternalSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value) {
  auto group = make_warp_group();
  transport.signal_internal(group, signalId, SignalOp::SIGNAL_ADD, value);
}

__global__ void waitAndReadUserSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out) {
  auto group = make_warp_group();
  transport.wait_signal_until(group, signalId, op, expected);
  if (group.is_leader()) {
    *out = transport.read_signal(signalId);
  }
}

__global__ void waitAndReadInternalSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out) {
  auto group = make_warp_group();
  transport.wait_internal_signal_until(group, signalId, op, expected);
  if (group.is_leader()) {
    *out = transport.read_internal_signal(signalId);
  }
}

__global__ void readUserAndInternalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t userId,
    uint64_t internalId,
    uint64_t* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = transport.read_signal(userId);
    out[1] = transport.read_internal_signal(internalId);
  }
}

template <typename T>
__device__ T reductionValue(float value) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half(value);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16_rn(value);
  } else {
    return static_cast<T>(value);
  }
}

template <typename T>
__global__ void fillReductionInputKernel(
    MultimemNvlTransportDevice transport,
    float value,
    std::size_t elems,
    std::size_t sourceOffsetElems) {
  auto* source = reinterpret_cast<T*>(transport.localData) + sourceOffsetElems;
  for (std::size_t i = threadIdx.x; i < elems; i += blockDim.x) {
    source[i] = reductionValue<T>(value);
  }
}

template <typename T, bool kAccF32>
__global__ void loadReduceKernel(
    MultimemNvlTransportDevice transport,
    T* output,
    std::size_t elems,
    std::size_t sourceOffsetElems) {
  auto group = make_warp_group();
  const auto* source =
      reinterpret_cast<const T*>(transport.multimemData) + sourceOffsetElems;
  multimem::load_reduce_at<T, multimem::MultimemRedOp::Add, kAccF32>(
      group, output, source, elems);
}

__global__ void stageLayoutKernel(
    MultimemNvlTransportDevice transport,
    StageLayoutResult* results) {
  auto group = make_block_group();
  const auto layout = multimem::make_stage_layout<uint32_t>(transport, group);
  if (group.is_leader()) {
    results[group.group_id] = StageLayoutResult{
        .groupBeginBytes = layout.groupBeginBytes,
        .stagingBytes = layout.stagingBytes,
        .signalBase = layout.signalBase,
        .signalsPerLane = layout.signalsPerLane,
        .pipelineDepth = layout.pipelineDepth,
    };
  }
}

template <typename T>
void launchFillReductionInputTyped(
    MultimemNvlTransportDevice transport,
    float value,
    std::size_t elems,
    std::size_t sourceOffsetElems,
    cudaStream_t stream) {
  fillReductionInputKernel<T>
      <<<1, 32, 0, stream>>>(transport, value, elems, sourceOffsetElems);
  PIPES_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launchLoadReduceTyped(
    MultimemNvlTransportDevice transport,
    bool accF32,
    void* output,
    std::size_t elems,
    std::size_t sourceOffsetElems,
    cudaStream_t stream) {
  if (accF32) {
    loadReduceKernel<T, true><<<1, 32, 0, stream>>>(
        transport, static_cast<T*>(output), elems, sourceOffsetElems);
  } else {
    loadReduceKernel<T, false><<<1, 32, 0, stream>>>(
        transport, static_cast<T*>(output), elems, sourceOffsetElems);
  }
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace

void launchSetUserSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream) {
  setUserSignalKernel<<<1, 32, 0, stream>>>(transport, signalId, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchSetInternalSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream) {
  setInternalSignalKernel<<<1, 32, 0, stream>>>(transport, signalId, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchAddUserSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream) {
  addUserSignalKernel<<<1, 32, 0, stream>>>(transport, signalId, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchAddInternalSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream) {
  addInternalSignalKernel<<<1, 32, 0, stream>>>(transport, signalId, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchWaitAndReadUserSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out,
    cudaStream_t stream) {
  waitAndReadUserSignalKernel<<<1, 32, 0, stream>>>(
      transport, signalId, op, expected, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchWaitAndReadInternalSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out,
    cudaStream_t stream) {
  waitAndReadInternalSignalKernel<<<1, 32, 0, stream>>>(
      transport, signalId, op, expected, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchReadUserAndInternal(
    MultimemNvlTransportDevice transport,
    uint64_t userId,
    uint64_t internalId,
    uint64_t* out,
    cudaStream_t stream) {
  readUserAndInternalKernel<<<1, 32, 0, stream>>>(
      transport, userId, internalId, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchFillReductionInput(
    MultimemNvlTransportDevice transport,
    MultimemReductionTestType type,
    float value,
    std::size_t elems,
    std::size_t sourceOffsetElems,
    cudaStream_t stream) {
  switch (type) {
    case MultimemReductionTestType::Float:
      return launchFillReductionInputTyped<float>(
          transport, value, elems, sourceOffsetElems, stream);
    case MultimemReductionTestType::Int32:
      return launchFillReductionInputTyped<int32_t>(
          transport, value, elems, sourceOffsetElems, stream);
    case MultimemReductionTestType::Float16:
      return launchFillReductionInputTyped<__half>(
          transport, value, elems, sourceOffsetElems, stream);
    case MultimemReductionTestType::Bfloat16:
      return launchFillReductionInputTyped<__nv_bfloat16>(
          transport, value, elems, sourceOffsetElems, stream);
  }
}

void launchLoadReduce(
    MultimemNvlTransportDevice transport,
    MultimemReductionTestType type,
    bool accF32,
    void* output,
    std::size_t elems,
    std::size_t sourceOffsetElems,
    cudaStream_t stream) {
  switch (type) {
    case MultimemReductionTestType::Float:
      return launchLoadReduceTyped<float>(
          transport, accF32, output, elems, sourceOffsetElems, stream);
    case MultimemReductionTestType::Int32:
      return launchLoadReduceTyped<int32_t>(
          transport, accF32, output, elems, sourceOffsetElems, stream);
    case MultimemReductionTestType::Float16:
      return launchLoadReduceTyped<__half>(
          transport, accF32, output, elems, sourceOffsetElems, stream);
    case MultimemReductionTestType::Bfloat16:
      return launchLoadReduceTyped<__nv_bfloat16>(
          transport, accF32, output, elems, sourceOffsetElems, stream);
  }
}

void launchStageLayout(
    MultimemNvlTransportDevice transport,
    StageLayoutResult* results,
    uint32_t numGroups,
    cudaStream_t stream) {
  stageLayoutKernel<<<numGroups, 32, 0, stream>>>(transport, results);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
