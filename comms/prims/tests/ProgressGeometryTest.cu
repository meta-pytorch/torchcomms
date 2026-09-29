// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/tests/ProgressGeometryTest.cuh"
#include "comms/prims/transport/P2pIbTransportProgressImpl.cuh"

namespace comms::prims::test {
namespace {

__global__ void chunk_payload_kernel(
    IbChannelLayout layout,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    std::size_t* out) {
  ThreadGroup g{
      threadIdx.x,
      blockDim.x,
      blockIdx.x,
      blockIdx.x,
      gridDim.x,
      SyncScope::BLOCK};

  const detail::ProgressGeometry geometry =
      detail::make_progress_geometry<protocol::LL>(
          layout, g, nbytes, maxSignalBytes, "ProgressGeometryTest");

  if (g.is_leader()) {
    *out = geometry.chunkPayload;
  }
}

__global__ void blocking_chunk_payload_kernel(
    IbChannelLayout layout,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    std::size_t* out) {
  ThreadGroup g{
      threadIdx.x,
      blockDim.x,
      blockIdx.x,
      blockIdx.x,
      gridDim.x,
      SyncScope::BLOCK};

  const detail::SendRecvGeometry geometry =
      detail::calcGeometry(protocol::LL{}, layout, g, nbytes, maxSignalBytes);

  if (g.is_leader()) {
    *out = geometry.chunkPayload;
  }
}

IbChannelLayout makeLayout(
    std::size_t perChannelBufferSize,
    int pipelineDepth) {
  IbChannelLayout layout{};
  // One logical channel, so group_id 0 stays in bounds.
  layout.numChannels = 1;
  layout.maxChannels = 1;
  layout.numLanes = 1;
  layout.pipelineDepth = pipelineDepth;
  layout.perChannelBufferSize = perChannelBufferSize;
  return layout;
}

} // namespace

void launch_ll_chunk_payload(
    std::size_t perChannelBufferSize,
    int pipelineDepth,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    std::size_t* out_d) {
  chunk_payload_kernel<<<1, 32>>>(
      makeLayout(perChannelBufferSize, pipelineDepth),
      nbytes,
      maxSignalBytes,
      out_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launch_ll_blocking_chunk_payload(
    std::size_t perChannelBufferSize,
    int pipelineDepth,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    std::size_t* out_d) {
  blocking_chunk_payload_kernel<<<1, 32>>>(
      makeLayout(perChannelBufferSize, pipelineDepth),
      nbytes,
      maxSignalBytes,
      out_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
