// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/BroadcastTest.cuh"

#include "comms/common/CudaWrap.h"
#include "comms/pipes/collectives/broadcast/BroadcastTopologies.cuh"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::collectives::test {

// =============================================================================
// Templated Kernel for Direct Template API Testing
// =============================================================================

/**
 * Device kernel template that directly invokes broadcast<TopologyTag>().
 *
 * This kernel template is the primary way to test the broadcast template API.
 * It exercises the full template dispatch mechanism: TopologyTag ->
 * TopologyTraits
 * -> execute().
 */
template <typename TopologyTag>
__global__ void testBroadcastKernel(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes) {
  broadcast<TopologyTag>(
      buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
}

/**
 * Host wrapper template that launches testBroadcastKernel<TopologyTag>.
 */
template <typename TopologyTag>
void testBroadcast(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim,
    cudaStream_t stream) {
  void* args[] = {
      &buff_d, &my_rank_id, &root_rank_id, &transports_per_rank, &nbytes};
  PIPES_CUDA_CHECK(
      comms::common::launchKernel(
          (void*)testBroadcastKernel<TopologyTag>,
          dim3(numBlocks, 1, 1),
          dim3(blockSize, 1, 1),
          args,
          stream,
          clusterDim));
  PIPES_KERNEL_LAUNCH_CHECK();
}

// Explicit instantiations for the three topology tags
template void testBroadcast<FlatTag>(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim,
    cudaStream_t stream);

template void testBroadcast<RingTag>(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim,
    cudaStream_t stream);

template void testBroadcast<BinomialTreeTag>(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim,
    cudaStream_t stream);

// =============================================================================
// Legacy Non-Templated Wrappers (delegate to template versions)
// =============================================================================

void testBroadcastFlat(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim,
    cudaStream_t stream) {
  testBroadcast<FlatTag>(
      buff_d,
      my_rank_id,
      root_rank_id,
      transports_per_rank,
      nbytes,
      numBlocks,
      blockSize,
      clusterDim,
      stream);
}

void testBroadcastBinomialTree(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim,
    cudaStream_t stream) {
  testBroadcast<BinomialTreeTag>(
      buff_d,
      my_rank_id,
      root_rank_id,
      transports_per_rank,
      nbytes,
      numBlocks,
      blockSize,
      clusterDim,
      stream);
}

void testBroadcastRing(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim,
    cudaStream_t stream) {
  testBroadcast<RingTag>(
      buff_d,
      my_rank_id,
      root_rank_id,
      transports_per_rank,
      nbytes,
      numBlocks,
      blockSize,
      clusterDim,
      stream);
}

/**
 * Device kernel that invokes the adaptive broadcast logic.
 * Selects between FlatTag and RingTag based on message size.
 */
__global__ void testBroadcastAdaptiveKernel(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes) {
  // Adaptive logic: use RingTag for large messages (>= 4MB) with nranks > 2
  constexpr std::size_t kRingThreshold = 4 * 1024 * 1024;
  const auto nranks = transports_per_rank.size();

  if (nbytes >= kRingThreshold && nranks > 2) {
    broadcast<RingTag>(
        buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
  } else {
    broadcast<FlatTag>(
        buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
  }
}

void testBroadcastAdaptive(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim,
    cudaStream_t stream) {
  void* args[] = {
      &buff_d, &my_rank_id, &root_rank_id, &transports_per_rank, &nbytes};
  PIPES_CUDA_CHECK(
      comms::common::launchKernel(
          (void*)testBroadcastAdaptiveKernel,
          dim3(numBlocks, 1, 1),
          dim3(blockSize, 1, 1),
          args,
          stream,
          clusterDim));
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::collectives::test
