// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <optional>

namespace comms::common {

// Default cluster size for spread cluster launch: this is the number of blocks
// in a cluster; all blocks in a cluster are launched on the sam GPC.
constexpr int kDefaultClusterSize = 4;

/**
 * launchKernel - Launch a kernel with optional cluster configuration
 *
 * This function provides a unified interface for launching kernels either
 * with standard cudaLaunchKernel or with clustered spread scheduling via
 * cudaLaunchKernelExC.
 *
 * When cluster launch is enabled (clusterDim has value):
 * - Sets up cudaLaunchAttributeClusterDimension with the specified cluster size
 * - Uses cudaClusterSchedulingPolicySpread to spread clusters across GPCs
 * - This can reduce memory stall and L2 cache churn for better performance
 *
 * @param kernelFunc Pointer to the kernel function
 * @param gridDim Grid dimensions for the kernel launch
 * @param blockDim Block dimensions for the kernel launch
 * @param args Array of pointers to kernel arguments
 * @param stream CUDA stream for the kernel launch
 * @param clusterDim Optional cluster dimensions. If nullopt, standard
 *                   kernel launch is used; otherwise clustered launch is used.
 * @return cudaError_t CUDA error code from the kernel launch
 *
 * Example usage:
 *   // Standard launch
 *   launchKernel(kernel, gridDim, blockDim, args, stream);
 *
 *   // Clustered launch with custom size
 *   launchKernel(kernel, gridDim, blockDim, args, stream, dim3(4, 1, 1));
 */
inline cudaError_t launchKernel(
    void* kernelFunc,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    cudaStream_t stream,
    std::optional<dim3> clusterDim = std::nullopt) {
  if (!clusterDim) {
    return cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream);
  }

  cudaLaunchConfig_t launchConfig = {};
  launchConfig.gridDim = gridDim;
  launchConfig.blockDim = blockDim;
  launchConfig.dynamicSmemBytes = 0;
  launchConfig.stream = stream;

  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = clusterDim->x;
  attrs[0].val.clusterDim.y = clusterDim->y;
  attrs[0].val.clusterDim.z = clusterDim->z;
  // Spread clusters across GPCs for better load balancing
  attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[1].val.clusterSchedulingPolicyPreference =
      cudaClusterSchedulingPolicySpread;
  launchConfig.attrs = attrs;
  launchConfig.numAttrs = 2;

  return cudaLaunchKernelExC(&launchConfig, kernelFunc, args);
}

} // namespace comms::common
