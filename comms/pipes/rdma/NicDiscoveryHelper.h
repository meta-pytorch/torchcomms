// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>

namespace comms::pipes {

/**
 * Thin wrapper around GpuNicDiscovery to isolate rdma-core headers
 * (infiniband/mlx5dv.h) from compilation units that include nvidia-doca
 * headers. Both define the same structs (mlx5dv_ah, mlx5dv_pd, mlx5dv_obj),
 * causing redefinition errors when included together.
 */
struct NicDiscoveryHelper {
  /**
   * Get PCIe bus ID string from CUDA device.
   * Delegates to GpuNicDiscovery::getCudaPciBusId().
   */
  static std::string getCudaPciBusId(int cudaDevice);

  /**
   * Discover the best NIC for a given CUDA device.
   * Returns the device name (e.g., "mlx5_0") of the best candidate.
   *
   * @param cudaDevice CUDA device index
   * @param ibHca NCCL_IB_HCA-style filter string (empty = no filtering)
   */
  static std::string discoverBestNic(int cudaDevice, const std::string& ibHca);
};

} // namespace comms::pipes
