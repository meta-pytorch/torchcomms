// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <folly/logging/xlog.h>

#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/testinfra/BenchmarkTestFixture.h"

namespace comms::pipes::benchmark {

/**
 * Format a byte count as a human-readable string (e.g., "128KB", "1.5GB").
 * Uses floating-point division to avoid integer overflow at large sizes.
 */
inline std::string format_bytes(std::size_t bytes) {
  constexpr double kKB = 1024.0;
  constexpr double kMB = 1024.0 * 1024.0;
  constexpr double kGB = 1024.0 * 1024.0 * 1024.0;

  double val = static_cast<double>(bytes);

  if (val < kKB) {
    return std::to_string(bytes) + "B";
  }
  if (val < kMB) {
    std::ostringstream oss;
    oss << static_cast<std::size_t>(val / kKB) << "KB";
    return oss.str();
  }
  if (val < kGB) {
    std::ostringstream oss;
    oss << static_cast<std::size_t>(val / kMB) << "MB";
    return oss.str();
  }
  std::ostringstream oss;
  oss << static_cast<std::size_t>(val / kGB) << "GB";
  return oss.str();
}

/**
 * Base fixture for collective benchmarks that use NCCL for comparison.
 *
 * Provides common SetUp/TearDown (CUDA device, NCCL communicator, stream)
 * and a bootstrap-based NCCL ID exchange helper.
 */
class NcclBenchmarkFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));

    NCCL_CHECK_VOID(
        ncclCommInitRank(&ncclComm_, worldSize, get_nccl_id(), globalRank));
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    NCCL_CHECK_VOID(ncclCommDestroy(ncclComm_));
    CUDA_CHECK_VOID(cudaStreamDestroy(stream_));
    BenchmarkTestFixture::TearDown();
  }

  /**
   * Exchange a NCCL unique ID across all ranks via the bootstrap allGather.
   * Rank 0 generates the ID; all others receive it.
   */
  ncclUniqueId get_nccl_id() {
    ncclUniqueId id{};
    if (globalRank == 0) {
      ncclResult_t res = ncclGetUniqueId(&id);
      if (res != ncclSuccess) {
        throw std::runtime_error(
            "Failed to get NCCL unique ID: " +
            std::string(ncclGetErrorString(res)));
      }
    }
    // Broadcast NCCL ID using bootstrap allGather
    std::vector<ncclUniqueId> allIds(worldSize);
    allIds[globalRank] = id;
    auto result =
        bootstrap
            ->allGather(
                allIds.data(), sizeof(ncclUniqueId), globalRank, worldSize)
            .get();
    if (result != 0) {
      throw std::runtime_error(
          "Failed to receive NCCL ID on rank " + std::to_string(globalRank));
    }
    id = allIds[0]; // Take rank 0's ID
    return id;
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

} // namespace comms::pipes::benchmark
