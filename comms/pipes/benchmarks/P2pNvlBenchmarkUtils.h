// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include "comms/pipes/P2pNvlTransportDevice.cuh"

namespace comms::pipes::benchmark {

// Test configuration struct
struct BenchmarkConfig {
  std::size_t nBytes = 0;
  std::size_t stagedBufferSize = 0;
  int numBlocks = 0;
  int numThreads = 0;
  std::size_t pipelineDepth = 4;
  std::size_t chunkSize = 512 * 1024; // 512KB default
  SyncScope groupScope = SyncScope::WARP; // Thread group scope for parallelism
  bool spreadClusterLaunch = false; // Use spread cluster kernel launch
  std::string name;
};

// Result struct for collecting benchmark data
struct BenchmarkResult {
  std::string testName;
  std::size_t messageSize{};
  std::size_t stagingBufferSize{};
  std::size_t pipelineDepth{};
  std::size_t chunkSize{};
  int numBlocks{};
  int numThreads{};
  float ncclBandwidth{};
  float p2pBandwidth{};
  float ncclTime{};
  float p2pTime{};
  float p2pSpeedup{}; // P2P vs NCCL
};

// Format bytes as human-readable size string (e.g., "8KB", "256MB", "1GB")
inline std::string formatSize(std::size_t bytes) {
  std::stringstream ss;
  if (bytes >= 1024 * 1024 * 1024) {
    ss << std::fixed << std::setprecision(0)
       << (bytes / (1024.0 * 1024.0 * 1024.0)) << "GB";
  } else if (bytes >= 1024 * 1024) {
    ss << std::fixed << std::setprecision(0) << (bytes / (1024.0 * 1024.0))
       << "MB";
  } else if (bytes >= 1024) {
    ss << std::fixed << std::setprecision(0) << (bytes / 1024.0) << "KB";
  } else {
    ss << bytes << "B";
  }
  return ss.str();
}

} // namespace comms::pipes::benchmark
