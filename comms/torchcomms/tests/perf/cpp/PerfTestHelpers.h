// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/Device.h>
#include "comms/torchcomms/TorchComm.hpp"

// Timing utilities
struct PerfResult {
  int64_t message_size_bytes;
  int num_ranks;
  int iterations;
  double total_time_us;
  double avg_time_us;
  double min_time_us;
  double max_time_us;
  double bus_bw_gbps; // Bus bandwidth in GB/s
};

// Performance test parameters
struct PerfParams {
  bool async = false;
  int warmup_iterations = 5;
  int measure_iterations = 1000;
  // Number of iterations between stream synchronizations during measurement.
  // If 0, only synchronize after all iterations complete.
  int iteration_window = 0;
  // Message size range in bytes (powers of 2)
  int64_t min_size = 4; // 4 bytes
  int64_t max_size = 67108864; // 64 MB
  // Scaling factor for message sizes (default 2 = powers of 2)
  int size_scaling_factor = 2;
  // Data type
  at::ScalarType dtype = at::kFloat;
};

class PerfTimer {
 public:
  enum class State { kIdle, kRunning, kStopped };

  PerfTimer();
  void start();
  void stop();
  void reset();
  double elapsed_us();
  double elapsed_ms();

 private:
  at::cuda::CUDAEvent start_event_;
  at::cuda::CUDAEvent end_event_;
  State state_;
};

// Helper functions
void printPerfResult(const PerfResult& result, int rank);
void printPerfHeader(int rank);

// Create tensor helper
at::Tensor createTensor(
    int64_t num_elements,
    int rank,
    c10::DeviceType device_type = c10::DeviceType::CUDA,
    at::ScalarType dtype = at::kFloat);

// ============================================================================
// Collective Performance Helper Functions
// These can be called standalone or from tests
// Each function iterates over message sizes and prints results
// ============================================================================

// AllReduce performance helper
void runAllReducePerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params);

// AllGather performance helper (tensor list output)
void runAllGatherPerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params);

// AllGatherSingle performance helper (single tensor output)
void runAllGatherSinglePerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params);

// ReduceScatter performance helper (tensor list input)
void runReduceScatterPerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params);

// ReduceScatterSingle performance helper (single tensor input)
void runReduceScatterSinglePerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params);

// AllToAll performance helper (tensor list)
void runAllToAllPerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params);

// AllToAllSingle performance helper (single tensor)
void runAllToAllSinglePerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params);

// Broadcast performance helper
void runBroadcastPerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params,
    int root = 0);

// Reduce performance helper
void runReducePerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params,
    int root = 0);

// Scatter performance helper
void runScatterPerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params,
    int root = 0);

// Gather performance helper
void runGatherPerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params,
    int root = 0);

// SendRecv performance helper (ping-pong between rank 0 and 1)
void runSendRecvPerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params);

// Barrier performance helper
void runBarrierPerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params);

// Print usage information
void printUsage(const char* program_name);
