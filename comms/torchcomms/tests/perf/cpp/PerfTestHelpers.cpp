// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/tests/perf/cpp/PerfTestHelpers.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace torch::comms;

// PerfTimer implementation using CUDA events
PerfTimer::PerfTimer() : state_(State::kIdle) {}

void PerfTimer::start() {
  if (state_ != State::kIdle) {
    throw std::runtime_error("start called when not idle");
  }
  start_event_.record();
  state_ = State::kRunning;
}

void PerfTimer::stop() {
  if (state_ != State::kRunning) {
    throw std::runtime_error("stop called when not running");
  }
  end_event_.record();
  state_ = State::kStopped;
}

void PerfTimer::reset() {
  if (state_ != State::kStopped) {
    throw std::runtime_error("reset called when not stopped");
  }
  state_ = State::kIdle;
}

double PerfTimer::elapsed_us() {
  if (state_ == State::kIdle) {
    throw std::runtime_error("elapsed_us called before start");
  }
  if (state_ == State::kRunning) {
    throw std::runtime_error("elapsed_us called before stop");
  }
  // Synchronize to ensure events are complete
  end_event_.synchronize();
  // elapsed_time_ms returns time in milliseconds
  float elapsed_ms = start_event_.elapsed_time(end_event_);
  return static_cast<double>(elapsed_ms) * 1000.0; // Convert to microseconds
}

double PerfTimer::elapsed_ms() {
  return elapsed_us() / 1000.0;
}

// Helper functions
void printPerfHeader(int rank) {
  if (rank != 0) {
    return;
  }
  std::cout << std::left << std::setw(15) << "Size(B)" << std::setw(10)
            << "Ranks" << std::setw(10) << "Iters" << std::setw(15) << "Avg(us)"
            << std::setw(15) << "Min(us)" << std::setw(15) << "Max(us)"
            << std::setw(15) << "BusBw(GB/s)" << std::endl;
  std::cout << std::string(95, '-') << std::endl;
}

void printPerfResult(const PerfResult& result, int rank) {
  if (rank != 0) {
    return;
  }
  std::cout << std::left << std::setw(15) << result.message_size_bytes
            << std::setw(10) << result.num_ranks << std::setw(10)
            << result.iterations << std::setw(15) << std::fixed
            << std::setprecision(2) << result.avg_time_us << std::setw(15)
            << result.min_time_us << std::setw(15) << result.max_time_us
            << std::setw(15) << result.bus_bw_gbps << std::endl;
}

at::Tensor createTensor(
    int64_t num_elements,
    int rank,
    c10::DeviceType device_type,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type);
  return at::ones({num_elements}, options) * static_cast<float>(rank + 1);
}

// ============================================================================
// Dispatch and Utility Functions
// ============================================================================

void printUsage(const char* program_name) {
  std::cout
      << "Usage: " << program_name << " <collective> [options]\n"
      << "\n"
      << "Collectives:\n"
      << "  all_reduce             - AllReduce collective\n"
      << "  all_gather             - AllGather collective (tensor list output)\n"
      << "  all_gather_single      - AllGather collective (single tensor output)\n"
      << "  reduce_scatter         - ReduceScatter collective (tensor list input)\n"
      << "  reduce_scatter_single  - ReduceScatter (single tensor input)\n"
      << "  all_to_all             - AllToAll collective (tensor list)\n"
      << "  all_to_all_single      - AllToAll collective (single tensor)\n"
      << "  broadcast              - Broadcast collective\n"
      << "  reduce                 - Reduce collective\n"
      << "  scatter                - Scatter collective\n"
      << "  gather                 - Gather collective\n"
      << "  send_recv              - Send/Recv point-to-point (ping-pong)\n"
      << "  barrier                - Barrier collective\n"
      << "  all                    - Run all collectives\n"
      << "\n"
      << "Options:\n"
      << "  --async                - Run async mode (default: sync)\n"
      << "  --warmup <n>           - Number of warmup iterations (default: 5)\n"
      << "  --iters <n>            - Number of measurement iterations (default: 1000)\n"
      << "  --window <n>           - Iterations between stream syncs (default: 0)\n"
      << "  --min-size <n>         - Min message size in bytes (default: 4)\n"
      << "  --max-size <n>         - Max message size in bytes (default: 67108864)\n"
      << "  --size-scaling-factor <n> - Size multiplier between tests (default: 2)\n"
      << "  --dtype <type>         - Data type: float32, float16, bfloat16, float64,\n"
      << "                           int32, int64 (default: float32)\n"
      << "  --help, -h             - Show this help message\n"
      << std::endl;
}
