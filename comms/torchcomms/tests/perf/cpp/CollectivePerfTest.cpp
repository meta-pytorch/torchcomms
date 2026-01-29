// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/tests/perf/cpp/PerfTestHelpers.h"

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

at::ScalarType parseDtype(const std::string& dtype_str) {
  if (dtype_str == "float" || dtype_str == "float32" || dtype_str == "fp32") {
    return at::kFloat;
  } else if (
      dtype_str == "half" || dtype_str == "float16" || dtype_str == "fp16") {
    return at::kHalf;
  } else if (dtype_str == "bfloat16" || dtype_str == "bf16") {
    return at::kBFloat16;
  } else if (
      dtype_str == "double" || dtype_str == "float64" || dtype_str == "fp64") {
    return at::kDouble;
  } else if (dtype_str == "int" || dtype_str == "int32") {
    return at::kInt;
  } else if (dtype_str == "long" || dtype_str == "int64") {
    return at::kLong;
  }
  throw std::runtime_error("Unknown dtype: " + dtype_str);
}

std::string dtypeToString(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return "float32";
    case at::kHalf:
      return "float16";
    case at::kBFloat16:
      return "bfloat16";
    case at::kDouble:
      return "float64";
    case at::kInt:
      return "int32";
    case at::kLong:
      return "int64";
    default:
      throw std::runtime_error("Unknown dtype");
  }
}

// Parse human-readable size strings like "4M", "4MB", "1G", "1GB", "512K", etc.
// Also accepts plain numbers (bytes).
int64_t parseSize(const std::string& size_str) {
  if (size_str.empty()) {
    throw std::runtime_error("Empty size string");
  }

  // Find where the numeric part ends
  size_t pos = 0;
  while (pos < size_str.size() &&
         (std::isdigit(size_str[pos]) || size_str[pos] == '.')) {
    pos++;
  }

  if (pos == 0) {
    throw std::runtime_error("Invalid size: " + size_str);
  }

  double value = std::stod(size_str.substr(0, pos));
  std::string suffix = size_str.substr(pos);

  // Convert suffix to uppercase for case-insensitive matching
  for (char& c : suffix) {
    c = std::toupper(c);
  }

  int64_t multiplier = 1;
  if (suffix.empty() || suffix == "B") {
    multiplier = 1;
  } else if (suffix == "K" || suffix == "KB") {
    multiplier = 1024;
  } else if (suffix == "M" || suffix == "MB") {
    multiplier = 1024 * 1024;
  } else if (suffix == "G" || suffix == "GB") {
    multiplier = 1024LL * 1024 * 1024;
  } else if (suffix == "T" || suffix == "TB") {
    multiplier = 1024LL * 1024 * 1024 * 1024;
  } else {
    throw std::runtime_error("Unknown size suffix: " + suffix);
  }

  return static_cast<int64_t>(value * multiplier);
}

std::string validateParams(
    const std::string& collective,
    const PerfParams& params) {
  // Validate collective name
  static const std::vector<std::string> valid_collectives = {
      "all_reduce",
      "allreduce",
      "all_gather",
      "allgather",
      "all_gather_single",
      "allgathersingle",
      "reduce_scatter",
      "reducescatter",
      "reduce_scatter_single",
      "reducescattersingle",
      "all_to_all",
      "alltoall",
      "all_to_all_single",
      "alltoallsingle",
      "broadcast",
      "reduce",
      "scatter",
      "gather",
      "send_recv",
      "sendrecv",
      "barrier",
      "all" // Run all collectives
  };

  bool valid_collective = false;
  for (const auto& valid : valid_collectives) {
    if (collective == valid) {
      valid_collective = true;
      break;
    }
  }
  if (!valid_collective) {
    return "Unknown collective '" + collective + "'";
  }

  // Validate size_scaling_factor
  if (params.size_scaling_factor < 2) {
    return "size_scaling_factor must be at least 2";
  }

  // Validate min_size and max_size
  if (params.min_size <= 0) {
    return "min_size must be positive";
  }
  if (params.max_size < params.min_size) {
    return "max_size must be >= min_size";
  }

  // Validate that max_size is reachable from min_size via scaling factor
  int64_t size = params.min_size;
  while (size < params.max_size) {
    size *= params.size_scaling_factor;
  }
  if (size != params.max_size && params.min_size != params.max_size) {
    return "max_size must be min_size * size_scaling_factor^n for some integer n";
  }

  // Validate dtype divides sizes evenly
  int64_t element_size = at::elementSize(params.dtype);
  if (params.min_size % element_size != 0) {
    return "min_size must be divisible by dtype element size (" +
        std::to_string(element_size) + " bytes)";
  }
  if (params.max_size % element_size != 0) {
    return "max_size must be divisible by dtype element size (" +
        std::to_string(element_size) + " bytes)";
  }

  // Validate iteration counts
  if (params.warmup_iterations < 0) {
    return "warmup_iterations must be non-negative";
  }
  if (params.measure_iterations <= 0) {
    return "measure_iterations must be positive";
  }
  if (params.iteration_window < 0) {
    return "iteration_window must be non-negative";
  }

  return ""; // Valid
}

} // namespace

int main(int argc, char** argv) {
  // Parse command-line arguments
  std::string collective = "all";
  PerfParams params;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "--async") {
      params.async = true;
    } else if (arg == "--warmup" && i + 1 < argc) {
      params.warmup_iterations = std::atoi(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      params.measure_iterations = std::atoi(argv[++i]);
    } else if (arg == "--window" && i + 1 < argc) {
      params.iteration_window = std::atoi(argv[++i]);
    } else if (arg == "--min-size" && i + 1 < argc) {
      params.min_size = parseSize(argv[++i]);
    } else if (arg == "--max-size" && i + 1 < argc) {
      params.max_size = parseSize(argv[++i]);
    } else if (arg == "--size-scaling-factor" && i + 1 < argc) {
      params.size_scaling_factor = std::atoi(argv[++i]);
    } else if (arg == "--dtype" && i + 1 < argc) {
      params.dtype = parseDtype(argv[++i]);
    } else if (arg[0] != '-') {
      // Positional argument - collective name
      collective = std::move(arg);
    }
  }

  // Validate all parameters
  std::string error = validateParams(collective, params);
  if (!error.empty()) {
    std::cerr << "Error: " << error << "\n\n";
    printUsage(argv[0]);
    return 1;
  }

  // Create communicator
  const char* test_device_env = std::getenv("TEST_DEVICE");
  std::string device_str = test_device_env ? test_device_env : "cuda";
  c10::Device device = c10::Device(device_str);

  const char* test_backend_env = std::getenv("TEST_BACKEND");
  if (!test_backend_env) {
    std::cerr << "Error: TEST_BACKEND environment variable is not set\n";
    return 1;
  }
  std::string backend = test_backend_env;

  torch::comms::CommOptions options;
  auto fast_init_mode_env = std::getenv("TEST_FAST_INIT_MODE");
  if (fast_init_mode_env) {
    options.hints["fastInitMode"] = std::string(fast_init_mode_env);
  }
  auto comm =
      torch::comms::new_comm(backend, device, "collective_perf_test", options);
  int rank = comm->getRank();
  int num_ranks = comm->getSize();

  if (rank == 0) {
    std::cout << "TorchComms Collective Performance Test\n"
              << "======================================\n"
              << "Backend: " << comm->getBackend() << "\n"
              << "Device: " << comm->getDevice() << "\n"
              << "Ranks: " << num_ranks << "\n"
              << "Collective: " << collective << "\n"
              << "Mode: " << (params.async ? "async" : "sync") << "\n"
              << "Dtype: " << dtypeToString(params.dtype) << "\n"
              << "Warmup: " << params.warmup_iterations << "\n"
              << "Iterations: " << params.measure_iterations << "\n"
              << "Window: " << params.iteration_window << "\n"
              << "Size range: " << params.min_size << " - " << params.max_size
              << " bytes (x" << params.size_scaling_factor << ")\n"
              << std::endl;
  }

  // Run performance tests
  if (collective == "all" || collective == "all_reduce") {
    runAllReducePerf(comm, params);
  }
  if (collective == "all" || collective == "all_gather") {
    runAllGatherPerf(comm, params);
  }
  if (collective == "all" || collective == "all_gather_single") {
    runAllGatherSinglePerf(comm, params);
  }
  if (collective == "all" || collective == "reduce_scatter") {
    runReduceScatterPerf(comm, params);
  }
  if (collective == "all" || collective == "reduce_scatter_single") {
    runReduceScatterSinglePerf(comm, params);
  }
  if (collective == "all" || collective == "all_to_all") {
    runAllToAllPerf(comm, params);
  }
  if (collective == "all" || collective == "all_to_all_single") {
    runAllToAllSinglePerf(comm, params);
  }
  if (collective == "all" || collective == "broadcast") {
    runBroadcastPerf(comm, params);
  }
  if (collective == "all" || collective == "reduce") {
    runReducePerf(comm, params);
  }
  if (collective == "all" || collective == "scatter") {
    runScatterPerf(comm, params);
  }
  if (collective == "all" || collective == "gather") {
    runGatherPerf(comm, params);
  }
  if (collective == "all" || collective == "send_recv") {
    runSendRecvPerf(comm, params);
  }
  if (collective == "all" || collective == "barrier") {
    runBarrierPerf(comm, params);
  }

  if (rank == 0) {
    std::cout << "\nPerformance test completed.\n";
  }

  comm->finalize();

  return 0;
}
