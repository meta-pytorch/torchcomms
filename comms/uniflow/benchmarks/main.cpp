// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <getopt.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "comms/uniflow/benchmarks/BenchmarkRunner.h"
#include "comms/uniflow/benchmarks/Bootstrap.h"
#include "comms/uniflow/benchmarks/Rendezvous.h"
#include "comms/uniflow/benchmarks/Reporter.h"
#include "comms/uniflow/benchmarks/bench/RdmaBandwidthBenchmark.h"
#include "comms/uniflow/logging/Logger.h"
// ConnectionSetup/NVLink/NcclSendRecv/SendRecv benchmarks are NVIDIA-only
// (NVLink/NCCL transports, or not yet wired for the AMD/hipify build) and are
// compiled out on AMD.
#ifndef __HIP_PLATFORM_AMD__
#include "comms/uniflow/benchmarks/bench/ConnectionSetupBenchmark.h"
#include "comms/uniflow/benchmarks/bench/NVLinkBandwidthBenchmark.h"
#include "comms/uniflow/benchmarks/bench/NcclSendRecvBenchmark.h"
#include "comms/uniflow/benchmarks/bench/SendRecvBandwidthBenchmark.h"
#endif

namespace {

struct CliOptions {
  std::string benchmark{"all"};
  std::string transport{"nvlink"};
  std::string direction{"both"};
  std::string format{"table"};
  std::string outputPath;
  size_t minSize{1};
  size_t maxSize{1UL << 30};
  int iterations{100};
  int warmup{10};
  int loopCount{1};
  int batchSize{1};
  int txDepth{1};
  int numNics{0};
  size_t chunkSize{512 * 1024};
  int cudaDevice{-1};
  std::vector<int> cudaDevices;
  std::vector<std::vector<std::string>> gpuNicGroups;
  bool bidirectional{false};
  bool dataDirect{false};
  std::vector<int> numStreams{1, 2, 4, 8};
  std::string topology{"fanout"};
  int pipelineDepth{2};
  size_t slabSize{0};
  int slabNum{0};
  std::vector<std::string> rdmaDevices;
};

std::vector<int> parseIntList(const std::string& s) {
  std::vector<int> result;
  std::istringstream iss(s);
  std::string token;
  while (std::getline(iss, token, ',')) {
    try {
      result.push_back(std::stoi(token));
    } catch (const std::exception&) {
      std::cerr << "Invalid integer in list: '" << token << "'\n";
      std::exit(1);
    }
  }
  return result;
}

std::vector<std::string> parseStringList(const std::string& s) {
  std::vector<std::string> result;
  std::istringstream iss(s);
  std::string token;
  while (std::getline(iss, token, ',')) {
    if (!token.empty()) {
      result.push_back(token);
    }
  }
  return result;
}

/*
 * Parse a per-GPU NIC map: groups separated by ';', NICs within a group by ','.
 * e.g. "mlx5_0,mlx5_1;mlx5_2,mlx5_3" -> [[mlx5_0, mlx5_1], [mlx5_2, mlx5_3]].
 */
std::vector<std::vector<std::string>> parseNicGroups(const std::string& s) {
  std::vector<std::vector<std::string>> groups;
  std::istringstream iss(s);
  std::string group;
  while (std::getline(iss, group, ';')) {
    /*
     * Preserve empty groups (leading or adjacent ';'). Dropping them would
     * silently collapse the map and shift each GPU's NICs onto the wrong GPU;
     * keeping them lets the per-GPU count check and NIC selection report a
     * clear error for the empty slot instead.
     */
    groups.push_back(parseStringList(group));
  }
  /*
   * getline emits no token after a trailing ';', so a trailing empty group must
   * be appended explicitly to stay consistent with leading/adjacent empties
   * (e.g. "mlx5_0;" -> two groups, the second empty).
   */
  if (!s.empty() && s.back() == ';') {
    groups.emplace_back();
  }
  return groups;
}

void printUsage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [OPTIONS]\n"
      << "\n"
      << "Options:\n"
      << "  --benchmark <name>     Benchmark to run (default: all)\n"
      << "  --transport <type>     Transport backend: nvlink|rdma (default: nvlink)\n"
      << "  --min-size <bytes>     Minimum message size (default: 1)\n"
      << "  --max-size <bytes>     Maximum message size (default: 1073741824)\n"
      << "  --iterations <n>       Iterations per size (default: 100)\n"
      << "  --warmup <n>           Warmup iterations (default: 10)\n"
      << "  --loop-count <n>       Transport calls per timed iteration (default: 1)\n"
      << "  --bidirectional        Both ranks transfer simultaneously (default: unidirectional)\n"
      << "  --direction <dir>      put|get|both (default: both)\n"
      << "  --num-streams <list>   Comma-separated stream counts (default: 1,2,4,8)\n"
      << "  --output <path>        CSV output file path\n"
      << "  --format <fmt>         table|csv|both (default: table)\n"
      << "  --rdma-devices <list>  Comma-separated RDMA device names (default: auto-discover)\n"
      << "  --batch-size <n>       Number of requests per transport call (default: 1)\n"
      << "  --tx-depth <n>         Outstanding transport calls before waiting (default: 1)\n"
      << "  --num-nics <n>         Cap number of NICs to use (default: 0 = all)\n"
      << "  --chunk-size <bytes>   RDMA transfer chunk size in bytes (default: 524288)\n"
      << "  --cuda-device <id>     GPU device index for buffer allocation (default: CPU memory)\n"
      << "  --topology <type>      Send/recv pattern: fanout|fanin (default: fanout)\n"
      << "  --pipeline-depth <n>   Send/recv staging pipeline depth (default: 2)\n"
      << "  --slab-size <bytes>    Staging slab size in bytes (default: chunk-size)\n"
      << "  --slab-num <n>         Number of staging slabs (default: pipeline-depth)\n"
      << "  --cuda-devices <list>  Comma-separated GPU indices for single-process multi-GPU (overrides --cuda-device)\n"
      << "  --gpu-nics <groups>    Per-GPU NIC map for multi-GPU: ';'-separated groups of comma-separated NICs, one per --cuda-devices entry\n"
      << "  --data-direct          Register GPU memory over the mlx5 Data Direct path (default: off)\n"
      << "  --list                 List available benchmarks\n"
      << "  --help                 Show this help message\n"
      << "\n"
      << "Environment variables:\n"
      << "  MASTER_ADDR            Address of rank 0 (required for multi-rank)\n"
      << "  MASTER_PORT            Port of rank 0 (default: 29500)\n"
      << "  RANK                   This process's rank (default: 0)\n"
      << "  WORLD_SIZE             Total number of ranks (default: 1)\n"
      << "  LOCAL_RANK             GPU device index (default: 0)\n";
}

CliOptions parseArgs(int argc, char** argv) {
  CliOptions opts;
  bool listMode = false;

  static struct option longOpts[] = {
      {"benchmark", required_argument, nullptr, 'b'},
      {"transport", required_argument, nullptr, 't'},
      {"min-size", required_argument, nullptr, 'm'},
      {"max-size", required_argument, nullptr, 'M'},
      {"iterations", required_argument, nullptr, 'i'},
      {"warmup", required_argument, nullptr, 'w'},
      {"loop-count", required_argument, nullptr, 'L'},
      {"bidirectional", no_argument, nullptr, 'B'},
      {"direction", required_argument, nullptr, 'd'},
      {"num-streams", required_argument, nullptr, 's'},
      {"output", required_argument, nullptr, 'o'},
      {"format", required_argument, nullptr, 'f'},
      {"rdma-devices", required_argument, nullptr, 'r'},
      {"batch-size", required_argument, nullptr, 'T'},
      {"tx-depth", required_argument, nullptr, 257},
      {"num-nics", required_argument, nullptr, 258},
      {"chunk-size", required_argument, nullptr, 256},
      {"cuda-device", required_argument, nullptr, 'c'},
      {"topology", required_argument, nullptr, 259},
      {"pipeline-depth", required_argument, nullptr, 260},
      {"slab-size", required_argument, nullptr, 261},
      {"slab-num", required_argument, nullptr, 262},
      {"data-direct", no_argument, nullptr, 263},
      {"cuda-devices", required_argument, nullptr, 264},
      {"gpu-nics", required_argument, nullptr, 265},
      {"list", no_argument, nullptr, 'l'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0},
  };

  int opt;
  while (
      (opt = getopt_long(
           argc, argv, "b:t:m:M:i:w:L:Bd:s:o:f:r:T:c:lh", longOpts, nullptr)) !=
      -1) {
    switch (opt) {
      case 'b':
        opts.benchmark = optarg;
        break;
      case 't':
        opts.transport = optarg;
        break;
      case 'm':
        try {
          opts.minSize = std::stoull(optarg);
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --min-size: '" << optarg << "'\n";
          std::exit(1);
        }
        break;
      case 'M':
        try {
          opts.maxSize = std::stoull(optarg);
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --max-size: '" << optarg << "'\n";
          std::exit(1);
        }
        break;
      case 'i':
        try {
          opts.iterations = std::stoi(optarg);
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --iterations: '" << optarg << "'\n";
          std::exit(1);
        }
        break;
      case 'w':
        try {
          opts.warmup = std::stoi(optarg);
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --warmup: '" << optarg << "'\n";
          std::exit(1);
        }
        break;
      case 'L':
        try {
          opts.loopCount = std::stoi(optarg);
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --loop-count: '" << optarg << "'\n";
          std::exit(1);
        }
        break;
      case 'B':
        opts.bidirectional = true;
        break;
      case 263:
        opts.dataDirect = true;
        break;
      case 264:
        opts.cudaDevices = parseIntList(optarg);
        break;
      case 265:
        opts.gpuNicGroups = parseNicGroups(optarg);
        break;
      case 'd':
        opts.direction = optarg;
        break;
      case 's':
        opts.numStreams = parseIntList(optarg);
        break;
      case 'o':
        opts.outputPath = optarg;
        break;
      case 'f':
        opts.format = optarg;
        break;
      case 'r':
        opts.rdmaDevices = parseStringList(optarg);
        break;
      case 'T':
        try {
          opts.batchSize = std::stoi(optarg);
          if (opts.batchSize < 1) {
            std::cerr << "Invalid value for --batch-size: must be >= 1\n";
            std::exit(1);
          }
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --batch-size: '" << optarg << "'\n";
          std::exit(1);
        }
        break;
      case 257:
        try {
          opts.txDepth = std::stoi(optarg);
          if (opts.txDepth < 1) {
            std::cerr << "Invalid value for --tx-depth: must be >= 1\n";
            std::exit(1);
          }
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --tx-depth: '" << optarg << "'\n";
          std::exit(1);
        }
        break;
      case 258:
        try {
          opts.numNics = std::stoi(optarg);
          if (opts.numNics < 0) {
            std::cerr << "Invalid value for --num-nics: must be >= 0\n";
            std::exit(1);
          }
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --num-nics: '" << optarg << "'\n";
          std::exit(1);
        }
        break;
      case 256:
        try {
          opts.chunkSize = std::stoull(optarg);
          if (opts.chunkSize < 1) {
            std::cerr << "Invalid value for --chunk-size: must be >= 1\n";
            std::exit(1);
          }
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --chunk-size: '" << optarg << "'\n";
          std::exit(1);
        }
        break;
      case 'c':
        try {
          opts.cudaDevice = std::stoi(optarg);
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --cuda-device: '" << optarg << "'\n";
          std::exit(1);
        }
        break;
      case 259:
        opts.topology = optarg;
        if (opts.topology != "fanout" && opts.topology != "fanin") {
          std::cerr << "Invalid value for --topology: '" << optarg
                    << "' (expected fanout|fanin)\n";
          std::exit(1);
        }
        break;
      case 260:
        try {
          opts.pipelineDepth = std::stoi(optarg);
          if (opts.pipelineDepth < 1 || opts.pipelineDepth > 65535) {
            std::cerr
                << "Invalid value for --pipeline-depth: must be in [1, 65535]\n";
            std::exit(1);
          }
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --pipeline-depth: '" << optarg
                    << "'\n";
          std::exit(1);
        }
        break;
      case 261:
        try {
          opts.slabSize = std::stoull(optarg);
          if (opts.slabSize < 1) {
            std::cerr << "Invalid value for --slab-size: must be >= 1\n";
            std::exit(1);
          }
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --slab-size: '" << optarg << "'\n";
          std::exit(1);
        }
        break;
      case 262:
        try {
          opts.slabNum = std::stoi(optarg);
          if (opts.slabNum < 1) {
            std::cerr << "Invalid value for --slab-num: must be >= 1\n";
            std::exit(1);
          }
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --slab-num: '" << optarg << "'\n";
          std::exit(1);
        }
        break;
      case 'l':
        listMode = true;
        break;
      case 'h':
      default:
        printUsage(argv[0]);
        std::exit(opt == 'h' ? 0 : 1);
    }
  }

  if (listMode) {
    opts.benchmark = "__list__";
  }

  return opts;
}

} // namespace

int main(int argc, char** argv) {
  // Default to error-only unless SPDLOG_LEVEL env var is set.
  auto* logger = uniflow::logging::getLogger();
  if (std::getenv("SPDLOG_LEVEL") == nullptr) {
    logger->set_level(spdlog::level::err);
  }

  auto opts = parseArgs(argc, argv);

  uniflow::benchmark::BenchmarkRunner runner;
  runner.registerBenchmark(
      std::make_unique<uniflow::benchmark::RdmaBandwidthBenchmark>(
          opts.rdmaDevices));
#ifndef __HIP_PLATFORM_AMD__
  runner.registerBenchmark(
      std::make_unique<uniflow::benchmark::ConnectionSetupBenchmark>());
  runner.registerBenchmark(
      std::make_unique<uniflow::benchmark::NVLinkBandwidthBenchmark>());
  runner.registerBenchmark(
      std::make_unique<uniflow::benchmark::NcclSendRecvBenchmark>());
  runner.registerBenchmark(
      std::make_unique<uniflow::benchmark::SendRecvBandwidthBenchmark>(
          opts.rdmaDevices));
#endif

  if (opts.benchmark == "__list__") {
    std::cout << "Available benchmarks:\n";
    for (const auto& name : runner.listBenchmarks()) {
      std::cout << "  " << name << "\n";
    }
    return 0;
  }

  uniflow::benchmark::BootstrapConfig bootstrap;
  try {
    bootstrap = uniflow::benchmark::BootstrapConfig::fromEnv();
  } catch (const std::exception& e) {
    UNIFLOW_LOG_ERROR("Bootstrap failed: {}", e.what());
    return 1;
  }

  uniflow::benchmark::BenchmarkConfig config;
  config.minSize = opts.minSize;
  config.maxSize = opts.maxSize;
  config.iterations = opts.iterations;
  config.warmupIterations = opts.warmup;
  config.loopCount = opts.loopCount;
  config.bidirectional = opts.bidirectional;
  config.dataDirect = opts.dataDirect;
  config.direction = opts.direction;
  config.batchSize = opts.batchSize;
  config.txDepth = opts.txDepth;
  config.numNics = opts.numNics;
  config.chunkSize = opts.chunkSize;
  config.cudaDevice = opts.cudaDevice;
  config.cudaDevices = opts.cudaDevices;
  config.gpuNicGroups = opts.gpuNicGroups;
  config.numStreams = opts.numStreams;
  config.topology = opts.topology;
  config.pipelineDepth = opts.pipelineDepth;
  config.slabSize = opts.slabSize;
  config.slabNum = opts.slabNum;

  UNIFLOW_LOG_INFO(
      "Rank {}/{} starting benchmark (transport={})",
      bootstrap.rank,
      bootstrap.worldSize,
      opts.transport);

  auto peersResult = uniflow::benchmark::Rendezvous::establish(bootstrap);
  if (!peersResult) {
    UNIFLOW_LOG_ERROR("Rendezvous failed: {}", peersResult.error().toString());
    return 1;
  }
  auto peers = std::move(peersResult).value();

  UNIFLOW_LOG_INFO("Rendezvous complete: {} peer(s) connected", peers.size());

  std::vector<uniflow::benchmark::BenchmarkResult> results;
  if (opts.benchmark == "all") {
    results = runner.runAll(config, peers, bootstrap);
  } else {
    results = runner.runByName(opts.benchmark, config, peers, bootstrap);
  }

  if (bootstrap.isRank0()) {
    if (opts.format == "table" || opts.format == "both") {
      uniflow::benchmark::Reporter::printHeader(
          bootstrap, opts.transport, std::cout);
      uniflow::benchmark::Reporter::printTable(results, std::cout);
    }

    if (opts.format == "csv" || opts.format == "both") {
      if (!opts.outputPath.empty()) {
        std::ofstream ofs(opts.outputPath);
        if (ofs.is_open()) {
          uniflow::benchmark::Reporter::printCSV(results, ofs);
          UNIFLOW_LOG_INFO("CSV results written to {}", opts.outputPath);
        } else {
          UNIFLOW_LOG_ERROR("Failed to open output file: {}", opts.outputPath);
        }
      } else {
        uniflow::benchmark::Reporter::printCSV(results, std::cout);
      }
    }
  }

  if (results.empty() && bootstrap.isRank0()) {
    std::cout
        << "No benchmark results. Register transport benchmarks to run.\n";
  }

  return 0;
}
