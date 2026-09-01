// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <getopt.h>

#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "comms/uniflow/benchmarks/BenchmarkRunner.h"
#include "comms/uniflow/benchmarks/Bootstrap.h"
#include "comms/uniflow/benchmarks/Rendezvous.h"
#include "comms/uniflow/benchmarks/Reporter.h"
#include "comms/uniflow/benchmarks/bench/RdmaBandwidthBenchmark.h"
#include "comms/uniflow/benchmarks/bench/SendRecvBandwidthBenchmark.h"
#include "comms/uniflow/benchmarks/bench/TcpBandwidthBenchmark.h"
#include "comms/uniflow/logging/Logger.h"

// ConnectionSetup/NVLink/NcclSendRecv benchmarks are NVIDIA-only (they depend
// on the NVLink transport or NCCL) and are compiled out on AMD. The SendRecv
// bandwidth benchmark is hipified and built on both platforms.
#ifndef __HIP_PLATFORM_AMD__
#include "comms/uniflow/benchmarks/bench/ConnectionSetupBenchmark.h"
#include "comms/uniflow/benchmarks/bench/NVLinkBandwidthBenchmark.h"
#include "comms/uniflow/benchmarks/bench/NcclSendRecvBenchmark.h"
#else
// AMD intra-node tier (HIP IPC over XGMI). Its target is only in
// uniflow_bench's deps under ovr_config//gpu:amd, so the include must be
// guarded to match -- unguarded it breaks the NVIDIA build with a missing
// header.
#include "comms/uniflow/benchmarks/bench/XgmiBandwidthBenchmark.h"
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
  bool noVerify{false};
  bool tcpAsyncH2d{true};
  std::vector<int> numStreams{1, 2, 4, 8};
  std::string topology{"fanout"};
  int pipelineDepth{2};
  size_t slabSize{0};
  int slabNum{0};
  std::vector<std::string> rdmaDevices;
  // Selects which local IPv6 address the TCP transport binds and advertises,
  // and nothing else. Which NIC actually carries the bytes is a separate axis
  // -- see --tcp-bind-dev below.
  std::string tcpIface{"eth2"};
  // Parallel TCP data sockets per bound device. Tracks the TcpTransportConfig
  // default; the total is this times the device count, and 1 with no bound
  // devices keeps the pre-lane wire format.
  size_t tcpSocketsPerNic{4};
  // Pin TCP egress to --tcp-iface via SO_BINDTODEVICE. Opt-in: without it
  // --tcp-iface only selects a source address and routing chooses the NIC.
  bool tcpBindDev{false};
  // Devices to stripe TCP lanes across, e.g. "eth1,eth2". Overrides
  // --tcp-bind-dev. Names are per-host, so the two hosts may need different
  // lists for the same pair of physical ports.
  std::string tcpBindDevs;
  std::string barrierDir;
  int barrierRanks{0};
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

// --tcp-bind-devs wins over --tcp-bind-dev, which is just the single-device
// shorthand for whatever --tcp-iface names. Empty means no device binding.
//
// --tcp-iface keeps selecting the source address either way, so when striping
// it wants to name one of the bound devices: pointing it at an unbound NIC
// advertises an address on a device carrying no lanes.
std::vector<std::string> tcpBindDevList(const CliOptions& opts) {
  std::vector<std::string> devices;
  if (!opts.tcpBindDevs.empty()) {
    std::istringstream iss(opts.tcpBindDevs);
    std::string token;
    while (std::getline(iss, token, ',')) {
      if (!token.empty()) {
        devices.push_back(token);
      }
    }
    if (devices.empty()) {
      std::cerr << "Invalid --tcp-bind-devs: '" << opts.tcpBindDevs << "'\n";
      std::exit(1);
    }
  } else if (opts.tcpBindDev) {
    devices.push_back(opts.tcpIface);
  }
  return devices;
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
      << "  --transport <type>     Transport backend: nvlink|rdma|tcp (default: nvlink)\n"
      << "  --min-size <bytes>     Minimum message size (default: 1)\n"
      << "  --max-size <bytes>     Maximum message size (default: 1073741824)\n"
      << "  --iterations <n>       Iterations per size (default: 100)\n"
      << "  --warmup <n>           Warmup iterations (default: 10)\n"
      << "  --loop-count <n>       Transport calls per timed iteration (default: 1)\n"
      << "  --bidirectional        Both ranks transfer simultaneously (default: unidirectional)\n"
      << "  --no-verify            Skip the pre-timing correctness sweep (tcp_bandwidth)\n"
      << "  --direction <dir>      put|get|both (default: both)\n"
      << "  --num-streams <list>   Comma-separated stream counts (default: 1,2,4,8)\n"
      << "  --output <path>        CSV output file path\n"
      << "  --format <fmt>         table|csv|both (default: table)\n"
      << "  --rdma-devices <list>  Comma-separated RDMA device names (default: auto-discover)\n"
      << "  --tcp-iface <name>     Interface whose address the TCP transport binds and\n"
      << "                         advertises (default: eth2). Selects the source address\n"
      << "                         only -- see --tcp-bind-dev to pin the egress NIC\n"
      << "  --no-tcp-async-h2d    Disable asynchronous TCP get() H2D (default: enabled)\n"
      << "  --tcp-sockets-per-nic <n>  Parallel TCP data sockets per bound device\n"
      << "                         (default: 4). Total lanes is this times the device\n"
      << "                         count, so 4 with two --tcp-bind-devs is 8 lanes. Both\n"
      << "                         peers must agree; 1 with no bound devices keeps the\n"
      << "                         pre-lane wire format\n"
      << "  --tcp-bind-dev         Pin TCP egress to --tcp-iface via SO_BINDTODEVICE\n"
      << "                         (default: off, so routing picks the egress NIC and\n"
      << "                         --tcp-iface only selects the source address)\n"
      << "  --tcp-bind-devs <l>    Comma-separated devices to stripe lanes across, e.g.\n"
      << "                         \"eth1,eth2\". Lane i goes to device i%count, one\n"
      << "                         listener per device. Overrides --tcp-bind-dev; both\n"
      << "                         peers must name the same number of devices; lanes\n"
      << "                         are per device, so each gets --tcp-sockets-per-nic\n"
      << "  --batch-size <n>       Number of requests per transport call (default: 1)\n"
      << "  --tx-depth <n>         Outstanding transport calls before waiting (default: 1)\n"
      << "  --num-nics <n>         Cap number of NICs to use (default: 0 = all\n"
      << "                         for RDMA, the transport's own default for TCP,\n"
      << "                         which is 2). On TCP a request above the number\n"
      << "                         of usable ports the host has is clamped to it,\n"
      << "                         with a warning; there is no fixed upper limit.\n"
      << "  --chunk-size <bytes>   RDMA transfer chunk size in bytes (default: 524288)\n"
      << "  --cuda-device <id>     GPU device index for buffer allocation (default: CPU memory)\n"
      << "  --topology <type>      Send/recv pattern: fanout|fanin (default: fanout)\n"
      << "  --pipeline-depth <n>   Send/recv staging pipeline depth (default: 2)\n"
      << "  --slab-size <bytes>    Staging slab size in bytes (default: chunk-size)\n"
      << "  --slab-num <n>         Number of staging slabs (default: pipeline-depth)\n"
      << "  --cuda-devices <list>  Comma-separated GPU indices for single-process multi-GPU (overrides --cuda-device)\n"
      << "  --gpu-nics <groups>    Per-GPU NIC map for multi-GPU: ';'-separated groups of comma-separated NICs, one per --cuda-devices entry\n"
      << "  --data-direct          Register GPU memory over the mlx5 Data Direct path (default: off)\n"
      << "  --measurement-barrier-dir <path>  Shared dir used to line up the timed\n"
      << "                         loops of N independent aggregate instances. Without\n"
      << "                         it their windows stagger and summed bandwidth is\n"
      << "                         overstated (measured overlap 1.0-2.1 of 8)\n"
      << "  --measurement-barrier-ranks <n>   Number of instances to wait for\n"
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
      {"tcp-iface", required_argument, nullptr, 266},
      {"no-tcp-async-h2d", no_argument, nullptr, 269},
      {"tcp-sockets-per-nic", required_argument, nullptr, 270},
      {"tcp-bind-dev", no_argument, nullptr, 271},
      {"tcp-bind-devs", required_argument, nullptr, 272},
      {"measurement-barrier-dir", required_argument, nullptr, 280},
      {"measurement-barrier-ranks", required_argument, nullptr, 281},
      {"no-verify", no_argument, nullptr, 267},
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
      case 267:
        opts.noVerify = true;
        break;
      case 264:
        opts.cudaDevices = parseIntList(optarg);
        break;
      case 265:
        opts.gpuNicGroups = parseNicGroups(optarg);
        break;
      case 266:
        opts.tcpIface = optarg;
        break;
      case 269:
        opts.tcpAsyncH2d = false;
        break;
      case 270:
        try {
          const int parsed = std::stoi(optarg);
          if (parsed < 1) {
            std::cerr
                << "Invalid value for --tcp-sockets-per-nic: must be >= 1\n";
            std::exit(1);
          }
          opts.tcpSocketsPerNic = static_cast<size_t>(parsed);
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --tcp-sockets-per-nic: '" << optarg
                    << "'\n";
          std::exit(1);
        }
        break;
      case 271:
        opts.tcpBindDev = true;
        break;
      case 272:
        opts.tcpBindDevs = optarg;
        break;
      case 280:
        opts.barrierDir = optarg;
        break;
      case 281:
        try {
          int parsed = std::stoi(optarg);
          if (parsed < 1) {
            std::cerr
                << "Invalid value for --measurement-barrier-ranks: must be >= 1\n";
            std::exit(1);
          }
          opts.barrierRanks = parsed;
        } catch (const std::exception&) {
          std::cerr << "Invalid value for --measurement-barrier-ranks: '"
                    << optarg << "'\n";
          std::exit(1);
        }
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
  runner.registerBenchmark(
      std::make_unique<uniflow::benchmark::TcpBandwidthBenchmark>(
          opts.tcpIface,
          opts.tcpAsyncH2d,
          opts.tcpSocketsPerNic,
          tcpBindDevList(opts)));
#ifndef __HIP_PLATFORM_AMD__
  runner.registerBenchmark(
      std::make_unique<uniflow::benchmark::ConnectionSetupBenchmark>());
  runner.registerBenchmark(
      std::make_unique<uniflow::benchmark::NVLinkBandwidthBenchmark>());
  runner.registerBenchmark(
      std::make_unique<uniflow::benchmark::NcclSendRecvBenchmark>());
#else
  runner.registerBenchmark(
      std::make_unique<uniflow::benchmark::XgmiBandwidthBenchmark>());
#endif
  runner.registerBenchmark(
      std::make_unique<uniflow::benchmark::SendRecvBandwidthBenchmark>(
          opts.rdmaDevices));

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
  config.verify = !opts.noVerify;
  config.barrierDir = opts.barrierDir;
  config.barrierRanks = opts.barrierRanks;
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
