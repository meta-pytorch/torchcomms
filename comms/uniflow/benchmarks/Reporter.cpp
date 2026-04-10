// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/Reporter.h"

#include <iomanip>
#include <map>
#include <sstream>

namespace uniflow::benchmark {

namespace {

std::string formatSize(size_t bytes) {
  std::ostringstream oss;
  oss << std::setw(12) << bytes;
  return oss.str();
}

std::string formatDouble(double val, int width = 12, int precision = 2) {
  std::ostringstream oss;
  oss << std::setw(width) << std::fixed << std::setprecision(precision) << val;
  return oss.str();
}

std::string formatInt(int val, int width = 8) {
  std::ostringstream oss;
  oss << std::setw(width) << val;
  return oss.str();
}

} // namespace

void Reporter::printHeader(
    const BootstrapConfig& config,
    const std::string& transport,
    std::ostream& os) {
  os << "======================================================================\n"
     << "                    Uniflow Transport Benchmark\n"
     << "  Transport: " << transport << "    Ranks: " << config.worldSize
     << "    LocalRank: " << config.localRank << "\n"
     << "======================================================================\n"
     << "\n";
}

void Reporter::printTable(
    const std::vector<BenchmarkResult>& results,
    std::ostream& os) {
  if (results.empty()) {
    os << "No benchmark results to report.\n";
    return;
  }

  std::map<std::string, std::vector<const BenchmarkResult*>> groups;
  for (const auto& r : results) {
    std::string key = r.benchmarkName;
    if (!r.direction.empty()) {
      key += " (" + r.direction + ")";
    }
    groups[key].push_back(&r);
  }

  for (const auto& [groupName, groupResults] : groups) {
    os << "-- " << groupName << " ";
    for (size_t i = groupName.size() + 4; i < 72; ++i) {
      os << '-';
    }
    os << "\n";

    bool hasBandwidth = false;
    bool hasMsgRate = false;
    bool hasStreams = false;
    for (const auto* r : groupResults) {
      if (r->bandwidthGBs > 0) {
        hasBandwidth = true;
      }
      if (r->messageRateMops > 0) {
        hasMsgRate = true;
      }
      if (r->numStreams > 0) {
        hasStreams = true;
      }
    }

    if (hasStreams) {
      os << " Streams";
    }
    os << "   Size (B)    Iters";
    if (hasBandwidth) {
      os << "    BW (GB/s)";
    }
    os << "   Lat avg(us)   Lat p50(us)   Lat p99(us)";
    if (hasMsgRate) {
      os << "     Mops/s";
    }
    os << "\n";

    for (const auto* r : groupResults) {
      if (hasStreams) {
        os << formatInt(r->numStreams, 8);
      }
      os << formatSize(r->messageSize) << formatInt(r->iterations, 8);
      if (hasBandwidth) {
        os << formatDouble(r->bandwidthGBs, 13);
      }
      os << formatDouble(r->latency.avg, 14) << formatDouble(r->latency.p50, 13)
         << formatDouble(r->latency.p99, 13);
      if (hasMsgRate) {
        os << formatDouble(r->messageRateMops, 10);
      }
      os << "\n";
    }
    os << "\n";
  }
}

void Reporter::printCSV(
    const std::vector<BenchmarkResult>& results,
    std::ostream& os) {
  os << "benchmark,transport,direction,size_bytes,iterations,"
        "bw_gbps,lat_avg_us,lat_p50_us,lat_p99_us,"
        "lat_min_us,lat_max_us,msg_rate_mops,num_streams\n";

  for (const auto& r : results) {
    os << r.benchmarkName << "," << r.transport << "," << r.direction << ","
       << r.messageSize << "," << r.iterations << "," << std::fixed
       << std::setprecision(4) << r.bandwidthGBs << "," << r.latency.avg << ","
       << r.latency.p50 << "," << r.latency.p99 << "," << r.latency.min << ","
       << r.latency.max << "," << r.messageRateMops << "," << r.numStreams
       << "\n";
  }
}

} // namespace uniflow::benchmark
