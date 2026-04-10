// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/Bootstrap.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace uniflow::benchmark {

namespace {

std::string requireEnv(const char* name) {
  const char* val = std::getenv(name);
  if (!val) {
    throw std::runtime_error(
        std::string("Missing required env var: ") + name +
        ". Launch with torchrun or set MASTER_ADDR, MASTER_PORT, RANK, "
        "WORLD_SIZE, LOCAL_RANK manually.");
  }
  return std::string(val);
}

int32_t requireEnvInt(const char* name) {
  auto val = requireEnv(name);
  try {
    return static_cast<int32_t>(std::stoi(val));
  } catch (const std::exception&) {
    throw std::runtime_error(
        std::string("Invalid integer for ") + name + ": '" + val + "'");
  }
}

} // namespace

BootstrapConfig BootstrapConfig::fromEnv() {
  BootstrapConfig config;
  config.masterAddr = requireEnv("MASTER_ADDR");
  config.masterPort = requireEnvInt("MASTER_PORT");
  config.rank = requireEnvInt("RANK");
  config.worldSize = requireEnvInt("WORLD_SIZE");
  config.localRank = requireEnvInt("LOCAL_RANK");

  if (config.worldSize < 1) {
    throw std::runtime_error("WORLD_SIZE must be >= 1");
  }
  if (config.rank < 0 || config.rank >= config.worldSize) {
    throw std::runtime_error(
        "RANK must be in [0, WORLD_SIZE). Got RANK=" +
        std::to_string(config.rank) +
        ", WORLD_SIZE=" + std::to_string(config.worldSize));
  }
  if (config.localRank < 0) {
    throw std::runtime_error(
        "LOCAL_RANK must be >= 0. Got LOCAL_RANK=" +
        std::to_string(config.localRank));
  }

  return config;
}

} // namespace uniflow::benchmark
