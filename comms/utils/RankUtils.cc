// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/RankUtils.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// Logic ported from scheduler
bool getIsTorchRun() {
  const char* localRank = std::getenv("LOCAL_RANK");
  return localRank != nullptr;
}

// Logic ported from scheduler
bool getIsSlurmJob() {
  const char* slurmJobId = std::getenv("SLURM_JOB_ID");
  return (slurmJobId != nullptr) && !getIsTorchRun();
}

enum RunType {
  TorchRun,
  SlurmJob,
  UnknownRun,
};

RunType getRunType() {
  if (getIsTorchRun()) {
    return RunType::TorchRun;
  } else if (getIsSlurmJob()) {
    return RunType::SlurmJob;
  }
  return RunType::UnknownRun;
}

} // namespace

/* static */
std::optional<int64_t> RankUtils::getInt64FromEnv(const char* envVar) {
  char* envVarValue = getenv(envVar);
  if (envVarValue && strlen(envVarValue)) {
    // Parse with std::stoll rather than folly::tryTo so this shared utility
    // (and the backends that link it, e.g. XCCL) does not pull in a folly
    // build dependency just for an env-var-to-int conversion.
    try {
      size_t pos = 0;
      std::string value(envVarValue);
      int64_t parsed = std::stoll(value, &pos);
      // std::stoll already skips leading whitespace, so trim trailing
      // whitespace too before the comparison so that values like "  42  "
      // are accepted. This matches folly::tryTo's behavior: it tolerates
      // surrounding whitespace but rejects other garbage.
      while (!value.empty() &&
             std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
      }
      if (pos == value.size()) {
        return parsed;
      }
    } catch (const std::exception&) {
      // Not a valid integer; fall through to nullopt.
    }
  }
  return std::nullopt;
}

/* static */
std::optional<int64_t> RankUtils::getWorldSize() {
  return getInt64FromEnv("WORLD_SIZE");
}

/* static */
std::optional<int64_t> RankUtils::getGlobalRank() {
  return getInt64FromEnv("RANK");
}

// Logic ported from scheduler
/* static */ std::optional<int64_t> RankUtils::getLocalRank() {
  switch (getRunType()) {
    case RunType::TorchRun:
      return getInt64FromEnv("LOCAL_RANK");
    case RunType::SlurmJob:
      return getInt64FromEnv("SLURM_LOCALID");
    default:
      return std::nullopt;
  }
}
