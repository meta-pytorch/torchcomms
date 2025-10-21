// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/RankUtils.h"

#include <cstdlib>

#include <folly/Conv.h>

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
    if (auto result = folly::tryTo<int64_t>(envVarValue); result.hasValue()) {
      return result.value();
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
