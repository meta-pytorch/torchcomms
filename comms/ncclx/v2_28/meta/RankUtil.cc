// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/RankUtil.h"

#include <cstdlib>

#include <folly/Conv.h>

/* static */
std::optional<int64_t> RankUtil::getInt64FromEnv(const char* envVar) {
  char* envVarValue = getenv(envVar);
  if (envVarValue && strlen(envVarValue)) {
    if (auto result = folly::tryTo<int64_t>(envVarValue); result.hasValue()) {
      return result.value();
    }
  }
  return std::nullopt;
}

/* static */
std::optional<int64_t> RankUtil::getWorldSize() {
  return getInt64FromEnv("WORLD_SIZE");
}

/* static */
std::optional<int64_t> RankUtil::getGlobalRank() {
  return getInt64FromEnv("RANK");
}
