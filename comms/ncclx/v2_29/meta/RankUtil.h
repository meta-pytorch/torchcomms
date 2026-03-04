// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <optional>

class RankUtil {
 public:
  static std::optional<int64_t> getInt64FromEnv(const char* envVar);
  static std::optional<int64_t> getWorldSize();
  static std::optional<int64_t> getGlobalRank();
};
