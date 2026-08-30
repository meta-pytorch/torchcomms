// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gtest/gtest.h>

#include "comms/ctran/utils/CtranLogger.h"

namespace ctran::bootstrap::testing {

class CtranLoggingEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    logging::configureStandaloneCtranLogging(spdlog::level::info);
  }
};

inline ::testing::Environment* registerCtranLoggingEnvironment() {
  return ::testing::AddGlobalTestEnvironment(new CtranLoggingEnvironment);
}

} // namespace ctran::bootstrap::testing
