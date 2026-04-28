// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Optional.h>
#include <folly/String.h>
#include <chrono>
#include <string>
#include <unordered_map>

#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/logs/provider.h"

/**
 * Otel ScubaOtel for FB cloud infra, like fair-sc, coreweave, etc.
 */
class ScubaOtel {
 public:
  explicit ScubaOtel(folly::StringPiece dataset);

  size_t addRawData(
      const std::string& dataset,
      const std::string& message,
      folly::Optional<std::chrono::milliseconds> timeout);

  size_t addSample(const std::string& dataset,
      std::unordered_map<std::string, std::string> normalMap,
      std::unordered_map<std::string, int64_t> intMap,
      std::unordered_map<std::string, double> doubleMap);

 private:
  std::string tableName_;
  ::opentelemetry::nostd::shared_ptr<const opentelemetry::context::RuntimeContextStorage> storage_;
  ::opentelemetry::nostd::shared_ptr<::opentelemetry::logs::Logger> logger_;
};

void initLoggerProvider();
