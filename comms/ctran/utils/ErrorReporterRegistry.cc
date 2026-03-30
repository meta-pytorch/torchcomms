// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/ErrorReporterRegistry.h"
#include "comms/ctran/utils/NcclxErrorReporter.h"

#include <unordered_map>

namespace ctran {

namespace {

std::unordered_map<ReporterType, ErrorReporterFactory>&
getFactoryRegistry() {
  static std::unordered_map<ReporterType, ErrorReporterFactory> registry;
  return registry;
}

} // namespace

void registerErrorReporterFactory(
    ReporterType type,
    ErrorReporterFactory factory) {
  getFactoryRegistry()[type] = std::move(factory);
}

std::unique_ptr<IErrorReporter> createErrorReporter(
    ReporterType type,
    CtranComm* comm) {
  auto& registry = getFactoryRegistry();
  auto it = registry.find(type);
  if (it != registry.end()) {
    return it->second(comm);
  }
  return std::make_unique<NcclxErrorReporter>();
}

} // namespace ctran
