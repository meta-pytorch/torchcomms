// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace ctran {

enum class ErrorReportKind {
  GENERAL_ERROR,
  NIC_EVENT,
};

// Data struct capturing error information, decoupled from both the error
// source and the reporter implementation. Passed to
// IErrorReporter::reportError().
struct ErrorReport {
  ErrorReportKind kind{ErrorReportKind::GENERAL_ERROR};

  // Common fields
  std::string errorMessage;
  std::vector<std::string> stackTrace;

  // NIC-specific fields (populated when kind == NIC_EVENT)
  std::string deviceName;
  int port{0};
  std::string nicStatus; // "UP" or "DOWN"
};

} // namespace ctran
