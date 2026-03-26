// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/utils/IErrorReporter.h"

namespace ctran {

// Free functions for thread-local error reporter access.
// The thread-local reporter is used by ErrorStackTraceUtil and
// ProcessGlobalErrorsUtil to dispatch errors to the appropriate scuba table.
void setThreadLocalErrorReporter(IErrorReporter* reporter);
IErrorReporter* getThreadLocalErrorReporter();

// RAII guard that sets a thread-local IErrorReporter* for the current scope.
// On destruction, restores the previous reporter. This allows nested guards
// (e.g., if a collective calls into another collective).
class ErrorReporterGuard {
 public:
  explicit ErrorReporterGuard(IErrorReporter* reporter);
  ~ErrorReporterGuard();

  ErrorReporterGuard(const ErrorReporterGuard&) = delete;
  ErrorReporterGuard& operator=(const ErrorReporterGuard&) = delete;
  ErrorReporterGuard(ErrorReporterGuard&&) = delete;
  ErrorReporterGuard& operator=(ErrorReporterGuard&&) = delete;

 private:
  IErrorReporter* prev_;
};

} // namespace ctran
