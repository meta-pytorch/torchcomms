// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/ErrorReporterGuard.h"

namespace ctran {

namespace {
thread_local IErrorReporter* kThreadLocalErrorReporter = nullptr;
} // namespace

void setThreadLocalErrorReporter(IErrorReporter* reporter) {
  kThreadLocalErrorReporter = reporter;
}

IErrorReporter* getThreadLocalErrorReporter() {
  return kThreadLocalErrorReporter;
}

ErrorReporterGuard::ErrorReporterGuard(IErrorReporter* reporter)
    : prev_(kThreadLocalErrorReporter) {
  kThreadLocalErrorReporter = reporter;
}

ErrorReporterGuard::~ErrorReporterGuard() {
  kThreadLocalErrorReporter = prev_;
}

} // namespace ctran
