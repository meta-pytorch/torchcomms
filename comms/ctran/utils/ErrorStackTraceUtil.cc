// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <fmt/printf.h>
#include <folly/ExceptionString.h>
#include <folly/debugging/exception_tracer/SmartExceptionTracer.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/utils/ErrorReporterGuard.h"
#include "comms/ctran/utils/ErrorStackTraceUtil.h"
#include "comms/ctran/utils/NcclxErrorReporter.h"
#include "comms/utils/Conversion.h"

#include "comms/utils/logger/EventsScubaUtil.h"
#include "comms/utils/logger/ProcessGlobalErrorsUtil.h"

/* static */
commResult_t ErrorStackTraceUtil::log(commResult_t result) {
  logErrorMessage(::meta::comms::commCodeToString(result));
  return result;
}

/* static */
void ErrorStackTraceUtil::logErrorMessage(std::string errorMessage) {
  ctran::ErrorReport report;
  report.kind = ctran::ErrorReportKind::GENERAL_ERROR;
  report.errorMessage = std::move(errorMessage);

  auto* reporter = ctran::getThreadLocalErrorReporter();
  if (reporter) {
    reporter->reportError(report);
  } else {
    // Fallback: use default NcclxErrorReporter when no thread-local is set
    ctran::NcclxErrorReporter defaultReporter;
    defaultReporter.reportError(report);
  }
}
