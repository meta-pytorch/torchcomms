// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <fmt/printf.h>
#include <folly/ExceptionString.h>
#include <folly/debugging/exception_tracer/SmartExceptionTracer.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/utils/ErrorStackTraceUtil.h"
#include "comms/utils/Conversion.h"

#include "comms/utils/logger/EventsScubaUtil.h"
#include "comms/utils/logger/ProcessGlobalErrorsUtil.h"

namespace {
void addToProcessGlobalErrors(EventsScubaUtil::SampleGuard& sampleGuard) {
  const auto& sample = sampleGuard.sample();
  ProcessGlobalErrorsUtil::addErrorAndStackTrace(
      sample.exceptionMessage, sample.stackTrace);
}
} // namespace

/* static */
commResult_t ErrorStackTraceUtil::log(commResult_t result) {
  logErrorMessage(::meta::comms::commCodeToString(result));
  return result;
}

/* static */
void ErrorStackTraceUtil::logErrorMessage(std::string errorMessage) {
  auto sampleGuard = EVENTS_SCUBA_UTIL_SAMPLE_GUARD("ERROR");
  sampleGuard.sample().setError(errorMessage);
  addToProcessGlobalErrors(sampleGuard);
}
