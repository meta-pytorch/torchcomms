// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/NcclxErrorReporter.h"

#include "comms/utils/logger/EventsScubaUtil.h"
#include "comms/utils/logger/ProcessGlobalErrorsUtil.h"
#include "comms/utils/logger/ScubaLogger.h"

namespace ctran {

void NcclxErrorReporter::reportError(const ErrorReport& report) {
  if (report.kind == ErrorReportKind::GENERAL_ERROR) {
    // Write error to nccl_structured_logging via EventsScubaUtil
    auto sampleGuard = EVENTS_SCUBA_UTIL_SAMPLE_GUARD("ERROR");
    sampleGuard.sample().setError(report.errorMessage);

    // Always update process-global error state
    const auto& sample = sampleGuard.sample();
    ProcessGlobalErrorsUtil::addErrorAndStackTrace(
        sample.exceptionMessage, sample.stackTrace);
  } else if (report.kind == ErrorReportKind::NIC_EVENT) {
    // Write NIC event to nccl_structured_logging
    NcclScubaSample nicEvent("NIC_EVENT");
    nicEvent.addNormal("device", report.deviceName);
    nicEvent.addInt("port", report.port);
    nicEvent.addNormal("status", report.nicStatus);
    SCUBA_nccl_structured_logging.addSample(std::move(nicEvent));
  }
}

} // namespace ctran
