// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/profiler/DefaultGpeProfilerReporter.h"

#include <memory>
#include <string>
#include <string_view>

#include "comms/utils/logger/EventMgr.h"
#include "comms/utils/logger/ScubaLogger.h"

namespace ctran {

void DefaultGpeProfilerReporter::report(const GpeProfilerReport& report) {
  // Copy message string_view into std::string here; the
  // CtranProfilerGpeEvent owns it for the async Scuba flush.
  // rank and commHash are added by the CommEvent base class via
  // logMetaData — do NOT pass them explicitly here (would duplicate
  // the Scuba columns).
  NcclScubaEvent scubaEvent(
      std::make_unique<CtranProfilerGpeEvent>(
          report.logMetaData,
          "ctranProfilerGpeV1",
          report.opCount,
          report.opType,
          std::string(tracePointName(report.tracePoint)),
          report.iterUs,
          report.durationUs,
          report.aborted,
          std::string(report.message)));
  scubaEvent.record();
}

} // namespace ctran
