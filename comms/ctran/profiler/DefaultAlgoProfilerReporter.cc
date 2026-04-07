// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/profiler/DefaultAlgoProfilerReporter.h"
#include "comms/utils/logger/EventMgr.h"
#include "comms/utils/logger/ScubaLogger.h"

namespace ctran {

void DefaultAlgoProfilerReporter::report(const AlgoProfilerReport& report) {
  if (!report.algoContext) {
    return;
  }
  NcclScubaEvent scubaEvent(
      std::make_unique<CtranProfilerAlgoEvent>(
          report.logMetaData,
          "algoProfilingV2",
          "",
          0,
          report.algoContext->peerRank,
          report.algoContext->deviceName,
          "",
          report.algoContext->algorithmName,
          report.algoContext->sendContext.messageSizes,
          report.algoContext->recvContext.messageSizes,
          "",
          report.algoContext->sendContext.totalBytes,
          report.algoContext->recvContext.totalBytes,
          report.bufferRegistrationTimeUs,
          report.controlSyncTimeUs,
          report.dataTransferTimeUs,
          report.opCount,
          report.readyTs,
          report.controlTs,
          report.timeFromDataToCollEndUs,
          report.collectiveDurationUs));
  scubaEvent.record();
}

} // namespace ctran
