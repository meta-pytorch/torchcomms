#include "RcclxScubaEvent.h"
#include "comms/utils/logger/NcclScubaSample.h"

RcclxScubaEvent::RcclxScubaEvent(
    const int rank,
    const uint64_t commHash,
    const meta::colltrace::CollTraceInfo* collTraceData) {
  if (collTraceData == nullptr) {
    return;
  }

  ncclEvent_.sample_.addInt("rank", rank);
  ncclEvent_.sample_.addInt("commHash", commHash);

  ncclEvent_.sample_.addInt("collId", collTraceData->collId);
  ncclEvent_.sample_.addNormal("opName", collTraceData->opName);
  ncclEvent_.sample_.addInt("opCount", collTraceData->opCount);
  ncclEvent_.sample_.addNormal("dataType", collTraceData->dataType);
  ncclEvent_.sample_.addInt("count", collTraceData->count);
  ncclEvent_.sample_.addNormal("algoName", collTraceData->algoName);
  ncclEvent_.sample_.addDouble("latencyMs", collTraceData->latencyMs);

  ncclEvent_.sample_.addInt(
      "startTimestampMs",
      std::chrono::duration_cast<std::chrono::microseconds>(
          collTraceData->startTs.time_since_epoch())
          .count());
  ncclEvent_.sample_.addInt(
      "enqueueTimestampMs",
      std::chrono::duration_cast<std::chrono::microseconds>(
          collTraceData->enqueueTs.time_since_epoch())
          .count());
  ncclEvent_.sample_.addInt(
      "interCollTimeUs", collTraceData->interCollTime.count());
}

void RcclxScubaEvent::record(const std::string& stage) {
  if (!stage.empty()) {
    ncclEvent_.sample_.addNormal("stage", stage);
  }
  ncclEvent_.record();
}
