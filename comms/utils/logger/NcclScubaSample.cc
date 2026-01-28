// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/NcclScubaSample.h"

#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars

NcclScubaSample::NcclScubaSample(std::string type, ScubaLogType logType)
    : CommsScubaSample(std::move(type)), logType_(logType) {}

NcclScubaSample::ScubaLogType NcclScubaSample::getLogType() {
  return logType_;
}

void NcclScubaSample::addTagSet(
    const std::string& key,
    const std::set<std::string>& value) {
  CommsScubaSample::addTagSet(key, value);
}

bool NcclScubaSample::shouldCaptureStackTrace() const {
  return NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED;
}

void NcclScubaSample::setExceptionInfo(const std::exception& ex) {
  setError(folly::exceptionStr(ex).toStdString());
}

void NcclScubaSample::setData(std::string data) {
  addNormal("event_data", std::move(data));
}

void NcclScubaSample::setExecResult(std::string result) {
  // TODO: We should change the field name to "exec_result" to be more generic
  // but this will break existing dashboards, so we might need more testing
  // before we can do that.
  addNormal("nccl_result", std::move(result));
}

void NcclScubaSample::setCommunicatorMetadata(const CommLogData* commMetadata) {
  if (commMetadata == nullptr) {
    return;
  }

  addInt("rank", commMetadata->rank);
  addInt("nRanks", commMetadata->nRanks);
  addInt("commId", commMetadata->commId);
  addInt("commHash", commMetadata->commHash);
  addNormal("commDesc", commMetadata->commDesc);
}
