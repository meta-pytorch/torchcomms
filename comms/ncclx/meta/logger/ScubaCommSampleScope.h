// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

#include "comms/utils/commSpecs.h" // CommLogData
#include "comms/utils/logger/EventsScubaUtil.h"
#include "comms/utils/logger/NcclScubaSample.h"

namespace meta::comms::ncclx {

// RAII scope that opens an EventsScubaUtil "INIT"/"TERMINATE" sample and
// attaches the communicator's Scuba metadata (or none, when `logMetaData` is
// null). The sample is logged when the scope leaves the enclosing function --
// the same lifetime as the hand-rolled guard it replaces. Hoisted out of the
// forked upstream `init.cc` so that file stays close to pristine NCCL.
//
// `fileName` / `line` / `functionName` are threaded from the call site (via the
// NCCLX_SCUBA_COMM_SAMPLE macro below) so the logged sample keeps the forked
// call site's identity, not this seam's.
class ScubaCommSampleScope {
 public:
  ScubaCommSampleScope(
      const std::string& eventName,
      const char* fileName,
      int line,
      const char* functionName,
      const CommLogData* logMetaData)
      : guard_(eventName, fileName, line, functionName) {
    guard_.sample().setCommunicatorMetadata(logMetaData);
  }

  ScubaCommSampleScope(const ScubaCommSampleScope&) = delete;
  ScubaCommSampleScope& operator=(const ScubaCommSampleScope&) = delete;
  ScubaCommSampleScope(ScubaCommSampleScope&&) = delete;
  ScubaCommSampleScope& operator=(ScubaCommSampleScope&&) = delete;

 private:
  ::EventsScubaUtil::SampleGuard guard_;
};

} // namespace meta::comms::ncclx

// Opens a communicator-scoped Scuba sample for the enclosing function, matching
// the call site's __FILE__ / __LINE__ / __FUNCTION__. `comm` may be null.
// Requires ncclCommLogData() (meta/wrapper/NcclCommLogData.h) to be visible at
// the call site.
#define NCCLX_SCUBA_COMM_SAMPLE(eventName, comm)                     \
  ::meta::comms::ncclx::ScubaCommSampleScope ncclxScubaSampleScope_( \
      (eventName),                                                   \
      __FILE__,                                                      \
      __LINE__,                                                      \
      __FUNCTION__,                                                  \
      (comm) ? &ncclCommLogData(comm) : nullptr)
