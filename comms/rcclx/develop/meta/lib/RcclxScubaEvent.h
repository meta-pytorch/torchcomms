// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/rcclx/develop/meta/lib/CollTraceUtils.h"
#include "comms/utils/logger/ScubaLogger.h"

class RcclxScubaEvent {
 public:
  explicit RcclxScubaEvent(
      const int rank,
      const uint64_t commHash,
      const meta::colltrace::CollTraceInfo* collTraceData);
  void record(const std::string& stage);

 private:
  NcclScubaEvent ncclEvent_{""};
};
