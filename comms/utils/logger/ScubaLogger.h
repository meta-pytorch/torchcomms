// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <string>

#include <folly/stop_watch.h>

#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/DataTableWrapper.h"
#include "comms/utils/logger/EventMgr.h"
#include "comms/utils/logger/NcclScubaSample.h"

struct NcclScubaEvent {
  void startAndRecord();
  void stopAndRecord();
  void lapAndRecord(const std::string& stage = "");
  void record();
  void record(const std::string& stage);
  void setLogMetatData(const CommLogData* logMetaData);

  explicit NcclScubaEvent(const std::string& stage);
  explicit NcclScubaEvent(const CommLogData* logMetaData);
  explicit NcclScubaEvent(const std::unique_ptr<LoggerEvent> loggerEvent);

  NcclScubaEvent(const std::string& stage, const CommLogData* logMetaData);

  NcclScubaSample sample_;
  folly::stop_watch<std::chrono::microseconds> timer_;
  std::string stage_{};
  LoggerEventType type_;
};

void ncclLogToScuba(LoggerEventType event, NcclScubaSample& sample);
DataTableWrapper* getTablePtrFromEvent(LoggerEventType event);
