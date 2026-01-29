// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <exception>
#include <set>
#include <string>
#include <vector>

#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/CommsScubaSample.h"

// See rfe/scubadata/ScubaDataSample.h
// We can't use it here to avoid all nccl headers including this type.
// Keys are scuba column names
// Each sample must explicitly define its own type so that different types of
// events within the system can be observed in different way.
class NcclScubaSample : public comms::CommsScubaSample {
 public:
  enum ScubaLogType {
    REGULAR,
    LITE // Only reports bare minimum common fields. Used for heavy duty tables
  };

  explicit NcclScubaSample(std::string type, ScubaLogType logType = REGULAR);

  // Only allow moves not copies
  NcclScubaSample(NcclScubaSample&& other) = default;
  NcclScubaSample& operator=(NcclScubaSample&& other) = default;
  ~NcclScubaSample() override = default;

  ScubaLogType getLogType();

  void addTagSet(const std::string& key, const std::set<std::string>& value);

  // Helper to set exception info and collect stack traces
  void setExceptionInfo(const std::exception& ex);

  // Set custom data attribute
  void setData(std::string data);

  // Add communicator metadata details to the sample
  void setCommunicatorMetadata(const CommLogData* commMetadata);
  void setExecResult(std::string result);

  // explicit copy function to avoid implicit copy constructor
  NcclScubaSample makeCopy() const {
    return NcclScubaSample(*this);
  }

 protected:
  bool shouldCaptureStackTrace() const override;

 private:
  NcclScubaSample(const NcclScubaSample& other) = default;
  NcclScubaSample& operator=(const NcclScubaSample& other) = default;

  ScubaLogType logType_;
};
