// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/utils/ErrorReport.h"

namespace ctran {

// Abstract interface for reporting errors to a scuba backend.
// Implementations can target different tables (e.g., nccl_structured_logging
// for NCCLX, mccl_operation_trace for MCCL).
class IErrorReporter {
 public:
  virtual ~IErrorReporter() = default;
  virtual void reportError(const ErrorReport& report) = 0;
};

} // namespace ctran
