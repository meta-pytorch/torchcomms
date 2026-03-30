// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/utils/IErrorReporter.h"

namespace ctran {

// Default error reporter that logs to nccl_structured_logging scuba table.
// Used when no MCCL-specific reporter is registered (i.e., NCCLX path).
class NcclxErrorReporter : public IErrorReporter {
 public:
  ~NcclxErrorReporter() override = default;
  void reportError(const ErrorReport& report) override;
};

} // namespace ctran
