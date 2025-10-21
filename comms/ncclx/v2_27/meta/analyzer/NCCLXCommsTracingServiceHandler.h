// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>

#include <folly/Synchronized.h>

#include "comms/analyzer/if/gen-cpp2/CommsTracingService.h"

namespace ncclx {

class NCCLXCommsTracingServiceHandler
    : public apache::thrift::ServiceHandler<comms::CommsTracingService> {
 public:
  NCCLXCommsTracingServiceHandler();

  folly::coro::Task<std::unique_ptr<comms::GetCommsResponse>> co_getComms(
      std::unique_ptr<comms::GetCommsRequest> request) override;

 private:
  std::chrono::nanoseconds jobStartTimeNs_;

  // Currently, trainer step info isn't exported to NCCLX.
  // As an approximation, do the following:
  // - if the current step number is different than the step number
  //   on the previous request, then assume the step started now
  // - if the current step number is the same as the step number on
  //   the previous request, do not update the timestamp.
  struct StepInfo {
    int64_t stepOnLastRequest{0};
    std::chrono::nanoseconds lastRequestTsNs;
  };
  folly::Synchronized<StepInfo> stepInfo_;
};

} // namespace ncclx
