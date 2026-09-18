// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <utility>

#include "comms/common/fault_tolerance/AbortTypes.h"
#include "comms/ctran/utils/Exception.h"

namespace ctran::utils {

inline comms::fault_tolerance::AbortReason abortReason(commResult_t result) {
  using comms::fault_tolerance::AbortReason;

  switch (result) {
    case commRemoteError:
      // commRemoteError is CTRAN's legacy bucket for remote/transport
      // failures. Callers with a more specific local cause should construct
      // AbortInfo directly instead of using this coarse mapping.
      return AbortReason::NETWORK_ERROR;
    case commTimeout:
      return AbortReason::TIMED_OUT;
    case commUserAbort:
      return AbortReason::ABORTED;
    default:
      return AbortReason::INTERNAL_ERROR;
  }
}

inline comms::fault_tolerance::AbortInfo abortInfo(
    const Exception& exception,
    std::string context) {
  return {
      .reason = abortReason(exception.result()),
      .context = std::move(context),
  };
}

} // namespace ctran::utils
