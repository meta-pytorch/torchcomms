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

inline const char* abortReasonName(comms::fault_tolerance::AbortReason reason) {
  using comms::fault_tolerance::AbortReason;

  switch (reason) {
    case AbortReason::NONE:
      return "none";
    case AbortReason::ABORTED:
      return "aborted";
    case AbortReason::TIMED_OUT:
      return "timed_out";
    case AbortReason::BOOTSTRAP_POLL:
      return "bootstrap_poll";
    case AbortReason::NETWORK_ERROR:
      return "network_error";
    case AbortReason::INTERNAL_ERROR:
      return "internal_error";
  }
  return "unknown";
}

} // namespace ctran::utils
