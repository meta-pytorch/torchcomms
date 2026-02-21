// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/ProcessGlobalErrorsUtil.h"

#include <folly/Singleton.h>
#include <folly/Synchronized.h>

#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/BackendTopologyUtil.h"
#include "comms/utils/logger/ScubaLogger.h"

namespace {
std::chrono::milliseconds nowTs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());
}

std::string readHostname() {
  auto topology = BackendTopologyUtil::getBackendTopology("/etc/fbwhoami");
  if (topology.has_value()) {
    return topology->host;
  }
  return "";
}

std::string readScaleupDomain() {
  auto topology = BackendTopologyUtil::getBackendTopology("/etc/fbwhoami");
  if (topology.has_value()) {
    return topology->scaleUp.domain;
  }
  return "";
}

struct AllStateTag {};
static folly::
    Singleton<folly::Synchronized<ProcessGlobalErrorsUtil::State>, AllStateTag>
        kAllState;
} // namespace

/* static */
void ProcessGlobalErrorsUtil::setNic(
    const std::string& devName,
    int port,
    std::optional<std::string> errorMessage) {
  auto statePtr = kAllState.try_get();
  if (!statePtr) {
    return;
  }
  statePtr->withWLock([&](auto& state) mutable {
    if (errorMessage.has_value()) {
      NicError nicError;
      nicError.timestampMs = nowTs();
      nicError.errorMessage = std::move(errorMessage.value());
      state.badNics[devName][port] = std::move(nicError);
    } else {
      state.badNics[devName].erase(port);
      if (state.badNics[devName].empty()) {
        state.badNics.erase(devName);
      }
    }
  });

  NcclScubaSample nicEvent("NIC_EVENT");
  nicEvent.addNormal("device", devName);
  nicEvent.addInt("port", port);
  nicEvent.addNormal("status", errorMessage.has_value() ? "DOWN" : "UP");
  SCUBA_nccl_structured_logging.addSample(std::move(nicEvent));
}

/* static */
void ProcessGlobalErrorsUtil::addErrorAndStackTrace(
    std::string errorMessage,
    std::vector<std::string> stackTrace) {
  if (NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES == 0) {
    return;
  }
  auto statePtr = kAllState.try_get();
  if (!statePtr) {
    return;
  }
  statePtr->withWLock([&](auto& state) {
    // If we are at capacity, remove the earliest one
    if (state.errorAndStackTraces.size() ==
        NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES) {
      state.errorAndStackTraces.pop_front();
    }
    ErrorAndStackTrace errorAndStackTrace;
    errorAndStackTrace.timestampMs = nowTs();
    errorAndStackTrace.errorMessage = std::move(errorMessage);
    errorAndStackTrace.stackTrace = std::move(stackTrace);
    state.errorAndStackTraces.push_back(std::move(errorAndStackTrace));
  });
}

/* static */
void ProcessGlobalErrorsUtil::addIbCompletionError(IbCompletionError error) {
  if (NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES == 0) {
    return;
  }
  auto statePtr = kAllState.try_get();
  if (!statePtr) {
    return;
  }
  statePtr->withWLock([&](auto& state) {
    // If we are at capacity, remove the earliest one
    if (state.ibCompletionErrors.size() ==
        NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES) {
      state.ibCompletionErrors.pop_front();
    }
    error.timestampMs = nowTs();
    state.ibCompletionErrors.push_back(std::move(error));
  });
}

/* static */
void ProcessGlobalErrorsUtil::clearIbCompletionErrors() {
  auto statePtr = kAllState.try_get();
  if (!statePtr) {
    return;
  }
  statePtr->withWLock([](auto& state) { state.ibCompletionErrors.clear(); });
}

/* static */
void ProcessGlobalErrorsUtil::addCudaError(CudaError error) {
  if (NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES == 0) {
    return;
  }
  auto statePtr = kAllState.try_get();
  if (!statePtr) {
    return;
  }
  statePtr->withWLock([&](auto& state) {
    if (state.cudaErrors.size() ==
        NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES) {
      state.cudaErrors.pop_front();
    }
    error.timestampMs = nowTs();
    state.cudaErrors.push_back(std::move(error));
  });
}

/* static */
void ProcessGlobalErrorsUtil::clearCudaErrors() {
  auto statePtr = kAllState.try_get();
  if (!statePtr) {
    return;
  }
  statePtr->withWLock([](auto& state) { state.cudaErrors.clear(); });
}

/* static */
ProcessGlobalErrorsUtil::State ProcessGlobalErrorsUtil::getAllState() {
  auto statePtr = kAllState.try_get();
  if (!statePtr) {
    return ProcessGlobalErrorsUtil::State{};
  }
  return statePtr->copy();
}

/* static */
std::string ProcessGlobalErrorsUtil::getHostname() {
  auto statePtr = kAllState.try_get();
  if (!statePtr) {
    return "";
  }
  return statePtr->withWLock([](auto& state) -> std::string {
    if (state.hostname.empty()) {
      state.hostname = readHostname();
    }
    return state.hostname;
  });
}

/* static */
std::string ProcessGlobalErrorsUtil::getScaleupDomain() {
  auto statePtr = kAllState.try_get();
  if (!statePtr) {
    return "";
  }
  return statePtr->withWLock([](auto& state) -> std::string {
    if (state.scaleupDomain.empty()) {
      state.scaleupDomain = readScaleupDomain();
    }
    return state.scaleupDomain;
  });
}
