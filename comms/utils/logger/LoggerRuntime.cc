// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/LoggerRuntime.h"

#include <mutex>

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/DataTableWrapper.h"
#include "comms/utils/logger/LoggingFormat.h"

namespace meta::comms::logger {
namespace {

struct RuntimeState {
  std::mutex mutex;
  bool initialized{false};
};

RuntimeState& getRuntimeState() {
  static RuntimeState state;
  return state;
}

} // namespace

void initCommLoggerRuntime() {
  auto& state = getRuntimeState();
  std::lock_guard lock{state.mutex};
  if (state.initialized) {
    return;
  }

  setSubSystemMask(parseDebugSubsysMask(NCCL_DEBUG_SUBSYS.c_str()));
  DataTableWrapper::init();
  state.initialized = true;
}

void shutdownCommLoggerRuntime() {
  auto& state = getRuntimeState();
  std::lock_guard lock{state.mutex};
  if (!state.initialized) {
    return;
  }

  DataTableWrapper::shutdown();
  state.initialized = false;
}

} // namespace meta::comms::logger
