// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/window/WinHintUtils.h"

#include "comms/ctran/hints/HintUtils.h"
#include "comms/ctran/utils/Checks.h"

namespace meta::comms::hints {

namespace {
const std::string kWindowBufferLocation = "window_buffer_location";
const std::string kWindowSignalBufSize = "window_signal_bufsize";
const std::string kWinRegisterIpcOnly = "win_register_ipc_only";
const std::string kWinRegisterEnableSignal = "win_register_enable_signal";
} // namespace

void WinHintUtils::init(kvType& kv) {
  kv[kWindowBufferLocation] = "gpu";
}

commResult_t
WinHintUtils::set(const std::string& key, const std::string& val, kvType& kv) {
  std::string b;

  if (key == kWindowBufferLocation) {
    FB_COMMCHECK(HintUtils::clean_location_string(val, b));
    kv[key] = b;
  } else if (key == kWindowSignalBufSize) {
    kv[key] = val;
  } else if (key == kWinRegisterIpcOnly) {
    if (val == "1" || val == "true") {
      kv[key] = "1";
    } else if (val == "0" || val == "false") {
      kv[key] = "0";
    } else {
      return commInvalidArgument;
    }
  } else if (key == kWinRegisterEnableSignal) {
    if (val == "1" || val == "true") {
      kv[key] = "1";
    } else if (val == "0" || val == "false") {
      kv[key] = "0";
    } else {
      return commInvalidArgument;
    }
  } else {
    return commInvalidArgument;
  }

  return commSuccess;
}

const std::vector<std::string>& WinHintUtils::keys() {
  static std::vector<std::string> kKeys = {
      kWindowBufferLocation,
  };
  return kKeys;
}

bool WinHintUtils::parseBool(const std::string& val) {
  return val == "1" || val == "true";
}

} // namespace meta::comms::hints
