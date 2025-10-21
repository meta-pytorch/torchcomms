// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/window/WinHintUtils.h"

#include "comms/ctran/hints/HintUtils.h"
#include "comms/ctran/utils/Checks.h"

namespace meta::comms::hints {

namespace {
const std::string kWindowBufferLocation = "window_buffer_location";
const std::string kWindowSignalBufSize = "window_signal_bufsize";
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

} // namespace meta::comms::hints
