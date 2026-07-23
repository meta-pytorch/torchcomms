// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/ErrorStackUtil.h"

#include <array>
#include <sstream>
#include <string_view>

#if __has_include(<dwarf.h>)
#include <folly/debugging/symbolizer/Symbolizer.h>
#endif
#include <folly/String.h>

#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars

namespace {
// Leading native-stack frames that belong to the logging / Scuba plumbing
// rather than the real error site. We drop frames from the top of the captured
// stack until the first frame that does NOT match any of these markers, so the
// recorded stack starts near where the error was actually reported. A
// name-based filter is used (instead of a fixed skip count) because setError()
// is reached through several different call chains (direct CTRAN
// ErrorStackTraceUtil, the NcclLogFormatter CTRAN hook, and the NCCL debug
// path), each with a different amount of plumbing on top.
constexpr std::array<std::string_view, 12> kInternalFrameMarkers = {
    "folly::symbolizer",
    "getStackTraceStr",
    "NcclScubaSample",
    "EventsScubaUtil",
    "logErrorToScuba",
    "ErrorStackTraceUtil",
    "NcclLogFormatter",
    "folly::LogStreamProcessor",
    "folly::LogStreamVoidify",
    "folly::LogStream",
    "folly::XlogCategoryInfo",
    "folly::LogCategory",
};

bool isInternalFrame(const std::string& frame) {
  for (const auto& marker : kInternalFrameMarkers) {
    if (frame.find(marker) != std::string::npos) {
      return true;
    }
  }
  return false;
}

// Erase the leading plumbing frames (see kInternalFrameMarkers).
void skipInternalFrames(std::vector<std::string>& frames) {
  auto firstReal = frames.begin();
  while (firstReal != frames.end() && isInternalFrame(*firstReal)) {
    ++firstReal;
  }
  frames.erase(frames.begin(), firstReal);
}
} // namespace

namespace meta::comms::logger {

std::vector<std::string> captureNativeErrorStack() {
  std::vector<std::string> stackTraceMangled;

  // Get stack trace (requires elfutils/libdwarf for folly Symbolizer)
#if __has_include(<dwarf.h>)
  if (NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED) {
    std::stringstream ss;
    ss << folly::symbolizer::getStackTraceStr();
    // @lint-ignore CLANGTIDY
    folly::split('\n', ss.str(), stackTraceMangled);
    for (auto& line : stackTraceMangled) {
      auto demangledLine = folly::demangle(line.c_str()).toStdString();
      line.swap(demangledLine);
    }
    // Drop the leading logging / Scuba plumbing frames so the recorded stack
    // starts near the real error site.
    skipInternalFrames(stackTraceMangled);
  }
#endif

  return stackTraceMangled;
}

} // namespace meta::comms::logger
