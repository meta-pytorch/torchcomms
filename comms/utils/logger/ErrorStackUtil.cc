// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/ErrorStackUtil.h"

#include <array>
#include <string_view>

#include <folly/String.h>
#include <folly/debugging/symbolizer/Symbolizer.h>

namespace {
// Leading native-stack frames that belong to the logging / Scuba plumbing
// rather than the real error site. We drop frames from the top of the captured
// stack until the first frame that does NOT match any of these markers, so the
// recorded stack starts near where the error was actually reported. A
// name-based filter is used (instead of a fixed skip count) because setError()
// is reached through several different call chains (the CTRAN CERR / NCCL ERR
// logErrorToScuba path, the NcclLogFormatter CTRAN hook, and the NCCL debug
// path), each with a different amount of plumbing on top.
constexpr std::array<std::string_view, 11> kInternalFrameMarkers = {
    "folly::symbolizer",
    "getStackTraceStr",
    "NcclScubaSample",
    "EventsScubaUtil",
    "logErrorToScuba",
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

namespace detail {

std::vector<std::string> normalizeStackTrace(const std::string& trace) {
  // folly returns "" both from the no-symbolizer stub and, per Symbolizer.h,
  // from the real implementation when no trace is available. Splitting "" would
  // yield one empty element -- a bogus frame where there should be none.
  if (trace.empty()) {
    return {};
  }

  std::vector<std::string> frames;
  // @lint-ignore CLANGTIDY
  folly::split('\n', trace, frames, /* ignoreEmpty */ true);
  for (auto& line : frames) {
    auto demangledLine = folly::demangle(line.c_str()).toStdString();
    line.swap(demangledLine);
  }
  // Drop the leading logging / Scuba plumbing frames so the recorded stack
  // starts near the real error site.
  skipInternalFrames(frames);
  return frames;
}

} // namespace detail

std::vector<std::string> captureNativeErrorStack() {
  return detail::normalizeStackTrace(folly::symbolizer::getStackTraceStr());
}

} // namespace meta::comms::logger
