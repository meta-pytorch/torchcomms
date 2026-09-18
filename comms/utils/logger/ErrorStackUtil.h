// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <vector>

namespace meta::comms::logger {

// Capture the native symbolized error stack once, dropping the leading logging
// / Scuba plumbing frames. Returns an empty vector when the build has no
// symbolizer support and when no trace could be obtained, so callers must treat
// an empty result as "no stack available" rather than as an error. The result
// can be shared across all error reporters so the expensive capture runs only
// once per error.
std::vector<std::string> captureNativeErrorStack();

namespace detail {
// Turns a raw symbolizer trace into frames: one per line, demangled, with the
// leading logging / Scuba plumbing removed. An empty trace yields an empty
// vector. Exposed so that case can be tested without building against a folly
// that has no symbolizer.
std::vector<std::string> normalizeStackTrace(const std::string& trace);
} // namespace detail

} // namespace meta::comms::logger
