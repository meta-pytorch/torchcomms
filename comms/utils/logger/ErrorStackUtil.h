// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <vector>

namespace meta::comms::logger {

// Capture the native symbolized error stack once, dropping the leading logging
// / Scuba plumbing frames. Returns the symbolized stack when symbolizer support
// is available, and an empty vector when it is unavailable (no dwarf.h). The
// result can be shared across all error reporters so the expensive capture runs
// only once per error.
std::vector<std::string> captureNativeErrorStack();

} // namespace meta::comms::logger
