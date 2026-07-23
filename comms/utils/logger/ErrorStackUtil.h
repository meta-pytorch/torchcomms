// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <vector>

namespace meta::comms::logger {

// Capture the native symbolized error stack once, dropping the leading logging
// / Scuba plumbing frames. Returns an empty vector when stack capture is
// disabled (NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED off) or unavailable (no
// dwarf.h). The result can be shared across all error reporters so the
// expensive capture runs only once per error.
std::vector<std::string> captureNativeErrorStack();

} // namespace meta::comms::logger
