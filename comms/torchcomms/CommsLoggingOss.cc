// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/CommsLogging.h"

namespace meta::comms::logger {

// The core OSS library loads before any backend plugin. CERR already emits an
// error log, and the core wheel has no backend telemetry sink, so keep its
// secondary reporting hook local and dependency-free. Backend libraries retain
// the complete reporting implementation for their own error paths.
void logCommErrorToScuba(commResult_t, const std::string&) {}

} // namespace meta::comms::logger
