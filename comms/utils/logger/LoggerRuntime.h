// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace meta::comms::logger {

// Initializes process-global logging state once. Calls before shutdown keep the
// subsystem mask and structured-event configuration from the first call.
void initCommLoggerRuntime();
void shutdownCommLoggerRuntime();

} // namespace meta::comms::logger
