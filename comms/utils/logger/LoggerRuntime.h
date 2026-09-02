// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace meta::comms::logger {

// Initializes logging state once per linkage image. Calls before shutdown keep
// the subsystem mask and structured-event configuration from the first call.
void initCommLoggerRuntime();
void shutdownCommLoggerRuntime();

} // namespace meta::comms::logger
