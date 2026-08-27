// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace ctran::logging {

/**
 * Initialize logging for Ctran. By default it initializes once globally and
 * is a no-op on future calls in the process.
 *
 * `alwaysInit` reconfigures the CTRAN loggers for tests. Process-global shared
 * runtime state, including structured logging tables and the subsystem mask,
 * remains initialized once until explicitly shut down.
 */
void initCtranLogging(bool alwaysInit = false);

}; // namespace ctran::logging
