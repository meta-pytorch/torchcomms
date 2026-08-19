// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace ctran::logging {

/**
 * Initialize logging for Ctran. By default it only initializes once globlally
 * and no-op for future calls on the process.
 *
 * @param alwaysInit If true, always initialize logging, for testing purpose.
 */
void initCtranLogging(bool alwaysInit = false);

}; // namespace ctran::logging
