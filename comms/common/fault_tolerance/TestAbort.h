// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>

#include "comms/common/fault_tolerance/AbortDevice.cuh"

namespace comms::fault_tolerance::testing {

/**
 * Wall-clock budget applied to device waits in tests.
 *
 * Long enough that a healthy multi-rank test never trips it, short enough that
 * a wedged one fails while the test runner is still watching.
 */
inline constexpr std::chrono::milliseconds kTestAbortTimeout{60000};

/**
 * Returns a device handle backed by a process-wide `TRAP` mode `Abort`.
 *
 * Tests have no communicator, so a default-constructed `AbortDevice` is
 * disabled and every wait it guards becomes unbounded: a wedged kernel then
 * hangs until the test runner kills the job, with no indication of which wait
 * stalled. This handle gives those waits `kTestAbortTimeout` and traps on
 * expiry, which is the diagnosable failure the standalone Prims `Timeout`
 * used to provide.
 *
 * The handle is unstarted; kernels copy it per block and call `start()`.
 * Callers wanting a different budget copy the handle and call
 * `setOpTimeoutMs()` on the copy.
 *
 * Throws `std::runtime_error` if the shared state cannot be mapped for the
 * current CUDA device.
 */
AbortDevice testAbortDevice();

/** Returns a process-wide test handle using production `SKIP` behavior. */
AbortDevice testSkipAbortDevice();

} // namespace comms::fault_tolerance::testing
