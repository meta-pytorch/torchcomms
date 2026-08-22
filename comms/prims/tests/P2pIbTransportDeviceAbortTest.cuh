// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef COMMS_PRIMS_TESTS_P2P_IB_TRANSPORT_DEVICE_ABORT_TEST_CUH_
#define COMMS_PRIMS_TESTS_P2P_IB_TRANSPORT_DEVICE_ABORT_TEST_CUH_

#include <cstdint>

#include "comms/common/fault_tolerance/AbortDevice.cuh"

namespace comms::prims::test {

/*
 * Runs `wait_signal(group, signal, expected, abort)` on a locally constructed
 * IBRC transport and stores the wait's return value in `waitResult`.
 *
 * Both entry points exercise the same wait through the two call paths that
 * production uses: the `P2pIbTransportDevice` dispatcher and the IBRC backend
 * directly. `expected == 0` is satisfied immediately by a zeroed signal;
 * anything higher never completes, so the wait can only return by observing
 * the abort handle.
 *
 * Asynchronous: the caller synchronizes and reads `waitResult` itself, which
 * is what lets a test abort the handle while the kernel is spinning.
 *
 * `enteredWait` must point at host-mapped memory. The kernel raises it, with a
 * system release fence, immediately before entering the wait. A test that
 * aborts mid-flight polls it first so it knows the kernel is resident and at
 * the wait, rather than sleeping and hoping. Without that, a slow launch turns
 * the abort-during-wait case into the pre-abort case with every assertion still
 * passing. May be null when the caller does not need the handshake.
 */
void launchIbWrapperWaitSignal(
    uint64_t* signal,
    bool* waitResult,
    uint64_t expected,
    comms::fault_tolerance::AbortDevice abort,
    uint32_t* enteredWait = nullptr);

void launchIbrcWaitSignal(
    uint64_t* signal,
    bool* waitResult,
    uint64_t expected,
    comms::fault_tolerance::AbortDevice abort,
    uint32_t* enteredWait = nullptr);

} // namespace comms::prims::test

#endif // COMMS_PRIMS_TESTS_P2P_IB_TRANSPORT_DEVICE_ABORT_TEST_CUH_
