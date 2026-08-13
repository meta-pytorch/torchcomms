// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef COMMS_PRIMS_TESTS_P2P_IB_TRANSPORT_DEVICE_ABORT_TEST_CUH_
#define COMMS_PRIMS_TESTS_P2P_IB_TRANSPORT_DEVICE_ABORT_TEST_CUH_

#include <cstdint>

#include "comms/common/fault_tolerance/AbortDevice.cuh"

namespace comms::prims::test {

void launchIbWrapperWaitSignalAbortCompileCheck(
    uint64_t* signal,
    bool* success);

void launchIbrcWaitSignalWithDisabledAbort(uint64_t* signal, bool* success);

void launchIbWrapperWaitSignalWithPreAbortedSkip(
    uint64_t* signal,
    bool* success,
    comms::fault_tolerance::AbortDevice abort);

void launchIbrcWaitSignalWithPreAbortedSkip(
    uint64_t* signal,
    bool* success,
    comms::fault_tolerance::AbortDevice abort);

} // namespace comms::prims::test

#endif // COMMS_PRIMS_TESTS_P2P_IB_TRANSPORT_DEVICE_ABORT_TEST_CUH_
