// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef COMMS_PRIMS_TESTS_P2P_IB_TRANSPORT_DEVICE_ABORT_TEST_CUH_
#define COMMS_PRIMS_TESTS_P2P_IB_TRANSPORT_DEVICE_ABORT_TEST_CUH_

#include <cstdint>

namespace comms::prims::test {

void launchIbWrapperWaitSignalAbortCompileCheck();

void launchIbrcWaitSignalWithDisabledAbort(uint64_t* signal, bool* success);

} // namespace comms::prims::test

#endif // COMMS_PRIMS_TESTS_P2P_IB_TRANSPORT_DEVICE_ABORT_TEST_CUH_
