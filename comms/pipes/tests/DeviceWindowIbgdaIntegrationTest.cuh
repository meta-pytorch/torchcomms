// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/window/DeviceWindow.cuh"

namespace comms::pipes::test {

/**
 * Test kernel: DeviceWindow put with ringDb=false + signal_peer
 *
 * Sender posts dw.put(group, peer, ..., ringDb=false) which holds the WQE
 * back without ringing the doorbell, then dw.signal_peer() rings the
 * doorbell and submits both WQEs in a single batch — exercising the
 * DeviceWindow → IBGDA put_group_global ringDb forwarding.
 */
void testDeviceWindowPutNoDbAndSignal(
    DeviceWindow& dw,
    int targetRank,
    LocalBufferRegistration src,
    std::size_t nbytes,
    int signalId);

/**
 * Test kernel: Wait for a peer signal via DeviceWindow::wait_signal
 *
 * Receiver waits until the aggregate signal across all peers reaches
 * expectedValue.
 */
void testDeviceWindowWaitSignal(
    DeviceWindow& dw,
    int signalId,
    uint64_t expectedValue);

} // namespace comms::pipes::test
