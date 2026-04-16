// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/DeviceWindowIbgdaIntegrationTest.cuh"

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/window/DeviceWindow.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes::test {

// =============================================================================
// Kernel: DeviceWindow put with ringDb=false + signal_peer
//
// dw.put(group, peer, ..., ringDb=false) prepares and marks the WQE ready
// in the QP ring buffer but skips the submit step entirely (sq_wqe_pi not
// advanced, NIC not notified). The subsequent dw.signal_peer() submits its
// own WQE, which atomic_max's sq_wqe_pi past both WQEs and rings a single
// doorbell — covering the held-back put and the signal in one ring.
// =============================================================================

__global__ void deviceWindowPutNoDbAndSignalKernel(
    DeviceWindow dw,
    int targetRank,
    LocalBufferRegistration src,
    std::size_t nbytes,
    int signalId) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    dw.put(
        group,
        targetRank,
        /*dst_offset=*/0,
        src,
        /*src_offset=*/0,
        nbytes,
        /*ringDb=*/false);
    dw.signal_peer(targetRank, signalId);
  }
}

void testDeviceWindowPutNoDbAndSignal(
    DeviceWindow& dw,
    int targetRank,
    LocalBufferRegistration src,
    std::size_t nbytes,
    int signalId) {
  deviceWindowPutNoDbAndSignalKernel<<<1, 1>>>(
      dw, targetRank, src, nbytes, signalId);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Kernel: Wait for aggregate signal via DeviceWindow::wait_signal
// =============================================================================

__global__ void deviceWindowWaitSignalKernel(
    DeviceWindow dw,
    int signalId,
    uint64_t expectedValue) {
  auto group = make_block_group();
  dw.wait_signal(group, signalId, CmpOp::CMP_GE, expectedValue);
}

void testDeviceWindowWaitSignal(
    DeviceWindow& dw,
    int signalId,
    uint64_t expectedValue) {
  deviceWindowWaitSignalKernel<<<1, 32>>>(dw, signalId, expectedValue);
  CUDACHECK_TEST(cudaGetLastError());
}

} // namespace comms::pipes::test
