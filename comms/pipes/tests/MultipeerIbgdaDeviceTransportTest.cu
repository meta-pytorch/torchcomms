// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultipeerIbgdaDeviceTransport.cuh"
#include "comms/pipes/tests/MultipeerIbgdaDeviceTransportTest.cuh"

namespace comms::pipes::tests {

// =============================================================================
// Device-side test kernel for rank mapping logic
// =============================================================================

__global__ void testRankMappingKernel(
    int myRank,
    int nRanks,
    int* results,
    int* expectedResults,
    int numTestCases,
    bool* success) {
  *success = true;

  // Create transport with empty peer transports (we only test rank mapping)
  MultipeerIbgdaDeviceTransport transport(
      myRank, nRanks, DeviceSpan<P2pIbgdaTransportDevice>());

  // Verify basic properties
  if (transport.myRank != myRank) {
    *success = false;
    return;
  }
  if (transport.nRanks != nRanks) {
    *success = false;
    return;
  }
  if (transport.numPeers() != nRanks - 1) {
    *success = false;
    return;
  }

  // Test indexToRank mapping for all peer indices
  for (int i = 0; i < numTestCases; ++i) {
    results[i] = transport.indexToRank(i);
    if (results[i] != expectedResults[i]) {
      *success = false;
    }
  }
}

// =============================================================================
// Wrapper function to launch the kernel (called from .cc test file)
// =============================================================================

void runTestRankMappingKernel(
    int myRank,
    int nRanks,
    int* d_results,
    int* d_expected,
    int numTestCases,
    bool* d_success) {
  testRankMappingKernel<<<1, 1>>>(
      myRank, nRanks, d_results, d_expected, numTestCases, d_success);
}

} // namespace comms::pipes::tests
