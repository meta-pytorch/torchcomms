// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/MultiPeerDeviceTransportTest.cuh"

#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes::test {

// =============================================================================
// MultiPeerDeviceTransport Construction Test
// =============================================================================

__global__ void multiPeerDeviceTransportConstructionKernel(
    int myRank,
    int nRanks,
    Transport* transports,
    int* results) {
  // Construct MultiPeerDeviceTransport
  DeviceSpan<Transport> transportsSpan(transports, nRanks);
  MultiPeerDeviceTransport transport(myRank, nRanks, transportsSpan);

  // Verify accessors return correct values
  results[0] = transport.rank();
  results[1] = transport.n_ranks();
}

void testMultiPeerDeviceTransportConstruction(
    int myRank,
    int nRanks,
    int* results) {
  // Allocate minimal buffers for construction test
  Transport* transports = nullptr;

  CUDACHECK_TEST(cudaMalloc(&transports, nRanks * sizeof(Transport)));

  CUDACHECK_TEST(cudaMemset(transports, 0, nRanks * sizeof(Transport)));

  multiPeerDeviceTransportConstructionKernel<<<1, 1>>>(
      myRank, nRanks, transports, results);
  CUDACHECK_TEST(cudaGetLastError());

  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaFree(transports));
}

// =============================================================================
// Get Transport Type Test
// =============================================================================

__global__ void getTransportTypeKernel(Transport* transport, int* results) {
  // Check if transport type is SELF
  results[0] = (transport->type == TransportType::SELF) ? 1 : 0;
}

void testGetTransportType(void* transport_d, int* results) {
  getTransportTypeKernel<<<1, 1>>>(
      static_cast<Transport*>(transport_d), results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// Peer Iteration Helpers Test
// =============================================================================

__global__ void peerIterationHelpersKernel(
    int myRank,
    int nRanks,
    Transport* transports,
    int* results) {
  // Construct MultiPeerDeviceTransport
  DeviceSpan<Transport> transportsSpan(transports, nRanks);
  MultiPeerDeviceTransport transport(myRank, nRanks, transportsSpan);

  // Test num_peers()
  results[0] = transport.num_peers();

  // Test peer_index_to_rank() for each peer index
  int numPeers = transport.num_peers();
  for (int i = 0; i < numPeers; ++i) {
    results[1 + i] = transport.peer_index_to_rank(i);
  }
}

void testPeerIterationHelpers(int myRank, int nRanks, int* results) {
  // Allocate minimal buffers for construction test
  Transport* transports = nullptr;

  CUDACHECK_TEST(cudaMalloc(&transports, nRanks * sizeof(Transport)));

  CUDACHECK_TEST(cudaMemset(transports, 0, nRanks * sizeof(Transport)));

  peerIterationHelpersKernel<<<1, 1>>>(myRank, nRanks, transports, results);
  CUDACHECK_TEST(cudaGetLastError());

  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaFree(transports));
}

} // namespace comms::pipes::test
