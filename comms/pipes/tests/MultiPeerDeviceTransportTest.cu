// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/MultiPeerDeviceTransportTest.cuh"

#include <vector>

#include "comms/pipes/DeviceSignal.cuh"
#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes::test {

// =============================================================================
// DeviceSignal Construction Test
// =============================================================================

__global__ void deviceSignalConstructionKernel(
    int myRank,
    int nRanks,
    int signalCount,
    SignalState* localInbox,
    SignalState** peerInboxPtrs,
    int* results) {
  // Construct DeviceSignal with provided buffers
  // peerInboxPtrs has nPeers entries (not nRanks)
  int nPeers = nRanks - 1;
  DeviceSpan<SignalState> inboxSpan(localInbox, signalCount);
  DeviceSpan<SignalState*> peerPtrsSpan(peerInboxPtrs, nPeers);

  DeviceSignal signal(myRank, nRanks, signalCount, inboxSpan, peerPtrsSpan);

  // Verify accessors return correct values
  results[0] = signal.rank();
  results[1] = signal.n_ranks();
  results[2] = signal.signal_count();
}

void testDeviceSignalConstruction(
    int myRank,
    int nRanks,
    int signalCount,
    int* results) {
  // Allocate minimal buffers for construction test
  // peerInboxPtrs has nPeers entries (not nRanks)
  int nPeers = nRanks - 1;
  SignalState* localInbox = nullptr;
  SignalState** peerInboxPtrs = nullptr;

  cudaMalloc(&localInbox, signalCount * sizeof(SignalState));
  cudaMalloc(&peerInboxPtrs, nPeers * sizeof(SignalState*));
  cudaMemset(localInbox, 0, signalCount * sizeof(SignalState));
  cudaMemset(peerInboxPtrs, 0, nPeers * sizeof(SignalState*));

  deviceSignalConstructionKernel<<<1, 1>>>(
      myRank, nRanks, signalCount, localInbox, peerInboxPtrs, results);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  cudaFree(localInbox);
  cudaFree(peerInboxPtrs);
}

// =============================================================================
// =============================================================================
// MultiPeerDeviceTransport Construction Test
// =============================================================================

__global__ void multiPeerDeviceTransportConstructionKernel(
    int myRank,
    int nRanks,
    Transport* transports,
    SignalState* signalInbox,
    SignalState** signalPeerPtrs,
    int* results) {
  // Construct component objects
  // signalPeerPtrs has nPeers entries (not nRanks)
  int nPeers = nRanks - 1;
  DeviceSpan<SignalState> inboxSpan(signalInbox, 1); // signalCount=1
  DeviceSpan<SignalState*> signalPeerSpan(signalPeerPtrs, nPeers);
  DeviceSignal signal(myRank, nRanks, 1, inboxSpan, signalPeerSpan);

  // Construct MultiPeerDeviceTransport
  DeviceSpan<Transport> transportsSpan(transports, nRanks);
  MultiPeerDeviceTransport transport(myRank, nRanks, transportsSpan, signal);

  // Verify accessors return correct values
  results[0] = transport.rank();
  results[1] = transport.n_ranks();
}

void testMultiPeerDeviceTransportConstruction(
    int myRank,
    int nRanks,
    int* results) {
  // Allocate minimal buffers for construction test
  // signalPeerPtrs has nPeers entries (not nRanks)
  int nPeers = nRanks - 1;
  Transport* transports = nullptr;
  SignalState* signalInbox = nullptr;
  SignalState** signalPeerPtrs = nullptr;

  CUDACHECK_TEST(cudaMalloc(&transports, nRanks * sizeof(Transport)));
  cudaMalloc(&signalInbox, sizeof(SignalState)); // signalCount=1
  cudaMalloc(&signalPeerPtrs, nPeers * sizeof(SignalState*));

  CUDACHECK_TEST(cudaMemset(transports, 0, nRanks * sizeof(Transport)));
  cudaMemset(signalInbox, 0, sizeof(SignalState));
  cudaMemset(signalPeerPtrs, 0, nPeers * sizeof(SignalState*));

  multiPeerDeviceTransportConstructionKernel<<<1, 1>>>(
      myRank, nRanks, transports, signalInbox, signalPeerPtrs, results);
  CUDACHECK_TEST(cudaGetLastError());

  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaFree(transports));
  cudaFree(signalInbox);
  cudaFree(signalPeerPtrs);
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
    SignalState* signalInbox,
    SignalState** signalPeerPtrs,
    int* results) {
  // Construct component objects
  // signalPeerPtrs has nPeers entries (not nRanks)
  int nPeers = nRanks - 1;
  DeviceSpan<SignalState> inboxSpan(signalInbox, 1); // signalCount=1
  DeviceSpan<SignalState*> signalPeerSpan(signalPeerPtrs, nPeers);
  DeviceSignal signal(myRank, nRanks, 1, inboxSpan, signalPeerSpan);

  // Construct MultiPeerDeviceTransport
  DeviceSpan<Transport> transportsSpan(transports, nRanks);
  MultiPeerDeviceTransport transport(myRank, nRanks, transportsSpan, signal);

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
  // signalPeerPtrs has nPeers entries (not nRanks)
  int nPeers = nRanks - 1;
  Transport* transports = nullptr;
  SignalState* signalInbox = nullptr;
  SignalState** signalPeerPtrs = nullptr;

  CUDACHECK_TEST(cudaMalloc(&transports, nRanks * sizeof(Transport)));
  cudaMalloc(&signalInbox, sizeof(SignalState)); // signalCount=1
  cudaMalloc(&signalPeerPtrs, nPeers * sizeof(SignalState*));

  CUDACHECK_TEST(cudaMemset(transports, 0, nRanks * sizeof(Transport)));
  cudaMemset(signalInbox, 0, sizeof(SignalState));
  cudaMemset(signalPeerPtrs, 0, nPeers * sizeof(SignalState*));

  peerIterationHelpersKernel<<<1, 1>>>(
      myRank, nRanks, transports, signalInbox, signalPeerPtrs, results);
  CUDACHECK_TEST(cudaGetLastError());

  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaFree(transports));
  cudaFree(signalInbox);
  cudaFree(signalPeerPtrs);
}

} // namespace comms::pipes::test
