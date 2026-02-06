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
  DeviceSpan<SignalState> inboxSpan(localInbox, signalCount);
  DeviceSpan<SignalState*> peerPtrsSpan(peerInboxPtrs, nRanks);

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
  SignalState* localInbox = nullptr;
  SignalState** peerInboxPtrs = nullptr;

  cudaMalloc(&localInbox, signalCount * sizeof(SignalState));
  cudaMalloc(&peerInboxPtrs, nRanks * sizeof(SignalState*));
  cudaMemset(localInbox, 0, signalCount * sizeof(SignalState));
  cudaMemset(peerInboxPtrs, 0, nRanks * sizeof(SignalState*));

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
  DeviceSpan<SignalState> inboxSpan(signalInbox, 1); // signalCount=1
  DeviceSpan<SignalState*> signalPeerSpan(signalPeerPtrs, nRanks);
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
  Transport* transports = nullptr;
  SignalState* signalInbox = nullptr;
  SignalState** signalPeerPtrs = nullptr;

  CUDACHECK_TEST(cudaMalloc(&transports, nRanks * sizeof(Transport)));
  cudaMalloc(&signalInbox, sizeof(SignalState)); // signalCount=1
  cudaMalloc(&signalPeerPtrs, nRanks * sizeof(SignalState*));

  CUDACHECK_TEST(cudaMemset(transports, 0, nRanks * sizeof(Transport)));
  cudaMemset(signalInbox, 0, sizeof(SignalState));
  cudaMemset(signalPeerPtrs, 0, nRanks * sizeof(SignalState*));

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
  DeviceSpan<SignalState> inboxSpan(signalInbox, 1); // signalCount=1
  DeviceSpan<SignalState*> signalPeerSpan(signalPeerPtrs, nRanks);
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
  Transport* transports = nullptr;
  SignalState* signalInbox = nullptr;
  SignalState** signalPeerPtrs = nullptr;

  CUDACHECK_TEST(cudaMalloc(&transports, nRanks * sizeof(Transport)));
  cudaMalloc(&signalInbox, sizeof(SignalState)); // signalCount=1
  cudaMalloc(&signalPeerPtrs, nRanks * sizeof(SignalState*));

  CUDACHECK_TEST(cudaMemset(transports, 0, nRanks * sizeof(Transport)));
  cudaMemset(signalInbox, 0, sizeof(SignalState));
  cudaMemset(signalPeerPtrs, 0, nRanks * sizeof(SignalState*));

  peerIterationHelpersKernel<<<1, 1>>>(
      myRank, nRanks, transports, signalInbox, signalPeerPtrs, results);
  CUDACHECK_TEST(cudaGetLastError());

  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaFree(transports));
  cudaFree(signalInbox);
  cudaFree(signalPeerPtrs);
}

// =============================================================================
// Peer Index Conversion Roundtrip Test
// =============================================================================

__global__ void peerIndexConversionRoundtripKernel(
    int myRank,
    int nRanks,
    DeviceSpan<Transport> transports,
    DeviceSpan<SignalState> signalInbox,
    DeviceSpan<SignalState*> signalPeerPtrs,
    int* results) {
  // Construct component objects
  DeviceSignal signal(myRank, nRanks, 1, signalInbox, signalPeerPtrs);

  // Construct MultiPeerDeviceTransport
  MultiPeerDeviceTransport transport(myRank, nRanks, transports, signal);

  int numPeers = transport.num_peers();
  int idx = 0;

  // results layout:
  //   [0]                     = numPeers
  //   [1 .. numPeers]         = rank_to_peer_index for each non-self rank
  //   [numPeers+1 .. 2*numPeers] = roundtrip:
  //   peer_index_to_rank(rank_to_peer_index(rank)) [2*numPeers+1 .. 3*numPeers]
  //   = roundtrip: rank_to_peer_index(peer_index_to_rank(i)) [3*numPeers+1] =
  //   get_self_transport()->type [3*numPeers+2 .. 4*numPeers+1] =
  //   get_peer_transport(i)->type

  results[idx++] = numPeers;

  // Test rank_to_peer_index() for each non-self rank
  for (int rank = 0; rank < nRanks; ++rank) {
    if (rank == myRank) {
      continue;
    }
    results[idx++] = transport.rank_to_peer_index(rank);
  }

  // Roundtrip: rank -> peer_index -> rank (should be identity for non-self)
  for (int rank = 0; rank < nRanks; ++rank) {
    if (rank == myRank) {
      continue;
    }
    int peerIdx = transport.rank_to_peer_index(rank);
    results[idx++] = transport.peer_index_to_rank(peerIdx);
  }

  // Roundtrip: peer_index -> rank -> peer_index (should be identity)
  for (int i = 0; i < numPeers; ++i) {
    int rank = transport.peer_index_to_rank(i);
    results[idx++] = transport.rank_to_peer_index(rank);
  }

  // Test get_self_transport() type
  results[idx++] = static_cast<int>(transport.get_self_transport()->type);

  // Test get_peer_transport() types
  for (int i = 0; i < numPeers; ++i) {
    results[idx++] = static_cast<int>(transport.get_peer_transport(i)->type);
  }
}

void testPeerIndexConversionRoundtrip(int myRank, int nRanks, int* results) {
  // Allocate minimal buffers for construction test
  Transport* transportsRaw = nullptr;
  SignalState* signalInboxRaw = nullptr;
  SignalState** signalPeerPtrsRaw = nullptr;
  SignalState* countersRaw = nullptr;
  BarrierState* barriersRaw = nullptr;
  BarrierState** barrierPeerPtrsRaw = nullptr;

  CUDACHECK_TEST(cudaMalloc(&transportsRaw, nRanks * sizeof(Transport)));
  cudaMalloc(&signalInboxRaw, sizeof(SignalState));
  cudaMalloc(&signalPeerPtrsRaw, nRanks * sizeof(SignalState*));
  cudaMalloc(&countersRaw, sizeof(SignalState));
  cudaMalloc(&barriersRaw, sizeof(BarrierState));
  cudaMalloc(&barrierPeerPtrsRaw, nRanks * sizeof(BarrierState*));

  // Set up transports: self = SELF type, peers = P2P_NVL type.
  // We only need correct type tags for this test (no actual data transfer).
  // Zero-init the memory, then set the type field for each transport.
  CUDACHECK_TEST(cudaMemset(transportsRaw, 0, nRanks * sizeof(Transport)));
  for (int i = 0; i < nRanks; ++i) {
    TransportType type =
        (i == myRank) ? TransportType::SELF : TransportType::P2P_NVL;
    CUDACHECK_TEST(cudaMemcpy(
        &transportsRaw[i].type,
        &type,
        sizeof(TransportType),
        cudaMemcpyHostToDevice));
  }

  cudaMemset(signalInboxRaw, 0, sizeof(SignalState));
  cudaMemset(signalPeerPtrsRaw, 0, nRanks * sizeof(SignalState*));
  cudaMemset(countersRaw, 0, sizeof(SignalState));
  cudaMemset(barriersRaw, 0, sizeof(BarrierState));
  cudaMemset(barrierPeerPtrsRaw, 0, nRanks * sizeof(BarrierState*));

  // Construct DeviceSpans on host side
  DeviceSpan<Transport> transports(transportsRaw, nRanks);
  DeviceSpan<SignalState> signalInbox(signalInboxRaw, 1);
  DeviceSpan<SignalState*> signalPeerPtrs(signalPeerPtrsRaw, nRanks);
  DeviceSpan<SignalState> counters(countersRaw, 1);
  DeviceSpan<BarrierState> barriers(barriersRaw, 1);
  DeviceSpan<BarrierState*> barrierPeerPtrs(barrierPeerPtrsRaw, nRanks);

  peerIndexConversionRoundtripKernel<<<1, 1>>>(
      myRank, nRanks, transports, signalInbox, signalPeerPtrs, results);
  CUDACHECK_TEST(cudaGetLastError());

  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaFree(transportsRaw));
  cudaFree(signalInboxRaw);
  cudaFree(signalPeerPtrsRaw);
  cudaFree(countersRaw);
  cudaFree(barriersRaw);
  cudaFree(barrierPeerPtrsRaw);
}

} // namespace comms::pipes::test
