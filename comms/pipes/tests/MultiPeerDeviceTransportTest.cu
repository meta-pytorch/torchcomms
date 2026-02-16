// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/MultiPeerDeviceTransportTest.cuh"

#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/window/DeviceWindowBarrier.cuh"
#include "comms/pipes/window/DeviceWindowMemory.cuh"
#include "comms/pipes/window/DeviceWindowSignal.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

namespace comms::pipes::test {

// Helper: allocate zeroed device memory and wrap in DeviceSpan
template <typename T>
DeviceSpan<T> makeZeroedSpan(
    meta::comms::DeviceBuffer& buf,
    std::size_t count) {
  CUDACHECK_TEST(cudaMemset(buf.get(), 0, count * sizeof(T)));
  return DeviceSpan<T>(static_cast<T*>(buf.get()), count);
}

// =============================================================================
// DeviceWindowSignal Construction Test
// =============================================================================

__global__ void deviceSignalConstructionKernel(
    int myRank,
    int nRanks,
    int signalCount,
    DeviceSpan<SignalState> localInbox,
    DeviceSpan<DeviceSpan<SignalState>> peerSignals,
    int* results) {
  DeviceWindowSignal signal(
      myRank, nRanks, signalCount, localInbox, peerSignals);

  // Verify accessors return correct values
  results[0] = signal.rank();
  results[1] = signal.n_ranks();
  results[2] = signal.signal_count();
}

void testDeviceWindowSignalConstruction(
    int myRank,
    int nRanks,
    int signalCount,
    int* results) {
  int nPeers = nRanks - 1;
  meta::comms::DeviceBuffer localInboxBuf(signalCount * sizeof(SignalState));
  meta::comms::DeviceBuffer peerSignalsBuf(
      nPeers * sizeof(DeviceSpan<SignalState>));

  auto localInbox = makeZeroedSpan<SignalState>(localInboxBuf, signalCount);
  auto peerSignals =
      makeZeroedSpan<DeviceSpan<SignalState>>(peerSignalsBuf, nPeers);

  deviceSignalConstructionKernel<<<1, 1>>>(
      myRank, nRanks, signalCount, localInbox, peerSignals, results);

  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindowBarrier Construction Test
// =============================================================================

__global__ void deviceBarrierConstructionKernel(
    int myRank,
    int nRanks,
    DeviceSpan<BarrierState> localBarriers,
    DeviceSpan<BarrierState*> peerBarrierPtrs,
    int* results) {
  DeviceWindowBarrier barrier(myRank, nRanks, localBarriers, peerBarrierPtrs);

  // Verify accessors return correct values
  results[0] = barrier.rank();
  results[1] = barrier.n_ranks();
}

void testDeviceWindowBarrierConstruction(int myRank, int nRanks, int* results) {
  int nPeers = nRanks - 1;
  meta::comms::DeviceBuffer localBarriersBuf(sizeof(BarrierState));
  meta::comms::DeviceBuffer peerBarrierPtrsBuf(nPeers * sizeof(BarrierState*));

  auto localBarriers = makeZeroedSpan<BarrierState>(localBarriersBuf, 1);
  auto peerBarrierPtrs =
      makeZeroedSpan<BarrierState*>(peerBarrierPtrsBuf, nPeers);

  deviceBarrierConstructionKernel<<<1, 1>>>(
      myRank, nRanks, localBarriers, peerBarrierPtrs, results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// MultiPeerDeviceTransport Construction Test
// =============================================================================

__global__ void multiPeerDeviceTransportConstructionKernel(
    int myRank,
    int nRanks,
    DeviceSpan<Transport> transports,
    DeviceSpan<SignalState> signalInbox,
    DeviceSpan<DeviceSpan<SignalState>> peerSignals,
    DeviceSpan<BarrierState> barriers,
    DeviceSpan<BarrierState*> barrierPeerPtrs,
    int* results) {
  // Construct component objects
  DeviceWindowSignal signal(myRank, nRanks, 1, signalInbox, peerSignals);
  DeviceWindowBarrier barrier(myRank, nRanks, barriers, barrierPeerPtrs);
  DeviceWindowMemory wm(signal, barrier);

  // Construct MultiPeerDeviceTransport
  MultiPeerDeviceTransport transport(myRank, nRanks, transports, wm);

  // Verify accessors return correct values
  results[0] = transport.rank();
  results[1] = transport.n_ranks();
}

void testMultiPeerDeviceTransportConstruction(
    int myRank,
    int nRanks,
    int* results) {
  int nPeers = nRanks - 1;
  meta::comms::DeviceBuffer transportsBuf(nRanks * sizeof(Transport));
  meta::comms::DeviceBuffer signalInboxBuf(sizeof(SignalState));
  meta::comms::DeviceBuffer peerSignalsBuf(
      nPeers * sizeof(DeviceSpan<SignalState>));
  meta::comms::DeviceBuffer barriersBuf(sizeof(BarrierState));
  meta::comms::DeviceBuffer barrierPeerPtrsBuf(nPeers * sizeof(BarrierState*));

  auto transports = makeZeroedSpan<Transport>(transportsBuf, nRanks);
  auto signalInbox = makeZeroedSpan<SignalState>(signalInboxBuf, 1);
  auto peerSignals =
      makeZeroedSpan<DeviceSpan<SignalState>>(peerSignalsBuf, nPeers);
  auto barriers = makeZeroedSpan<BarrierState>(barriersBuf, 1);
  auto barrierPeerPtrs =
      makeZeroedSpan<BarrierState*>(barrierPeerPtrsBuf, nPeers);

  multiPeerDeviceTransportConstructionKernel<<<1, 1>>>(
      myRank,
      nRanks,
      transports,
      signalInbox,
      peerSignals,
      barriers,
      barrierPeerPtrs,
      results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// Self-Transport Put Test
// =============================================================================

__global__ void selfTransportPutKernel(
    Transport* transport,
    char* dst_d,
    const char* src_d,
    std::size_t nbytes) {
  // Use self transport's put() method for local copy
  auto group = make_warp_group();
  transport->self.put(group, dst_d, src_d, nbytes);
}

void testSelfTransportPut(
    void* transport_d,
    char* dst_d,
    const char* src_d,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  selfTransportPutKernel<<<numBlocks, blockSize>>>(
      static_cast<Transport*>(transport_d), dst_d, src_d, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
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
    DeviceSpan<Transport> transports,
    DeviceSpan<SignalState> signalInbox,
    DeviceSpan<DeviceSpan<SignalState>> peerSignals,
    DeviceSpan<BarrierState> barriers,
    DeviceSpan<BarrierState*> barrierPeerPtrs,
    int* results) {
  // Construct component objects
  DeviceWindowSignal signal(myRank, nRanks, 1, signalInbox, peerSignals);
  DeviceWindowBarrier barrier(myRank, nRanks, barriers, barrierPeerPtrs);
  DeviceWindowMemory wm(signal, barrier);

  // Construct MultiPeerDeviceTransport
  MultiPeerDeviceTransport transport(myRank, nRanks, transports, wm);

  // Test num_peers()
  results[0] = transport.num_peers();

  // Test peer_index_to_rank() for each peer index
  int numPeers = transport.num_peers();
  for (int i = 0; i < numPeers; ++i) {
    results[1 + i] = transport.peer_index_to_rank(i);
  }
}

void testPeerIterationHelpers(int myRank, int nRanks, int* results) {
  int nPeers = nRanks - 1;
  meta::comms::DeviceBuffer transportsBuf(nRanks * sizeof(Transport));
  meta::comms::DeviceBuffer signalInboxBuf(sizeof(SignalState));
  meta::comms::DeviceBuffer peerSignalsBuf(
      nPeers * sizeof(DeviceSpan<SignalState>));
  meta::comms::DeviceBuffer barriersBuf(sizeof(BarrierState));
  meta::comms::DeviceBuffer barrierPeerPtrsBuf(nPeers * sizeof(BarrierState*));

  auto transports = makeZeroedSpan<Transport>(transportsBuf, nRanks);
  auto signalInbox = makeZeroedSpan<SignalState>(signalInboxBuf, 1);
  auto peerSignals =
      makeZeroedSpan<DeviceSpan<SignalState>>(peerSignalsBuf, nPeers);
  auto barriers = makeZeroedSpan<BarrierState>(barriersBuf, 1);
  auto barrierPeerPtrs =
      makeZeroedSpan<BarrierState*>(barrierPeerPtrsBuf, nPeers);

  peerIterationHelpersKernel<<<1, 1>>>(
      myRank,
      nRanks,
      transports,
      signalInbox,
      peerSignals,
      barriers,
      barrierPeerPtrs,
      results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// Peer Index Conversion Roundtrip Test
// =============================================================================

__global__ void peerIndexConversionRoundtripKernel(
    int myRank,
    int nRanks,
    DeviceSpan<Transport> transports,
    DeviceSpan<SignalState> signalInbox,
    DeviceSpan<DeviceSpan<SignalState>> peerSignals,
    DeviceSpan<BarrierState> barriers,
    DeviceSpan<BarrierState*> barrierPeerPtrs,
    int* results) {
  // Construct component objects
  DeviceWindowSignal signal(myRank, nRanks, 1, signalInbox, peerSignals);
  DeviceWindowBarrier barrier(myRank, nRanks, barriers, barrierPeerPtrs);
  DeviceWindowMemory wm(signal, barrier);

  // Construct MultiPeerDeviceTransport
  MultiPeerDeviceTransport transport(myRank, nRanks, transports, wm);

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
  for (int rank = 0; rank < nRanks; ++rank) {
    if (rank == myRank)
      continue;
    results[idx++] = static_cast<int>(transport.get_peer_transport(rank)->type);
  }
}

void testPeerIndexConversionRoundtrip(int myRank, int nRanks, int* results) {
  int nPeers = nRanks - 1;
  meta::comms::DeviceBuffer transportsBuf(nRanks * sizeof(Transport));
  meta::comms::DeviceBuffer signalInboxBuf(sizeof(SignalState));
  meta::comms::DeviceBuffer peerSignalsBuf(
      nPeers * sizeof(DeviceSpan<SignalState>));
  meta::comms::DeviceBuffer barriersBuf(sizeof(BarrierState));
  meta::comms::DeviceBuffer barrierPeerPtrsBuf(nPeers * sizeof(BarrierState*));

  // Set up transports: self = SELF type, peers = P2P_NVL type.
  auto* transportsPtr = static_cast<Transport*>(transportsBuf.get());
  CUDACHECK_TEST(
      cudaMemset(transportsBuf.get(), 0, nRanks * sizeof(Transport)));
  for (int i = 0; i < nRanks; ++i) {
    TransportType type =
        (i == myRank) ? TransportType::SELF : TransportType::P2P_NVL;
    CUDACHECK_TEST(cudaMemcpy(
        &transportsPtr[i].type,
        &type,
        sizeof(TransportType),
        cudaMemcpyHostToDevice));
  }

  DeviceSpan<Transport> transports(transportsPtr, nRanks);
  auto signalInbox = makeZeroedSpan<SignalState>(signalInboxBuf, 1);
  auto peerSignals =
      makeZeroedSpan<DeviceSpan<SignalState>>(peerSignalsBuf, nPeers);
  auto barriers = makeZeroedSpan<BarrierState>(barriersBuf, 1);
  auto barrierPeerPtrs =
      makeZeroedSpan<BarrierState*>(barrierPeerPtrsBuf, nPeers);

  peerIndexConversionRoundtripKernel<<<1, 1>>>(
      myRank,
      nRanks,
      transports,
      signalInbox,
      peerSignals,
      barriers,
      barrierPeerPtrs,
      results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceWindowMemory Accessors Test
// =============================================================================

__global__ void deviceWindowMemoryAccessorsKernel(
    int myRank,
    int nRanks,
    int signalCount,
    DeviceSpan<SignalState> signalInbox,
    DeviceSpan<DeviceSpan<SignalState>> peerSignals,
    DeviceSpan<BarrierState> barriers,
    DeviceSpan<BarrierState*> barrierPeerPtrs,
    int* results) {
  // Construct primitives
  DeviceWindowSignal signal(
      myRank, nRanks, signalCount, signalInbox, peerSignals);
  DeviceWindowBarrier barrier(myRank, nRanks, barriers, barrierPeerPtrs);

  // Construct DeviceWindowMemory
  DeviceWindowMemory wm(signal, barrier);

  // Verify signal() accessor returns correct metadata
  results[0] = wm.signal().rank();
  results[1] = wm.signal().n_ranks();
  results[2] = wm.signal().signal_count();

  // Verify barrier() accessor returns correct metadata
  results[3] = wm.barrier().rank();
  results[4] = wm.barrier().n_ranks();
}

void testDeviceWindowMemoryAccessors(
    int myRank,
    int nRanks,
    int signalCount,
    int* results) {
  int nPeers = nRanks - 1;
  meta::comms::DeviceBuffer signalInboxBuf(signalCount * sizeof(SignalState));
  meta::comms::DeviceBuffer peerSignalsBuf(
      nPeers * sizeof(DeviceSpan<SignalState>));
  meta::comms::DeviceBuffer barriersBuf(sizeof(BarrierState));
  meta::comms::DeviceBuffer barrierPeerPtrsBuf(nPeers * sizeof(BarrierState*));

  auto signalInbox = makeZeroedSpan<SignalState>(signalInboxBuf, signalCount);
  auto peerSignals =
      makeZeroedSpan<DeviceSpan<SignalState>>(peerSignalsBuf, nPeers);
  auto barriers = makeZeroedSpan<BarrierState>(barriersBuf, 1);
  auto barrierPeerPtrs =
      makeZeroedSpan<BarrierState*>(barrierPeerPtrsBuf, nPeers);

  deviceWindowMemoryAccessorsKernel<<<1, 1>>>(
      myRank,
      nRanks,
      signalCount,
      signalInbox,
      peerSignals,
      barriers,
      barrierPeerPtrs,
      results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

} // namespace comms::pipes::test
