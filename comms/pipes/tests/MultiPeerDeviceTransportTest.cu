// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/MultiPeerDeviceTransportTest.cuh"

#include <vector>

#include "comms/pipes/DeviceBarrier.cuh"
#include "comms/pipes/DeviceCounter.cuh"
#include "comms/pipes/DeviceSignal.cuh"
#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Transport.cuh"
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
// DeviceSignal Construction Test
// =============================================================================

__global__ void deviceSignalConstructionKernel(
    int myRank,
    int nRanks,
    int signalCount,
    DeviceSpan<SignalState> localInbox,
    DeviceSpan<SignalState*> peerInboxPtrs,
    int* results) {
  DeviceSignal signal(myRank, nRanks, signalCount, localInbox, peerInboxPtrs);

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
  meta::comms::DeviceBuffer localInboxBuf(signalCount * sizeof(SignalState));
  meta::comms::DeviceBuffer peerInboxPtrsBuf(nRanks * sizeof(SignalState*));

  auto localInbox = makeZeroedSpan<SignalState>(localInboxBuf, signalCount);
  auto peerInboxPtrs = makeZeroedSpan<SignalState*>(peerInboxPtrsBuf, nRanks);

  deviceSignalConstructionKernel<<<1, 1>>>(
      myRank, nRanks, signalCount, localInbox, peerInboxPtrs, results);

  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceCounter Construction Test
// =============================================================================

__global__ void deviceCounterConstructionKernel(
    int counterCount,
    SignalState* counters,
    uint32_t* results) {
  // Construct DeviceCounter with provided buffers
  DeviceSpan<SignalState> countersSpan(counters, counterCount);
  DeviceCounter counter(countersSpan);

  // Verify accessors return correct values
  results[0] = counter.counter_count();
}

void testDeviceCounterConstruction(int counterCount, uint32_t* results) {
  // Allocate minimal buffers for construction test
  SignalState* counters = nullptr;

  cudaMalloc(&counters, counterCount * sizeof(SignalState));
  cudaMemset(counters, 0, counterCount * sizeof(SignalState));

  deviceCounterConstructionKernel<<<1, 1>>>(counterCount, counters, results);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  cudaFree(counters);
}

// =============================================================================
// DeviceBarrier Construction Test
// =============================================================================

__global__ void deviceBarrierConstructionKernel(
    int myRank,
    int nRanks,
    DeviceSpan<BarrierState> localBarriers,
    DeviceSpan<BarrierState*> peerBarrierPtrs,
    int* results) {
  DeviceBarrier barrier(myRank, nRanks, localBarriers, peerBarrierPtrs);

  // Verify accessors return correct values
  results[0] = barrier.rank();
  results[1] = barrier.n_ranks();
}

void testDeviceBarrierConstruction(int myRank, int nRanks, int* results) {
  int nPeers = nRanks - 1;
  meta::comms::DeviceBuffer localBarriersBuf(sizeof(BarrierState));
  meta::comms::DeviceBuffer peerBarrierPtrsBuf(nPeers * sizeof(BarrierState*));

  auto localBarriers = makeZeroedSpan<BarrierState>(localBarriersBuf, 1);
  auto peerBarrierPtrs =
      makeZeroedSpan<BarrierState*>(peerBarrierPtrsBuf, nPeers);

  deviceBarrierConstructionKernel<<<1, 1>>>(
      myRank, nRanks, localBarriers, peerBarrierPtrs, results);

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
    DeviceSpan<SignalState*> signalPeerPtrs,
    DeviceSpan<SignalState> counters,
    DeviceSpan<BarrierState> barriers,
    DeviceSpan<BarrierState*> barrierPeerPtrs,
    int* results) {
  // Construct component objects
  DeviceSignal signal(myRank, nRanks, 1, signalInbox, signalPeerPtrs);
  DeviceCounter counter(counters);
  DeviceBarrier barrier(myRank, nRanks, barriers, barrierPeerPtrs);

  // Construct MultiPeerDeviceTransport
  MultiPeerDeviceTransport transport(
      myRank, nRanks, transports, signal, counter, barrier);

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
  meta::comms::DeviceBuffer signalPeerPtrsBuf(nPeers * sizeof(SignalState*));
  meta::comms::DeviceBuffer countersBuf(sizeof(SignalState));
  meta::comms::DeviceBuffer barriersBuf(sizeof(BarrierState));
  meta::comms::DeviceBuffer barrierPeerPtrsBuf(nPeers * sizeof(BarrierState*));

  auto transports = makeZeroedSpan<Transport>(transportsBuf, nRanks);
  auto signalInbox = makeZeroedSpan<SignalState>(signalInboxBuf, 1);
  auto signalPeerPtrs =
      makeZeroedSpan<SignalState*>(signalPeerPtrsBuf, nPeers);
  auto counters = makeZeroedSpan<SignalState>(countersBuf, 1);
  auto barriers = makeZeroedSpan<BarrierState>(barriersBuf, 1);
  auto barrierPeerPtrs =
      makeZeroedSpan<BarrierState*>(barrierPeerPtrsBuf, nPeers);

  multiPeerDeviceTransportConstructionKernel<<<1, 1>>>(
      myRank,
      nRanks,
      transports,
      signalInbox,
      signalPeerPtrs,
      counters,
      barriers,
      barrierPeerPtrs,
      results);
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
    DeviceSpan<SignalState*> signalPeerPtrs,
    DeviceSpan<SignalState> counters,
    DeviceSpan<BarrierState> barriers,
    DeviceSpan<BarrierState*> barrierPeerPtrs,
    int* results) {
  // Construct component objects
  DeviceSignal signal(myRank, nRanks, 1, signalInbox, signalPeerPtrs);
  DeviceCounter counter(counters);
  DeviceBarrier barrier(myRank, nRanks, barriers, barrierPeerPtrs);

  // Construct MultiPeerDeviceTransport
  MultiPeerDeviceTransport transport(
      myRank, nRanks, transports, signal, counter, barrier);

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
  meta::comms::DeviceBuffer signalPeerPtrsBuf(nPeers * sizeof(SignalState*));
  meta::comms::DeviceBuffer countersBuf(sizeof(SignalState));
  meta::comms::DeviceBuffer barriersBuf(sizeof(BarrierState));
  meta::comms::DeviceBuffer barrierPeerPtrsBuf(nPeers * sizeof(BarrierState*));

  auto transports = makeZeroedSpan<Transport>(transportsBuf, nRanks);
  auto signalInbox = makeZeroedSpan<SignalState>(signalInboxBuf, 1);
  auto signalPeerPtrs =
      makeZeroedSpan<SignalState*>(signalPeerPtrsBuf, nPeers);
  auto counters = makeZeroedSpan<SignalState>(countersBuf, 1);
  auto barriers = makeZeroedSpan<BarrierState>(barriersBuf, 1);
  auto barrierPeerPtrs =
      makeZeroedSpan<BarrierState*>(barrierPeerPtrsBuf, nPeers);

  peerIterationHelpersKernel<<<1, 1>>>(
      myRank,
      nRanks,
      transports,
      signalInbox,
      signalPeerPtrs,
      counters,
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
    DeviceSpan<SignalState*> signalPeerPtrs,
    DeviceSpan<SignalState> counters,
    DeviceSpan<BarrierState> barriers,
    DeviceSpan<BarrierState*> barrierPeerPtrs,
    int* results) {
  // Construct component objects
  DeviceSignal signal(myRank, nRanks, 1, signalInbox, signalPeerPtrs);
  DeviceCounter counter(counters);
  DeviceBarrier barrier(myRank, nRanks, barriers, barrierPeerPtrs);

  // Construct MultiPeerDeviceTransport
  MultiPeerDeviceTransport transport(
      myRank, nRanks, transports, signal, counter, barrier);

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
  int nPeers = nRanks - 1;
  meta::comms::DeviceBuffer transportsBuf(nRanks * sizeof(Transport));
  meta::comms::DeviceBuffer signalInboxBuf(sizeof(SignalState));
  meta::comms::DeviceBuffer signalPeerPtrsBuf(nPeers * sizeof(SignalState*));
  meta::comms::DeviceBuffer countersBuf(sizeof(SignalState));
  meta::comms::DeviceBuffer barriersBuf(sizeof(BarrierState));
  meta::comms::DeviceBuffer barrierPeerPtrsBuf(nPeers * sizeof(BarrierState*));

  // Set up transports: self = SELF type, peers = P2P_NVL type.
  auto* transportsPtr = static_cast<Transport*>(transportsBuf.get());
  CUDACHECK_TEST(cudaMemset(transportsBuf.get(), 0, nRanks * sizeof(Transport)));
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
  auto signalPeerPtrs =
      makeZeroedSpan<SignalState*>(signalPeerPtrsBuf, nPeers);
  auto counters = makeZeroedSpan<SignalState>(countersBuf, 1);
  auto barriers = makeZeroedSpan<BarrierState>(barriersBuf, 1);
  auto barrierPeerPtrs =
      makeZeroedSpan<BarrierState*>(barrierPeerPtrsBuf, nPeers);

  peerIndexConversionRoundtripKernel<<<1, 1>>>(
      myRank,
      nRanks,
      transports,
      signalInbox,
      signalPeerPtrs,
      counters,
      barriers,
      barrierPeerPtrs,
      results);
  CUDACHECK_TEST(cudaGetLastError());
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

// =============================================================================
// DeviceCounter Increment and Read Test
// =============================================================================

__global__ void counterIncrementAndReadKernel(
    int counterCount,
    SignalState* counters,
    uint64_t* results) {
  auto group = make_warp_group();

  DeviceSpan<SignalState> countersSpan(counters, counterCount);
  DeviceCounter counter(countersSpan);

  // Increment counter 0 with value=1
  counter.increment_counter(group, 0, 1);
  group.sync();

  // Read back and store result
  results[0] = counter.read_counter(0);
}

void testCounterIncrementAndRead(int counterCount, uint64_t* results) {
  SignalState* counters = nullptr;

  cudaMalloc(&counters, counterCount * sizeof(SignalState));
  cudaMemset(counters, 0, counterCount * sizeof(SignalState));

  counterIncrementAndReadKernel<<<1, 32>>>(counterCount, counters, results);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  cudaFree(counters);
}

// =============================================================================
// DeviceCounter Value Accumulation Test
// =============================================================================

__global__ void counterValueAccumulationKernel(
    int counterCount,
    SignalState* counters,
    uint64_t* results) {
  auto group = make_warp_group();

  DeviceSpan<SignalState> countersSpan(counters, counterCount);
  DeviceCounter counter(countersSpan);

  // Increment 3 times with value=1
  counter.increment_counter(group, 0, 1);
  group.sync();
  counter.increment_counter(group, 0, 1);
  group.sync();
  counter.increment_counter(group, 0, 1);
  group.sync();

  // Read back - should be 3
  results[0] = counter.read_counter(0);
}

void testCounterValueAccumulation(int counterCount, uint64_t* results) {
  SignalState* counters = nullptr;

  cudaMalloc(&counters, counterCount * sizeof(SignalState));
  cudaMemset(counters, 0, counterCount * sizeof(SignalState));

  counterValueAccumulationKernel<<<1, 32>>>(counterCount, counters, results);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  cudaFree(counters);
}

// =============================================================================
// DeviceCounter Increment Custom Value Test
// =============================================================================

__global__ void counterIncrementCustomValueKernel(
    int counterCount,
    SignalState* counters,
    uint64_t incrementValue,
    uint64_t* results) {
  auto group = make_warp_group();

  DeviceSpan<SignalState> countersSpan(counters, counterCount);
  DeviceCounter counter(countersSpan);

  // Increment counter 0 with custom value
  counter.increment_counter(group, 0, incrementValue);
  group.sync();

  // Read back
  results[0] = counter.read_counter(0);
}

void testCounterIncrementCustomValue(
    int counterCount,
    uint64_t incrementValue,
    uint64_t* results) {
  SignalState* counters = nullptr;

  cudaMalloc(&counters, counterCount * sizeof(SignalState));
  cudaMemset(counters, 0, counterCount * sizeof(SignalState));

  counterIncrementCustomValueKernel<<<1, 32>>>(
      counterCount, counters, incrementValue, results);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  cudaFree(counters);
}

// =============================================================================
// DeviceCounter Wait CMP_GE Test
// =============================================================================

__global__ void
waitCounterCmpGeKernel(int counterCount, SignalState* counters, int* results) {
  auto group = make_warp_group();

  DeviceSpan<SignalState> countersSpan(counters, counterCount);
  DeviceCounter counter(countersSpan);

  // Increment counter to 5
  for (int i = 0; i < 5; ++i) {
    counter.increment_counter(group, 0, 1);
    group.sync();
  }

  // Wait for counter >= 5 (should pass immediately)
  counter.wait_counter(group, 0, CmpOp::CMP_GE, 5);

  // If we get here, wait succeeded
  results[0] = 1;
}

void testWaitCounterCmpGe(int counterCount, int* results) {
  SignalState* counters = nullptr;

  cudaMalloc(&counters, counterCount * sizeof(SignalState));
  cudaMemset(counters, 0, counterCount * sizeof(SignalState));

  waitCounterCmpGeKernel<<<1, 32>>>(counterCount, counters, results);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  cudaFree(counters);
}

// =============================================================================
// DeviceCounter Wait CMP_EQ Test
// =============================================================================

__global__ void
waitCounterCmpEqKernel(int counterCount, SignalState* counters, int* results) {
  auto group = make_warp_group();

  DeviceSpan<SignalState> countersSpan(counters, counterCount);
  DeviceCounter counter(countersSpan);

  // Increment counter to 3
  for (int i = 0; i < 3; ++i) {
    counter.increment_counter(group, 0, 1);
    group.sync();
  }

  // Wait for counter == 3 (should pass immediately)
  counter.wait_counter(group, 0, CmpOp::CMP_EQ, 3);

  // If we get here, wait succeeded
  results[0] = 1;
}

void testWaitCounterCmpEq(int counterCount, int* results) {
  SignalState* counters = nullptr;

  cudaMalloc(&counters, counterCount * sizeof(SignalState));
  cudaMemset(counters, 0, counterCount * sizeof(SignalState));

  waitCounterCmpEqKernel<<<1, 32>>>(counterCount, counters, results);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  cudaFree(counters);
}

// =============================================================================
// DeviceCounter Reset Single Counter Test
// =============================================================================

__global__ void
resetCounterKernel(int counterCount, SignalState* counters, uint64_t* results) {
  auto group = make_warp_group();

  DeviceSpan<SignalState> countersSpan(counters, counterCount);
  DeviceCounter counter(countersSpan);

  // Increment counter to 5
  for (int i = 0; i < 5; ++i) {
    counter.increment_counter(group, 0, 1);
    group.sync();
  }

  // Reset the counter
  counter.reset_counter(0);

  // Read back - should be 0
  results[0] = counter.read_counter(0);
}

void testResetCounter(int counterCount, uint64_t* results) {
  SignalState* counters = nullptr;

  cudaMalloc(&counters, counterCount * sizeof(SignalState));
  cudaMemset(counters, 0, counterCount * sizeof(SignalState));

  resetCounterKernel<<<1, 32>>>(counterCount, counters, results);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  cudaFree(counters);
}

// =============================================================================
// DeviceCounter Reset All Counters Test
// =============================================================================

__global__ void resetAllCountersKernel(
    int counterCount,
    SignalState* counters,
    uint64_t* results) {
  auto group = make_warp_group();

  DeviceSpan<SignalState> countersSpan(counters, counterCount);
  DeviceCounter counter(countersSpan);

  // Increment counters to different values
  counter.increment_counter(group, 0, 10);
  group.sync();
  counter.increment_counter(group, 1, 20);
  group.sync();
  counter.increment_counter(group, 2, 30);
  group.sync();

  // Reset all counters
  counter.reset_all_counters();

  // Read back all - should all be 0
  for (int i = 0; i < counterCount; ++i) {
    results[i] = counter.read_counter(i);
  }
}

void testResetAllCounters(int counterCount, uint64_t* results) {
  SignalState* counters = nullptr;

  cudaMalloc(&counters, counterCount * sizeof(SignalState));
  cudaMemset(counters, 0, counterCount * sizeof(SignalState));

  resetAllCountersKernel<<<1, 32>>>(counterCount, counters, results);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  cudaFree(counters);
}

// =============================================================================
// DeviceCounter Multiple Counter Independence Test
// =============================================================================

__global__ void multipleCounterIndependenceKernel(
    int counterCount,
    SignalState* counters,
    uint64_t* results) {
  auto group = make_warp_group();

  DeviceSpan<SignalState> countersSpan(counters, counterCount);
  DeviceCounter counter(countersSpan);

  // Increment counter 0 with value=10
  counter.increment_counter(group, 0, 10);
  group.sync();

  // Increment counter 1 with value=20
  counter.increment_counter(group, 1, 20);
  group.sync();

  // Increment counter 2 with value=30
  counter.increment_counter(group, 2, 30);
  group.sync();

  // Read back each counter
  results[0] = counter.read_counter(0);
  results[1] = counter.read_counter(1);
  results[2] = counter.read_counter(2);
}

void testMultipleCounterIndependence(int counterCount, uint64_t* results) {
  SignalState* counters = nullptr;

  cudaMalloc(&counters, counterCount * sizeof(SignalState));
  cudaMemset(counters, 0, counterCount * sizeof(SignalState));

  multipleCounterIndependenceKernel<<<1, 32>>>(counterCount, counters, results);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  cudaFree(counters);
}

} // namespace comms::pipes::test
