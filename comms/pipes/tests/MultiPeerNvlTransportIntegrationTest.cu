// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/MultiPeerNvlTransportIntegrationTest.cuh"

#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes::test {

// =============================================================================
// MultiPeerDeviceTransport Accessors Test
// =============================================================================

__global__ void multiPeerDeviceTransportAccessorsKernel(
    MultiPeerDeviceTransport& transport,
    int* results) {
  results[0] = transport.rank();
  results[1] = transport.n_ranks();
  results[2] = transport.num_peers();
}

void testMultiPeerDeviceTransportAccessors(
    MultiPeerDeviceTransport& transport,
    int* results) {
  multiPeerDeviceTransportAccessorsKernel<<<1, 1>>>(transport, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Single-Peer Send/Recv Tests
// =============================================================================

__global__ void singlePeerSendKernel(
    MultiPeerDeviceTransport& transport,
    int peerRank,
    void* srcBuff,
    std::size_t nbytes) {
  auto group = make_warp_group();
  transport.send(peerRank, group, srcBuff, nbytes);
}

void testSinglePeerSend(
    MultiPeerDeviceTransport& transport,
    int peerRank,
    void* srcBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  singlePeerSendKernel<<<numBlocks, blockSize>>>(
      transport, peerRank, srcBuff, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
}

__global__ void singlePeerRecvKernel(
    MultiPeerDeviceTransport& transport,
    int peerRank,
    void* dstBuff,
    std::size_t nbytes) {
  auto group = make_warp_group();
  transport.recv(peerRank, group, dstBuff, nbytes);
}

void testSinglePeerRecv(
    MultiPeerDeviceTransport& transport,
    int peerRank,
    void* dstBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  singlePeerRecvKernel<<<numBlocks, blockSize>>>(
      transport, peerRank, dstBuff, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Multi-Peer Send/Recv Test (Parallel via Partition)
// =============================================================================

__global__ void multiPeerSendRecvAllPeersKernel(
    MultiPeerDeviceTransport& transport,
    void** srcBuffs,
    void** dstBuffs,
    std::size_t nbytesPerPeer) {
  auto group = make_warp_group();

  int myRank = transport.rank();
  int numPeers = transport.num_peers();

  // Partition into send and recv groups (interleaved for SM balance)
  auto [partition_id, send_recv_group] = group.partition_interleaved(2);

  // Further partition across peers
  auto [peer_idx, group_per_peer] =
      send_recv_group.partition_interleaved(numPeers);

  // Map peer_idx to actual rank (skip self)
  int peer_rank = peer_idx < myRank ? peer_idx : peer_idx + 1;

  if (partition_id == 0) {
    transport.send(
        peer_rank, group_per_peer, srcBuffs[peer_idx], nbytesPerPeer, peer_idx);
  } else {
    transport.recv(
        peer_rank, group_per_peer, dstBuffs[peer_idx], nbytesPerPeer, peer_idx);
  }
}

void testMultiPeerSendRecvAllPeers(
    MultiPeerDeviceTransport& transport,
    void** srcBuffs,
    void** dstBuffs,
    std::size_t nbytesPerPeer,
    int numBlocks,
    int blockSize) {
  multiPeerSendRecvAllPeersKernel<<<numBlocks, blockSize>>>(
      transport, srcBuffs, dstBuffs, nbytesPerPeer);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Transport Types Test
// =============================================================================

__global__ void transportTypesKernel(
    const MultiPeerDeviceTransport& transport,
    int* results) {
  // Output numPeers in results[0]
  results[0] = transport.num_peers();

  // Self transport type
  int myRank = transport.rank();
  results[1 + myRank] = static_cast<int>(transport.get_self_transport()->type);

  // Peer transport types
  int nRanks = transport.n_ranks();
  for (int r = 0; r < nRanks; ++r) {
    if (r == myRank)
      continue;
    results[1 + r] = static_cast<int>(transport.get_peer_transport(r)->type);
  }
}

void testTransportTypes(
    const MultiPeerDeviceTransport& transport,
    int* results) {
  transportTypesKernel<<<1, 1>>>(transport, results);
  CUDACHECK_TEST(cudaGetLastError());
}

} // namespace comms::pipes::test
