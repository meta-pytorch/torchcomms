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
    MultiPeerDeviceTransport transport,
    int* results) {
  results[0] = transport.rank();
  results[1] = transport.n_ranks();
  results[2] = transport.num_peers();
}

void testMultiPeerDeviceTransportAccessors(
    MultiPeerDeviceTransport transport,
    int* results) {
  multiPeerDeviceTransportAccessorsKernel<<<1, 1>>>(transport, results);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Single-Peer Send/Recv Tests
// =============================================================================

__global__ void singlePeerSendKernel(
    MultiPeerDeviceTransport transport,
    int peerIndex,
    void* srcBuff,
    std::size_t nbytes) {
  auto group = make_warp_group();
  transport.send(peerIndex, group, srcBuff, nbytes);
}

void testSinglePeerSend(
    MultiPeerDeviceTransport transport,
    int peerIndex,
    void* srcBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  singlePeerSendKernel<<<numBlocks, blockSize>>>(
      transport, peerIndex, srcBuff, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
}

__global__ void singlePeerRecvKernel(
    MultiPeerDeviceTransport transport,
    int peerIndex,
    void* dstBuff,
    std::size_t nbytes) {
  auto group = make_warp_group();
  transport.recv(peerIndex, group, dstBuff, nbytes);
}

void testSinglePeerRecv(
    MultiPeerDeviceTransport transport,
    int peerIndex,
    void* dstBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  singlePeerRecvKernel<<<numBlocks, blockSize>>>(
      transport, peerIndex, dstBuff, nbytes);
  CUDACHECK_TEST(cudaGetLastError());
}

// =============================================================================
// Multi-Peer Send/Recv Tests (Peer Iteration)
// =============================================================================

__global__ void multiPeerSendAllPeersKernel(
    MultiPeerDeviceTransport transport,
    void** srcBuffs,
    std::size_t nbytesPerPeer) {
  auto group = make_warp_group();

  // send() now takes peer index directly — no conversion needed
  int numPeers = transport.num_peers();
  for (int i = 0; i < numPeers; ++i) {
    transport.send(i, group, srcBuffs[i], nbytesPerPeer, i);
  }
}

void testMultiPeerSendAllPeers(
    MultiPeerDeviceTransport transport,
    void** srcBuffs,
    std::size_t nbytesPerPeer,
    int numBlocks,
    int blockSize) {
  multiPeerSendAllPeersKernel<<<numBlocks, blockSize>>>(
      transport, srcBuffs, nbytesPerPeer);
  CUDACHECK_TEST(cudaGetLastError());
}

__global__ void multiPeerRecvAllPeersKernel(
    MultiPeerDeviceTransport transport,
    void** dstBuffs,
    std::size_t nbytesPerPeer) {
  auto group = make_warp_group();

  // recv() now takes peer index directly — no conversion needed
  int numPeers = transport.num_peers();
  for (int i = 0; i < numPeers; ++i) {
    transport.recv(i, group, dstBuffs[i], nbytesPerPeer, i);
  }
}

void testMultiPeerRecvAllPeers(
    MultiPeerDeviceTransport transport,
    void** dstBuffs,
    std::size_t nbytesPerPeer,
    int numBlocks,
    int blockSize) {
  multiPeerRecvAllPeersKernel<<<numBlocks, blockSize>>>(
      transport, dstBuffs, nbytesPerPeer);
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
  int numPeers = transport.num_peers();
  for (int peerIdx = 0; peerIdx < numPeers; ++peerIdx) {
    int rank = transport.peer_index_to_rank(peerIdx);
    results[1 + rank] =
        static_cast<int>(transport.get_peer_transport(peerIdx)->type);
  }
}

void testTransportTypes(
    const MultiPeerDeviceTransport& transport,
    int* results) {
  transportTypesKernel<<<1, 1>>>(transport, results);
  CUDACHECK_TEST(cudaGetLastError());
}

} // namespace comms::pipes::test
