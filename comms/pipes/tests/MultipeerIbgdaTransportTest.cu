// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/tests/MultipeerIbgdaTransportTest.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace comms::pipes::test {

// =============================================================================
// Kernel: Put with signal
// =============================================================================

__global__ void putSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  // Only thread 0 performs the put_signal
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto work =
        transport->put_signal(localBuf, remoteBuf, nbytes, signalId, signalVal);
    transport->wait_local(work);
  }
}

void testPutSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  putSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, localBuf, remoteBuf, nbytes, signalId, signalVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Wait for signal
// =============================================================================

__global__ void waitSignalKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    IbgdaCmpOp cmp,
    uint64_t expectedSignal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport->wait_signal(signalId, cmp, expectedSignal);
  }
}

void testWaitSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    int signalId,
    IbgdaCmpOp cmp,
    uint64_t expectedSignal,
    int numBlocks,
    int blockSize) {
  waitSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, signalId, cmp, expectedSignal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Multiple put_signal operations
// =============================================================================

__global__ void multiplePutSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t bytesPerPut,
    int signalId,
    int numPuts) {
  // Only thread 0 performs the puts
  auto group = make_block_group();
  if (group.is_global_leader()) {
    for (int i = 0; i < numPuts; i++) {
      IbgdaLocalBuffer srcBuf = localBuf.subBuffer(i * bytesPerPut);
      IbgdaRemoteBuffer dstBuf = remoteBuf.subBuffer(i * bytesPerPut);

      // Signal value is i+1 (cumulative count)
      auto work =
          transport->put_signal(srcBuf, dstBuf, bytesPerPut, signalId, 1);
      transport->wait_local(work);
    }
  }
}

void testMultiplePutSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t bytesPerPut,
    int signalId,
    int numPuts,
    int numBlocks,
    int blockSize) {
  multiplePutSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, localBuf, remoteBuf, bytesPerPut, signalId, numPuts);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Signal only (no data)
// =============================================================================

__global__ void signalOnlyKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t signalVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto work = transport->signal(signalId, signalVal);
    transport->wait_local(work);
  }
}

void testSignalOnly(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  signalOnlyKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, signalId, signalVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Reset signal
// =============================================================================

__global__ void resetSignalKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // reset_signal is now synchronous (includes fences and wait internally)
    transport->reset_signal(signalId);
  }
}

void testResetSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    int signalId,
    int numBlocks,
    int blockSize) {
  resetSignalKernel<<<numBlocks, blockSize>>>(deviceTransportPtr, signalId);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Put only (no signal)
// =============================================================================

__global__ void putOnlyKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto work = transport->put(localBuf, remoteBuf, nbytes);
    transport->wait_local(work);
  }
}

void testPutOnly(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  putOnlyKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, localBuf, remoteBuf, nbytes);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Read signal value
// =============================================================================

__global__ void readSignalKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t* result) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    *result = transport->read_signal(signalId);
  }
}

void testReadSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    int signalId,
    uint64_t* d_result,
    int numBlocks,
    int blockSize) {
  readSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, signalId, d_result);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Fill buffer with pattern
// =============================================================================

__global__ void
fillPatternKernel(uint8_t* buffer, std::size_t nbytes, uint8_t baseValue) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = idx; i < nbytes; i += stride) {
    buffer[i] = static_cast<uint8_t>(baseValue + (i % 256));
  }
}

void fillBufferWithPattern(
    void* buffer,
    std::size_t nbytes,
    uint8_t baseValue,
    int numBlocks,
    int blockSize) {
  fillPatternKernel<<<numBlocks, blockSize>>>(
      static_cast<uint8_t*>(buffer), nbytes, baseValue);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Verify buffer pattern
// =============================================================================

__global__ void verifyPatternKernel(
    const uint8_t* buffer,
    std::size_t nbytes,
    uint8_t expectedBaseValue,
    int* errorCount) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = idx; i < nbytes; i += stride) {
    uint8_t expected = static_cast<uint8_t>(expectedBaseValue + (i % 256));
    if (buffer[i] != expected) {
      atomicAdd(errorCount, 1);
    }
  }
}

void verifyBufferPattern(
    const void* buffer,
    std::size_t nbytes,
    uint8_t expectedBaseValue,
    int* errorCount,
    int numBlocks,
    int blockSize) {
  verifyPatternKernel<<<numBlocks, blockSize>>>(
      static_cast<const uint8_t*>(buffer),
      nbytes,
      expectedBaseValue,
      errorCount);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Wait for ready signal, then put with signal
// =============================================================================

__global__ void waitReadyThenPutSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int readySignalId,
    uint64_t readySignalVal,
    int dataSignalId,
    uint64_t dataSignalVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // Wait for receiver to signal that its buffer is ready
    transport->wait_signal(readySignalId, IbgdaCmpOp::GE, readySignalVal);

    // Now put data with signal
    auto work = transport->put_signal(
        localBuf, remoteBuf, nbytes, dataSignalId, dataSignalVal);
    transport->wait_local(work);
  }
}

void testWaitReadyThenPutSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int readySignalId,
    uint64_t readySignalVal,
    int dataSignalId,
    uint64_t dataSignalVal,
    int numBlocks,
    int blockSize) {
  waitReadyThenPutSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr,
      localBuf,
      remoteBuf,
      nbytes,
      readySignalId,
      readySignalVal,
      dataSignalId,
      dataSignalVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Bidirectional - one thread does put_signal, another does wait_signal
// =============================================================================

__global__ void bidirectionalPutWaitKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int sendSignalId,
    uint64_t sendSignalVal,
    int recvSignalId,
    uint64_t recvSignalVal) {
  // Use ThreadGroup to partition threads:
  // Thread 0 (leader): send data via put_signal
  // Thread 1: wait for incoming signal
  auto group = make_block_group();
  if (group.group_id == 0) {
    if (group.is_leader()) {
      // Send data to peer
      auto work = transport->put_signal(
          localBuf, remoteBuf, nbytes, sendSignalId, sendSignalVal);
      transport->wait_local(work);
    } else if (group.thread_id_in_group == 1) {
      // Wait for data from peer
      transport->wait_signal(recvSignalId, IbgdaCmpOp::GE, recvSignalVal);
    }
  }
}

void testBidirectionalPutWait(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int sendSignalId,
    uint64_t sendSignalVal,
    int recvSignalId,
    uint64_t recvSignalVal,
    int numBlocks,
    int blockSize) {
  bidirectionalPutWaitKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr,
      localBuf,
      remoteBuf,
      nbytes,
      sendSignalId,
      sendSignalVal,
      recvSignalId,
      recvSignalVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Multi-peer bidirectional - partition groups by peer, then send/recv
// =============================================================================

__global__ void allToAllSendKernel(
    P2pIbgdaTransportDevice** peerTransports,
    IbgdaLocalBuffer* localSendBufs,
    IbgdaRemoteBuffer* peerRecvBufs,
    int myRank,
    std::size_t nbytes,
    int numPeers) {
  // Partition groups by peer: each peer gets a subset of groups
  // [peer_id, per_peer_group] = group.partition(numPeers)
  auto group = make_block_group();
  auto [peerId, perPeerGroup] = group.partition(numPeers);

  // Only the leader of each peer group sends data
  P2pIbgdaTransportDevice* transport = peerTransports[peerId];

  if (perPeerGroup.is_leader()) {
    // Send data to this peer
    // Signal ID = myRank (sender's rank - receiver waits on this)
    auto work = transport->put_signal(
        localSendBufs[peerId], peerRecvBufs[peerId], nbytes, myRank, 1);
    transport->wait_local(work);
  }
}

__global__ void allToAllWaitKernel(
    P2pIbgdaTransportDevice** peerTransports,
    int* peerRanks,
    int numPeers) {
  // Partition groups by peer: each peer gets a subset of groups
  auto group = make_block_group();
  auto [peerId, perPeerGroup] = group.partition(numPeers);

  // Only the leader waits for signal from each peer
  P2pIbgdaTransportDevice* transport = peerTransports[peerId];
  int peerRank = peerRanks[peerId];

  if (perPeerGroup.is_leader()) {
    // Wait for signal from this peer
    // Signal ID = peerRank (sender's rank)
    transport->wait_signal(peerRank, IbgdaCmpOp::GE, 1);
  }
}

void testAllToAll(
    P2pIbgdaTransportDevice** peerTransports,
    IbgdaLocalBuffer* localSendBufs,
    IbgdaRemoteBuffer* peerRecvBufs,
    int* peerRanks,
    int myRank,
    std::size_t nbytes,
    int numPeers,
    int numBlocks,
    int blockSize) {
  // Phase 1: All ranks send data to all peers
  allToAllSendKernel<<<numBlocks, blockSize>>>(
      peerTransports, localSendBufs, peerRecvBufs, myRank, nbytes, numPeers);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

void testAllToAllWait(
    P2pIbgdaTransportDevice** peerTransports,
    int* peerRanks,
    int numPeers,
    int numBlocks,
    int blockSize) {
  // Phase 2: Wait for signals from all peers
  allToAllWaitKernel<<<numBlocks, blockSize>>>(
      peerTransports, peerRanks, numPeers);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

} // namespace comms::pipes::test
