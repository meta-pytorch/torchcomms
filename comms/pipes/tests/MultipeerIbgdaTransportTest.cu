// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/tests/MultipeerIbgdaTransportTest.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace comms::pipes::test {

// =============================================================================
// Kernel: Put data + signal remote (adaptive-routing safe, with NIC fence)
// =============================================================================

__global__ void putAndSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport->put(localBuf, remoteBuf, nbytes, signalId, signalVal);
    transport->fence();
  }
}

void testPutAndSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  putAndSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, localBuf, remoteBuf, nbytes, signalId, signalVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Group-collaborative put + signal (warp group)
// =============================================================================

__global__ void putAndSignalGroupKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  auto group = make_warp_group();

  // Group-cooperative put with signal (single put+signal, not two puts)
  transport->put(group, localBuf, remoteBuf, nbytes, signalId, signalVal);

  transport->fence(group);
}

void testPutAndSignalGroup(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  putAndSignalGroupKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, localBuf, remoteBuf, nbytes, signalId, signalVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Multi-warp group-collaborative put + signal
// Each warp partitions data manually, then calls group-scope put + signal
// =============================================================================

__global__ void putAndSignalGroupMultiWarpKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  auto group = make_warp_group();

  // Manually partition data across all warp groups
  std::size_t chunkSize = nbytes / group.total_groups;
  std::size_t offset = group.group_id * chunkSize;
  std::size_t myBytes = (group.group_id == group.total_groups - 1)
      ? (nbytes - offset)
      : chunkSize;

  IbgdaLocalBuffer myLocalBuf = localBuf.subBuffer(offset);
  IbgdaRemoteBuffer myRemoteBuf = remoteBuf.subBuffer(offset);

  // Each warp group does put + signal (each signal adds signalVal)
  transport->put(group, myLocalBuf, myRemoteBuf, myBytes, signalId, signalVal);

  transport->fence(group);
}

void testPutAndSignalGroupMultiWarp(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  putAndSignalGroupMultiWarpKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, localBuf, remoteBuf, nbytes, signalId, signalVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Block-scope group-collaborative put + signal
// Each block partitions data manually, then calls group-scope put + signal
// =============================================================================

__global__ void putAndSignalGroupBlockKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  auto group = make_block_group();

  // Manually partition data across all block groups
  std::size_t chunkSize = nbytes / group.total_groups;
  std::size_t offset = group.group_id * chunkSize;
  std::size_t myBytes = (group.group_id == group.total_groups - 1)
      ? (nbytes - offset)
      : chunkSize;

  IbgdaLocalBuffer myLocalBuf = localBuf.subBuffer(offset);
  IbgdaRemoteBuffer myRemoteBuf = remoteBuf.subBuffer(offset);

  // Each block group does put + signal (each signal adds signalVal)
  transport->put(group, myLocalBuf, myRemoteBuf, myBytes, signalId, signalVal);

  transport->flush(group);
}

void testPutAndSignalGroupBlock(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  putAndSignalGroupBlockKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, localBuf, remoteBuf, nbytes, signalId, signalVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Wait for signal (volatile spin on local signal buffer)
// =============================================================================

__global__ void waitSignalKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t expectedSignal) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    transport->wait_signal(signalId, expectedSignal);
  }
}

void testWaitSignal(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t expectedSignal,
    int numBlocks,
    int blockSize) {
  waitSignalKernel<<<numBlocks, blockSize>>>(
      transport, signalId, expectedSignal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Multiple put + signal operations
// =============================================================================

__global__ void multiplePutAndSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t bytesPerPut,
    int signalId,
    int numPuts) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    for (int i = 0; i < numPuts; i++) {
      IbgdaLocalBuffer srcBuf = localBuf.subBuffer(i * bytesPerPut);
      IbgdaRemoteBuffer dstBuf = remoteBuf.subBuffer(i * bytesPerPut);

      transport->put(srcBuf, dstBuf, bytesPerPut, signalId, 1);
      transport->fence();
    }
  }
}

void testMultiplePutAndSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t bytesPerPut,
    int signalId,
    int numPuts,
    int numBlocks,
    int blockSize) {
  multiplePutAndSignalKernel<<<numBlocks, blockSize>>>(
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
    transport->signal(signalId, signalVal);
    transport->flush();
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
// Kernel: Put only (no signal)
// =============================================================================

__global__ void putOnlyKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport->put(localBuf, remoteBuf, nbytes);
    transport->flush();
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
// Kernel: Wait for ready signal, then put + signal
// =============================================================================

__global__ void waitReadyThenPutAndSignalKernel(
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
    // Wait for receiver to signal that its buffer is ready (local inbox)
    transport->wait_signal(readySignalId, readySignalVal);

    // Now put data and signal completion (remote outbox)
    transport->put(localBuf, remoteBuf, nbytes, dataSignalId, dataSignalVal);
    transport->flush();
  }
}

void testWaitReadyThenPutAndSignal(
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
  waitReadyThenPutAndSignalKernel<<<numBlocks, blockSize>>>(
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
// Kernel: Bidirectional - thread 0 does put+signal, thread 1 does wait
// =============================================================================

__global__ void bidirectionalPutAndWaitKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int sendSignalId,
    uint64_t sendSignalVal,
    int recvSignalId,
    uint64_t recvSignalVal) {
  auto group = make_block_group();
  if (group.group_id == 0) {
    if (group.is_leader()) {
      // Send data to peer (remote outbox)
      transport->put(localBuf, remoteBuf, nbytes, sendSignalId, sendSignalVal);
      transport->flush();
    } else if (group.thread_id_in_group == 1) {
      // Wait for data from peer (local inbox)
      transport->wait_signal(recvSignalId, recvSignalVal);
    }
  }
}

void testBidirectionalPutAndWait(
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
  bidirectionalPutAndWaitKernel<<<numBlocks, blockSize>>>(
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
// Kernel: All-to-all send phase - partition groups by peer
// =============================================================================

__global__ void allToAllSendKernel(
    P2pIbgdaTransportDevice** peerTransports,
    IbgdaLocalBuffer* localSendBufs,
    IbgdaRemoteBuffer* peerRecvBufs,
    int myRank,
    std::size_t nbytes,
    int numPeers) {
  auto group = make_block_group();
  auto [peerId, perPeerGroup] = group.partition(numPeers);

  P2pIbgdaTransportDevice* transport = peerTransports[peerId];

  if (perPeerGroup.is_leader()) {
    // Send data to this peer with signal (slot 0)
    transport->put(
        localSendBufs[peerId],
        peerRecvBufs[peerId],
        nbytes,
        0, // signalId
        1);
    transport->flush();
  }
}

__global__ void allToAllWaitKernel(
    P2pIbgdaTransportDevice** peerTransports,
    int numPeers) {
  auto group = make_block_group();
  auto [peerId, perPeerGroup] = group.partition(numPeers);

  if (perPeerGroup.is_leader()) {
    // Wait for signal from this peer (local inbox, slot 0)
    peerTransports[peerId]->wait_signal(0, 1);
  }
}

void testAllToAll(
    P2pIbgdaTransportDevice** peerTransports,
    IbgdaLocalBuffer* localSendBufs,
    IbgdaRemoteBuffer* peerRecvBufs,
    int myRank,
    std::size_t nbytes,
    int numPeers,
    int numBlocks,
    int blockSize) {
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
    int numPeers,
    int numBlocks,
    int blockSize) {
  allToAllWaitKernel<<<numBlocks, blockSize>>>(peerTransports, numPeers);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Put data + signal remote + counter via companion QP
// =============================================================================

__global__ void putSignalCounterKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int counterId,
    uint64_t counterVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport->put(
        localDataBuf,
        remoteDataBuf,
        nbytes,
        signalId,
        signalVal,
        counterId,
        counterVal);
    transport->wait_counter(counterId, counterVal);
  }
}

void testPutSignalCounter(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localDataBuf,
    const IbgdaRemoteBuffer& remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int counterId,
    uint64_t counterVal,
    int numBlocks,
    int blockSize) {
  putSignalCounterKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      signalId,
      signalVal,
      counterId,
      counterVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Wait for local counter to reach expected value (volatile spin)
// =============================================================================

__global__ void waitCounterKernel(
    P2pIbgdaTransportDevice* transport,
    int counterId,
    uint64_t expectedVal) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    transport->wait_counter(counterId, expectedVal);
  }
}

void testWaitCounter(
    P2pIbgdaTransportDevice* transport,
    int counterId,
    uint64_t expectedVal,
    int numBlocks,
    int blockSize) {
  waitCounterKernel<<<numBlocks, blockSize>>>(
      transport, counterId, expectedVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Multi-QP put + signal (Level 1 — transparent QP selection)
// =============================================================================
//
// Each block puts its chunk of totalBytes using block-scope group put.
// QP selection is handled internally by active_qp() inside the transport —
// no manual blockIdx % numQps needed. This verifies that the Level 1
// multi-QP design works transparently.

__global__ void multiQpPutAndSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t totalBytes,
    int signalId,
    uint64_t signalVal) {
  auto nBlocks = gridDim.x;
  std::size_t chunkSize = totalBytes / nBlocks;
  std::size_t myOffset = blockIdx.x * chunkSize;
  std::size_t myBytes =
      (blockIdx.x == nBlocks - 1) ? (totalBytes - myOffset) : chunkSize;

  IbgdaLocalBuffer myLocalBuf = localBuf.subBuffer(myOffset);
  IbgdaRemoteBuffer myRemoteBuf = remoteBuf.subBuffer(myOffset);

  auto group = make_block_group();

  // QP selection is transparent — transport->active_qp() selects per blockIdx
  transport->put(group, myLocalBuf, myRemoteBuf, myBytes, signalId, signalVal);

  transport->fence(group);
}

void testMultiQpPutAndSignal(
    P2pIbgdaTransportDevice* transport,
    int numQps,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t totalBytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  (void)numQps; // unused with Level 1 — QP selection is internal
  multiQpPutAndSignalKernel<<<numBlocks, blockSize>>>(
      transport, localBuf, remoteBuf, totalBytes, signalId, signalVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

} // namespace comms::pipes::test
