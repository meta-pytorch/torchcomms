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
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto work = transport->put_signal(
        localBuf, remoteBuf, nbytes, remoteSignalBuf, signalId, signalVal);
    transport->wait_local(work);
  }
}

void testPutAndSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  putAndSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr,
      localBuf,
      remoteBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Warp-group-collaborative put + signal
// =============================================================================

__global__ void putAndSignalGroupKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  auto group = make_warp_group();

  transport->put_group_local(group, localBuf, remoteBuf, nbytes);

  auto work = transport->put_signal_group_local(
      group, localBuf, remoteBuf, nbytes, remoteSignalBuf, signalId, signalVal);

  if (group.is_leader()) {
    transport->wait_local(work);
  }
}

void testPutAndSignalGroup(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  putAndSignalGroupKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr,
      localBuf,
      remoteBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Multi-warp group-collaborative put + signal
// =============================================================================

__global__ void putAndSignalGroupMultiWarpKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  auto group = make_warp_group();

  auto work = transport->put_signal_group_global(
      group, localBuf, remoteBuf, nbytes, remoteSignalBuf, signalId, signalVal);

  if (group.is_leader()) {
    transport->wait_local(work);
  }
}

void testPutAndSignalGroupMultiWarp(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  putAndSignalGroupMultiWarpKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr,
      localBuf,
      remoteBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Block-scope group-collaborative put + signal
// Multiple blocks use put_group_global, each leader signals
// =============================================================================

__global__ void putAndSignalGroupBlockKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  auto group = make_block_group();

  auto work = transport->put_signal_group_global(
      group, localBuf, remoteBuf, nbytes, remoteSignalBuf, signalId, signalVal);

  if (group.is_leader()) {
    transport->wait_local(work);
  }
}

void testPutAndSignalGroupBlock(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  putAndSignalGroupBlockKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr,
      localBuf,
      remoteBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal);
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
    volatile uint64_t* localSignalBuf,
    int signalId,
    uint64_t expectedSignal) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    while (localSignalBuf[signalId] < expectedSignal) {
      // spin
    }
  }
}

void testWaitSignal(
    uint64_t* localSignalBuf,
    int signalId,
    uint64_t expectedSignal,
    int numBlocks,
    int blockSize) {
  waitSignalKernel<<<numBlocks, blockSize>>>(
      localSignalBuf, signalId, expectedSignal);
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
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    int numPuts) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    for (int i = 0; i < numPuts; i++) {
      IbgdaLocalBuffer srcBuf = localBuf.subBuffer(i * bytesPerPut);
      IbgdaRemoteBuffer dstBuf = remoteBuf.subBuffer(i * bytesPerPut);

      auto work = transport->put_signal(
          srcBuf, dstBuf, bytesPerPut, remoteSignalBuf, signalId, 1);
      transport->wait_local(work);
    }
  }
}

void testMultiplePutAndSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t bytesPerPut,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numPuts,
    int numBlocks,
    int blockSize) {
  multiplePutAndSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr,
      localBuf,
      remoteBuf,
      bytesPerPut,
      remoteSignalBuf,
      signalId,
      numPuts);
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
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto work = transport->signal_remote(remoteSignalBuf, signalId, signalVal);
    transport->wait_local(work);
  }
}

void testSignalOnly(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  signalOnlyKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, remoteSignalBuf, signalId, signalVal);
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
// Kernel: Reset signal
// =============================================================================

__global__ void resetSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // reset_signal is now synchronous (includes fences and wait internally)
    transport->reset_signal(remoteSignalBuf, signalId);
  }
}

void testResetSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numBlocks,
    int blockSize) {
  resetSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, remoteSignalBuf, signalId);
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
    volatile uint64_t* localSignalBuf,
    int readySignalId,
    uint64_t readySignalVal,
    IbgdaRemoteBuffer remoteSignalBuf,
    int dataSignalId,
    uint64_t dataSignalVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // Wait for receiver to signal that its buffer is ready
    while (localSignalBuf[readySignalId] < readySignalVal) {
      // spin
    }

    // Now put data and signal completion
    auto work = transport->put_signal(
        localBuf,
        remoteBuf,
        nbytes,
        remoteSignalBuf,
        dataSignalId,
        dataSignalVal);
    transport->wait_local(work);
  }
}

void testWaitReadyThenPutAndSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    uint64_t* localSignalBuf,
    int readySignalId,
    uint64_t readySignalVal,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int dataSignalId,
    uint64_t dataSignalVal,
    int numBlocks,
    int blockSize) {
  waitReadyThenPutAndSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr,
      localBuf,
      remoteBuf,
      nbytes,
      localSignalBuf,
      readySignalId,
      readySignalVal,
      remoteSignalBuf,
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
    IbgdaRemoteBuffer remoteSignalBuf,
    int sendSignalId,
    uint64_t sendSignalVal,
    volatile uint64_t* localSignalBuf,
    int recvSignalId,
    uint64_t recvSignalVal) {
  auto group = make_block_group();
  if (group.group_id == 0) {
    if (group.is_leader()) {
      // Send data to peer
      auto work = transport->put_signal(
          localBuf,
          remoteBuf,
          nbytes,
          remoteSignalBuf,
          sendSignalId,
          sendSignalVal);
      transport->wait_local(work);
    } else if (group.thread_id_in_group == 1) {
      // Wait for data from peer
      while (localSignalBuf[recvSignalId] < recvSignalVal) {
        // spin
      }
    }
  }
}

void testBidirectionalPutAndWait(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int sendSignalId,
    uint64_t sendSignalVal,
    uint64_t* localSignalBuf,
    int recvSignalId,
    uint64_t recvSignalVal,
    int numBlocks,
    int blockSize) {
  bidirectionalPutAndWaitKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr,
      localBuf,
      remoteBuf,
      nbytes,
      remoteSignalBuf,
      sendSignalId,
      sendSignalVal,
      localSignalBuf,
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
    IbgdaRemoteBuffer* remoteSignalBufs,
    int myRank,
    std::size_t nbytes,
    int numPeers) {
  auto group = make_block_group();
  auto [peerId, perPeerGroup] = group.partition(numPeers);

  P2pIbgdaTransportDevice* transport = peerTransports[peerId];

  if (perPeerGroup.is_leader()) {
    // Send data to this peer
    auto work = transport->put_signal(
        localSendBufs[peerId],
        peerRecvBufs[peerId],
        nbytes,
        remoteSignalBufs[peerId],
        myRank,
        1);
    transport->wait_local(work);
  }
}

__global__ void allToAllWaitKernel(
    volatile uint64_t* localSignalBuf,
    int* peerRanks,
    int numPeers) {
  auto group = make_block_group();
  auto [peerId, perPeerGroup] = group.partition(numPeers);

  if (perPeerGroup.is_leader()) {
    // Wait for signal from this peer
    // Signal ID = peerRank (sender's rank)
    int peerRank = peerRanks[peerId];
    while (localSignalBuf[peerRank] < 1) {
      // spin
    }
  }
}

void testAllToAll(
    P2pIbgdaTransportDevice** peerTransports,
    IbgdaLocalBuffer* localSendBufs,
    IbgdaRemoteBuffer* peerRecvBufs,
    IbgdaRemoteBuffer* remoteSignalBufs,
    int myRank,
    std::size_t nbytes,
    int numPeers,
    int numBlocks,
    int blockSize) {
  allToAllSendKernel<<<numBlocks, blockSize>>>(
      peerTransports,
      localSendBufs,
      peerRecvBufs,
      remoteSignalBufs,
      myRank,
      nbytes,
      numPeers);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

void testAllToAllWait(
    uint64_t* localSignalBuf,
    int* peerRanks,
    int numPeers,
    int numBlocks,
    int blockSize) {
  allToAllWaitKernel<<<numBlocks, blockSize>>>(
      localSignalBuf, peerRanks, numPeers);
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
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    uint64_t counterVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport->put_signal_counter_remote(
        localDataBuf,
        remoteDataBuf,
        nbytes,
        remoteSignalBuf,
        signalId,
        signalVal,
        localCounterBuf,
        counterId,
        counterVal);
  }
}

void testPutSignalCounter(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localDataBuf,
    const IbgdaRemoteBuffer& remoteDataBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    const IbgdaLocalBuffer& localCounterBuf,
    int counterId,
    uint64_t counterVal,
    int numBlocks,
    int blockSize) {
  putSignalCounterKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal,
      localCounterBuf,
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
    volatile uint64_t* counterBuf,
    int counterId,
    uint64_t expectedVal) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    volatile uint64_t* ctr = counterBuf + counterId;
    while (*ctr < expectedVal) {
    }
  }
}

void testWaitCounter(
    uint64_t* counterBuf,
    int counterId,
    uint64_t expectedVal,
    int numBlocks,
    int blockSize) {
  waitCounterKernel<<<numBlocks, blockSize>>>(
      counterBuf, counterId, expectedVal);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

} // namespace comms::pipes::test
