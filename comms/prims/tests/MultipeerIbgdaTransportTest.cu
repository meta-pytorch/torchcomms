// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/MultipeerIbgdaTransportTest.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "comms/prims/transport/P2pIbTransportProgressImpl.cuh"

namespace comms::prims::test {

// =============================================================================
// Kernel: Put data + signal remote (adaptive-routing safe, with NIC flush)
// =============================================================================

__global__ void putAndSignalKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport.put(localBuf, remoteBuf, nbytes, signalId, signalVal);
    transport.flush();
  }
}

void testPutAndSignal(
    P2pIbTransportDevice deviceTransportPtr,
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
// Kernel: Explicit cooperative put + signal (warp group)
// =============================================================================

__global__ void putAndSignalGroupKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  auto group = make_warp_group();

  // Explicitly shard the provided buffer across warp lanes.
  transport.put_cooperative(
      group, localBuf, remoteBuf, nbytes, signalId, signalVal);

  transport.flush(group);
}

void testPutAndSignalGroup(
    P2pIbTransportDevice deviceTransportPtr,
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
// Kernel: Multi-warp presharded group put + signal
// Each warp partitions data manually, then calls group-scope put + signal
// =============================================================================

__global__ void putAndSignalGroupMultiWarpKernel(
    P2pIbTransportDevice transport,
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
  transport.put(group, myLocalBuf, myRemoteBuf, myBytes, signalId, signalVal);

  transport.flush(group);
}

void testPutAndSignalGroupMultiWarp(
    P2pIbTransportDevice deviceTransportPtr,
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
// Kernel: Block-scope presharded group put + signal
// Each block partitions data manually, then calls group-scope put + signal
// =============================================================================

__global__ void putAndSignalGroupBlockKernel(
    P2pIbTransportDevice transport,
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
  transport.put(group, myLocalBuf, myRemoteBuf, myBytes, signalId, signalVal);

  transport.flush(group);
}

void testPutAndSignalGroupBlock(
    P2pIbTransportDevice deviceTransportPtr,
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
// Kernel: Wait for signal (acquire polling on local signal buffer)
// =============================================================================

__global__ void waitSignalKernel(
    P2pIbTransportDevice transport,
    int signalId,
    uint64_t expectedSignal) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    transport.wait_signal(signalId, expectedSignal);
  }
}

void testWaitSignal(
    P2pIbTransportDevice transport,
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
    P2pIbTransportDevice transport,
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

      transport.put(srcBuf, dstBuf, bytesPerPut, signalId, 1);
      transport.flush();
    }
  }
}

void testMultiplePutAndSignal(
    P2pIbTransportDevice deviceTransportPtr,
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
    P2pIbTransportDevice transport,
    int signalId,
    uint64_t signalVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport.signal(signalId, signalVal);
    transport.flush();
  }
}

void testSignalOnly(
    P2pIbTransportDevice deviceTransportPtr,
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
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport.put(localBuf, remoteBuf, nbytes);
    transport.flush();
  }
}

void testPutOnly(
    P2pIbTransportDevice deviceTransportPtr,
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
// Kernel: Pipeline geometry snapshot
// =============================================================================

__global__ void pipelineGeometryKernel(
    P2pIbTransportDevice transport,
    uint64_t* output) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    output[0] = static_cast<uint64_t>(transport.pipeline_depth());
    output[1] = static_cast<uint64_t>(transport.pipeline_window());
    output[2] = static_cast<uint64_t>(transport.pipeline_chunk());
  }
}

void testPipelineGeometry(
    P2pIbTransportDevice transport,
    uint64_t* output,
    int numBlocks,
    int blockSize) {
  pipelineGeometryKernel<<<numBlocks, blockSize>>>(transport, output);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// =============================================================================
// Kernel: Blocking send/recv and resumable progress send/recv
// =============================================================================

bool supportsProgressSendRecv() {
#ifdef __HIP_PLATFORM_AMD__
  return false;
#else
  return true;
#endif
}

__global__ void sendRecvKernel(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send) {
  auto group = make_block_group();
  Timeout timeout(kDefaultDeviceTimeoutCycles);
  timeout.start();
  if (send) {
    transport->send(group, buffer, nbytes, maxSignalBytes, timeout);
  } else {
    transport->recv(group, buffer, nbytes, maxSignalBytes, timeout);
  }
}

__global__ void twoCallSendThenRecvKernel(
    P2pIbTransportDevice transport,
    const void* sendBuffer,
    void* recvBuffer,
    std::size_t firstBytes,
    std::size_t secondBytes,
    std::size_t maxSignalBytes) {
  auto group = make_block_group();
  Timeout timeout(kDefaultDeviceTimeoutCycles);
  timeout.start();
  auto* sendBytes = static_cast<const char*>(sendBuffer);
  auto* recvBytes = static_cast<char*>(recvBuffer);

  transport.send(group, sendBytes, firstBytes, maxSignalBytes, timeout);
  transport.recv(group, recvBytes, firstBytes, maxSignalBytes, timeout);
  transport.send(
      group, sendBytes + firstBytes, secondBytes, maxSignalBytes, timeout);
  transport.recv(
      group, recvBytes + firstBytes, secondBytes, maxSignalBytes, timeout);
}

void testSendRecv(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize) {
  sendRecvKernel<<<numBlocks, blockSize>>>(
      transport, buffer, nbytes, maxSignalBytes, send);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

void testTwoCallSendThenRecv(
    P2pIbTransportDevice transport,
    const void* sendBuffer,
    void* recvBuffer,
    std::size_t firstBytes,
    std::size_t secondBytes,
    std::size_t maxSignalBytes,
    int numBlocks,
    int blockSize) {
  twoCallSendThenRecvKernel<<<numBlocks, blockSize>>>(
      transport,
      sendBuffer,
      recvBuffer,
      firstBytes,
      secondBytes,
      maxSignalBytes);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

#ifndef __HIP_PLATFORM_AMD__
__global__ void progressSendRecvKernel(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send,
    uint64_t* waitingCount) {
  auto group = make_block_group();
  Timeout timeout(kDefaultDeviceTimeoutCycles);
  timeout.start();
  uint64_t waits = 0;
  if (send) {
    transport->init_send_progress(group, nbytes, maxSignalBytes);
    IbgdaSendRecvProgressStatus status;
    do {
      status = transport->progress_send_once(
          group, buffer, nbytes, maxSignalBytes, timeout);
      waits += status == IbgdaSendRecvProgressStatus::Waiting;
    } while (status != IbgdaSendRecvProgressStatus::Done);
  } else {
    transport->init_recv_progress(group, nbytes, maxSignalBytes);
    IbgdaSendRecvProgressStatus status;
    do {
      status = transport->progress_recv_once(
          group, buffer, nbytes, maxSignalBytes, timeout);
      waits += status == IbgdaSendRecvProgressStatus::Waiting;
    } while (status != IbgdaSendRecvProgressStatus::Done);
  }
  if (waitingCount != nullptr && group.is_leader()) {
    waitingCount[group.group_id] = waits;
  }
}

__global__ void progressReservationKernel(
    P2pIbgdaTransportDevice* transport,
    int64_t* output,
    std::size_t sendBytes,
    std::size_t recvBytes) {
  auto group = make_block_group();
  transport->init_send_progress(group, sendBytes);
  transport->init_recv_progress(group, recvBytes);

  if (group.is_leader()) {
    const auto& protoSlot =
        transport->local_channel_slot<protocol::Simple>(group.group_id);
    output[0] = protoSlot.sendProgress.nextStep;
    output[1] = protoSlot.recvProgress.nextStep;
  }
}

template <typename Transport>
__device__ IbgdaRegisteredSendProgressStatus postRegisteredSend(
    Transport& transport,
    ThreadGroup& group,
    const IbgdaLocalBuffer& source,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    const Timeout& timeout,
    RegisteredSendObservation* observation) {
  transport.init_registered_send_progress(group, nbytes, maxSignalBytes);
  IbgdaRegisteredSendProgressStatus status;
  do {
    status = transport.progress_registered_send_once(
        group, source, nbytes, maxSignalBytes, timeout);
    if (group.is_leader() && observation != nullptr) {
      observation->record(status);
    }
  } while (status != IbgdaRegisteredSendProgressStatus::Posted &&
           status != IbgdaRegisteredSendProgressStatus::Drained);
  return status;
}

template <typename Transport>
__device__ void drainRegisteredSends(
    Transport& transport,
    ThreadGroup& group,
    const Timeout& timeout,
    RegisteredSendObservation* observation) {
  IbgdaRegisteredSendProgressStatus status;
  do {
    status = transport.progress_registered_send_drain_once(group, timeout);
    if (group.is_leader() && observation != nullptr) {
      observation->record(status);
    }
  } while (status != IbgdaRegisteredSendProgressStatus::Drained);
}

__global__ void registeredSendRecvKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer source,
    void* recvBuffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send,
    RegisteredSendObservation* observation,
    bool blocking,
    bool overwriteAfterDrain,
    uint8_t overwriteValue,
    bool zeroByteAfterPosted) {
  auto group = make_block_group();
  Timeout timeout(kDefaultDeviceTimeoutCycles);
  timeout.start();
  if (send) {
    if (blocking) {
      transport.send_registered(group, source, nbytes, maxSignalBytes, timeout);
      if (group.is_leader() && observation != nullptr) {
        ++observation->drainedCount;
      }
    } else {
      const auto status = postRegisteredSend(
          transport,
          group,
          source,
          nbytes,
          maxSignalBytes,
          timeout,
          observation);
      if (zeroByteAfterPosted) {
        transport.init_registered_send_progress(group, 0, maxSignalBytes);
        const auto zeroByteStatus = transport.progress_registered_send_once(
            group, IbgdaLocalBuffer{}, 0, maxSignalBytes, timeout);
        if (group.is_leader() && observation != nullptr) {
          observation->record(zeroByteStatus);
        }
      }
      if (status != IbgdaRegisteredSendProgressStatus::Drained) {
        drainRegisteredSends(transport, group, timeout, observation);
      }
    }
    if (overwriteAfterDrain) {
      auto* bytes = static_cast<uint8_t*>(source.ptr);
      for (std::size_t i = group.thread_id_in_group; i < nbytes;
           i += group.group_size) {
        bytes[i] = overwriteValue;
      }
      group.sync();
    }
  } else {
    transport.recv(group, recvBuffer, nbytes, maxSignalBytes, timeout);
  }
}

__global__ void mixedRegisteredAndStagedSendRecvKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer sendBuffer,
    void* recvBuffer,
    std::size_t firstBytes,
    std::size_t secondBytes,
    std::size_t thirdBytes,
    std::size_t maxSignalBytes,
    bool send) {
  auto group = make_block_group();
  Timeout timeout(kDefaultDeviceTimeoutCycles);
  timeout.start();
  if (send) {
    (void)postRegisteredSend(
        *transport,
        group,
        sendBuffer,
        firstBytes,
        maxSignalBytes,
        timeout,
        nullptr);
    transport->send(
        group,
        static_cast<const char*>(sendBuffer.ptr) + firstBytes,
        secondBytes,
        maxSignalBytes,
        timeout);
    (void)postRegisteredSend(
        *transport,
        group,
        sendBuffer.subBuffer(firstBytes + secondBytes),
        thirdBytes,
        maxSignalBytes,
        timeout,
        nullptr);
    drainRegisteredSends(*transport, group, timeout, nullptr);
    return;
  }

  auto* output = static_cast<char*>(recvBuffer);
  transport->recv(group, output, firstBytes, maxSignalBytes, timeout);
  transport->recv(
      group, output + firstBytes, secondBytes, maxSignalBytes, timeout);
  transport->recv(
      group,
      output + firstBytes + secondBytes,
      thirdBytes,
      maxSignalBytes,
      timeout);
}

__global__ void fillTransportStagingKernel(
    P2pIbgdaTransportDevice* transport,
    bool sendStaging,
    std::size_t offset,
    std::size_t nbytes,
    uint8_t value) {
  auto group = make_block_group();
  auto& layout = transport->channel_layout();
  char* staging = sendStaging ? layout.sendStagingPtr : layout.recvStagingPtr;
  staging +=
      static_cast<std::size_t>(group.group_id) * layout.perChannelBufferSize +
      offset;
  for (std::size_t i = group.thread_id_in_group; i < nbytes;
       i += group.group_size) {
    staging[i] = static_cast<char>(value);
  }
}

__global__ void verifyTransportStagingKernel(
    P2pIbgdaTransportDevice* transport,
    bool sendStaging,
    std::size_t offset,
    std::size_t nbytes,
    uint8_t expected,
    int* errorCount) {
  auto group = make_block_group();
  const auto& layout = transport->channel_layout();
  const char* staging =
      sendStaging ? layout.sendStagingPtr : layout.recvStagingPtr;
  staging +=
      static_cast<std::size_t>(group.group_id) * layout.perChannelBufferSize +
      offset;
  for (std::size_t i = group.thread_id_in_group; i < nbytes;
       i += group.group_size) {
    if (static_cast<uint8_t>(staging[i]) != expected) {
      atomicAdd(errorCount, 1);
    }
  }
}

#endif

void testProgressSendRecv(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize,
    uint64_t* waitingCount) {
#ifdef __HIP_PLATFORM_AMD__
  (void)transport;
  (void)buffer;
  (void)nbytes;
  (void)maxSignalBytes;
  (void)send;
  (void)numBlocks;
  (void)blockSize;
  (void)waitingCount;
  throw std::runtime_error("progress send/recv is NVIDIA-only");
#else
  progressSendRecvKernel<<<numBlocks, blockSize>>>(
      transport, buffer, nbytes, maxSignalBytes, send, waitingCount);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
  // The caller synchronizes immediately after this helper so runtime kernel
  // failures are reported through the test's CUDA check rather than as skips.
#endif
}

void testProgressReservations(
    P2pIbgdaTransportDevice* transport,
    int64_t* output,
    std::size_t sendBytes,
    std::size_t recvBytes,
    int numBlocks,
    int blockSize) {
#ifdef __HIP_PLATFORM_AMD__
  (void)transport;
  (void)output;
  (void)sendBytes;
  (void)recvBytes;
  (void)numBlocks;
  (void)blockSize;
  throw std::runtime_error("progress send/recv is NVIDIA-only");
#else
  progressReservationKernel<<<numBlocks, blockSize>>>(
      transport, output, sendBytes, recvBytes);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
#endif
}

void testRegisteredSendRecv(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& source,
    void* recvBuffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize,
    RegisteredSendObservation* observation,
    bool blocking,
    bool overwriteAfterDrain,
    uint8_t overwriteValue,
    bool zeroByteAfterPosted) {
#ifdef __HIP_PLATFORM_AMD__
  (void)transport;
  (void)source;
  (void)recvBuffer;
  (void)nbytes;
  (void)maxSignalBytes;
  (void)send;
  (void)numBlocks;
  (void)blockSize;
  (void)observation;
  (void)blocking;
  (void)overwriteAfterDrain;
  (void)overwriteValue;
  (void)zeroByteAfterPosted;
  throw std::runtime_error("registered-source send is NVIDIA-only");
#else
  P2pIbTransportDevice unifiedTransport(transport);
  registeredSendRecvKernel<<<numBlocks, blockSize>>>(
      unifiedTransport,
      source,
      recvBuffer,
      nbytes,
      maxSignalBytes,
      send,
      observation,
      blocking,
      overwriteAfterDrain,
      overwriteValue,
      zeroByteAfterPosted);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
#endif
}

void testMixedRegisteredAndStagedSendRecv(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& sendBuffer,
    void* recvBuffer,
    std::size_t firstBytes,
    std::size_t secondBytes,
    std::size_t thirdBytes,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize) {
#ifdef __HIP_PLATFORM_AMD__
  (void)transport;
  (void)sendBuffer;
  (void)recvBuffer;
  (void)firstBytes;
  (void)secondBytes;
  (void)thirdBytes;
  (void)maxSignalBytes;
  (void)send;
  (void)numBlocks;
  (void)blockSize;
  throw std::runtime_error("registered-source send is NVIDIA-only");
#else
  mixedRegisteredAndStagedSendRecvKernel<<<numBlocks, blockSize>>>(
      transport,
      sendBuffer,
      recvBuffer,
      firstBytes,
      secondBytes,
      thirdBytes,
      maxSignalBytes,
      send);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
#endif
}

void testFillTransportStaging(
    P2pIbgdaTransportDevice* transport,
    bool sendStaging,
    std::size_t offset,
    std::size_t nbytes,
    uint8_t value,
    int numBlocks,
    int blockSize) {
#ifdef __HIP_PLATFORM_AMD__
  (void)transport;
  (void)sendStaging;
  (void)offset;
  (void)nbytes;
  (void)value;
  (void)numBlocks;
  (void)blockSize;
  throw std::runtime_error("registered-source send is NVIDIA-only");
#else
  fillTransportStagingKernel<<<numBlocks, blockSize>>>(
      transport, sendStaging, offset, nbytes, value);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
#endif
}

void testVerifyTransportStaging(
    P2pIbgdaTransportDevice* transport,
    bool sendStaging,
    std::size_t offset,
    std::size_t nbytes,
    uint8_t expected,
    int* errorCount,
    int numBlocks,
    int blockSize) {
#ifdef __HIP_PLATFORM_AMD__
  (void)transport;
  (void)sendStaging;
  (void)offset;
  (void)nbytes;
  (void)expected;
  (void)errorCount;
  (void)numBlocks;
  (void)blockSize;
  throw std::runtime_error("registered-source send is NVIDIA-only");
#else
  verifyTransportStagingKernel<<<numBlocks, blockSize>>>(
      transport, sendStaging, offset, nbytes, expected, errorCount);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
#endif
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
    P2pIbTransportDevice transport,
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
    transport.wait_signal(readySignalId, readySignalVal);

    // Now put data and signal completion (remote outbox)
    transport.put(localBuf, remoteBuf, nbytes, dataSignalId, dataSignalVal);
    transport.flush();
  }
}

void testWaitReadyThenPutAndSignal(
    P2pIbTransportDevice deviceTransportPtr,
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
    P2pIbTransportDevice transport,
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
      transport.put(localBuf, remoteBuf, nbytes, sendSignalId, sendSignalVal);
      transport.flush();
    } else if (group.thread_id_in_group == 1) {
      // Wait for data from peer (local inbox)
      transport.wait_signal(recvSignalId, recvSignalVal);
    }
  }
}

void testBidirectionalPutAndWait(
    P2pIbTransportDevice deviceTransportPtr,
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
    P2pIbTransportDevice* peerTransports,
    IbgdaLocalBuffer* localSendBufs,
    IbgdaRemoteBuffer* peerRecvBufs,
    int myRank,
    std::size_t nbytes,
    int numPeers) {
  auto group = make_block_group();
  auto [peerId, perPeerGroup] = group.partition(numPeers);

  P2pIbTransportDevice transport = peerTransports[peerId];

  if (perPeerGroup.is_leader()) {
    // Send data to this peer with signal (slot 0)
    transport.put(
        localSendBufs[peerId],
        peerRecvBufs[peerId],
        nbytes,
        0, // signalId
        1);
    transport.flush();
  }
}

__global__ void allToAllWaitKernel(
    P2pIbTransportDevice* peerTransports,
    int numPeers) {
  auto group = make_block_group();
  auto [peerId, perPeerGroup] = group.partition(numPeers);

  if (perPeerGroup.is_leader()) {
    // Wait for signal from this peer (local inbox, slot 0)
    peerTransports[peerId].wait_signal(0, 1);
  }
}

void testAllToAll(
    P2pIbTransportDevice* peerTransports,
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
    P2pIbTransportDevice* peerTransports,
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
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int counterId,
    uint64_t counterVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport.put(
        localDataBuf,
        remoteDataBuf,
        nbytes,
        signalId,
        signalVal,
        counterId,
        counterVal);
    transport.wait_counter(counterId, counterVal);
  }
}

void testPutSignalCounter(
    P2pIbTransportDevice deviceTransportPtr,
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
// Kernel: Wait for local counter to reach expected value (acquire polling)
// =============================================================================

__global__ void waitCounterKernel(
    P2pIbTransportDevice transport,
    int counterId,
    uint64_t expectedVal) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    transport.wait_counter(counterId, expectedVal);
  }
}

void testWaitCounter(
    P2pIbTransportDevice transport,
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
    P2pIbTransportDevice transport,
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

  // QP selection is transparent — transport.active_qp() selects per blockIdx
  transport.put(group, myLocalBuf, myRemoteBuf, myBytes, signalId, signalVal);

  transport.flush(group);
}

void testMultiQpPutAndSignal(
    P2pIbTransportDevice transport,
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

} // namespace comms::prims::test
