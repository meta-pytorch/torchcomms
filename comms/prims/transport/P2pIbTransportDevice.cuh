// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"

#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"
#include "comms/prims/transport/ibrc/P2pIbrcTransportDevice.cuh"

namespace comms::prims {

__device__ __forceinline__ void P2pIbTransportDevice::signal(
    int signalId,
    uint64_t signalVal) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->signal(signalId, signalVal);
  } else {
    ibgda->signal(signalId, signalVal);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::signal(
    ThreadGroup& group,
    int signalId,
    uint64_t signalVal) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->signal(group, signalId, signalVal);
  } else {
    ibgda->signal(group, signalId, signalVal);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::put(
    ThreadGroup& group,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int counterId,
    uint64_t counterVal) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->put(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        signalId,
        signalVal,
        counterId,
        counterVal);
  } else {
    ibgda->put(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        signalId,
        signalVal,
        counterId,
        counterVal);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::put(
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int counterId,
    uint64_t counterVal) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->put(
        localBuf,
        remoteBuf,
        nbytes,
        signalId,
        signalVal,
        counterId,
        counterVal);
  } else {
    ibgda->put(
        localBuf,
        remoteBuf,
        nbytes,
        signalId,
        signalVal,
        counterId,
        counterVal);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::put_cooperative(
    ThreadGroup& group,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int counterId,
    uint64_t counterVal) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->put_cooperative(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        signalId,
        signalVal,
        counterId,
        counterVal);
  } else {
    ibgda->put_cooperative(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        signalId,
        signalVal,
        counterId,
        counterVal);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::wait_signal(
    ThreadGroup& group,
    int signalId,
    uint64_t expected,
    const Timeout& timeout) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->wait_signal(group, signalId, expected, timeout);
  } else {
    ibgda->wait_signal(group, signalId, expected, timeout);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::wait_signal(
    int signalId,
    uint64_t expected,
    const Timeout& timeout) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->wait_signal(signalId, expected, timeout);
  } else {
    ibgda->wait_signal(signalId, expected, timeout);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::wait_counter(
    ThreadGroup& group,
    int counterId,
    uint64_t expected,
    const Timeout& timeout) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->wait_counter(group, counterId, expected, timeout);
  } else {
    ibgda->wait_counter(group, counterId, expected, timeout);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::wait_counter(
    int counterId,
    uint64_t expected,
    const Timeout& timeout) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->wait_counter(counterId, expected, timeout);
  } else {
    ibgda->wait_counter(counterId, expected, timeout);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::reset_signal(
    ThreadGroup& group,
    int signalId) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->reset_signal(group, signalId);
  } else {
    ibgda->reset_signal(group, signalId);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::reset_signal(
    int signalId) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->reset_signal(signalId);
  } else {
    ibgda->reset_signal(signalId);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::reset_counter(
    ThreadGroup& group,
    int counterId) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->reset_counter(group, counterId);
  } else {
    ibgda->reset_counter(group, counterId);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::reset_counter(
    int counterId) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->reset_counter(counterId);
  } else {
    ibgda->reset_counter(counterId);
  }
}

__device__ __forceinline__ uint64_t
P2pIbTransportDevice::read_signal(int signalId) const {
  if (type == P2pIbBackendType::IBRC) {
    return ibrc->read_signal(signalId);
  }
  return ibgda->read_signal(signalId);
}

__device__ __forceinline__ uint64_t
P2pIbTransportDevice::read_counter(int counterId) const {
  if (type == P2pIbBackendType::IBRC) {
    return ibrc->read_counter(counterId);
  }
  return ibgda->read_counter(counterId);
}

__device__ __forceinline__ void P2pIbTransportDevice::signal(
    const IbgdaRemoteBuffer& signalBuf,
    uint64_t signalVal) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->signal(signalBuf, signalVal);
  } else {
    ibgda->signal(signalBuf, signalVal);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::signal(
    ThreadGroup& group,
    const IbgdaRemoteBuffer& signalBuf,
    uint64_t signalVal) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->signal(group, signalBuf, signalVal);
  } else {
    ibgda->signal(group, signalBuf, signalVal);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::put(
    ThreadGroup& group,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& signalBuf,
    uint64_t signalVal,
    const IbgdaLocalBuffer& counterBuf,
    uint64_t counterVal) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->put(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        signalBuf,
        signalVal,
        counterBuf,
        counterVal);
  } else {
    ibgda->put(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        signalBuf,
        signalVal,
        counterBuf,
        counterVal);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::put(
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& signalBuf,
    uint64_t signalVal,
    const IbgdaLocalBuffer& counterBuf,
    uint64_t counterVal) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->put(
        localBuf,
        remoteBuf,
        nbytes,
        signalBuf,
        signalVal,
        counterBuf,
        counterVal);
  } else {
    ibgda->put(
        localBuf,
        remoteBuf,
        nbytes,
        signalBuf,
        signalVal,
        counterBuf,
        counterVal);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::put_cooperative(
    ThreadGroup& group,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& signalBuf,
    uint64_t signalVal,
    const IbgdaLocalBuffer& counterBuf,
    uint64_t counterVal) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->put_cooperative(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        signalBuf,
        signalVal,
        counterBuf,
        counterVal);
  } else {
    ibgda->put_cooperative(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        signalBuf,
        signalVal,
        counterBuf,
        counterVal);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::put_cooperative(
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& signalBuf,
    uint64_t signalVal,
    const IbgdaLocalBuffer& counterBuf,
    uint64_t counterVal) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  ThreadGroup solo{0, 1, 0, blockIdx.x, 1, SyncScope::THREAD};
#else
  ThreadGroup solo{0, 1, 0, 0, 1, SyncScope::THREAD};
#endif
  put_cooperative(
      solo,
      localBuf,
      remoteBuf,
      nbytes,
      signalBuf,
      signalVal,
      counterBuf,
      counterVal);
}

__device__ __forceinline__ void P2pIbTransportDevice::wait_signal(
    ThreadGroup& group,
    const IbgdaLocalBuffer& signalBuf,
    uint64_t expected,
    const Timeout& timeout) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->wait_signal(group, signalBuf, expected, timeout);
  } else {
    ibgda->wait_signal(group, signalBuf, expected, timeout);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::wait_signal(
    const IbgdaLocalBuffer& signalBuf,
    uint64_t expected,
    const Timeout& timeout) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->wait_signal(signalBuf, expected, timeout);
  } else {
    ibgda->wait_signal(signalBuf, expected, timeout);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::wait_counter(
    ThreadGroup& group,
    const IbgdaLocalBuffer& counterBuf,
    uint64_t expected,
    const Timeout& timeout) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->wait_counter(group, counterBuf, expected, timeout);
  } else {
    ibgda->wait_counter(group, counterBuf, expected, timeout);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::wait_counter(
    const IbgdaLocalBuffer& counterBuf,
    uint64_t expected,
    const Timeout& timeout) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->wait_counter(counterBuf, expected, timeout);
  } else {
    ibgda->wait_counter(counterBuf, expected, timeout);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::reset_signal(
    ThreadGroup& group,
    const IbgdaLocalBuffer& signalBuf) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->reset_signal(group, signalBuf);
  } else {
    ibgda->reset_signal(group, signalBuf);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::reset_signal(
    const IbgdaLocalBuffer& signalBuf) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->reset_signal(signalBuf);
  } else {
    ibgda->reset_signal(signalBuf);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::reset_counter(
    ThreadGroup& group,
    const IbgdaLocalBuffer& counterBuf) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->reset_counter(group, counterBuf);
  } else {
    ibgda->reset_counter(group, counterBuf);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::reset_counter(
    const IbgdaLocalBuffer& counterBuf) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->reset_counter(counterBuf);
  } else {
    ibgda->reset_counter(counterBuf);
  }
}

__device__ __forceinline__ uint64_t
P2pIbTransportDevice::read_signal(const IbgdaLocalBuffer& signalBuf) const {
  if (type == P2pIbBackendType::IBRC) {
    return ibrc->read_signal(signalBuf);
  }
  return ibgda->read_signal(signalBuf);
}

__device__ __forceinline__ uint64_t
P2pIbTransportDevice::read_counter(const IbgdaLocalBuffer& counterBuf) const {
  if (type == P2pIbBackendType::IBRC) {
    return ibrc->read_counter(counterBuf);
  }
  return ibgda->read_counter(counterBuf);
}

__device__ __forceinline__ void P2pIbTransportDevice::flush(
    ThreadGroup& group) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->flush(group);
  } else {
    ibgda->flush(group);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::flush() {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->flush();
  } else {
    ibgda->flush();
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::fence(
    ThreadGroup& group) {
  flush(group);
}

__device__ __forceinline__ void P2pIbTransportDevice::fence() {
  flush();
}

// ===========================================================================
// Pipelined send/recv — forwarded to the active backend.
// ===========================================================================

template <typename CopyOp, typename... Args>
__device__ __forceinline__ void P2pIbTransportDevice::send(
    ThreadGroup& group,
    const void* __restrict__ src,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    Args... args) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->send<CopyOp>(group, src, nbytes, max_signal_bytes, timeout, args...);
  } else {
    ibgda->send<CopyOp>(group, src, nbytes, max_signal_bytes, timeout, args...);
  }
}

template <typename CopyOp, typename... Args>
__device__ __forceinline__ void P2pIbTransportDevice::recv(
    ThreadGroup& group,
    void* __restrict__ dst,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    Args... args) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->recv<CopyOp>(group, dst, nbytes, max_signal_bytes, timeout, args...);
  } else {
    ibgda->recv<CopyOp>(group, dst, nbytes, max_signal_bytes, timeout, args...);
  }
}

template <typename CopyOp, typename... Args>
__device__ __forceinline__ void P2pIbTransportDevice::forward(
    ThreadGroup& group,
    void* __restrict__ dst,
    P2pIbTransportDevice& fwd,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    Args... args) {
  if (type == P2pIbBackendType::IBRC && fwd.type == P2pIbBackendType::IBRC) {
    ibrc->forward<CopyOp>(
        group, dst, *fwd.ibrc, nbytes, max_signal_bytes, timeout, args...);
    return;
  }
  if (type == P2pIbBackendType::IBGDA && fwd.type == P2pIbBackendType::IBGDA) {
    ibgda->forward<CopyOp>(
        group, dst, *fwd.ibgda, nbytes, max_signal_bytes, timeout, args...);
    return;
  }
  if (group.is_leader()) {
    printf("[PIPES] FATAL: mixed IB backend forward is unsupported\n");
  }
  PIPES_DEVICE_TRAP();
}

__device__ __forceinline__ std::size_t P2pIbTransportDevice::pipeline_window()
    const {
  if (type == P2pIbBackendType::IBRC) {
    return ibrc->pipeline_window();
  }
  return ibgda->pipeline_window();
}

__device__ __forceinline__ std::size_t
P2pIbTransportDevice::blocking_payload_window() const {
  if (type == P2pIbBackendType::IBRC) {
    return ibrc->blocking_payload_window();
  }
  return ibgda->blocking_payload_window();
}

__device__ __forceinline__ void P2pIbTransportDevice::init_send_progress(
    ThreadGroup& group,
    std::size_t nbytes,
    std::size_t max_signal_bytes) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->init_send_progress(group, nbytes, max_signal_bytes);
  } else {
    ibgda->init_send_progress(group, nbytes, max_signal_bytes);
  }
}

__device__ __forceinline__ void P2pIbTransportDevice::init_recv_progress(
    ThreadGroup& group,
    std::size_t nbytes,
    std::size_t max_signal_bytes) {
  if (type == P2pIbBackendType::IBRC) {
    ibrc->init_recv_progress(group, nbytes, max_signal_bytes);
  } else {
    ibgda->init_recv_progress(group, nbytes, max_signal_bytes);
  }
}

template <typename CopyOp, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus
P2pIbTransportDevice::progress_send_once(
    ThreadGroup& group,
    const void* __restrict__ src,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    Args... args) {
  if (type == P2pIbBackendType::IBRC) {
    return ibrc->progress_send_once<CopyOp>(
        group, src, nbytes, max_signal_bytes, timeout, args...);
  }
  return ibgda->progress_send_once<CopyOp>(
      group, src, nbytes, max_signal_bytes, timeout, args...);
}

template <typename CopyOp, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus
P2pIbTransportDevice::progress_recv_once(
    ThreadGroup& group,
    void* __restrict__ dst,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    Args... args) {
  if (type == P2pIbBackendType::IBRC) {
    return ibrc->progress_recv_once<CopyOp>(
        group, dst, nbytes, max_signal_bytes, timeout, args...);
  }
  return ibgda->progress_recv_once<CopyOp>(
      group, dst, nbytes, max_signal_bytes, timeout, args...);
}

} // namespace comms::prims
