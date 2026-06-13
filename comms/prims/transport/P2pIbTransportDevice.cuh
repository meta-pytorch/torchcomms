// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"

#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"
#include "comms/prims/transport/ibrc/P2pIbrcTransportDevice.cuh"

namespace comms::prims {

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
  ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
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

} // namespace comms::prims
