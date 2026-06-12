// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace comms::prims {

class P2pIbgdaTransportDevice;
class P2pIbrcTransportDevice;

enum class P2pIbBackendType : uint8_t {
  IBGDA,
  IBRC,
};

struct P2pIbTransportDevice {
  P2pIbBackendType type{P2pIbBackendType::IBGDA};
  union {
    P2pIbgdaTransportDevice* ibgda;
    P2pIbrcTransportDevice* ibrc;
  };

  IBGDA_HOST_DEVICE P2pIbTransportDevice() : ibgda(nullptr) {}
  IBGDA_HOST_DEVICE explicit P2pIbTransportDevice(P2pIbgdaTransportDevice* p)
      : type(P2pIbBackendType::IBGDA), ibgda(p) {}
  IBGDA_HOST_DEVICE explicit P2pIbTransportDevice(P2pIbrcTransportDevice* p)
      : type(P2pIbBackendType::IBRC), ibrc(p) {}

  IBGDA_HOST_DEVICE P2pIbTransportDevice(const P2pIbTransportDevice&) = default;
  IBGDA_HOST_DEVICE P2pIbTransportDevice& operator=(
      const P2pIbTransportDevice&) = default;

  // Common explicit-buffer IB device API. Backend-owned slot-index helpers and
  // send/recv staging remain backend-specific until IBRC owns matching state.
  __device__ void signal(
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1);

  __device__ void signal(
      ThreadGroup& group,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1);

  __device__ void put(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf = {},
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1);

  __device__ void put(
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf = {},
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1);

  __device__ void put_cooperative(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf = {},
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1);

  __device__ void put_cooperative(
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf = {},
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1);

  __device__ void wait_signal(
      ThreadGroup& group,
      const IbgdaLocalBuffer& signalBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout());

  __device__ void wait_signal(
      const IbgdaLocalBuffer& signalBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout());

  __device__ void wait_counter(
      ThreadGroup& group,
      const IbgdaLocalBuffer& counterBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout());

  __device__ void wait_counter(
      const IbgdaLocalBuffer& counterBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout());

  __device__ void reset_signal(
      ThreadGroup& group,
      const IbgdaLocalBuffer& signalBuf);

  __device__ void reset_signal(const IbgdaLocalBuffer& signalBuf);

  __device__ void reset_counter(
      ThreadGroup& group,
      const IbgdaLocalBuffer& counterBuf);

  __device__ void reset_counter(const IbgdaLocalBuffer& counterBuf);

  __device__ uint64_t read_signal(const IbgdaLocalBuffer& signalBuf) const;

  __device__ uint64_t read_counter(const IbgdaLocalBuffer& counterBuf) const;

  __device__ void flush(ThreadGroup& group);

  __device__ void flush();

  __device__ void fence(ThreadGroup& group);

  __device__ void fence();
};

static_assert(std::is_standard_layout_v<P2pIbTransportDevice>);
static_assert(std::is_trivially_copyable_v<P2pIbTransportDevice>);

} // namespace comms::prims
