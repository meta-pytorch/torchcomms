// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "comms/prims/core/DeviceMacros.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace comms::prims {

struct Memcpy;
struct PipesTraceAllReduceContext;
struct PipesTraceProgressState;

class P2pIbgdaTransportDevice;
class P2pIbrcTransportDevice;

enum class P2pIbBackendType : uint8_t {
  IBGDA,
  IBRC,
};

enum class IbgdaSendRecvProgressStatus : uint8_t {
  Waiting,
  Progressed,
  Done,
};

enum class IbgdaRegisteredSendProgressStatus : uint8_t {
  Waiting,
  Progressed,
  Posted,
  Drained,
};

namespace detail {

template <typename Transport>
__device__ __forceinline__ void init_registered_send_progress(
    Transport& transport,
    ThreadGroup& group,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0);

template <typename Transport>
__device__ __forceinline__ IbgdaRegisteredSendProgressStatus
progress_registered_send_once(
    Transport& transport,
    ThreadGroup& group,
    const IbgdaLocalBuffer& src,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout());

template <typename Transport>
__device__ __forceinline__ IbgdaRegisteredSendProgressStatus
progress_registered_send_drain_once(
    Transport& transport,
    ThreadGroup& group,
    const Timeout& timeout = Timeout());

template <typename Transport>
__device__ __forceinline__ void send_registered(
    Transport& transport,
    ThreadGroup& group,
    const IbgdaLocalBuffer& src,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout());

template <typename Transport, typename CopyOp, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus
progress_send_once_with_trace(
    Transport& transport,
    ThreadGroup& group,
    const void* __restrict__ src,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    const PipesTraceAllReduceContext& traceContext,
    PipesTraceProgressState& traceState,
    Args... args);

} // namespace detail

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

  __device__ void signal(int signalId, uint64_t signalVal = 1);

  __device__ void
  signal(ThreadGroup& group, int signalId, uint64_t signalVal = 1);

  __device__ IbLocalCompletionTicket
  put(ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      int signalId = -1,
      uint64_t signalVal = 1,
      int counterId = -1,
      uint64_t counterVal = 1);

  __device__ IbLocalCompletionTicket
  put(const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      int signalId = -1,
      uint64_t signalVal = 1,
      int counterId = -1,
      uint64_t counterVal = 1);

  __device__ void put_cooperative(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      int signalId = -1,
      uint64_t signalVal = 1,
      int counterId = -1,
      uint64_t counterVal = 1);

  __device__ void wait_signal(
      ThreadGroup& group,
      int signalId,
      uint64_t expected,
      const Timeout& timeout = Timeout());

  __device__ void wait_signal(
      int signalId,
      uint64_t expected,
      const Timeout& timeout = Timeout());

  __device__ void wait_counter(
      ThreadGroup& group,
      int counterId,
      uint64_t expected,
      const Timeout& timeout = Timeout());

  __device__ void wait_counter(
      int counterId,
      uint64_t expected,
      const Timeout& timeout = Timeout());

  __device__ void reset_signal(ThreadGroup& group, int signalId);

  __device__ void reset_signal(int signalId);

  __device__ void reset_counter(ThreadGroup& group, int counterId);

  __device__ void reset_counter(int counterId);

  __device__ uint64_t read_signal(int signalId) const;

  __device__ uint64_t read_counter(int counterId) const;

  __device__ void signal(
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1);

  __device__ void signal(
      ThreadGroup& group,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1);

  __device__ IbLocalCompletionTicket
  put(ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1,
      bool signalPerLane = false);

  __device__ IbLocalCompletionTicket
  put(const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1);

  __device__ void put_cooperative(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1);

  __device__ void put_cooperative(
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
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

  __device__ void wait_local(
      ThreadGroup& group,
      const IbLocalCompletionTicket& ticket,
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

  __device__ __forceinline__ void require_ibgda(
      ThreadGroup& group,
      const char* operation) const;

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void send(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args);

  template <typename = void>
  __device__ __forceinline__ void send_registered(
      ThreadGroup& group,
      const IbgdaLocalBuffer& src,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout());

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void recv(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args);

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void forward(
      ThreadGroup& group,
      void* __restrict__ dst,
      P2pIbTransportDevice& fwd,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args);

  __device__ __forceinline__ std::size_t pipeline_window() const;

  __device__ __forceinline__ int pipeline_depth() const;

  __device__ __forceinline__ std::size_t pipeline_chunk() const;

  template <typename = void>
  __device__ __forceinline__ void init_send_progress(
      ThreadGroup& group,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0);

  template <typename = void>
  __device__ __forceinline__ void init_registered_send_progress(
      ThreadGroup& group,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0);

  template <typename = void>
  __device__ __forceinline__ void init_recv_progress(
      ThreadGroup& group,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0);

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus progress_send_once(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args);

  template <typename = void>
  __device__ __forceinline__ IbgdaRegisteredSendProgressStatus
  progress_registered_send_once(
      ThreadGroup& group,
      const IbgdaLocalBuffer& src,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout());

  template <typename = void>
  __device__ __forceinline__ IbgdaRegisteredSendProgressStatus
  progress_registered_send_drain_once(
      ThreadGroup& group,
      const Timeout& timeout = Timeout());

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus
  progress_send_once_with_trace(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      std::size_t max_signal_bytes,
      const Timeout& timeout,
      const PipesTraceAllReduceContext& traceContext,
      PipesTraceProgressState& traceState,
      Args... args);

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus progress_recv_once(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args);

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus
  progress_recv_once_with_trace(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      std::size_t max_signal_bytes,
      const Timeout& timeout,
      const PipesTraceAllReduceContext& traceContext,
      PipesTraceProgressState& traceState,
      Args... args);
};

static_assert(std::is_standard_layout_v<P2pIbTransportDevice>);
static_assert(std::is_trivially_copyable_v<P2pIbTransportDevice>);

} // namespace comms::prims
