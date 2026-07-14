// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#ifdef __HIP_PLATFORM_AMD__
#include "HipDeviceCompat.h"
#else
#include <cuda/atomic>
#endif

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "comms/prims/core/DeviceMacros.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibrc/IbrcTypes.h"

namespace comms::prims {

// Default bound for device-side waits on the CPU progress thread (flush /
// reserve). Mirrors IBGDA's kDefaultDeviceTimeoutCycles: converts an indefinite
// hang (a stalled progress thread that never publishes an error) into a bounded
// trap.
inline constexpr uint64_t kIbrcDefaultDeviceTimeoutCycles = 10'000'000'000ULL;

#if PIPES_IS_DEVICE_COMPILE
#define IBRC_CHECK_SLOT_ID(id, count, kind)             \
  do {                                                  \
    if (!((id) >= 0 && (id) < (count))) {               \
      printf(                                           \
          "P2pIbrcTransportDevice: " kind               \
          " id %d out of range [0, %d) at "             \
          "%s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n", \
          (int)(id),                                    \
          (int)(count),                                 \
          __FILE__,                                     \
          __LINE__,                                     \
          blockIdx.x,                                   \
          blockIdx.y,                                   \
          blockIdx.z,                                   \
          threadIdx.x,                                  \
          threadIdx.y,                                  \
          threadIdx.z);                                 \
      PIPES_DEVICE_TRAP();                              \
    }                                                   \
  } while (0)
#else
#define IBRC_CHECK_SLOT_ID(id, count, kind) assert((id) >= 0 && (id) < (count))
#endif

/**
 * Device-side IBRC peer handle.
 *
 * IBRC uses a GPU-visible command queue per peer/QP/NIC. Device code reserves a
 * queue slot, writes an IbrcDesc, then publishes ready_seq with release
 * ordering. The CPU progress thread consumes descriptors, posts the verbs work
 * requests on the matching QP, and advances ci after polling the CQE. Optional
 * local counters are updated by the CPU proxy after polling that CQE.
 */
class P2pIbrcTransportDevice {
 public:
  P2pIbrcTransportDevice() = default;

  __host__ __device__ P2pIbrcTransportDevice(
      DeviceSpan<IbrcCmdQueueDevice> queues,
      uint32_t nics,
      uint32_t maxChannels,
      uint32_t qpsPerConnection,
      DeviceSpan<IbLocalChannel> localChannels,
      IbgdaRemoteBuffer ownedRemoteSignalBuf = {},
      IbgdaLocalBuffer ownedLocalSignalBuf = {},
      IbgdaLocalBuffer ownedCounterDeviceBuf = {},
      IbgdaLocalBuffer ownedCounterHostBuf = {},
      int numSignalSlots = 0,
      int numCounterSlots = 0,
      IbChannelLayout channelLayout = {})
      : cmdQueues(queues),
        numNics(nics),
        maxChannels_(maxChannels),
        qpsPerConnection_(qpsPerConnection),
        localChannels_(localChannels),
        ownedRemoteSignalBuf_(ownedRemoteSignalBuf),
        ownedLocalSignalBuf_(ownedLocalSignalBuf),
        ownedCounterDeviceBuf_(ownedCounterDeviceBuf),
        ownedCounterHostBuf_(ownedCounterHostBuf),
        numSignalSlots_(numSignalSlots),
        numCounterSlots_(numCounterSlots),
        channelLayout_(channelLayout) {}

  __device__ void put(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      int signalId = -1,
      uint64_t signalVal = 1,
      int counterId = -1,
      uint64_t counterVal = 1) {
    IbgdaRemoteBuffer sigSlot =
        (signalId >= 0) ? remote_signal_slot(signalId) : IbgdaRemoteBuffer{};
    IbgdaLocalBuffer ctrSlot =
        (counterId >= 0) ? counter_host_slot(counterId) : IbgdaLocalBuffer{};
    put(group,
        localBuf,
        remoteBuf,
        nbytes,
        sigSlot,
        signalVal,
        ctrSlot,
        counterVal);
  }

  __device__ void put(
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      int signalId = -1,
      uint64_t signalVal = 1,
      int counterId = -1,
      uint64_t counterVal = 1) {
    ThreadGroup solo = make_thread_solo();
    put(solo,
        localBuf,
        remoteBuf,
        nbytes,
        signalId,
        signalVal,
        counterId,
        counterVal);
  }

  __device__ void put_cooperative(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      int signalId = -1,
      uint64_t signalVal = 1,
      int counterId = -1,
      uint64_t counterVal = 1) {
    put(group,
        localBuf,
        remoteBuf,
        nbytes,
        signalId,
        signalVal,
        counterId,
        counterVal);
  }

  __device__ void
  signal(ThreadGroup& group, int signalId, uint64_t signalVal = 1) {
    signal(group, remote_signal_slot(signalId), signalVal);
  }

  __device__ void signal(int signalId, uint64_t signalVal = 1) {
    ThreadGroup solo = make_thread_solo();
    signal(solo, signalId, signalVal);
  }

  __device__ void wait_signal(
      ThreadGroup& group,
      int signalId,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    wait_signal(group, local_signal_slot(signalId), expected, timeout);
  }

  __device__ void wait_signal(
      int signalId,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    ThreadGroup solo = make_thread_solo();
    wait_signal(solo, signalId, expected, timeout);
  }

  __device__ void wait_counter(
      ThreadGroup& group,
      int counterId,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    wait_counter(group, counter_device_slot(counterId), expected, timeout);
  }

  __device__ void wait_counter(
      int counterId,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    ThreadGroup solo = make_thread_solo();
    wait_counter(solo, counterId, expected, timeout);
  }

  __device__ void reset_signal(ThreadGroup& group, int signalId) {
    reset_signal(group, local_signal_slot(signalId));
  }

  __device__ void reset_signal(int signalId) {
    ThreadGroup solo = make_thread_solo();
    reset_signal(solo, signalId);
  }

  __device__ void reset_counter(ThreadGroup& group, int counterId) {
    reset_counter(group, counter_device_slot(counterId));
  }

  __device__ void reset_counter(int counterId) {
    ThreadGroup solo = make_thread_solo();
    reset_counter(solo, counterId);
  }

  __device__ uint64_t read_signal(int signalId) const {
    return read_signal(local_signal_slot(signalId));
  }

  __device__ uint64_t read_counter(int counterId) const {
    return read_counter(counter_device_slot(counterId));
  }

  // Public raw put/signal/flush/fence APIs default to the Send direction.
  // Recv-direction operations are reserved for the send/recv protocol internals
  // that explicitly pass IbDirection::Recv.
  __device__ void signal(
      ThreadGroup& group,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      IbDirection direction = IbDirection::Send) {
    if (group.is_leader()) {
      if (signalBuf.ptr == nullptr) {
        trap("P2pIbrcTransportDevice: signal buffer is null");
      }
      validate_group_scope(group);
      const uint32_t queueId = control_queue_id(group, direction);
      const uint32_t nicId = nic_for_queue(queueId);
      IbrcDesc desc{};
      desc.signal_addr = reinterpret_cast<uint64_t>(signalBuf.ptr);
      desc.signal_value = signalVal;
      desc.signal_rkey_device_order = signalBuf.rkey_per_device[nicId].value;
      desc.op = static_cast<uint16_t>(IbrcOp::SIGNAL);
      desc.flags = IBRC_HAS_SIGNAL | IBRC_SIGNAL_ADD;
      desc.ready_seq = kIbrcInvalidReadySeq;
      enqueue(queueId, desc);
    }
    group.sync();
  }

  __device__ void signal(
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      IbDirection direction = IbDirection::Send) {
    ThreadGroup solo = make_thread_solo();
    signal(solo, signalBuf, signalVal, direction);
  }

  __device__ void put(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1) {
    const bool hasData = nbytes > 0;
    const bool hasSignal = signalBuf.ptr != nullptr;
    const bool hasCounter = counterBuf.ptr != nullptr;
    if (hasData) {
      if (localBuf.ptr == nullptr || remoteBuf.ptr == nullptr) {
        trap("P2pIbrcTransportDevice: put data buffer is null");
      }
      threadfence_system();
    }
    group.sync();

    if (group.is_leader()) {
      validate_group_scope(group);
      const uint32_t queueId = select_put_queue_id(group, IbDirection::Send);
      const uint32_t nicId = nic_for_queue(queueId);
      IbrcDesc desc{};
      desc.op = static_cast<uint16_t>(hasData ? IbrcOp::PUT : IbrcOp::SIGNAL);
      desc.ready_seq = kIbrcInvalidReadySeq;

      if (hasData) {
        desc.local_addr = reinterpret_cast<uint64_t>(localBuf.ptr);
        desc.remote_addr = reinterpret_cast<uint64_t>(remoteBuf.ptr);
        desc.bytes = nbytes;
        desc.lkey_device_order = localBuf.lkey_per_device[nicId].value;
        desc.rkey_device_order = remoteBuf.rkey_per_device[nicId].value;
      }

      if (hasSignal) {
        desc.signal_addr = reinterpret_cast<uint64_t>(signalBuf.ptr);
        desc.signal_value = signalVal;
        desc.signal_rkey_device_order = signalBuf.rkey_per_device[nicId].value;
        desc.flags |= IBRC_HAS_SIGNAL | IBRC_SIGNAL_ADD;
      }

      if (hasCounter) {
        desc.counter_addr = reinterpret_cast<uint64_t>(counterBuf.ptr);
        desc.counter_value = counterVal;
        desc.flags |= IBRC_HAS_COUNTER;
      }

      enqueue(queueId, desc);
    }
    group.sync();
  }

  __device__ void put(
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1) {
    ThreadGroup solo = make_thread_solo();
    put(solo,
        localBuf,
        remoteBuf,
        nbytes,
        signalBuf,
        signalVal,
        counterBuf,
        counterVal);
  }

  __device__ void put_cooperative(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1) {
    put(group,
        localBuf,
        remoteBuf,
        nbytes,
        signalBuf,
        signalVal,
        counterBuf,
        counterVal);
  }

  __device__ void put_cooperative(
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1) {
    ThreadGroup solo = make_thread_solo();
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

  __device__ void wait_signal(
      ThreadGroup& group,
      const IbgdaLocalBuffer& signalBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    wait_local(group, signalBuf.ptr, expected, timeout, "signal");
  }

  __device__ void wait_signal(
      const IbgdaLocalBuffer& signalBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    ThreadGroup solo = make_thread_solo();
    wait_signal(solo, signalBuf, expected, timeout);
  }

  __device__ void reset_signal(
      ThreadGroup& group,
      const IbgdaLocalBuffer& signalBuf) {
    reset_local(group, signalBuf.ptr, "signal");
  }

  __device__ void reset_signal(const IbgdaLocalBuffer& signalBuf) {
    ThreadGroup solo = make_thread_solo();
    reset_signal(solo, signalBuf);
  }

  __device__ void reset_counter(
      ThreadGroup& group,
      const IbgdaLocalBuffer& counterBuf) {
    reset_local(group, counterBuf.ptr, "counter");
  }

  __device__ void reset_counter(const IbgdaLocalBuffer& counterBuf) {
    ThreadGroup solo = make_thread_solo();
    reset_counter(solo, counterBuf);
  }

  __device__ void wait_counter(
      ThreadGroup& group,
      const IbgdaLocalBuffer& counterBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    wait_local(group, counterBuf.ptr, expected, timeout, "counter");
  }

  __device__ void wait_counter(
      const IbgdaLocalBuffer& counterBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    ThreadGroup solo = make_thread_solo();
    wait_counter(solo, counterBuf, expected, timeout);
  }

  __device__ uint64_t read_signal(const IbgdaLocalBuffer& signalBuf) const {
    return load_acquire_system_u64(signalBuf.ptr);
  }

  __device__ uint64_t read_counter(const IbgdaLocalBuffer& counterBuf) const {
    return load_acquire_system_u64(counterBuf.ptr);
  }

  __device__ void flush(
      ThreadGroup& group,
      IbDirection direction = IbDirection::Send) {
    if (group.is_leader()) {
      validate_group_scope(group);
      drain_channel_queues(group, direction);
    }
    group.sync();
  }

  __device__ void flush(IbDirection direction = IbDirection::Send) {
    ThreadGroup solo = make_thread_solo();
    flush(solo, direction);
  }

  __device__ void fence(
      ThreadGroup& group,
      IbDirection direction = IbDirection::Send) {
    flush(group, direction);
  }

  __device__ void fence(IbDirection direction = IbDirection::Send) {
    flush(direction);
  }

  // ===========================================================================
  // Pipelined send/recv — delegated to the shared stateless IbSendRecvOps.
  // ===========================================================================
  //
  // The send/recv algorithm is transport-agnostic and lives in
  // `IbSendRecvOps` (P2pIbTransportDeviceDecl.cuh). The protocol state is
  // owned by this backend device; each method routes every transport op through
  // `*this`, so IBRC reuses IBGDA's send/recv unchanged.

  __device__ __forceinline__ IbLocalChannel& local_channel(uint32_t channelId) {
    validate_channel_id(channelId);
    return localChannels_[channelId];
  }

  __device__ __forceinline__ IbLocalChannel& local_channel(ThreadGroup& group) {
    return local_channel(group.group_id);
  }

  __host__ __device__ IbChannelLayout& channel_layout() {
    return channelLayout_;
  }

  __host__ __device__ const IbChannelLayout& channel_layout() const {
    return channelLayout_;
  }

  __device__ __forceinline__ std::size_t pipeline_window() const {
    return IbSendRecvOps{}.pipeline_window(channelLayout_);
  }

  __device__ __forceinline__ void init_send_progress(
      ThreadGroup& group,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0) {
    IbSendRecvOps{}.init_send_progress(
        *this, channelLayout_, group, nbytes, max_signal_bytes);
  }

  __device__ __forceinline__ void init_recv_progress(
      ThreadGroup& group,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0) {
    IbSendRecvOps{}.init_recv_progress(
        *this, channelLayout_, group, nbytes, max_signal_bytes);
  }

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus progress_send_once(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
    return IbSendRecvOps{}.progress_send_once<P2pIbrcTransportDevice, CopyOp>(
        *this,
        channelLayout_,
        group,
        src,
        nbytes,
        max_signal_bytes,
        timeout,
        args...);
  }

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus progress_recv_once(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
    return IbSendRecvOps{}.progress_recv_once<P2pIbrcTransportDevice, CopyOp>(
        *this,
        channelLayout_,
        group,
        dst,
        nbytes,
        max_signal_bytes,
        timeout,
        args...);
  }

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void send(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
    IbSendRecvOps{}.send<P2pIbrcTransportDevice, CopyOp>(
        *this,
        channelLayout_,
        group,
        src,
        nbytes,
        max_signal_bytes,
        timeout,
        args...);
  }

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void recv(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
    IbSendRecvOps{}.recv<P2pIbrcTransportDevice, CopyOp>(
        *this,
        channelLayout_,
        group,
        dst,
        nbytes,
        max_signal_bytes,
        timeout,
        args...);
  }

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void forward(
      ThreadGroup& group,
      void* __restrict__ dst,
      P2pIbrcTransportDevice& fwd,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
    IbSendRecvOps{}.forward<CopyOp>(
        *this,
        channelLayout_,
        group,
        dst,
        fwd.channelLayout_,
        fwd,
        nbytes,
        max_signal_bytes,
        timeout,
        args...);
  }

 private:
  __device__ __forceinline__ uint32_t num_qp_lanes() const {
    return numNics * qpsPerConnection_;
  }

  __device__ __forceinline__ uint32_t num_qps_per_peer_per_nic() const {
    return maxChannels_ * kIbDirections * qpsPerConnection_;
  }

  __device__ __forceinline__ void validate_channel_id(
      uint32_t channelId) const {
#if PIPES_IS_DEVICE_COMPILE
    if (blockIdx.y != 0 || blockIdx.z != 0 || blockDim.y != 1 ||
        blockDim.z != 1) {
      printf(
          "P2pIbrcTransportDevice: channel QP selection currently "
          "supports only 1D grids and 1D thread blocks, got "
          "blockIdx=(%u,%u,%u) blockDim=(%u,%u,%u)\n",
          blockIdx.x,
          blockIdx.y,
          blockIdx.z,
          blockDim.x,
          blockDim.y,
          blockDim.z);
      PIPES_DEVICE_TRAP();
    }
#endif
    if (cmdQueues.empty()) {
      trap("P2pIbrcTransportDevice: no command queues");
    }
    if (numNics == 0 || qpsPerConnection_ == 0 || maxChannels_ == 0 ||
        localChannels_.empty() || channelId >= maxChannels_) {
      printf(
          "P2pIbrcTransportDevice: invalid channel QP state channel_id=%u "
          "maxChannels=%u qpsPerConnection=%u numNics=%u stateSize=%u\n",
          channelId,
          maxChannels_,
          qpsPerConnection_,
          numNics,
          static_cast<unsigned>(localChannels_.size()));
      PIPES_DEVICE_TRAP();
    }
  }

  __device__ __forceinline__ void validate_group_scope(
      const ThreadGroup& group) const {
    if (group.scope == SyncScope::CLUSTER) {
      trap("P2pIbrcTransportDevice: cluster-scope ThreadGroup unsupported");
    }
    validate_channel_id(group.group_id);
  }

  __device__ __forceinline__ IbQpState& qp_state(
      uint32_t channelId,
      IbDirection direction) const {
    validate_channel_id(channelId);
    auto& channel = localChannels_[channelId];
    return direction == IbDirection::Send ? channel.sendQp : channel.recvQp;
  }

  __device__ __forceinline__ uint32_t queue_for_lane(
      uint32_t channelId,
      IbDirection direction,
      uint32_t laneOrdinal) const {
    validate_channel_id(channelId);
    const uint32_t lanes = num_qp_lanes();
    if (laneOrdinal >= lanes) {
      trap("P2pIbrcTransportDevice: lane ordinal out of range");
    }
    const uint32_t nicId = laneOrdinal % numNics;
    const uint32_t qpIndex = laneOrdinal / numNics;
    const uint32_t directionIndex = static_cast<uint32_t>(direction);
    const uint32_t qpSlot =
        ((channelId * kIbDirections + directionIndex) * qpsPerConnection_) +
        qpIndex;
    if (qpSlot >= num_qps_per_peer_per_nic()) {
      trap("P2pIbrcTransportDevice: QP slot out of range");
    }
    const uint32_t queueId = qpSlot * numNics + nicId;
    if (queueId >= cmdQueues.size()) {
      trap("P2pIbrcTransportDevice: command queue id out of range");
    }
    return queueId;
  }

  __device__ __forceinline__ uint32_t
  control_queue_id(const ThreadGroup& group, IbDirection direction) const {
    return queue_for_lane(group.group_id, direction, 0);
  }

  __device__ __forceinline__ uint32_t
  select_put_queue_id(const ThreadGroup& group, IbDirection direction) {
    const uint32_t channelId = group.group_id;
    validate_channel_id(channelId);
    const uint32_t lanes = num_qp_lanes();
    if (lanes == 1) {
      return control_queue_id(group, direction);
    }
    const uint32_t seq =
        fetch_add_system_u32(&qp_state(channelId, direction).cursor, 1U);
    return queue_for_lane(channelId, direction, seq % lanes);
  }

  __device__ __forceinline__ uint32_t nic_for_queue(uint32_t queueId) const {
    if (numNics == 0) {
      trap("P2pIbrcTransportDevice: no NICs");
    }
    return queueId % numNics;
  }

  __device__ void drain_queue(const IbrcCmdQueueDevice& queue) const {
    const uint64_t target = load_acquire_system_u64(queue.pi);
    Timeout timeout{kIbrcDefaultDeviceTimeoutCycles};
    timeout.start();
    while (load_acquire_system_u64(queue.ci) < target) {
      check_status(queue);
      if (timeout.checkExpired()) {
        printf("P2pIbrcTransportDevice: flush timed out\n");
        PIPES_DEVICE_TRAP();
      }
    }
  }

  __device__ void drain_channel_queues(
      const ThreadGroup& group,
      IbDirection direction) const {
    const uint32_t channelId = group.group_id;
    validate_channel_id(channelId);
    const uint32_t lanes = num_qp_lanes();
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      drain_queue(cmdQueues[queue_for_lane(channelId, direction, lane)]);
    }
  }

  __device__ void check_channel_status(uint32_t channelId) const {
    validate_channel_id(channelId);
    const uint32_t lanes = num_qp_lanes();
    for (uint32_t dir = 0; dir < kIbDirections; ++dir) {
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        check_status(
            cmdQueues[queue_for_lane(
                channelId, static_cast<IbDirection>(dir), lane)]);
      }
    }
  }

  __device__ __forceinline__ uint64_t reserve(IbrcCmdQueueDevice& queue) const {
    const uint64_t seq = fetch_add_system_u64(queue.pi, 1);
    Timeout timeout{kIbrcDefaultDeviceTimeoutCycles};
    timeout.start();
    while (seq - load_acquire_system_u64(queue.ci) >= queue.depth) {
      check_status(queue);
      if (timeout.checkExpired()) {
        printf("P2pIbrcTransportDevice: reserve timed out\n");
        PIPES_DEVICE_TRAP();
      }
    }
    return seq;
  }

  __device__ __forceinline__ void enqueue(
      uint32_t queueId,
      const IbrcDesc& desc) const {
    IbrcCmdQueueDevice& queue = cmdQueues[queueId];
    check_status(queue);
    const uint64_t seq = reserve(queue);
    IbrcDesc& slot = queue.descs[seq & queue.mask];
    slot = desc;
    store_release_system_u64(&slot.ready_seq, seq);
  }

  __device__ __forceinline__ void check_status(
      const IbrcCmdQueueDevice& queue) const {
    if (queue.status == nullptr) {
      return;
    }
    if (load_acquire_system_u32(&queue.status->error) != 0) {
      printf(
          "P2pIbrcTransportDevice: queue error queue=%u code=%u\n",
          load_acquire_system_u32(&queue.status->error_queue),
          load_acquire_system_u32(&queue.status->error_code));
      PIPES_DEVICE_TRAP();
    }
  }

  __device__ void wait_local(
      ThreadGroup& group,
      const void* ptr,
      uint64_t expected,
      const Timeout& timeout,
      const char* kind) const {
    if (ptr == nullptr) {
      trap("P2pIbrcTransportDevice: wait buffer is null");
    }
    if (group.is_leader()) {
      validate_group_scope(group);
      while (load_acquire_system_u64(ptr) < expected) {
        check_channel_status(group.group_id);
        if (timeout.checkExpired()) {
          printf(
              "P2pIbrcTransportDevice: wait_%s timed out expected=%llu\n",
              kind,
              static_cast<unsigned long long>(expected));
          PIPES_DEVICE_TRAP();
        }
      }
    }
    group.sync();
  }

  __device__ void reset_local(ThreadGroup& group, void* ptr, const char* kind)
      const {
    (void)kind;
    if (ptr == nullptr) {
      trap("P2pIbrcTransportDevice: reset buffer is null");
    }
    if (group.is_leader()) {
      store_release_system_u64(static_cast<uint64_t*>(ptr), 0);
    }
    group.sync();
  }

  // System-scope release fence, portable across NVIDIA/AMD and host/device
  // passes. The PIPES_IS_DEVICE_COMPILE gate makes it a no-op in the host pass
  // so this header is parseable when included from a host .cc TU.
  __device__ __forceinline__ static void threadfence_system() {
#if PIPES_IS_DEVICE_COMPILE
#ifdef __HIP_PLATFORM_AMD__
    amd_fence_system();
#else
    __threadfence_system();
#endif
#endif
  }

  __device__ __forceinline__ static uint64_t load_acquire_system_u64(
      const void* ptr) {
    auto* slot = static_cast<uint64_t*>(const_cast<void*>(ptr));
#ifdef __HIP_PLATFORM_AMD__
    return __hip_atomic_load(slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return cuda::atomic_ref<uint64_t, cuda::thread_scope_system>{*slot}.load(
        cuda::memory_order_acquire);
#endif
  }

  __device__ __forceinline__ static uint32_t load_acquire_system_u32(
      const uint32_t* ptr) {
    auto* slot = const_cast<uint32_t*>(ptr);
#ifdef __HIP_PLATFORM_AMD__
    return __hip_atomic_load(slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return cuda::atomic_ref<uint32_t, cuda::thread_scope_system>{*slot}.load(
        cuda::memory_order_acquire);
#endif
  }

  __device__ __forceinline__ static uint64_t fetch_add_system_u64(
      uint64_t* ptr,
      uint64_t value) {
#ifdef __HIP_PLATFORM_AMD__
    return __hip_atomic_fetch_add(
        ptr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return cuda::atomic_ref<uint64_t, cuda::thread_scope_system>{*ptr}
        .fetch_add(value, cuda::memory_order_relaxed);
#endif
  }

  __device__ __forceinline__ static uint32_t fetch_add_system_u32(
      uint32_t* ptr,
      uint32_t value) {
#ifdef __HIP_PLATFORM_AMD__
    return __hip_atomic_fetch_add(
        ptr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return cuda::atomic_ref<uint32_t, cuda::thread_scope_system>{*ptr}
        .fetch_add(value, cuda::memory_order_relaxed);
#endif
  }

  __device__ __forceinline__ static void store_release_system_u64(
      uint64_t* ptr,
      uint64_t value) {
#ifdef __HIP_PLATFORM_AMD__
    __hip_atomic_store(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system>{*ptr}.store(
        value, cuda::memory_order_release);
#endif
  }

  __device__ IbgdaRemoteBuffer remote_signal_slot(int id) const {
    IBRC_CHECK_SLOT_ID(id, numSignalSlots_, "signal");
    return IbgdaRemoteBuffer(
        static_cast<uint64_t*>(ownedRemoteSignalBuf_.ptr) + id,
        ownedRemoteSignalBuf_.rkey_per_device);
  }

  __device__ IbgdaLocalBuffer local_signal_slot(int id) const {
    IBRC_CHECK_SLOT_ID(id, numSignalSlots_, "signal");
    return IbgdaLocalBuffer(
        static_cast<uint64_t*>(ownedLocalSignalBuf_.ptr) + id,
        ownedLocalSignalBuf_.lkey_per_device);
  }

  __device__ IbgdaLocalBuffer counter_device_slot(int id) const {
    IBRC_CHECK_SLOT_ID(id, numCounterSlots_, "counter");
    return IbgdaLocalBuffer(
        static_cast<uint64_t*>(ownedCounterDeviceBuf_.ptr) + id,
        ownedCounterDeviceBuf_.lkey_per_device);
  }

  __device__ IbgdaLocalBuffer counter_host_slot(int id) const {
    IBRC_CHECK_SLOT_ID(id, numCounterSlots_, "counter");
    return IbgdaLocalBuffer(
        static_cast<uint64_t*>(ownedCounterHostBuf_.ptr) + id,
        ownedCounterHostBuf_.lkey_per_device);
  }

  __device__ __forceinline__ static void trap(const char* msg) {
    printf("%s\n", msg);
    PIPES_DEVICE_TRAP();
  }

  DeviceSpan<IbrcCmdQueueDevice> cmdQueues{};
  uint32_t numNics{0};
  uint32_t maxChannels_{0};
  uint32_t qpsPerConnection_{0};
  DeviceSpan<IbLocalChannel> localChannels_{};
  IbgdaRemoteBuffer ownedRemoteSignalBuf_{};
  IbgdaLocalBuffer ownedLocalSignalBuf_{};
  IbgdaLocalBuffer ownedCounterDeviceBuf_{};
  IbgdaLocalBuffer ownedCounterHostBuf_{};
  int numSignalSlots_{0};
  int numCounterSlots_{0};
  IbChannelLayout channelLayout_{};
};

static_assert(std::is_standard_layout_v<P2pIbrcTransportDevice>);
static_assert(std::is_trivially_copyable_v<P2pIbrcTransportDevice>);

} // namespace comms::prims
