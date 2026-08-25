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
#include "comms/prims/transport/P2pIbTransportDeviceImpl.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibrc/IbrcTypes.h"

namespace comms::prims {

// Legacy bound for device-side waits on the CPU progress thread (flush /
// reserve), used only when no abort handle is wired. Mirrors IBGDA's
// kDefaultDeviceTimeoutCycles: it converts an indefinite hang (a stalled
// progress thread that never publishes an error) into a bounded trap.
//
// With a handle wired, the abort supersedes it and these waits unwind instead
// of trapping -- see the fault-vs-assertion note on `reserve()`. The trap is
// kept for the disabled case because silently dropping a put with no abort
// signal to report would be undetectable data loss, which is worse.
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
 * Group-scope put() returns a completion ticket in the leader thread; a later
 * group-scope wait_local() consumes that leader's ticket collectively.
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
      IbChannelLayout channelLayout = {},
      AbortDevice abort = {})
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
        channelLayout_(channelLayout),
        abort_(abort) {}

  // IBRC round-robins each send/recv chunk's RDMA_WRITE + DATA_READY fetch-add
  // across per-lane command queues / QPs when numLanes > 1 (select_put_queue_id
  // -> seq % num_qp_lanes()), and, with signalPerLane set on the send/recv
  // path, posts each chunk's DATA_READY fetch-add into that lane's own
  // single-writer slot (see put()). The CPU proxy drains each command queue in
  // FIFO order and posts the data write before the signal fetch-add on the same
  // QP, so lane L's slot advances monotonically in lane-L chunk order. The
  // receiver therefore waits on the specific lane that carried each chunk
  // (mirroring the sender's round-robin cursor via recvDataReadyLaneCursor; see
  // detail::wait_recv_data_ready), which removes the cross-lane hazard where a
  // fast lane's later chunk masks a slow lane's not-yet-landed data.

  __device__ IbLocalCompletionTicket
  put(ThreadGroup& group,
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
    return put(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        sigSlot,
        signalVal,
        ctrSlot,
        counterVal);
  }

  __device__ IbLocalCompletionTicket
  put(const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      int signalId = -1,
      uint64_t signalVal = 1,
      int counterId = -1,
      uint64_t counterVal = 1) {
    ThreadGroup solo = make_thread_solo();
    return put(
        solo,
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

  __device__ IbLocalCompletionTicket
  put(ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1,
      bool signalPerLane = false) {
    const bool hasData = nbytes > 0;
    const bool hasSignal = signalBuf.ptr != nullptr;
    const bool hasCounter = counterBuf.ptr != nullptr;
    if (hasData) {
      if (localBuf.ptr == nullptr || remoteBuf.ptr == nullptr) {
        trap("P2pIbrcTransportDevice: put data buffer is null");
      }
    }
    group.sync();

    IbLocalCompletionTicket completion;
    if (group.is_leader()) {
      if (hasData) {
        threadfence_system();
      }
      validate_group_scope(group);
      const uint32_t queueId = select_put_queue_id(group, IbDirection::Send);
      // queue_for_lane encodes the lane ordinal modulo the total lane count.
      const uint32_t laneOrdinal = queueId % num_qp_lanes();
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
        // Per-lane DATA_READY: when signalPerLane is set (the send/recv path),
        // offset the signal target by this put's lane ordinal so each QP lane
        // fetch-adds its own single-writer slot. The receiver mirrors the same
        // round-robin cursor and waits on that lane's slot
        // (detail::wait_recv_data_ready). Mirrors IBGDA put_impl's per-lane
        // effectiveSignalBuf offset. laneOrdinal == 0 (including numLanes == 1)
        // leaves the per-channel base slot, so raw put callers
        // (signalPerLane == false) are unchanged.
        const IbgdaRemoteBuffer effectiveSignalBuf = signalPerLane
            ? signalBuf.subBuffer(
                  sendRecvSignalSlotOffset(static_cast<int>(laneOrdinal)))
            : signalBuf;
        desc.signal_addr = reinterpret_cast<uint64_t>(effectiveSignalBuf.ptr);
        desc.signal_value = signalVal;
        desc.signal_rkey_device_order =
            effectiveSignalBuf.rkey_per_device[nicId].value;
        desc.flags |= IBRC_HAS_SIGNAL | IBRC_SIGNAL_ADD;
      }

      if (hasCounter) {
        desc.counter_addr = reinterpret_cast<uint64_t>(counterBuf.ptr);
        desc.counter_value = counterVal;
        desc.flags |= IBRC_HAS_COUNTER;
      }

      const uint64_t seq = enqueue(queueId, desc);
      if (seq != kIbrcInvalidReadySeq) {
        completion = IbLocalCompletionTicket{
            .completionId = laneOrdinal,
            .value = seq + 1,
        };
      }
      // A skipped enqueue leaves `completion` default-constructed, whose value
      // of 0 is already below every ci, so a later wait_local() on it returns
      // at once instead of blocking on a transfer that was never posted.
    }
    group.sync();
    return completion;
  }

  __device__ IbLocalCompletionTicket
  put(const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1) {
    ThreadGroup solo = make_thread_solo();
    return put(
        solo,
        localBuf,
        remoteBuf,
        nbytes,
        signalBuf,
        signalVal,
        counterBuf,
        counterVal);
  }

  __device__ void wait_local(
      ThreadGroup& group,
      const IbLocalCompletionTicket& ticket,
      const Timeout& timeout = Timeout()) const {
    if (group.is_leader()) {
      const auto& queue = cmdQueues[queue_for_lane(
          group.group_id, IbDirection::Send, ticket.completionId)];
      while (static_cast<int64_t>(
                 load_acquire_system_u64(queue.ci) - ticket.value) < 0) {
        check_status(queue);
        FT_ABORT_BREAK(
            timeout,
            "P2pIbrcTransportDevice: wait_local lane=%u expected=%llu",
            ticket.completionId,
            static_cast<unsigned long long>(ticket.value));
      }
    }
    group.sync();
  }

  __device__ __forceinline__ bool is_local_completion_ready(
      uint32_t channelId,
      const IbLocalCompletionTicket& ticket) const {
    const auto& queue = cmdQueues[queue_for_lane(
        channelId, IbDirection::Send, ticket.completionId)];
    check_status(queue);
    return static_cast<int64_t>(
               load_acquire_system_u64(queue.ci) - ticket.value) >= 0;
  }

  __device__ __forceinline__ void wait_local_completion(
      uint32_t channelId,
      const IbLocalCompletionTicket& ticket,
      const Timeout& timeout) const {
    const auto& queue = cmdQueues[queue_for_lane(
        channelId, IbDirection::Send, ticket.completionId)];
    while (static_cast<int64_t>(
               load_acquire_system_u64(queue.ci) - ticket.value) < 0) {
      check_status(queue);
      FT_ABORT_BREAK(
          timeout,
          "P2pIbrcTransportDevice: local completion lane=%u expected=%llu",
          ticket.completionId,
          static_cast<unsigned long long>(ticket.value));
    }
  }

  __device__ __forceinline__ uint32_t send_completion_lane_count() const {
    return num_qp_lanes();
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
  // Pipelined send/recv — delegated to shared detail helpers.
  // ===========================================================================
  //
  // The send/recv algorithm is transport-agnostic and lives in private helpers
  // in P2pIbTransportDeviceImpl.cuh. The protocol state is owned by this
  // backend device; each method routes every transport op through `*this`, so
  // IBRC reuses IBGDA's send/recv unchanged.

  __device__ __forceinline__ IbLocalChannel& local_channel(uint32_t channelId) {
    validate_channel_id(channelId);
    return localChannels_[channelId];
  }

  __device__ __forceinline__ IbLocalChannel& local_channel(ThreadGroup& group) {
    return local_channel(group.group_id);
  }

  // Per-protocol resources on a channel. `P::kProtoSlot` is a compile-time
  // constant, so this resolves to a fixed offset. What stays on the channel
  // itself is the state every protocol shares: sendQp, recvQp, and
  // recvDataReadyLaneCursor.
  //
  // No default on P: omitting the protocol must be a compile error, not a
  // silent read of the default protocol's cursors.
  template <typename P>
  __device__ __forceinline__ IbChannelProtoSlot& local_channel_slot(
      uint32_t channelId) {
    return local_channel(channelId).protos[P::kProtoSlot];
  }

  template <typename P>
  __device__ __forceinline__ IbChannelProtoSlot& local_channel_slot(
      ThreadGroup& group) {
    return local_channel_slot<P>(group.group_id);
  }

  __host__ __device__ IbChannelLayout& channel_layout() {
    return channelLayout_;
  }

  __host__ __device__ const IbChannelLayout& channel_layout() const {
    return channelLayout_;
  }

  __device__ __forceinline__ std::size_t pipeline_window() const {
    return channelLayout_.perChannelBufferSize != 0
        ? channelLayout_.perChannelBufferSize
        : channelLayout_.perChannelSize;
  }

  __device__ __forceinline__ std::size_t pipeline_window(
      int active_blocks) const {
    (void)active_blocks;
    return pipeline_window();
  }

  __device__ __forceinline__ int pipeline_depth() const {
    return channelLayout_.pipelineDepth;
  }

  __device__ __forceinline__ std::size_t pipeline_chunk() const {
    if (channelLayout_.pipelineDepth <= 0) {
      return 0;
    }
    return pipeline_window() /
        static_cast<std::size_t>(channelLayout_.pipelineDepth);
  }

  template <typename Proto = protocol::Simple>
  __device__ __forceinline__ void init_send_progress(
      ThreadGroup& group,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0) {
    detail::init_send_progress<P2pIbrcTransportDevice, Proto>(
        *this, group, nbytes, max_signal_bytes);
  }

  template <typename Proto = protocol::Simple>
  __device__ __forceinline__ void init_recv_progress(
      ThreadGroup& group,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0) {
    detail::init_recv_progress<P2pIbrcTransportDevice, Proto>(
        *this, group, nbytes, max_signal_bytes);
  }

  template <
      typename CopyOp = Memcpy,
      typename Proto = protocol::Simple,
      typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus progress_send_once(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
    return detail::progress_send_once<P2pIbrcTransportDevice, CopyOp, Proto>(
        *this, group, src, nbytes, max_signal_bytes, timeout, args...);
  }

  template <
      typename CopyOp = Memcpy,
      typename Proto = protocol::Simple,
      typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus progress_recv_once(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
    return detail::progress_recv_once<P2pIbrcTransportDevice, CopyOp, Proto>(
        *this, group, dst, nbytes, max_signal_bytes, timeout, args...);
  }

  // Templated for the same reason P2pIbTransportDevice templates its
  // dispatchers: the definitions live in the progress-impl header, which this
  // header deliberately does not include. A non-template body is compiled
  // eagerly in every translation unit, so the HIP/ROCm build -- which never
  // pulls in that impl header -- failed with -Werror,-Wundefined-inline. A
  // template body is only instantiated where it is actually called, i.e. where
  // the definition is visible.
  template <typename = void>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus
  progress_recv_acquire_once(
      ThreadGroup& group,
      std::size_t nbytes,
      std::size_t max_signal_bytes,
      const Timeout& timeout,
      detail::RecvChunkAcquisition& out) {
    return detail::
        progress_recv_acquire_once<P2pIbrcTransportDevice, protocol::Simple>(
            *this, group, nbytes, max_signal_bytes, timeout, out);
  }

  template <typename = void>
  __device__ __forceinline__ void progress_recv_release_once(
      ThreadGroup& group,
      const detail::RecvChunkAcquisition& view) {
    detail::
        progress_recv_release_once<P2pIbrcTransportDevice, protocol::Simple>(
            *this, group, view);
  }

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void send(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
    detail::send<P2pIbrcTransportDevice, CopyOp>(
        *this, group, src, nbytes, max_signal_bytes, timeout, args...);
  }

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void recv(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
    detail::recv<P2pIbrcTransportDevice, CopyOp>(
        *this, group, dst, nbytes, max_signal_bytes, timeout, args...);
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
    detail::forward<CopyOp>(
        *this, group, dst, fwd, nbytes, max_signal_bytes, timeout, args...);
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

  // Selects the command queue for one data put. The Send cursor is advanced
  // exactly once per data put here; control ops (signal / SLOT_FREE) use
  // control_queue_id (lane 0) and never advance it, keeping the receiver's
  // recvDataReadyLaneCursor mirror in lock-step. `numLanes == 1` stays on lane
  // 0.
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

  // Stays void on abort: a drain that stops early has nothing to report to a
  // caller that is itself about to unwind, and the abort is already recorded
  // for the host.
  // Bounded by the fixed proxy watchdog, not by any caller deadline. This wait
  // is the contract between one rank's SM and its host proxy, and that contract
  // has its own duration; the enclosing collective's deadline is a different
  // bound with a different owner. A cycle compare is a register op, whereas the
  // abort flag lives in mapped host memory, so the loop never reads it.
  __device__ void drain_queue(const IbrcCmdQueueDevice& queue) const {
    const uint64_t target = load_acquire_system_u64(queue.pi);
    const uint64_t start = gpu_clock64();
    const bool ftEnabled = abort_.isEnabled();
    while (load_acquire_system_u64(queue.ci) < target) {
      check_status(queue);
      if (gpu_clock64() - start >= kIbrcDefaultDeviceTimeoutCycles) {
        if (ftEnabled) {
          // The proxy stopped draining. Latching the reason is what lets the
          // rest of the kernel unwind: every other wait observes it through
          // shared state on its own next poll.
          abort_.setAbort(
              comms::fault_tolerance::AbortReason::IBRC_PROXY_TIMEOUT,
              "IBRC proxy watchdog: flush drain");
          return;
        }
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

  // Claims one command-queue slot, spinning while the ring is full.
  //
  // Returns kIbrcInvalidReadySeq when the wait ended in an abort. A full ring
  // means the CPU proxy is behind or stopped, which is a *fault*, not a
  // programming error, so it must not trap: a device trap kills the CUDA
  // context for the whole process, which is precisely the outcome fault
  // tolerance exists to avoid. Reporting the failure instead lets enqueue()
  // skip the descriptor, put() hand back a ticket that is already satisfied,
  // and the block unwind to the end of the kernel on its own.
  //
  // The abort is checked before the fetch_add so an already-aborted producer
  // claims nothing. Aborting *during* the wait still leaves the claimed
  // sequence unpublished, which wedges this queue's proxy cursor -- acceptable
  // because a latched abort is terminal for the communicator, and the progress
  // thread is stopped by teardown rather than by draining.
  //
  // The bound is a *cycle* watchdog, not an abort poll, and it is armed in
  // every mode. That is deliberate on two counts.
  //
  // First, it is the contract between this kernel and the host proxy: a
  // submitted descriptor must be consumed within a bounded time, and that
  // obligation does not change because fault tolerance happens to be on.
  // Gating the watchdog on `!isEnabled()` -- which is what this used to do --
  // meant enabling FT *removed* the only bound, leaving an explicit host abort
  // as the sole exit.
  //
  // Second, polling the abort here would cost a mapped-host read per stalled
  // iteration, and the throttle that would otherwise amortise it cannot live
  // in `abort_` (see the member). A `clock64()` delta is a register compare and
  // is accurate enough for a multi-second bound.
  //
  // So this loop never reads shared state; on expiry it *writes* it, latching
  // IBRC_PROXY_TIMEOUT so every other wait in the kernel unwinds through the
  // normal path. The trade: a block parked here does not observe someone
  // else's abort promptly -- it leaves on its own budget instead. Liveness is
  // what the contract requires, and that is bounded either way.
  __device__ __forceinline__ uint64_t reserve(IbrcCmdQueueDevice& queue) const {
    // One-shot, before anything is claimed: a producer on an already-aborted
    // communicator must leave the wire untouched, so no sequence number is
    // taken and no descriptor is published. This is the only shared-state read
    // on the producer path, and it is not in a loop.
    if (abort_.isAborted()) {
      return kIbrcInvalidReadySeq;
    }
    const uint64_t seq = fetch_add_system_u64(queue.pi, 1);
    if (seq - load_acquire_system_u64(queue.ci) < queue.depth) {
      // Fast path: ring has space. No clock read, no abort read, no copy --
      // this is the whole cost of FT on the healthy producer path.
      return seq;
    }
    // Bounded by the fixed proxy watchdog: this is the SM-to-proxy contract and
    // it is honoured on its own terms, not rescaled by whatever deadline the
    // enclosing collective happens to carry. One shared read for `isEnabled()`,
    // then pure register compares for the rest of the stall.
    const uint64_t start = gpu_clock64();
    const bool ftEnabled = abort_.isEnabled();
    while (seq - load_acquire_system_u64(queue.ci) >= queue.depth) {
      check_status(queue);
      if (gpu_clock64() - start >= kIbrcDefaultDeviceTimeoutCycles) {
        if (ftEnabled) {
          abort_.setAbort(
              comms::fault_tolerance::AbortReason::IBRC_PROXY_TIMEOUT,
              "IBRC proxy watchdog: reserve on a full ring");
          return kIbrcInvalidReadySeq;
        }
        printf("P2pIbrcTransportDevice: reserve timed out\n");
        PIPES_DEVICE_TRAP();
      }
    }
    return seq;
  }

  // Returns kIbrcInvalidReadySeq when the command was skipped because the
  // communicator aborted. Nothing is published in that case, so the CPU proxy
  // never sees a partially written descriptor.
  __device__ __forceinline__ uint64_t
  enqueue(uint32_t queueId, const IbrcDesc& desc) const {
    IbrcCmdQueueDevice& queue = cmdQueues[queueId];
    check_status(queue);
    const uint64_t seq = reserve(queue);
    if (seq == kIbrcInvalidReadySeq) {
      return kIbrcInvalidReadySeq;
    }
    IbrcDesc& slot = queue.descs[seq & queue.mask];
    slot = desc;
    store_release_system_u64(&slot.ready_seq, seq);
    return seq;
  }

  // An error published by the CPU proxy is a remote fault, not a local bug, so
  // with a handle wired it is latched on the abort and every wait in this
  // kernel unwinds. Without one there is nowhere to record it and no way for
  // the host to learn why, so the legacy trap stands.
  __device__ __forceinline__ void check_status(
      const IbrcCmdQueueDevice& queue) const {
    if (queue.status == nullptr) {
      return;
    }
    if (load_acquire_system_u32(&queue.status->error) == 0) {
      return;
    }
    printf(
        "P2pIbrcTransportDevice: queue error queue=%u code=%u\n",
        load_acquire_system_u32(&queue.status->error_queue),
        load_acquire_system_u32(&queue.status->error_code));
    if (!abort_.isEnabled()) {
      PIPES_DEVICE_TRAP();
    }
    abort_.setAbort(comms::fault_tolerance::AbortReason::ABORTED);
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
        FT_ABORT_BREAK(
            timeout,
            "P2pIbrcTransportDevice: wait_%s expected=%llu",
            kind,
            static_cast<unsigned long long>(expected));
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

  // Communicator abort handle, baked in when the host writes this device slot.
  //
  // Held as a member rather than threaded through put()/signal()/flush()
  // because the waits that need it -- reserve() and drain_queue() -- sit behind
  // APIs that take no timeout. Passing it per call would push an abort
  // parameter onto every IBRC producer, and the callers have nothing to do with
  // the answer: these waits terminate themselves. Default-constructed (no
  // handle) keeps the legacy cycle-deadline trap.
  //
  // `AbortFlag`, not `AbortDevice`, and the type is the point.
  //
  // This object lives in device memory, one slot per peer, and every block
  // talking to that peer sees the same one. An `AbortDevice` here would carry
  // mutable poll-throttle state into shared memory: several blocks writing
  // `nextPollCycles_` non-atomically, each gating its polls on an absolute
  // `clock64()` stamped by whichever SM wrote last. `clock64()` is per-SM on
  // NVIDIA, so a block on a lagging SM could sit below a leading SM's throttle
  // and stop polling for far longer than the interval -- missing the abort it
  // was parked waiting for. `AbortFlag` has no such state and no way to poll,
  // so that cannot be written here at all.
  //
  // What remains is what a shared handle can legitimately do: report whether
  // FT is on, and *write* a terminal reason via a system-scope CAS. The waits
  // bound themselves on the device clock instead of on shared reads.
  AbortFlag abort_{};
};

static_assert(std::is_standard_layout_v<P2pIbrcTransportDevice>);
static_assert(std::is_trivially_copyable_v<P2pIbrcTransportDevice>);

} // namespace comms::prims
