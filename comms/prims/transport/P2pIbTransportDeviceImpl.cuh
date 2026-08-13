// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <type_traits>

#include "comms/prims/core/LlxPacket.cuh"
#include "comms/prims/core/MemcpyCopyOp.cuh"
#include "comms/prims/trace/PipesTraceTypes.h"
#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"

namespace comms::prims {

namespace detail {
// Query whether a CopyOp policy is variable-size (e.g. AnsCompress, which
// produces a data-dependent compressed payload and needs the variable-size
// transport protocol). Policies that don't declare `kVariableSize` (Memcpy,
// TileReduce, MemcpyAndSelfCopy, …) are treated as fixed-size.
template <typename C, typename = void>
struct copyop_variable_size : std::false_type {};
template <typename C>
struct copyop_variable_size<C, std::void_t<decltype(C::kVariableSize)>>
    : std::bool_constant<C::kVariableSize> {};
template <typename C>
inline constexpr bool copyop_variable_size_v = copyop_variable_size<C>::value;

__device__ __forceinline__ bool fine_trace_enabled(
    const PipesTraceAllReduceContext* context) {
  return context != nullptr && context->trace.ring != nullptr &&
      context->trace.writeIndex != nullptr;
}

__device__ __forceinline__ void trace_allreduce_event(
    const PipesTraceAllReduceContext* context,
    PipesTraceEventType type,
    uint8_t qpLane,
    std::size_t bytes) {
  if (!fine_trace_enabled(context)) {
    return;
  }
  auto tagged = *context;
  tagged.qpLane = qpLane;
  tagged.bytes = bytes > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(bytes);
  write_pipes_trace_allreduce(tagged, type);
}

__device__ __forceinline__ void trace_allreduce_event(
    const PipesTraceAllReduceContext* context,
    PipesTraceEventType type,
    uint8_t qpLane) {
  trace_allreduce_event(
      context, type, qpLane, context == nullptr ? 0 : context->bytes);
}
} // namespace detail

#if PIPES_IS_DEVICE_COMPILE
__device__ __forceinline__ uint32_t trace_ibgda_step(std::size_t value) {
  constexpr std::size_t kMaxTraceStep = static_cast<std::size_t>(UINT32_MAX);
  return value > kMaxTraceStep ? UINT32_MAX : static_cast<uint32_t>(value);
}

__device__ __forceinline__ void trace_ibgda_event(
    PipesTraceHandle trace,
    uint8_t self_rank,
    PipesTraceEventType type,
    uint32_t step,
    uint16_t group_id) {
  // write_pipes_trace short-circuits when trace.ring == nullptr, so an
  // unconfigured handle has effectively no cost.
  write_pipes_trace(trace, type, step, group_id, self_rank);
}
#endif

/**
 * Protocol policy selecting the send/recv wire format AND its packet geometry.
 * The shared send/recv loop iterates in PAYLOAD bytes and derives physical
 * staging offsets, RDMA lengths, and signal/counter thresholds in WIRE bytes
 * via wire_bytes():
 *   kData        payload bytes per packet / alignment quantum
 *   kPacketBytes wire bytes per packet (== kData for Simple: no flag)
 *   max_payload  wire-bytes -> payload capacity
 *   wire_bytes   payload-bytes -> wire bytes (round up to a whole packet)
 * `Simple` is nccl-"simple" put(data)+explicit-signal: a degenerate packet of
 * 16 contiguous data bytes, no flag, so wire == payload. Low-latency (LL)
 * policies are added in later diffs. Tags live in `protocol` to avoid
 * collisions with the generic name `Simple` at comms::prims scope.
 */
namespace protocol {
struct Simple {
  // Resource slot this protocol owns on every channel. Slot 0 is the default
  // protocol; additional protocols take higher slots (see kNumProtoSlots).
  static constexpr int kProtoSlot = 0;
  static constexpr std::size_t kData = 16; // protocol alignment quantum
  static constexpr std::size_t kPacketBytes = 16; // wire == payload
  __host__ __device__ static constexpr std::size_t max_payload(
      std::size_t wireBytes) {
    return wireBytes;
  }
  __host__ __device__ static constexpr std::size_t wire_bytes(
      std::size_t payloadBytes) {
    return (payloadBytes + kData - 1) / kData * kPacketBytes;
  }
};
} // namespace protocol

// Low-latency (data+flag) wire format. Packet-geometry policy is LlxPacket<4,4>
// (8 B packet = 4 B data + 4 B flag), so wire == 2x payload; readiness is the
// inline flag (no DATA_READY -- see consumeRecvBuf(LL)). LL reuses the same
// channel/state as Simple (shared IbChannelLayout, staging, and progress
// cursor); forward is not implemented for LL yet.
namespace protocol {
struct LL {
  // Slot 1: LL owns its own per-channel resource slot. The two protocols must
  // not share one -- the progress cursors, staging ring position, and
  // cumulative signal counters are all persistent across kernel launches, so a
  // slot left mid-stream by one protocol is meaningless to the other. The
  // channel's QPs and its lane cursor ARE shared; only slot-indexed state is
  // duplicated.
  static constexpr int kProtoSlot = 1;
  using Packet = LlxPacket<4, 4>;
  static constexpr std::size_t kData = Packet::kData;
  static constexpr std::size_t kPacketBytes = Packet::kPacketBytes;
  __host__ __device__ static std::size_t max_payload(std::size_t wireBytes) {
    return Packet::max_payload(wireBytes);
  }
  __host__ __device__ static std::size_t wire_bytes(std::size_t payloadBytes) {
    return Packet::wire_bytes(payloadBytes);
  }
};
} // namespace protocol

namespace detail {

template <typename P, typename Transport>
__device__ __forceinline__ IbChannelProgress& progress_send_slot(
    Transport& transport,
    ThreadGroup& group);

template <typename P, typename Transport>
__device__ __forceinline__ IbChannelProgress& progress_recv_slot(
    Transport& transport,
    ThreadGroup& group);

/**
 * Every channel resource one (channel, protocol) transfer addresses, resolved
 * in one place.
 *
 * Callers must never derive a channel or slot index themselves: the whole point
 * is that the protocol coordinate lives here and nowhere else, so adding a
 * protocol cannot silently miss a site. `groupId` stays the LOGICAL channel --
 * which is also the QP channel -- and the protocol is a separate compile-time
 * coordinate, never folded into it.
 */
struct ChannelSlotView {
  IbLocalChannel& channel; ///< shared: sendQp, recvQp, recvDataReadyLaneCursor
  IbChannelProtoSlot& local; ///< this protocol's cursors and local buffer views
  IbRemoteChannel remote; ///< this protocol's peer-side views
  std::size_t stagingBase; ///< byte offset of this slot's staging window
};

template <typename P, typename Transport>
__device__ __forceinline__ ChannelSlotView acquire_channel(
    Transport& transport,
    const IbChannelLayout& channelLayout,
    ThreadGroup& group);

__device__ __forceinline__ static std::size_t valid_payload_bytes(
    std::size_t byteOffset,
    std::size_t chunkBytes,
    std::size_t payloadBytes);

__device__ __forceinline__ static std::size_t align_protocol_bytes(
    std::size_t nbytes);

__device__ __forceinline__ static uint64_t round_up_to_multiple(
    uint64_t value,
    std::size_t alignment);

__device__ __forceinline__ static std::size_t signal_alignment(
    std::size_t maxSignalBytes,
    std::size_t perBlockSlot);

__device__ __forceinline__ static std::size_t
tail_padding_for_signal_granularity(
    uint64_t baseByte,
    std::size_t maxSignalBytes,
    std::size_t perBlockSlot,
    std::size_t payloadBytes);

__device__ __forceinline__ std::size_t pipeline_window(
    const IbChannelLayout& channelLayout);

__device__ __forceinline__ std::size_t pipeline_chunk(
    const IbChannelLayout& channelLayout);

__device__ __forceinline__ void assert_progress_slot_idle(
    ThreadGroup& group,
    const IbChannelProgress& slot,
    const char* opName);

template <typename P, typename Transport>
__device__ __forceinline__ void prepare_send_slot(
    Transport& transport,
    ThreadGroup& group,
    uint32_t slotId,
    uint64_t generation,
    const Timeout& timeout = Timeout());

template <typename P, typename Transport>
__device__ __forceinline__ void record_send_completion(
    Transport& transport,
    uint32_t channelId,
    uint32_t slotId,
    uint64_t generation,
    const IbLocalCompletionTicket& ticket);

template <typename Transport>
__device__ __forceinline__ void init_send_progress(
    Transport& transport,
    ThreadGroup& group,
    std::size_t nbytes,
    std::size_t max_signal_bytes);

template <typename Transport>
__device__ __forceinline__ void init_recv_progress(
    Transport& transport,
    ThreadGroup& group,
    std::size_t nbytes,
    std::size_t max_signal_bytes);

template <
    typename Transport,
    typename CopyOp,
    typename Proto = protocol::Simple,
    typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus progress_send_once(
    Transport& transport,
    ThreadGroup& group,
    const void* __restrict__ src,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    Args... args);

template <
    typename Transport,
    typename CopyOp,
    typename Proto = protocol::Simple,
    typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus progress_recv_once(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    Args... args);

template <typename Transport, typename CopyOp, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus
progress_recv_once_with_trace(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    const PipesTraceAllReduceContext& traceContext,
    PipesTraceProgressState& traceState,
    Args... args);

/**
 * Blocking transport-agnostic pipelined RDMA send/recv helpers.
 *
 * Resumable progress definitions live in
 * `P2pIbTransportProgressImpl.cuh` so progress-only changes do not invalidate
 * blocking-only collective kernels.
 */

// What the leader put hands to the wire: the DATA_READY signal slot + credit.
// protocol::Simple returns the remote DATA_READY slot + credit; protocol::LL
// returns an empty buffer (its inline flag is the readiness mark, so the put
// carries data only).
struct SendSignal {
  IbgdaRemoteBuffer buf;
  uint64_t val;
};

/**
 * Blocking wait for one receive chunk's DATA_READY on its round-robin lane.
 *
 * Both IB backends round-robin each chunk's RDMA_WRITE + DATA_READY fetch-add
 * across `numLanes` single-writer slots. Chunk i rides lane `i % numLanes`,
 * driven by the sender's free-running per-(channel, Send) cursor, which the
 * receiver mirrors in `localChannel.recvDataReadyLaneCursor`. Waiting on that
 * lane's own slot (not the summed cumulative) guarantees chunk i's RDMA_WRITE
 * has landed, because the lane's single RC QP delivers the DATA_READY fetch-add
 * only after its data write. This removes the cross-lane out-of-order hazard
 * where a fast lane's later chunk pushes the summed DATA_READY past chunk i's
 * threshold while chunk i's data (on a slow lane) is still in flight. When
 * `numLanes` is 1, this degenerates to exactly the single-slot cumulative wait
 * on lane 0. On success the leader advances `recvDataReadyLaneCursor` and this
 * lane's `recvLaneExpected` by exactly one chunk.
 */
template <typename Transport>
__device__ __forceinline__ void wait_recv_data_ready(
    Transport& transport,
    ThreadGroup& group,
    IbLocalChannel& localChannel,
    const IbgdaLocalBuffer& localDataReady,
    std::size_t chunkBytes,
    const Timeout& timeout) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const uint32_t numLanes =
      static_cast<uint32_t>(transport.channel_layout().numLanes);
  const uint32_t lanes = numLanes == 0 ? 1U : numLanes;
  if (group.is_leader()) {
    // Simple's slot unconditionally: an explicit DATA_READY signal IS Simple's
    // readiness mark. A protocol that carries its flag inline never reaches
    // here, so there is no other slot to select. The lane cursor is
    // channel-scoped (it mirrors the shared sendQp.cursor); the per-lane
    // expected totals are slot-scoped, mirroring Simple's own DATA_READY slots.
    IbChannelProtoSlot& protoSlot =
        localChannel.protos[protocol::Simple::kProtoSlot];
    // Truncate recvDataReadyLaneCursor to 32 bits BEFORE the modulo so the lane
    // matches the sender's uint32 Send cursor once it wraps at 2^32; otherwise
    // a non-power-of-two numLanes would desync the lane after wrap.
    const uint32_t lane =
        static_cast<uint32_t>(localChannel.recvDataReadyLaneCursor) % lanes;
    const uint64_t expected = protoSlot.recvLaneExpected[lane] + chunkBytes;
    const IbgdaLocalBuffer laneBuf = localDataReady.subBuffer(
        sendRecvSignalSlotOffset(static_cast<int>(lane)));
    ThreadGroup solo{
        0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
    transport.wait_signal(solo, laneBuf, expected, timeout);
    protoSlot.recvLaneExpected[lane] = expected;
    ++localChannel.recvDataReadyLaneCursor;
  }
  group.sync();
#else
  (void)transport;
  (void)group;
  (void)localChannel;
  (void)localDataReady;
  (void)chunkBytes;
  (void)timeout;
#endif
}

/**
 * send — send one block's tile via pipelined RDMA.
 *
 * Copies src -> sendStaging, then RDMA puts sendStaging -> peer's
 * recvStaging. For this call, each logical slot contributes one
 * perBlockSlot-sized region for this group. If nbytes > perBlockSlot, send()
 * advances through multiple ring positions. max_signal_bytes can further
 * subdivide each perBlockSlot into multiple signaled sub-chunks, enabling
 * finer-grained overlap at the receiver.
 *
 * Signaling protocol (per group):
 *   LOCAL_DONE — completion ticket returned by each RDMA put. Blocking send
 *                waits on the latest channel frontier before overwriting
 *                local sendStaging.
 *   SLOT_FREE  — receiver increments by bytesThis for each signaled byte
 *                range. send waits before overwriting recvStaging.
 *   DATA_READY — sender increments by bytesThis, piggybacked on put.
 *                recv waits on this before reading recvStaging.
 *
 * The channel progress cursor persists across calls, so send() resumes the
 * staging-ring cursor and protocol sequence numbers on each invocation. This
 * allows callers to pipeline across repeated send() calls without a separate
 * drain.
 *
 * The caller must keep the transport layout stable while a sequence is in
 * flight. `max_signal_bytes` may vary across calls because it changes only
 * sub-chunk signaling, not the fixed channel staging layout.
 *
 * @param transport       Owning transport used for every transport op.
 * @param group           ThreadGroup (all threads participate in memcpy,
 *                        leader does RDMA ops).
 * @param src             Source data for this block's tile.
 * @param nbytes          Bytes to send for this group. Internally consumed
 *                        in perBlockSlot-sized pieces, or smaller sub-chunks
 *                        when max_signal_bytes is set.
 * @param max_signal_bytes Max bytes per signaled sub-chunk within one
 *                        perBlockSlot. 0 means one signal per perBlockSlot.
 * @param timeout         Optional timeout for wait operations.
 */
// Per-call geometry for the blocking send()/recv() loops. One definition serves
// both directions -- send and recv share the same layout, so the caller
// resolves its own progress slot separately. Tag-dispatched on the protocol
// `P`, but `P` is an unused tag today (single-space, 16B); per-protocol packet
// geometry is added in a later diff. Blocking-path analog of
// make_progress_geometry().
struct SendRecvGeometry {
  // See ProgressGeometry: logical channel and protocol resource slot stay
  // separate coordinates.
  int groupId;
  int slotIndex;
  std::size_t perBlockSlotWire; // physical (wire) bytes per (slot, group)
  std::size_t perBlockSlotPayload; // payload capacity of that region
  std::size_t chunkPayload; // max payload per signaled chunk (>= 1 packet)
  std::size_t pipelineBytesPayload; // ring window in payload bytes
  std::size_t pipelineBytesWire; // ring window in wire bytes
  std::size_t
      payloadProtocolBytes; // payload bytes rounded up to kData; loop bound
};

template <typename P>
__device__ __forceinline__ SendRecvGeometry calcGeometry(
    P,
    const IbChannelLayout& channelLayout,
    ThreadGroup& group,
    std::size_t nbytes,
    std::size_t max_signal_bytes) {
  const int groupId = group.group_id;
  // group_id selects a LOGICAL channel, so bound it by numChannels.
  // maxChannels counts resource SLOTS (numChannels * kNumProtoSlots) and would
  // be kNumProtoSlots-times too loose here.
  const int numChannels = channelLayout.numChannels;
  if (groupId >= numChannels) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: send/recv group_id=%u >= numChannels=%d\n",
          groupId,
          numChannels);
    }
    PIPES_DEVICE_TRAP();
  }
  const std::size_t perBlockSlotWire = pipeline_chunk(channelLayout);
  if (perBlockSlotWire == 0) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: send/recv perBlockSlot=0 "
          "(perChannelBufferSize=%llu, pipelineDepth=%d)\n",
          (unsigned long long)pipeline_window(channelLayout),
          channelLayout.pipelineDepth);
    }
    PIPES_DEVICE_TRAP();
  }
  const std::size_t perBlockSlotPayload = P::max_payload(perBlockSlotWire);
  // Chunk size in PAYLOAD bytes, aligned down to whole packets (kData).
  std::size_t chunkPayload =
      (max_signal_bytes > 0 && max_signal_bytes < perBlockSlotPayload)
      ? (max_signal_bytes / P::kData * P::kData)
      : perBlockSlotPayload;
  if (chunkPayload == 0) {
    chunkPayload = perBlockSlotPayload;
  }
  const std::size_t pipelineBytesWire = pipeline_window(channelLayout);
  // Payload bytes rounded up to a whole packet (kData). For Simple == round-16.
  const std::size_t payloadProtocolBytes =
      (nbytes + P::kData - 1) / P::kData * P::kData;
  return SendRecvGeometry{
      .groupId = groupId,
      .slotIndex = channelLayout.protoChannelSlot(groupId, P::kProtoSlot),
      .perBlockSlotWire = perBlockSlotWire,
      .perBlockSlotPayload = perBlockSlotPayload,
      .chunkPayload = chunkPayload,
      .pipelineBytesPayload = P::max_payload(pipelineBytesWire),
      .pipelineBytesWire = pipelineBytesWire,
      .payloadProtocolBytes = payloadProtocolBytes,
  };
}

template <typename CopyOp = Memcpy, typename... Args>
__device__ __forceinline__ SendSignal prepareSendBuf(
    protocol::Simple,
    ThreadGroup& group,
    char* staging,
    const char* src,
    std::size_t payloadBytes,
    std::size_t nbytes,
    std::size_t dataOff,
    uint64_t /*flagVal*/,
    const IbRemoteChannel& remoteChannel,
    uint64_t signalVal,
    Args... args) {
#if PIPES_IS_DEVICE_COMPILE
  const std::size_t validBytes =
      valid_payload_bytes(dataOff, payloadBytes, nbytes);
  if (validBytes > 0) {
    CopyOp::send(staging, src, validBytes, group, dataOff, args...);
  }
  group.sync();
  return SendSignal{remoteChannel.dataReady, signalVal};
#else
  (void)group;
  (void)staging;
  (void)src;
  (void)payloadBytes;
  (void)nbytes;
  (void)dataOff;
  (void)remoteChannel;
  (void)signalVal;
  ((void)args, ...);
  return SendSignal{};
#endif
}

// Simple decode: wait for the sender's DATA_READY on this chunk's lane, then
// cooperative CopyOp::recv from contiguous staging.
template <typename CopyOp = Memcpy, typename Transport, typename... Args>
__device__ __forceinline__ void consumeRecvBuf(
    protocol::Simple,
    Transport& transport,
    ThreadGroup& group,
    IbLocalChannel& localChannel,
    const IbgdaLocalBuffer& localDataReady,
    char* dst,
    const char* staging,
    std::size_t payloadBytes,
    std::size_t nbytes,
    std::size_t dataOff,
    uint64_t /*flagVal*/,
    uint64_t waitCredit,
    const Timeout& timeout,
    const PipesTraceAllReduceContext* traceContext,
    Args... args) {
#if PIPES_IS_DEVICE_COMPILE
  const uint32_t numLanes =
      static_cast<uint32_t>(transport.channel_layout().numLanes);
  const uint8_t qpLane = static_cast<uint8_t>(
      numLanes == 0 ? 0 : localChannel.recvDataReadyLaneCursor % numLanes);
  if (group.is_leader()) {
    trace_allreduce_event(
        traceContext,
        PipesTraceEventType::kAllReduceDataReadyWaitBegin,
        qpLane,
        waitCredit);
  }
  wait_recv_data_ready(
      transport, group, localChannel, localDataReady, waitCredit, timeout);
  if (group.is_leader()) {
    trace_allreduce_event(
        traceContext,
        PipesTraceEventType::kAllReduceDataReadyWaitEnd,
        qpLane,
        waitCredit);
  }
  const std::size_t validBytes =
      valid_payload_bytes(dataOff, payloadBytes, nbytes);
  if (validBytes > 0) {
    if (group.is_leader()) {
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceReduceCopyBegin,
          qpLane,
          validBytes);
    }
    CopyOp::recv(dst, staging, validBytes, group, dataOff, args...);
    if (group.is_leader()) {
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceReduceCopyEnd,
          qpLane,
          validBytes);
    }
  }
  group.sync();
#else
  (void)transport;
  (void)group;
  (void)localChannel;
  (void)localDataReady;
  (void)dst;
  (void)staging;
  (void)payloadBytes;
  (void)nbytes;
  (void)dataOff;
  (void)waitCredit;
  (void)timeout;
  (void)traceContext;
  ((void)args, ...);
#endif
}

// Simple forward: wait for the upstream chunk's DATA_READY, then a single fused
// CopyOp::forward transforms recvStaging -> dst + fwdStaging, and return the
// DATA_READY SendSignal that the relay put piggybacks. This is the forward
// analog of consumeRecvBuf (recv-side readiness) fused with prepareSendBuf
// (relay signal); LL later overrides it to poll inline flags, repack the fwd
// staging, and return an empty signal.
template <
    typename CopyOp = Memcpy,
    typename Transport,
    typename FwdTransport,
    typename... Args>
__device__ __forceinline__ SendSignal prepareForwardBuf(
    protocol::Simple,
    Transport& transport,
    FwdTransport& fwdTransport,
    ThreadGroup& group,
    IbLocalChannel& recvLocalChannel,
    const IbgdaLocalBuffer& recvDataReady,
    char* dst,
    char* fwdStaging,
    const char* recvStaging,
    std::size_t payloadBytes,
    std::size_t nbytes,
    std::size_t dataOff,
    uint64_t recvWaitCredit,
    const IbRemoteChannel& fwdRemoteChannel,
    uint64_t fwdSignalVal,
    uint32_t fwdSlot,
    uint64_t fwdPipelineCycle,
    const Timeout& timeout,
    const PipesTraceAllReduceContext* recvTraceContext,
    const PipesTraceAllReduceContext* sendTraceContext,
    Args... args) {
#if PIPES_IS_DEVICE_COMPILE
  // Both waits must clear before CopyOp::forward, which reads recvStaging and
  // writes fwdStaging; they are independent, so the order is a latency choice,
  // not a correctness one. Upstream readiness goes first because it is the
  // remote-dependent wait: by the time the peer's data lands, the local NIC has
  // usually already released this fwdStaging slot, so prepare_send_slot
  // degenerates to a single completion check instead of a CQ-polling spin.
  // Folded in here rather than left to the caller -- both are correctness
  // requirements of this function, not of its call site.
  const uint32_t recvNumLanes =
      static_cast<uint32_t>(transport.channel_layout().numLanes);
  const uint8_t recvQpLane = static_cast<uint8_t>(
      recvNumLanes == 0
          ? 0
          : recvLocalChannel.recvDataReadyLaneCursor % recvNumLanes);
  if (group.is_leader()) {
    trace_allreduce_event(
        recvTraceContext,
        PipesTraceEventType::kAllReduceDataReadyWaitBegin,
        recvQpLane,
        recvWaitCredit);
  }
  wait_recv_data_ready(
      transport,
      group,
      recvLocalChannel,
      recvDataReady,
      recvWaitCredit,
      timeout);
  if (group.is_leader()) {
    trace_allreduce_event(
        recvTraceContext,
        PipesTraceEventType::kAllReduceDataReadyWaitEnd,
        recvQpLane,
        recvWaitCredit);
    trace_allreduce_event(
        sendTraceContext,
        PipesTraceEventType::kAllReduceLocalCompletionWaitBegin,
        static_cast<uint8_t>(kPipesTraceQpLaneMask),
        fwdSignalVal);
  }
  prepare_send_slot<protocol::Simple>(
      fwdTransport, group, fwdSlot, fwdPipelineCycle, timeout);
  if (group.is_leader()) {
    trace_allreduce_event(
        sendTraceContext,
        PipesTraceEventType::kAllReduceLocalCompletionWaitEnd,
        static_cast<uint8_t>(kPipesTraceQpLaneMask),
        fwdSignalVal);
  }
  const std::size_t validBytes =
      valid_payload_bytes(dataOff, payloadBytes, nbytes);
  if (validBytes > 0) {
    if (group.is_leader()) {
      trace_allreduce_event(
          recvTraceContext,
          PipesTraceEventType::kAllReduceStageCopyBegin,
          recvQpLane,
          validBytes);
    }
    CopyOp::forward(
        dst, fwdStaging, recvStaging, validBytes, group, dataOff, args...);
    if (group.is_leader()) {
      trace_allreduce_event(
          recvTraceContext,
          PipesTraceEventType::kAllReduceStageCopyEnd,
          recvQpLane,
          validBytes);
    }
  }
  group.sync();
  return SendSignal{fwdRemoteChannel.dataReady, fwdSignalVal};
#else
  (void)transport;
  (void)fwdTransport;
  (void)group;
  (void)recvLocalChannel;
  (void)recvDataReady;
  (void)dst;
  (void)fwdStaging;
  (void)recvStaging;
  (void)payloadBytes;
  (void)nbytes;
  (void)dataOff;
  (void)recvWaitCredit;
  (void)fwdRemoteChannel;
  (void)fwdSignalVal;
  (void)fwdSlot;
  (void)fwdPipelineCycle;
  (void)timeout;
  (void)recvTraceContext;
  (void)sendTraceContext;
  ((void)args, ...);
  return SendSignal{};
#endif
}
// LL encode: pack payload + trailing flag=flagVal into staging via
// LLImpl::pack; the put carries NO DATA_READY (empty signal) -- the inline
// flag is the readiness mark. Ignores the remote signal slot/value that
// Simple uses.
template <typename CopyOp = Memcpy, typename... Args>
__device__ __forceinline__ SendSignal prepareSendBuf(
    protocol::LL,
    ThreadGroup& group,
    char* staging,
    const char* src,
    std::size_t payloadBytes,
    std::size_t nbytes,
    std::size_t dataOff,
    uint64_t flagVal,
    const IbRemoteChannel& remoteChannel,
    uint64_t signalVal,
    Args... args) {
  using P = LlxPacket<4, 4>;
  static_assert(
      has_sendLL_v<CopyOp, P>,
      "LL send path requires a CopyOp with a packet-aware sendLL<P>(); Memcpy "
      "provides one. A reduce/convert CopyOp must supply its own -- a plain "
      "contiguous copy cannot address the data+flag interleaved staging");
#if PIPES_IS_DEVICE_COMPILE
  (void)remoteChannel;
  (void)signalVal;
  // Clamp to the REAL payload before handing it to the codec. payloadBytes is
  // rounded up to kData for the wire/credit stream, but the caller's src only
  // holds nbytes -- packing the padded length reads up to kData-1 bytes past
  // it, which faults on a tightly-sized device allocation. Staging offsets and
  // the RDMA length stay padded; only the codec is clamped. Both ranks derive
  // validBytes from values they already agree on, so packet_count() matches on
  // each side. Mirrors the Simple overload above.
  const std::size_t validBytes =
      valid_payload_bytes(dataOff, payloadBytes, nbytes);
  if (validBytes > 0) {
    CopyOp::template sendLL<P>(
        group,
        staging,
        src,
        validBytes,
        dataOff,
        static_cast<typename P::FlagType>(flagVal),
        args...);
  }
  return SendSignal{IbgdaRemoteBuffer{}, /*val=*/0};
#else
  (void)group;
  (void)staging;
  (void)src;
  (void)payloadBytes;
  (void)nbytes;
  (void)dataOff;
  (void)flagVal;
  (void)remoteChannel;
  (void)signalVal;
  ((void)args, ...);
  return SendSignal{};
#endif
}

// LL decode: poll the inline flags == flagVal and copy staging -> dst via
// LLImpl::unpack. No signal wait; ignores the local channel/credit Simple
// uses.
template <typename CopyOp = Memcpy, typename Transport, typename... Args>
__device__ __forceinline__ void consumeRecvBuf(
    protocol::LL,
    Transport& transport,
    ThreadGroup& group,
    IbLocalChannel& localChannel,
    const IbgdaLocalBuffer& localDataReady,
    char* dst,
    const char* staging,
    std::size_t payloadBytes,
    std::size_t nbytes,
    std::size_t dataOff,
    uint64_t flagVal,
    uint64_t waitCredit,
    const Timeout& timeout,
    Args... args) {
  using P = LlxPacket<4, 4>;
  static_assert(
      has_recvLL_v<CopyOp, P>,
      "LL recv path requires a CopyOp with a packet-aware recvLL<P>(); Memcpy "
      "provides one. A reduce/convert CopyOp must supply its own -- a plain "
      "contiguous copy cannot address the data+flag interleaved staging");
#if PIPES_IS_DEVICE_COMPILE
  (void)transport;
  (void)localChannel;
  (void)localDataReady;
  (void)waitCredit;
  // Same clamp as the send side: unpacking the padded length writes up to
  // kData-1 bytes past the caller's dst, silently corrupting whatever follows.
  const std::size_t validBytes =
      valid_payload_bytes(dataOff, payloadBytes, nbytes);
  if (validBytes > 0) {
    CopyOp::template recvLL<P>(
        group,
        dst,
        staging,
        validBytes,
        dataOff,
        static_cast<typename P::FlagType>(flagVal),
        timeout,
        args...);
  }
#else
  (void)transport;
  (void)group;
  (void)localChannel;
  (void)localDataReady;
  (void)dst;
  (void)staging;
  (void)payloadBytes;
  (void)nbytes;
  (void)dataOff;
  (void)flagVal;
  (void)waitCredit;
  (void)timeout;
  ((void)args, ...);
#endif
}

template <
    typename Transport,
    typename CopyOp = Memcpy,
    typename Proto = protocol::Simple,
    typename... Args>
__device__ __forceinline__ void send_impl(
    Transport& transport,
    ThreadGroup& group,
    const void* __restrict__ src,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout(),
    const PipesTraceAllReduceContext* traceContext = nullptr,
    Args... args) {
  // The variable-size (compressed) loop below keeps its encode inline rather
  // than behind the prepareSendBuf/consumeRecvBuf seam, so it is Simple-shaped
  // and a non-default protocol would silently drive the wrong encode. Forbid
  // the pairing; there is no LL-over-compressed use case today.
  static_assert(
      !detail::copyop_variable_size_v<CopyOp> ||
          std::is_same_v<Proto, protocol::Simple>,
      "variable-size CopyOps (e.g. AnsCompress) are supported on "
      "protocol::Simple only; the compressed loop is not behind the "
      "prepareSendBuf/consumeRecvBuf seam.");
#if !PIPES_IS_DEVICE_COMPILE
  (void)transport;
  (void)group;
  (void)src;
  (void)nbytes;
  (void)max_signal_bytes;
  (void)timeout;
  (void)traceContext;
#else
  if (nbytes == 0) {
    return;
  }
  auto& channelLayout = transport.channel_layout();
  const SendRecvGeometry geometry =
      calcGeometry(Proto{}, channelLayout, group, nbytes, max_signal_bytes);
  const int groupId = geometry.groupId;
  const std::size_t perBlockSlotWire = geometry.perBlockSlotWire;
  const std::size_t perBlockSlotPayload = geometry.perBlockSlotPayload;
  [[maybe_unused]] const std::size_t chunkPayload = geometry.chunkPayload;
  [[maybe_unused]] const std::size_t pipelineBytesPayload =
      geometry.pipelineBytesPayload;
  const std::size_t pipelineBytesWire = geometry.pipelineBytesWire;
  [[maybe_unused]] const std::size_t payloadProtocolBytes =
      geometry.payloadProtocolBytes;

  auto& state = progress_send_slot<Proto>(transport, group);
  const ChannelSlotView ch =
      acquire_channel<Proto>(transport, channelLayout, group);
  IbLocalChannel& localChannel = ch.channel;
  const IbgdaLocalBuffer localSlotFree = ch.local.slotFree;
  const IbRemoteChannel remoteChannel = ch.remote;
  assert_progress_slot_idle(group, state, "send");
  const uint64_t baseByte = static_cast<uint64_t>(state.nextStep);
  // Category of the previous op on this slot, read before the leader overwrites
  // it below. Used only to make the compressed slot-alignment trap precise
  // about a fixed->variable transition (byte cursor vs sub-chunk cursor).
  [[maybe_unused]] const bool prevOpVariableSize = state.activeVariableSize;
  const std::size_t protocolTailPadding = tail_padding_for_signal_granularity(
      baseByte, max_signal_bytes, perBlockSlotPayload, nbytes);
  [[maybe_unused]] const uint64_t payloadBaseByte = baseByte;
  [[maybe_unused]] const std::size_t protocolBytes =
      payloadProtocolBytes + protocolTailPadding;
  if (group.is_leader()) {
    // Record this op's CopyOp category in the persistent slot state so the
    // cross-category contract is explicit (see IbChannelProgress). Safe to
    // switch categories between ops: nextStep is a slot-pinned wire-byte
    // cursor.
    state.activeVariableSize = detail::copyop_variable_size_v<CopyOp>;
    state.activeStage = detail::IbSendRecvProgressStage::Busy;
    state.activeBaseStep = static_cast<int64_t>(baseByte);
    state.activeNextByte = 0;
    state.activeTailPadding = protocolTailPadding;
    trace_allreduce_event(
        traceContext,
        PipesTraceEventType::kAllReducePathStaged,
        static_cast<uint8_t>(kPipesTraceQpLaneMask));
  }

  if constexpr (detail::copyop_variable_size_v<CopyOp>) {
    // Variable-size (compressed) send. A compressed sub-chunk's on-wire size is
    // data-dependent, so the staging ring reserves a fixed worst-case region
    // (`chunkStride`) per sub-chunk while the RDMA put writes only the bytes
    // the CopyOp actually produced. Flow control still runs in WIRE bytes (the
    // same unit as the plain path above), so the shared per-channel progress
    // cursor, DATA_READY/SLOT_FREE signals, and lane/completion machinery are
    // reused unchanged and a plain send can safely follow a compressed one on
    // the same channel. The cursor advances in whole slots (`perBlockSlot`):
    // the last sub-chunk of every slot carries the slot's unused tail as
    // flow-control credit so the cumulative stream lands exactly on a slot
    // boundary.
    //
    // Cross-category staging-reuse contract (shared physical ring): the fixed
    // and variable paths share one physical staging ring, but a switch between
    // categories can NOT reuse a slot region while the previous category still
    // has NIC reads in flight. Reuse is gated by the per-slot completion
    // handshake, which is category-agnostic: every put records its completion
    // via record_send_completion(groupId, ringSlot, cycle), and every slot is
    // re-armed by prepare_send_slot(ringSlot, cycle) below, which blocks until
    // the NIC has finished the prior use of that ring slot. Because both paths
    // key on the same (channel, slot) completion state and advance the same
    // wire-byte cursor, a following op -- fixed after variable or vice versa --
    // waits out any in-flight read before overwriting staging. Two ops on a
    // channel never overlap (assert_progress_slot_idle traps on a non-Done
    // predecessor) and signal_alignment() pins the shared cursor to whole
    // perBlockSlot strides for both categories, so a switch always resumes on a
    // slot boundary. No separate cross-category drain is therefore required.
    const std::size_t perBlockSlot = perBlockSlotWire;
    const std::size_t pipelineBytes = pipelineBytesWire;
    const int pipelineDepth = channelLayout.pipelineDepth;
    // nvcompdx requires 512-byte-aligned staging regions; every compressed
    // sub-chunk starts at groupId*pipelineBytes + ringSlot*perBlockSlot +
    // subStep*chunkStride, so perBlockSlot must be 512-aligned for those starts
    // to be aligned (chunkStride already is).
    if ((perBlockSlot & 511ULL) != 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: compressed send perBlockSlot=%llu not 512-aligned; "
            "size the per-channel staging so each block's slot is a multiple of "
            "the NIC burst alignment.\n",
            (unsigned long long)perBlockSlot);
      }
      PIPES_DEVICE_TRAP();
    }
    // The compressed cursor is slot-granular and requires a slot-aligned start.
    // `signal_alignment()` pins the persistent per-channel cursor to whole
    // perBlockSlot strides for BOTH plain and compressed ops, so a preceding
    // plain (byte-granular) op always leaves this aligned. This remains as a
    // defensive invariant check: fail fast rather than corrupt if some future
    // path advances the cursor on a sub-slot boundary.
    if ((baseByte % perBlockSlot) != 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: compressed send requires a slot-aligned start "
            "cursor (baseByte=%llu, perBlockSlot=%llu, prevOpVariableSize=%d); a "
            "preceding op left the per-channel cursor mid-slot, which would "
            "reinterpret its byte cursor as a compressed sub-chunk cursor.\n",
            (unsigned long long)baseByte,
            (unsigned long long)perBlockSlot,
            (int)prevOpVariableSize);
      }
      PIPES_DEVICE_TRAP();
    }
    // For the 0 sentinel (or an over-large request) pick the largest chunk
    // whose worst-case ANS-expanded staging still fits one perBlockSlot, via
    // the policy's max_safe_chunk_size_for_slot(). Using perBlockSlot directly
    // would make worst_case_chunk_stride() exceed the slot and trap, since a
    // compressed sub-chunk's worst case is ~1.3x its uncompressed input. send()
    // and recv() derive the identical value from the shared perBlockSlot, so
    // the two sides agree on the chunking without any exchange.
    std::size_t chunkSize =
        (max_signal_bytes > 0 && max_signal_bytes < perBlockSlot)
        ? (max_signal_bytes & ~511ULL)
        : CopyOp::max_safe_chunk_size_for_slot(perBlockSlot);
    if (chunkSize == 0) {
      chunkSize = CopyOp::max_safe_chunk_size_for_slot(perBlockSlot);
    }
    const std::size_t chunkStride = CopyOp::worst_case_chunk_stride(chunkSize);
    if (chunkStride == 0 || chunkStride > perBlockSlot) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: compressed send perBlockSlot=%llu < chunkStride="
            "%llu (chunkSize=%llu). Increase the per-channel staging or reduce "
            "max_signal_bytes so each slot fits at least one worst-case-expanded "
            "sub-chunk.\n",
            (unsigned long long)perBlockSlot,
            (unsigned long long)chunkStride,
            (unsigned long long)chunkSize);
      }
      PIPES_DEVICE_TRAP();
    }
    const std::size_t chunksPerSlot = perBlockSlot / chunkStride;
    const std::size_t totalChunks = (nbytes + chunkSize - 1) / chunkSize;
    const std::size_t numSlots =
        (totalChunks + chunksPerSlot - 1) / chunksPerSlot;
    const std::size_t baseSlot =
        static_cast<std::size_t>(baseByte) / perBlockSlot;

    for (std::size_t s = 0; s < totalChunks; ++s) {
      const std::size_t slotIdx = s / chunksPerSlot;
      const std::size_t subStep = s % chunksPerSlot;
      const bool isLastInSlot =
          (subStep == chunksPerSlot - 1) || (s == totalChunks - 1);
      const std::size_t subStart =
          slotIdx * perBlockSlot + subStep * chunkStride;
      const std::size_t subEnd =
          isLastInSlot ? (slotIdx + 1) * perBlockSlot : subStart + chunkStride;
      const std::size_t protocolBytesThis = subEnd - subStart;
      const uint64_t protocolStreamEnd = baseByte + subEnd;

      const std::size_t absSlot = baseSlot + slotIdx;
      const int ringSlot = static_cast<int>(absSlot % pipelineDepth);
      const uint64_t pipelineCycle = absSlot / pipelineDepth;
      const std::size_t stagingOff =
          static_cast<std::size_t>(groupId) * pipelineBytes +
          static_cast<std::size_t>(ringSlot) * perBlockSlot +
          subStep * chunkStride;
      const std::size_t dataOff = s * chunkSize;
      const std::size_t bytesThis =
          (dataOff + chunkSize <= nbytes) ? chunkSize : (nbytes - dataOff);

      // (1) Wait for NIC to finish with this slot's local sendStaging.
      prepare_send_slot<Proto>(
          transport, group, ringSlot, pipelineCycle, timeout);

      // (2) Cooperative compress: src -> local sendStaging via CopyOp. The
      //     return value is the compressed byte count the leader uses to size
      //     the RDMA put.
      const std::size_t copyResult = CopyOp::send(
          channelLayout.sendStagingPtr + stagingOff,
          static_cast<const char*>(src) + dataOff,
          bytesThis,
          group,
          dataOff,
          args...);
      group.sync();

      // (3) Backpressure: wait for receiver to free this slot's recvStaging.
      if (protocolStreamEnd > pipelineBytes) {
        transport.wait_signal(
            group, localSlotFree, protocolStreamEnd - pipelineBytes, timeout);
      }

      // (4) Leader-only RDMA put with fused signal. The put length is the
      //     compressed size; DATA_READY advances by the reserved wire stride so
      //     the receiver's cumulative threshold is data-independent. The
      //     preceding group.sync() gives the happens-before so the leader's
      //     __threadfence_system() flushes every thread's compressed writes to
      //     system scope before the WQE is posted (only the leader pays the
      //     system fence).
      group.sync();
      if (group.is_leader()) {
        __threadfence_system();
        if (copyResult > chunkStride) {
          printf(
              "[PIPES] FATAL: compressed send copyResult=%llu > chunkStride="
              "%llu (chunkSize=%llu). Compressed sub-chunk exceeded its "
              "reserved worst-case slot region; refusing to truncate the RDMA "
              "put (would corrupt decompression).\n",
              (unsigned long long)copyResult,
              (unsigned long long)chunkStride,
              (unsigned long long)chunkSize);
          PIPES_DEVICE_TRAP();
        }
        ThreadGroup solo{
            0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
        const auto completion = transport.put(
            solo,
            channelLayout.sendStagingBuf.subBuffer(stagingOff),
            remoteChannel.recvStaging.subBuffer(stagingOff),
            copyResult,
            remoteChannel.dataReady,
            protocolBytesThis,
            /*counterBuf=*/{},
            /*counterVal=*/0,
            /*signalPerLane=*/true);
        record_send_completion<Proto>(
            transport,
            static_cast<uint32_t>(groupId),
            ringSlot,
            pipelineCycle,
            completion);
      }
      group.sync();
    }

    if (group.is_leader()) {
      state.nextStep = static_cast<int64_t>(baseByte + numSlots * perBlockSlot);
      state.activeStage = detail::IbSendRecvProgressStage::Done;
      state.activeBaseStep = 0;
      state.activeNextByte = 0;
      state.activeTailPadding = 0;
    }
    group.sync();
  } else {
    // The loop iterates in PAYLOAD bytes; physical staging offsets, RDMA
    // lengths, and signal/counter thresholds are derived in WIRE bytes via
    // Proto::wire_bytes() (1:1 for Simple, kPacketBytes:kData for LL). Tail
    // padding rides the final signal/counter credit only -- the RDMA write
    // covers valid payload/wire bytes.
    for (std::size_t dataOff = 0; dataOff < payloadProtocolBytes;) {
      const uint64_t streamPayload = payloadBaseByte + dataOff;
      const std::size_t pipelineOff =
          static_cast<std::size_t>(streamPayload % pipelineBytesPayload);
      const int slot = static_cast<int>(pipelineOff / perBlockSlotPayload);
      const std::size_t chunkOff = pipelineOff - slot * perBlockSlotPayload;
      const std::size_t slotRemaining = perBlockSlotPayload - chunkOff;
      const std::size_t dataRemaining = payloadProtocolBytes - dataOff;
      std::size_t payloadBytes =
          chunkPayload < dataRemaining ? chunkPayload : dataRemaining;
      payloadBytes =
          payloadBytes < slotRemaining ? payloadBytes : slotRemaining;
      const bool isFinalChunk = dataOff + payloadBytes >= payloadProtocolBytes;

      // Wire-space derivations (== payload for Simple: wire_bytes is identity).
      const std::size_t bytesThis = Proto::wire_bytes(payloadBytes);
      // Tail padding is a payload-space alignment credit; convert it to wire so
      // it matches the wire flow-control stream (streamWire =
      // wire_bytes(cursor)). Identity for Simple; kPacketBytes:kData for LL.
      const std::size_t protocolBytesThis = bytesThis +
          (isFinalChunk ? Proto::wire_bytes(protocolTailPadding) : 0);
      const std::size_t validBytes =
          valid_payload_bytes(dataOff, payloadBytes, nbytes);
      const std::size_t stagingOff = ch.stagingBase +
          static_cast<std::size_t>(slot) * perBlockSlotWire +
          Proto::wire_bytes(chunkOff);
      const uint64_t streamWire = Proto::wire_bytes(streamPayload);
      const uint64_t protocolStreamEnd = streamWire + protocolBytesThis;
      const uint64_t pipelineCycle = streamPayload / pipelineBytesPayload;
      // flagVal (a per-ring-pass counter) for this chunk's slot; LL stamps it
      // into every packet flag. Offset by +1 so the flag is never zero.
      const uint64_t flagVal = streamPayload / pipelineBytesPayload + 1;

      // (1) Wait for NIC to finish with this slot's local sendStaging.
      if (group.is_leader()) {
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceLocalCompletionWaitBegin,
            static_cast<uint8_t>(kPipesTraceQpLaneMask),
            bytesThis);
      }
      prepare_send_slot<Proto>(transport, group, slot, pipelineCycle, timeout);
      if (group.is_leader()) {
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceLocalCompletionWaitEnd,
            static_cast<uint8_t>(kPipesTraceQpLaneMask),
            bytesThis);
      }

      // (2) Cooperative copy: src -> local sendStaging via CopyOp.
      if (group.is_leader()) {
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceStageCopyBegin,
            static_cast<uint8_t>(kPipesTraceQpLaneMask),
            validBytes);
      }
      const SendSignal sig = prepareSendBuf<CopyOp>(
          Proto{},
          group,
          channelLayout.sendStagingPtr + stagingOff,
          static_cast<const char*>(src) + dataOff,
          payloadBytes,
          nbytes,
          dataOff,
          flagVal,
          remoteChannel,
          protocolBytesThis,
          args...);
      if (group.is_leader()) {
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceStageCopyEnd,
            static_cast<uint8_t>(kPipesTraceQpLaneMask),
            validBytes);
      }

      // (3) Backpressure: wait for receiver to free this byte range's
      //     recvStaging offset. Symmetric with DATA_READY.
      if (protocolStreamEnd > pipelineBytesWire) {
        if (group.is_leader()) {
          trace_allreduce_event(
              traceContext,
              PipesTraceEventType::kAllReduceRemoteSlotFreeWaitBegin,
              static_cast<uint8_t>(kPipesTraceQpLaneMask),
              protocolBytesThis);
        }
        transport.wait_signal(
            group,
            localSlotFree,
            protocolStreamEnd - pipelineBytesWire,
            timeout);
        if (group.is_leader()) {
          trace_allreduce_event(
              traceContext,
              PipesTraceEventType::kAllReduceRemoteSlotFreeWaitEnd,
              static_cast<uint8_t>(kPipesTraceQpLaneMask),
              protocolBytesThis);
        }
      }

      // (4) Leader-only single-WQE RDMA put with fused signal.
      if (group.is_leader()) {
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceSendSyncBegin,
            static_cast<uint8_t>(kPipesTraceQpLaneMask),
            bytesThis);
      }
      group.sync();
      if (group.is_leader()) {
        __threadfence_system();
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceSendSyncEnd,
            static_cast<uint8_t>(kPipesTraceQpLaneMask),
            bytesThis);
        const uint32_t numLanes = static_cast<uint32_t>(channelLayout.numLanes);
        const uint8_t qpLane = static_cast<uint8_t>(
            numLanes == 0 ? 0 : localChannel.sendQp.cursor % numLanes);
        ThreadGroup solo{
            0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceWqeSubmitBegin,
            qpLane,
            bytesThis);
        const auto completion = transport.put(
            solo,
            channelLayout.sendStagingBuf.subBuffer(stagingOff),
            remoteChannel.recvStaging.subBuffer(stagingOff),
            bytesThis,
            sig.buf,
            sig.val,
            /*counterBuf=*/{},
            /*counterVal=*/0,
            /*signalPerLane=*/true);
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceWqeSubmitEnd,
            qpLane,
            bytesThis);
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceBookkeepingBegin,
            qpLane,
            protocolBytesThis);
        record_send_completion<Proto>(
            transport,
            static_cast<uint32_t>(groupId),
            slot,
            pipelineCycle,
            completion);
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceBookkeepingEnd,
            qpLane,
            protocolBytesThis);
      }
      group.sync();
      dataOff += payloadBytes;
    }

    if (group.is_leader()) {
      state.nextStep = static_cast<int64_t>(baseByte + protocolBytes);
      state.activeStage = detail::IbSendRecvProgressStage::Done;
      state.activeBaseStep = 0;
      state.activeNextByte = 0;
      state.activeTailPadding = 0;
    }
    group.sync();
  }
#endif
}

template <
    typename Transport,
    typename CopyOp = Memcpy,
    typename Proto = protocol::Simple,
    typename... Args>
__device__ __forceinline__ void send(
    Transport& transport,
    ThreadGroup& group,
    const void* __restrict__ src,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout(),
    Args... args) {
  send_impl<Transport, CopyOp, Proto>(
      transport,
      group,
      src,
      nbytes,
      max_signal_bytes,
      timeout,
      nullptr,
      args...);
}

template <
    typename Transport,
    typename CopyOp = Memcpy,
    typename Proto = protocol::Simple,
    typename... Args>
__device__ __forceinline__ void send_with_fine_trace(
    Transport& transport,
    ThreadGroup& group,
    const void* __restrict__ src,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    const PipesTraceAllReduceContext& traceContext,
    Args... args) {
  send_impl<Transport, CopyOp, Proto>(
      transport,
      group,
      src,
      nbytes,
      max_signal_bytes,
      timeout,
      &traceContext,
      args...);
}

/**
 * recv — receive one block's tile from pipelined RDMA.
 *
 * Waits for data to arrive in recvStaging, then copies recvStaging -> dst.
 * For this call, each logical slot contributes one perBlockSlot-sized region
 * for this group. If nbytes > perBlockSlot, recv() advances through multiple
 * ring positions. max_signal_bytes controls sub-chunk granularity and must
 * match the sender.
 *
 * Signaling protocol (per group, symmetric with send):
 *   DATA_READY — sender increments by bytesThis after RDMA put completes.
 *                recv waits on this before copying from recvStaging.
 *   SLOT_FREE  — recv increments by bytesThis (symmetric with DATA_READY)
 *                to release backpressure on sender.
 *
 * @param transport       Owning transport used for every transport op.
 * @param group           ThreadGroup (all threads participate in memcpy,
 *                        leader does signal ops).
 * @param dst             Destination for this block's tile.
 * @param nbytes          Bytes to receive for this group. Internally
 *                        consumed in perBlockSlot-sized pieces, or smaller
 *                        sub-chunks when max_signal_bytes is set.
 * @param max_signal_bytes Max bytes per signaled sub-chunk within one
 *                        perBlockSlot. 0 means one signal per perBlockSlot.
 *                        Must match the sender's value.
 * @param timeout         Optional timeout for wait operations.
 */
template <
    typename Transport,
    typename CopyOp = Memcpy,
    typename Proto = protocol::Simple,
    typename... Args>
__device__ __forceinline__ void recv_impl(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout(),
    const PipesTraceAllReduceContext* traceContext = nullptr,
    Args... args) {
  // The variable-size (compressed) loop below keeps its encode inline rather
  // than behind the prepareSendBuf/consumeRecvBuf seam, so it is Simple-shaped
  // and a non-default protocol would silently drive the wrong encode. Forbid
  // the pairing; there is no LL-over-compressed use case today.
  static_assert(
      !detail::copyop_variable_size_v<CopyOp> ||
          std::is_same_v<Proto, protocol::Simple>,
      "variable-size CopyOps (e.g. AnsCompress) are supported on "
      "protocol::Simple only; the compressed loop is not behind the "
      "prepareSendBuf/consumeRecvBuf seam.");
#if !PIPES_IS_DEVICE_COMPILE
  (void)transport;
  (void)group;
  (void)dst;
  (void)nbytes;
  (void)max_signal_bytes;
  (void)timeout;
  (void)traceContext;
#else
  if (nbytes == 0) {
    return;
  }
  auto& channelLayout = transport.channel_layout();
  const SendRecvGeometry geometry =
      calcGeometry(Proto{}, channelLayout, group, nbytes, max_signal_bytes);
  const int groupId = geometry.groupId;
  const std::size_t perBlockSlotWire = geometry.perBlockSlotWire;
  const std::size_t perBlockSlotPayload = geometry.perBlockSlotPayload;
  [[maybe_unused]] const std::size_t chunkPayload = geometry.chunkPayload;
  [[maybe_unused]] const std::size_t pipelineBytesPayload =
      geometry.pipelineBytesPayload;
  const std::size_t pipelineBytesWire = geometry.pipelineBytesWire;
  [[maybe_unused]] const std::size_t payloadProtocolBytes =
      geometry.payloadProtocolBytes;

  auto& state = progress_recv_slot<Proto>(transport, group);
  const ChannelSlotView ch =
      acquire_channel<Proto>(transport, channelLayout, group);
  IbLocalChannel& localChannel = ch.channel;
  const IbgdaLocalBuffer localDataReady = ch.local.dataReady;
  const IbRemoteChannel remoteChannel = ch.remote;
  assert_progress_slot_idle(group, state, "recv");
  const uint64_t baseByte = static_cast<uint64_t>(state.nextStep);
  // Category of the previous op on this slot, read before the leader overwrites
  // it below. Used only to make the compressed slot-alignment trap precise
  // about a fixed->variable transition (byte cursor vs sub-chunk cursor).
  [[maybe_unused]] const bool prevOpVariableSize = state.activeVariableSize;
  const std::size_t protocolTailPadding = tail_padding_for_signal_granularity(
      baseByte, max_signal_bytes, perBlockSlotPayload, nbytes);
  [[maybe_unused]] const uint64_t payloadBaseByte = baseByte;
  [[maybe_unused]] const std::size_t protocolBytes =
      payloadProtocolBytes + protocolTailPadding;
  if (group.is_leader()) {
    // Record this op's CopyOp category in the persistent slot state so the
    // cross-category contract is explicit (see IbChannelProgress). Safe to
    // switch categories between ops: nextStep is a slot-pinned wire-byte
    // cursor.
    state.activeVariableSize = detail::copyop_variable_size_v<CopyOp>;
    state.activeStage = detail::IbSendRecvProgressStage::Busy;
    state.activeBaseStep = static_cast<int64_t>(baseByte);
    state.activeNextByte = 0;
    state.activeTailPadding = protocolTailPadding;
  }

  if constexpr (detail::copyop_variable_size_v<CopyOp>) {
    // Variable-size (compressed) recv, mirror of send(): the staging ring
    // reserves a fixed worst-case region per sub-chunk while the CopyOp
    // decompresses only the bytes that arrived (AnsCompress reads its own
    // in-staging size header). Flow control runs in WIRE bytes with the same
    // gap-carried, slot-granular cursor as the sender, so DATA_READY/SLOT_FREE
    // thresholds match the sender's exactly.
    //
    // Cross-category staging-reuse contract (shared physical ring): mirror of
    // send(). Reuse of a recvStaging slot across a fixed<->variable switch is
    // serialized by the same cumulative DATA_READY/SLOT_FREE handshake on the
    // shared wire-byte cursor -- the sender never overwrites a slot the
    // receiver has not yet released -- so no separate cross-category drain is
    // needed.
    const std::size_t perBlockSlot = perBlockSlotWire;
    const int pipelineDepth = channelLayout.pipelineDepth;
    if ((perBlockSlot & 511ULL) != 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: compressed recv perBlockSlot=%llu not 512-aligned; "
            "size the per-channel staging so each block's slot is a multiple of "
            "the NIC burst alignment.\n",
            (unsigned long long)perBlockSlot);
      }
      PIPES_DEVICE_TRAP();
    }
    // `signal_alignment()` pins the persistent per-channel cursor to whole
    // perBlockSlot strides for both plain and compressed ops, so this is always
    // aligned in practice. Kept as a defensive invariant check (mirror of the
    // send-side guard above).
    if ((baseByte % perBlockSlot) != 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: compressed recv requires a slot-aligned start "
            "cursor (baseByte=%llu, perBlockSlot=%llu, prevOpVariableSize=%d); a "
            "preceding op left the per-channel cursor mid-slot, which would "
            "reinterpret its byte cursor as a compressed sub-chunk cursor.\n",
            (unsigned long long)baseByte,
            (unsigned long long)perBlockSlot,
            (int)prevOpVariableSize);
      }
      PIPES_DEVICE_TRAP();
    }
    // For the 0 sentinel (or an over-large request) pick the largest chunk
    // whose worst-case ANS-expanded staging still fits one perBlockSlot, via
    // the policy's max_safe_chunk_size_for_slot(). Using perBlockSlot directly
    // would make worst_case_chunk_stride() exceed the slot and trap, since a
    // compressed sub-chunk's worst case is ~1.3x its uncompressed input. send()
    // and recv() derive the identical value from the shared perBlockSlot, so
    // the two sides agree on the chunking without any exchange.
    std::size_t chunkSize =
        (max_signal_bytes > 0 && max_signal_bytes < perBlockSlot)
        ? (max_signal_bytes & ~511ULL)
        : CopyOp::max_safe_chunk_size_for_slot(perBlockSlot);
    if (chunkSize == 0) {
      chunkSize = CopyOp::max_safe_chunk_size_for_slot(perBlockSlot);
    }
    const std::size_t chunkStride = CopyOp::worst_case_chunk_stride(chunkSize);
    if (chunkStride == 0 || chunkStride > perBlockSlot) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: compressed recv perBlockSlot=%llu < chunkStride="
            "%llu (chunkSize=%llu).\n",
            (unsigned long long)perBlockSlot,
            (unsigned long long)chunkStride,
            (unsigned long long)chunkSize);
      }
      PIPES_DEVICE_TRAP();
    }
    const std::size_t chunksPerSlot = perBlockSlot / chunkStride;
    const std::size_t totalChunks = (nbytes + chunkSize - 1) / chunkSize;
    const std::size_t numSlots =
        (totalChunks + chunksPerSlot - 1) / chunksPerSlot;
    const std::size_t baseSlot =
        static_cast<std::size_t>(baseByte) / perBlockSlot;

    for (std::size_t s = 0; s < totalChunks; ++s) {
      const std::size_t slotIdx = s / chunksPerSlot;
      const std::size_t subStep = s % chunksPerSlot;
      const bool isLastInSlot =
          (subStep == chunksPerSlot - 1) || (s == totalChunks - 1);
      const std::size_t subStart =
          slotIdx * perBlockSlot + subStep * chunkStride;
      const std::size_t subEnd =
          isLastInSlot ? (slotIdx + 1) * perBlockSlot : subStart + chunkStride;
      const std::size_t protocolBytesThis = subEnd - subStart;

      const std::size_t absSlot = baseSlot + slotIdx;
      const int ringSlot = static_cast<int>(absSlot % pipelineDepth);
      const std::size_t stagingOff =
          static_cast<std::size_t>(groupId) * pipelineBytesWire +
          static_cast<std::size_t>(ringSlot) * perBlockSlot +
          subStep * chunkStride;
      const std::size_t dataOff = s * chunkSize;
      const std::size_t bytesThis =
          (dataOff + chunkSize <= nbytes) ? chunkSize : (nbytes - dataOff);

      // (1) Wait for sender's DATA_READY (reserved wire stride, gap-carried).
      wait_recv_data_ready(
          transport,
          group,
          localChannel,
          localDataReady,
          protocolBytesThis,
          timeout);

      // (2) Cooperative decompress: local recvStaging -> dst via CopyOp.
      CopyOp::recv(
          static_cast<char*>(dst) + dataOff,
          channelLayout.recvStagingPtr + stagingOff,
          bytesThis,
          group,
          dataOff,
          args...);
      group.sync();

      // (3) Signal SLOT_FREE to sender (same reserved wire stride).
      transport.signal(
          group, remoteChannel.slotFree, protocolBytesThis, IbDirection::Recv);
    }

    if (group.is_leader()) {
      state.nextStep = static_cast<int64_t>(baseByte + numSlots * perBlockSlot);
      state.activeStage = detail::IbSendRecvProgressStage::Done;
      state.activeBaseStep = 0;
      state.activeNextByte = 0;
      state.activeTailPadding = 0;
    }
    group.sync();
  } else {
    // Payload-space iteration; wire-space staging/threshold derivations via
    // Proto::wire_bytes() (identity for Simple). Tail padding rides the final
    // SLOT_FREE credit only; the recv copies valid payload bytes.
    for (std::size_t dataOff = 0; dataOff < payloadProtocolBytes;) {
      const uint64_t streamPayload = payloadBaseByte + dataOff;
      const std::size_t pipelineOff =
          static_cast<std::size_t>(streamPayload % pipelineBytesPayload);
      const int slot = static_cast<int>(pipelineOff / perBlockSlotPayload);
      const std::size_t chunkOff = pipelineOff - slot * perBlockSlotPayload;
      const std::size_t slotRemaining = perBlockSlotPayload - chunkOff;
      const std::size_t dataRemaining = payloadProtocolBytes - dataOff;
      std::size_t payloadBytes =
          chunkPayload < dataRemaining ? chunkPayload : dataRemaining;
      payloadBytes =
          payloadBytes < slotRemaining ? payloadBytes : slotRemaining;
      const bool isFinalChunk = dataOff + payloadBytes >= payloadProtocolBytes;

      const std::size_t bytesThis = Proto::wire_bytes(payloadBytes);
      // Tail padding is a payload-space alignment credit; convert it to wire so
      // it matches the wire flow-control stream (streamWire =
      // wire_bytes(cursor)). Identity for Simple; kPacketBytes:kData for LL.
      const std::size_t protocolBytesThis = bytesThis +
          (isFinalChunk ? Proto::wire_bytes(protocolTailPadding) : 0);
      const std::size_t stagingOff = ch.stagingBase +
          static_cast<std::size_t>(slot) * perBlockSlotWire +
          Proto::wire_bytes(chunkOff);
      // flagVal (a per-ring-pass counter) for this chunk's slot; LL stamps it
      // into every packet flag.
      const uint64_t flagVal = streamPayload / pipelineBytesPayload + 1;

      // (1)+(2) Wait for the chunk to be ready (DATA_READY signal or, for LL,
      //         the inline flag) and cooperatively copy recvStaging -> dst.
      const uint32_t numLanes = static_cast<uint32_t>(channelLayout.numLanes);
      const uint8_t qpLane = static_cast<uint8_t>(
          numLanes == 0 ? 0 : localChannel.recvDataReadyLaneCursor % numLanes);
      consumeRecvBuf<CopyOp>(
          Proto{},
          transport,
          group,
          localChannel,
          localDataReady,
          static_cast<char*>(dst) + dataOff,
          channelLayout.recvStagingPtr + stagingOff,
          payloadBytes,
          nbytes,
          dataOff,
          flagVal,
          protocolBytesThis,
          timeout,
          traceContext,
          args...);

      if (group.is_leader()) {
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceBookkeepingBegin,
            qpLane,
            protocolBytesThis);
      }
      transport.signal(
          group, remoteChannel.slotFree, protocolBytesThis, IbDirection::Recv);
      dataOff += payloadBytes;
      if (group.is_leader()) {
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceBookkeepingEnd,
            qpLane,
            protocolBytesThis);
      }
    }

    if (group.is_leader()) {
      state.nextStep = static_cast<int64_t>(baseByte + protocolBytes);
      state.activeStage = detail::IbSendRecvProgressStage::Done;
      state.activeBaseStep = 0;
      state.activeNextByte = 0;
      state.activeTailPadding = 0;
    }
    group.sync();
  }
#endif
}

template <
    typename Transport,
    typename CopyOp = Memcpy,
    typename Proto = protocol::Simple,
    typename... Args>
__device__ __forceinline__ void recv(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout(),
    Args... args) {
  recv_impl<Transport, CopyOp, Proto>(
      transport,
      group,
      dst,
      nbytes,
      max_signal_bytes,
      timeout,
      nullptr,
      args...);
}

template <
    typename Transport,
    typename CopyOp = Memcpy,
    typename Proto = protocol::Simple,
    typename... Args>
__device__ __forceinline__ void recv_with_fine_trace(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    const PipesTraceAllReduceContext& traceContext,
    Args... args) {
  recv_impl<Transport, CopyOp, Proto>(
      transport,
      group,
      dst,
      nbytes,
      max_signal_bytes,
      timeout,
      &traceContext,
      args...);
}

/**
 * forward — receive data and forward it to the next peer in a ring.
 *
 * Combines recv + send in a single method, sharing the staging buffer to
 * avoid an extra copy. The CopyOp::forward() method receives three
 * buffers: dst (application output), fwd_staging (next peer's send staging),
 * and staging (this transport's recv staging). This enables fused
 * receive-reduce-forward patterns.
 *
 * Signal ordering invariant (critical for ring deadlock avoidance):
 *   1. Wait DATA_READY from sender (this transport)
 *   2. Wait for local completion on fwd transport's sendStaging
 *   3. CopyOp::forward(dst, fwd_staging, staging, ...)
 *   4. Signal SLOT_FREE to sender (this transport) — BEFORE step 5
 *   5. Wait SLOT_FREE from fwd transport's receiver
 *   6. threadfence_system + RDMA put via fwd transport
 *
 * Step 4 before step 5 breaks the circular dependency in rings: each rank
 * releases its predecessor's staging before waiting on its successor.
 *
 * Protocol compatibility with send() and recv():
 *
 * forward acts as a recv on "this" transport and a send on "fwd".
 * The signal protocol is wire-compatible:
 *
 *   Recv side (this transport):
 *     - Uses this channel's recv progress cursor.
 *     - Waits DATA_READY on this channel's local data-ready signal.
 *     - Signals SLOT_FREE on the remote channel's slot-free signal.
 *
 *   Fwd side (fwd transport):
 *     - Uses the forward channel's send progress cursor.
 *     - Waits on the forward channel's local-completion ticket.
 *     - Waits SLOT_FREE on the forward channel's local slot-free signal.
 *     - RDMA puts with DATA_READY on the forward remote channel and
 *       returns a ticket covering local completion of the data put.
 *
 * Any chain of send → forward* → recv is therefore valid: each
 * forward consumes exactly the signals its predecessor produces
 * and produces exactly the signals its successor expects.
 *
 * @param transport       Recv-side transport (this peer's receiver).
 * @param group           ThreadGroup (all threads participate).
 * @param dst             Application destination (may be nullptr if
 *                        CopyOp handles it, e.g. reduce-scatter).
 * @param fwdTransport    Forward transport (sends to next peer in ring).
 * @param nbytes          Bytes to receive and forward.
 * @param max_signal_bytes Max bytes per signaled sub-chunk. 0 =
 * perBlockSlot.
 * @param timeout         Optional timeout for wait operations.
 * @param args            Extra args forwarded to CopyOp::forward.
 */
template <
    typename CopyOp = Memcpy,
    typename Transport,
    typename Proto = protocol::Simple,
    typename... Args>
__device__ __forceinline__ void forward_impl(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    Transport& fwdTransport,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout(),
    const PipesTraceAllReduceContext* recvTraceContext = nullptr,
    const PipesTraceAllReduceContext* sendTraceContext = nullptr,
    Args... args) {
#if PIPES_IS_DEVICE_COMPILE
#ifdef __HIP_PLATFORM_AMD__
  static_assert(
      sizeof(CopyOp) == 0,
      "detail::forward() requires NVIDIA GPU (DOCA/IBGDA)");
#endif
  if (nbytes == 0) {
    return;
  }
  auto& channelLayout = transport.channel_layout();
  auto& fwdChannelLayout = fwdTransport.channel_layout();
  const int groupId = group.group_id;

  // Per-side payload/wire geometry (Proto-tagged). calcGeometry validates
  // groupId < maxChannels and perBlockSlot != 0 for each channel. The loop
  // iterates in PAYLOAD bytes; staging offsets, RDMA lengths, and
  // signal/counter thresholds are derived in WIRE bytes via Proto::wire_bytes()
  // (identity for Simple; kPacketBytes:kData for LL).
  const SendRecvGeometry recvGeo =
      calcGeometry(Proto{}, channelLayout, group, nbytes, max_signal_bytes);
  const SendRecvGeometry fwdGeo =
      calcGeometry(Proto{}, fwdChannelLayout, group, nbytes, max_signal_bytes);
  const std::size_t payloadProtocolBytes = recvGeo.payloadProtocolBytes;

  // --- recv side (this transport) ---
  auto& recvSlotState = progress_recv_slot<Proto>(transport, group);
  const ChannelSlotView recvCh =
      acquire_channel<Proto>(transport, channelLayout, group);
  IbLocalChannel& recvLocalChannel = recvCh.channel;
  const IbgdaLocalBuffer recvDataReady = recvCh.local.dataReady;
  const IbRemoteChannel recvRemoteChannel = recvCh.remote;
  assert_progress_slot_idle(group, recvSlotState, "forward recv");
  const uint64_t recvBaseByte = static_cast<uint64_t>(recvSlotState.nextStep);
  const std::size_t recvProtocolTailPadding =
      tail_padding_for_signal_granularity(
          recvBaseByte, max_signal_bytes, recvGeo.perBlockSlotPayload, nbytes);
  const uint64_t recvPayloadBaseByte = recvBaseByte;
  const std::size_t recvProtocolBytes =
      payloadProtocolBytes + recvProtocolTailPadding;

  // --- fwd side (fwd transport) ---
  auto& fwdSlotState = progress_send_slot<Proto>(fwdTransport, group);
  const ChannelSlotView fwdCh =
      acquire_channel<Proto>(fwdTransport, fwdChannelLayout, group);
  IbLocalChannel& fwdLocalChannel = fwdCh.channel;
  const IbgdaLocalBuffer fwdSlotFree = fwdCh.local.slotFree;
  const IbRemoteChannel fwdRemoteChannel = fwdCh.remote;
  assert_progress_slot_idle(group, fwdSlotState, "forward send");
  const uint64_t fwdBaseByte = static_cast<uint64_t>(fwdSlotState.nextStep);
  const std::size_t fwdProtocolTailPadding =
      tail_padding_for_signal_granularity(
          fwdBaseByte, max_signal_bytes, fwdGeo.perBlockSlotPayload, nbytes);
  const uint64_t fwdPayloadBaseByte = fwdBaseByte;
  const std::size_t fwdProtocolBytes =
      payloadProtocolBytes + fwdProtocolTailPadding;
  if (group.is_leader()) {
    recvSlotState.activeStage = detail::IbSendRecvProgressStage::Busy;
    recvSlotState.activeBaseStep = static_cast<int64_t>(recvBaseByte);
    recvSlotState.activeNextByte = 0;
    recvSlotState.activeTailPadding = recvProtocolTailPadding;
    fwdSlotState.activeStage = detail::IbSendRecvProgressStage::Busy;
    fwdSlotState.activeBaseStep = static_cast<int64_t>(fwdBaseByte);
    fwdSlotState.activeNextByte = 0;
    fwdSlotState.activeTailPadding = fwdProtocolTailPadding;
    trace_allreduce_event(
        sendTraceContext,
        PipesTraceEventType::kAllReducePathStaged,
        static_cast<uint8_t>(kPipesTraceQpLaneMask));
  }

  for (std::size_t dataOff = 0; dataOff < payloadProtocolBytes;) {
    // --- Recv side offsets (ring math in PAYLOAD, physical offset in WIRE) ---
    const uint64_t recvStreamPayload = recvPayloadBaseByte + dataOff;
    const std::size_t recvPipelineOff = static_cast<std::size_t>(
        recvStreamPayload % recvGeo.pipelineBytesPayload);
    const int recvSlot =
        static_cast<int>(recvPipelineOff / recvGeo.perBlockSlotPayload);
    const std::size_t recvChunkOff =
        recvPipelineOff - recvSlot * recvGeo.perBlockSlotPayload;
    const std::size_t recvStagingOff = recvCh.stagingBase +
        static_cast<std::size_t>(recvSlot) * recvGeo.perBlockSlotWire +
        Proto::wire_bytes(recvChunkOff);
    const std::size_t recvSlotRemaining =
        recvGeo.perBlockSlotPayload - recvChunkOff;

    // --- Fwd side offsets ---
    const uint64_t fwdStreamPayload = fwdPayloadBaseByte + dataOff;
    const std::size_t fwdPipelineOff = static_cast<std::size_t>(
        fwdStreamPayload % fwdGeo.pipelineBytesPayload);
    const int fwdSlot =
        static_cast<int>(fwdPipelineOff / fwdGeo.perBlockSlotPayload);
    const std::size_t fwdChunkOff =
        fwdPipelineOff - fwdSlot * fwdGeo.perBlockSlotPayload;
    const std::size_t fwdStagingOff = fwdCh.stagingBase +
        static_cast<std::size_t>(fwdSlot) * fwdGeo.perBlockSlotWire +
        Proto::wire_bytes(fwdChunkOff);
    const std::size_t fwdSlotRemaining =
        fwdGeo.perBlockSlotPayload - fwdChunkOff;

    // --- Chunk sizing in PAYLOAD; wire lengths via Proto::wire_bytes() ---
    const std::size_t dataRemaining = payloadProtocolBytes - dataOff;
    std::size_t payloadBytes = recvGeo.chunkPayload < fwdGeo.chunkPayload
        ? recvGeo.chunkPayload
        : fwdGeo.chunkPayload;
    payloadBytes = payloadBytes < dataRemaining ? payloadBytes : dataRemaining;
    payloadBytes =
        payloadBytes < recvSlotRemaining ? payloadBytes : recvSlotRemaining;
    payloadBytes =
        payloadBytes < fwdSlotRemaining ? payloadBytes : fwdSlotRemaining;
    const bool isFinalChunk = dataOff + payloadBytes >= payloadProtocolBytes;
    const std::size_t bytesThis = Proto::wire_bytes(payloadBytes);
    const std::size_t recvProtocolBytesThis = bytesThis +
        (isFinalChunk ? Proto::wire_bytes(recvProtocolTailPadding) : 0);
    const std::size_t fwdProtocolBytesThis = bytesThis +
        (isFinalChunk ? Proto::wire_bytes(fwdProtocolTailPadding) : 0);
    const uint64_t fwdStreamWire = Proto::wire_bytes(fwdStreamPayload);
    const uint64_t fwdProtocolStreamEnd = fwdStreamWire + fwdProtocolBytesThis;
    const uint64_t fwdPipelineCycle =
        fwdStreamPayload / fwdGeo.pipelineBytesPayload;

    // (1) prepareForwardBuf: fwd slot-reuse backpressure + recv-side readiness
    //     + fused transform recvStaging -> dst + fwdStaging, returning the
    //     relay SendSignal for the put. Tag-dispatched on Proto.
    const SendSignal sig = prepareForwardBuf<CopyOp>(
        Proto{},
        transport,
        fwdTransport,
        group,
        recvLocalChannel,
        recvDataReady,
        dst ? static_cast<char*>(dst) + dataOff : nullptr,
        fwdChannelLayout.sendStagingPtr + fwdStagingOff,
        channelLayout.recvStagingPtr + recvStagingOff,
        payloadBytes,
        nbytes,
        dataOff,
        recvProtocolBytesThis,
        fwdRemoteChannel,
        fwdProtocolBytesThis,
        static_cast<uint32_t>(fwdSlot),
        fwdPipelineCycle,
        timeout,
        recvTraceContext,
        sendTraceContext,
        args...);

    transport.signal(
        group,
        recvRemoteChannel.slotFree,
        recvProtocolBytesThis,
        IbDirection::Recv);

    // (5) Wait for fwd receiver's SLOT_FREE (backpressure on fwd's
    //     recvStaging).
    if (fwdProtocolStreamEnd > fwdGeo.pipelineBytesWire) {
      if (group.is_leader()) {
        trace_allreduce_event(
            sendTraceContext,
            PipesTraceEventType::kAllReduceRemoteSlotFreeWaitBegin,
            static_cast<uint8_t>(kPipesTraceQpLaneMask),
            fwdProtocolBytesThis);
      }
      fwdTransport.wait_signal(
          group,
          fwdSlotFree,
          fwdProtocolStreamEnd - fwdGeo.pipelineBytesWire,
          timeout);
      if (group.is_leader()) {
        trace_allreduce_event(
            sendTraceContext,
            PipesTraceEventType::kAllReduceRemoteSlotFreeWaitEnd,
            static_cast<uint8_t>(kPipesTraceQpLaneMask),
            fwdProtocolBytesThis);
      }
    }

    // (6) Leader-only RDMA put via the forwarding transport.
    if (group.is_leader()) {
      trace_allreduce_event(
          sendTraceContext,
          PipesTraceEventType::kAllReduceSendSyncBegin,
          static_cast<uint8_t>(kPipesTraceQpLaneMask),
          bytesThis);
    }
    group.sync();
    if (group.is_leader()) {
      __threadfence_system();
      trace_allreduce_event(
          sendTraceContext,
          PipesTraceEventType::kAllReduceSendSyncEnd,
          static_cast<uint8_t>(kPipesTraceQpLaneMask),
          bytesThis);
      const uint32_t numLanes =
          static_cast<uint32_t>(fwdChannelLayout.numLanes);
      const uint8_t qpLane = static_cast<uint8_t>(
          numLanes == 0 ? 0 : fwdLocalChannel.sendQp.cursor % numLanes);
      ThreadGroup solo{
          0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
      trace_allreduce_event(
          sendTraceContext,
          PipesTraceEventType::kAllReduceWqeSubmitBegin,
          qpLane,
          bytesThis);
      const auto completion = fwdTransport.put(
          solo,
          fwdChannelLayout.sendStagingBuf.subBuffer(fwdStagingOff),
          fwdRemoteChannel.recvStaging.subBuffer(fwdStagingOff),
          bytesThis,
          sig.buf,
          sig.val,
          /*counterBuf=*/{},
          /*counterVal=*/0,
          /*signalPerLane=*/true);
      trace_allreduce_event(
          sendTraceContext,
          PipesTraceEventType::kAllReduceWqeSubmitEnd,
          qpLane,
          bytesThis);
      trace_allreduce_event(
          sendTraceContext,
          PipesTraceEventType::kAllReduceBookkeepingBegin,
          qpLane,
          fwdProtocolBytesThis);
      record_send_completion<Proto>(
          fwdTransport,
          static_cast<uint32_t>(groupId),
          fwdSlot,
          fwdPipelineCycle,
          completion);
      trace_allreduce_event(
          sendTraceContext,
          PipesTraceEventType::kAllReduceBookkeepingEnd,
          qpLane,
          fwdProtocolBytesThis);
    }
    group.sync();
    dataOff += payloadBytes;
  }

  // Update shared byte cursors for both recv and fwd sides.
  if (group.is_leader()) {
    recvSlotState.nextStep =
        static_cast<int64_t>(recvBaseByte + recvProtocolBytes);
    recvSlotState.activeStage = detail::IbSendRecvProgressStage::Done;
    recvSlotState.activeBaseStep = 0;
    recvSlotState.activeNextByte = 0;
    recvSlotState.activeTailPadding = 0;
    fwdSlotState.nextStep =
        static_cast<int64_t>(fwdBaseByte + fwdProtocolBytes);
    fwdSlotState.activeStage = detail::IbSendRecvProgressStage::Done;
    fwdSlotState.activeBaseStep = 0;
    fwdSlotState.activeNextByte = 0;
    fwdSlotState.activeTailPadding = 0;
  }
  group.sync();
#else
  (void)transport;
  (void)group;
  (void)dst;
  (void)fwdTransport;
  (void)nbytes;
  (void)max_signal_bytes;
  (void)timeout;
  (void)recvTraceContext;
  (void)sendTraceContext;
#endif
}

template <
    typename CopyOp = Memcpy,
    typename Transport,
    typename Proto = protocol::Simple,
    typename... Args>
__device__ __forceinline__ void forward(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    Transport& fwdTransport,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout(),
    Args... args) {
  forward_impl<CopyOp, Transport, Proto>(
      transport,
      group,
      dst,
      fwdTransport,
      nbytes,
      max_signal_bytes,
      timeout,
      nullptr,
      nullptr,
      args...);
}

template <
    typename CopyOp = Memcpy,
    typename Transport,
    typename Proto = protocol::Simple,
    typename... Args>
__device__ __forceinline__ void forward_with_fine_trace(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    Transport& fwdTransport,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    const PipesTraceAllReduceContext& recvTraceContext,
    const PipesTraceAllReduceContext& sendTraceContext,
    Args... args) {
  forward_impl<CopyOp, Transport, Proto>(
      transport,
      group,
      dst,
      fwdTransport,
      nbytes,
      max_signal_bytes,
      timeout,
      &recvTraceContext,
      &sendTraceContext,
      args...);
}

/**
 * Maximum bytes one channel can send without blocking on pipeline backpressure.
 */
__device__ __forceinline__ std::size_t pipeline_window(
    const IbChannelLayout& channelLayout) {
  return channelLayout.perChannelBufferSize != 0
      ? channelLayout.perChannelBufferSize
      : channelLayout.perChannelSize;
}

__device__ __forceinline__ std::size_t pipeline_chunk(
    const IbChannelLayout& channelLayout) {
  if (channelLayout.pipelineDepth <= 0) {
    return 0;
  }
  return pipeline_window(channelLayout) /
      static_cast<std::size_t>(channelLayout.pipelineDepth);
}

/**
 * Stateful send/recv cursors advance in 16-byte protocol quanta. That keeps
 * the staging stream on the same granularity as the vectorized local staging
 * copies while preserving caller-facing payload byte counts; padding is
 * transport-private and is never exposed to CopyOp callbacks.
 */
__device__ __forceinline__ static std::size_t align_protocol_bytes(
    std::size_t nbytes) {
  return (nbytes + 15ULL) & ~15ULL;
}

__device__ __forceinline__ static uint64_t round_up_to_multiple(
    uint64_t value,
    std::size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const uint64_t alignment64 = static_cast<uint64_t>(alignment);
  return ((value + alignment64 - 1) / alignment64) * alignment64;
}

// Granularity to which the persistent per-channel cursor advances BETWEEN
// operations (via tail_padding_for_signal_granularity). This is intentionally
// ALWAYS `perBlockSlot`, independent of `maxSignalBytes`.
//
// Rationale: a channel's stateful cursor (IbChannelProgress::nextStep) is
// shared by every op on that channel, plain OR compressed. The compressed
// (variable-size CopyOp) send()/recv() path requires a slot-aligned start
// cursor -- it lays sub-chunks out at `slotIdx*perBlockSlot +
// subStep*chunkStride` and nvcompdx needs 512-byte-aligned staging, both of
// which only hold when each slot begins exactly on a perBlockSlot boundary
// (it traps via `baseByte % perBlockSlot != 0` otherwise). If the plain path
// were allowed to round the cursor to a sub-slot (maxSignalBytes) boundary --
// which in general does NOT divide perBlockSlot -- a later compressed op on
// the same channel would start mid-slot and trap. So the cursor stride is
// pinned to whole slots for both paths.
//
// `maxSignalBytes` still controls INTRA-transfer signaling granularity (the
// per-signaled-chunk size) via calcGeometry()'s `chunkPayload`; it just must
// not drive the between-op cursor stride. Kept as a parameter so the signature
// and all call sites are unchanged.
__device__ __forceinline__ static std::size_t signal_alignment(
    [[maybe_unused]] std::size_t maxSignalBytes,
    std::size_t perBlockSlot) {
  return perBlockSlot;
}

/**
 * Pad the current operation's protocol byte stream to the signaling boundary.
 *
 * Padding is credit-only: payload copies and RDMA writes still cover only
 * aligned payload protocol bytes. The final DATA_READY/SLOT_FREE update carries
 * this tail padding so the next operation starts on an aligned protocol cursor
 * without needing a future recv to publish padding credit.
 */
__device__ __forceinline__ static std::size_t
tail_padding_for_signal_granularity(
    uint64_t baseByte,
    std::size_t maxSignalBytes,
    std::size_t perBlockSlot,
    std::size_t payloadBytes) {
  const std::size_t alignment = signal_alignment(maxSignalBytes, perBlockSlot);
  if (alignment == 0) {
    return 0;
  }
  const uint64_t payloadEnd = baseByte + align_protocol_bytes(payloadBytes);
  return static_cast<std::size_t>(
      round_up_to_multiple(payloadEnd, alignment) - payloadEnd);
}

__device__ __forceinline__ static std::size_t valid_payload_bytes(
    std::size_t byteOffset,
    std::size_t chunkBytes,
    std::size_t payloadBytes) {
  if (byteOffset >= payloadBytes) {
    return 0;
  }
  const std::size_t remaining = payloadBytes - byteOffset;
  return chunkBytes < remaining ? chunkBytes : remaining;
}

__device__ __forceinline__ void validate_progress_group(
    const IbChannelLayout& channelLayout,
    ThreadGroup& group) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  // group_id selects a LOGICAL channel; maxChannels counts resource slots.
  if (channelLayout.numChannels <= 0) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: send/recv numChannels must be > 0, got %d\n",
          channelLayout.numChannels);
    }
    PIPES_DEVICE_TRAP();
  }
  if (group.group_id >= static_cast<uint32_t>(channelLayout.numChannels)) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: progress group_id=%u out of range [0, %d)\n",
          group.group_id,
          channelLayout.numChannels);
    }
    PIPES_DEVICE_TRAP();
  }
#else
  (void)channelLayout;
  (void)group;
#endif
}

template <typename P, typename Transport>
__device__ __forceinline__ IbChannelProgress& progress_send_slot(
    Transport& transport,
    ThreadGroup& group) {
  validate_progress_group(transport.channel_layout(), group);
  return transport.template local_channel_slot<P>(group).sendProgress;
}

template <typename P, typename Transport>
__device__ __forceinline__ IbChannelProgress& progress_recv_slot(
    Transport& transport,
    ThreadGroup& group) {
  validate_progress_group(transport.channel_layout(), group);
  return transport.template local_channel_slot<P>(group).recvProgress;
}

// No default on P: omitting the protocol must be a compile error, not a silent
// read of the default protocol's slot.
template <typename P, typename Transport>
__device__ __forceinline__ ChannelSlotView acquire_channel(
    Transport& transport,
    const IbChannelLayout& channelLayout,
    ThreadGroup& group) {
  validate_progress_group(channelLayout, group);
  const int channelId = static_cast<int>(group.group_id);
  const int slotIndex =
      channelLayout.protoChannelSlot(channelId, P::kProtoSlot);
  return ChannelSlotView{
      .channel = transport.local_channel(static_cast<uint32_t>(channelId)),
      .local = transport.template local_channel_slot<P>(
          static_cast<uint32_t>(channelId)),
      .remote = makeIbRemoteChannel(channelLayout, slotIndex),
      .stagingBase =
          static_cast<std::size_t>(slotIndex) * pipeline_window(channelLayout),
  };
}

/**
 * Trap if a caller tries to start a second send/recv before the first ends.
 *
 * The broadcast is the ordering point for init callers: if the leader sees a
 * non-idle slot, every thread traps before any caller can store new state.
 */
__device__ __forceinline__ void assert_progress_slot_idle(
    ThreadGroup& group,
    const IbChannelProgress& state,
    const char* direction) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  uint32_t idle = 1;
  if (group.is_leader()) {
    const auto activeStage = state.activeStage;
    idle = activeStage == detail::IbSendRecvProgressStage::Done ? 1U : 0U;
    if (!idle) {
      printf(
          "[PIPES] FATAL: %s requested with outstanding %s progress "
          "for group_id=%u stage=%d nextByte=%llu\n",
          direction,
          direction,
          group.group_id,
          static_cast<int>(activeStage),
          static_cast<unsigned long long>(state.activeNextByte));
    }
  }
  idle = group.broadcast<uint32_t>(idle);
  if (!idle) {
    PIPES_DEVICE_TRAP();
  }
#else
  (void)group;
  (void)state;
  (void)direction;
#endif
}

template <typename P, typename Transport>
__device__ __forceinline__ void prepare_send_slot(
    Transport& transport,
    ThreadGroup& group,
    uint32_t slotId,
    uint64_t generation,
    const Timeout& timeout) {
  if (group.is_leader()) {
    auto& slot = transport.template local_channel_slot<P>(group.group_id)
                     .sendCompletionSlots[slotId];
    if (slot.generation != generation) {
      const uint64_t pending = slot.laneMask;
      const uint32_t numLanes = transport.send_completion_lane_count();
      for (uint32_t laneId = 0; laneId < numLanes; ++laneId) {
        if ((pending & (1ULL << laneId)) == 0) {
          continue;
        }
        transport.wait_local_completion(
            group.group_id,
            IbLocalCompletionTicket{
                .completionId = laneId,
                .value = slot.values[laneId],
            },
            timeout);
      }
      slot.laneMask = 0;
      slot.generation = generation;
    }
  }
  group.sync();
}

template <typename P, typename Transport>
__device__ __forceinline__ void record_send_completion(
    Transport& transport,
    uint32_t channelId,
    uint32_t slotId,
    uint64_t generation,
    const IbLocalCompletionTicket& ticket) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  auto& slot = transport.template local_channel_slot<P>(channelId)
                   .sendCompletionSlots[slotId];
  slot.generation = generation;
  slot.values[ticket.completionId] = ticket.value;
  slot.laneMask |= 1ULL << ticket.completionId;
#else
  (void)transport;
  (void)channelId;
  (void)slotId;
  (void)generation;
  (void)ticket;
#endif
}
} // namespace detail

} // namespace comms::prims
