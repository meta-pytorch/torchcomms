// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/prims/transport/P2pIbTransportDeviceImpl.cuh"

namespace comms::prims {
namespace detail {

// Readiness is carried as a word rather than a bool so one broadcast can report
// three states, keeping the answer group-uniform without a second round trip.
constexpr uint32_t kProgressNotReady = 0U;
constexpr uint32_t kProgressReady = 1U;
constexpr uint32_t kProgressAborted = 2U;

/**
 * Physical staging range for the next resumable progress step.
 *
 * `stagingOff` is an offset into the transport-owned send/recv staging
 * buffers. `dataOff` is the matching protocol offset into the caller's user
 * buffer. `bytes` never crosses a per-block staging partition or the
 * reserved protocol byte count. `streamEnd` is the absolute protocol byte
 * value after this chunk and is used as the DATA_READY and SLOT_FREE readiness
 * threshold. `slotId` and `pipelineGeneration` identify the local completion
 * frontier that protects this staging range. `dataOff` is a protocol offset;
 * callers mask it against the payload byte count before invoking user-buffer
 * copy callbacks.
 */
struct ProgressChunk {
  std::size_t stagingOff; // WIRE offset into send/recv staging
  std::size_t dataOff; // PAYLOAD offset into the user buffer
  std::size_t payloadBytes; // cursor advance (payload)
  std::size_t wireBytes; // RDMA length + signal/counter delta (wire)
  uint64_t streamEndWire; // absolute WIRE byte value after this chunk
  uint64_t flagVal; // packet flag/generation (ring cursor); ignored by Simple
  uint32_t slotId; // local-completion slot for send-staging backpressure
  uint64_t pipelineGeneration; // slot reuse generation (prepare_send_slot)
};

/**
 * Register-only geometry for one resumable progress call.
 *
 * Derived in registers on every call rather than stored: only the two caller
 * inputs it cannot reach on its own (`activeUserBytes`, `activeMaxSignalBytes`)
 * live in `IbChannelProgress`; the rest come from `transport.channel_layout()`
 * and `group`, so caching them would duplicate HBM state for no gain.
 */
struct ProgressGeometry {
  // Two independent coordinates, never merged: `groupId` is the LOGICAL channel
  // (and therefore the QP channel); `slotIndex` is this protocol's flat
  // resource slot, addressing slot-indexed storage. Merging them would force a
  // division to recover one from the other on the QP path.
  int groupId;
  int slotIndex;
  std::size_t payloadBytes; // raw nbytes, for valid-byte masking
  std::size_t
      protocolBytes; // payload rounded up to Proto::kData (cursor bound)
  std::size_t perBlockSlotWire; // physical (wire) bytes per (slot, group)
  std::size_t perBlockSlotPayload; // payload capacity of that region
  std::size_t perChannelBufferSize; // wire ring window (group stride)
  std::size_t chunkPayload; // max payload per chunk (>= 1 packet)
  int pipelineDepth;
};

__device__ __forceinline__ void store_progress_state(
    ThreadGroup& group,
    IbChannelProgress& slot,
    const IbChannelProgress& state);

__device__ __forceinline__ void abandon_progress_state(
    ThreadGroup& group,
    IbChannelProgress& slot,
    IbChannelProgress& state);

/*
 * Where the abort is consulted on the ready path, and why it is not a
 * standalone entry guard.
 *
 * Every abort check in this file used to hang off a wait, and each consults the
 * abort only on its *not-ready* branch. So a call that found everything ready
 * reached the put, the fused DATA_READY, or the SLOT_FREE credit having never
 * looked at the abort -- and publishing any of those after an abort is what
 * principle 4 forbids: a false signal releases a peer that is correctly blocked
 * and stops it ever reaching its own deadline, so the fault reads to that peer
 * as success.
 *
 * The obvious fix -- one guard at the top of each progress call -- adds a
 * group-wide broadcast to every *healthy* attempt, and Tree scheduling and
 * batched send/recv drive these in tight loops. Instead the verdict rides out
 * through synchronization that already exists:
 *
 *   - `try_prepare_send_slot()` and both `progress_recv_ready()` overloads
 *     already broadcast a readiness verdict; the leader now decides the abort
 *     inside that same block and returns `kProgressAborted` through it.
 *   - The pre-put `group.sync()` becomes a broadcast carrying the verdict,
 *     which covers a call resuming at WaitSlotFree -- that path skips slot
 *     preparation, and skips the SLOT_FREE wait entirely for a chunk inside the
 *     pipeline depth.
 *
 * Net added barriers on the healthy path: zero, except LL's ready path, which
 * had no existing broadcast to fold into and now pays the one the not-ready
 * path already paid.
 */

template <typename Proto = protocol::Simple>
__device__ __forceinline__ ProgressGeometry make_progress_geometry(
    const IbChannelLayout& channelLayout,
    ThreadGroup& group,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const char* opName);

__device__ __forceinline__ std::size_t active_payload_offset(
    const IbChannelProgress& state);

__device__ __forceinline__ void reserve_progress_step(
    ThreadGroup& group,
    IbChannelProgress& slot,
    IbChannelProgress& state,
    const ProgressGeometry& geometry);

__device__ __forceinline__ void validate_send_progress_stage(
    ThreadGroup& group,
    const IbChannelProgress& state);

__device__ __forceinline__ void validate_recv_progress_stage(
    ThreadGroup& group,
    const IbChannelProgress& state);

__device__ __forceinline__ void transition_progress_stage(
    ThreadGroup& group,
    IbChannelProgress& state,
    detail::IbSendRecvProgressStage next);

template <typename Proto = protocol::Simple>
__device__ __forceinline__ ProgressChunk next_chunk(
    const IbChannelLayout& channelLayout,
    const IbChannelProgress& state,
    const ProgressGeometry& geometry);

template <typename P, typename Transport>
// Returns one of kProgressNotReady / kProgressReady / kProgressAborted, already
// broadcast so every thread in the group agrees.
__device__ __forceinline__ uint32_t try_prepare_send_slot(
    Transport& transport,
    ThreadGroup& group,
    uint32_t slotId,
    uint64_t generation,
    const AbortDevice& abortDevice = AbortDevice());

/**
 * Initialize transport-owned state for one pipelined send operation.
 *
 * The transport reserves the sender-side byte stream for `group.group_id`
 * and starts the internal state in the sender state machine unless
 * `nbytes == 0`. It captures `src`, so `progress_send_once()` advances through
 * the same buffer on every call and cannot be handed a different one.
 *
 * The send progress slot for this group must be idle. Re-initializing a
 * group while a previous send is outstanding traps with a diagnostic instead
 * of silently overwriting the in-flight byte range.
 *
 * Channel count and per-channel staging geometry are fixed in
 * `IbChannelLayout`. `max_signal_bytes == 0` sends one signal per
 * per-channel staging partition; smaller non-zero values split that
 * partition into multiple signaled sub-chunks for finer overlap with the
 * receiver.
 *
 * Zero-byte sends mark the internal state `Done` without reading or
 * validating staging geometry. This matches the blocking `send()` no-op
 * behavior and lets schedulers treat empty operations uniformly.
 *
 * @param group Thread group that will execute all later progress calls.
 * @param src Source user buffer. The range `[src, src + nbytes)` must remain
 *            valid until the operation reports `Done`.
 * @param nbytes Number of user-buffer bytes to send for this group.
 * @param max_signal_bytes Maximum signaled sub-chunk size, or 0 for default.
 */
template <typename Transport, typename Proto>
__device__ __forceinline__ void init_send_progress(
    Transport& transport,
    ThreadGroup& group,
    const void* __restrict__ src,
    std::size_t nbytes,
    std::size_t max_signal_bytes) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  auto& channelLayout = transport.channel_layout();
  auto& slot = progress_send_slot<Proto>(transport, group);
  assert_progress_slot_idle(group, slot, "send");
  IbChannelProgress state{};
  // The send path only reads through this; see IbChannelProgress.
  state.activeUserBuf = const_cast<void*>(src);
  state.activeUserBytes = nbytes;
  state.activeMaxSignalBytes = max_signal_bytes;
  state.activeStage = nbytes == 0
      ? detail::IbSendRecvProgressStage::Done
      : detail::IbSendRecvProgressStage::WaitLocalCompletion;
  if (nbytes == 0) {
    store_progress_state(group, slot, state);
    return;
  }
  // Validate the transfer before reserving the transport byte cursor. The
  // reservation runs in Proto's payload/wire geometry so init and every later
  // progress_send_once() agree on the cursor + tail padding.
  const ProgressGeometry geometry = make_progress_geometry<Proto>(
      channelLayout, group, nbytes, max_signal_bytes, "init_send_progress");
  reserve_progress_step(group, slot, state, geometry);
  store_progress_state(group, slot, state);
#else
  (void)transport;
  (void)group;
  (void)src;
  (void)nbytes;
  (void)max_signal_bytes;
#endif
}

template <typename Transport>
__device__ __forceinline__ void init_registered_send_progress(
    Transport& transport,
    ThreadGroup& group,
    const IbgdaLocalBuffer& src,
    std::size_t nbytes,
    std::size_t max_signal_bytes) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (nbytes > 0) {
    __threadfence_system();
    group.sync();
  }
  // activeUserBuf stays null: this path never copies through send staging, so
  // the registered descriptor below is the only source it uses.
  init_send_progress(
      transport, group, /*src=*/nullptr, nbytes, max_signal_bytes);
  // Written after the delegated init, which builds its own local state and
  // stores it. store_progress_state() does not carry this field, so the slot
  // keeps what is written here for the whole operation.
  auto& slot = progress_send_slot<protocol::Simple>(transport, group);
  if (group.is_leader()) {
    slot.activeRegisteredBuf = src;
  }
  group.sync();
#else
  (void)transport;
  (void)group;
  (void)src;
  (void)nbytes;
  (void)max_signal_bytes;
#endif
}

/**
 * Initialize transport-owned state for one pipelined recv operation.
 *
 * The transport reserves the receiver-side byte stream for `group.group_id`
 * and starts the internal state in the receiver state machine unless
 * `nbytes == 0`. It captures `dst`, so `progress_recv_once()` advances through
 * the same buffer on every call and cannot be handed a different one.
 *
 * The recv progress slot for this group must be idle. Re-initializing a
 * group while a previous recv is outstanding traps with a diagnostic instead
 * of silently overwriting the in-flight byte range.
 *
 * The sender and receiver must use compatible `max_signal_bytes` for a
 * logical transfer. Channel count and staging geometry are fixed in the
 * transport layout; `max_signal_bytes` only controls sub-chunk signaling.
 *
 * Zero-byte receives mark the internal state `Done` without reading or
 * validating staging geometry. This matches the blocking `recv()` no-op
 * behavior and lets schedulers treat empty operations uniformly.
 *
 * @param group Thread group that will execute all later progress calls.
 * @param dst Destination user buffer. The range `[dst, dst + nbytes)` must
 *            remain valid until the operation reports `Done`.
 * @param nbytes Number of user-buffer bytes to receive for this group.
 * @param max_signal_bytes Maximum signaled sub-chunk size, or 0 for default.
 */
template <typename Transport, typename Proto>
__device__ __forceinline__ void init_recv_progress(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    std::size_t nbytes,
    std::size_t max_signal_bytes) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  auto& channelLayout = transport.channel_layout();
  auto& slot = progress_recv_slot<Proto>(transport, group);
  assert_progress_slot_idle(group, slot, "recv");
  IbChannelProgress state{};
  state.activeUserBuf = dst;
  state.activeUserBytes = nbytes;
  state.activeMaxSignalBytes = max_signal_bytes;
  state.activeStage = nbytes == 0
      ? detail::IbSendRecvProgressStage::Done
      : detail::IbSendRecvProgressStage::WaitDataReady;
  if (nbytes == 0) {
    store_progress_state(group, slot, state);
    return;
  }
  // Validate the transfer before reserving the transport byte cursor. The
  // reservation runs in Proto's payload/wire geometry so init and every later
  // progress_recv_once() agree on the cursor + tail padding.
  const ProgressGeometry geometry = make_progress_geometry<Proto>(
      channelLayout, group, nbytes, max_signal_bytes, "init_recv_progress");
  reserve_progress_step(group, slot, state, geometry);
  store_progress_state(group, slot, state);
#else
  (void)transport;
  (void)group;
  (void)dst;
  (void)nbytes;
  (void)max_signal_bytes;
#endif
}

/**
 * Attempt bounded progress on one initialized send.
 *
 * This method advances at most one staged copy plus one RDMA put for the
 * current chunk. It never spins on local completion or SLOT_FREE: if either
 * dependency is not ready, it returns immediately so a higher-level scheduler
 * can try another independent lane. If an `AbortDevice` is enabled, it is
 * checked only at those readiness points and should already have been started
 * by the caller.
 *
 * The send path first checks local completion before reusing the local
 * send-staging range, then copies user data into send-staging through
 * `CopyOp::send`, waits for SLOT_FREE before reusing the peer's recv-staging
 * range, and finally issues an RDMA put that piggybacks DATA_READY and records
 * the returned completion ticket per chunk. Returning `Done` means
 * the reserved byte range has been posted; later slot reuse waits for local
 * completion.
 *
 * `CopyOp` must expose `send(dst, src, bytes, group, dataOffset, args...)`.
 * The default `Memcpy` copies bytes cooperatively across the supplied
 * `ThreadGroup`; custom copy ops may use `args` to pass reduction or
 * conversion context.
 *
 * @param transport Owning transport used for every transport op.
 * @param group Thread group matching the one used during initialization.
 * @param abortDevice Optional device abortDevice checked while dependencies
 * wait.
 * @param args Additional arguments forwarded to `CopyOp::send`.
 */
// ---- Resumable-progress protocol seams (tag-dispatched) ------------------
// These isolate the four protocol-specific steps of progress_send_once() /
// progress_recv_once() so the resumable state machine stays protocol-agnostic.
// Only the protocol::Simple overloads exist here (behavior identical to the
// former inline code); later protocols (e.g. protocol::LL) add their own
// overloads without touching the state machine. Split differently from the
// blocking prepareSendBuf()/consumeRecvBuf(): encode and signal are issued at
// different stages, and recv needs a non-blocking readiness check.

// Encode one send chunk into send-staging (Simple: contiguous CopyOp::send).
template <typename CopyOp = Memcpy, typename... Args>
__device__ __forceinline__ void progress_send_prepare_buf(
    protocol::Simple,
    ThreadGroup& group,
    const IbChannelLayout& channelLayout,
    const ProgressChunk& chunk,
    const void* __restrict__ src,
    std::size_t payloadBytes,
    Args... args) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const std::size_t validBytes =
      valid_payload_bytes(chunk.dataOff, chunk.payloadBytes, payloadBytes);
  if (validBytes > 0) {
    CopyOp::send(
        channelLayout.sendStagingPtr + chunk.stagingOff,
        static_cast<const char*>(src) + chunk.dataOff,
        validBytes,
        group,
        chunk.dataOff,
        args...);
  }
#else
  (void)group;
  (void)channelLayout;
  (void)chunk;
  (void)src;
  (void)payloadBytes;
  ((void)args, ...);
#endif
}

// The DATA_READY signal the leader put piggybacks (Simple: remote slot +
// cumulative-bytes credit).
__device__ __forceinline__ SendSignal progress_send_signal(
    protocol::Simple,
    const IbRemoteChannel& remoteChannel,
    uint64_t signalVal) {
  return SendSignal{remoteChannel.dataReady, signalVal};
}

// ---- LL (data + inline flag) progress send overloads ---------------------
// The LL siblings of the Simple send seams above. Encode packs payload+flag
// via the CopyOp's packet-aware sendLL<P> hook (Memcpy delegates to
// LLImpl::pack) so the put carries the readiness mark inline; the signal is
// therefore empty (no DATA_READY). This mirrors the blocking LL path
// (prepareSendBuf(protocol::LL)) -- the CopyOp must provide sendLL<P>, so a
// reduce/convert op can plug in; a plain contiguous copy cannot address the
// packet-interleaved staging.

// Encode one send chunk into send-staging (LL: CopyOp::sendLL<P>,
// payload+flag).
template <typename CopyOp = Memcpy, typename... Args>
__device__ __forceinline__ void progress_send_prepare_buf(
    protocol::LL,
    ThreadGroup& group,
    const IbChannelLayout& channelLayout,
    const ProgressChunk& chunk,
    const void* __restrict__ src,
    std::size_t payloadBytes,
    Args... args) {
  using P = LlxPacket<4, 4>;
  static_assert(
      has_sendLL_v<CopyOp, P>,
      "LL progress send requires a CopyOp with a packet-aware sendLL<P>(); "
      "Memcpy provides one. A reduce/convert CopyOp must supply its own -- a "
      "plain contiguous copy cannot address the data+flag interleaved staging");
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const std::size_t validBytes =
      valid_payload_bytes(chunk.dataOff, chunk.payloadBytes, payloadBytes);
  CopyOp::template sendLL<P>(
      group,
      channelLayout.sendStagingPtr + chunk.stagingOff,
      static_cast<const char*>(src) + chunk.dataOff,
      validBytes,
      chunk.dataOff,
      static_cast<typename P::FlagType>(chunk.flagVal),
      args...);
#else
  (void)group;
  (void)channelLayout;
  (void)chunk;
  (void)src;
  (void)payloadBytes;
  ((void)args, ...);
#endif
}

// The signal the leader put piggybacks (LL: none -- the inline packet flag is
// the readiness mark, so the put carries data only).
__device__ __forceinline__ SendSignal progress_send_signal(
    protocol::LL,
    const IbRemoteChannel& /*remoteChannel*/,
    uint64_t /*signalVal*/) {
  return SendSignal{IbgdaRemoteBuffer{}, /*val=*/0};
}

template <typename Transport, typename CopyOp, typename Proto, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus progress_send_once_impl(
    Transport& transport,
    ThreadGroup& group,
    const AbortDevice& abortDevice,
    const PipesTraceAllReduceContext* traceContext,
    PipesTraceProgressState* traceState,
    Args... args) {
  // The progress API drives the FIXED-size protocol only: it signals in wire
  // bytes and ignores CopyOp::send()'s returned wire size, so a variable-size
  // policy (e.g. AnsCompress) would put/signal the wrong length and corrupt the
  // stream. Forbid it explicitly; variable-size CopyOps must use the blocking
  // send(). See D108485978 review.
  static_assert(
      !detail::copyop_variable_size_v<CopyOp>,
      "progress_send_once() supports fixed-size CopyOps only; use the "
      "blocking send() for variable-size CopyOps such as AnsCompress.");
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef __HIP_PLATFORM_AMD__
  static_assert(
      sizeof(CopyOp) == 0, "detail::progress_send_once() requires NVIDIA GPU");
#endif
  auto& channelLayout = transport.channel_layout();
  auto& progressSlot = progress_send_slot<Proto>(transport, group);
  IbChannelProgress state = progressSlot;
  if (state.activeStage == detail::IbSendRecvProgressStage::Done) {
    return IbgdaSendRecvProgressStatus::Done;
  }
  const ProgressGeometry progress_params = make_progress_geometry<Proto>(
      channelLayout,
      group,
      state.activeUserBytes,
      state.activeMaxSignalBytes,
      "progress_send_once");
  if (active_payload_offset(state) >= progress_params.protocolBytes) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: progress_send_once payloadOffset=%llu >= "
          "protocolBytes=%llu without Done stage\n",
          static_cast<unsigned long long>(active_payload_offset(state)),
          static_cast<unsigned long long>(progress_params.protocolBytes));
    }
    PIPES_DEVICE_TRAP();
  }
  validate_send_progress_stage(group, state);

  const detail::IbSendRecvProgressStage initialStage = state.activeStage;
  const std::size_t initialNextByte = state.activeNextByte;
  const std::size_t pipelineBytesWire = progress_params.perBlockSlotWire *
      static_cast<std::size_t>(channelLayout.pipelineDepth);
  const ChannelSlotView ch =
      acquire_channel<Proto>(transport, channelLayout, group);
  IbLocalChannel& localChannel = ch.channel;
  const IbgdaLocalBuffer localSlotFree = ch.local.slotFree;
  const IbRemoteChannel remoteChannel = ch.remote;

  if (state.activeStage ==
      detail::IbSendRecvProgressStage::WaitLocalCompletion) {
    const ProgressChunk chunk =
        next_chunk<Proto>(channelLayout, state, progress_params);
    const uint32_t slotReadiness = try_prepare_send_slot<Proto>(
        transport, group, chunk.slotId, chunk.pipelineGeneration, abortDevice);
    if (slotReadiness != kProgressReady) {
      if (fine_trace_enabled(traceContext) && traceState != nullptr &&
          !traceState->localCompletionWaitOpen && group.is_leader()) {
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceLocalCompletionWaitBegin,
            static_cast<uint8_t>(kPipesTraceQpLaneMask),
            chunk.wireBytes);
        traceState->localCompletionWaitOpen = true;
      }
      if (slotReadiness == kProgressAborted) {
        abandon_progress_state(group, progressSlot, state);
        return IbgdaSendRecvProgressStatus::Aborted;
      }
      return IbgdaSendRecvProgressStatus::Waiting;
    }
    if (fine_trace_enabled(traceContext) && traceState != nullptr &&
        traceState->localCompletionWaitOpen && group.is_leader()) {
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceLocalCompletionWaitEnd,
          static_cast<uint8_t>(kPipesTraceQpLaneMask),
          chunk.wireBytes);
      traceState->localCompletionWaitOpen = false;
    }

    if (group.is_leader()) {
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceStageCopyBegin,
          static_cast<uint8_t>(kPipesTraceQpLaneMask),
          chunk.payloadBytes);
    }
    progress_send_prepare_buf<CopyOp>(
        Proto{},
        group,
        channelLayout,
        chunk,
        state.activeUserBuf,
        progress_params.payloadBytes,
        args...);
    if (group.is_leader()) {
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceStageCopyEnd,
          static_cast<uint8_t>(kPipesTraceQpLaneMask),
          chunk.payloadBytes);
    }
    group.sync();
    transition_progress_stage(
        group, state, detail::IbSendRecvProgressStage::WaitSlotFree);
  }

  if (state.activeStage == detail::IbSendRecvProgressStage::WaitSlotFree) {
    const ProgressChunk chunk =
        next_chunk<Proto>(channelLayout, state, progress_params);
    const bool isFinalChunk =
        chunk.dataOff + chunk.payloadBytes >= progress_params.protocolBytes;
    const uint64_t protocolStreamEnd = chunk.streamEndWire +
        (isFinalChunk ? Proto::wire_bytes(state.activeTailPadding) : 0);
    if (protocolStreamEnd > pipelineBytesWire) {
      const uint64_t expected = protocolStreamEnd - pipelineBytesWire;
      uint32_t ready = 1;
      unsigned long long current = 0;
      if (group.is_leader()) {
        current = static_cast<unsigned long long>(
            transport.read_signal(localSlotFree));
        ready = current >= expected ? 1U : 0U;
        if (!ready) {
          if (FT_ABORT_CHECK(
                  abortDevice,
                  "progress_send_once waiting for SLOT_FREE expected>=%llu, "
                  "current=%llu",
                  static_cast<unsigned long long>(expected),
                  current)) {
            ready = kProgressAborted;
          }
        }
      }
      ready = group.broadcast<uint32_t>(ready);
      if (ready == kProgressAborted) {
        abandon_progress_state(group, progressSlot, state);
        return IbgdaSendRecvProgressStatus::Aborted;
      }
      if (!ready) {
        if (fine_trace_enabled(traceContext) && traceState != nullptr &&
            !traceState->remoteSlotFreeWaitOpen && group.is_leader()) {
          trace_allreduce_event(
              traceContext,
              PipesTraceEventType::kAllReduceRemoteSlotFreeWaitBegin,
              static_cast<uint8_t>(kPipesTraceQpLaneMask),
              chunk.wireBytes);
          traceState->remoteSlotFreeWaitOpen = true;
        }
        if (state.activeStage != initialStage ||
            state.activeNextByte != initialNextByte) {
          store_progress_state(group, progressSlot, state);
          return IbgdaSendRecvProgressStatus::Progressed;
        }
        return IbgdaSendRecvProgressStatus::Waiting;
      }
      if (fine_trace_enabled(traceContext) && traceState != nullptr &&
          traceState->remoteSlotFreeWaitOpen && group.is_leader()) {
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceRemoteSlotFreeWaitEnd,
            static_cast<uint8_t>(kPipesTraceQpLaneMask),
            chunk.wireBytes);
        traceState->remoteSlotFreeWaitOpen = false;
      }
    }

    // Last gate before the put and its fused DATA_READY.
    //
    // A call that *resumes* at WaitSlotFree skipped `try_prepare_send_slot`,
    // and the SLOT_FREE wait above is itself skipped for any chunk inside the
    // pipeline depth -- so on that path nothing has consulted the abort yet.
    // This replaces the `group.sync()` that already stood here rather than
    // adding a barrier: a broadcast is the same single synchronization, and it
    // carries the verdict out with it.
    uint32_t sendAborted = 0;
    if (group.is_leader()) {
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceSendSyncBegin,
          static_cast<uint8_t>(kPipesTraceQpLaneMask),
          chunk.wireBytes);
      sendAborted =
          FT_ABORT_CHECK(abortDevice, "send publish on an aborted communicator")
          ? 1U
          : 0U;
    }
    if (group.broadcast<uint32_t>(sendAborted) != 0U) {
      abandon_progress_state(group, progressSlot, state);
      return IbgdaSendRecvProgressStatus::Aborted;
    }
    if (group.is_leader()) {
      __threadfence_system();
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceSendSyncEnd,
          static_cast<uint8_t>(kPipesTraceQpLaneMask),
          chunk.wireBytes);
      const uint32_t numLanes = static_cast<uint32_t>(channelLayout.numLanes);
      const uint8_t qpLane = static_cast<uint8_t>(
          numLanes == 0 ? 0 : localChannel.sendQp.cursor % numLanes);
      ThreadGroup solo{
          0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
      const std::size_t protocolBytesThis = chunk.wireBytes +
          (isFinalChunk ? Proto::wire_bytes(state.activeTailPadding) : 0);
      const SendSignal sig =
          progress_send_signal(Proto{}, remoteChannel, protocolBytesThis);
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceWqeSubmitBegin,
          qpLane,
          protocolBytesThis);
      const auto completion = transport.put(
          solo,
          channelLayout.sendStagingBuf.subBuffer(chunk.stagingOff),
          remoteChannel.recvStaging.subBuffer(chunk.stagingOff),
          chunk.wireBytes,
          sig.buf,
          sig.val,
          /*counterBuf=*/{},
          /*counterVal=*/0,
          /*signalPerLane=*/true);
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceWqeSubmitEnd,
          qpLane,
          protocolBytesThis);
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceBookkeepingBegin,
          qpLane,
          protocolBytesThis);
      record_send_completion<Proto>(
          transport,
          static_cast<uint32_t>(progress_params.groupId),
          chunk.slotId,
          chunk.pipelineGeneration,
          completion);
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceBookkeepingEnd,
          qpLane,
          protocolBytesThis);
    }
    group.sync();

    state.activeNextByte += chunk.payloadBytes;
    if (active_payload_offset(state) >= progress_params.protocolBytes) {
      transition_progress_stage(
          group, state, detail::IbSendRecvProgressStage::Done);
      store_progress_state(group, progressSlot, state);
      return IbgdaSendRecvProgressStatus::Done;
    }
    transition_progress_stage(
        group, state, detail::IbSendRecvProgressStage::WaitLocalCompletion);
  }

  // A full non-final chunk can cycle WaitLocalCompletion -> WaitSlotFree ->
  // WaitLocalCompletion in one call, leaving the stage unchanged while nextByte
  // advances. Check both fields so that case reports Progressed.
  if (state.activeStage != initialStage ||
      state.activeNextByte != initialNextByte) {
    store_progress_state(group, progressSlot, state);
    return IbgdaSendRecvProgressStatus::Progressed;
  }
  return IbgdaSendRecvProgressStatus::Waiting;
#else
  (void)transport;
  (void)group;
  (void)abortDevice;
  (void)traceContext;
  (void)traceState;
  return IbgdaSendRecvProgressStatus::Done;
#endif
}

template <typename Transport>
__device__ __forceinline__ IbgdaRegisteredSendProgressStatus
progress_registered_send_once(
    Transport& transport,
    ThreadGroup& group,
    const AbortDevice& abortDevice) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  auto& channelLayout = transport.channel_layout();
  auto& progressSlot = progress_send_slot<protocol::Simple>(transport, group);
  IbChannelProgress state = progressSlot;
  if (state.activeStage == detail::IbSendRecvProgressStage::Done) {
    return IbgdaRegisteredSendProgressStatus::Posted;
  }

  const ProgressGeometry geometry = make_progress_geometry<protocol::Simple>(
      channelLayout,
      group,
      state.activeUserBytes,
      state.activeMaxSignalBytes,
      "progress_registered_send_once");
  if (active_payload_offset(state) >= geometry.protocolBytes) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: progress_registered_send_once payloadOffset=%llu "
          ">= protocolBytes=%llu without Done stage\n",
          static_cast<unsigned long long>(active_payload_offset(state)),
          static_cast<unsigned long long>(geometry.protocolBytes));
    }
    PIPES_DEVICE_TRAP();
  }
  validate_send_progress_stage(group, state);

  const detail::IbSendRecvProgressStage initialStage = state.activeStage;
  const std::size_t initialNextByte = state.activeNextByte;
  const std::size_t pipelineBytesWire = geometry.perBlockSlotWire *
      static_cast<std::size_t>(channelLayout.pipelineDepth);
  const ChannelSlotView ch =
      acquire_channel<protocol::Simple>(transport, channelLayout, group);
  const IbgdaLocalBuffer localSlotFree = ch.local.slotFree;
  const IbRemoteChannel remoteChannel = ch.remote;

  if (state.activeStage ==
      detail::IbSendRecvProgressStage::WaitLocalCompletion) {
    const ProgressChunk chunk =
        next_chunk<protocol::Simple>(channelLayout, state, geometry);
    const uint32_t slotReadiness = try_prepare_send_slot<protocol::Simple>(
        transport, group, chunk.slotId, chunk.pipelineGeneration, abortDevice);
    if (slotReadiness != kProgressReady) {
      if (slotReadiness == kProgressAborted) {
        abandon_progress_state(group, progressSlot, state);
        return IbgdaRegisteredSendProgressStatus::Aborted;
      }
      return IbgdaRegisteredSendProgressStatus::Waiting;
    }
    transition_progress_stage(
        group, state, detail::IbSendRecvProgressStage::WaitSlotFree);
  }

  if (state.activeStage == detail::IbSendRecvProgressStage::WaitSlotFree) {
    const ProgressChunk chunk =
        next_chunk<protocol::Simple>(channelLayout, state, geometry);
    const bool isFinalChunk =
        chunk.dataOff + chunk.payloadBytes >= geometry.protocolBytes;
    const uint64_t protocolStreamEnd = chunk.streamEndWire +
        (isFinalChunk ? protocol::Simple::wire_bytes(state.activeTailPadding)
                      : 0);
    if (protocolStreamEnd > pipelineBytesWire) {
      const uint64_t expected = protocolStreamEnd - pipelineBytesWire;
      uint32_t ready = 1;
      unsigned long long current = 0;
      if (group.is_leader()) {
        current = static_cast<unsigned long long>(
            transport.read_signal(localSlotFree));
        ready = current >= expected ? 1U : 0U;
        if (!ready) {
          if (FT_ABORT_CHECK(
                  abortDevice,
                  "progress_registered_send_once waiting for SLOT_FREE "
                  "expected>=%llu, current=%llu",
                  static_cast<unsigned long long>(expected),
                  current)) {
            ready = kProgressAborted;
          }
        }
      }
      ready = group.broadcast<uint32_t>(ready);
      if (ready == kProgressAborted) {
        abandon_progress_state(group, progressSlot, state);
        return IbgdaRegisteredSendProgressStatus::Aborted;
      }
      if (!ready) {
        if (state.activeStage != initialStage ||
            state.activeNextByte != initialNextByte) {
          store_progress_state(group, progressSlot, state);
          return IbgdaRegisteredSendProgressStatus::Progressed;
        }
        return IbgdaRegisteredSendProgressStatus::Waiting;
      }
    }

    const std::size_t validBytes = valid_payload_bytes(
        chunk.dataOff, chunk.payloadBytes, geometry.payloadBytes);
    const std::size_t protocolBytesThis = chunk.wireBytes +
        (isFinalChunk ? protocol::Simple::wire_bytes(state.activeTailPadding)
                      : 0);

    // Same gate as the plain send path, for the same reason: a resumed call
    // entering at WaitSlotFree consulted nothing, and this replaces the
    // `group.sync()` that already stood here rather than adding a barrier.
    uint32_t sendAborted = 0;
    if (group.is_leader()) {
      sendAborted =
          FT_ABORT_CHECK(
              abortDevice, "registered send publish on an aborted communicator")
          ? 1U
          : 0U;
    }
    if (group.broadcast<uint32_t>(sendAborted) != 0U) {
      abandon_progress_state(group, progressSlot, state);
      return IbgdaRegisteredSendProgressStatus::Aborted;
    }
    if (group.is_leader()) {
      __threadfence_system();
      ThreadGroup solo{
          0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
      const auto completion = transport.put(
          solo,
          state.activeRegisteredBuf.subBuffer(chunk.dataOff),
          remoteChannel.recvStaging.subBuffer(chunk.stagingOff),
          validBytes,
          remoteChannel.dataReady,
          protocolBytesThis,
          {},
          0,
          true);
      record_send_completion<protocol::Simple>(
          transport,
          static_cast<uint32_t>(geometry.groupId),
          chunk.slotId,
          chunk.pipelineGeneration,
          completion);
    }
    group.sync();

    state.activeNextByte += chunk.payloadBytes;
    if (active_payload_offset(state) >= geometry.protocolBytes) {
      transition_progress_stage(
          group, state, detail::IbSendRecvProgressStage::Done);
      store_progress_state(group, progressSlot, state);
      return IbgdaRegisteredSendProgressStatus::Posted;
    }
    transition_progress_stage(
        group, state, detail::IbSendRecvProgressStage::WaitLocalCompletion);
  }

  if (state.activeStage != initialStage ||
      state.activeNextByte != initialNextByte) {
    store_progress_state(group, progressSlot, state);
    return IbgdaRegisteredSendProgressStatus::Progressed;
  }
  return IbgdaRegisteredSendProgressStatus::Waiting;
#else
  (void)transport;
  (void)group;
  (void)abortDevice;
  return IbgdaRegisteredSendProgressStatus::Drained;
#endif
}

/**
 * Drain outstanding local send completions for this channel, one bounded pass.
 *
 * Terminal once the abort is latched, which is the drain-side counterpart of
 * `abandon_progress_state()`. Without it this call is not terminal at all: the
 * abort path persists the lanes it did NOT drain (`slot.laneMask = pending`
 * below) and stops the outer loop early, so every later call re-finds the same
 * lanes and returns `Aborted` again forever. A caller that loops until
 * `Drained`
 * -- `ReduceScatterDirectIbV2.cu` does exactly that -- then spins for the life
 * of the kernel. The `activeStage == Done` short-circuit that saves the other
 * progress entry points has no analogue here, because the drain reads the
 * completion lane masks rather than the progress slot.
 *
 * The masks are deliberately left ALONE rather than cleared. Clearing would
 * tell `try_prepare_send_slot()` that those completions have landed when they
 * have not, dropping the one guard that stops a later operation overwriting
 * send-staging while the NIC is still reading out of it. Reporting `Drained`
 * costs nothing by comparison: after an abort the channel is not expected to
 * carry meaningful traffic anyway, and recovery is a `reconfigure()`.
 *
 * The call that first observes the abort mid-scan still reports `Aborted`, so
 * the signal is not lost; only subsequent calls short-circuit.
 */
template <typename Transport>
__device__ __forceinline__ IbgdaRegisteredSendProgressStatus
progress_registered_send_drain_once(
    Transport& transport,
    ThreadGroup& group,
    const AbortDevice& abortDevice) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  uint32_t result =
      static_cast<uint32_t>(IbgdaRegisteredSendProgressStatus::Drained);
  if (group.is_leader() && !abortDevice.isAborted()) {
    bool foundPending = false;
    bool madeProgress = false;
    bool aborted = false;
    auto& channel =
        transport.template local_channel_slot<protocol::Simple>(group.group_id);
    const uint32_t numLanes = transport.send_completion_lane_count();
    const int pipelineDepth = transport.channel_layout().pipelineDepth;

    for (int slotId = 0; slotId < pipelineDepth && !aborted; ++slotId) {
      auto& slot = channel.sendCompletionSlots[slotId];
      uint64_t pending = slot.laneMask;
      for (uint32_t laneId = 0; laneId < numLanes; ++laneId) {
        const uint64_t laneBit = 1ULL << laneId;
        if ((pending & laneBit) == 0) {
          continue;
        }
        const IbLocalCompletionTicket ticket{
            .completionId = laneId,
            .value = slot.values[laneId],
        };
        if (transport.is_local_completion_ready(
                group.group_id, ticket, abortDevice)) {
          pending &= ~laneBit;
          madeProgress = true;
          continue;
        }
        foundPending = true;
        aborted = FT_ABORT_CHECK(
            abortDevice,
            "registered send local completion timed out slot=%d lane=%u",
            slotId,
            laneId);
        if (aborted) {
          break;
        }
      }
      // Persist the lanes drained so far even when unwinding, so a retry does
      // not re-wait on completions that already landed.
      slot.laneMask = pending;
    }

    if (aborted) {
      result =
          static_cast<uint32_t>(IbgdaRegisteredSendProgressStatus::Aborted);
    } else if (foundPending) {
      result = static_cast<uint32_t>(
          madeProgress ? IbgdaRegisteredSendProgressStatus::Progressed
                       : IbgdaRegisteredSendProgressStatus::Waiting);
    }
  }
  result = group.broadcast<uint32_t>(result);
  return static_cast<IbgdaRegisteredSendProgressStatus>(result);
#else
  (void)transport;
  (void)group;
  (void)abortDevice;
  return IbgdaRegisteredSendProgressStatus::Drained;
#endif
}

template <typename Transport>
__device__ __forceinline__ void send_registered(
    Transport& transport,
    ThreadGroup& group,
    const IbgdaLocalBuffer& src,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const AbortDevice& abortDevice) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  init_registered_send_progress(
      transport, group, src, nbytes, max_signal_bytes);
  IbgdaRegisteredSendProgressStatus status;
  // Both loops are unbounded by design: they spin until the NIC makes progress.
  // `Aborted` is the only escape when the NIC never will, so it must terminate
  // them - otherwise an abort on a dead NIC deadlocks here, which is the exact
  // failure this abort plumbing exists to break.
  do {
    status = progress_registered_send_once(transport, group, abortDevice);
  } while (status != IbgdaRegisteredSendProgressStatus::Posted &&
           status != IbgdaRegisteredSendProgressStatus::Drained &&
           status != IbgdaRegisteredSendProgressStatus::Aborted);
  while (status != IbgdaRegisteredSendProgressStatus::Drained &&
         status != IbgdaRegisteredSendProgressStatus::Aborted) {
    status = progress_registered_send_drain_once(transport, group, abortDevice);
  }
#else
  (void)transport;
  (void)group;
  (void)src;
  (void)nbytes;
  (void)max_signal_bytes;
  (void)abortDevice;
#endif
}

template <typename Transport, typename CopyOp, typename Proto, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus progress_send_once(
    Transport& transport,
    ThreadGroup& group,
    const AbortDevice& abortDevice,
    Args... args) {
  return progress_send_once_impl<Transport, CopyOp, Proto>(
      transport, group, abortDevice, nullptr, nullptr, args...);
}

template <typename Transport, typename CopyOp, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus
progress_send_once_with_trace(
    Transport& transport,
    ThreadGroup& group,
    const AbortDevice& abortDevice,
    const PipesTraceAllReduceContext& traceContext,
    PipesTraceProgressState& traceState,
    Args... args) {
  return progress_send_once_impl<Transport, CopyOp, protocol::Simple>(
      transport, group, abortDevice, &traceContext, &traceState, args...);
}

/**
 * Non-blocking poll for one receive chunk's DATA_READY on its round-robin lane.
 *
 * Leader-only. Mirrors wait_recv_data_ready's readiness test without spinning:
 * returns true when the chunk's DATA_READY has landed on its lane, advancing
 * `recvDataReadyLaneCursor`/`recvLaneExpected` by exactly one chunk on that
 * (and only that) return. A false return leaves all receiver state untouched so
 * the caller can retry the same chunk on a later progress attempt.
 * `currentOut`/`expectedOut` are set for the caller's abortDevice diagnostic.
 */
template <typename Transport>
__device__ __forceinline__ bool poll_recv_data_ready(
    Transport& transport,
    IbLocalChannel& localChannel,
    const IbgdaLocalBuffer& localDataReady,
    std::size_t chunkBytes,
    unsigned long long& currentOut,
    unsigned long long& expectedOut) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const uint32_t numLanes =
      static_cast<uint32_t>(transport.channel_layout().numLanes);
  const uint32_t lanes = numLanes == 0 ? 1U : numLanes;
  // Simple's slot unconditionally -- see wait_recv_data_ready.
  IbChannelProtoSlot& protoSlot =
      localChannel.protos[protocol::Simple::kProtoSlot];
  // Truncate to 32 bits before the modulo to match the sender's uint32 cursor
  // wrap (see wait_recv_data_ready).
  const uint32_t lane =
      static_cast<uint32_t>(localChannel.recvDataReadyLaneCursor) % lanes;
  const uint64_t expected = protoSlot.recvLaneExpected[lane] + chunkBytes;
  const IbgdaLocalBuffer laneBuf = localDataReady.subBuffer(
      sendRecvSignalSlotOffset(static_cast<int>(lane)));
  const uint64_t current = transport.read_signal(laneBuf);
  currentOut = static_cast<unsigned long long>(current);
  expectedOut = static_cast<unsigned long long>(expected);
  if (current < expected) {
    return false;
  }
  protoSlot.recvLaneExpected[lane] = expected;
  ++localChannel.recvDataReadyLaneCursor;
  return true;
#else
  (void)transport;
  (void)localChannel;
  (void)localDataReady;
  (void)chunkBytes;
  currentOut = 0;
  expectedOut = 0;
  return false;
#endif
}

/**
 * Attempt bounded progress on one initialized recv.
 *
 * This method advances at most one recv-staging copy for the current chunk.
 * It never spins on DATA_READY: if the sender has not signaled the next
 * chunk, it returns `Waiting` immediately. If an `AbortDevice` is enabled, it
 * is checked only while the DATA_READY dependency is not ready and should
 * already have been started by the caller.
 *
 * When DATA_READY reaches the chunk's `streamEnd`, the recv path copies from
 * transport-owned recv-staging into the caller's destination through
 * `CopyOp::recv`, then signals SLOT_FREE per chunk back to the
 * sender. Returning `Done` means the reserved byte range has completed.
 *
 * `CopyOp` must expose `recv(dst, src, bytes, group, dataOffset, args...)`.
 * The default `Memcpy` copies bytes cooperatively across the supplied
 * `ThreadGroup`; custom copy ops may use `args` to pass reduction or
 * conversion context.
 *
 * @param transport Owning transport used for every transport op.
 * @param group Thread group matching the one used during initialization.
 * @param abortDevice Optional device abortDevice checked while dependencies
 * wait.
 * @param args Additional arguments forwarded to `CopyOp::recv`.
 */
// Non-blocking readiness check for one recv chunk (Simple: poll the round-robin
// DATA_READY lane that carried this chunk; leader-only + broadcast, no spin).
// `channelLayout`/`chunk` are unused by Simple (they carry the recv staging +
// packet geometry the LL overload polls); the seam passes them so both
// protocols share the call site in progress_recv_once.
template <typename Transport>
__device__ __forceinline__ uint32_t progress_recv_ready(
    protocol::Simple,
    Transport& transport,
    ThreadGroup& group,
    const IbChannelLayout& channelLayout,
    const ProgressChunk& chunk,
    IbLocalChannel& localChannel,
    const IbgdaLocalBuffer& localDataReady,
    uint64_t waitCredit,
    const AbortDevice& abortDevice,
    const PipesTraceAllReduceContext* traceContext,
    PipesTraceProgressState* traceState,
    uint8_t qpLane) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  (void)channelLayout;
  (void)chunk;
  uint32_t ready = 1;
  if (group.is_leader()) {
    unsigned long long current = 0;
    unsigned long long expected = 0;
    // Asked before readiness, so the already-landed path is covered too. This
    // function already broadcasts, so the check is free of extra barriers; a
    // separate entry guard would add one to every healthy poll. Without it a
    // chunk whose data had arrived went on to consume it and credit SLOT_FREE
    // without ever consulting the abort.
    if (FT_ABORT_CHECK(
            abortDevice, "recv progress on an aborted communicator")) {
      ready = kProgressAborted;
    } else {
      ready = poll_recv_data_ready(
                  transport,
                  localChannel,
                  localDataReady,
                  waitCredit,
                  current,
                  expected)
          ? 1U
          : 0U;
      if (!ready) {
        if (fine_trace_enabled(traceContext) && traceState != nullptr &&
            !traceState->dataReadyWaitOpen) {
          trace_allreduce_event(
              traceContext,
              PipesTraceEventType::kAllReduceDataReadyWaitBegin,
              qpLane,
              waitCredit);
          traceState->dataReadyWaitOpen = true;
        }
        if (FT_ABORT_CHECK(
                abortDevice,
                "progress_recv_once waiting for DATA_READY expected>=%llu, "
                "current=%llu",
                expected,
                current)) {
          ready = kProgressAborted;
        }
      } else if (
          fine_trace_enabled(traceContext) && traceState != nullptr &&
          traceState->dataReadyWaitOpen) {
        trace_allreduce_event(
            traceContext,
            PipesTraceEventType::kAllReduceDataReadyWaitEnd,
            qpLane,
            waitCredit);
        traceState->dataReadyWaitOpen = false;
      }
    }
  }
  // Broadcast makes the answer group-uniform, which is what lets abort ride out
  // as `kProgressAborted` instead of an indistinguishable "not ready" that the
  // caller's driver would retry forever.
  return group.broadcast<uint32_t>(ready);
#else
  (void)transport;
  (void)group;
  (void)channelLayout;
  (void)chunk;
  (void)localChannel;
  (void)localDataReady;
  (void)waitCredit;
  (void)abortDevice;
  (void)traceContext;
  (void)traceState;
  (void)qpLane;
  return kProgressReady;
#endif
}

// Decode one ready recv chunk from recv-staging into dst (Simple: contiguous
// CopyOp::recv).
template <typename CopyOp = Memcpy, typename... Args>
__device__ __forceinline__ void progress_recv_consume_buf(
    protocol::Simple,
    ThreadGroup& group,
    const IbChannelLayout& channelLayout,
    const ProgressChunk& chunk,
    void* __restrict__ dst,
    std::size_t payloadBytes,
    const PipesTraceAllReduceContext* traceContext,
    uint8_t qpLane,
    Args... args) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const std::size_t validBytes =
      valid_payload_bytes(chunk.dataOff, chunk.payloadBytes, payloadBytes);
  if (validBytes > 0) {
    if (group.is_leader()) {
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceReduceCopyBegin,
          qpLane,
          validBytes);
    }
    CopyOp::recv(
        static_cast<char*>(dst) + chunk.dataOff,
        channelLayout.recvStagingPtr + chunk.stagingOff,
        validBytes,
        group,
        chunk.dataOff,
        args...);
    if (group.is_leader()) {
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceReduceCopyEnd,
          qpLane,
          validBytes);
    }
  }
#else
  (void)group;
  (void)channelLayout;
  (void)chunk;
  (void)dst;
  (void)payloadBytes;
  (void)traceContext;
  (void)qpLane;
  ((void)args, ...);
#endif
}

// Non-blocking readiness check for one recv chunk (LL: cooperative,
// non-spinning poll of every packet's inline flag == chunk.flagVal, then a
// group AND-reduce). There is no DATA_READY counter for LL, so
// transport/localChannel/waitCredit are unused. The trace parameters are part
// of the shared seam that progress_recv_once_impl calls through; LL emits no
// trace events, so they are accepted and ignored.
//
// Landing constraint: leaving localChannel untouched here skips the
// receiver-side mirror advance that IbgdaBuffer.h documents as a cross-protocol
// invariant -- put_impl bumps the send cursor unconditionally, so without a
// matching receiver-side advance the round-robin lane mapping desyncs and a
// later Simple collective on the same channel can wait on a lane the sender
// never used. D115669516 completes that advance; it must land in the same batch
// as this diff and the ones that enable LL progress on top of it.
template <typename Transport>
__device__ __forceinline__ uint32_t progress_recv_ready(
    protocol::LL,
    Transport& transport,
    ThreadGroup& group,
    const IbChannelLayout& channelLayout,
    const ProgressChunk& chunk,
    IbLocalChannel& localChannel,
    const IbgdaLocalBuffer& localDataReady,
    uint64_t waitCredit,
    const AbortDevice& abortDevice,
    const PipesTraceAllReduceContext* traceContext,
    PipesTraceProgressState* traceState,
    uint8_t qpLane) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  using P = LlxPacket<4, 4>;
  (void)transport;
  (void)localDataReady;
  (void)waitCredit;
  (void)traceContext;
  (void)traceState;
  (void)qpLane;
  const char* staging = channelLayout.recvStagingPtr + chunk.stagingOff;
  // Packet geometry lives in LLImpl, which owns the format; this seam only
  // decides what to do with the verdict -- report not-ready on false, rather
  // than spin.
  // `all_flags_set` is group-collective, so every thread evaluates it; the
  // verdict below is then decided once by the leader and broadcast.
  const bool flagsSet = LLImpl<P>::all_flags_set(
      group,
      staging,
      P::max_payload(chunk.wireBytes),
      static_cast<typename P::FlagType>(chunk.flagVal));
  // Only the leader checks, so the answer has to be broadcast before it leaves:
  // a per-thread verdict here would split the group at the caller's next
  // collective step.
  //
  // The abort is asked *before* readiness, and the ready path now leaves
  // through the same single broadcast as the not-ready path rather than
  // returning early. Previously a chunk whose flags were already set returned
  // `kProgressReady` without ever consulting the abort, and the caller went on
  // to consume the packet and credit SLOT_FREE. One broadcast on every path is
  // the floor here -- unlike Simple, LL's ready path had no existing barrier to
  // fold into.
  uint32_t status = kProgressNotReady;
  if (group.is_leader()) {
    if (FT_ABORT_CHECK(
            abortDevice,
            "progress_recv_once(LL) waiting for packet flags flagVal=%llu, "
            "wireBytes=%llu",
            static_cast<unsigned long long>(chunk.flagVal),
            static_cast<unsigned long long>(chunk.wireBytes))) {
      status = kProgressAborted;
    } else if (flagsSet) {
      // LL carries no DATA_READY, but its put still advanced the sender's
      // IbQpState::cursor -- select_put_lane_ordinal() increments that cursor
      // on every put regardless of protocol, and it is channel-scoped, not
      // slot-scoped. recvDataReadyLaneCursor mirrors it, so the mirror has to
      // advance here too: consuming an LL chunk without it leaves the two one
      // chunk apart per LL transfer, and Simple's next receive on this channel
      // then waits on a lane the sender never wrote. Matches the unconditional
      // bump in poll_recv_data_ready() (a single lane makes it a no-op).
      ++localChannel.recvDataReadyLaneCursor;
      status = kProgressReady;
    }
  }
  return group.broadcast<uint32_t>(status);
#else
  (void)transport;
  (void)group;
  (void)channelLayout;
  (void)chunk;
  (void)localChannel;
  (void)localDataReady;
  (void)waitCredit;
  (void)abortDevice;
  return kProgressReady;
#endif
}

// Decode one ready recv chunk from recv-staging into dst via the CopyOp's
// packet-aware recvLL<P> hook (Memcpy delegates to LLImpl::unpack; a
// reduce/convert op sums each packet's payload into the accumulator).
// progress_recv_ready already confirmed every packet flag == chunk.flagVal, so
// the decode does not spin. Mirrors the blocking LL path
// (consumeRecvBuf(protocol::LL)) -- this is the CopyOp dispatch point for the
// LL progress recv path.
//
// traceContext/qpLane are named rather than swept into `Args...`: the pack is
// forwarded to the CopyOp, so leaving them unnamed would silently hand the
// reduce op two extra arguments instead of failing to compile.
template <typename CopyOp = Memcpy, typename... Args>
__device__ __forceinline__ void progress_recv_consume_buf(
    protocol::LL,
    ThreadGroup& group,
    const IbChannelLayout& channelLayout,
    const ProgressChunk& chunk,
    void* __restrict__ dst,
    std::size_t payloadBytes,
    const PipesTraceAllReduceContext* traceContext,
    uint8_t qpLane,
    Args... args) {
  using P = LlxPacket<4, 4>;
  (void)traceContext;
  (void)qpLane;
  static_assert(
      has_recvLL_v<CopyOp, P>,
      "LL progress recv requires a CopyOp with a packet-aware recvLL<P>(); "
      "Memcpy provides one. A reduce/convert CopyOp must supply its own -- a "
      "plain contiguous copy cannot address the data+flag interleaved staging");
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const std::size_t validBytes =
      valid_payload_bytes(chunk.dataOff, chunk.payloadBytes, payloadBytes);
  // progress_recv_ready() above already confirmed every packet flag equals
  // chunk.flagVal, so unpack's readiness spin exits on its first load and a
  // disabled AbortDevice cannot hang here. If that pre-check ever stops being
  // exhaustive, thread progress_recv_once()'s abortDevice through instead.
  CopyOp::template recvLL<P>(
      group,
      static_cast<char*>(dst) + chunk.dataOff,
      channelLayout.recvStagingPtr + chunk.stagingOff,
      validBytes,
      chunk.dataOff,
      static_cast<typename P::FlagType>(chunk.flagVal),
      AbortDevice(),
      args...);
#else
  (void)group;
  (void)channelLayout;
  (void)chunk;
  (void)dst;
  (void)payloadBytes;
  ((void)args, ...);
#endif
}

template <typename Transport, typename CopyOp, typename Proto, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus progress_recv_once_impl(
    Transport& transport,
    ThreadGroup& group,
    const AbortDevice& abortDevice,
    const PipesTraceAllReduceContext* traceContext,
    PipesTraceProgressState* traceState,
    Args... args) {
  // Mirror of progress_send_once: the progress API is fixed-size only. A
  // variable-size policy would mis-size the DATA_READY/SLOT_FREE protocol and
  // corrupt the stream; use the blocking recv() instead. See D108485978 review.
  static_assert(
      !detail::copyop_variable_size_v<CopyOp>,
      "progress_recv_once() supports fixed-size CopyOps only; use the "
      "blocking recv() for variable-size CopyOps such as AnsCompress.");
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef __HIP_PLATFORM_AMD__
  static_assert(
      sizeof(CopyOp) == 0, "detail::progress_recv_once() requires NVIDIA GPU");
#endif
  auto& channelLayout = transport.channel_layout();
  auto& progressSlot = progress_recv_slot<Proto>(transport, group);
  IbChannelProgress state = progressSlot;
  if (state.activeStage == detail::IbSendRecvProgressStage::Done) {
    return IbgdaSendRecvProgressStatus::Done;
  }
  const ProgressGeometry progress_params = make_progress_geometry<Proto>(
      channelLayout,
      group,
      state.activeUserBytes,
      state.activeMaxSignalBytes,
      "progress_recv_once");
  if (active_payload_offset(state) >= progress_params.protocolBytes) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: progress_recv_once payloadOffset=%llu >= "
          "protocolBytes=%llu without Done stage\n",
          static_cast<unsigned long long>(active_payload_offset(state)),
          static_cast<unsigned long long>(progress_params.protocolBytes));
    }
    PIPES_DEVICE_TRAP();
  }
  validate_recv_progress_stage(group, state);

  const ProgressChunk chunk =
      next_chunk<Proto>(channelLayout, state, progress_params);
  const bool isFinalChunk =
      chunk.dataOff + chunk.payloadBytes >= progress_params.protocolBytes;
  const std::size_t protocolBytesThis = chunk.wireBytes +
      (isFinalChunk ? Proto::wire_bytes(state.activeTailPadding) : 0);
  const ChannelSlotView ch =
      acquire_channel<Proto>(transport, channelLayout, group);
  IbLocalChannel& localChannel = ch.channel;
  const IbgdaLocalBuffer localDataReady = ch.local.dataReady;
  const IbRemoteChannel remoteChannel = ch.remote;
  const uint32_t numLanes = static_cast<uint32_t>(channelLayout.numLanes);
  const uint8_t qpLane = static_cast<uint8_t>(
      numLanes == 0 ? 0 : localChannel.recvDataReadyLaneCursor % numLanes);
  const uint32_t recvReadiness = progress_recv_ready(
      Proto{},
      transport,
      group,
      channelLayout,
      chunk,
      localChannel,
      localDataReady,
      protocolBytesThis,
      abortDevice,
      traceContext,
      traceState,
      qpLane);
  if (recvReadiness != kProgressReady) {
    if (recvReadiness == kProgressAborted) {
      abandon_progress_state(group, progressSlot, state);
      return IbgdaSendRecvProgressStatus::Aborted;
    }
    return IbgdaSendRecvProgressStatus::Waiting;
  }

  progress_recv_consume_buf<CopyOp>(
      Proto{},
      group,
      channelLayout,
      chunk,
      state.activeUserBuf,
      progress_params.payloadBytes,
      traceContext,
      qpLane,
      args...);
  group.sync();

  if (group.is_leader()) {
    trace_allreduce_event(
        traceContext,
        PipesTraceEventType::kAllReduceBookkeepingBegin,
        qpLane,
        protocolBytesThis);
  }
  transport.signal(
      group, remoteChannel.slotFree, protocolBytesThis, IbDirection::Recv);

  state.activeNextByte += chunk.payloadBytes;
  if (active_payload_offset(state) >= progress_params.protocolBytes) {
    transition_progress_stage(
        group, state, detail::IbSendRecvProgressStage::Done);
    store_progress_state(group, progressSlot, state);
    if (group.is_leader()) {
      trace_allreduce_event(
          traceContext,
          PipesTraceEventType::kAllReduceBookkeepingEnd,
          qpLane,
          protocolBytesThis);
    }
    return IbgdaSendRecvProgressStatus::Done;
  }

  store_progress_state(group, progressSlot, state);
  if (group.is_leader()) {
    trace_allreduce_event(
        traceContext,
        PipesTraceEventType::kAllReduceBookkeepingEnd,
        qpLane,
        protocolBytesThis);
  }
  return IbgdaSendRecvProgressStatus::Progressed;
#else
  (void)transport;
  (void)group;
  (void)abortDevice;
  (void)traceContext;
  (void)traceState;
  return IbgdaSendRecvProgressStatus::Done;
#endif
}

template <typename Transport, typename CopyOp, typename Proto, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus progress_recv_once(
    Transport& transport,
    ThreadGroup& group,
    const AbortDevice& abortDevice,
    Args... args) {
  return progress_recv_once_impl<Transport, CopyOp, Proto>(
      transport, group, abortDevice, nullptr, nullptr, args...);
}

template <typename Transport, typename CopyOp, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus
progress_recv_once_with_trace(
    Transport& transport,
    ThreadGroup& group,
    const AbortDevice& abortDevice,
    const PipesTraceAllReduceContext& traceContext,
    PipesTraceProgressState& traceState,
    Args... args) {
  return progress_recv_once_impl<Transport, CopyOp, protocol::Simple>(
      transport, group, abortDevice, &traceContext, &traceState, args...);
}

/**
 * Acquire one landed recv chunk WITHOUT consuming it.
 *
 * progress_recv_once() fuses wait, CopyOp and SLOT_FREE into one call, which
 * forces a caller to finish with a chunk before it can look at the next peer.
 * A reduce-scatter that sums N peers into one accumulator needs the opposite:
 * hold all N peers' chunks for the same byte range at once, reduce across them
 * in registers, and only then release all N slots. Splitting the wait from the
 * release is what makes that possible; fusing them would force an accumulator
 * round-trip through HBM per peer.
 *
 * Advances the progress cursor, so successive calls hand back successive
 * chunks and several acquisitions may be outstanding at once. The slot stays
 * owned by the caller until progress_recv_release_once() returns its credit;
 * every acquire that returns Progressed must eventually be released, or the
 * sender starves for SLOT_FREE.
 *
 * @return Waiting if the chunk has not landed, Progressed if `out` is valid,
 *         Done if the stream is already complete.
 */
template <typename Transport, typename Proto>
__device__ __forceinline__ IbgdaSendRecvProgressStatus
progress_recv_acquire_once(
    Transport& transport,
    ThreadGroup& group,
    const AbortDevice& abortDevice,
    RecvChunkAcquisition& out) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  auto& channelLayout = transport.channel_layout();
  auto& progressSlot = progress_recv_slot<Proto>(transport, group);
  IbChannelProgress state = progressSlot;
  if (state.activeStage == detail::IbSendRecvProgressStage::Done) {
    return IbgdaSendRecvProgressStatus::Done;
  }
  const ProgressGeometry geometry = make_progress_geometry<Proto>(
      channelLayout,
      group,
      state.activeUserBytes,
      state.activeMaxSignalBytes,
      "progress_recv_acquire_once");
  // Parity with progress_recv_once: a cursor past the end of the stream while
  // the stage is not Done means the progress state is corrupted or
  // desynchronised from the sender. Should be unreachable given the Done
  // transition below, but this is the diagnostic that catches that bug class on
  // the fused path, and the split path is no less prone to it.
  if (active_payload_offset(state) >= geometry.protocolBytes) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: progress_recv_acquire_once payloadOffset=%llu >= "
          "protocolBytes=%llu without Done stage\n",
          static_cast<unsigned long long>(active_payload_offset(state)),
          static_cast<unsigned long long>(geometry.protocolBytes));
    }
    PIPES_DEVICE_TRAP();
  }
  validate_recv_progress_stage(group, state);

  const ProgressChunk chunk = next_chunk<Proto>(channelLayout, state, geometry);
  const bool isFinalChunk =
      chunk.dataOff + chunk.payloadBytes >= geometry.protocolBytes;
  const std::size_t protocolBytesThis = chunk.wireBytes +
      (isFinalChunk ? Proto::wire_bytes(state.activeTailPadding) : 0);
  const ChannelSlotView ch =
      acquire_channel<Proto>(transport, channelLayout, group);
  // qpLane mirrors progress_recv_once_impl: DATA_READY is delivered round-robin
  // across lanes, so the readiness poll must look at the lane that carried THIS
  // chunk. Tracing is not plumbed through the acquire/release path, so the
  // trace context and state are null here.
  const uint32_t numLanes = static_cast<uint32_t>(channelLayout.numLanes);
  const uint8_t qpLane = static_cast<uint8_t>(
      numLanes == 0 ? 0 : ch.channel.recvDataReadyLaneCursor % numLanes);
  const uint32_t recvReadiness = progress_recv_ready(
      Proto{},
      transport,
      group,
      channelLayout,
      chunk,
      ch.channel,
      ch.local.dataReady,
      protocolBytesThis,
      abortDevice,
      nullptr,
      nullptr,
      qpLane);
  if (recvReadiness != kProgressReady) {
    if (recvReadiness == kProgressAborted) {
      abandon_progress_state(group, progressSlot, state);
      return IbgdaSendRecvProgressStatus::Aborted;
    }
    return IbgdaSendRecvProgressStatus::Waiting;
  }

  out.staging = channelLayout.recvStagingPtr + chunk.stagingOff;
  out.validBytes = valid_payload_bytes(
      chunk.dataOff, chunk.payloadBytes, geometry.payloadBytes);
  out.dataOff = chunk.dataOff;
  out.protocolBytes = protocolBytesThis;

  // Advance the cursor here, not in release: successive acquires must hand
  // back successive chunks so a caller can hold several at once. The slot is
  // still owned by the caller until it releases this view.
  state.activeNextByte += chunk.payloadBytes;
  if (active_payload_offset(state) >= geometry.protocolBytes) {
    transition_progress_stage(
        group, state, detail::IbSendRecvProgressStage::Done);
  }
  store_progress_state(group, progressSlot, state);
  return IbgdaSendRecvProgressStatus::Progressed;
#else
  (void)transport;
  (void)group;
  (void)abortDevice;
  (void)out;
  return IbgdaSendRecvProgressStatus::Done;
#endif
}

/**
 * Return the SLOT_FREE credit for a chunk a previous
 * progress_recv_acquire_once() handed back.
 *
 * The caller MUST have finished reading `view.staging` and synchronised every
 * thread that read it first: this frees the slot for the sender to overwrite,
 * so releasing early corrupts data still being consumed. Does not touch the
 * progress cursor -- acquire already advanced it -- so several acquisitions may
 * be outstanding at once.
 *
 * Release them in ACQUISITION ORDER per channel. SLOT_FREE is a cumulative byte
 * counter and the sender gates on `current >= streamEnd - pipelineWindow`, so
 * the credit records how many bytes were freed, never which chunk. An
 * out-of-order release can therefore satisfy the sender's threshold while an
 * older chunk is still being read, and the sender then overwrites that slot --
 * silent data corruption, not a fault. At pipelineDepth 2 with equal chunks,
 * releasing chunk 2 before chunk 1 already lets chunk 3 land on chunk 1's slot.
 * Supporting genuine out-of-order release would need per-slot credits instead
 * of one counter; no caller needs that today.
 */
template <typename Transport, typename Proto>
__device__ __forceinline__ void progress_recv_release_once(
    Transport& transport,
    ThreadGroup& group,
    const AbortDevice& abortDevice,
    const RecvChunkAcquisition& view) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (view.staging == nullptr) {
    return;
  }
  // SLOT_FREE is a peer-visible credit: it tells the sender this range has been
  // consumed and may be overwritten. After an abort the consumption either did
  // not happen or is not trustworthy, so crediting it releases a peer that is
  // correctly blocked -- principle 4. This function had no abort parameter at
  // all and signalled unconditionally, which is why the split-receive path
  // stayed uncontained while the fused one was fixed.
  //
  // Leader-decides-and-broadcasts, because a subset skipping the signal while
  // the rest enter `transport.signal()` would split the group.
  uint32_t aborted = 0;
  if (group.is_leader()) {
    aborted = FT_ABORT_CHECK(
                  abortDevice, "recv slot release on an aborted communicator")
        ? 1U
        : 0U;
  }
  if (group.broadcast<uint32_t>(aborted) != 0U) {
    return;
  }
  auto& channelLayout = transport.channel_layout();
  const ChannelSlotView ch =
      acquire_channel<Proto>(transport, channelLayout, group);
  transport.signal(
      group, ch.remote.slotFree, view.protocolBytes, IbDirection::Recv);
#else
  (void)transport;
  (void)group;
  (void)abortDevice;
  (void)view;
#endif
}

/**
 * Commit an updated local progress state back to its transport-owned slot.
 * The trailing sync orders the leader's store before later group work.
 */
__device__ __forceinline__ void store_progress_state(
    ThreadGroup& group,
    IbChannelProgress& slot,
    const IbChannelProgress& state) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (group.is_leader()) {
    slot.activeNextByte = state.activeNextByte;
    slot.activeTailPadding = state.activeTailPadding;
    slot.activeBaseStep = state.activeBaseStep;
    slot.activeUserBytes = state.activeUserBytes;
    slot.activeMaxSignalBytes = state.activeMaxSignalBytes;
    slot.activeUserBuf = state.activeUserBuf;
    slot.activeStage = state.activeStage;
  }
  group.sync();
#else
  (void)group;
  (void)slot;
  (void)state;
#endif
}

/**
 * Drive an aborted progress slot to its terminal stage.
 *
 * A transfer that unwinds on abort leaves the slot mid-state machine, and
 * `assert_progress_slot_idle()` traps on anything but `Done`. The next kernel
 * queued on this channel would therefore trap inside init_send_progress() --
 * turning a clean abort into a device fault one kernel later. Marking the slot
 * terminal converts that into a clean exit, and keeps the collective path free
 * of abort bookkeeping: every later progress call on this slot short-circuits
 * to `Done`, so a driver loop simply drains and the kernel exits.
 *
 * This buys LIVENESS, not a usable channel. After an abort the DATA_READY /
 * SLOT_FREE counters are skewed against the peer's, so traffic that follows on
 * this channel is not meaningful -- which the abort contract already says
 * ("results from work that completes after an abort should be treated as the
 * reason for abort"). Recovery is still a `reconfigure()` that destroys and
 * rebuilds the transport with zeroed state. What changes is that the host now
 * reaches that point, instead of losing the CUDA context to a trap first.
 *
 * The reserved byte range is deliberately NOT returned to `slot.nextStep`. The
 * peer may still land an RDMA write into it, so the cursor stays advanced and
 * the range is abandoned rather than recycled.
 *
 * Bypasses transition_progress_stage() on purpose: abandoning is not a protocol
 * transition, and it is legal from every stage.
 */
__device__ __forceinline__ void abandon_progress_state(
    ThreadGroup& group,
    IbChannelProgress& slot,
    IbChannelProgress& state) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  state.activeStage = detail::IbSendRecvProgressStage::Done;
  state.activeNextByte = 0;
  state.activeTailPadding = 0;
  state.activeBaseStep = 0;
  store_progress_state(group, slot, state);
#else
  (void)group;
  (void)slot;
  (void)state;
#endif
}

/**
 * Validate and return the static staging geometry for one progress call.
 *
 * This helper is called by the public init APIs and each progress attempt. It
 * verifies that the group participates in this transfer and that the
 * requested active block count can fit at least one 16-byte-aligned region in
 * each staging slot.
 *
 * On success, `perBlockSlot` is the caller's partition within each logical
 * staging slot and `chunkSize` is the maximum signaled sub-chunk size. A zero
 * `maxSignalBytes` means one chunk per `perBlockSlot`; otherwise the value is
 * rounded down to the same 16-byte alignment used by the blocking send/recv
 * path. Invalid geometry traps on device because continuing would corrupt
 * another block group's staging partition.
 *
 * The public init methods handle zero-byte operations before calling this
 * helper. A progress call should only see zero bytes when its state is
 * already `Done`. Non-empty payload byte counts are rounded up to 16-byte
 * protocol counts here; copy callbacks still see only valid payload bytes.
 */
template <typename Proto>
__device__ __forceinline__ ProgressGeometry make_progress_geometry(
    const IbChannelLayout& channelLayout,
    ThreadGroup& group,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const char* opName) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (nbytes == 0) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: %s saw non-Done progress state for zero-byte "
          "transfer\n",
          opName);
    }
    PIPES_DEVICE_TRAP();
  }
  const int groupId = static_cast<int>(group.group_id);
  // LOGICAL bound -- see calcGeometry.
  const int numChannels = channelLayout.numChannels;
  if (groupId < 0 || groupId >= numChannels) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: %s group_id=%d >= numChannels=%d\n",
          opName,
          groupId,
          numChannels);
    }
    PIPES_DEVICE_TRAP();
  }

  // Chunks align to lcm(kData, widest reduced element), not to kData alone --
  // see calcGeometry for the full reasoning: a reducing CopyOp needs each chunk
  // to hold a whole number of elements, LL's kData of 4 splits a double/int64,
  // and the alignment must come from the protocol rather than the CopyOp so the
  // Memcpy send lane and the IbReduceCopy recv lane agree on the chunk
  // boundary. Both terms are payload bytes; this is NOT kPacketBytes, which is
  // a wire size. No-op for Simple (kData == 16 already covers 8).
  constexpr std::size_t kDataBytes = static_cast<std::size_t>(Proto::kData);
  static_assert(
      (kDataBytes & (kDataBytes - 1)) == 0,
      "kData must be a power of two, so that max() is its lcm with "
      "kMaxReducedTypeBytes");
  constexpr std::size_t kChunkAlign =
      kDataBytes > kMaxReducedTypeBytes ? kDataBytes : kMaxReducedTypeBytes;

  const std::size_t perBlockSlotWire = pipeline_chunk(channelLayout);
  const std::size_t perBlockSlotPayload = Proto::max_payload(perBlockSlotWire);
  // Widened from `== 0` to one aligned chunk unit: the chunk sizing below
  // rounds to kChunkAlign, so a slot that cannot hold one such unit would be
  // overrun by the smallest legal chunk. Same single branch as before, so this
  // costs nothing extra on the progress hot path.
  if (perBlockSlotPayload < kChunkAlign) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: %s perBlockSlotPayload=%llu holds no whole chunk "
          "unit (perChannelBufferSize=%llu, pipelineDepth=%d)\n",
          opName,
          (unsigned long long)perBlockSlotPayload,
          (unsigned long long)pipeline_window(channelLayout),
          channelLayout.pipelineDepth);
    }
    PIPES_DEVICE_TRAP();
  }
  const std::size_t perChannelBufferSize = pipeline_window(channelLayout);

  // Cursor + chunk sizing run in PAYLOAD bytes, aligned to whole packets
  // (Proto::kData); wire quantities are derived via Proto::wire_bytes. For
  // Simple (kData == 16) this is the original 16B alignment.
  const std::size_t protocolBytes =
      (nbytes + Proto::kData - 1) / Proto::kData * Proto::kData;
  // Guaranteed >= kChunkAlign by the slot check above.
  const std::size_t alignedSlotPayload =
      perBlockSlotPayload / kChunkAlign * kChunkAlign;
  std::size_t chunkPayload =
      (max_signal_bytes > 0 && max_signal_bytes < alignedSlotPayload)
      ? (max_signal_bytes / kChunkAlign * kChunkAlign)
      : alignedSlotPayload;
  if (chunkPayload == 0) {
    // Sub-unit request: clamp to one aligned unit, the finest granularity that
    // still holds whole packets and whole elements. See calcGeometry --
    // max_signal_bytes is a MAXIMUM, so neither the whole slot nor rounding up
    // would respect it.
    chunkPayload = kChunkAlign;
  }
  return ProgressGeometry{
      .groupId = groupId,
      .slotIndex = channelLayout.protoChannelSlot(groupId, Proto::kProtoSlot),
      .payloadBytes = nbytes,
      .protocolBytes = protocolBytes,
      .perBlockSlotWire = perBlockSlotWire,
      .perBlockSlotPayload = perBlockSlotPayload,
      .perChannelBufferSize = perChannelBufferSize,
      .chunkPayload = chunkPayload,
      .pipelineDepth = channelLayout.pipelineDepth,
  };
#else
  (void)channelLayout;
  (void)group;
  (void)nbytes;
  (void)max_signal_bytes;
  (void)opName;
  return {};
#endif
}

__device__ __forceinline__ std::size_t active_payload_offset(
    const IbChannelProgress& state) {
  return state.activeNextByte;
}

/**
 * Reserve a non-overlapping protocol byte range for one progress state.
 *
 * Blocking send()/recv() read the channel cursor at call entry and commit it
 * at completion. Progress init reserves immediately because operations may
 * complete across many bounded calls. The active byte cursor tracks payload
 * protocol bytes; final signals/counters carry any tail padding reserved here.
 */
__device__ __forceinline__ void reserve_progress_step(
    ThreadGroup& group,
    IbChannelProgress& slot,
    IbChannelProgress& state,
    const ProgressGeometry& geometry) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  uint64_t baseStep = 0;
  uint64_t protocolTailPadding = 0;
  if (group.is_leader()) {
    baseStep = static_cast<uint64_t>(slot.nextStep);
    protocolTailPadding = tail_padding_for_signal_granularity(
        baseStep,
        geometry.chunkPayload,
        geometry.perBlockSlotPayload,
        geometry.payloadBytes);
    slot.nextStep = static_cast<int64_t>(
        baseStep + geometry.protocolBytes + protocolTailPadding);
  }
  baseStep = group.broadcast<uint64_t>(baseStep);
  protocolTailPadding = group.broadcast<uint64_t>(protocolTailPadding);
  state.activeBaseStep = static_cast<int64_t>(baseStep);
  state.activeNextByte = 0;
  state.activeTailPadding = static_cast<std::size_t>(protocolTailPadding);
#else
  (void)group;
  (void)slot;
  (void)state;
  (void)geometry;
#endif
}

/**
 * Trap if a send progress state is not in a sender-owned stage.
 *
 * Without this check, corrupted or mismatched transport-owned state could
 * return `Waiting` forever because no send-side transition would match.
 * Trapping turns that misuse into an immediate device failure with a clear
 * diagnostic.
 */
__device__ __forceinline__ void validate_send_progress_stage(
    ThreadGroup& group,
    const IbChannelProgress& state) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (state.activeStage !=
          detail::IbSendRecvProgressStage::WaitLocalCompletion &&
      state.activeStage != detail::IbSendRecvProgressStage::WaitSlotFree &&
      state.activeStage != detail::IbSendRecvProgressStage::Done) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: progress_send_once invalid stage=%d\n",
          static_cast<int>(state.activeStage));
    }
    PIPES_DEVICE_TRAP();
  }
#endif
}

/**
 * Trap if a recv progress state is not in a receiver-owned stage.
 *
 * Receiver progress is only valid while waiting for DATA_READY. Sender
 * stages are protocol misuse and cannot make progress on the recv path.
 */
__device__ __forceinline__ void validate_recv_progress_stage(
    ThreadGroup& group,
    const IbChannelProgress& state) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (state.activeStage != detail::IbSendRecvProgressStage::WaitDataReady &&
      state.activeStage != detail::IbSendRecvProgressStage::Done) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: progress_recv_once invalid stage=%d\n",
          static_cast<int>(state.activeStage));
    }
    PIPES_DEVICE_TRAP();
  }
#endif
}

/**
 * Return whether `current -> next` is a valid progress-state transition.
 */
__device__ __forceinline__ bool is_valid_progress_transition(
    detail::IbSendRecvProgressStage current,
    detail::IbSendRecvProgressStage next) {
  switch (current) {
    case detail::IbSendRecvProgressStage::WaitLocalCompletion:
      return next == detail::IbSendRecvProgressStage::WaitSlotFree;
    case detail::IbSendRecvProgressStage::WaitSlotFree:
      return next == detail::IbSendRecvProgressStage::WaitLocalCompletion ||
          next == detail::IbSendRecvProgressStage::Done;
    case detail::IbSendRecvProgressStage::WaitDataReady:
      return next == detail::IbSendRecvProgressStage::Done;
    case detail::IbSendRecvProgressStage::Done:
      return false;
    case detail::IbSendRecvProgressStage::Busy:
      return false;
  }
  return false;
}

/**
 * Apply one legal progress-state transition.
 *
 * The explicit transition table keeps the send/recv state machine local and
 * auditable. If future progress states are added, this switch must opt into
 * each new legal edge instead of allowing silent fallthrough.
 */
__device__ __forceinline__ void transition_progress_stage(
    ThreadGroup& group,
    IbChannelProgress& state,
    detail::IbSendRecvProgressStage next) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const detail::IbSendRecvProgressStage current = state.activeStage;
  if (!is_valid_progress_transition(current, next)) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: invalid progress transition stage=%d -> %d\n",
          static_cast<int>(current),
          static_cast<int>(next));
    }
    PIPES_DEVICE_TRAP();
  }
  state.activeStage = next;
#endif
}

/**
 * Map the state's logical protocol byte cursor to the next staging-ring
 * chunk.
 *
 * The transport stores each channel as one contiguous pipeline window split
 * into `pipelineDepth` slots. The protocol cursor advances in bytes, not
 * slots, so `baseStep + nextByte` is reduced modulo the per-channel window
 * to pick the slot and offset.
 *
 * The returned chunk is clipped by three boundaries: the configured
 * sub-chunk size, remaining protocol bytes, and remaining bytes in the
 * current per-channel staging slice. This keeps every progress call
 * bounded and prevents a single RDMA put or recv copy from spanning two
 * staging slots.
 */
template <typename Proto>
__device__ __forceinline__ ProgressChunk next_chunk(
    const IbChannelLayout& channelLayout,
    const IbChannelProgress& state,
    const ProgressGeometry& geometry) {
  // The ring cursor advances in PAYLOAD bytes; staging offsets, RDMA lengths,
  // and readiness thresholds are derived in WIRE bytes via Proto::wire_bytes
  // (1:1 for Simple, kPacketBytes:kData for LL).
  const uint64_t streamStart =
      static_cast<uint64_t>(state.activeBaseStep) + state.activeNextByte;
  (void)channelLayout;
  const std::size_t pipelineBytesPayload =
      Proto::max_payload(geometry.perChannelBufferSize);
  const std::size_t pipelineOff =
      static_cast<std::size_t>(streamStart % pipelineBytesPayload);
  const int slot = static_cast<int>(pipelineOff / geometry.perBlockSlotPayload);
  const std::size_t chunkOff = pipelineOff -
      static_cast<std::size_t>(slot) * geometry.perBlockSlotPayload;
  const std::size_t slotRemaining = geometry.perBlockSlotPayload - chunkOff;
  const std::size_t payloadNextByte = active_payload_offset(state);
  const std::size_t dataRemaining = geometry.protocolBytes - payloadNextByte;
  std::size_t payloadBytes = geometry.chunkPayload < dataRemaining
      ? geometry.chunkPayload
      : dataRemaining;
  payloadBytes = payloadBytes < slotRemaining ? payloadBytes : slotRemaining;
  return ProgressChunk{
      .stagingOff = static_cast<std::size_t>(geometry.slotIndex) *
              geometry.perChannelBufferSize +
          static_cast<std::size_t>(slot) * geometry.perBlockSlotWire +
          Proto::wire_bytes(chunkOff),
      .dataOff = payloadNextByte,
      .payloadBytes = payloadBytes,
      .wireBytes = Proto::wire_bytes(payloadBytes),
      .streamEndWire =
          Proto::wire_bytes(streamStart) + Proto::wire_bytes(payloadBytes),
      .flagVal = streamStart / pipelineBytesPayload + 1,
      .slotId = static_cast<uint32_t>(slot),
      .pipelineGeneration = streamStart / pipelineBytesPayload,
  };
}

template <typename P, typename Transport>
__device__ __forceinline__ uint32_t try_prepare_send_slot(
    Transport& transport,
    ThreadGroup& group,
    uint32_t slotId,
    uint64_t generation,
    const AbortDevice& abortDevice) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  uint32_t ready = kProgressReady;
  if (group.is_leader()) {
    // Consulted on the ready path too, not just when the slot is busy.
    //
    // This function already broadcasts its verdict, so folding the abort into
    // it costs no extra barrier -- which is the whole point: a standalone entry
    // guard would add a group-wide broadcast to every healthy progress attempt,
    // and Tree scheduling and batched send/recv call this in tight loops.
    //
    // Checking only inside the not-ready branch below is what left the hole: a
    // call that finds the slot already free never looked at the abort and went
    // on to stage the payload and issue the put with its fused DATA_READY.
    if (FT_ABORT_CHECK(
            abortDevice, "send slot preparation on an aborted communicator")) {
      ready = kProgressAborted;
    } else {
      auto& slot = transport.template local_channel_slot<P>(group.group_id)
                       .sendCompletionSlots[slotId];
      if (slot.generation != generation) {
        uint64_t pending = slot.laneMask;
        const uint32_t numLanes = transport.send_completion_lane_count();
        for (uint32_t laneId = 0; laneId < numLanes; ++laneId) {
          const uint64_t laneBit = 1ULL << laneId;
          if ((pending & laneBit) == 0) {
            continue;
          }
          const IbLocalCompletionTicket ticket{
              .completionId = laneId,
              .value = slot.values[laneId],
          };
          if (transport.is_local_completion_ready(
                  group.group_id, ticket, abortDevice)) {
            pending &= ~laneBit;
          }
        }
        slot.laneMask = pending;
        if (pending == 0) {
          slot.generation = generation;
        } else {
          ready = kProgressNotReady;
          if (FT_ABORT_CHECK(
                  abortDevice,
                  "send slot local completion timed out slot=%u generation=%llu "
                  "pending=0x%llx",
                  slotId,
                  static_cast<unsigned long long>(generation),
                  static_cast<unsigned long long>(pending))) {
            ready = kProgressAborted;
          }
        }
      }
    }
  }
  return group.broadcast<uint32_t>(ready);
#else
  (void)transport;
  (void)group;
  (void)slotId;
  (void)generation;
  (void)abortDevice;
  return kProgressReady;
#endif
}

} // namespace detail
} // namespace comms::prims
