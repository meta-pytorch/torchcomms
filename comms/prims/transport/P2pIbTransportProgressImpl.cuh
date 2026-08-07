// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/prims/transport/P2pIbTransportDeviceImpl.cuh"

namespace comms::prims {
namespace detail {

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
  std::size_t stagingOff;
  std::size_t dataOff;
  std::size_t bytes;
  uint64_t streamEnd;
  uint32_t slotId;
  uint64_t pipelineGeneration;
};

/**
 * Register-only geometry for one resumable progress call.
 *
 * This is intentionally not stored in `IbChannelProgress`: callers pass the
 * same static geometry to init and progress, and each progress call derives
 * these values in registers instead of reloading duplicated fields from
 * HBM-backed progress state.
 */
struct ProgressGeometry {
  int groupId;
  std::size_t payloadBytes;
  std::size_t protocolBytes;
  std::size_t perBlockSlot;
  std::size_t perChannelBufferSize;
  std::size_t chunkSize;
  int pipelineDepth;
};

__device__ __forceinline__ void store_progress_state(
    ThreadGroup& group,
    IbChannelProgress& slot,
    const IbChannelProgress& state);

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

__device__ __forceinline__ ProgressChunk next_chunk(
    const IbChannelLayout& channelLayout,
    const IbChannelProgress& state,
    const ProgressGeometry& geometry);

template <typename Transport>
__device__ __forceinline__ bool try_prepare_send_slot(
    Transport& transport,
    ThreadGroup& group,
    uint32_t slotId,
    uint64_t generation,
    const Timeout& timeout = Timeout());

/**
 * Initialize transport-owned state for one pipelined send operation.
 *
 * The transport reserves the sender-side byte stream for `group.group_id`
 * and starts the internal state in the sender state machine unless
 * `nbytes == 0`. It does not capture the source pointer; callers pass the
 * pointer to each `progress_send_once()` call.
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
 * @param nbytes Number of user-buffer bytes to send for this group.
 * @param max_signal_bytes Maximum signaled sub-chunk size, or 0 for default.
 */
template <typename Transport>
__device__ __forceinline__ void init_send_progress(
    Transport& transport,
    ThreadGroup& group,
    std::size_t nbytes,
    std::size_t max_signal_bytes) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  auto& channelLayout = transport.channel_layout();
  auto& slot = progress_send_slot(transport, group);
  assert_progress_slot_idle(group, slot, "send");
  IbChannelProgress state{};
  state.activeStage = nbytes == 0
      ? detail::IbSendRecvProgressStage::Done
      : detail::IbSendRecvProgressStage::WaitLocalCompletion;
  if (nbytes == 0) {
    store_progress_state(group, slot, state);
    return;
  }
  // Validate the transfer before reserving the transport byte cursor.
  const ProgressGeometry geometry = make_progress_geometry(
      channelLayout, group, nbytes, max_signal_bytes, "init_send_progress");
  reserve_progress_step(group, slot, state, geometry);
  store_progress_state(group, slot, state);
#else
  (void)transport;
  (void)group;
  (void)nbytes;
  (void)max_signal_bytes;
#endif
}

/**
 * Initialize transport-owned state for one pipelined recv operation.
 *
 * The transport reserves the receiver-side byte stream for `group.group_id`
 * and starts the internal state in the receiver state machine unless
 * `nbytes == 0`. It does not capture the destination pointer; callers pass
 * the pointer to each `progress_recv_once()` call.
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
 * @param nbytes Number of user-buffer bytes to receive for this group.
 * @param max_signal_bytes Maximum signaled sub-chunk size, or 0 for default.
 */
template <typename Transport>
__device__ __forceinline__ void init_recv_progress(
    Transport& transport,
    ThreadGroup& group,
    std::size_t nbytes,
    std::size_t max_signal_bytes) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  auto& channelLayout = transport.channel_layout();
  auto& slot = progress_recv_slot(transport, group);
  assert_progress_slot_idle(group, slot, "recv");
  IbChannelProgress state{};
  state.activeStage = nbytes == 0
      ? detail::IbSendRecvProgressStage::Done
      : detail::IbSendRecvProgressStage::WaitDataReady;
  if (nbytes == 0) {
    store_progress_state(group, slot, state);
    return;
  }
  // Validate the transfer before reserving the transport byte cursor.
  const ProgressGeometry geometry = make_progress_geometry(
      channelLayout, group, nbytes, max_signal_bytes, "init_recv_progress");
  reserve_progress_step(group, slot, state, geometry);
  store_progress_state(group, slot, state);
#else
  (void)transport;
  (void)group;
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
 * can try another independent lane. If a `Timeout` is enabled, it is checked
 * only at those readiness points and should already have been started by the
 * caller.
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
 * @param src Source user buffer. The range `[src, src + nbytes)` must remain
 *            valid until `Done`.
 * @param nbytes Number of user-buffer bytes from the matching init call.
 * @param max_signal_bytes Maximum signaled sub-chunk size from init.
 * @param timeout Optional device timeout checked while dependencies wait.
 * @param args Additional arguments forwarded to `CopyOp::send`.
 */
template <typename Transport, typename CopyOp, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus progress_send_once(
    Transport& transport,
    ThreadGroup& group,
    const void* __restrict__ src,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
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
  auto& progressSlot = progress_send_slot(transport, group);
  IbChannelProgress state = progressSlot;
  if (state.activeStage == detail::IbSendRecvProgressStage::Done) {
    return IbgdaSendRecvProgressStatus::Done;
  }
  const ProgressGeometry progress_params = make_progress_geometry(
      channelLayout, group, nbytes, max_signal_bytes, "progress_send_once");
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
  const std::size_t pipelineBytes = progress_params.perBlockSlot *
      static_cast<std::size_t>(channelLayout.pipelineDepth);
  IbLocalChannel& localChannel =
      transport.local_channel(static_cast<uint32_t>(progress_params.groupId));
  const IbgdaLocalBuffer localSlotFree = localChannel.slotFree;
  const IbRemoteChannel remoteChannel =
      makeIbRemoteChannel(channelLayout, progress_params.groupId);

  if (state.activeStage ==
      detail::IbSendRecvProgressStage::WaitLocalCompletion) {
    const ProgressChunk chunk =
        next_chunk(channelLayout, state, progress_params);
    if (!try_prepare_send_slot(
            transport,
            group,
            chunk.slotId,
            chunk.pipelineGeneration,
            timeout)) {
      return IbgdaSendRecvProgressStatus::Waiting;
    }

    const std::size_t validBytes = valid_payload_bytes(
        chunk.dataOff, chunk.bytes, progress_params.payloadBytes);
    if (validBytes > 0) {
      CopyOp::send(
          channelLayout.sendStagingPtr + chunk.stagingOff,
          static_cast<const char*>(src) + chunk.dataOff,
          validBytes,
          group,
          chunk.dataOff,
          args...);
    }
    group.sync();
    transition_progress_stage(
        group, state, detail::IbSendRecvProgressStage::WaitSlotFree);
  }

  if (state.activeStage == detail::IbSendRecvProgressStage::WaitSlotFree) {
    const ProgressChunk chunk =
        next_chunk(channelLayout, state, progress_params);
    const bool isFinalChunk =
        chunk.dataOff + chunk.bytes >= progress_params.protocolBytes;
    const uint64_t protocolStreamEnd =
        chunk.streamEnd + (isFinalChunk ? state.activeTailPadding : 0);
    if (protocolStreamEnd > pipelineBytes) {
      const uint64_t expected = protocolStreamEnd - pipelineBytes;
      uint32_t ready = 1;
      unsigned long long current = 0;
      if (group.is_leader()) {
        current = static_cast<unsigned long long>(
            transport.read_signal(localSlotFree));
        ready = current >= expected ? 1U : 0U;
        if (!ready) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "progress_send_once waiting for SLOT_FREE expected>=%llu, "
              "current=%llu",
              static_cast<unsigned long long>(expected),
              current);
        }
      }
      ready = group.broadcast<uint32_t>(ready);
      if (!ready) {
        if (state.activeStage != initialStage ||
            state.activeNextByte != initialNextByte) {
          store_progress_state(group, progressSlot, state);
          return IbgdaSendRecvProgressStatus::Progressed;
        }
        return IbgdaSendRecvProgressStatus::Waiting;
      }
    }

    group.sync();
    if (group.is_leader()) {
      __threadfence_system();
      ThreadGroup solo{
          0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
      const std::size_t protocolBytesThis =
          chunk.bytes + (isFinalChunk ? state.activeTailPadding : 0);
      const auto completion = transport.put(
          solo,
          channelLayout.sendStagingBuf.subBuffer(chunk.stagingOff),
          remoteChannel.recvStaging.subBuffer(chunk.stagingOff),
          chunk.bytes,
          remoteChannel.dataReady,
          protocolBytesThis,
          /*counterBuf=*/{},
          /*counterVal=*/0,
          /*signalPerLane=*/true);
      record_send_completion(
          transport,
          static_cast<uint32_t>(progress_params.groupId),
          chunk.slotId,
          chunk.pipelineGeneration,
          completion);
    }
    group.sync();

    state.activeNextByte += chunk.bytes;
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
  (void)src;
  (void)nbytes;
  (void)max_signal_bytes;
  (void)timeout;
  return IbgdaSendRecvProgressStatus::Done;
#endif
}

/**
 * Non-blocking poll for one receive chunk's DATA_READY on its round-robin lane.
 *
 * Leader-only. Mirrors wait_recv_data_ready's readiness test without spinning:
 * returns true when the chunk's DATA_READY has landed on its lane, advancing
 * `recvDataReadyLaneCursor`/`recvLaneExpected` by exactly one chunk on that
 * (and only that) return. A false return leaves all receiver state untouched so
 * the caller can retry the same chunk on a later progress attempt.
 * `currentOut`/`expectedOut` are set for the caller's timeout diagnostic.
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
  // Truncate to 32 bits before the modulo to match the sender's uint32 cursor
  // wrap (see wait_recv_data_ready).
  const uint32_t lane =
      static_cast<uint32_t>(localChannel.recvDataReadyLaneCursor) % lanes;
  const uint64_t expected = localChannel.recvLaneExpected[lane] + chunkBytes;
  const IbgdaLocalBuffer laneBuf = localDataReady.subBuffer(
      sendRecvSignalSlotOffset(static_cast<int>(lane)));
  const uint64_t current = transport.read_signal(laneBuf);
  currentOut = static_cast<unsigned long long>(current);
  expectedOut = static_cast<unsigned long long>(expected);
  if (current < expected) {
    return false;
  }
  localChannel.recvLaneExpected[lane] = expected;
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
 * chunk, it returns `Waiting` immediately. If a `Timeout` is enabled, it is
 * checked only while the DATA_READY dependency is not ready and should
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
 * @param dst Destination user buffer. The range `[dst, dst + nbytes)` must
 *            remain valid until `Done`.
 * @param nbytes Number of user-buffer bytes from the matching init call.
 * @param max_signal_bytes Maximum signaled sub-chunk size from init.
 * @param timeout Optional device timeout checked while dependencies wait.
 * @param args Additional arguments forwarded to `CopyOp::recv`.
 */
template <typename Transport, typename CopyOp, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus progress_recv_once(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    std::size_t nbytes,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
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
  auto& progressSlot = progress_recv_slot(transport, group);
  IbChannelProgress state = progressSlot;
  if (state.activeStage == detail::IbSendRecvProgressStage::Done) {
    return IbgdaSendRecvProgressStatus::Done;
  }
  const ProgressGeometry progress_params = make_progress_geometry(
      channelLayout, group, nbytes, max_signal_bytes, "progress_recv_once");
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

  const ProgressChunk chunk = next_chunk(channelLayout, state, progress_params);
  const bool isFinalChunk =
      chunk.dataOff + chunk.bytes >= progress_params.protocolBytes;
  const std::size_t protocolBytesThis =
      chunk.bytes + (isFinalChunk ? state.activeTailPadding : 0);
  IbLocalChannel& localChannel =
      transport.local_channel(static_cast<uint32_t>(progress_params.groupId));
  const IbgdaLocalBuffer localDataReady = localChannel.dataReady;
  const IbRemoteChannel remoteChannel =
      makeIbRemoteChannel(channelLayout, progress_params.groupId);
  uint32_t ready = 1;
  if (group.is_leader()) {
    // Poll the specific round-robin lane that carried this chunk and commit
    // recvDataReadyLaneCursor/recvLaneExpected only on a ready result.
    unsigned long long current = 0;
    unsigned long long expected = 0;
    ready = poll_recv_data_ready(
                transport,
                localChannel,
                localDataReady,
                protocolBytesThis,
                current,
                expected)
        ? 1U
        : 0U;
    if (!ready) {
      TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
          timeout,
          "progress_recv_once waiting for DATA_READY expected>=%llu, "
          "current=%llu",
          expected,
          current);
    }
  }
  ready = group.broadcast<uint32_t>(ready);
  if (!ready) {
    return IbgdaSendRecvProgressStatus::Waiting;
  }

  const std::size_t validBytes = valid_payload_bytes(
      chunk.dataOff, chunk.bytes, progress_params.payloadBytes);
  if (validBytes > 0) {
    CopyOp::recv(
        static_cast<char*>(dst) + chunk.dataOff,
        channelLayout.recvStagingPtr + chunk.stagingOff,
        validBytes,
        group,
        chunk.dataOff,
        args...);
  }
  group.sync();

  transport.signal(
      group, remoteChannel.slotFree, protocolBytesThis, IbDirection::Recv);

  state.activeNextByte += chunk.bytes;
  if (active_payload_offset(state) >= progress_params.protocolBytes) {
    transition_progress_stage(
        group, state, detail::IbSendRecvProgressStage::Done);
    store_progress_state(group, progressSlot, state);
    return IbgdaSendRecvProgressStatus::Done;
  }

  store_progress_state(group, progressSlot, state);
  return IbgdaSendRecvProgressStatus::Progressed;
#else
  (void)transport;
  (void)group;
  (void)dst;
  (void)nbytes;
  (void)max_signal_bytes;
  (void)timeout;
  return IbgdaSendRecvProgressStatus::Done;
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
  const int maxGroups = channelLayout.maxChannels;
  if (groupId < 0 || groupId >= maxGroups) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: %s group_id=%d >= maxGroups=%d\n",
          opName,
          groupId,
          maxGroups);
    }
    PIPES_DEVICE_TRAP();
  }

  const std::size_t perBlockSlot = pipeline_chunk(channelLayout);
  if (perBlockSlot == 0) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: %s perBlockSlot=0 "
          "(perChannelBufferSize=%llu, pipelineDepth=%d)\n",
          opName,
          (unsigned long long)pipeline_window(channelLayout),
          channelLayout.pipelineDepth);
    }
    PIPES_DEVICE_TRAP();
  }
  const std::size_t perChannelBufferSize = pipeline_window(channelLayout);

  const std::size_t protocolBytes = align_protocol_bytes(nbytes);
  std::size_t chunkSize =
      (max_signal_bytes > 0 && max_signal_bytes < perBlockSlot)
      ? (max_signal_bytes & ~15ULL)
      : perBlockSlot;
  if (chunkSize == 0) {
    chunkSize = perBlockSlot;
  }
  return ProgressGeometry{
      .groupId = groupId,
      .payloadBytes = nbytes,
      .protocolBytes = protocolBytes,
      .perBlockSlot = perBlockSlot,
      .perChannelBufferSize = perChannelBufferSize,
      .chunkSize = chunkSize,
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
        geometry.chunkSize,
        geometry.perBlockSlot,
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
__device__ __forceinline__ ProgressChunk next_chunk(
    const IbChannelLayout& channelLayout,
    const IbChannelProgress& state,
    const ProgressGeometry& geometry) {
  const uint64_t streamStart =
      static_cast<uint64_t>(state.activeBaseStep) + state.activeNextByte;
  (void)channelLayout;
  const std::size_t pipelineBytes = geometry.perChannelBufferSize;
  const std::size_t pipelineOff =
      static_cast<std::size_t>(streamStart % pipelineBytes);
  const int slot = static_cast<int>(pipelineOff / geometry.perBlockSlot);
  const std::size_t slotOff =
      static_cast<std::size_t>(slot) * geometry.perBlockSlot;
  const std::size_t chunkOff =
      pipelineOff - static_cast<std::size_t>(slot) * geometry.perBlockSlot;
  const std::size_t slotRemaining = geometry.perBlockSlot - chunkOff;
  const std::size_t payloadNextByte = active_payload_offset(state);
  const std::size_t dataRemaining = geometry.protocolBytes - payloadNextByte;
  std::size_t bytes =
      geometry.chunkSize < dataRemaining ? geometry.chunkSize : dataRemaining;
  bytes = bytes < slotRemaining ? bytes : slotRemaining;
  return ProgressChunk{
      .stagingOff = static_cast<std::size_t>(geometry.groupId) * pipelineBytes +
          slotOff + chunkOff,
      .dataOff = payloadNextByte,
      .bytes = bytes,
      .streamEnd = streamStart + bytes,
      .slotId = static_cast<uint32_t>(slot),
      .pipelineGeneration = streamStart / pipelineBytes,
  };
}

template <typename Transport>
__device__ __forceinline__ bool try_prepare_send_slot(
    Transport& transport,
    ThreadGroup& group,
    uint32_t slotId,
    uint64_t generation,
    const Timeout& timeout) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  uint32_t ready = 1;
  if (group.is_leader()) {
    auto& slot =
        transport.local_channel(group.group_id).sendCompletionSlots[slotId];
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
        if (transport.is_local_completion_ready(group.group_id, ticket)) {
          pending &= ~laneBit;
        }
      }
      slot.laneMask = pending;
      if (pending == 0) {
        slot.generation = generation;
      } else {
        ready = 0;
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "send slot local completion timed out slot=%u generation=%llu "
            "pending=0x%llx",
            slotId,
            static_cast<unsigned long long>(generation),
            static_cast<unsigned long long>(pending));
      }
    }
  }
  ready = group.broadcast<uint32_t>(ready);
  return ready != 0;
#else
  (void)transport;
  (void)group;
  (void)slotId;
  (void)generation;
  (void)timeout;
  return true;
#endif
}

} // namespace detail
} // namespace comms::prims
