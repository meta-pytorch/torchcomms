// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "comms/prims/core/DeviceMacros.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/trace/PipesTraceTypes.h"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace comms::prims {

struct Memcpy;

class P2pIbgdaTransportDevice;
class P2pIbrcTransportDevice;

enum class P2pIbBackendType : uint8_t {
  IBGDA,
  IBRC,
};

/**
 * Result of one bounded send/recv progress attempt.
 *
 * `progress_send_once()` and `progress_recv_once()` are intended for callers
 * that multiplex several independent transfers from one kernel. A single call
 * may complete immediately, advance one protocol step, or find that the next
 * signal/counter dependency is not ready yet.
 *
 * `Waiting` means no user-visible data movement or protocol signal was
 * issued. The caller may retry the same transport operation later after
 * making progress on another lane. `Progressed` means the call advanced the
 * operation but more calls are required. `Done` means the transfer has
 * completed the byte range reserved by the corresponding init call.
 */
enum class IbgdaSendRecvProgressStatus : uint8_t {
  Waiting,
  Progressed,
  Done,
};

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

template <typename Transport>
__device__ __forceinline__ IbChannelProgress& progress_send_slot(
    Transport& transport,
    ThreadGroup& group);

template <typename Transport>
__device__ __forceinline__ IbChannelProgress& progress_recv_slot(
    Transport& transport,
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

template <typename Transport>
__device__ __forceinline__ void prepare_send_slot(
    Transport& transport,
    ThreadGroup& group,
    uint32_t slotId,
    uint64_t generation,
    const Timeout& timeout = Timeout());

template <typename Transport>
__device__ __forceinline__ void record_send_completion(
    Transport& transport,
    uint32_t channelId,
    uint32_t slotId,
    uint64_t generation,
    const IbLocalCompletionTicket& ticket);

/**
 * Transport-agnostic pipelined RDMA send/recv helpers.
 *
 * These helpers implement the blocking `send`/`recv`/`forward`, the resumable
 * `init_*_progress` / `progress_*_once` pair, and their private helpers. The
 * send/recv algorithm is independent of the underlying transport: it only
 * needs the common explicit-buffer IB device ops `put`, `wait_local`,
 * `signal`, `wait_signal`, `wait_counter`, `read_signal`, and `read_counter`.
 *
 * Every public entry point takes the owning transport and routes every
 * transport op through it. `P2pIbgdaTransportDevice` and
 * `P2pIbrcTransportDevice` own the channel layout/state and use these helpers
 * only for the shared send/recv algorithm.
 *
 * Blocking and resumable send/forward use the same put() completion-ticket
 * stream for local staging reuse.
 */
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
    std::size_t max_signal_bytes = 0) {
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
    std::size_t max_signal_bytes = 0) {
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
template <typename Transport, typename CopyOp = Memcpy, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus progress_send_once(
    Transport& transport,
    ThreadGroup& group,
    const void* __restrict__ src,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout(),
    Args... args) {
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

    __threadfence_system();
    group.sync();
    if (group.is_leader()) {
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
    // Truncate recvDataReadyLaneCursor to 32 bits BEFORE the modulo so the lane
    // matches the sender's uint32 Send cursor once it wraps at 2^32; otherwise
    // a non-power-of-two numLanes would desync the lane after wrap.
    const uint32_t lane =
        static_cast<uint32_t>(localChannel.recvDataReadyLaneCursor) % lanes;
    const uint64_t expected = localChannel.recvLaneExpected[lane] + chunkBytes;
    const IbgdaLocalBuffer laneBuf = localDataReady.subBuffer(
        sendRecvSignalSlotOffset(static_cast<int>(lane)));
    ThreadGroup solo{
        0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
    transport.wait_signal(solo, laneBuf, expected, timeout);
    localChannel.recvLaneExpected[lane] = expected;
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
template <typename Transport, typename CopyOp = Memcpy, typename... Args>
__device__ __forceinline__ IbgdaSendRecvProgressStatus progress_recv_once(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout(),
    Args... args) {
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
template <typename Transport, typename CopyOp = Memcpy, typename... Args>
__device__ __forceinline__ void send(
    Transport& transport,
    ThreadGroup& group,
    const void* __restrict__ src,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout(),
    Args... args) {
#if !PIPES_IS_DEVICE_COMPILE
  (void)transport;
  (void)group;
  (void)src;
  (void)nbytes;
  (void)max_signal_bytes;
  (void)timeout;
#else
  if (nbytes == 0) {
    return;
  }
  auto& channelLayout = transport.channel_layout();
  const std::size_t payloadProtocolBytes = align_protocol_bytes(nbytes);

  const int groupId = group.group_id;
  const int maxGroups = channelLayout.maxChannels;
  if (groupId >= maxGroups) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: send group_id=%u >= maxGroups=%d\n",
          groupId,
          maxGroups);
    }
    PIPES_DEVICE_TRAP();
  }

  const std::size_t perBlockSlot = pipeline_chunk(channelLayout);
  if (perBlockSlot == 0) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: send perBlockSlot=0 "
          "(perChannelBufferSize=%llu, pipelineDepth=%d)\n",
          (unsigned long long)pipeline_window(channelLayout),
          channelLayout.pipelineDepth);
    }
    PIPES_DEVICE_TRAP();
  }

  std::size_t chunkSize =
      (max_signal_bytes > 0 && max_signal_bytes < perBlockSlot)
      ? (max_signal_bytes & ~15ULL)
      : perBlockSlot;
  if (chunkSize == 0) {
    chunkSize = perBlockSlot;
  }

  auto& state = progress_send_slot(transport, group);
  IbLocalChannel& localChannel =
      transport.local_channel(static_cast<uint32_t>(groupId));
  const IbgdaLocalBuffer localSlotFree = localChannel.slotFree;
  const IbRemoteChannel remoteChannel =
      makeIbRemoteChannel(channelLayout, groupId);
  assert_progress_slot_idle(group, state, "send");
  const uint64_t baseByte = static_cast<uint64_t>(state.nextStep);
  const std::size_t protocolTailPadding = tail_padding_for_signal_granularity(
      baseByte, max_signal_bytes, perBlockSlot, nbytes);
  const uint64_t payloadBaseByte = baseByte;
  const std::size_t protocolBytes = payloadProtocolBytes + protocolTailPadding;
  const std::size_t pipelineBytes = pipeline_window(channelLayout);
  if (group.is_leader()) {
    state.activeStage = detail::IbSendRecvProgressStage::Busy;
    state.activeBaseStep = static_cast<int64_t>(baseByte);
    state.activeNextByte = 0;
    state.activeTailPadding = protocolTailPadding;
  }

  for (std::size_t dataOff = 0; dataOff < payloadProtocolBytes;) {
    const uint64_t streamStart = payloadBaseByte + dataOff;
    const std::size_t pipelineOff =
        static_cast<std::size_t>(streamStart % pipelineBytes);
    const int slot = static_cast<int>(pipelineOff / perBlockSlot);
    const std::size_t slotOff = static_cast<std::size_t>(slot) * perBlockSlot;
    const std::size_t chunkOff = pipelineOff - slot * perBlockSlot;
    const std::size_t slotRemaining = perBlockSlot - chunkOff;
    const std::size_t dataRemaining = payloadProtocolBytes - dataOff;
    std::size_t bytesThis =
        chunkSize < dataRemaining ? chunkSize : dataRemaining;
    bytesThis = bytesThis < slotRemaining ? bytesThis : slotRemaining;
    const bool isFinalChunk = dataOff + bytesThis >= payloadProtocolBytes;
    const std::size_t protocolBytesThis =
        bytesThis + (isFinalChunk ? protocolTailPadding : 0);
    const std::size_t stagingOff =
        static_cast<std::size_t>(groupId) * pipelineBytes + slotOff + chunkOff;
    const uint64_t protocolStreamEnd = streamStart + protocolBytesThis;
    const uint64_t pipelineCycle = streamStart / pipelineBytes;

    // (1) Wait for NIC to finish with this slot's local sendStaging.
    prepare_send_slot(transport, group, slot, pipelineCycle, timeout);

    // (2) Cooperative copy: src -> local sendStaging via CopyOp.
    const std::size_t validBytes =
        valid_payload_bytes(dataOff, bytesThis, nbytes);
    if (validBytes > 0) {
      CopyOp::send(
          channelLayout.sendStagingPtr + stagingOff,
          static_cast<const char*>(src) + dataOff,
          validBytes,
          group,
          dataOff,
          args...);
    }
    group.sync();

    // (3) Backpressure: wait for receiver to free this byte range's
    //     recvStaging offset. Symmetric with DATA_READY.
    if (protocolStreamEnd > pipelineBytes) {
      transport.wait_signal(
          group, localSlotFree, protocolStreamEnd - pipelineBytes, timeout);
    }

    // (4) threadfence_system + leader-only RDMA put with fused signal.
    //     stores are visible to the NIC before the leader posts the WQE.
    __threadfence_system();
    group.sync();
    if (group.is_leader()) {
      ThreadGroup solo{
          0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
      const auto completion = transport.put(
          solo,
          channelLayout.sendStagingBuf.subBuffer(stagingOff),
          remoteChannel.recvStaging.subBuffer(stagingOff),
          bytesThis,
          remoteChannel.dataReady,
          protocolBytesThis,
          /*counterBuf=*/{},
          /*counterVal=*/0,
          /*signalPerLane=*/true);
      record_send_completion(
          transport,
          static_cast<uint32_t>(groupId),
          slot,
          pipelineCycle,
          completion);
    }
    group.sync();
    dataOff += bytesThis;
  }

  if (group.is_leader()) {
    state.nextStep = static_cast<int64_t>(baseByte + protocolBytes);
    state.activeStage = detail::IbSendRecvProgressStage::Done;
    state.activeBaseStep = 0;
    state.activeNextByte = 0;
    state.activeTailPadding = 0;
  }
  group.sync();
#endif
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
template <typename Transport, typename CopyOp = Memcpy, typename... Args>
__device__ __forceinline__ void recv(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout(),
    Args... args) {
#if !PIPES_IS_DEVICE_COMPILE
  (void)transport;
  (void)group;
  (void)dst;
  (void)nbytes;
  (void)max_signal_bytes;
  (void)timeout;
#else
  if (nbytes == 0) {
    return;
  }
  auto& channelLayout = transport.channel_layout();
  const std::size_t payloadProtocolBytes = align_protocol_bytes(nbytes);

  const int groupId = group.group_id;
  const int maxGroups = channelLayout.maxChannels;
  if (groupId >= maxGroups) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: recv group_id=%u >= maxGroups=%d\n",
          groupId,
          maxGroups);
    }
    PIPES_DEVICE_TRAP();
  }

  const std::size_t perBlockSlot = pipeline_chunk(channelLayout);
  if (perBlockSlot == 0) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: recv perBlockSlot=0 "
          "(perChannelBufferSize=%llu, pipelineDepth=%d)\n",
          (unsigned long long)pipeline_window(channelLayout),
          channelLayout.pipelineDepth);
    }
    PIPES_DEVICE_TRAP();
  }

  std::size_t chunkSize =
      (max_signal_bytes > 0 && max_signal_bytes < perBlockSlot)
      ? (max_signal_bytes & ~15ULL)
      : perBlockSlot;
  if (chunkSize == 0) {
    chunkSize = perBlockSlot;
  }

  auto& state = progress_recv_slot(transport, group);
  IbLocalChannel& localChannel =
      transport.local_channel(static_cast<uint32_t>(groupId));
  const IbgdaLocalBuffer localDataReady = localChannel.dataReady;
  const IbRemoteChannel remoteChannel =
      makeIbRemoteChannel(channelLayout, groupId);
  assert_progress_slot_idle(group, state, "recv");
  const uint64_t baseByte = static_cast<uint64_t>(state.nextStep);
  const std::size_t protocolTailPadding = tail_padding_for_signal_granularity(
      baseByte, max_signal_bytes, perBlockSlot, nbytes);
  const uint64_t payloadBaseByte = baseByte;
  const std::size_t protocolBytes = payloadProtocolBytes + protocolTailPadding;
  const std::size_t pipelineBytes = pipeline_window(channelLayout);
  if (group.is_leader()) {
    state.activeStage = detail::IbSendRecvProgressStage::Busy;
    state.activeBaseStep = static_cast<int64_t>(baseByte);
    state.activeNextByte = 0;
    state.activeTailPadding = protocolTailPadding;
  }

  for (std::size_t dataOff = 0; dataOff < payloadProtocolBytes;) {
    const uint64_t streamStart = payloadBaseByte + dataOff;
    const std::size_t pipelineOff =
        static_cast<std::size_t>(streamStart % pipelineBytes);
    const int slot = static_cast<int>(pipelineOff / perBlockSlot);
    const std::size_t slotOff = static_cast<std::size_t>(slot) * perBlockSlot;
    const std::size_t chunkOff = pipelineOff - slot * perBlockSlot;
    const std::size_t slotRemaining = perBlockSlot - chunkOff;
    const std::size_t dataRemaining = payloadProtocolBytes - dataOff;
    std::size_t bytesThis =
        chunkSize < dataRemaining ? chunkSize : dataRemaining;
    bytesThis = bytesThis < slotRemaining ? bytesThis : slotRemaining;
    const bool isFinalChunk = dataOff + bytesThis >= payloadProtocolBytes;
    const std::size_t protocolBytesThis =
        bytesThis + (isFinalChunk ? protocolTailPadding : 0);
    const std::size_t stagingOff =
        static_cast<std::size_t>(groupId) * pipelineBytes + slotOff + chunkOff;

    // (1) Wait for sender's DATA_READY on the specific round-robin lane that
    //     carried this chunk (mirrors the sender's per-channel Send cursor).
    wait_recv_data_ready(
        transport,
        group,
        localChannel,
        localDataReady,
        protocolBytesThis,
        timeout);

    // (2) Cooperative copy: local recvStaging -> dst via CopyOp.
    const std::size_t validBytes =
        valid_payload_bytes(dataOff, bytesThis, nbytes);
    if (validBytes > 0) {
      CopyOp::recv(
          static_cast<char*>(dst) + dataOff,
          channelLayout.recvStagingPtr + stagingOff,
          validBytes,
          group,
          dataOff,
          args...);
    }
    group.sync();

    transport.signal(
        group, remoteChannel.slotFree, protocolBytesThis, IbDirection::Recv);
    dataOff += bytesThis;
  }

  if (group.is_leader()) {
    state.nextStep = static_cast<int64_t>(baseByte + protocolBytes);
    state.activeStage = detail::IbSendRecvProgressStage::Done;
    state.activeBaseStep = 0;
    state.activeNextByte = 0;
    state.activeTailPadding = 0;
  }
  group.sync();
#endif
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
template <typename CopyOp = Memcpy, typename Transport, typename... Args>
__device__ __forceinline__ void forward(
    Transport& transport,
    ThreadGroup& group,
    void* __restrict__ dst,
    Transport& fwdTransport,
    std::size_t nbytes,
    std::size_t max_signal_bytes = 0,
    const Timeout& timeout = Timeout(),
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
  const std::size_t payloadProtocolBytes = align_protocol_bytes(nbytes);

  const int groupId = group.group_id;

  // --- recv side (this transport) ---
  const int recvMaxGroups = channelLayout.maxChannels;
  if (groupId >= recvMaxGroups) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: forward recv maxGroups=%d groupId=%u\n",
          recvMaxGroups,
          groupId);
    }
    PIPES_DEVICE_TRAP();
  }

  const std::size_t recvPerBlockSlot = pipeline_chunk(channelLayout);
  if (recvPerBlockSlot == 0) {
    if (group.is_leader()) {
      printf("[PIPES] FATAL: forward recvPerBlockSlot=0\n");
    }
    PIPES_DEVICE_TRAP();
  }

  // --- fwd side (fwd transport) ---
  const int fwdMaxGroups = fwdChannelLayout.maxChannels;
  if (groupId >= fwdMaxGroups) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: forward fwd maxGroups=%d groupId=%u\n",
          fwdMaxGroups,
          groupId);
    }
    PIPES_DEVICE_TRAP();
  }

  const std::size_t fwdPerBlockSlot = pipeline_chunk(fwdChannelLayout);
  if (fwdPerBlockSlot == 0) {
    if (group.is_leader()) {
      printf("[PIPES] FATAL: forward fwdPerBlockSlot=0\n");
    }
    PIPES_DEVICE_TRAP();
  }

  // Chunk sizes for recv and fwd sides
  std::size_t recvChunkSize =
      (max_signal_bytes > 0 && max_signal_bytes < recvPerBlockSlot)
      ? (max_signal_bytes & ~15ULL)
      : recvPerBlockSlot;
  if (recvChunkSize == 0) {
    recvChunkSize = recvPerBlockSlot;
  }
  std::size_t fwdChunkSize =
      (max_signal_bytes > 0 && max_signal_bytes < fwdPerBlockSlot)
      ? (max_signal_bytes & ~15ULL)
      : fwdPerBlockSlot;
  if (fwdChunkSize == 0) {
    fwdChunkSize = fwdPerBlockSlot;
  }

  auto& recvSlotState = progress_recv_slot(transport, group);
  IbLocalChannel& recvLocalChannel =
      transport.local_channel(static_cast<uint32_t>(groupId));
  const IbgdaLocalBuffer recvDataReady = recvLocalChannel.dataReady;
  const IbRemoteChannel recvRemoteChannel =
      makeIbRemoteChannel(channelLayout, groupId);
  assert_progress_slot_idle(group, recvSlotState, "forward recv");
  const uint64_t recvBaseByte = static_cast<uint64_t>(recvSlotState.nextStep);
  const std::size_t recvProtocolTailPadding =
      tail_padding_for_signal_granularity(
          recvBaseByte, max_signal_bytes, recvPerBlockSlot, nbytes);
  const uint64_t recvPayloadBaseByte = recvBaseByte;
  const std::size_t recvProtocolBytes =
      payloadProtocolBytes + recvProtocolTailPadding;
  const std::size_t recvPipelineBytes = pipeline_window(channelLayout);

  auto& fwdSlotState = progress_send_slot(fwdTransport, group);
  IbLocalChannel& fwdLocalChannel =
      fwdTransport.local_channel(static_cast<uint32_t>(groupId));
  const IbgdaLocalBuffer fwdSlotFree = fwdLocalChannel.slotFree;
  const IbRemoteChannel fwdRemoteChannel =
      makeIbRemoteChannel(fwdChannelLayout, groupId);
  assert_progress_slot_idle(group, fwdSlotState, "forward send");
  const uint64_t fwdBaseByte = static_cast<uint64_t>(fwdSlotState.nextStep);
  const std::size_t fwdProtocolTailPadding =
      tail_padding_for_signal_granularity(
          fwdBaseByte, max_signal_bytes, fwdPerBlockSlot, nbytes);
  const uint64_t fwdPayloadBaseByte = fwdBaseByte;
  const std::size_t fwdProtocolBytes =
      payloadProtocolBytes + fwdProtocolTailPadding;
  const std::size_t fwdPipelineBytes = pipeline_window(fwdChannelLayout);
  if (group.is_leader()) {
    recvSlotState.activeStage = detail::IbSendRecvProgressStage::Busy;
    recvSlotState.activeBaseStep = static_cast<int64_t>(recvBaseByte);
    recvSlotState.activeNextByte = 0;
    recvSlotState.activeTailPadding = recvProtocolTailPadding;
    fwdSlotState.activeStage = detail::IbSendRecvProgressStage::Busy;
    fwdSlotState.activeBaseStep = static_cast<int64_t>(fwdBaseByte);
    fwdSlotState.activeNextByte = 0;
    fwdSlotState.activeTailPadding = fwdProtocolTailPadding;
  }

  for (std::size_t dataOff = 0; dataOff < payloadProtocolBytes;) {
    // --- Recv side offsets ---
    const uint64_t recvStreamStart = recvPayloadBaseByte + dataOff;
    const std::size_t recvPipelineOff =
        static_cast<std::size_t>(recvStreamStart % recvPipelineBytes);
    const int recvSlot = static_cast<int>(recvPipelineOff / recvPerBlockSlot);
    const std::size_t recvSlotOff =
        static_cast<std::size_t>(recvSlot) * recvPerBlockSlot;
    const std::size_t recvChunkOff =
        recvPipelineOff - recvSlot * recvPerBlockSlot;
    const std::size_t recvStagingOff =
        static_cast<std::size_t>(groupId) * recvPipelineBytes + recvSlotOff +
        recvChunkOff;

    // --- Fwd side offsets ---
    const uint64_t fwdStreamStart = fwdPayloadBaseByte + dataOff;
    const std::size_t fwdPipelineOff =
        static_cast<std::size_t>(fwdStreamStart % fwdPipelineBytes);
    const int fwdSlot = static_cast<int>(fwdPipelineOff / fwdPerBlockSlot);
    const std::size_t fwdSlotOff =
        static_cast<std::size_t>(fwdSlot) * fwdPerBlockSlot;
    const std::size_t fwdChunkOff = fwdPipelineOff - fwdSlot * fwdPerBlockSlot;
    const std::size_t fwdStagingOff =
        static_cast<std::size_t>(groupId) * fwdPipelineBytes + fwdSlotOff +
        fwdChunkOff;
    const std::size_t recvSlotRemaining = recvPerBlockSlot - recvChunkOff;
    const std::size_t fwdSlotRemaining = fwdPerBlockSlot - fwdChunkOff;
    const std::size_t dataRemaining = payloadProtocolBytes - dataOff;
    std::size_t bytesThis =
        recvChunkSize < fwdChunkSize ? recvChunkSize : fwdChunkSize;
    bytesThis = bytesThis < dataRemaining ? bytesThis : dataRemaining;
    bytesThis = bytesThis < recvSlotRemaining ? bytesThis : recvSlotRemaining;
    bytesThis = bytesThis < fwdSlotRemaining ? bytesThis : fwdSlotRemaining;
    const bool isFinalChunk = dataOff + bytesThis >= payloadProtocolBytes;
    const std::size_t recvProtocolBytesThis =
        bytesThis + (isFinalChunk ? recvProtocolTailPadding : 0);
    const std::size_t fwdProtocolBytesThis =
        bytesThis + (isFinalChunk ? fwdProtocolTailPadding : 0);
    const uint64_t fwdProtocolStreamEnd = fwdStreamStart + fwdProtocolBytesThis;
    const uint64_t fwdPipelineCycle = fwdStreamStart / fwdPipelineBytes;

    // (1) Wait for the upstream sender's DATA_READY on the specific round-robin
    //     lane that carried this chunk (mirrors the upstream sender's Send
    //     cursor via recvLocalChannel.recvDataReadyLaneCursor).
    wait_recv_data_ready(
        transport,
        group,
        recvLocalChannel,
        recvDataReady,
        recvProtocolBytesThis,
        timeout);

    // (2) Wait for local completion on fwd's sendStaging.
    prepare_send_slot(fwdTransport, group, fwdSlot, fwdPipelineCycle, timeout);

    // (3) CopyOp::forward — transform recv staging -> dst + fwd staging.
    const std::size_t validBytes =
        valid_payload_bytes(dataOff, bytesThis, nbytes);
    if (validBytes > 0) {
      CopyOp::forward(
          dst ? static_cast<char*>(dst) + dataOff : nullptr,
          fwdChannelLayout.sendStagingPtr + fwdStagingOff,
          channelLayout.recvStagingPtr + recvStagingOff,
          validBytes,
          group,
          dataOff,
          args...);
    }
    group.sync();

    transport.signal(
        group,
        recvRemoteChannel.slotFree,
        recvProtocolBytesThis,
        IbDirection::Recv);

    // (5) Wait for fwd receiver's SLOT_FREE (backpressure on fwd's
    //     recvStaging).
    if (fwdProtocolStreamEnd > fwdPipelineBytes) {
      fwdTransport.wait_signal(
          group, fwdSlotFree, fwdProtocolStreamEnd - fwdPipelineBytes, timeout);
    }

    // (6) threadfence_system + RDMA put via fwd transport.
    __threadfence_system();
    group.sync();
    if (group.is_leader()) {
      ThreadGroup solo{
          0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
      const auto completion = fwdTransport.put(
          solo,
          fwdChannelLayout.sendStagingBuf.subBuffer(fwdStagingOff),
          fwdRemoteChannel.recvStaging.subBuffer(fwdStagingOff),
          bytesThis,
          fwdRemoteChannel.dataReady,
          fwdProtocolBytesThis,
          /*counterBuf=*/{},
          /*counterVal=*/0,
          /*signalPerLane=*/true);
      record_send_completion(
          fwdTransport,
          static_cast<uint32_t>(groupId),
          fwdSlot,
          fwdPipelineCycle,
          completion);
    }
    group.sync();
    dataOff += bytesThis;
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
#endif
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

__device__ __forceinline__ static std::size_t signal_alignment(
    std::size_t maxSignalBytes,
    std::size_t perBlockSlot) {
  const bool usesPartialSlot =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlot;
  std::size_t alignment =
      usesPartialSlot ? (maxSignalBytes & ~15ULL) : perBlockSlot;
  return alignment == 0 ? perBlockSlot : alignment;
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
  if (channelLayout.maxChannels <= 0) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: send/recv maxChannels must be > 0, got %d\n",
          channelLayout.maxChannels);
    }
    PIPES_DEVICE_TRAP();
  }
  if (group.group_id >= static_cast<uint32_t>(channelLayout.maxChannels)) {
    if (group.is_leader()) {
      printf(
          "[PIPES] FATAL: progress group_id=%u out of range [0, %d)\n",
          group.group_id,
          channelLayout.maxChannels);
    }
    PIPES_DEVICE_TRAP();
  }
#else
  (void)channelLayout;
  (void)group;
#endif
}

template <typename Transport>
__device__ __forceinline__ IbChannelProgress& progress_send_slot(
    Transport& transport,
    ThreadGroup& group) {
  validate_progress_group(transport.channel_layout(), group);
  return transport.local_channel(group).sendProgress;
}

template <typename Transport>
__device__ __forceinline__ IbChannelProgress& progress_recv_slot(
    Transport& transport,
    ThreadGroup& group) {
  validate_progress_group(transport.channel_layout(), group);
  return transport.local_channel(group).recvProgress;
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

template <typename Transport>
__device__ __forceinline__ void prepare_send_slot(
    Transport& transport,
    ThreadGroup& group,
    uint32_t slotId,
    uint64_t generation,
    const Timeout& timeout) {
  if (group.is_leader()) {
    auto& slot =
        transport.local_channel(group.group_id).sendCompletionSlots[slotId];
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

template <typename Transport>
__device__ __forceinline__ void record_send_completion(
    Transport& transport,
    uint32_t channelId,
    uint32_t slotId,
    uint64_t generation,
    const IbLocalCompletionTicket& ticket) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  auto& slot = transport.local_channel(channelId).sendCompletionSlots[slotId];
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

  // Common slot-index IB device API.
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

  // Common explicit-buffer IB device API.
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

  // Pipelined send/recv — forwarded to the active backend's shared helpers.
  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void send(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args);

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

  // Total staging bytes for one channel, forwarded to the active backend.
  __device__ __forceinline__ std::size_t pipeline_window() const;

  // Slots per channel, forwarded to the active backend.
  __device__ __forceinline__ int pipeline_depth() const;

  // Slot/chunk bytes, forwarded to the active backend.
  __device__ __forceinline__ std::size_t pipeline_chunk() const;

  __device__ __forceinline__ void init_send_progress(
      ThreadGroup& group,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0);

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

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus progress_recv_once(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args);
};

static_assert(std::is_standard_layout_v<P2pIbTransportDevice>);
static_assert(std::is_trivially_copyable_v<P2pIbTransportDevice>);

} // namespace comms::prims
