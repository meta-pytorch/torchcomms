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

namespace detail {
__host__ __device__ constexpr std::size_t ib_send_recv_credit_quantum(
    std::size_t perBlockSlot,
    int pipelineDepth) {
  const std::size_t pipelineBytes =
      perBlockSlot * static_cast<std::size_t>(pipelineDepth);
  std::size_t quantum = pipelineBytes / 4;
  if (quantum < 16ULL) {
    quantum = 16ULL;
  }
  if (quantum > perBlockSlot) {
    quantum = perBlockSlot;
  }
  quantum &= ~15ULL;
  return quantum == 0 ? 16ULL : quantum;
}

__host__ __device__ constexpr std::size_t ib_send_recv_nic_done_credit_quantum(
    std::size_t perBlockSlot,
    int pipelineDepth) {
  return ib_send_recv_credit_quantum(perBlockSlot, pipelineDepth);
}

__host__ __device__ constexpr std::size_t ib_send_recv_slot_free_credit_quantum(
    std::size_t perBlockSlot,
    int pipelineDepth) {
  return ib_send_recv_credit_quantum(perBlockSlot, pipelineDepth);
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
 * IbSendRecvDevice — transport-agnostic pipelined RDMA send/recv.
 *
 * This class owns master's full async send/recv algorithm: the blocking
 * `send`/`recv`/`forward`, the resumable `init_*_progress` /
 * `progress_*_once` pair, and all of their private helpers. The send/recv
 * algorithm is independent of the underlying transport: it only needs the
 * common explicit-buffer IB device ops `put` (fused signal+counter),
 * `signal`, `wait_signal`, `wait_counter`, `read_signal`, and `read_counter`.
 *
 * Every public method takes the owning transport as a `Transport& transport`
 * parameter and routes every transport op through it. `P2pIbgdaTransportDevice`
 * and `P2pIbrcTransportDevice` hold one `IbSendRecvDevice sendRecv_` member and
 * delegate their `send`/`recv`/`forward`/progress methods to it, passing
 * `*this` as the transport. The send/recv protocol state lives in
 * `sendRecvState_`, populated by the host transport builder.
 */
class IbSendRecvDevice {
 public:
  IbSendRecvState sendRecvState_{};

  __host__ __device__ IbSendRecvDevice() = default;

  __host__ __device__ explicit IbSendRecvDevice(IbSendRecvState initialState)
      : sendRecvState_(initialState) {}

  __host__ __device__ const IbSendRecvState& send_recv_state() const {
    return sendRecvState_;
  }

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
  __device__ __forceinline__ void init_send_progress(
      ThreadGroup& group,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const int progressIndex = progress_send_index(group);
    auto& slot = progress_state_slot(group, progressIndex);
    assert_progress_slot_idle(group, slot, "send");
    IbSendRecvState::ProgressSlot state{};
    state.reuseCreditStep = slot.reuseCreditStep;
    state.activeStage = nbytes == 0
        ? detail::IbSendRecvProgressStage::Done
        : detail::IbSendRecvProgressStage::WaitNicDone;
    if (nbytes == 0) {
      store_progress_state(group, progressIndex, state);
      return;
    }
    // Validate the transfer before reserving the transport byte cursor.
    const ProgressGeometry geometry = make_progress_geometry(
        group, nbytes, max_signal_bytes, "init_send_progress");
    state.activeBaseStep =
        reserve_progress_step(group, progressIndex, geometry.protocolBytes);
    store_progress_state(group, progressIndex, state);
#else
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
  __device__ __forceinline__ void init_recv_progress(
      ThreadGroup& group,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const int progressIndex = progress_recv_index(group);
    auto& slot = progress_state_slot(group, progressIndex);
    assert_progress_slot_idle(group, slot, "recv");
    IbSendRecvState::ProgressSlot state{};
    state.reuseCreditStep = slot.reuseCreditStep;
    state.activeStage = nbytes == 0
        ? detail::IbSendRecvProgressStage::Done
        : detail::IbSendRecvProgressStage::WaitDataReady;
    if (nbytes == 0) {
      store_progress_state(group, progressIndex, state);
      return;
    }
    // Validate the transfer before reserving the transport byte cursor.
    const ProgressGeometry geometry = make_progress_geometry(
        group, nbytes, max_signal_bytes, "init_recv_progress");
    state.activeBaseStep =
        reserve_progress_step(group, progressIndex, geometry.protocolBytes);
    store_progress_state(group, progressIndex, state);
#else
    (void)group;
    (void)nbytes;
    (void)max_signal_bytes;
#endif
  }

  /**
   * Attempt bounded progress on one initialized send.
   *
   * This method advances at most one staged copy plus one RDMA put for the
   * current chunk. It never spins on NIC_DONE or SLOT_FREE: if either
   * dependency is not ready, it returns immediately so a higher-level scheduler
   * can try another independent lane. If a `Timeout` is enabled, it is checked
   * only at those readiness points and should already have been started by the
   * caller.
   *
   * The send path first waits for NIC_DONE before reusing the local
   * send-staging range, then copies user data into send-staging through
   * `CopyOp::send`, waits for SLOT_FREE before reusing the peer's recv-staging
   * range, and finally issues an RDMA put that piggybacks DATA_READY and may
   * batch NIC_DONE into the local counter. Returning `Done` means the reserved
   * byte range has completed.
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
        sizeof(CopyOp) == 0,
        "IbSendRecvDevice::progress_send_once() requires NVIDIA GPU");
#endif
    const int progressIndex = progress_send_index(group);
    IbSendRecvState::ProgressSlot state =
        progress_state_slot(group, progressIndex);
    if (state.activeStage == detail::IbSendRecvProgressStage::Done) {
      return IbgdaSendRecvProgressStatus::Done;
    }
    const ProgressGeometry progress_params = make_progress_geometry(
        group, nbytes, max_signal_bytes, "progress_send_once");
    if (state.activeNextByte >= progress_params.protocolBytes) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: progress_send_once nextByte=%llu >= "
            "protocolBytes=%llu without Done stage\n",
            static_cast<unsigned long long>(state.activeNextByte),
            static_cast<unsigned long long>(progress_params.protocolBytes));
      }
      PIPES_DEVICE_TRAP();
    }
    validate_send_progress_stage(group, state);

    const detail::IbSendRecvProgressStage initialStage = state.activeStage;
    const std::size_t initialNextByte = state.activeNextByte;
    const std::size_t pipelineBytes = progress_params.perBlockSlot *
        static_cast<std::size_t>(sendRecvState_.pipelineDepth);

    if (state.activeStage == detail::IbSendRecvProgressStage::WaitNicDone) {
      const ProgressChunk chunk = next_chunk(state, progress_params);
      if (chunk.streamEnd > pipelineBytes) {
        const uint64_t expected = chunk.streamEnd - pipelineBytes;
        uint32_t ready = 1;
        unsigned long long current = 0;
        if (group.is_leader()) {
          current = static_cast<unsigned long long>(
              transport.read_counter(sendRecvState_.localCounterBuf.subBuffer(
                  progress_params.groupId * sizeof(uint64_t))));
          ready = current >= expected ? 1U : 0U;
          if (!ready) {
            TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
                timeout,
                "progress_send_once waiting for NIC_DONE expected>=%llu, "
                "current=%llu",
                static_cast<unsigned long long>(expected),
                current);
          }
        }
        ready = group.broadcast<uint32_t>(ready);
        if (!ready) {
          return IbgdaSendRecvProgressStatus::Waiting;
        }
      }

      const std::size_t validBytes = valid_payload_bytes(
          chunk.dataOff, chunk.bytes, progress_params.payloadBytes);
      if (validBytes > 0) {
        CopyOp::send(
            sendRecvState_.sendStagingPtr + chunk.stagingOff,
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
      const ProgressChunk chunk = next_chunk(state, progress_params);
      if (chunk.streamEnd > pipelineBytes) {
        const uint64_t expected = chunk.streamEnd - pipelineBytes;
        uint32_t ready = 1;
        unsigned long long current = 0;
        if (group.is_leader()) {
          current = static_cast<unsigned long long>(
              transport.read_signal(sendRecvState_.localSignalBuf.subBuffer(
                  (sendRecvState_.maxGroups + progress_params.groupId) *
                  sizeof(uint64_t))));
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
            store_progress_state(group, progressIndex, state);
            return IbgdaSendRecvProgressStatus::Progressed;
          }
          return IbgdaSendRecvProgressStatus::Waiting;
        }
      }

      __threadfence_system();
      group.sync();
      IbgdaLocalBuffer counterBuf{};
      uint64_t counterVal = 0;
      if (group.is_leader()) {
        if (should_post_nic_done_credit(
                chunk.streamEnd, state.reuseCreditStep, progress_params)) {
          counterVal =
              reuse_credit_value(chunk.streamEnd, state.reuseCreditStep);
          counterBuf = sendRecvState_.localCounterCompletionBuf.subBuffer(
              progress_params.groupId * sizeof(uint64_t));
          state.reuseCreditStep = static_cast<int64_t>(chunk.streamEnd);
        }
        ThreadGroup solo{
            0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
        transport.put(
            solo,
            sendRecvState_.sendStagingBuf.subBuffer(chunk.stagingOff),
            sendRecvState_.recvStagingBuf.subBuffer(chunk.stagingOff),
            chunk.bytes,
            sendRecvState_.remoteSignalBuf.subBuffer(
                progress_params.groupId * sizeof(uint64_t)),
            chunk.bytes,
            counterBuf,
            counterVal);
      }
      group.sync();

      state.activeNextByte += chunk.bytes;
      if (state.activeNextByte >= progress_params.protocolBytes) {
        transition_progress_stage(
            group, state, detail::IbSendRecvProgressStage::Done);
        store_progress_state(group, progressIndex, state);
        return IbgdaSendRecvProgressStatus::Done;
      }
      transition_progress_stage(
          group, state, detail::IbSendRecvProgressStage::WaitNicDone);
    }

    // A full non-final chunk can cycle WaitNicDone -> WaitSlotFree ->
    // WaitNicDone in one call, leaving the stage unchanged while nextByte
    // advances. Check both fields so that case reports Progressed.
    if (state.activeStage != initialStage ||
        state.activeNextByte != initialNextByte) {
      store_progress_state(group, progressIndex, state);
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
   * `CopyOp::recv`, then may signal batched SLOT_FREE credit back to the
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
        sizeof(CopyOp) == 0,
        "IbSendRecvDevice::progress_recv_once() requires NVIDIA GPU");
#endif
    const int progressIndex = progress_recv_index(group);
    IbSendRecvState::ProgressSlot state =
        progress_state_slot(group, progressIndex);
    if (state.activeStage == detail::IbSendRecvProgressStage::Done) {
      return IbgdaSendRecvProgressStatus::Done;
    }
    const ProgressGeometry progress_params = make_progress_geometry(
        group, nbytes, max_signal_bytes, "progress_recv_once");
    if (state.activeNextByte >= progress_params.protocolBytes) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: progress_recv_once nextByte=%llu >= "
            "protocolBytes=%llu without Done stage\n",
            static_cast<unsigned long long>(state.activeNextByte),
            static_cast<unsigned long long>(progress_params.protocolBytes));
      }
      PIPES_DEVICE_TRAP();
    }
    validate_recv_progress_stage(group, state);

    const ProgressChunk chunk = next_chunk(state, progress_params);
    const uint64_t expected = chunk.streamEnd;
    uint32_t ready = 1;
    unsigned long long current = 0;
    if (group.is_leader()) {
      current = static_cast<unsigned long long>(
          transport.read_signal(sendRecvState_.localSignalBuf.subBuffer(
              progress_params.groupId * sizeof(uint64_t))));
      ready = current >= expected ? 1U : 0U;
      if (!ready) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "progress_recv_once waiting for DATA_READY expected>=%llu, "
            "current=%llu",
            static_cast<unsigned long long>(expected),
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
          sendRecvState_.recvStagingPtr + chunk.stagingOff,
          validBytes,
          group,
          chunk.dataOff,
          args...);
    }
    group.sync();

    uint64_t slotFreeVal = 0;
    if (group.is_leader()) {
      if (should_post_slot_free_credit(
              chunk.streamEnd, state.reuseCreditStep, progress_params)) {
        slotFreeVal =
            reuse_credit_value(chunk.streamEnd, state.reuseCreditStep);
        state.reuseCreditStep = static_cast<int64_t>(chunk.streamEnd);
      }
    }
    slotFreeVal = group.broadcast<uint64_t>(slotFreeVal);
    if (slotFreeVal != 0) {
      transport.signal(
          group,
          sendRecvState_.remoteSignalBuf.subBuffer(
              (sendRecvState_.maxGroups + progress_params.groupId) *
              sizeof(uint64_t)),
          slotFreeVal,
          IbDirection::Recv);
    }

    state.activeNextByte += chunk.bytes;
    if (state.activeNextByte >= progress_params.protocolBytes) {
      transition_progress_stage(
          group, state, detail::IbSendRecvProgressStage::Done);
      store_progress_state(group, progressIndex, state);
      return IbgdaSendRecvProgressStatus::Done;
    }

    store_progress_state(group, progressIndex, state);
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
   *   NIC_DONE   — loopback counter incremented by NIC after each RDMA put.
   *                send waits on this before overwriting local sendStaging.
   *   SLOT_FREE  — receiver increments by bytesThis for each signaled byte
   *                range. send waits before overwriting recvStaging.
   *   DATA_READY — sender increments by bytesThis, piggybacked on put.
   *                recv waits on this before reading recvStaging.
   *
   * state[].nextStep persists across calls, so send() resumes the staging-ring
   * cursor and protocol sequence numbers on each invocation. This allows
   * callers to pipeline across repeated send() calls without a separate drain.
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
    const std::size_t protocolBytes = align_protocol_bytes(nbytes);

    const int groupId = group.group_id;
    const int maxGroups = sendRecvState_.maxGroups;
    if (groupId >= maxGroups) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: send group_id=%u >= maxGroups=%d\n",
            groupId,
            maxGroups);
      }
      PIPES_DEVICE_TRAP();
    }

    const std::size_t perBlockSlot =
        (sendRecvState_.dataBufferSize / maxGroups) & ~15ULL;
    if (perBlockSlot == 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: send perBlockSlot=0 "
            "(dataBufferSize=%llu, maxGroups=%d)\n",
            (unsigned long long)sendRecvState_.dataBufferSize,
            maxGroups);
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

    const int pipelineDepth = sendRecvState_.pipelineDepth;
    const std::size_t dataBufferSize = sendRecvState_.dataBufferSize;
    const int stateIndex = progress_send_index(group);
    auto& state = progress_state_slot(group, stateIndex);
    assert_progress_slot_idle(group, state, "send");
    const uint64_t baseByte = static_cast<uint64_t>(state.nextStep);
    const std::size_t pipelineBytes = perBlockSlot * pipelineDepth;
    if (group.is_leader()) {
      state.activeStage = detail::IbSendRecvProgressStage::Busy;
      state.activeBaseStep = static_cast<int64_t>(baseByte);
      state.activeNextByte = 0;
    }

    for (std::size_t dataOff = 0; dataOff < protocolBytes;) {
      const uint64_t streamStart = baseByte + dataOff;
      const std::size_t pipelineOff =
          static_cast<std::size_t>(streamStart % pipelineBytes);
      const int slot = static_cast<int>(pipelineOff / perBlockSlot);
      const std::size_t slotOff = slot * dataBufferSize;
      const std::size_t chunkOff = pipelineOff - slot * perBlockSlot;
      const std::size_t slotRemaining = perBlockSlot - chunkOff;
      const std::size_t dataRemaining = protocolBytes - dataOff;
      std::size_t bytesThis =
          chunkSize < dataRemaining ? chunkSize : dataRemaining;
      bytesThis = bytesThis < slotRemaining ? bytesThis : slotRemaining;
      const std::size_t stagingOff =
          slotOff + groupId * perBlockSlot + chunkOff;
      const uint64_t streamEnd = streamStart + bytesThis;

      // (1) Wait for NIC to finish with this slot's local sendStaging.
      if (streamEnd > pipelineBytes) {
        transport.wait_counter(
            group,
            sendRecvState_.localCounterBuf.subBuffer(
                groupId * sizeof(uint64_t)),
            streamEnd - pipelineBytes,
            timeout);
      }

      // (2) Cooperative copy: src -> local sendStaging via CopyOp.
      const std::size_t validBytes =
          valid_payload_bytes(dataOff, bytesThis, nbytes);
      if (validBytes > 0) {
        CopyOp::send(
            sendRecvState_.sendStagingPtr + stagingOff,
            static_cast<const char*>(src) + dataOff,
            validBytes,
            group,
            dataOff,
            args...);
      }
      group.sync();

      // (3) Backpressure: wait for receiver to free this byte range's
      //     recvStaging offset. Symmetric with DATA_READY.
      if (streamEnd > pipelineBytes) {
        transport.wait_signal(
            group,
            sendRecvState_.localSignalBuf.subBuffer(
                (maxGroups + groupId) * sizeof(uint64_t)),
            streamEnd - pipelineBytes,
            timeout);
      }

      // (4) threadfence_system + leader-only single-WQE RDMA put with
      //     fused signal+counter. All threads fence to ensure memcpy
      //     stores are visible to the NIC before the leader posts the WQE.
      __threadfence_system();
      group.sync();
      if (group.is_leader()) {
        IbgdaLocalBuffer counterBuf{};
        uint64_t counterVal = 0;
        if (should_post_nic_done_credit(
                streamEnd, state.reuseCreditStep, perBlockSlot)) {
          counterVal = reuse_credit_value(streamEnd, state.reuseCreditStep);
          counterBuf = sendRecvState_.localCounterCompletionBuf.subBuffer(
              groupId * sizeof(uint64_t));
          state.reuseCreditStep = static_cast<int64_t>(streamEnd);
        }
        ThreadGroup solo{
            0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
        transport.put(
            solo,
            sendRecvState_.sendStagingBuf.subBuffer(stagingOff),
            sendRecvState_.recvStagingBuf.subBuffer(stagingOff),
            bytesThis,
            sendRecvState_.remoteSignalBuf.subBuffer(
                groupId * sizeof(uint64_t)),
            bytesThis,
            counterBuf,
            counterVal);
      }
      group.sync();
      dataOff += bytesThis;
    }

    if (group.is_leader()) {
      state.nextStep = static_cast<int64_t>(baseByte + protocolBytes);
      state.activeStage = detail::IbSendRecvProgressStage::Done;
      state.activeBaseStep = 0;
      state.activeNextByte = 0;
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
    const std::size_t protocolBytes = align_protocol_bytes(nbytes);

    const int groupId = group.group_id;
    const int maxGroups = sendRecvState_.maxGroups;
    if (groupId >= maxGroups) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: recv group_id=%u >= maxGroups=%d\n",
            groupId,
            maxGroups);
      }
      PIPES_DEVICE_TRAP();
    }

    const std::size_t perBlockSlot =
        (sendRecvState_.dataBufferSize / maxGroups) & ~15ULL;
    if (perBlockSlot == 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: recv perBlockSlot=0 "
            "(dataBufferSize=%llu, maxGroups=%d)\n",
            (unsigned long long)sendRecvState_.dataBufferSize,
            maxGroups);
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

    const int pipelineDepth = sendRecvState_.pipelineDepth;
    const std::size_t dataBufferSize = sendRecvState_.dataBufferSize;
    const int stateIndex = progress_recv_index(group);
    auto& state = progress_state_slot(group, stateIndex);
    assert_progress_slot_idle(group, state, "recv");
    const uint64_t baseByte = static_cast<uint64_t>(state.nextStep);
    const std::size_t pipelineBytes = perBlockSlot * pipelineDepth;
    if (group.is_leader()) {
      state.activeStage = detail::IbSendRecvProgressStage::Busy;
      state.activeBaseStep = static_cast<int64_t>(baseByte);
      state.activeNextByte = 0;
    }

    for (std::size_t dataOff = 0; dataOff < protocolBytes;) {
      const uint64_t streamStart = baseByte + dataOff;
      const std::size_t pipelineOff =
          static_cast<std::size_t>(streamStart % pipelineBytes);
      const int slot = static_cast<int>(pipelineOff / perBlockSlot);
      const std::size_t slotOff = slot * dataBufferSize;
      const std::size_t chunkOff = pipelineOff - slot * perBlockSlot;
      const std::size_t slotRemaining = perBlockSlot - chunkOff;
      const std::size_t dataRemaining = protocolBytes - dataOff;
      std::size_t bytesThis =
          chunkSize < dataRemaining ? chunkSize : dataRemaining;
      bytesThis = bytesThis < slotRemaining ? bytesThis : slotRemaining;
      const std::size_t stagingOff =
          slotOff + groupId * perBlockSlot + chunkOff;
      const uint64_t streamEnd = streamStart + bytesThis;

      // (1) Wait for sender's DATA_READY signal.
      transport.wait_signal(
          group,
          sendRecvState_.localSignalBuf.subBuffer(groupId * sizeof(uint64_t)),
          streamEnd,
          timeout);

      // (2) Cooperative copy: local recvStaging -> dst via CopyOp.
      const std::size_t validBytes =
          valid_payload_bytes(dataOff, bytesThis, nbytes);
      if (validBytes > 0) {
        CopyOp::recv(
            static_cast<char*>(dst) + dataOff,
            sendRecvState_.recvStagingPtr + stagingOff,
            validBytes,
            group,
            dataOff,
            args...);
      }
      group.sync();

      // (3) Signal SLOT_FREE to sender. Sender waits on the cumulative byte
      //     threshold before reusing remote recvStaging at the same offset.
      uint64_t slotFreeVal = 0;
      if (group.is_leader()) {
        if (should_post_slot_free_credit(
                streamEnd, state.reuseCreditStep, perBlockSlot)) {
          slotFreeVal = reuse_credit_value(streamEnd, state.reuseCreditStep);
          state.reuseCreditStep = static_cast<int64_t>(streamEnd);
        }
      }
      slotFreeVal = group.broadcast<uint64_t>(slotFreeVal);
      if (slotFreeVal != 0) {
        transport.signal(
            group,
            sendRecvState_.remoteSignalBuf.subBuffer(
                (maxGroups + groupId) * sizeof(uint64_t)),
            slotFreeVal,
            IbDirection::Recv);
      }
      dataOff += bytesThis;
    }

    if (group.is_leader()) {
      state.nextStep = static_cast<int64_t>(baseByte + protocolBytes);
      state.activeStage = detail::IbSendRecvProgressStage::Done;
      state.activeBaseStep = 0;
      state.activeNextByte = 0;
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
   *   2. Wait NIC_DONE on fwd transport's sendStaging (backpressure)
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
   *     - Waits NIC_DONE on the forward channel's local completion counter.
   *     - Waits SLOT_FREE on the forward channel's local slot-free signal.
   *     - RDMA puts with DATA_READY on the forward remote channel and may
   *       batch NIC_DONE credit to the local completion counter.
   *
   * Any chain of send → forward* → recv is therefore valid: each
   * forward consumes exactly the signals its predecessor produces
   * and produces exactly the signals its successor expects.
   *
   * @param transport       Recv-side transport (this peer's receiver).
   * @param group           ThreadGroup (all threads participate).
   * @param dst             Application destination (may be nullptr if
   *                        CopyOp handles it, e.g. reduce-scatter).
   * @param fwdDevice       Forward-side send/recv device (next peer).
   * @param fwdTransport    Forward transport (sends to next peer in ring).
   * @param nbytes          Bytes to receive and forward.
   * @param max_signal_bytes Max bytes per signaled sub-chunk. 0 = perBlockSlot.
   * @param timeout         Optional timeout for wait operations.
   * @param args            Extra args forwarded to CopyOp::forward.
   */
  template <typename CopyOp = Memcpy, typename Transport, typename... Args>
  __device__ __forceinline__ void forward(
      Transport& transport,
      ThreadGroup& group,
      void* __restrict__ dst,
      IbSendRecvDevice& fwdDevice,
      Transport& fwdTransport,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
#if PIPES_IS_DEVICE_COMPILE
#ifdef __HIP_PLATFORM_AMD__
    static_assert(
        sizeof(CopyOp) == 0,
        "IbSendRecvDevice::forward() requires NVIDIA GPU (DOCA/IBGDA)");
#endif
    if (nbytes == 0) {
      return;
    }
    const std::size_t protocolBytes = align_protocol_bytes(nbytes);

    const int groupId = group.group_id;

    // --- recv side (this transport) ---
    const int recvMaxGroups = sendRecvState_.maxGroups;
    if (groupId >= recvMaxGroups) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: forward recv maxGroups=%d groupId=%u\n",
            recvMaxGroups,
            groupId);
      }
      PIPES_DEVICE_TRAP();
    }

    const std::size_t recvPerBlockSlot =
        (sendRecvState_.dataBufferSize / recvMaxGroups) & ~15ULL;
    if (recvPerBlockSlot == 0) {
      if (group.is_leader()) {
        printf("[PIPES] FATAL: forward recvPerBlockSlot=0\n");
      }
      PIPES_DEVICE_TRAP();
    }

    // --- fwd side (fwd transport) ---
    const int fwdMaxGroups = fwdDevice.sendRecvState_.maxGroups;
    if (groupId >= fwdMaxGroups) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: forward fwd maxGroups=%d groupId=%u\n",
            fwdMaxGroups,
            groupId);
      }
      PIPES_DEVICE_TRAP();
    }

    const std::size_t fwdPerBlockSlot =
        (fwdDevice.sendRecvState_.dataBufferSize / fwdMaxGroups) & ~15ULL;
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

    const int recvPipelineDepth = sendRecvState_.pipelineDepth;
    const std::size_t recvDataBufSize = sendRecvState_.dataBufferSize;
    const int recvStateIndex = progress_recv_index(group);
    auto& recvSlotState = progress_state_slot(group, recvStateIndex);
    assert_progress_slot_idle(group, recvSlotState, "forward recv");
    const uint64_t recvBaseByte = static_cast<uint64_t>(recvSlotState.nextStep);
    const std::size_t recvPipelineBytes = recvPerBlockSlot * recvPipelineDepth;

    const int fwdPipelineDepth = fwdDevice.sendRecvState_.pipelineDepth;
    const std::size_t fwdDataBufSize = fwdDevice.sendRecvState_.dataBufferSize;
    const int fwdStateIndex = fwdDevice.progress_send_index(group);
    auto& fwdSlotState = fwdDevice.progress_state_slot(group, fwdStateIndex);
    fwdDevice.assert_progress_slot_idle(group, fwdSlotState, "forward send");
    const uint64_t fwdBaseByte = static_cast<uint64_t>(fwdSlotState.nextStep);
    const std::size_t fwdPipelineBytes = fwdPerBlockSlot * fwdPipelineDepth;
    if (group.is_leader()) {
      recvSlotState.activeStage = detail::IbSendRecvProgressStage::Busy;
      recvSlotState.activeBaseStep = static_cast<int64_t>(recvBaseByte);
      recvSlotState.activeNextByte = 0;
      fwdSlotState.activeStage = detail::IbSendRecvProgressStage::Busy;
      fwdSlotState.activeBaseStep = static_cast<int64_t>(fwdBaseByte);
      fwdSlotState.activeNextByte = 0;
    }

    for (std::size_t dataOff = 0; dataOff < protocolBytes;) {
      // --- Recv side offsets ---
      const uint64_t recvStreamStart = recvBaseByte + dataOff;
      const std::size_t recvPipelineOff =
          static_cast<std::size_t>(recvStreamStart % recvPipelineBytes);
      const int recvSlot = static_cast<int>(recvPipelineOff / recvPerBlockSlot);
      const std::size_t recvSlotOff = recvSlot * recvDataBufSize;
      const std::size_t recvChunkOff =
          recvPipelineOff - recvSlot * recvPerBlockSlot;
      const std::size_t recvStagingOff =
          recvSlotOff + groupId * recvPerBlockSlot + recvChunkOff;

      // --- Fwd side offsets ---
      const uint64_t fwdStreamStart = fwdBaseByte + dataOff;
      const std::size_t fwdPipelineOff =
          static_cast<std::size_t>(fwdStreamStart % fwdPipelineBytes);
      const int fwdSlot = static_cast<int>(fwdPipelineOff / fwdPerBlockSlot);
      const std::size_t fwdSlotOff = fwdSlot * fwdDataBufSize;
      const std::size_t fwdChunkOff =
          fwdPipelineOff - fwdSlot * fwdPerBlockSlot;
      const std::size_t fwdStagingOff =
          fwdSlotOff + groupId * fwdPerBlockSlot + fwdChunkOff;
      const std::size_t recvSlotRemaining = recvPerBlockSlot - recvChunkOff;
      const std::size_t fwdSlotRemaining = fwdPerBlockSlot - fwdChunkOff;
      const std::size_t dataRemaining = protocolBytes - dataOff;
      std::size_t bytesThis =
          recvChunkSize < fwdChunkSize ? recvChunkSize : fwdChunkSize;
      bytesThis = bytesThis < dataRemaining ? bytesThis : dataRemaining;
      bytesThis = bytesThis < recvSlotRemaining ? bytesThis : recvSlotRemaining;
      bytesThis = bytesThis < fwdSlotRemaining ? bytesThis : fwdSlotRemaining;
      const uint64_t recvStreamEnd = recvStreamStart + bytesThis;
      const uint64_t fwdStreamEnd = fwdStreamStart + bytesThis;

      // (1) Wait for sender's DATA_READY.
      transport.wait_signal(
          group,
          sendRecvState_.localSignalBuf.subBuffer(groupId * sizeof(uint64_t)),
          recvStreamEnd,
          timeout);

      // (2) Wait for NIC_DONE on fwd's sendStaging (backpressure).
      if (fwdStreamEnd > fwdPipelineBytes) {
        fwdTransport.wait_counter(
            group,
            fwdDevice.sendRecvState_.localCounterBuf.subBuffer(
                groupId * sizeof(uint64_t)),
            fwdStreamEnd - fwdPipelineBytes,
            timeout);
      }

      // (3) CopyOp::forward — transform recv staging -> dst + fwd staging.
      const std::size_t validBytes =
          valid_payload_bytes(dataOff, bytesThis, nbytes);
      if (validBytes > 0) {
        CopyOp::forward(
            dst ? static_cast<char*>(dst) + dataOff : nullptr,
            fwdDevice.sendRecvState_.sendStagingPtr + fwdStagingOff,
            sendRecvState_.recvStagingPtr + recvStagingOff,
            validBytes,
            group,
            dataOff,
            args...);
      }
      group.sync();

      // (4) SLOT_FREE, posted unbatched every chunk. This must happen before
      //     the step-5 wait to break circular ring dependency. Advance
      //     reuseCreditStep so an interleaved recv() on this shared slot
      //     computes reuse credit from a current cursor.
      transport.signal(
          group,
          sendRecvState_.remoteSignalBuf.subBuffer(
              (recvMaxGroups + groupId) * sizeof(uint64_t)),
          bytesThis,
          IbDirection::Recv);
      if (group.is_leader()) {
        recvSlotState.reuseCreditStep = static_cast<int64_t>(recvStreamEnd);
      }

      // (5) Wait for fwd receiver's SLOT_FREE (backpressure on fwd's
      //     recvStaging).
      if (fwdStreamEnd > fwdPipelineBytes) {
        fwdTransport.wait_signal(
            group,
            fwdDevice.sendRecvState_.localSignalBuf.subBuffer(
                (fwdMaxGroups + groupId) * sizeof(uint64_t)),
            fwdStreamEnd - fwdPipelineBytes,
            timeout);
      }

      // (6) threadfence_system + RDMA put via fwd transport.
      __threadfence_system();
      group.sync();
      if (group.is_leader()) {
        fwdSlotState.reuseCreditStep = static_cast<int64_t>(fwdStreamEnd);
        ThreadGroup solo{
            0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
        fwdTransport.put(
            solo,
            fwdDevice.sendRecvState_.sendStagingBuf.subBuffer(fwdStagingOff),
            fwdDevice.sendRecvState_.recvStagingBuf.subBuffer(fwdStagingOff),
            bytesThis,
            fwdDevice.sendRecvState_.remoteSignalBuf.subBuffer(
                groupId * sizeof(uint64_t)),
            bytesThis,
            fwdDevice.sendRecvState_.localCounterCompletionBuf.subBuffer(
                groupId * sizeof(uint64_t)),
            bytesThis);
      }
      group.sync();
      dataOff += bytesThis;
    }

    // Update shared byte cursors for both recv and fwd sides.
    if (group.is_leader()) {
      recvSlotState.nextStep =
          static_cast<int64_t>(recvBaseByte + protocolBytes);
      recvSlotState.activeStage = detail::IbSendRecvProgressStage::Done;
      recvSlotState.activeBaseStep = 0;
      recvSlotState.activeNextByte = 0;
      fwdSlotState.nextStep = static_cast<int64_t>(fwdBaseByte + protocolBytes);
      fwdSlotState.activeStage = detail::IbSendRecvProgressStage::Done;
      fwdSlotState.activeBaseStep = 0;
      fwdSlotState.activeNextByte = 0;
    }
    group.sync();
#else
    (void)transport;
    (void)group;
    (void)dst;
    (void)fwdDevice;
    (void)fwdTransport;
    (void)nbytes;
    (void)max_signal_bytes;
    (void)timeout;
#endif
  }

  /**
   * Maximum bytes a block can send without blocking on pipeline backpressure.
   *
   * The staging buffer is split into pipelineDepth slots, each with a fixed
   * per-channel partition. A block can fill all its slots before the NIC must
   * drain any of them, so the non-blocking window is:
   *   perChannelSize * pipelineDepth
   *
   * Callers should loop over their data in pipeline_window-sized chunks so
   * that send()/forward() never stall waiting for a free slot.
   */
  __device__ __forceinline__ std::size_t pipeline_window() const {
    const std::size_t per_block_slot =
        (sendRecvState_.dataBufferSize / sendRecvState_.maxGroups) & ~15ULL;
    return per_block_slot * sendRecvState_.pipelineDepth;
  }

 private:
  /**
   * Physical staging range for the next resumable progress step.
   *
   * `stagingOff` is an offset into the transport-owned send/recv staging
   * buffers. `dataOff` is the matching protocol offset into the caller's user
   * buffer. `bytes` never crosses a per-block staging partition or the
   * reserved protocol byte count. `streamEnd` is the absolute protocol byte
   * value after this chunk and is used as the DATA_READY, SLOT_FREE, and
   * NIC_DONE readiness threshold. `dataOff` is a protocol offset; callers mask
   * it against the payload byte count before invoking user-buffer copy
   * callbacks.
   */
  struct ProgressChunk {
    std::size_t stagingOff;
    std::size_t dataOff;
    std::size_t bytes;
    uint64_t streamEnd;
  };

  /**
   * Register-only geometry for one resumable progress call.
   *
   * This is intentionally not stored in `IbSendRecvState::ProgressSlot`:
   * callers pass the same static geometry to init and progress, and each
   * progress call derives these values in registers instead of reloading
   * duplicated fields from HBM-backed progress state.
   */
  struct ProgressGeometry {
    int groupId;
    std::size_t payloadBytes;
    std::size_t protocolBytes;
    std::size_t perBlockSlot;
    std::size_t chunkSize;
  };

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

  /**
   * Return the internal send progress slot index for `group`.
   */
  __device__ __forceinline__ int progress_send_index(ThreadGroup& group) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return progress_index_for_group(group, 0);
#else
    (void)group;
    return 0;
#endif
  }

  /**
   * Return the internal recv progress slot index for `group`.
   */
  __device__ __forceinline__ int progress_recv_index(ThreadGroup& group) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return progress_index_for_group(group, sendRecvState_.maxGroups);
#else
    (void)group;
    return 0;
#endif
  }

  /**
   * Validate a progress group and map it into the transport-owned slot array.
   */
  __device__ __forceinline__ int progress_index_for_group(
      ThreadGroup& group,
      int baseIndex) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (sendRecvState_.state.empty() ||
        sendRecvState_.state.data() == nullptr) {
      if (group.is_leader()) {
        printf("[PIPES] FATAL: send/recv state is null\n");
      }
      PIPES_DEVICE_TRAP();
    }
    if (sendRecvState_.maxGroups <= 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: send/recv maxGroups must be > 0, got %d\n",
            sendRecvState_.maxGroups);
      }
      PIPES_DEVICE_TRAP();
    }
    const auto requiredStateSlots =
        static_cast<uint32_t>(2 * sendRecvState_.maxGroups);
    if (sendRecvState_.state.size() < requiredStateSlots ||
        group.group_id >= static_cast<uint32_t>(sendRecvState_.maxGroups)) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: progress group_id=%u out of range [0, %d), "
            "state slots=%u\n",
            group.group_id,
            sendRecvState_.maxGroups,
            static_cast<unsigned>(sendRecvState_.state.size()));
      }
      PIPES_DEVICE_TRAP();
    }
    return baseIndex + static_cast<int>(group.group_id);
#else
    (void)group;
    (void)baseIndex;
    return 0;
#endif
  }

  /**
   * Return a reference to a transport-owned progress state slot.
   */
  __device__ __forceinline__ IbSendRecvState::ProgressSlot& progress_state_slot(
      ThreadGroup& group,
      int progressIndex) const {
    (void)group;
    return sendRecvState_.state[progressIndex];
  }

  /**
   * Trap if a caller tries to start a second send/recv before the first ends.
   *
   * The broadcast is the ordering point for init callers: if the leader sees a
   * non-idle slot, every thread traps before any caller can store new state.
   */
  __device__ __forceinline__ void assert_progress_slot_idle(
      ThreadGroup& group,
      const IbSendRecvState::ProgressSlot& state,
      const char* direction) const {
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
      int progressIndex,
      const IbSendRecvState::ProgressSlot& state) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (group.is_leader()) {
      auto& slot = sendRecvState_.state[progressIndex];
      slot.reuseCreditStep = state.reuseCreditStep;
      slot.activeNextByte = state.activeNextByte;
      slot.activeBaseStep = state.activeBaseStep;
      slot.activeStage = state.activeStage;
    }
    group.sync();
#else
    (void)group;
    (void)progressIndex;
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
      ThreadGroup& group,
      std::size_t nbytes,
      std::size_t max_signal_bytes,
      const char* opName) const {
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
    const int maxGroups = sendRecvState_.maxGroups;
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

    const std::size_t perBlockSlot =
        (sendRecvState_.dataBufferSize / maxGroups) & ~15ULL;
    if (perBlockSlot == 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: %s perBlockSlot=0 "
            "(dataBufferSize=%llu, maxGroups=%d)\n",
            opName,
            (unsigned long long)sendRecvState_.dataBufferSize,
            maxGroups);
      }
      PIPES_DEVICE_TRAP();
    }

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
        .chunkSize = chunkSize,
    };
#else
    (void)group;
    (void)nbytes;
    (void)max_signal_bytes;
    (void)opName;
    return {};
#endif
  }

  __device__ __forceinline__ std::size_t nic_done_credit_quantum(
      std::size_t perBlockSlot) const {
    return detail::ib_send_recv_nic_done_credit_quantum(
        perBlockSlot, sendRecvState_.pipelineDepth);
  }

  __device__ __forceinline__ std::size_t slot_free_credit_quantum(
      std::size_t perBlockSlot) const {
    return detail::ib_send_recv_slot_free_credit_quantum(
        perBlockSlot, sendRecvState_.pipelineDepth);
  }

  __device__ __forceinline__ bool should_post_nic_done_credit(
      uint64_t streamEnd,
      int64_t reuseCreditStep,
      const ProgressGeometry& geometry) const {
    return should_post_nic_done_credit(
        streamEnd, reuseCreditStep, geometry.perBlockSlot);
  }

  __device__ __forceinline__ bool should_post_nic_done_credit(
      uint64_t streamEnd,
      int64_t reuseCreditStep,
      std::size_t perBlockSlot) const {
    return should_post_credit_at_quantum(
        streamEnd, reuseCreditStep, nic_done_credit_quantum(perBlockSlot));
  }

  __device__ __forceinline__ bool should_post_slot_free_credit(
      uint64_t streamEnd,
      int64_t reuseCreditStep,
      const ProgressGeometry& geometry) const {
    return should_post_slot_free_credit(
        streamEnd, reuseCreditStep, geometry.perBlockSlot);
  }

  __device__ __forceinline__ bool should_post_slot_free_credit(
      uint64_t streamEnd,
      int64_t reuseCreditStep,
      std::size_t perBlockSlot) const {
    return should_post_credit_at_quantum(
        streamEnd, reuseCreditStep, slot_free_credit_quantum(perBlockSlot));
  }

  __device__ __forceinline__ bool should_post_credit_at_quantum(
      uint64_t streamEnd,
      int64_t reuseCreditStep,
      std::size_t creditQuantum) const {
    const uint64_t posted = reuse_credit_posted_step(reuseCreditStep);
    return streamEnd > posted && streamEnd - posted >= creditQuantum;
  }

  __device__ __forceinline__ uint64_t
  reuse_credit_value(uint64_t streamEnd, int64_t reuseCreditStep) const {
    return streamEnd - reuse_credit_posted_step(reuseCreditStep);
  }

  __device__ __forceinline__ static uint64_t reuse_credit_posted_step(
      int64_t reuseCreditStep) {
    return reuseCreditStep > 0 ? static_cast<uint64_t>(reuseCreditStep) : 0ULL;
  }

  /**
   * Reserve a non-overlapping protocol byte range for one progress state.
   *
   * Blocking send()/recv() read `state[].nextStep` at call entry and commit it
   * at completion. Progress init cannot wait until completion because progress
   * operations may complete across many bounded calls. Reserving at init gives
   * the transport-owned state a stable byte-stream base and makes later
   * blocking calls start after all in-flight progress ranges.
   */
  __device__ __forceinline__ int64_t reserve_progress_step(
      ThreadGroup& group,
      int stateIndex,
      std::size_t nbytes) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    uint64_t baseStep = 0;
    if (group.is_leader()) {
      auto& slot = sendRecvState_.state[stateIndex];
      baseStep = static_cast<uint64_t>(slot.nextStep);
      slot.nextStep = static_cast<int64_t>(baseStep + nbytes);
    }
    baseStep = group.broadcast<uint64_t>(baseStep);
    return static_cast<int64_t>(baseStep);
#else
    (void)group;
    (void)stateIndex;
    (void)nbytes;
    return 0;
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
      const IbSendRecvState::ProgressSlot& state) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (state.activeStage != detail::IbSendRecvProgressStage::WaitNicDone &&
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
      const IbSendRecvState::ProgressSlot& state) const {
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
      detail::IbSendRecvProgressStage next) const {
    switch (current) {
      case detail::IbSendRecvProgressStage::WaitNicDone:
        return next == detail::IbSendRecvProgressStage::WaitSlotFree;
      case detail::IbSendRecvProgressStage::WaitSlotFree:
        return next == detail::IbSendRecvProgressStage::WaitNicDone ||
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
      IbSendRecvState::ProgressSlot& state,
      detail::IbSendRecvProgressStage next) const {
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
   * The transport stores each logical slot as `dataBufferSize` bytes, split
   * into geometry-specific per-group partitions. The protocol cursor advances
   * in bytes, not slots, so `baseStep + nextByte` is reduced modulo
   * `perBlockSlot * pipelineDepth` to pick the ring slot and per-group offset.
   *
   * The returned chunk is clipped by three boundaries: the configured
   * sub-chunk size, remaining protocol bytes, and remaining bytes in the
   * current per-block staging partition. This keeps every progress call
   * bounded and prevents a single RDMA put or recv copy from spanning two
   * staging slots.
   */
  __device__ __forceinline__ ProgressChunk next_chunk(
      const IbSendRecvState::ProgressSlot& state,
      const ProgressGeometry& geometry) const {
    const uint64_t streamStart =
        static_cast<uint64_t>(state.activeBaseStep) + state.activeNextByte;
    const std::size_t pipelineBytes = geometry.perBlockSlot *
        static_cast<std::size_t>(sendRecvState_.pipelineDepth);
    const std::size_t pipelineOff =
        static_cast<std::size_t>(streamStart % pipelineBytes);
    const int slot = static_cast<int>(pipelineOff / geometry.perBlockSlot);
    const std::size_t slotOff =
        static_cast<std::size_t>(slot) * sendRecvState_.dataBufferSize;
    const std::size_t chunkOff =
        pipelineOff - static_cast<std::size_t>(slot) * geometry.perBlockSlot;
    const std::size_t slotRemaining = geometry.perBlockSlot - chunkOff;
    const std::size_t dataRemaining =
        geometry.protocolBytes - state.activeNextByte;
    std::size_t bytes =
        geometry.chunkSize < dataRemaining ? geometry.chunkSize : dataRemaining;
    bytes = bytes < slotRemaining ? bytes : slotRemaining;
    return ProgressChunk{
        .stagingOff = slotOff +
            static_cast<std::size_t>(geometry.groupId) * geometry.perBlockSlot +
            chunkOff,
        .dataOff = state.activeNextByte,
        .bytes = bytes,
        .streamEnd = streamStart + bytes,
    };
  }
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

  // Common slot-index IB device API.
  __device__ void signal(int signalId, uint64_t signalVal = 1);

  __device__ void
  signal(ThreadGroup& group, int signalId, uint64_t signalVal = 1);

  __device__ void put(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      int signalId = -1,
      uint64_t signalVal = 1,
      int counterId = -1,
      uint64_t counterVal = 1);

  __device__ void put(
      const IbgdaLocalBuffer& localBuf,
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

  __device__ void put(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1);

  __device__ void put(
      const IbgdaLocalBuffer& localBuf,
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

  // Pipelined send/recv — forwarded to the active backend's IbSendRecvDevice.
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

  // Per-block pipelined staging window — forwarded to the active backend.
  __device__ __forceinline__ std::size_t pipeline_window() const;

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
