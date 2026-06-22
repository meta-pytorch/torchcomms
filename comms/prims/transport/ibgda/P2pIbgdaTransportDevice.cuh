// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// `<cuda/atomic>` is NVIDIA-only. AMD uses `__hip_atomic_*` builtins
// (already pulled in transitively via `HipDeviceCompat.h` below).
#ifndef __HIP_PLATFORM_AMD__
#include <cuda/atomic>
#endif

#include <cerrno>
#include <cstddef>
#include <cstdint>

// On NVIDIA, pull in the DOCA verbs headers + the Meta-internal
// `DocaVerbsUtils.cuh` wrapper that defines `comms::prims::doca_fence`.
// On AMD, the DOCA headers don't exist; the DocaCompat shim provides
// `doca_*` type aliases, constants, function shims, and `doca_fence`.
#ifdef __HIP_PLATFORM_AMD__
#include "comms/prims/transport/amd/DocaCompat.h"
#else
#include <device/doca_gpunetio_dev_verbs_counter.cuh>
#include <device/doca_gpunetio_dev_verbs_onesided.cuh>

#include "comms/prims/platform/DocaVerbsUtils.cuh"
#endif
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/DeviceMacros.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/trace/PipesTraceTypes.h"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace comms::prims {

struct Memcpy;

inline constexpr uint64_t kDefaultDeviceTimeoutCycles = 10'000'000'000ULL;

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

// `PIPES_DEVICE_TRAP()` is defined in `comms/prims/core/DeviceMacros.cuh` and
// is intentionally available across all `comms/prims` device headers.

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

// Slot-id bounds checks for the slot-index API. Catches both
// out-of-range slot ids and slot-index calls made when the transport was
// constructed with no owned signal/counter buffer (numSlots == 0).
#if PIPES_IS_DEVICE_COMPILE
#define IBGDA_CHECK_SLOT_ID(id, count, kind)            \
  do {                                                  \
    if (!((id) >= 0 && (id) < (count))) {               \
      printf(                                           \
          "P2pIbgdaTransportDevice: " kind              \
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
#define IBGDA_CHECK_SLOT_ID(id, count, kind) assert((id) >= 0 && (id) < (count))
#endif

/**
 * NicDeviceIbgdaResources - Per-NIC bundle of QPs and sink lkey
 *
 * Owns the QPs (primary + companion for compound put+signal+counter ops)
 * and the sink lkey for atomic FA responses on a single NIC. The
 * P2pIbgdaTransportDevice holds a `DeviceSpan<NicDeviceIbgdaResources>` indexed
 * by physical NIC slot.
 */
struct NicDeviceIbgdaResources {
  DeviceSpan<doca_gpu_dev_verbs_qp*> qps{};
  DeviceSpan<doca_gpu_dev_verbs_qp*> companion_qps{};
  NetworkLKey sink_lkey{};
  int device_id{0};

  __host__ __device__ int get_nic_id() const {
    return device_id;
  }
};

inline constexpr int kIbgdaMaxQpLanesPerBlock = 64;

struct IbgdaBlockQpState {
  uint32_t put_rr{0};
  uint64_t pending_flush_lanes_mask{0};
  uint64_t last_flush_wqe[kIbgdaMaxQpLanesPerBlock]{};
};

/**
 * P2pIbgdaTransportDevice - Device-side per-peer RDMA transport handle
 *
 * Every method has two overloads:
 *   Group-scope: put(group, ...) — all threads in group must call.
 *     QP selection is owned by the physical CUDA block. The block
 *     round-robins put operations across NIC-first QP lanes and preserves
 *     signal/flush ordering for operations issued by the same block_id.
 *     Data transfer uses the exact buffer span supplied by the caller.
 *     Threads in the group coordinate the operation; callers that want the
 *     transport to shard a larger buffer should use put_cooperative().
 *     Signal/counter/flush are leader-only with group.sync().
 *
 *   Thread-scope: put(...) — single thread calls.
 *     QP selection uses the caller's physical blockIdx.x. Implemented as a
 *     thin wrapper: creates a solo ThreadGroup with block_id=blockIdx.x, then
 *     forwards to the group-scope implementation.
 *
 * CRITICAL: Do not rely on scope-family mixing for synchronization.
 *   Thread-scope wrappers do not synchronize with other threads in the block.
 *   put(group,...) -> signal(group,0) does not by itself order prior puts
 *   issued on other QP lanes. Call fence(group) or flush(group) before a
 *   standalone signal when the signal is meant to announce completion of prior
 *   puts.
 *
 * CRITICAL: Same-block warp-scope batching is not a supported ordering
 * contract in this implementation. The block_id owns one logical ordered
 * stream. If multiple independent warps use the same block_id, this transport
 * does not infer cross-warp order for a later signal() or flush(); the caller
 * must use a CTA-level barrier or another protocol-level synchronization
 * before issuing the covering signal/flush. In particular, do not rely on:
 *
 *   warp0: put(A); put(B); signal(S);
 *   warp1: put(C); put(D); signal(S);
 *
 * to mean that either signal covers both warps' puts. Each warp only has its
 * own program order, and the relative order between the warps is unspecified.
 * A fused put+signal from multiple warps is only self-ordered for each
 * operation's own put before its own signal; it is not a cross-warp batch
 * completion signal. Full warp-specialized concurrent issue needs future
 * per-QP-lane reservation/ordering state.
 *
 * Signal WQEs use the IB FENCE bit, which orders the signal after prior WQEs
 * on the same QP only. A fused put(..., signal) posts the put and signal on
 * the same QP, so the signal covers that put. A standalone signal uses the
 * block's control lane and does not cover earlier round-robined puts unless
 * the caller explicitly calls fence()/flush() first.
 * put() returns void — completion via wait_signal/wait_counter/flush.
 *
 * Two API layers:
 *   1. Slot-index API: resolve owned buffers by slot index, then forward
 *      to explicit-buffer methods. Requires owned buffers set in constructor.
 *   2. Explicit-buffer API: caller provides pre-resolved buffer pointers.
 *      Buffer ptr==nullptr means "disabled" (no signal/counter).
 */
class P2pIbgdaTransportDevice {
 public:
  // Default ctor required so an array of these can be cudaMemcpy'd from host
  // (see MultipeerIbgdaTransportCuda.cu::buildDeviceTransportsOnGpu). Do not
  // call methods on a default-constructed instance — nicDevices_ is empty.
  P2pIbgdaTransportDevice() = default;

  /**
   * Construct a per-peer device transport handle.
   *
   * Each P2p instance owns one peer's NICs. Each NicDeviceIbgdaResources
   * carries its main and companion QPs plus a sink lkey. Lane selection is
   * block-owned: each physical CUDA block round-robins its puts across
   * numNics * qpsPerBlockPerNic lanes, using NIC-first lane ordinals.
   *
   * Single-NIC usage: pass a 1-element nicDevices span. All ops fall through
   * to NIC 0.
   *
   * @param nicDevices          GPU span of per-NIC bundles (length =
   *                              numNics). Each NicDeviceIbgdaResources owns
   *                              maxGroups * qpsPerBlockPerNic main QPs and
   *                              maxGroups companion QPs.
   * @param ownedRemoteSignalBuf  Remote-side signal outbox: writing here
   *                              targets the peer's local signal inbox.
   *                              Used by the slot-index signal API.
   * @param ownedLocalSignalBuf   Local signal inbox: receives signals from
   *                              the peer. Used by the slot-index
   *                              wait_signal/reset_signal/read_signal APIs.
   * @param ownedCounterBuf       Local counter buffer for compound
   *                              put+signal+counter and the slot-index
   *                              counter APIs. May be empty if not used.
   * @param numSignalSlots        Number of uint64_t slots in the owned
   *                              signal buffers. Used to bounds-check
   *                              signalId. Zero disables the slot-index
   *                              signal API.
   * @param numCounterSlots       Number of uint64_t slots in the owned
   *                              counter buffer. Zero disables the
   *                              slot-index counter API.
   * @param sendRecvState         Optional pipelined send/recv protocol state.
   *                              When empty, send()/recv() are unavailable.
   */
  __host__ __device__ P2pIbgdaTransportDevice(
      DeviceSpan<NicDeviceIbgdaResources> nicDevices,
      IbgdaRemoteBuffer ownedRemoteSignalBuf = {},
      IbgdaLocalBuffer ownedLocalSignalBuf = {},
      IbgdaLocalBuffer ownedCounterBuf = {},
      int numSignalSlots = 0,
      int numCounterSlots = 0,
      int maxGroups = 0,
      int qpsPerBlockPerNic = 1,
      DeviceSpan<IbgdaBlockQpState> blockQpState = {},
      IbSendRecvState sendRecvState = {})
      : nicDevices_(nicDevices),
        ownedRemoteSignalBuf_(ownedRemoteSignalBuf),
        ownedLocalSignalBuf_(ownedLocalSignalBuf),
        ownedCounterBuf_(ownedCounterBuf),
        numSignalSlots_(numSignalSlots),
        numCounterSlots_(numCounterSlots),
        maxGroups_(maxGroups),
        qpsPerBlockPerNic_(qpsPerBlockPerNic),
        blockQpState_(blockQpState),
        sendRecvState_(sendRecvState) {}

  // =========================================================================
  // Slot-Index API (resolves owned buffers, forwards to explicit-buffer API)
  // =========================================================================

  /**
   * put (group-scope, slot-index) - RDMA Write with slot-index signal/counter.
   *
   * Resolves signal/counter slots from owned buffers, then forwards to the
   * explicit-buffer put().
   *
   * @param group       Thread group; all threads must call. Group cooperates
   *                    on WQE construction; leader posts signal/counter.
   * @param localBuf    Source buffer on this GPU (registered for RDMA).
   * @param remoteBuf   Destination buffer on the peer.
   * @param nbytes      Number of bytes to transfer.
   * @param signalId    Slot index into the peer's signal inbox. -1 disables
   *                    signaling. Bounds-checked against numSignalSlots_.
   * @param signalVal   Value added to the peer's signal slot (atomic FA).
   * @param counterId   Slot index into the local counter buffer. -1 disables
   *                    the counter. Bounds-checked against numCounterSlots_.
   * @param counterVal  Value added to the local counter slot.
   */
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
        (counterId >= 0) ? counter_slot(counterId) : IbgdaLocalBuffer{};
    put(group,
        localBuf,
        remoteBuf,
        nbytes,
        sigSlot,
        signalVal,
        ctrSlot,
        counterVal);
  }

  /**
   * put_cooperative (group-scope, slot-index) - Shard a larger RDMA Write
   * across the threads in this group, with optional slot-index
   * signal/counter.
   *
   * This is the explicit helper for callers that want the transport to split
   * the provided buffer across the group. Plain put(group, ...) expects the
   * caller to pass the exact data range for this group.
   */
  __device__ void put_cooperative(
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
        (counterId >= 0) ? counter_slot(counterId) : IbgdaLocalBuffer{};
    put_cooperative(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        sigSlot,
        signalVal,
        ctrSlot,
        counterVal);
  }

  /**
   * put (thread-scope, slot-index) - Single-thread variant of slot-index put.
   * Caller is responsible for gating to one thread. Uses the caller's physical
   * block-owned QP resources.
   * Args match the group-scope overload.
   */
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

  /**
   * signal (group-scope, slot-index) - Fenced RDMA atomic add by slot index.
   *
   * Standalone signal posts on this block's control lane. It does not wait for
   * prior puts issued on other round-robined QP lanes. If this signal should
   * announce completion of earlier put() calls, the user/protocol must call
   * fence(group) or flush(group) before signal(group, ...).
   *
   * @param group     Thread group; all threads must call. Leader posts WQE.
   * @param signalId  Slot index into the peer's signal inbox (>= 0,
   *                  < numSignalSlots_).
   * @param signalVal Value added to the peer's signal slot.
   */
  __device__ void
  signal(ThreadGroup& group, int signalId, uint64_t signalVal = 1) {
    signal(group, remote_signal_slot(signalId), signalVal);
  }

  /**
   * signal (thread-scope, slot-index) - Single-thread variant. Uses the
   * caller's physical block-owned QP resources.
   */
  __device__ void signal(int signalId, uint64_t signalVal = 1) {
    ThreadGroup solo = make_thread_solo();
    signal(solo, signalId, signalVal);
  }

  /**
   * wait_signal (group-scope, slot-index) - Spin until local inbox slot >=
   * expected.
   *
   * @param group     Thread group; all threads must call. Leader spins, all
   *                  sync after.
   * @param signalId  Slot index into the local signal inbox.
   * @param expected  Threshold; wait returns when slot value >= expected.
   * @param timeout   Optional spin timeout. On expiry, prints diagnostic and
   *                  __trap()s.
   */
  __device__ void wait_signal(
      ThreadGroup& group,
      int signalId,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    wait_signal(group, local_signal_slot(signalId), expected, timeout);
  }

  /** wait_signal (thread-scope, slot-index) - Single-thread variant. */
  __device__ void wait_signal(
      int signalId,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    ThreadGroup solo = make_thread_solo();
    wait_signal(solo, signalId, expected, timeout);
  }

  /**
   * wait_counter (group-scope, slot-index) - Spin until local counter slot >=
   * expected.
   *
   * @param group     Thread group; all threads must call. Leader spins.
   * @param counterId Slot index into the local counter buffer.
   * @param expected  Threshold; wait returns when slot value >= expected.
   * @param timeout   Optional spin timeout.
   */
  __device__ void wait_counter(
      ThreadGroup& group,
      int counterId,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    wait_counter(group, counter_slot(counterId), expected, timeout);
  }

  /** wait_counter (thread-scope, slot-index) - Single-thread variant. */
  __device__ void wait_counter(
      int counterId,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    ThreadGroup solo = make_thread_solo();
    wait_counter(solo, counterId, expected, timeout);
  }

  /**
   * reset_signal (group-scope, slot-index) - Zero a local signal inbox slot.
   *
   * @param group    Thread group; all threads must call. Leader writes 0,
   *                 then a device-scope fence.
   * @param signalId Slot index into the local signal inbox.
   */
  __device__ void reset_signal(ThreadGroup& group, int signalId) {
    reset_signal(group, local_signal_slot(signalId));
  }

  /** reset_signal (thread-scope, slot-index) - Single-thread variant. */
  __device__ void reset_signal(int signalId) {
    ThreadGroup solo = make_thread_solo();
    reset_signal(solo, signalId);
  }

  /**
   * reset_counter (group-scope, slot-index) - Zero a local counter slot.
   *
   * @param group     Thread group; all threads must call. Leader writes 0.
   * @param counterId Slot index into the local counter buffer.
   */
  __device__ void reset_counter(ThreadGroup& group, int counterId) {
    reset_counter(group, counter_slot(counterId));
  }

  /** reset_counter (thread-scope, slot-index) - Single-thread variant. */
  __device__ void reset_counter(int counterId) {
    ThreadGroup solo = make_thread_solo();
    reset_counter(solo, counterId);
  }

  /**
   * read_signal (slot-index) - Non-blocking acquire read of a local signal
   * inbox slot.
   *
   * @param signalId Slot index into the local signal inbox.
   * @return         Current value of the slot.
   */
  __device__ uint64_t read_signal(int signalId) const {
    return read_signal(local_signal_slot(signalId));
  }

  /**
   * read_counter (slot-index) - Non-blocking acquire read of a local counter
   * slot.
   *
   * @param counterId Slot index into the local counter buffer.
   * @return          Current value of the slot.
   */
  __device__ uint64_t read_counter(int counterId) const {
    return read_counter(counter_slot(counterId));
  }

  // =========================================================================
  // Explicit-Buffer API (caller provides pre-resolved buffer pointers)
  // =========================================================================

  // =========================================================================
  // Data Transfer
  // =========================================================================

  /**
   * put (group-scope) - Group-local RDMA Write with optional signal /
   * counter.
   *
   * All threads in the group must call. The provided localBuf/remoteBuf/nbytes
   * are treated as the exact range for this group; put() does not shard a
   * larger user buffer. Use put_cooperative() for the convenience behavior that
   * splits the provided range across the group.
   *
   * Returns void; completion is observed via wait_signal/wait_counter/flush.
   *
   * NOTE: signalBuf is intentionally NOT defaulted, even though `= {}` would
   * mean "no signal". Defaulting it would make put(group, local, remote, n)
   * ambiguous against the slot-index overload. Pass IbgdaRemoteBuffer{}
   * explicitly for no-signal puts, or use the slot-index overload.
   *
   * @param group      Thread group; all threads must call.
   * @param localBuf   Source buffer on this GPU.
   * @param remoteBuf  Destination buffer on the peer.
   * @param nbytes     Number of bytes to transfer.
   * @param signalBuf  Pre-resolved remote signal slot. ptr==nullptr disables
   *                   signaling; otherwise leader posts a FENCEd atomic FA so
   *                   the signal arrives after the put completes at the NIC.
   * @param signalVal  Value added to *signalBuf (atomic FA).
   * @param counterBuf Pre-resolved local counter slot. ptr==nullptr disables
   *                   the counter. With a put, the counter waits for the put
   *                   WQE through the companion QP.
   * @param counterVal Value added to *counterBuf.
   */
  __device__ void put(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1) {
    put_impl(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        signalBuf,
        signalVal,
        counterBuf,
        counterVal);
  }

  /**
   * put (thread-scope) - Single-thread variant. Caller gates.
   *
   * signalBuf intentionally not defaulted (see group-scope sibling above).
   */
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

  /**
   * put_cooperative (group-scope) - Group-cooperative RDMA Write with optional
   * signal/counter.
   *
   * All threads in the group must call. The transport splits the provided
   * range across group lanes and posts one data WQE per lane, then the leader
   * posts any requested signal/counter WQE.
   */
  __device__ void put_cooperative(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer& counterBuf = {},
      uint64_t counterVal = 1) {
    put_cooperative_impl(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        signalBuf,
        signalVal,
        counterBuf,
        counterVal);
  }

  // =========================================================================
  // Signal (always fenced)
  // =========================================================================

  /**
   * signal (group-scope) - Fenced RDMA atomic add to a remote signal slot.
   *
   * Standalone signal posts on this block's control lane. The IB FENCE bit
   * orders it after earlier WQEs on that same QP only; it does not drain the
   * block's round-robined put lanes. If this signal is meant to cover previous
   * put() calls, the user/protocol must call fence(group) or flush(group)
   * before signal(group, ...). A fused put(..., signal) remains self-ordered
   * because the put and signal are posted on the same QP.
   *
   * @param group     Thread group; all threads must call. Leader posts WQE,
   *                  all sync.
   * @param signalBuf Pre-resolved remote signal slot (must point to the
   *                  exact uint64_t slot).
   * @param signalVal Value added to *signalBuf (atomic FA).
   */
  __device__ void signal(
      ThreadGroup& group,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1) {
    if (group.is_leader()) {
      validate_group_scope(group);
      IbgdaLane lane = control_lane(group.block_id);
      const uint64_t signalTicket = signal_fenced(lane, signalBuf, signalVal);
      record_signal_wqe(lane, signalTicket);
    }
    group.sync();
  }

  /** signal (thread-scope) - Single-thread variant. */
  __device__ void signal(
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1) {
    ThreadGroup solo = make_thread_solo();
    signal(solo, signalBuf, signalVal);
  }

  // =========================================================================
  // Synchronization
  // =========================================================================

  /**
   * wait_signal (group-scope) - Spin until *signalBuf >= expected.
   *
   * @param group     Thread group; all threads must call. Leader spins, all
   *                  sync after.
   * @param signalBuf Pre-resolved local signal slot.
   * @param expected  Threshold; returns when slot value >= expected.
   * @param timeout   Optional spin timeout. On expiry, prints diagnostic and
   *                  __trap()s.
   */
  __device__ void wait_signal(
      ThreadGroup& group,
      const IbgdaLocalBuffer& signalBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    wait_signal_impl(group, signalBuf, expected, timeout);
  }

  /** wait_signal (thread-scope) - Single-thread variant. */
  __device__ void wait_signal(
      const IbgdaLocalBuffer& signalBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    ThreadGroup solo = make_thread_solo();
    wait_signal(solo, signalBuf, expected, timeout);
  }

  /**
   * wait_counter (group-scope) - Spin until *counterBuf >= expected.
   *
   * @param group      Thread group; all threads must call. Leader spins.
   * @param counterBuf Pre-resolved local counter slot.
   * @param expected   Threshold; returns when slot value >= expected.
   * @param timeout    Optional spin timeout.
   */
  __device__ void wait_counter(
      ThreadGroup& group,
      const IbgdaLocalBuffer& counterBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    wait_counter_impl(group, counterBuf, expected, timeout);
  }

  /** wait_counter (thread-scope) - Single-thread variant. */
  __device__ void wait_counter(
      const IbgdaLocalBuffer& counterBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    ThreadGroup solo = make_thread_solo();
    wait_counter(solo, counterBuf, expected, timeout);
  }

  /**
   * flush (group-scope) - Wait for this block's locally tracked transport
   * operations to complete.
   *
   * Flush waits for the block's recorded data put and signal WQEs. Counter
   * WQEs are local-tracking operations and are intentionally not part of the
   * flush set.
   *
   * @param group Thread group; all threads must call. Leader waits, all sync.
   */
  __device__ void flush(ThreadGroup& group) {
    if (group.is_leader()) {
      validate_group_scope(group);
      drain_flush_lanes(group.block_id);
    }
    group.sync();
  }

  /** flush (thread-scope) - Single-thread variant. */
  __device__ void flush() {
    ThreadGroup solo = make_thread_solo();
    flush(solo);
  }

  /**
   * fence (group-scope) - Drain all locally tracked WQEs for this block.
   *
   * Aliased to flush(). Use this before a standalone signal when that signal
   * must announce completion of prior round-robined puts from the same block.
   *
   * @param group Thread group; all threads must call.
   */
  __device__ void fence(ThreadGroup& group) {
    flush(group);
  }

  /** fence (thread-scope) - Single-thread variant. */
  __device__ void fence() {
    flush();
  }

  // =========================================================================
  // Reset
  // =========================================================================

  /**
   * reset_signal (group-scope) - Zero a local signal slot.
   *
   * @param group     Thread group; all threads must call. Leader writes 0,
   *                  then a device-scope fence.
   * @param signalBuf Pre-resolved local signal slot.
   */
  __device__ void reset_signal(
      ThreadGroup& group,
      const IbgdaLocalBuffer& signalBuf) {
    reset_local_impl(group, signalBuf);
  }

  /** reset_signal (thread-scope) - Single-thread variant. */
  __device__ void reset_signal(const IbgdaLocalBuffer& signalBuf) {
    ThreadGroup solo = make_thread_solo();
    reset_signal(solo, signalBuf);
  }

  /**
   * reset_counter (group-scope) - Zero a local counter slot.
   *
   * @param group      Thread group; all threads must call. Leader writes 0.
   * @param counterBuf Pre-resolved local counter slot.
   */
  __device__ void reset_counter(
      ThreadGroup& group,
      const IbgdaLocalBuffer& counterBuf) {
    reset_local_impl(group, counterBuf);
  }

  /** reset_counter (thread-scope) - Single-thread variant. */
  __device__ void reset_counter(const IbgdaLocalBuffer& counterBuf) {
    ThreadGroup solo = make_thread_solo();
    reset_counter(solo, counterBuf);
  }

  // =========================================================================
  // Non-blocking reads (no QP, no group). Buffer must point to exact slot.
  // =========================================================================

  /**
   * read_signal - Non-blocking acquire read of a local signal slot.
   *
   * @param signalBuf Pre-resolved local signal slot.
   * @return          Current value of *signalBuf.
   */
  __device__ uint64_t read_signal(const IbgdaLocalBuffer& signalBuf) const {
    return load_acquire_system_u64(signalBuf.ptr);
  }

  /**
   * read_counter - Non-blocking acquire read of a local counter slot.
   *
   * @param counterBuf Pre-resolved local counter slot.
   * @return           Current value of *counterBuf.
   */
  __device__ uint64_t read_counter(const IbgdaLocalBuffer& counterBuf) const {
    return load_acquire_system_u64(counterBuf.ptr);
  }

  // =========================================================================
  // Private: _impl methods + internal building blocks
  // =========================================================================

 private:
  struct IbgdaLane {
    uint32_t nic_id{0};
    uint32_t qp_index{0};
    uint32_t lane_ordinal{0};
    uint32_t block_id{0};
    doca_gpu_dev_verbs_qp* qp{nullptr};
    doca_gpu_dev_verbs_qp* companion_qp{nullptr};
  };

  struct IbgdaPutSignalTickets {
    uint64_t put_wqe{0};
    uint64_t signal_wqe{0};
  };

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

  __device__ __forceinline__ void validate_block_id(uint32_t blockId) const {
    if (blockIdx.y != 0 || blockIdx.z != 0 || blockDim.y != 1 ||
        blockDim.z != 1) {
      printf(
          "[PIPES] FATAL: IBGDA per-block QP selection currently supports "
          "only 1D grids and 1D thread blocks, got blockIdx=(%u,%u,%u) "
          "blockDim=(%u,%u,%u)\n",
          blockIdx.x,
          blockIdx.y,
          blockIdx.z,
          blockDim.x,
          blockDim.y,
          blockDim.z);
      PIPES_DEVICE_TRAP();
    }
    if (blockId >= static_cast<uint32_t>(maxGroups_) || blockQpState_.empty()) {
      printf(
          "[PIPES] FATAL: IBGDA block_id=%u out of range [0, %d) "
          "or block QP state missing\n",
          blockId,
          maxGroups_);
      PIPES_DEVICE_TRAP();
    }
  }

  __device__ __forceinline__ void validate_group_scope(
      const ThreadGroup& group) const {
    if (group.scope == SyncScope::CLUSTER) {
      printf(
          "[PIPES] FATAL: IBGDA per-block QP selection does not support "
          "cluster-scope ThreadGroup yet\n");
      PIPES_DEVICE_TRAP();
    }
    validate_block_id(group.block_id);
  }

  __device__ __forceinline__ IbgdaLane
  lane_from_ordinal(uint32_t blockId, uint32_t laneOrdinal) const {
    validate_block_id(blockId);
    const uint32_t numNics = static_cast<uint32_t>(nicDevices_.size());
    if (numNics == 0) {
      printf(
          "P2pIbgdaTransportDevice: transport not initialized "
          "(peer not materialized? call get_device_handle(peers) first) "
          "at %s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",
          __FILE__,
          __LINE__,
          blockIdx.x,
          blockIdx.y,
          blockIdx.z,
          threadIdx.x,
          threadIdx.y,
          threadIdx.z);
      PIPES_DEVICE_TRAP();
    }
    const uint32_t nicId = laneOrdinal % numNics;
    const uint32_t qpIndex = laneOrdinal / numNics;
    const NicDeviceIbgdaResources& nic = nicDevices_[nicId];
    const uint32_t qpId = blockId * qpsPerBlockPerNic_ + qpIndex;
    if (qpIndex >= static_cast<uint32_t>(qpsPerBlockPerNic_) ||
        qpId >= nic.qps.size() || blockId >= nic.companion_qps.size()) {
      printf(
          "[PIPES] FATAL: invalid IBGDA lane block=%u nic=%u qpIndex=%u "
          "qpsPerBlockPerNic=%d qps=%u companionQps=%u\n",
          blockId,
          nicId,
          qpIndex,
          qpsPerBlockPerNic_,
          static_cast<unsigned>(nic.qps.size()),
          static_cast<unsigned>(nic.companion_qps.size()));
      PIPES_DEVICE_TRAP();
    }
    return IbgdaLane{
        .nic_id = nicId,
        .qp_index = qpIndex,
        .lane_ordinal = laneOrdinal,
        .block_id = blockId,
        .qp = nic.qps[qpId],
        .companion_qp = nic.companion_qps[blockId]};
  }

  __device__ __forceinline__ uint32_t
  select_put_lane_ordinal(uint32_t blockId) {
    validate_block_id(blockId);
    if (nicDevices_.empty()) {
      printf(
          "P2pIbgdaTransportDevice: transport not initialized "
          "(peer not materialized? call get_device_handle(peers) first) "
          "at %s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",
          __FILE__,
          __LINE__,
          blockIdx.x,
          blockIdx.y,
          blockIdx.z,
          threadIdx.x,
          threadIdx.y,
          threadIdx.z);
      PIPES_DEVICE_TRAP();
    }
    const uint32_t numLanes =
        static_cast<uint32_t>(nicDevices_.size() * qpsPerBlockPerNic_);
    if (numLanes == 1) {
      return 0;
    }
    uint32_t seq = atomicAdd(&blockQpState_[blockId].put_rr, 1U);
    return seq % numLanes;
  }

  __device__ __forceinline__ IbgdaLane select_put_lane(uint32_t blockId) {
    return lane_from_ordinal(blockId, select_put_lane_ordinal(blockId));
  }

  __device__ __forceinline__ IbgdaLane control_lane(uint32_t blockId) const {
    return lane_from_ordinal(blockId, 0);
  }

  __device__ __forceinline__ uint32_t num_qp_lanes() const {
    return static_cast<uint32_t>(nicDevices_.size() * qpsPerBlockPerNic_);
  }

  __device__ __forceinline__ void atomic_max_u64(uint64_t* ptr, uint64_t val) {
    atomicMax(
        reinterpret_cast<unsigned long long*>(ptr),
        static_cast<unsigned long long>(val));
  }

  __device__ __forceinline__ void atomic_or_u64(uint64_t* ptr, uint64_t val) {
    atomicOr(
        reinterpret_cast<unsigned long long*>(ptr),
        static_cast<unsigned long long>(val));
  }

  __device__ __forceinline__ uint64_t
  atomic_exchange_u64(uint64_t* ptr, uint64_t val) {
    return atomicExch(
        reinterpret_cast<unsigned long long*>(ptr),
        static_cast<unsigned long long>(val));
  }

  __device__ __forceinline__ void record_put_wqe(
      const IbgdaLane& lane,
      uint64_t ticket) {
    record_flush_wqe(lane, ticket);
  }

  __device__ __forceinline__ void record_flush_wqe(
      const IbgdaLane& lane,
      uint64_t ticket) {
    auto& state = blockQpState_[lane.block_id];
    atomic_max_u64(&state.last_flush_wqe[lane.lane_ordinal], ticket);
    atomic_or_u64(&state.pending_flush_lanes_mask, 1ULL << lane.lane_ordinal);
  }

  __device__ __forceinline__ void record_signal_wqe(
      const IbgdaLane& lane,
      uint64_t ticket) {
    record_flush_wqe(lane, ticket);
  }

  __device__ void wait_local_on_qp(
      doca_gpu_dev_verbs_qp* qp,
      doca_gpu_dev_verbs_ticket_t ticket,
      Timeout timeout = Timeout()) {
    if (!timeout.isEnabled()) {
      doca_gpu_dev_verbs_wait<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp, ticket);
    } else {
      int status;
      do {
        status = doca_gpu_dev_verbs_poll_one_cq_at<
            DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            doca_gpu_dev_verbs_qp_get_cq_sq(qp), ticket);
        if (status == EBUSY) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "wait_local_on_qp timed out (ticket=%llu)",
              static_cast<unsigned long long>(ticket));
        }
      } while (status == EBUSY);
    }
  }

  __device__ void
  wait_lanes(uint32_t blockId, uint64_t mask, const uint64_t* tickets) {
    const uint32_t numLanes =
        static_cast<uint32_t>(nicDevices_.size() * qpsPerBlockPerNic_);
    for (uint32_t laneId = 0; laneId < numLanes; ++laneId) {
      if ((mask & (1ULL << laneId)) == 0) {
        continue;
      }
      IbgdaLane lane = lane_from_ordinal(blockId, laneId);
      wait_local_on_qp(lane.qp, tickets[laneId]);
    }
  }

  __device__ void drain_flush_lanes(uint32_t blockId) {
    auto& state = blockQpState_[blockId];
    const uint64_t mask =
        atomic_exchange_u64(&state.pending_flush_lanes_mask, 0);
    wait_lanes(blockId, mask, state.last_flush_wqe);
  }

  __device__ void put_impl(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal,
      const IbgdaLocalBuffer& counterBuf,
      uint64_t counterVal) {
    const bool hasSignal = signalBuf.ptr != nullptr;
    if (nbytes == 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: zero-byte IBGDA put is unsupported. "
            "Use signal() for no-data signaling.\n");
        PIPES_DEVICE_TRAP();
      }
      group.sync();
      return;
    }

    if (group.is_leader()) {
      validate_group_scope(group);
      const bool hasCounter = counterBuf.ptr != nullptr;
      IbgdaLane lane = select_put_lane(group.block_id);
      if (hasSignal && hasCounter) {
        const auto tickets = put_signal_counter_single_impl(
            lane,
            localBuf,
            remoteBuf,
            nbytes,
            signalBuf,
            signalVal,
            counterBuf,
            counterVal);
        record_signal_wqe(lane, tickets.signal_wqe);
      } else if (hasSignal) {
        const auto tickets = put_signal_single_impl(
            lane, localBuf, remoteBuf, nbytes, signalBuf, signalVal);
        record_signal_wqe(lane, tickets.signal_wqe);
      } else if (hasCounter) {
        const uint64_t putTicket = put_counter_single_impl(
            lane, localBuf, remoteBuf, nbytes, counterBuf, counterVal);
        record_put_wqe(lane, putTicket);
      } else {
        const uint64_t putTicket =
            put_single_impl(lane, localBuf, remoteBuf, nbytes);
        record_put_wqe(lane, putTicket);
      }
    }
    group.sync();
  }

  __device__ void put_cooperative_impl(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal,
      const IbgdaLocalBuffer& counterBuf,
      uint64_t counterVal) {
    const bool hasSignal = signalBuf.ptr != nullptr;
    if (nbytes == 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: zero-byte IBGDA put_cooperative is unsupported. "
            "Use signal() for no-data signaling.\n");
        PIPES_DEVICE_TRAP();
      }
      group.sync();
      return;
    }

    uint32_t laneOrdinal = 0;
    uint64_t lastPutWqeIdx = 0;
    if (group.is_leader()) {
      validate_group_scope(group);
      laneOrdinal = select_put_lane_ordinal(group.block_id);
    }
    laneOrdinal = group.broadcast<uint32_t>(laneOrdinal);
    IbgdaLane lane = lane_from_ordinal(group.block_id, laneOrdinal);

    lastPutWqeIdx =
        put_cooperative_data_impl(group, lane, localBuf, remoteBuf, nbytes);
    if (group.is_leader()) {
      record_put_wqe(lane, lastPutWqeIdx);
      const bool hasCounter = counterBuf.ptr != nullptr;
      if (hasSignal) {
        const uint64_t signalTicket = signal_fenced(lane, signalBuf, signalVal);
        record_signal_wqe(lane, signalTicket);
      }
      if (hasCounter) {
        counter_after_wqe_impl(lane, lastPutWqeIdx, counterBuf, counterVal);
      }
    }
    group.sync();
  }

  // --- wait_signal_impl ---
  //
  // Signal waits use system-scope acquire loads. This matches NCCLX GIN's
  // waitSignal path and avoids the heavier post-poll __threadfence_system().

  __device__ void wait_signal_impl(
      ThreadGroup& group,
      const IbgdaLocalBuffer& signalBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    if (group.is_leader()) {
      uint64_t current = load_acquire_system_u64(signalBuf.ptr);
      while (current < expected) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "wait_signal: expected>=%llu, current=%llu",
            static_cast<unsigned long long>(expected),
            static_cast<unsigned long long>(current));
        current = load_acquire_system_u64(signalBuf.ptr);
      }
    }
    group.sync();
  }

  // --- wait_counter_impl ---

  __device__ void wait_counter_impl(
      ThreadGroup& group,
      const IbgdaLocalBuffer& counterBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    if (group.is_leader()) {
      uint64_t current = load_acquire_system_u64(counterBuf.ptr);
      while (current < expected) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "wait_counter: expected>=%llu, current=%llu",
            static_cast<unsigned long long>(expected),
            static_cast<unsigned long long>(current));
        current = load_acquire_system_u64(counterBuf.ptr);
      }
    }
    group.sync();
  }

  // --- reset_local_impl: zero a local 64-bit slot ---
  //
  // The volatile store + group.sync() is sufficient for intra-group ordering.
  // __threadfence() (device scope) is a cheap belt-and-suspenders so that
  // threads in OTHER blocks observing the slot through the read/wait APIs see
  // the reset. We deliberately do NOT use __threadfence_system() here: nothing
  // off-device reads this slot — the NIC only writes it via remote signals,
  // and the host doesn't read it on the hot path.

  __device__ void reset_local_impl(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf) {
    if (group.is_leader()) {
      volatile uint64_t* slot = static_cast<volatile uint64_t*>(localBuf.ptr);
      *slot = 0;
      __threadfence();
    }
    group.sync();
  }

  // =========================================================================
  // Raw building blocks (single-thread, no gating, no sync)
  // =========================================================================

  // --- put_cooperative_data_impl: group-collaborative WQE construction ---

  __device__ uint64_t put_cooperative_data_impl(
      ThreadGroup& group,
      const IbgdaLane& lane,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes) {
    std::size_t chunkSize = nbytes / group.group_size;
    std::size_t offset = group.thread_id_in_group * chunkSize;
    std::size_t laneBytes = (group.thread_id_in_group == group.group_size - 1)
        ? (nbytes - offset)
        : chunkSize;

    IbgdaLocalBuffer laneBuf = localBuf.subBuffer(offset);
    IbgdaRemoteBuffer laneRemoteBuf = remoteBuf.subBuffer(offset);

    auto* qp = lane.qp;

    // Guard: group_size must fit within QP send queue depth
    if (group.is_leader()) {
      const uint16_t qp_depth = __ldg(&qp->sq_wqe_num);
      if (group.group_size > qp_depth) {
        printf(
            "[PIPES] FATAL: put group_size (%u) > QP depth (%u). "
            "Set NCCL_CTRAN_IBGDA_QP_DEPTH >= %u to avoid deadlock.\n",
            group.group_size,
            qp_depth,
            group.group_size);
        PIPES_DEVICE_TRAP();
      }
    }

    // Leader reserves WQE slots for all threads
    uint64_t base_wqe_idx = 0;
    if (group.is_leader()) {
      base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, group.group_size);
    }
    base_wqe_idx = group.broadcast<uint64_t>(base_wqe_idx);

    // Each thread prepares its WQE
    uint64_t wqe_idx = base_wqe_idx + group.thread_id_in_group;
    doca_gpu_dev_verbs_wqe* wqe_ptr =
        doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

    if (laneBytes == 0) {
      doca_gpu_dev_verbs_wqe_prepare_nop(
          qp,
          wqe_ptr,
          static_cast<uint16_t>(wqe_idx),
          DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
    } else {
      doca_gpu_dev_verbs_wqe_prepare_write(
          qp,
          wqe_ptr,
          static_cast<uint16_t>(wqe_idx),
          DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
          DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
          0,
          reinterpret_cast<uint64_t>(laneRemoteBuf.ptr),
          laneRemoteBuf.rkey_per_device[lane.nic_id].value,
          reinterpret_cast<uint64_t>(laneBuf.ptr),
          laneBuf.lkey_per_device[lane.nic_id].value,
          static_cast<uint32_t>(laneBytes));
    }

    group.sync();

    // Leader marks ready and rings doorbell
    if (group.is_leader()) {
      doca_gpu_dev_verbs_mark_wqes_ready<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
          qp, base_wqe_idx, base_wqe_idx + group.group_size - 1);
      doca_gpu_dev_verbs_submit<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
          DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
          qp, base_wqe_idx + group.group_size);
    }

    group.sync();

    return base_wqe_idx + group.group_size - 1;
  }

  // --- put_single_impl: one thread, one WQE ---

  __device__ uint64_t put_single_impl(
      const IbgdaLane& lane,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes) {
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_addr localAddr = {
        .addr = reinterpret_cast<uint64_t>(localBuf.ptr),
        .key = localBuf.lkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(remoteBuf.ptr),
        .key = remoteBuf.rkey_per_device[lane.nic_id].value};

    doca_gpu_dev_verbs_put<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
        DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>(
        lane.qp, remoteAddr, localAddr, nbytes, &ticket);
    return ticket;
  }

  __device__ IbgdaPutSignalTickets put_signal_single_impl(
      const IbgdaLane& lane,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal) {
    const NicDeviceIbgdaResources& nic = nicDevices_[lane.nic_id];
    doca_gpu_dev_verbs_addr localAddr = {
        .addr = reinterpret_cast<uint64_t>(localBuf.ptr),
        .key = localBuf.lkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(remoteBuf.ptr),
        .key = remoteBuf.rkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr sigRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(signalBuf.ptr),
        .key = signalBuf.rkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr sigSinkAddr = {
        .addr = 0, .key = nic.sink_lkey.value};

#ifdef __HIP_PLATFORM_AMD__
    pipes_gda::ActiveNicBackend amdNic{};
    uint64_t ticket = 0;
    pipes_gda::pipes_gda_gpu_dev_verbs_put_signal(
        amdNic,
        lane.qp,
        remoteAddr,
        localAddr,
        nbytes,
        sigRemoteAddr,
        sigSinkAddr,
        signalVal,
        &ticket);
    return IbgdaPutSignalTickets{ticket, ticket};
#else
    uint64_t numChunks = doca_gpu_dev_verbs_div_ceil_aligned_pow2(
        nbytes, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
    numChunks = numChunks > 1 ? numChunks : 1;
    uint64_t baseWqeIdx = doca_gpu_dev_verbs_reserve_wq_slots<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(lane.qp, numChunks + 1);
    uint64_t wqeIdx = baseWqeIdx;
    std::size_t remainingSize = nbytes;

#pragma unroll 1
    for (uint64_t i = 0; i < numChunks; ++i) {
      wqeIdx = baseWqeIdx + i;
      const std::size_t chunkSize =
          remainingSize > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
          ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
          : remainingSize;
      doca_gpu_dev_verbs_wqe* wqePtr =
          doca_gpu_dev_verbs_get_wqe_ptr(lane.qp, wqeIdx);
      doca_gpu_dev_verbs_wqe_prepare_write(
          lane.qp,
          wqePtr,
          wqeIdx,
          DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
          DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
          0,
          remoteAddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE),
          remoteAddr.key,
          localAddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE),
          localAddr.key,
          chunkSize);
      remainingSize -= chunkSize;
    }
    const uint64_t lastPutWqeIdx = wqeIdx;

    ++wqeIdx;
    doca_gpu_dev_verbs_wqe* wqePtr =
        doca_gpu_dev_verbs_get_wqe_ptr(lane.qp, wqeIdx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        lane.qp,
        wqePtr,
        wqeIdx,
        DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        sigRemoteAddr.addr,
        sigRemoteAddr.key,
        sigSinkAddr.addr,
        sigSinkAddr.key,
        sizeof(uint64_t),
        signalVal,
        0);
    doca_gpu_dev_verbs_mark_wqes_ready<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        lane.qp, baseWqeIdx, wqeIdx);
    doca_gpu_dev_verbs_submit<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(lane.qp, wqeIdx + 1);
    return IbgdaPutSignalTickets{lastPutWqeIdx, wqeIdx};
#endif
  }

  __device__ uint64_t put_counter_single_impl(
      const IbgdaLane& lane,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaLocalBuffer& counterBuf,
      uint64_t counterVal) {
    const NicDeviceIbgdaResources& nic = nicDevices_[lane.nic_id];
    doca_gpu_dev_verbs_addr localAddr = {
        .addr = reinterpret_cast<uint64_t>(localBuf.ptr),
        .key = localBuf.lkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(remoteBuf.ptr),
        .key = remoteBuf.rkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr counterRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(counterBuf.ptr),
        .key = counterBuf.lkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr counterSinkAddr = {
        .addr = 0, .key = nic.sink_lkey.value};

#ifdef __HIP_PLATFORM_AMD__
    // AMD: route data + local counter through put_signal_counter with the
    // signal disabled (sigRemoteAddr.addr == 0 short-circuits the atomic-FA
    // WQE in pipes_gda_gpu_dev_verbs_put_signal_counter).
    doca_gpu_dev_verbs_addr noSigRemoteAddr = {.addr = 0, .key = 0};
    doca_gpu_dev_verbs_addr noSigSinkAddr = {
        .addr = 0, .key = nic.sink_lkey.value};
    pipes_gda::ActiveNicBackend amdNic{};
    pipes_gda::pipes_gda_gpu_dev_verbs_put_signal_counter(
        amdNic,
        lane.qp,
        remoteAddr,
        localAddr,
        nbytes,
        noSigRemoteAddr,
        noSigSinkAddr,
        0,
        lane.companion_qp,
        counterRemoteAddr,
        counterSinkAddr,
        counterVal);
    return 0;
#else
    constexpr unsigned int kNumQps = 2;
    doca_gpu_dev_verbs_qp* qp = lane.qp;
    doca_gpu_dev_verbs_qp* companionQp = lane.companion_qp;

    uint64_t numChunks = doca_gpu_dev_verbs_div_ceil_aligned_pow2(
        nbytes, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
    numChunks = numChunks > 1 ? numChunks : 1;
    uint64_t baseWqeIdx = doca_gpu_dev_verbs_reserve_wq_slots<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, numChunks);
    uint64_t wqeIdx = baseWqeIdx;
    std::size_t remainingSize = nbytes;

#pragma unroll 1
    for (uint64_t i = 0; i < numChunks; ++i) {
      wqeIdx = baseWqeIdx + i;
      const std::size_t chunkSize =
          remainingSize > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
          ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
          : remainingSize;
      doca_gpu_dev_verbs_wqe* wqePtr =
          doca_gpu_dev_verbs_get_wqe_ptr(qp, wqeIdx);
      doca_gpu_dev_verbs_wqe_prepare_write(
          qp,
          wqePtr,
          wqeIdx,
          DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
          DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
          0,
          remoteAddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE),
          remoteAddr.key,
          localAddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE),
          localAddr.key,
          chunkSize);
      remainingSize -= chunkSize;
    }
    const uint64_t lastPutWqeIdx = wqeIdx;

    doca_gpu_dev_verbs_mark_wqes_ready<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        qp, baseWqeIdx, lastPutWqeIdx);

    uint64_t companionBaseWqeIdx = doca_gpu_dev_verbs_reserve_wq_slots<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(companionQp, 2);
    uint64_t companionWqeIdx = companionBaseWqeIdx;
    doca_gpu_dev_verbs_wqe* wqePtr =
        doca_gpu_dev_verbs_get_wqe_ptr(companionQp, companionWqeIdx);
    doca_gpu_dev_verbs_wqe_prepare_wait(
        companionQp,
        wqePtr,
        companionWqeIdx,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        lastPutWqeIdx,
        qp->cq_sq.cq_num);

    ++companionWqeIdx;
    wqePtr = doca_gpu_dev_verbs_get_wqe_ptr(companionQp, companionWqeIdx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companionQp,
        wqePtr,
        companionWqeIdx,
        DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        counterRemoteAddr.addr,
        counterRemoteAddr.key,
        counterSinkAddr.addr,
        counterSinkAddr.key,
        sizeof(uint64_t),
        counterVal,
        0);
    doca_gpu_dev_verbs_mark_wqes_ready<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        companionQp, companionBaseWqeIdx, companionWqeIdx);

    doca_gpu_dev_verbs_qp* qps[kNumQps] = {qp, companionQp};
    uint64_t prodIndices[kNumQps] = {lastPutWqeIdx + 1, companionWqeIdx + 1};
    doca_gpu_dev_verbs_submit_multi_qps<
        kNumQps,
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qps, prodIndices);
    return lastPutWqeIdx;
#endif
  }

  __device__ IbgdaPutSignalTickets put_signal_counter_single_impl(
      const IbgdaLane& lane,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal,
      const IbgdaLocalBuffer& counterBuf,
      uint64_t counterVal) {
#ifdef __HIP_PLATFORM_AMD__
    put_counter_single_impl(
        lane, localBuf, remoteBuf, nbytes, counterBuf, counterVal);
    const uint64_t signalTicket = signal_fenced(lane, signalBuf, signalVal);
    return IbgdaPutSignalTickets{signalTicket, signalTicket};
#else
    constexpr unsigned int kNumQps = 2;
    const NicDeviceIbgdaResources& nic = nicDevices_[lane.nic_id];
    doca_gpu_dev_verbs_qp* qp = lane.qp;
    doca_gpu_dev_verbs_qp* companionQp = lane.companion_qp;

    doca_gpu_dev_verbs_addr localAddr = {
        .addr = reinterpret_cast<uint64_t>(localBuf.ptr),
        .key = localBuf.lkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(remoteBuf.ptr),
        .key = remoteBuf.rkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr sigRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(signalBuf.ptr),
        .key = signalBuf.rkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr sigSinkAddr = {
        .addr = 0, .key = nic.sink_lkey.value};
    doca_gpu_dev_verbs_addr counterRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(counterBuf.ptr),
        .key = counterBuf.lkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr counterSinkAddr = {
        .addr = 0, .key = nic.sink_lkey.value};

    uint64_t numChunks = doca_gpu_dev_verbs_div_ceil_aligned_pow2(
        nbytes, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
    numChunks = numChunks > 1 ? numChunks : 1;
    uint64_t baseWqeIdx = doca_gpu_dev_verbs_reserve_wq_slots<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, numChunks + 1);
    uint64_t wqeIdx = baseWqeIdx;
    std::size_t remainingSize = nbytes;

#pragma unroll 1
    for (uint64_t i = 0; i < numChunks; ++i) {
      wqeIdx = baseWqeIdx + i;
      const std::size_t chunkSize =
          remainingSize > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
          ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
          : remainingSize;
      doca_gpu_dev_verbs_wqe* wqePtr =
          doca_gpu_dev_verbs_get_wqe_ptr(qp, wqeIdx);
      doca_gpu_dev_verbs_wqe_prepare_write(
          qp,
          wqePtr,
          wqeIdx,
          DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
          DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
          0,
          remoteAddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE),
          remoteAddr.key,
          localAddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE),
          localAddr.key,
          chunkSize);
      remainingSize -= chunkSize;
    }
    const uint64_t lastPutWqeIdx = wqeIdx;

    ++wqeIdx;
    doca_gpu_dev_verbs_wqe* wqePtr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqeIdx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        qp,
        wqePtr,
        wqeIdx,
        DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        sigRemoteAddr.addr,
        sigRemoteAddr.key,
        sigSinkAddr.addr,
        sigSinkAddr.key,
        sizeof(uint64_t),
        signalVal,
        0);
    const uint64_t signalWqeIdx = wqeIdx;
    doca_gpu_dev_verbs_mark_wqes_ready<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        qp, baseWqeIdx, signalWqeIdx);

    uint64_t companionBaseWqeIdx = doca_gpu_dev_verbs_reserve_wq_slots<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(companionQp, 2);
    uint64_t companionWqeIdx = companionBaseWqeIdx;
    wqePtr = doca_gpu_dev_verbs_get_wqe_ptr(companionQp, companionWqeIdx);
    doca_gpu_dev_verbs_wqe_prepare_wait(
        companionQp,
        wqePtr,
        companionWqeIdx,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        lastPutWqeIdx,
        qp->cq_sq.cq_num);

    ++companionWqeIdx;
    wqePtr = doca_gpu_dev_verbs_get_wqe_ptr(companionQp, companionWqeIdx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companionQp,
        wqePtr,
        companionWqeIdx,
        DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        counterRemoteAddr.addr,
        counterRemoteAddr.key,
        counterSinkAddr.addr,
        counterSinkAddr.key,
        sizeof(uint64_t),
        counterVal,
        0);
    doca_gpu_dev_verbs_mark_wqes_ready<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        companionQp, companionBaseWqeIdx, companionWqeIdx);

    doca_gpu_dev_verbs_qp* qps[kNumQps] = {qp, companionQp};
    uint64_t prodIndices[kNumQps] = {signalWqeIdx + 1, companionWqeIdx + 1};
    doca_gpu_dev_verbs_submit_multi_qps<
        kNumQps,
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qps, prodIndices);
    return IbgdaPutSignalTickets{lastPutWqeIdx, signalWqeIdx};
#endif
  }

  __device__ void counter_after_wqe_impl(
      const IbgdaLane& lane,
      uint64_t waitWqeIdx,
      const IbgdaLocalBuffer& counterBuf,
      uint64_t counterVal) {
    const NicDeviceIbgdaResources& nic = nicDevices_[lane.nic_id];
    doca_gpu_dev_verbs_qp* qp = lane.qp;
    doca_gpu_dev_verbs_qp* companionQp = lane.companion_qp;

    doca_gpu_dev_verbs_addr counterRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(counterBuf.ptr),
        .key = counterBuf.lkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr counterSinkAddr = {
        .addr = 0, .key = nic.sink_lkey.value};

#ifdef __HIP_PLATFORM_AMD__
    // AMD: pipes_gda doesn't expose inter-QP wait WQEs at the public API,
    // and signal_counter waits on mainQp's last reserved WQE — which is
    // exactly the most recent put issued before this call. Routing through
    // signal_counter with sig disabled (sigRemoteAddr.addr == 0) gives us
    // the same "wait on previous put + atomic counter" semantics.
    (void)waitWqeIdx;
    (void)qp;
    doca_gpu_dev_verbs_addr noSigRemoteAddr = {.addr = 0, .key = 0};
    doca_gpu_dev_verbs_addr noSigSinkAddr = {
        .addr = 0, .key = nic.sink_lkey.value};
    pipes_gda::ActiveNicBackend amdNic{};
    pipes_gda::pipes_gda_gpu_dev_verbs_signal_counter(
        amdNic,
        lane.qp,
        noSigRemoteAddr,
        noSigSinkAddr,
        0,
        companionQp,
        counterRemoteAddr,
        counterSinkAddr,
        counterVal);
#else
    uint64_t baseWqeIdx = doca_gpu_dev_verbs_reserve_wq_slots<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(companionQp, 2);
    uint64_t wqeIdx = baseWqeIdx;
    doca_gpu_dev_verbs_wqe* wqePtr =
        doca_gpu_dev_verbs_get_wqe_ptr(companionQp, wqeIdx);
    doca_gpu_dev_verbs_wqe_prepare_wait(
        companionQp,
        wqePtr,
        wqeIdx,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        waitWqeIdx,
        qp->cq_sq.cq_num);

    ++wqeIdx;
    wqePtr = doca_gpu_dev_verbs_get_wqe_ptr(companionQp, wqeIdx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companionQp,
        wqePtr,
        wqeIdx,
        DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        counterRemoteAddr.addr,
        counterRemoteAddr.key,
        counterSinkAddr.addr,
        counterSinkAddr.key,
        sizeof(uint64_t),
        counterVal,
        0);
    doca_gpu_dev_verbs_mark_wqes_ready<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        companionQp, baseWqeIdx, wqeIdx);
    doca_gpu_dev_verbs_submit<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(companionQp, wqeIdx + 1);
#endif
  }

  // --- signal_fenced: atomic fetch-add with NIC FENCE (always fenced) ---

  __device__ uint64_t signal_fenced(
      const IbgdaLane& lane,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal) {
    const NicDeviceIbgdaResources& nic = nicDevices_[lane.nic_id];
    doca_gpu_dev_verbs_qp* qp = lane.qp;
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(signalBuf.ptr),
        .key = signalBuf.rkey_per_device[lane.nic_id].value};
    doca_gpu_dev_verbs_addr sinkAddr = {.addr = 0, .key = nic.sink_lkey.value};

    uint64_t wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, 1);

    doca_gpu_dev_verbs_wqe* wqe_ptr =
        doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

    doca_gpu_dev_verbs_wqe_prepare_atomic(
        qp,
        wqe_ptr,
        static_cast<uint16_t>(wqe_idx),
        DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        static_cast<doca_gpu_dev_verbs_wqe_ctrl_flags>(
            DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE |
            DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FENCE),
        remoteAddr.addr,
        remoteAddr.key,
        sinkAddr.addr,
        sinkAddr.key,
        sizeof(uint64_t),
        signalVal,
        0);

    doca_gpu_dev_verbs_mark_wqes_ready<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, wqe_idx, wqe_idx);

    doca_gpu_dev_verbs_submit<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp, wqe_idx + 1);
    return wqe_idx;
  }

  // --- Slot resolution helpers ---
  //
  // Centralize bounds-check + pointer arithmetic for the slot-index API.
  // Every slot-index method goes through one of these so the bounds check
  // and the slot pointer can never drift apart.

  __device__ IbgdaRemoteBuffer remote_signal_slot(int id) const {
    IBGDA_CHECK_SLOT_ID(id, numSignalSlots_, "signal");
    return IbgdaRemoteBuffer(
        static_cast<uint64_t*>(ownedRemoteSignalBuf_.ptr) + id,
        ownedRemoteSignalBuf_.rkey_per_device);
  }

  __device__ IbgdaLocalBuffer local_signal_slot(int id) const {
    IBGDA_CHECK_SLOT_ID(id, numSignalSlots_, "signal");
    return IbgdaLocalBuffer(
        static_cast<uint64_t*>(ownedLocalSignalBuf_.ptr) + id,
        ownedLocalSignalBuf_.lkey_per_device);
  }

  __device__ IbgdaLocalBuffer counter_slot(int id) const {
    IBGDA_CHECK_SLOT_ID(id, numCounterSlots_, "counter");
    return IbgdaLocalBuffer(
        static_cast<uint64_t*>(ownedCounterBuf_.ptr) + id,
        ownedCounterBuf_.lkey_per_device);
  }

 public:
  // ===========================================================================
  // Pipelined Send/Recv (using transport-managed staging buffers)
  // ===========================================================================
  //
  // Public composable primitives for pipelined RDMA data transfer. Each block
  // owns one tile (a partition of the user's data). The transport manages
  // staging buffers internally — the user only provides src/dst pointers.
  //
  // Data flow:
  //
  //   SENDER (GPU A)                              RECEIVER (GPU B)
  //   ┌──────────┐                                ┌──────────┐
  //   │ user src │                                │ user dst │
  //   └────┬─────┘                                └────▲─────┘
  //        │ memcpy                                    │ memcpy
  //        ▼                                           │
  //   ┌────────────┐       RDMA put              ┌─────┴──────┐
  //   │sendStaging │ ─────────────────────────▶  │recvStaging │
  //   │  (GPU A)   │  + DATA_READY signal        │  (GPU B)   │
  //   └────────────┘  + NIC_DONE counter         └────────────┘
  //        ▲                                           │
  //        └───────────── SLOT_FREE signal ────────────┘
  //
  // Signal protocol (per block, 3 primitives):
  //   DATA_READY  — piggybacked on put (sender → receiver's signalBuf)
  //   SLOT_FREE   — explicit signal    (receiver → sender's signalBuf)
  //   NIC_DONE    — loopback counter   (NIC → sender's counterBuf)
  //
  // Terminology used below:
  //   slot             = one logical staging-ring entry of dataBufferSize
  //                      bytes. There are pipelineDepth slots in the ring.
  //   active_blocks    = number of participating block-groups in one
  //                      send()/recv() call. Must be <= maxGroups.
  //   perBlockSlot     = one block-group's partition within a slot:
  //                      (dataBufferSize / active_blocks) & ~15ULL
  //   sub-chunk        = one signaled byte range within a perBlockSlot. When
  //                      max_signal_bytes == 0, a sub-chunk is the whole
  //                      perBlockSlot. Otherwise:
  //                        chunkSize = floor16(min(perBlockSlot,
  //                                             max_signal_bytes))
  //   state[].nextStep = persistent 16-byte-aligned protocol cursor.
  //                      DATA_READY, SLOT_FREE, and NIC_DONE counters also
  //                      advance by protocol bytes, which keeps cursor state
  //                      independent of max_signal_bytes.
  //
  // Typical usage:
  //   auto [role, sub] = group.partition(2);
  //   std::size_t sectionBytes = transport->send_recv_state().dataBufferSize;
  //   for (std::size_t s = 0; s < totalBytes / sectionBytes; ++s) {
  //     TiledBuffer<char> tiles(data + s * sectionBytes, sectionBytes, sub);
  //     if (role == 0)
  //       transport->send(sub, tiles.data(), tiles.bytes(), active_blocks);
  //     else
  //       transport->recv(sub, tiles.data(), tiles.bytes(), active_blocks);
  //   }

  /**
   * Resumable send/recv compatibility with blocking send()/recv().
   *
   * The progress API deliberately uses the same wire protocol as blocking
   * send()/recv(). For a chunk with logical byte range [streamStart,
   * streamEnd), both APIs derive the same ring slot and staging offset from:
   *
   *   pipelineBytes = perBlockSlot * pipelineDepth
   *   pipelineOff   = streamStart % pipelineBytes
   *   slot          = pipelineOff / perBlockSlot
   *   stagingOff    = slot * dataBufferSize + groupId * perBlockSlot +
   *                   (pipelineOff - slot * perBlockSlot)
   *
   * Sender chunk state machine:
   *
   *   WaitNicDone
   *        |
   *        | local sendStaging is reusable; copy user src -> sendStaging
   *        v
   *   WaitSlotFree
   *        |
   *        | remote recvStaging is reusable; RDMA put + DATA_READY + NIC_DONE
   *        v
   *   Done for this chunk, then either WaitNicDone for next chunk or Done
   *
   * Receiver chunk state machine:
   *
   *   WaitDataReady
   *        |
   *        | DATA_READY reached streamEnd; copy recvStaging -> user dst
   *        v
   *   Signal SLOT_FREE, then either WaitDataReady for next chunk or Done
   *
   * Compatibility follows from using the same cumulative byte counters:
   * DATA_READY advances by bytesThis/chunk.bytes on each put, SLOT_FREE
   * advances by the same amount after recv copies out of staging, and NIC_DONE
   * advances by the same amount after the NIC completes the sender's WQE.
   * Blocking send()/recv() and async init share one transport-owned byte
   * cursor. Blocking calls advance it when the call completes; progress init
   * reserves that cursor range before returning so later blocking calls cannot
   * reuse protocol bytes while an async operation is in flight.
   *
   * Usage starts by initializing the transport-owned mutable state for one
   * logical transfer, then calling the matching progress method with the same
   * static geometry until it returns `Done`:
   *
   *   transport->init_send_progress(
   *       group, nbytes, active_blocks, max_signal_bytes);
   *   while (transport->progress_send_once(
   *              group, src, nbytes, active_blocks, max_signal_bytes, timeout)
   *          != IbgdaSendRecvProgressStatus::Done) {
   *     // Try another independent lane or return to the scheduler.
   *   }
   *
   * Receivers use the symmetric `init_recv_progress()` and
   * `progress_recv_once()` pair with the same `nbytes`, `active_blocks`, and
   * compatible `max_signal_bytes`. Zero-byte operations initialize directly to
   * `Done`, so callers can use the same loop shape for empty and non-empty
   * transfers.
   *
   * The transport-owned state slot stores the shared persistent protocol byte
   * cursor and only the active async stage, `activeNextByte`, and reserved
   * stream base. Immutable geometry such as `nbytes`, active block count, and
   * chunk sizing is intentionally kept out of HBM-backed state and recomputed
   * by each progress call from its arguments.
   *
   * The progress state is a property of this transport, indexed by
   * `group.group_id` and direction. A caller may have one send and one recv in
   * flight for the same group, but a second send or second recv init for the
   * same group before the previous operation reaches `Done` traps on device.
   */

  /**
   * Initialize transport-owned state for one pipelined send operation.
   *
   * The transport reserves the sender-side protocol byte stream for
   * `group.group_id` and starts the internal state in the sender state machine
   * unless `nbytes == 0`. Non-empty payload byte counts are rounded up to the
   * 16-byte wire granularity internally. It does not capture the source
   * pointer; callers pass the pointer to each `progress_send_once()` call.
   *
   * The send progress slot for this group must be idle. Re-initializing a
   * group while a previous send is outstanding traps with a diagnostic instead
   * of silently overwriting the in-flight byte range.
   *
   * `active_blocks == 0` means all configured groups participate. Non-zero
   * values must match the peer's recv-side initialization for the same logical
   * transfer. `max_signal_bytes == 0` sends one signal per per-block staging
   * partition; smaller non-zero values split that partition into multiple
   * signaled sub-chunks for finer overlap with the receiver.
   *
   * Zero-byte sends mark the internal state `Done` without reading or
   * validating staging geometry. This matches the blocking `send()` no-op
   * behavior and lets schedulers treat empty operations uniformly.
   *
   * @param group Thread group that will execute all later progress calls.
   * @param nbytes Number of user-buffer bytes to send for this group.
   * @param active_blocks Number of participating transport lanes, or 0 for
   *                      maxGroups.
   * @param max_signal_bytes Maximum signaled sub-chunk size, or 0 for default.
   */
  __device__ __forceinline__ void init_send_progress(
      ThreadGroup& group,
      std::size_t nbytes,
      int active_blocks = 0,
      std::size_t max_signal_bytes = 0) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const int progressIndex = progress_send_index(group);
    auto& slot = progress_state_slot(group, progressIndex);
    assert_progress_slot_idle(group, slot, "send");
    IbSendRecvState::ProgressSlot state{};
    state.activeStage = nbytes == 0
        ? detail::IbSendRecvProgressStage::Done
        : detail::IbSendRecvProgressStage::WaitNicDone;
    if (nbytes == 0) {
      store_progress_state(group, progressIndex, state);
      return;
    }
    // Validate the transfer before reserving the transport byte cursor.
    const ProgressGeometry geometry = make_progress_geometry(
        group, nbytes, active_blocks, max_signal_bytes, "init_send_progress");
    state.activeBaseStep =
        reserve_progress_step(group, progressIndex, geometry.protocolBytes);
    store_progress_state(group, progressIndex, state);
#endif
  }

  /**
   * Initialize transport-owned state for one pipelined recv operation.
   *
   * The transport reserves the receiver-side protocol byte stream for
   * `group.group_id` and starts the internal state in the receiver state
   * machine unless `nbytes == 0`. Non-empty payload byte counts are rounded up
   * to the 16-byte wire granularity internally. It does not capture the
   * destination pointer; callers pass the pointer to each
   * `progress_recv_once()` call.
   *
   * The recv progress slot for this group must be idle. Re-initializing a
   * group while a previous recv is outstanding traps with a diagnostic instead
   * of silently overwriting the in-flight byte range.
   *
   * The sender and receiver must use the same `active_blocks` and compatible
   * `max_signal_bytes` for a logical transfer. The staging offset and protocol
   * signal values are derived from those values, so a mismatch can make one
   * side wait on a different byte range than the other side produced.
   *
   * Zero-byte receives mark the internal state `Done` without reading or
   * validating staging geometry. This matches the blocking `recv()` no-op
   * behavior and lets schedulers treat empty operations uniformly.
   *
   * @param group Thread group that will execute all later progress calls.
   * @param nbytes Number of user-buffer bytes to receive for this group.
   * @param active_blocks Number of participating transport lanes, or 0 for
   *                      maxGroups.
   * @param max_signal_bytes Maximum signaled sub-chunk size, or 0 for default.
   */
  __device__ __forceinline__ void init_recv_progress(
      ThreadGroup& group,
      std::size_t nbytes,
      int active_blocks = 0,
      std::size_t max_signal_bytes = 0) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const int progressIndex = progress_recv_index(group);
    auto& slot = progress_state_slot(group, progressIndex);
    assert_progress_slot_idle(group, slot, "recv");
    IbSendRecvState::ProgressSlot state{};
    state.activeStage = nbytes == 0
        ? detail::IbSendRecvProgressStage::Done
        : detail::IbSendRecvProgressStage::WaitDataReady;
    if (nbytes == 0) {
      store_progress_state(group, progressIndex, state);
      return;
    }
    // Validate the transfer before reserving the transport byte cursor.
    const ProgressGeometry geometry = make_progress_geometry(
        group, nbytes, active_blocks, max_signal_bytes, "init_recv_progress");
    state.activeBaseStep =
        reserve_progress_step(group, progressIndex, geometry.protocolBytes);
    store_progress_state(group, progressIndex, state);
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
   * range, and finally issues an RDMA put that piggybacks DATA_READY and
   * records NIC_DONE in the local counter. Returning `Done` means the reserved
   * protocol byte range has completed. For unaligned payload sizes, the final
   * WQE may include transport-private padding; `CopyOp` is invoked only for
   * valid payload bytes.
   *
   * `CopyOp` must expose `send(dst, src, bytes, group, dataOffset, args...)`.
   * The default `Memcpy` copies bytes cooperatively across the supplied
   * `ThreadGroup`; custom copy ops may use `args` to pass reduction or
   * conversion context.
   *
   * @param group Thread group matching the one used during initialization.
   * @param src Source user buffer. The range `[src, src + nbytes)` must remain
   *            valid until `Done`.
   * @param nbytes Number of user-buffer bytes from the matching init call.
   * @param active_blocks Number of participating groups from init.
   * @param max_signal_bytes Maximum signaled sub-chunk size from init.
   * @param timeout Optional device timeout checked while dependencies wait.
   * @param args Additional arguments forwarded to `CopyOp::send`.
   */
  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus progress_send_once(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      int active_blocks = 0,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef __HIP_PLATFORM_AMD__
    static_assert(
        sizeof(CopyOp) == 0,
        "P2pIbgdaTransportDevice::progress_send_once() requires NVIDIA GPU");
#endif
    const int progressIndex = progress_send_index(group);
    IbSendRecvState::ProgressSlot state =
        progress_state_slot(group, progressIndex);
    if (state.activeStage == detail::IbSendRecvProgressStage::Done) {
      return IbgdaSendRecvProgressStatus::Done;
    }
    const ProgressGeometry progress_params = make_progress_geometry(
        group, nbytes, active_blocks, max_signal_bytes, "progress_send_once");
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
              read_counter(sendRecvState_.localCounterBuf.subBuffer(
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
              read_signal(sendRecvState_.localSignalBuf.subBuffer(
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
      if (group.is_leader()) {
        ThreadGroup solo{
            0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
        put(solo,
            sendRecvState_.sendStagingBuf.subBuffer(chunk.stagingOff),
            sendRecvState_.recvStagingBuf.subBuffer(chunk.stagingOff),
            chunk.bytes,
            sendRecvState_.remoteSignalBuf.subBuffer(
                progress_params.groupId * sizeof(uint64_t)),
            chunk.bytes,
            sendRecvState_.localCounterBuf.subBuffer(
                progress_params.groupId * sizeof(uint64_t)),
            chunk.bytes);
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
   * `CopyOp::recv`, then signals SLOT_FREE back to the sender. Returning `Done`
   * means the reserved protocol byte range has completed. For unaligned
   * payload sizes, the final WQE may include transport-private padding;
   * `CopyOp` is invoked only for valid payload bytes.
   *
   * `CopyOp` must expose `recv(dst, src, bytes, group, dataOffset, args...)`.
   * The default `Memcpy` copies bytes cooperatively across the supplied
   * `ThreadGroup`; custom copy ops may use `args` to pass reduction or
   * conversion context.
   *
   * @param group Thread group matching the one used during initialization.
   * @param dst Destination user buffer. The range `[dst, dst + nbytes)` must
   *            remain valid until `Done`.
   * @param nbytes Number of user-buffer bytes from the matching init call.
   * @param active_blocks Number of participating groups from init.
   * @param max_signal_bytes Maximum signaled sub-chunk size from init.
   * @param timeout Optional device timeout checked while dependencies wait.
   * @param args Additional arguments forwarded to `CopyOp::recv`.
   */
  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus progress_recv_once(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      int active_blocks = 0,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef __HIP_PLATFORM_AMD__
    static_assert(
        sizeof(CopyOp) == 0,
        "P2pIbgdaTransportDevice::progress_recv_once() requires NVIDIA GPU");
#endif
    const int progressIndex = progress_recv_index(group);
    IbSendRecvState::ProgressSlot state =
        progress_state_slot(group, progressIndex);
    if (state.activeStage == detail::IbSendRecvProgressStage::Done) {
      return IbgdaSendRecvProgressStatus::Done;
    }
    const ProgressGeometry progress_params = make_progress_geometry(
        group, nbytes, active_blocks, max_signal_bytes, "progress_recv_once");
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
          read_signal(sendRecvState_.localSignalBuf.subBuffer(
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

    signal(
        group,
        sendRecvState_.remoteSignalBuf.subBuffer(
            (sendRecvState_.maxGroups + progress_params.groupId) *
            sizeof(uint64_t)),
        chunk.bytes);

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
    return IbgdaSendRecvProgressStatus::Done;
#endif
  }

  /**
   * send — send one block's tile via pipelined RDMA.
   *
   * Copies src → sendStaging, then RDMA puts sendStaging → peer's recvStaging.
   * For this call, each logical slot contributes one perBlockSlot-sized region
   * for this group. If nbytes > perBlockSlot, send() advances through multiple
   * ring positions. max_signal_bytes can further subdivide each perBlockSlot
   * into multiple signaled sub-chunks, enabling finer-grained overlap at the
   * receiver.
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
   * The caller must keep the staging layout stable while a sequence is in
   * flight. Changing active_blocks changes the per-block staging partition, so
   * both sides must perform a higher-level barrier/quiescence step first.
   * max_signal_bytes may vary across calls with the same active_blocks.
   *
   * @param group           ThreadGroup (all threads participate in memcpy,
   *                        leader does RDMA ops).
   * @param src             Source data for this block's tile.
   * @param nbytes          Payload bytes to send for this group. The internal
   *                        protocol byte count is rounded up to 16 bytes and
   *                        consumed in perBlockSlot-sized pieces, or smaller
   *                        sub-chunks when max_signal_bytes is set.
   * @param active_blocks   Number of block-groups sharing each logical slot in
   *                        this call. 0 means use maxGroups.
   * @param max_signal_bytes Max bytes per signaled sub-chunk within one
   *                        perBlockSlot. 0 means one signal per perBlockSlot.
   * @param timeout         Optional timeout for wait operations.
   */
  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void send(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      int active_blocks = 0,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
#if !PIPES_IS_DEVICE_COMPILE
    (void)group;
    (void)src;
    (void)nbytes;
    (void)active_blocks;
    (void)max_signal_bytes;
    (void)timeout;
#else
    sendWithTrace<CopyOp>(
        group,
        src,
        nbytes,
        active_blocks,
        max_signal_bytes,
        timeout,
        {},
        0,
        args...);
#endif
  }

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void sendWithTrace(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      int active_blocks,
      std::size_t max_signal_bytes,
      const Timeout& timeout,
      PipesTraceHandle trace,
      uint8_t self_rank,
      Args... args) {
#if !PIPES_IS_DEVICE_COMPILE
    (void)group;
    (void)src;
    (void)nbytes;
    (void)active_blocks;
    (void)max_signal_bytes;
    (void)timeout;
    (void)trace;
    (void)self_rank;
#else
    if (nbytes == 0) {
      return;
    }
    const std::size_t protocolBytes = align_protocol_bytes(nbytes);

    const int groupId = group.group_id;
    const int effActive =
        active_blocks > 0 ? active_blocks : sendRecvState_.maxGroups;

    if (effActive > sendRecvState_.maxGroups) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: send active_blocks=%d > maxGroups=%d\n",
            effActive,
            sendRecvState_.maxGroups);
      }
      PIPES_DEVICE_TRAP();
    }
    if (groupId >= effActive) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: send group_id=%u >= active_blocks=%d\n",
            groupId,
            effActive);
      }
      PIPES_DEVICE_TRAP();
    }

    const std::size_t perBlockSlot =
        (sendRecvState_.dataBufferSize / effActive) & ~15ULL;
    if (perBlockSlot == 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: send perBlockSlot=0 "
            "(dataBufferSize=%llu, active_blocks=%d)\n",
            (unsigned long long)sendRecvState_.dataBufferSize,
            effActive);
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
    const int maxGroups = sendRecvState_.maxGroups;
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

    if (group.is_leader()) {
      trace_ibgda_event(
          trace,
          self_rank,
          PipesTraceEventType::kIbSendBegin,
          /*step=*/0,
          static_cast<uint16_t>(groupId));
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
        wait_counter(
            group,
            sendRecvState_.localCounterBuf.subBuffer(
                groupId * sizeof(uint64_t)),
            streamEnd - pipelineBytes,
            timeout);
      }

      // (2) Cooperative copy: src → local sendStaging via CopyOp.
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
        wait_signal(
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
        ThreadGroup solo{
            0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
        put(solo,
            sendRecvState_.sendStagingBuf.subBuffer(stagingOff),
            sendRecvState_.recvStagingBuf.subBuffer(stagingOff),
            bytesThis,
            sendRecvState_.remoteSignalBuf.subBuffer(
                groupId * sizeof(uint64_t)),
            bytesThis,
            sendRecvState_.localCounterBuf.subBuffer(
                groupId * sizeof(uint64_t)),
            bytesThis);
      }
      group.sync();
      dataOff += bytesThis;
    }

    if (group.is_leader()) {
      state.nextStep = static_cast<int64_t>(baseByte + protocolBytes);
      state.activeStage = detail::IbSendRecvProgressStage::Done;
      state.activeBaseStep = 0;
      state.activeNextByte = 0;
      trace_ibgda_event(
          trace,
          self_rank,
          PipesTraceEventType::kIbSendEnd,
          trace_ibgda_step(nbytes),
          static_cast<uint16_t>(groupId));
    }
    group.sync();
#endif
  }

  /**
   * recv — receive one block's tile from pipelined RDMA.
   *
   * Waits for data to arrive in recvStaging, then copies recvStaging → dst.
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
   * @param group           ThreadGroup (all threads participate in memcpy,
   *                        leader does signal ops).
   * @param dst             Destination for this block's tile.
   * @param nbytes          Payload bytes to receive for this group. The
   *                        internal protocol byte count is rounded up to 16
   *                        bytes and consumed in perBlockSlot-sized pieces, or
   *                        smaller sub-chunks when max_signal_bytes is set.
   * @param active_blocks   Number of block-groups sharing each logical slot in
   *                        this call. 0 means use maxGroups.
   * @param max_signal_bytes Max bytes per signaled sub-chunk within one
   *                        perBlockSlot. 0 means one signal per perBlockSlot.
   *                        Must match the sender's value.
   * @param timeout         Optional timeout for wait operations.
   */
  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void recv(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      int active_blocks = 0,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
#if !PIPES_IS_DEVICE_COMPILE
    (void)group;
    (void)dst;
    (void)nbytes;
    (void)active_blocks;
    (void)max_signal_bytes;
    (void)timeout;
#else
    recvWithTrace<CopyOp>(
        group,
        dst,
        nbytes,
        active_blocks,
        max_signal_bytes,
        timeout,
        {},
        0,
        args...);
#endif
  }

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void recvWithTrace(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      int active_blocks,
      std::size_t max_signal_bytes,
      const Timeout& timeout,
      PipesTraceHandle trace,
      uint8_t self_rank,
      Args... args) {
#if !PIPES_IS_DEVICE_COMPILE
    (void)group;
    (void)dst;
    (void)nbytes;
    (void)active_blocks;
    (void)max_signal_bytes;
    (void)timeout;
    (void)trace;
    (void)self_rank;
#else
    if (nbytes == 0) {
      return;
    }
    const std::size_t protocolBytes = align_protocol_bytes(nbytes);

    const int groupId = group.group_id;
    const int effActive =
        active_blocks > 0 ? active_blocks : sendRecvState_.maxGroups;

    if (effActive > sendRecvState_.maxGroups) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: recv active_blocks=%d > maxGroups=%d\n",
            effActive,
            sendRecvState_.maxGroups);
      }
      PIPES_DEVICE_TRAP();
    }
    if (groupId >= effActive) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: recv group_id=%u >= active_blocks=%d\n",
            groupId,
            effActive);
      }
      PIPES_DEVICE_TRAP();
    }

    const std::size_t perBlockSlot =
        (sendRecvState_.dataBufferSize / effActive) & ~15ULL;
    if (perBlockSlot == 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: recv perBlockSlot=0 "
            "(dataBufferSize=%llu, active_blocks=%d)\n",
            (unsigned long long)sendRecvState_.dataBufferSize,
            effActive);
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
    const int maxGroups = sendRecvState_.maxGroups;
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

    if (group.is_leader()) {
      trace_ibgda_event(
          trace,
          self_rank,
          PipesTraceEventType::kIbRecvBegin,
          /*step=*/0,
          static_cast<uint16_t>(groupId));
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
      wait_signal(
          group,
          sendRecvState_.localSignalBuf.subBuffer(groupId * sizeof(uint64_t)),
          streamEnd,
          timeout);

      // (2) Cooperative copy: local recvStaging → dst via CopyOp.
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
      signal(
          group,
          sendRecvState_.remoteSignalBuf.subBuffer(
              (maxGroups + groupId) * sizeof(uint64_t)),
          bytesThis);
      dataOff += bytesThis;
    }

    if (group.is_leader()) {
      state.nextStep = static_cast<int64_t>(baseByte + protocolBytes);
      state.activeStage = detail::IbSendRecvProgressStage::Done;
      state.activeBaseStep = 0;
      state.activeNextByte = 0;
      trace_ibgda_event(
          trace,
          self_rank,
          PipesTraceEventType::kIbRecvEnd,
          trace_ibgda_step(nbytes),
          static_cast<uint16_t>(groupId));
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
   *     - Reads state[maxGroups + groupId].nextStep (same index as recv)
   *     - Waits DATA_READY on localSignalBuf[groupId] (matches send's
   *       piggybacked signal on remoteSignalBuf[groupId])
   *     - Signals SLOT_FREE on remoteSignalBuf[maxGroups + groupId]
   *       (matches send's backpressure wait on localSignalBuf[maxGroups +
   *       groupId])
   *
   *   Fwd side (fwd transport):
   *     - Reads state[groupId].nextStep (same index as send)
   *     - Waits NIC_DONE on localCounterBuf[groupId] (matches send's
   *       self-counter)
   *     - Waits SLOT_FREE on localSignalBuf[maxGroups + groupId]
   *       (matches recv's backpressure release)
   *     - RDMA puts with DATA_READY on remoteSignalBuf[groupId]
   *       + NIC_DONE on localCounterBuf[groupId]
   *       (matches recv's DATA_READY wait)
   *
   * Any chain of send → forward* → recv is therefore valid: each
   * forward consumes exactly the signals its predecessor produces
   * and produces exactly the signals its successor expects.
   *
   * @param group           ThreadGroup (all threads participate).
   * @param dst             Application destination (may be nullptr if
   *                        CopyOp handles it, e.g. reduce-scatter).
   * @param fwd             Forward transport (sends to next peer in ring).
   * @param nbytes          Payload bytes to receive and forward. The internal
   *                        protocol byte count is rounded up to 16 bytes.
   * @param active_blocks   Number of block-groups sharing the slot. 0 =
   * maxGroups.
   * @param max_signal_bytes Max bytes per signaled sub-chunk. 0 = perBlockSlot.
   * @param timeout         Optional timeout for wait operations.
   * @param args            Extra args forwarded to CopyOp::forward.
   */
  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void forward(
      ThreadGroup& group,
      void* __restrict__ dst,
      P2pIbgdaTransportDevice& fwd,
      std::size_t nbytes,
      int active_blocks = 0,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout(),
      Args... args) {
    forwardWithTrace<CopyOp>(
        group,
        dst,
        fwd,
        nbytes,
        active_blocks,
        max_signal_bytes,
        timeout,
        {},
        0,
        args...);
  }

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void forwardWithTrace(
      ThreadGroup& group,
      void* __restrict__ dst,
      P2pIbgdaTransportDevice& fwd,
      std::size_t nbytes,
      int active_blocks,
      std::size_t max_signal_bytes,
      const Timeout& timeout,
      PipesTraceHandle trace,
      uint8_t self_rank,
      Args... args) {
#if PIPES_IS_DEVICE_COMPILE
#ifdef __HIP_PLATFORM_AMD__
    static_assert(
        false,
        "P2pIbgdaTransportDevice::forward() requires NVIDIA GPU (DOCA/IBGDA)");
#endif
    if (nbytes == 0) {
      return;
    }
    const std::size_t protocolBytes = align_protocol_bytes(nbytes);

    const int groupId = group.group_id;

    // --- recv side (this transport) ---
    const int recvEffActive =
        active_blocks > 0 ? active_blocks : sendRecvState_.maxGroups;
    if (recvEffActive > sendRecvState_.maxGroups || groupId >= recvEffActive) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: forward recv active_blocks=%d "
            "maxGroups=%d groupId=%u\n",
            recvEffActive,
            sendRecvState_.maxGroups,
            groupId);
      }
      PIPES_DEVICE_TRAP();
    }

    const std::size_t recvPerBlockSlot =
        (sendRecvState_.dataBufferSize / recvEffActive) & ~15ULL;
    if (recvPerBlockSlot == 0) {
      if (group.is_leader()) {
        printf("[PIPES] FATAL: forward recvPerBlockSlot=0\n");
      }
      PIPES_DEVICE_TRAP();
    }

    // --- fwd side (fwd transport) ---
    const int fwdEffActive =
        active_blocks > 0 ? active_blocks : fwd.sendRecvState_.maxGroups;
    if (fwdEffActive > fwd.sendRecvState_.maxGroups ||
        groupId >= fwdEffActive) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: forward fwd active_blocks=%d "
            "maxGroups=%d groupId=%u\n",
            fwdEffActive,
            fwd.sendRecvState_.maxGroups,
            groupId);
      }
      PIPES_DEVICE_TRAP();
    }

    const std::size_t fwdPerBlockSlot =
        (fwd.sendRecvState_.dataBufferSize / fwdEffActive) & ~15ULL;
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
    const int recvMaxGroups = sendRecvState_.maxGroups;
    const int recvStateIndex = progress_recv_index(group);
    auto& recvSlotState = progress_state_slot(group, recvStateIndex);
    assert_progress_slot_idle(group, recvSlotState, "forward recv");
    const uint64_t recvBaseByte = static_cast<uint64_t>(recvSlotState.nextStep);
    const std::size_t recvPipelineBytes = recvPerBlockSlot * recvPipelineDepth;

    const int fwdPipelineDepth = fwd.sendRecvState_.pipelineDepth;
    const std::size_t fwdDataBufSize = fwd.sendRecvState_.dataBufferSize;
    const int fwdMaxGroups = fwd.sendRecvState_.maxGroups;
    const int fwdStateIndex = fwd.progress_send_index(group);
    auto& fwdSlotState = fwd.progress_state_slot(group, fwdStateIndex);
    fwd.assert_progress_slot_idle(group, fwdSlotState, "forward send");
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

    if (group.is_leader()) {
      trace_ibgda_event(
          trace,
          self_rank,
          PipesTraceEventType::kIbForwardBegin,
          /*step=*/0,
          static_cast<uint16_t>(groupId));
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
      wait_signal(
          group,
          sendRecvState_.localSignalBuf.subBuffer(groupId * sizeof(uint64_t)),
          recvStreamEnd,
          timeout);

      // (2) Wait for NIC_DONE on fwd's sendStaging (backpressure).
      if (fwdStreamEnd > fwdPipelineBytes) {
        fwd.wait_counter(
            group,
            fwd.sendRecvState_.localCounterBuf.subBuffer(
                groupId * sizeof(uint64_t)),
            fwdStreamEnd - fwdPipelineBytes,
            timeout);
      }

      // (3) CopyOp::forward — transform recv staging → dst + fwd staging.
      const std::size_t validBytes =
          valid_payload_bytes(dataOff, bytesThis, nbytes);
      if (validBytes > 0) {
        CopyOp::forward(
            dst ? static_cast<char*>(dst) + dataOff : nullptr,
            fwd.sendRecvState_.sendStagingPtr + fwdStagingOff,
            sendRecvState_.recvStagingPtr + recvStagingOff,
            validBytes,
            group,
            dataOff,
            args...);
      }
      group.sync();

      // (4) Signal SLOT_FREE to sender (this transport).
      //     CRITICAL: must happen BEFORE waiting on fwd's SLOT_FREE (step 5)
      //     to break circular ring dependency.
      signal(
          group,
          sendRecvState_.remoteSignalBuf.subBuffer(
              (recvMaxGroups + groupId) * sizeof(uint64_t)),
          bytesThis);

      // (5) Wait for fwd receiver's SLOT_FREE (backpressure on fwd's
      //     recvStaging).
      if (fwdStreamEnd > fwdPipelineBytes) {
        fwd.wait_signal(
            group,
            fwd.sendRecvState_.localSignalBuf.subBuffer(
                (fwdMaxGroups + groupId) * sizeof(uint64_t)),
            fwdStreamEnd - fwdPipelineBytes,
            timeout);
      }

      // (6) threadfence_system + RDMA put via fwd transport.
      __threadfence_system();
      group.sync();
      if (group.is_leader()) {
        ThreadGroup solo{
            0, 1, group.group_id, group.block_id, 1, SyncScope::THREAD};
        fwd.put(
            solo,
            fwd.sendRecvState_.sendStagingBuf.subBuffer(fwdStagingOff),
            fwd.sendRecvState_.recvStagingBuf.subBuffer(fwdStagingOff),
            bytesThis,
            fwd.sendRecvState_.remoteSignalBuf.subBuffer(
                groupId * sizeof(uint64_t)),
            bytesThis,
            fwd.sendRecvState_.localCounterBuf.subBuffer(
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
      trace_ibgda_event(
          trace,
          self_rank,
          PipesTraceEventType::kIbForwardEnd,
          trace_ibgda_step(nbytes),
          static_cast<uint16_t>(groupId));
    }
    group.sync();
#endif
  }

  // Send/recv state accessors

  __host__ __device__ const IbSendRecvState& send_recv_state() const {
    return sendRecvState_;
  }

  /**
   * Maximum bytes a block can send without blocking on pipeline backpressure.
   *
   * The staging buffer is split into pipelineDepth slots, each divided evenly
   * across active_blocks. A block can fill all its slots before the NIC must
   * drain any of them, so the non-blocking window is:
   *   (dataBufferSize / active_blocks) * pipelineDepth
   *
   * Callers should loop over their data in pipeline_window-sized chunks so
   * that send()/forward() never stall waiting for a free slot.
   *
   * @param active_blocks  Total blocks sharing this transport (typically
   *                       gridDim.x).
   */
  __device__ __forceinline__ std::size_t pipeline_window(
      int active_blocks) const {
    const std::size_t per_block_slot =
        (sendRecvState_.dataBufferSize / active_blocks) & ~15ULL;
    return per_block_slot * sendRecvState_.pipelineDepth;
  }

 private:
  /**
   * Physical staging range for the next resumable progress step.
   *
   * `stagingOff` is an offset into the transport-owned send/recv staging
   * buffers. `dataOff` is the matching protocol offset into the caller's user
   * buffer.
   * `bytes` never crosses a per-block staging partition or the reserved
   * protocol byte count. `streamEnd` is the absolute protocol byte value after
   * this chunk and is used as the DATA_READY, SLOT_FREE, and NIC_DONE
   * readiness threshold. `dataOff` is a protocol offset; callers mask it
   * against the payload byte count before invoking user-buffer copy callbacks.
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
      int active_blocks,
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
    const int effActive =
        active_blocks > 0 ? active_blocks : sendRecvState_.maxGroups;
    if (effActive > sendRecvState_.maxGroups) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: %s active_blocks=%d > maxGroups=%d\n",
            opName,
            effActive,
            sendRecvState_.maxGroups);
      }
      PIPES_DEVICE_TRAP();
    }
    if (groupId < 0 || groupId >= effActive) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: %s group_id=%d >= active_blocks=%d\n",
            opName,
            groupId,
            effActive);
      }
      PIPES_DEVICE_TRAP();
    }

    const std::size_t perBlockSlot =
        (sendRecvState_.dataBufferSize / effActive) & ~15ULL;
    if (perBlockSlot == 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: %s perBlockSlot=0 "
            "(dataBufferSize=%llu, active_blocks=%d)\n",
            opName,
            (unsigned long long)sendRecvState_.dataBufferSize,
            effActive);
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
    (void)active_blocks;
    (void)max_signal_bytes;
    (void)opName;
    return {};
#endif
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

  // --- Members ---
  // Per-NIC bundles (qps + companion_qps + sink_lkey + device_id).
  DeviceSpan<NicDeviceIbgdaResources> nicDevices_{};

  // Owned signal/counter buffers (set by transport during construction)
  IbgdaRemoteBuffer ownedRemoteSignalBuf_{}; // outbox: signal peer's inbox
  IbgdaLocalBuffer ownedLocalSignalBuf_{}; // inbox: receive signals from peers
  IbgdaLocalBuffer ownedCounterBuf_{}; // local counter for companion QP

  int numSignalSlots_{0};
  int numCounterSlots_{0};
  int maxGroups_{0};
  int qpsPerBlockPerNic_{1};
  DeviceSpan<IbgdaBlockQpState> blockQpState_{};

  IbSendRecvState sendRecvState_{};
};

} // namespace comms::prims
