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
#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace comms::prims {

struct Memcpy;

inline constexpr uint64_t kDefaultDeviceTimeoutCycles = 10'000'000'000ULL;

// `PIPES_DEVICE_TRAP()` is defined in `comms/prims/core/DeviceMacros.cuh` and
// is intentionally available across all `comms/prims` device headers.
//
// `IbgdaSendRecvProgressStatus` and the pipelined send/recv algorithm now live
// in the shared `IbSendRecvDevice` (P2pIbTransportDeviceDecl.cuh); this class
// delegates its send/recv/forward/init/progress methods to a `sendRecv_`
// member.

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
   *                              maxChannels * qpDirectionCount *
   *                              qpsPerConnection main and companion QPs.
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
      int maxChannels = 0,
      int qpsPerConnection = 1,
      int qpDirectionCount = kIbDirections,
      DeviceSpan<IbLocalChannel> localChannels = {},
      IbSendRecvState sendRecvState = {})
      : nicDevices_(nicDevices),
        ownedRemoteSignalBuf_(ownedRemoteSignalBuf),
        ownedLocalSignalBuf_(ownedLocalSignalBuf),
        ownedCounterBuf_(ownedCounterBuf),
        numSignalSlots_(numSignalSlots),
        numCounterSlots_(numCounterSlots),
        maxChannels_(maxChannels),
        qpsPerConnection_(qpsPerConnection),
        qpDirectionCount_(qpDirectionCount),
        localChannels_(localChannels),
        sendRecv_(sendRecvState) {}

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
   * Caller is responsible for gating to one thread. Fixed-channel callers must
   * use the group-scope APIs because the solo group id is not a channel id.
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
   * signal (thread-scope, slot-index) - Single-thread variant. Fixed-channel
   * callers must use the group-scope APIs because the solo group id is not a
   * channel id.
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
   * put (thread-scope) - Single-thread variant. Caller gates. Fixed-channel
   * callers must use the group-scope APIs because the solo group id is not a
   * channel id.
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
      uint64_t signalVal = 1,
      IbDirection direction = IbDirection::Send) {
    if (group.is_leader()) {
      validate_group_scope(group);
      IbgdaLane lane = control_lane(group, direction);
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
   * flush set. Raw put/signal/fence users default to the Send direction;
   * receive-direction drains are only needed by protocols that explicitly post
   * receive-direction data WQEs.
   *
   * @param group Thread group; all threads must call. Leader waits, all sync.
   */
  __device__ void flush(
      ThreadGroup& group,
      IbDirection direction = IbDirection::Send) {
    if (group.is_leader()) {
      validate_group_scope(group);
      drain_flush_lanes(group, direction);
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
  __device__ void fence(
      ThreadGroup& group,
      IbDirection direction = IbDirection::Send) {
    flush(group, direction);
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
    uint32_t channel_id{0};
    uint32_t qp_slot_per_nic{0};
    uint32_t companion_slot_per_nic{0};
    IbDirection direction{IbDirection::Send};
    IbQpState* qp_state{nullptr};
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

  __device__ __forceinline__ void validate_channel_id(
      uint32_t channelId) const {
    if (blockIdx.y != 0 || blockIdx.z != 0 || blockDim.y != 1 ||
        blockDim.z != 1) {
      printf(
          "[PIPES] FATAL: IBGDA channel QP selection currently supports "
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
    if (channelId >= static_cast<uint32_t>(maxChannels_) ||
        localChannels_.empty()) {
      printf(
          "[PIPES] FATAL: IBGDA channel_id=%u out of range [0, %d) "
          "or channel state missing\n",
          channelId,
          maxChannels_);
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
    validate_channel_id(group.group_id);
  }

  __device__ __forceinline__ IbQpState& qp_state(
      uint32_t channelId,
      IbDirection direction) const {
    validate_channel_id(channelId);
    auto& channel = localChannels_[channelId];
    return direction == IbDirection::Send ? channel.sendQp : channel.recvQp;
  }

  __device__ __forceinline__ IbgdaLane lane_from_ordinal(
      uint32_t channelId,
      IbDirection direction,
      uint32_t laneOrdinal) const {
    validate_channel_id(channelId);
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
    const uint32_t directionIndex = static_cast<uint32_t>(direction);
    if (directionIndex >= static_cast<uint32_t>(qpDirectionCount_)) {
      printf(
          "[PIPES] FATAL: IBGDA direction=%u is unavailable for "
          "qpDirectionCount=%d\n",
          directionIndex,
          qpDirectionCount_);
      PIPES_DEVICE_TRAP();
    }
    const uint32_t qpSlotPerNic =
        ((channelId * static_cast<uint32_t>(qpDirectionCount_) +
          directionIndex) *
         static_cast<uint32_t>(qpsPerConnection_)) +
        qpIndex;
    const uint32_t companionSlotPerNic = qpSlotPerNic;
    const NicDeviceIbgdaResources& nic = nicDevices_[nicId];
    if (qpIndex >= static_cast<uint32_t>(qpsPerConnection_) ||
        qpSlotPerNic >= nic.qps.size() ||
        companionSlotPerNic >= nic.companion_qps.size()) {
      printf(
          "[PIPES] FATAL: invalid IBGDA lane channel=%u direction=%u nic=%u "
          "qpIndex=%u qpsPerConnection=%d qps=%u companionQps=%u\n",
          channelId,
          directionIndex,
          nicId,
          qpIndex,
          qpsPerConnection_,
          static_cast<unsigned>(nic.qps.size()),
          static_cast<unsigned>(nic.companion_qps.size()));
      PIPES_DEVICE_TRAP();
    }
    return IbgdaLane{
        .nic_id = nicId,
        .qp_index = qpIndex,
        .lane_ordinal = laneOrdinal,
        .channel_id = channelId,
        .qp_slot_per_nic = qpSlotPerNic,
        .companion_slot_per_nic = companionSlotPerNic,
        .direction = direction,
        .qp_state = &qp_state(channelId, direction),
        .qp = nic.qps[qpSlotPerNic],
        .companion_qp = nic.companion_qps[companionSlotPerNic]};
  }

  __device__ __forceinline__ uint32_t
  select_put_lane_ordinal(uint32_t channelId, IbDirection direction) {
    validate_channel_id(channelId);
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
        static_cast<uint32_t>(nicDevices_.size() * qpsPerConnection_);
    if (numLanes == 1) {
      return 0;
    }
    uint32_t seq = qp_state(channelId, direction).cursor++;
    return seq % numLanes;
  }

  __device__ __forceinline__ IbgdaLane
  select_put_lane(ThreadGroup& group, IbDirection direction) {
    const uint32_t channelId = group.group_id;
    return lane_from_ordinal(
        channelId, direction, select_put_lane_ordinal(channelId, direction));
  }

  __device__ __forceinline__ IbgdaLane
  control_lane(ThreadGroup& group, IbDirection direction) const {
    return lane_from_ordinal(group.group_id, direction, 0);
  }

  __device__ __forceinline__ uint32_t num_qp_lanes() const {
    return static_cast<uint32_t>(nicDevices_.size() * qpsPerConnection_);
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
    auto& state = *lane.qp_state;
    atomic_max_u64(&state.lastFlushWqe[lane.lane_ordinal], ticket);
    atomic_or_u64(&state.pendingFlushLanesMask, 1ULL << lane.lane_ordinal);
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

  __device__ void wait_lanes(
      uint32_t channelId,
      IbDirection direction,
      uint64_t mask,
      const uint64_t* tickets) {
    const uint32_t numLanes = num_qp_lanes();
    for (uint32_t laneId = 0; laneId < numLanes; ++laneId) {
      if ((mask & (1ULL << laneId)) == 0) {
        continue;
      }
      IbgdaLane lane = lane_from_ordinal(channelId, direction, laneId);
      wait_local_on_qp(lane.qp, tickets[laneId]);
    }
  }

  __device__ void drain_flush_lanes(ThreadGroup& group, IbDirection direction) {
    auto& state = qp_state(group.group_id, direction);
    const uint64_t mask = atomic_exchange_u64(&state.pendingFlushLanesMask, 0);
    wait_lanes(group.group_id, direction, mask, state.lastFlushWqe);
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
      IbgdaLane lane = select_put_lane(group, IbDirection::Send);
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
      laneOrdinal = select_put_lane_ordinal(group.group_id, IbDirection::Send);
    }
    laneOrdinal = group.broadcast<uint32_t>(laneOrdinal);
    IbgdaLane lane =
        lane_from_ordinal(group.group_id, IbDirection::Send, laneOrdinal);

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
    sendRecv_.init_send_progress(
        group, nbytes, active_blocks, max_signal_bytes);
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
    sendRecv_.init_recv_progress(
        group, nbytes, active_blocks, max_signal_bytes);
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
    return sendRecv_.progress_send_once<P2pIbgdaTransportDevice, CopyOp>(
        *this,
        group,
        src,
        nbytes,
        active_blocks,
        max_signal_bytes,
        timeout,
        args...);
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
    return sendRecv_.progress_recv_once<P2pIbgdaTransportDevice, CopyOp>(
        *this,
        group,
        dst,
        nbytes,
        active_blocks,
        max_signal_bytes,
        timeout,
        args...);
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
    if (group.is_leader()) {
      trace_ibgda_event(
          trace,
          self_rank,
          PipesTraceEventType::kIbSendBegin,
          /*step=*/0,
          static_cast<uint16_t>(group.group_id));
    }
    sendRecv_.send<P2pIbgdaTransportDevice, CopyOp>(
        *this,
        group,
        src,
        nbytes,
        active_blocks,
        max_signal_bytes,
        timeout,
        args...);
    if (group.is_leader()) {
      trace_ibgda_event(
          trace,
          self_rank,
          PipesTraceEventType::kIbSendEnd,
          trace_ibgda_step(nbytes),
          static_cast<uint16_t>(group.group_id));
    }
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
    if (group.is_leader()) {
      trace_ibgda_event(
          trace,
          self_rank,
          PipesTraceEventType::kIbRecvBegin,
          /*step=*/0,
          static_cast<uint16_t>(group.group_id));
    }
    sendRecv_.recv<P2pIbgdaTransportDevice, CopyOp>(
        *this,
        group,
        dst,
        nbytes,
        active_blocks,
        max_signal_bytes,
        timeout,
        args...);
    if (group.is_leader()) {
      trace_ibgda_event(
          trace,
          self_rank,
          PipesTraceEventType::kIbRecvEnd,
          trace_ibgda_step(nbytes),
          static_cast<uint16_t>(group.group_id));
    }
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
    if (nbytes == 0) {
      return;
    }
    if (group.is_leader()) {
      trace_ibgda_event(
          trace,
          self_rank,
          PipesTraceEventType::kIbForwardBegin,
          /*step=*/0,
          static_cast<uint16_t>(group.group_id));
    }
    sendRecv_.forward<CopyOp>(
        *this,
        group,
        dst,
        fwd.sendRecv_,
        fwd,
        nbytes,
        active_blocks,
        max_signal_bytes,
        timeout,
        args...);
    if (group.is_leader()) {
      trace_ibgda_event(
          trace,
          self_rank,
          PipesTraceEventType::kIbForwardEnd,
          trace_ibgda_step(nbytes),
          static_cast<uint16_t>(group.group_id));
    }
#endif
  }

  // Send/recv state accessors

  __host__ __device__ const IbSendRecvState& send_recv_state() const {
    return sendRecv_.send_recv_state();
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
    return sendRecv_.pipeline_window(active_blocks);
  }

 private:
  // --- Members ---
  // Per-NIC bundles (qps + companion_qps + sink_lkey + device_id).
  DeviceSpan<NicDeviceIbgdaResources> nicDevices_{};

  // Owned signal/counter buffers (set by transport during construction)
  IbgdaRemoteBuffer ownedRemoteSignalBuf_{}; // outbox: signal peer's inbox
  IbgdaLocalBuffer ownedLocalSignalBuf_{}; // inbox: receive signals from peers
  IbgdaLocalBuffer ownedCounterBuf_{}; // local counter for companion QP

  int numSignalSlots_{0};
  int numCounterSlots_{0};
  int maxChannels_{0};
  int qpsPerConnection_{1};
  int qpDirectionCount_{kIbDirections};
  DeviceSpan<IbLocalChannel> localChannels_{};

  IbSendRecvDevice sendRecv_{};
};

} // namespace comms::prims
