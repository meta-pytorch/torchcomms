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
// `DocaVerbsUtils.cuh` wrapper that defines `comms::pipes::doca_fence`.
// On AMD, the DOCA headers don't exist; the DocaCompat shim provides
// `doca_*` type aliases, constants, function shims, and `doca_fence`.
#ifdef __HIP_PLATFORM_AMD__
#include "comms/pipes/amd/DocaCompat.h"
#else
#include <device/doca_gpunetio_dev_verbs_counter.cuh>
#include <device/doca_gpunetio_dev_verbs_onesided.cuh>

#include "comms/pipes/DocaVerbsUtils.cuh"
#endif
#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/DeviceMacros.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {

struct Memcpy;

inline constexpr uint64_t kDefaultDeviceTimeoutCycles = 10'000'000'000ULL;

// `PIPES_DEVICE_TRAP()` is defined in `comms/pipes/DeviceMacros.cuh` and
// is intentionally available across all `comms/pipes` device headers.

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

/**
 * P2pIbgdaTransportDevice - Device-side per-peer RDMA transport handle
 *
 * Every method has two overloads:
 *   Group-scope: put(group, ...) — all threads in group must call.
 *     QP selection: single QP for now (multi-QP via group.group_id % numQps
 *     will be added in a follow-up diff).
 *     Data transfer uses the exact buffer span supplied by the caller.
 *     Threads in the group coordinate the operation; callers that want the
 *     transport to shard a larger buffer should use put_cooperative().
 *     Signal/counter/fence are leader-only with group.sync().
 *
 *   Thread-scope: put(...) — single thread calls.
 *     QP selection: always QP 0.
 *     Implemented as thin wrapper: creates solo ThreadGroup, forwards.
 *
 * CRITICAL: Do not mix scope families in an ordered sequence.
 *   put(group,...) -> signal(0) is BROKEN (different QPs, FENCE invalid).
 *   put(group,...) -> signal(group,0) is CORRECT (same QP).
 *
 * Signal is always fenced (NIC completes prior WQEs before signal).
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
   * carries its own primary and companion QPs and a sink lkey. The host-side
   * builder is responsible for peer-rotating the NicDeviceIbgdaResources[]
   * order so that `nic_qp_for_group(g)`'s nic_id (= g % nicDevices.size())
   * produces balanced thread-per-peer scatter when nicDevices.size() > 1.
   *
   * Single-NIC usage: pass a 1-element nicDevices span. All ops fall through
   * to NIC 0.
   *
   * @param nicDevices          GPU span of per-NIC bundles (length =
   *                              numNics). Each NicDeviceIbgdaResources owns
   *                              numQpsPerPeerPerNic primary + companion QP
   *                              pointers and the per-NIC sink lkey.
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
   * @param discardSignalSlot     Remote uint64_t slot used as a "throwaway"
   *                              signal target for counter-only puts (see
   *                              put_impl for rationale). The peer never
   *                              reads this slot. Required when counter is
   *                              used; ignored otherwise.
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
      IbgdaRemoteBuffer discardSignalSlot = {},
      IbSendRecvState sendRecvState = {})
      : nicDevices_(nicDevices),
        ownedRemoteSignalBuf_(ownedRemoteSignalBuf),
        ownedLocalSignalBuf_(ownedLocalSignalBuf),
        ownedCounterBuf_(ownedCounterBuf),
        discardSignalSlot_(discardSignalSlot),
        numSignalSlots_(numSignalSlots),
        numCounterSlots_(numCounterSlots),
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
   * Caller is responsible for gating to one thread. Uses QP 0.
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
    ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
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
   * Always FENCEd against preceding WQEs on the same QP, so signal arrives
   * after any prior put() completes at the NIC.
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

  /** signal (thread-scope, slot-index) - Single-thread variant. Uses QP 0. */
  __device__ void signal(int signalId, uint64_t signalVal = 1) {
    ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
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
    ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
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
    ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
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
    ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
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
    ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
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
   *                   the counter. With signalBuf set: companion-QP loopback
   *                   atomic. Counter-only uses a discard signal target plus
   *                   the same companion-QP loopback counter.
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
   * put (thread-scope) - Single-thread, QP 0. Caller gates.
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
    ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
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
   * Always FENCEd against preceding WQEs on the same QP, so signal arrives
   * after any prior put() completes at the NIC.
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
      signal_fenced(group.group_id, signalBuf, signalVal);
    }
    group.sync();
  }

  /** signal (thread-scope) - Single-thread variant. Uses QP 0. */
  __device__ void signal(
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal = 1) {
    ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
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
    ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
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
    ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
    wait_counter(solo, counterBuf, expected, timeout);
  }

  /**
   * flush (group-scope) - Wait for all in-flight transport operations to
   * complete on this group's QP.
   *
   * Drains the QP via a NOP WQE. Use this when callers want "wait for
   * completion" semantics independent of the underlying mechanism, so the
   * implementation can later evolve (e.g. cross-QP flush) without churning
   * call sites.
   *
   * @param group Thread group; all threads must call. Leader issues NOP
   *              WQE and waits, all sync.
   */
  __device__ void flush(ThreadGroup& group) {
    if (group.is_leader()) {
      flush_impl(group.group_id);
    }
    group.sync();
  }

  /** flush (thread-scope) - Single-thread variant. */
  __device__ void flush() {
    flush_impl(0);
  }

  /**
   * fence (group-scope) - Drain all pending WQEs on this group's QP.
   *
   * Aliased to flush(). Prefer flush() in new code.
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
    ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
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
    ThreadGroup solo{0, 1, 0, 1, SyncScope::THREAD};
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

  __device__ void finish_put_impl(
      ThreadGroup& group,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal,
      const IbgdaLocalBuffer& counterBuf,
      uint64_t counterVal) {
    const bool hasSignal = signalBuf.ptr != nullptr;
    const bool hasCounter = counterBuf.ptr != nullptr;

    // The DOCA verbs API exposes:
    //   - signal_fenced (atomic FA, FENCEd against prior put)
    //   - signal_counter (signal_fenced on primary QP + companion-QP loopback
    //     atomic for the local counter, both ordered against prior put)
    //
    // It does NOT expose a "counter-only" primitive. To keep put() async and
    // ordered for the counter-only case, we route through signal_counter with
    // a transport-owned discard slot as the signal target — the peer never
    // reads it, so the signal value is garbage by design.
    //
    // The discard-slot trick lets every put_impl branch be a single async
    // WQE post; the alternative (flush_impl + GPU atomicAdd) would silently
    // make counter-only puts synchronous and add a CQ-poll round-trip on
    // the hot path.
    if (group.is_leader()) {
      if (hasSignal && hasCounter) {
        signal_counter(
            group.group_id, signalBuf, signalVal, counterBuf, counterVal);
      } else if (hasSignal) {
        signal_fenced(group.group_id, signalBuf, signalVal);
      } else if (hasCounter) {
        signal_counter(
            group.group_id, discardSignalSlot_, 0, counterBuf, counterVal);
      }
    }
    group.sync();
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
    if (group.is_leader() && nbytes > 0) {
      put_single_impl(group.group_id, localBuf, remoteBuf, nbytes);
    }
    group.sync();
    finish_put_impl(group, signalBuf, signalVal, counterBuf, counterVal);
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
    put_cooperative_data_impl(group, localBuf, remoteBuf, nbytes);
    finish_put_impl(group, signalBuf, signalVal, counterBuf, counterVal);
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

  __device__ void put_cooperative_data_impl(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes) {
    if (nbytes == 0) {
      group.sync();
      return;
    }

    std::size_t chunkSize = nbytes / group.group_size;
    std::size_t offset = group.thread_id_in_group * chunkSize;
    std::size_t laneBytes = (group.thread_id_in_group == group.group_size - 1)
        ? (nbytes - offset)
        : chunkSize;

    IbgdaLocalBuffer laneBuf = localBuf.subBuffer(offset);
    IbgdaRemoteBuffer laneRemoteBuf = remoteBuf.subBuffer(offset);

    if (group.group_size == 1) {
      put_single_impl(group.group_id, laneBuf, laneRemoteBuf, laneBytes);
      return;
    }

    auto idx = nic_qp_for_group(group.group_id);
    const NicDeviceIbgdaResources& nic = nicDevices_[idx.nic_id];
    auto* qp = nic.qps[idx.qp_id];

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

    doca_gpu_dev_verbs_wqe_prepare_write(
        qp,
        wqe_ptr,
        static_cast<uint16_t>(wqe_idx),
        DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        0,
        reinterpret_cast<uint64_t>(laneRemoteBuf.ptr),
        laneRemoteBuf.rkey_per_device[idx.nic_id].value,
        reinterpret_cast<uint64_t>(laneBuf.ptr),
        laneBuf.lkey_per_device[idx.nic_id].value,
        static_cast<uint32_t>(laneBytes));

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
  }

  // --- put_single_impl: one thread, one WQE ---

  __device__ void put_single_impl(
      uint32_t group_id,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes) {
    auto idx = nic_qp_for_group(group_id);
    const NicDeviceIbgdaResources& nic = nicDevices_[idx.nic_id];
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_addr localAddr = {
        .addr = reinterpret_cast<uint64_t>(localBuf.ptr),
        .key = localBuf.lkey_per_device[idx.nic_id].value};
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(remoteBuf.ptr),
        .key = remoteBuf.rkey_per_device[idx.nic_id].value};

    doca_gpu_dev_verbs_put<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
        DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>(
        nic.qps[idx.qp_id], remoteAddr, localAddr, nbytes, &ticket);
  }

  // --- signal_fenced: atomic fetch-add with NIC FENCE (always fenced) ---

  __device__ void signal_fenced(
      uint32_t group_id,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal) {
    auto idx = nic_qp_for_group(group_id);
    const NicDeviceIbgdaResources& nic = nicDevices_[idx.nic_id];
    doca_gpu_dev_verbs_qp* qp = nic.qps[idx.qp_id];
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(signalBuf.ptr),
        .key = signalBuf.rkey_per_device[idx.nic_id].value};
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
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp, wqe_idx + 1);
  }

  // --- signal_counter: fenced signal + companion QP loopback counter ---

  __device__ void signal_counter(
      uint32_t group_id,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal,
      const IbgdaLocalBuffer& counterBuf,
      uint64_t counterVal) {
    auto idx = nic_qp_for_group(group_id);
    const NicDeviceIbgdaResources& nic = nicDevices_[idx.nic_id];
    doca_gpu_dev_verbs_addr sigRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(signalBuf.ptr),
        .key = signalBuf.rkey_per_device[idx.nic_id].value};
    doca_gpu_dev_verbs_addr sigSinkAddr = {
        .addr = 0, .key = nic.sink_lkey.value};

    doca_gpu_dev_verbs_addr counterRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(counterBuf.ptr),
        .key = counterBuf.lkey_per_device[idx.nic_id].value};
    doca_gpu_dev_verbs_addr counterSinkAddr = {
        .addr = 0, .key = nic.sink_lkey.value};

    doca_gpu_dev_verbs_signal_counter<
        DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD,
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
        nic.qps[idx.qp_id],
        sigRemoteAddr,
        sigSinkAddr,
        signalVal,
        nic.companion_qps[idx.qp_id],
        counterRemoteAddr,
        counterSinkAddr,
        counterVal);
  }

  // --- flush_impl: NOP WQE + wait ---

  __device__ void flush_impl(uint32_t group_id) {
    doca_fence<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(active_qp(group_id));
  }

  // --- wait_local_impl: CQ poll for specific WQE (internal use only) ---

  __device__ void wait_local_impl(
      uint32_t group_id,
      doca_gpu_dev_verbs_ticket_t ticket,
      Timeout timeout = Timeout()) {
    if (!timeout.isEnabled()) {
      doca_gpu_dev_verbs_wait<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(active_qp(group_id), ticket);
    } else {
      int status;
      do {
        status = doca_gpu_dev_verbs_poll_one_cq_at<
            DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            doca_gpu_dev_verbs_qp_get_cq_sq(active_qp(group_id)), ticket);
        if (status == EBUSY) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "wait_local_impl timed out (ticket=%llu)",
              static_cast<unsigned long long>(ticket));
        }
      } while (status == EBUSY);
    }
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
  //   stepState        = persistent byte cursor. DATA_READY, SLOT_FREE, and
  //                      NIC_DONE counters also advance by bytes, which keeps
  //                      cursor state independent of max_signal_bytes.
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
   * Blocking send()/recv() advance the transport cursor when the call
   * completes; progress init reserves that cursor range before returning so
   * multiple resumable states and later blocking calls cannot reuse the same
   * protocol bytes.
   *
   * Usage starts by initializing the transport-owned state for one logical
   * transfer, then calling the matching progress method until it returns
   * `Done`:
   *
   *   transport->init_send_progress(
   *       group, nbytes, active_blocks, max_signal_bytes);
   *   while (transport->progress_send_once(group, src, timeout) !=
   *          IbgdaSendRecvProgressStatus::Done) {
   *     // Try another independent lane or return to the scheduler.
   *   }
   *
   * Receivers use the symmetric `init_recv_progress()` and
   * `progress_recv_once()` pair with the same `active_blocks` and compatible
   * `max_signal_bytes`. Zero-byte operations initialize directly to `Done`, so
   * callers can use the same loop shape for empty and non-empty transfers.
   *
   * The progress state is a property of this transport, indexed by
   * `group.group_id` and direction. A caller may have one send and one recv in
   * flight for the same group, but a second send or second recv init for the
   * same group before the previous operation reaches `Done` traps on device.
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
   * @param active_blocks Number of participating groups, or 0 for maxGroups.
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
    detail::IbSendRecvProgressState state{
        .groupId = static_cast<int>(group.group_id),
        .activeBlocks =
            active_blocks > 0 ? active_blocks : sendRecvState_.maxGroups,
        .nbytes = nbytes,
        .maxSignalBytes = max_signal_bytes,
        .stage = nbytes == 0 ? detail::IbSendRecvProgressStage::Done
                             : detail::IbSendRecvProgressStage::WaitNicDone,
    };
    if (nbytes == 0) {
      store_progress_state(group, progressIndex, state);
      return;
    }
    init_progress_geometry(group, state);
    state.baseStep = reserve_progress_step(group, state.groupId, nbytes);
    store_progress_state(group, progressIndex, state);
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
   * @param active_blocks Number of participating groups, or 0 for maxGroups.
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
    detail::IbSendRecvProgressState state{
        .groupId = static_cast<int>(group.group_id),
        .activeBlocks =
            active_blocks > 0 ? active_blocks : sendRecvState_.maxGroups,
        .nbytes = nbytes,
        .maxSignalBytes = max_signal_bytes,
        .stage = nbytes == 0 ? detail::IbSendRecvProgressStage::Done
                             : detail::IbSendRecvProgressStage::WaitDataReady,
    };
    if (nbytes == 0) {
      store_progress_state(group, progressIndex, state);
      return;
    }
    init_progress_geometry(group, state);
    state.baseStep = reserve_progress_step(
        group, sendRecvState_.maxGroups + state.groupId, nbytes);
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
   * byte range has completed.
   *
   * `CopyOp` must expose `send(dst, src, bytes, group, dataOffset, args...)`.
   * The default `Memcpy` copies bytes cooperatively across the supplied
   * `ThreadGroup`; custom copy ops may use `args` to pass reduction or
   * conversion context.
   *
   * @param group Thread group matching the one used during initialization.
   * @param src Source user buffer. The range `[src, src + nbytes)` must remain
   *            valid until `Done`.
   * @param timeout Optional device timeout checked while dependencies wait.
   * @param args Additional arguments forwarded to `CopyOp::send`.
   */
  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus progress_send_once(
      ThreadGroup& group,
      const void* __restrict__ src,
      const Timeout& timeout = Timeout(),
      Args... args) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef __HIP_PLATFORM_AMD__
    static_assert(
        sizeof(CopyOp) == 0,
        "P2pIbgdaTransportDevice::progress_send_once() requires NVIDIA GPU");
#endif
    const int progressIndex = progress_send_index(group);
    detail::IbSendRecvProgressState state =
        progress_state_slot(group, progressIndex);
    if (state.stage == detail::IbSendRecvProgressStage::Done) {
      return IbgdaSendRecvProgressStatus::Done;
    }
    if (state.nextByte >= state.nbytes) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: progress_send_once nextByte=%llu >= nbytes=%llu "
            "without Done stage\n",
            static_cast<unsigned long long>(state.nextByte),
            static_cast<unsigned long long>(state.nbytes));
      }
      PIPES_DEVICE_TRAP();
    }
    validate_send_progress_stage(group, state);

    const detail::IbSendRecvProgressStage initialStage = state.stage;
    const std::size_t initialNextByte = state.nextByte;
    const std::size_t pipelineBytes = state.perBlockSlot *
        static_cast<std::size_t>(sendRecvState_.pipelineDepth);

    if (state.stage == detail::IbSendRecvProgressStage::WaitNicDone) {
      const ProgressChunk chunk = next_chunk(state);
      if (chunk.streamEnd > pipelineBytes) {
        const uint64_t expected = chunk.streamEnd - pipelineBytes;
        uint32_t ready = 1;
        unsigned long long current = 0;
        if (group.is_leader()) {
          current = static_cast<unsigned long long>(
              read_counter(sendRecvState_.localCounterBuf.subBuffer(
                  state.groupId * sizeof(uint64_t))));
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

      CopyOp::send(
          sendRecvState_.sendStagingPtr + chunk.stagingOff,
          static_cast<const char*>(src) + chunk.dataOff,
          chunk.bytes,
          group,
          chunk.dataOff,
          args...);
      group.sync();
      transition_progress_stage(
          group, state, detail::IbSendRecvProgressStage::WaitSlotFree);
    }

    if (state.stage == detail::IbSendRecvProgressStage::WaitSlotFree) {
      const ProgressChunk chunk = next_chunk(state);
      if (chunk.streamEnd > pipelineBytes) {
        const uint64_t expected = chunk.streamEnd - pipelineBytes;
        uint32_t ready = 1;
        unsigned long long current = 0;
        if (group.is_leader()) {
          current = static_cast<unsigned long long>(
              read_signal(sendRecvState_.localSignalBuf.subBuffer(
                  (sendRecvState_.maxGroups + state.groupId) *
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
          if (state.stage != initialStage ||
              state.nextByte != initialNextByte) {
            store_progress_state(group, progressIndex, state);
            return IbgdaSendRecvProgressStatus::Progressed;
          }
          return IbgdaSendRecvProgressStatus::Waiting;
        }
      }

      __threadfence_system();
      group.sync();
      if (group.is_leader()) {
        ThreadGroup solo{0, 1, group.group_id, 1, SyncScope::THREAD};
        put(solo,
            sendRecvState_.sendStagingBuf.subBuffer(chunk.stagingOff),
            sendRecvState_.recvStagingBuf.subBuffer(chunk.stagingOff),
            chunk.bytes,
            sendRecvState_.remoteSignalBuf.subBuffer(
                state.groupId * sizeof(uint64_t)),
            chunk.bytes,
            sendRecvState_.localCounterBuf.subBuffer(
                state.groupId * sizeof(uint64_t)),
            chunk.bytes);
      }
      group.sync();

      state.nextByte += chunk.bytes;
      if (state.nextByte >= state.nbytes) {
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
    if (state.stage != initialStage || state.nextByte != initialNextByte) {
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
   * means the reserved byte range has completed.
   *
   * `CopyOp` must expose `recv(dst, src, bytes, group, dataOffset, args...)`.
   * The default `Memcpy` copies bytes cooperatively across the supplied
   * `ThreadGroup`; custom copy ops may use `args` to pass reduction or
   * conversion context.
   *
   * @param group Thread group matching the one used during initialization.
   * @param dst Destination user buffer. The range `[dst, dst + nbytes)` must
   *            remain valid until `Done`.
   * @param timeout Optional device timeout checked while dependencies wait.
   * @param args Additional arguments forwarded to `CopyOp::recv`.
   */
  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ IbgdaSendRecvProgressStatus progress_recv_once(
      ThreadGroup& group,
      void* __restrict__ dst,
      const Timeout& timeout = Timeout(),
      Args... args) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef __HIP_PLATFORM_AMD__
    static_assert(
        sizeof(CopyOp) == 0,
        "P2pIbgdaTransportDevice::progress_recv_once() requires NVIDIA GPU");
#endif
    const int progressIndex = progress_recv_index(group);
    detail::IbSendRecvProgressState state =
        progress_state_slot(group, progressIndex);
    if (state.stage == detail::IbSendRecvProgressStage::Done) {
      return IbgdaSendRecvProgressStatus::Done;
    }
    if (state.nextByte >= state.nbytes) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: progress_recv_once nextByte=%llu >= nbytes=%llu "
            "without Done stage\n",
            static_cast<unsigned long long>(state.nextByte),
            static_cast<unsigned long long>(state.nbytes));
      }
      PIPES_DEVICE_TRAP();
    }
    validate_recv_progress_stage(group, state);

    const ProgressChunk chunk = next_chunk(state);
    const uint64_t expected = chunk.streamEnd;
    uint32_t ready = 1;
    unsigned long long current = 0;
    if (group.is_leader()) {
      current = static_cast<unsigned long long>(
          read_signal(sendRecvState_.localSignalBuf.subBuffer(
              state.groupId * sizeof(uint64_t))));
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

    CopyOp::recv(
        static_cast<char*>(dst) + chunk.dataOff,
        sendRecvState_.recvStagingPtr + chunk.stagingOff,
        chunk.bytes,
        group,
        chunk.dataOff,
        args...);
    group.sync();

    signal(
        group,
        sendRecvState_.remoteSignalBuf.subBuffer(
            (sendRecvState_.maxGroups + state.groupId) * sizeof(uint64_t)),
        chunk.bytes);

    state.nextByte += chunk.bytes;
    if (state.nextByte >= state.nbytes) {
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
   * stepState persists across calls, so send() resumes the staging-ring cursor
   * and protocol sequence numbers on each invocation. This allows callers to
   * pipeline across repeated send() calls without a separate drain.
   *
   * The caller must keep the staging layout stable while a sequence is in
   * flight. Changing active_blocks changes the per-block staging partition, so
   * both sides must perform a higher-level barrier/quiescence step first.
   * max_signal_bytes may vary across calls with the same active_blocks.
   *
   * @param group           ThreadGroup (all threads participate in memcpy,
   *                        leader does RDMA ops).
   * @param src             Source data for this block's tile.
   * @param nbytes          Bytes to send for this group. Internally consumed
   *                        in perBlockSlot-sized pieces, or smaller sub-chunks
   *                        when max_signal_bytes is set.
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
    if (nbytes == 0) {
      return;
    }

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
    const uint64_t baseByte =
        static_cast<uint64_t>(sendRecvState_.stepState[groupId]);
    const std::size_t pipelineBytes = perBlockSlot * pipelineDepth;

    for (std::size_t dataOff = 0; dataOff < nbytes;) {
      const uint64_t streamStart = baseByte + dataOff;
      const std::size_t pipelineOff =
          static_cast<std::size_t>(streamStart % pipelineBytes);
      const int slot = static_cast<int>(pipelineOff / perBlockSlot);
      const std::size_t slotOff = slot * dataBufferSize;
      const std::size_t chunkOff = pipelineOff - slot * perBlockSlot;
      const std::size_t slotRemaining = perBlockSlot - chunkOff;
      const std::size_t dataRemaining = nbytes - dataOff;
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
      CopyOp::send(
          sendRecvState_.sendStagingPtr + stagingOff,
          static_cast<const char*>(src) + dataOff,
          bytesThis,
          group,
          dataOff,
          args...);
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
        ThreadGroup solo{0, 1, group.group_id, 1, SyncScope::THREAD};
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
      sendRecvState_.stepState[groupId] =
          static_cast<int64_t>(baseByte + nbytes);
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
   * @param nbytes          Bytes to receive for this group. Internally
   *                        consumed in perBlockSlot-sized pieces, or smaller
   *                        sub-chunks when max_signal_bytes is set.
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
    if (nbytes == 0) {
      return;
    }

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
    const uint64_t baseByte = static_cast<uint64_t>(
        sendRecvState_.stepState[sendRecvState_.maxGroups + groupId]);
    const std::size_t pipelineBytes = perBlockSlot * pipelineDepth;

    for (std::size_t dataOff = 0; dataOff < nbytes;) {
      const uint64_t streamStart = baseByte + dataOff;
      const std::size_t pipelineOff =
          static_cast<std::size_t>(streamStart % pipelineBytes);
      const int slot = static_cast<int>(pipelineOff / perBlockSlot);
      const std::size_t slotOff = slot * dataBufferSize;
      const std::size_t chunkOff = pipelineOff - slot * perBlockSlot;
      const std::size_t slotRemaining = perBlockSlot - chunkOff;
      const std::size_t dataRemaining = nbytes - dataOff;
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
      CopyOp::recv(
          static_cast<char*>(dst) + dataOff,
          sendRecvState_.recvStagingPtr + stagingOff,
          bytesThis,
          group,
          dataOff,
          args...);
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
      sendRecvState_.stepState[sendRecvState_.maxGroups + groupId] =
          static_cast<int64_t>(baseByte + nbytes);
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
   *     - Reads stepState[maxGroups + groupId] (same index as recv)
   *     - Waits DATA_READY on localSignalBuf[groupId] (matches send's
   *       piggybacked signal on remoteSignalBuf[groupId])
   *     - Signals SLOT_FREE on remoteSignalBuf[maxGroups + groupId]
   *       (matches send's backpressure wait on localSignalBuf[maxGroups +
   *       groupId])
   *
   *   Fwd side (fwd transport):
   *     - Reads stepState[groupId] (same index as send)
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
   * @param nbytes          Bytes to receive and forward.
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
#if PIPES_IS_DEVICE_COMPILE
#ifdef __HIP_PLATFORM_AMD__
    static_assert(
        false,
        "P2pIbgdaTransportDevice::forward() requires NVIDIA GPU (DOCA/IBGDA)");
#endif
    if (nbytes == 0) {
      return;
    }

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
    const uint64_t recvBaseByte = static_cast<uint64_t>(
        sendRecvState_.stepState[sendRecvState_.maxGroups + groupId]);
    const std::size_t recvPipelineBytes = recvPerBlockSlot * recvPipelineDepth;

    const int fwdPipelineDepth = fwd.sendRecvState_.pipelineDepth;
    const std::size_t fwdDataBufSize = fwd.sendRecvState_.dataBufferSize;
    const int fwdMaxGroups = fwd.sendRecvState_.maxGroups;
    const uint64_t fwdBaseByte =
        static_cast<uint64_t>(fwd.sendRecvState_.stepState[groupId]);
    const std::size_t fwdPipelineBytes = fwdPerBlockSlot * fwdPipelineDepth;

    for (std::size_t dataOff = 0; dataOff < nbytes;) {
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
      const std::size_t dataRemaining = nbytes - dataOff;
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
      CopyOp::forward(
          dst ? static_cast<char*>(dst) + dataOff : nullptr,
          fwd.sendRecvState_.sendStagingPtr + fwdStagingOff,
          sendRecvState_.recvStagingPtr + recvStagingOff,
          bytesThis,
          group,
          dataOff,
          args...);
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
        ThreadGroup solo{0, 1, group.group_id, 1, SyncScope::THREAD};
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

    // Update step state for both recv and fwd sides.
    if (group.is_leader()) {
      sendRecvState_.stepState[sendRecvState_.maxGroups + groupId] =
          static_cast<int64_t>(recvBaseByte + nbytes);
      fwd.sendRecvState_.stepState[groupId] =
          static_cast<int64_t>(fwdBaseByte + nbytes);
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
   * buffers. `dataOff` is the matching offset into the caller's user buffer.
   * `bytes` never crosses a per-block staging partition or the requested byte
   * count. `streamEnd` is the absolute protocol byte value after this chunk and
   * is used as the DATA_READY, SLOT_FREE, and NIC_DONE readiness threshold.
   */
  struct ProgressChunk {
    std::size_t stagingOff;
    std::size_t dataOff;
    std::size_t bytes;
    uint64_t streamEnd;
  };

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
    if (sendRecvState_.progressState == nullptr) {
      if (group.is_leader()) {
        printf("[PIPES] FATAL: progressState is null\n");
      }
      PIPES_DEVICE_TRAP();
    }
    if (sendRecvState_.maxGroups <= 0 ||
        group.group_id >= static_cast<uint32_t>(sendRecvState_.maxGroups)) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: progress group_id=%u out of range [0, %d)\n",
            group.group_id,
            sendRecvState_.maxGroups);
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
  __device__ __forceinline__ detail::IbSendRecvProgressState&
  progress_state_slot(ThreadGroup& group, int progressIndex) const {
    (void)group;
    return sendRecvState_.progressState[progressIndex];
  }

  /**
   * Trap if a caller tries to start a second send/recv before the first ends.
   */
  __device__ __forceinline__ void assert_progress_slot_idle(
      ThreadGroup& group,
      const detail::IbSendRecvProgressState& state,
      const char* direction) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (state.stage != detail::IbSendRecvProgressStage::Done) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: init_%s_progress called with outstanding %s "
            "for group_id=%u stage=%d nextByte=%llu nbytes=%llu\n",
            direction,
            direction,
            group.group_id,
            static_cast<int>(state.stage),
            static_cast<unsigned long long>(state.nextByte),
            static_cast<unsigned long long>(state.nbytes));
      }
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
   */
  __device__ __forceinline__ void store_progress_state(
      ThreadGroup& group,
      int progressIndex,
      const detail::IbSendRecvProgressState& state) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (group.is_leader()) {
      sendRecvState_.progressState[progressIndex] = state;
    }
    group.sync();
#else
    (void)group;
    (void)progressIndex;
    (void)state;
#endif
  }

  /**
   * Validate and finalize the staging geometry stored in a progress state.
   *
   * This helper is called by the public init APIs before the internal state is
   * committed to the transport. It verifies that the state belongs to the
   * current `ThreadGroup`, that the group participates in this transfer, and
   * that the requested active block count can fit at least one 16-byte-aligned
   * region in each staging slot.
   *
   * On success, `perBlockSlot` is the caller's partition within each logical
   * staging slot and `chunkSize` is the maximum signaled sub-chunk size. A
   * zero `maxSignalBytes` means one chunk per `perBlockSlot`; otherwise the
   * value is rounded down to the same 16-byte alignment used by the blocking
   * send/recv path. Invalid geometry traps on device because continuing would
   * corrupt another block group's staging partition.
   *
   * The public init methods handle zero-byte operations before calling this
   * helper. This helper is therefore only responsible for valid, non-empty
   * transfer geometry.
   */
  __device__ __forceinline__ void init_progress_geometry(
      ThreadGroup& group,
      detail::IbSendRecvProgressState& state) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (state.groupId != static_cast<int>(group.group_id)) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: progress state group_id=%d but group.group_id=%u\n",
            state.groupId,
            group.group_id);
      }
      PIPES_DEVICE_TRAP();
    }
    if (state.activeBlocks > sendRecvState_.maxGroups) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: progress active_blocks=%d > maxGroups=%d\n",
            state.activeBlocks,
            sendRecvState_.maxGroups);
      }
      PIPES_DEVICE_TRAP();
    }
    if (state.groupId < 0 || state.groupId >= state.activeBlocks) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: progress group_id=%d >= active_blocks=%d\n",
            state.groupId,
            state.activeBlocks);
      }
      PIPES_DEVICE_TRAP();
    }

    state.perBlockSlot =
        (sendRecvState_.dataBufferSize / state.activeBlocks) & ~15ULL;
    if (state.perBlockSlot == 0) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: progress perBlockSlot=0 "
            "(dataBufferSize=%llu, active_blocks=%d)\n",
            (unsigned long long)sendRecvState_.dataBufferSize,
            state.activeBlocks);
      }
      PIPES_DEVICE_TRAP();
    }

    state.chunkSize =
        (state.maxSignalBytes > 0 && state.maxSignalBytes < state.perBlockSlot)
        ? (state.maxSignalBytes & ~15ULL)
        : state.perBlockSlot;
    if (state.chunkSize == 0) {
      state.chunkSize = state.perBlockSlot;
    }
#endif
  }

  /**
   * Reserve a non-overlapping protocol byte range for one progress state.
   *
   * Blocking send()/recv() read `stepState` at call entry and commit it at
   * completion. Progress init cannot wait until completion because progress
   * operations may complete across many bounded calls. Reserving at init gives
   * the transport-owned state a stable byte-stream base and makes later
   * blocking calls start after all in-flight progress ranges.
   */
  __device__ __forceinline__ int64_t reserve_progress_step(
      ThreadGroup& group,
      int stepIndex,
      std::size_t nbytes) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    uint64_t baseStep = 0;
    if (group.is_leader()) {
      baseStep = static_cast<uint64_t>(sendRecvState_.stepState[stepIndex]);
      sendRecvState_.stepState[stepIndex] =
          static_cast<int64_t>(baseStep + nbytes);
    }
    baseStep = group.broadcast<uint64_t>(baseStep);
    return static_cast<int64_t>(baseStep);
#else
    (void)group;
    (void)stepIndex;
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
      const detail::IbSendRecvProgressState& state) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (state.stage != detail::IbSendRecvProgressStage::WaitNicDone &&
        state.stage != detail::IbSendRecvProgressStage::WaitSlotFree &&
        state.stage != detail::IbSendRecvProgressStage::Done) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: progress_send_once invalid stage=%d\n",
            static_cast<int>(state.stage));
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
      const detail::IbSendRecvProgressState& state) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (state.stage != detail::IbSendRecvProgressStage::WaitDataReady &&
        state.stage != detail::IbSendRecvProgressStage::Done) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: progress_recv_once invalid stage=%d\n",
            static_cast<int>(state.stage));
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
      detail::IbSendRecvProgressState& state,
      detail::IbSendRecvProgressStage next) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const detail::IbSendRecvProgressStage current = state.stage;
    if (!is_valid_progress_transition(current, next)) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: invalid progress transition stage=%d -> %d\n",
            static_cast<int>(current),
            static_cast<int>(next));
      }
      PIPES_DEVICE_TRAP();
    }
    state.stage = next;
#endif
  }

  /**
   * Map the state's logical byte cursor to the next staging-ring chunk.
   *
   * The transport stores each logical slot as `dataBufferSize` bytes, split
   * into `activeBlocks` per-group partitions. The protocol cursor advances in
   * bytes, not slots, so `baseStep + nextByte` is reduced modulo
   * `perBlockSlot * pipelineDepth` to pick the ring slot and per-group offset.
   *
   * The returned chunk is clipped by three boundaries: the configured
   * sub-chunk size, remaining user bytes, and remaining bytes in the current
   * per-block staging partition. This keeps every progress call bounded and
   * prevents a single RDMA put or recv copy from spanning two staging slots.
   */
  __device__ __forceinline__ ProgressChunk
  next_chunk(const detail::IbSendRecvProgressState& state) const {
    const uint64_t streamStart =
        static_cast<uint64_t>(state.baseStep) + state.nextByte;
    const std::size_t pipelineBytes = state.perBlockSlot *
        static_cast<std::size_t>(sendRecvState_.pipelineDepth);
    const std::size_t pipelineOff =
        static_cast<std::size_t>(streamStart % pipelineBytes);
    const int slot = static_cast<int>(pipelineOff / state.perBlockSlot);
    const std::size_t slotOff =
        static_cast<std::size_t>(slot) * sendRecvState_.dataBufferSize;
    const std::size_t chunkOff =
        pipelineOff - static_cast<std::size_t>(slot) * state.perBlockSlot;
    const std::size_t slotRemaining = state.perBlockSlot - chunkOff;
    const std::size_t dataRemaining = state.nbytes - state.nextByte;
    std::size_t bytes =
        state.chunkSize < dataRemaining ? state.chunkSize : dataRemaining;
    bytes = bytes < slotRemaining ? bytes : slotRemaining;
    return ProgressChunk{
        .stagingOff = slotOff +
            static_cast<std::size_t>(state.groupId) * state.perBlockSlot +
            chunkOff,
        .dataOff = state.nextByte,
        .bytes = bytes,
        .streamEnd = streamStart + bytes,
    };
  }

  struct NicQpIndex {
    int nic_id;
    int qp_id;
  };

  /**
   * nic_qp_for_group - Single lookup: returns NIC/QP ids.
   *
   * Round-robin over NIC resources, then within the chosen NIC round-robin over
   * its QPs.
   */
  __device__ NicQpIndex nic_qp_for_group(uint32_t group_id) const {
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
    int nic_id = group_id % nicDevices_.size();
    const auto& qps = nicDevices_[nic_id].qps;
    if (qps.empty()) {
      printf(
          "P2pIbgdaTransportDevice: NIC %d has no qps at "
          "%s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",
          nic_id,
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
    int qp_id = (group_id / nicDevices_.size()) % qps.size();
    return {nic_id, qp_id};
  }

  __device__ doca_gpu_dev_verbs_qp* active_qp(uint32_t group_id) const {
    auto idx = nic_qp_for_group(group_id);
    return nicDevices_[idx.nic_id].qps[idx.qp_id];
  }

  __device__ doca_gpu_dev_verbs_qp* active_companion_qp(
      uint32_t group_id) const {
    auto idx = nic_qp_for_group(group_id);
    return nicDevices_[idx.nic_id].companion_qps[idx.qp_id];
  }

  // --- Members ---
  // Per-NIC bundles (qps + companion_qps + sink_lkey + device_id).
  DeviceSpan<NicDeviceIbgdaResources> nicDevices_{};

  // Owned signal/counter buffers (set by transport during construction)
  IbgdaRemoteBuffer ownedRemoteSignalBuf_{}; // outbox: signal peer's inbox
  IbgdaLocalBuffer ownedLocalSignalBuf_{}; // inbox: receive signals from peers
  IbgdaLocalBuffer ownedCounterBuf_{}; // local counter for companion QP

  IbgdaRemoteBuffer discardSignalSlot_{};

  int numSignalSlots_{0};
  int numCounterSlots_{0};

  IbSendRecvState sendRecvState_{};
};

} // namespace comms::pipes
