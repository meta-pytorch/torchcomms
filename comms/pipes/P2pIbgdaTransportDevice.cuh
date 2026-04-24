// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>

#include <device/doca_gpunetio_dev_verbs_counter.cuh>
#include <device/doca_gpunetio_dev_verbs_onesided.cuh>

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/DocaVerbsUtils.cuh"
#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {

inline constexpr uint64_t kDefaultDeviceTimeoutCycles = 10'000'000'000ULL;

// Slot-id bounds checks for the slot-index API. Catches both
// out-of-range slot ids and slot-index calls made when the transport was
// constructed with no owned signal/counter buffer (numSlots == 0).
#ifdef __CUDA_ARCH__
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
      __trap();                                         \
    }                                                   \
  } while (0)
#else
#define IBGDA_CHECK_SLOT_ID(id, count, kind) assert((id) >= 0 && (id) < (count))
#endif

/**
 * P2pIbgdaTransportDevice - Device-side per-peer RDMA transport handle
 *
 * Every method has two overloads:
 *   Group-scope: put(group, ...) — all threads in group must call.
 *     QP selection: single QP for now (multi-QP via group.group_id % numQps
 *     will be added in a follow-up diff).
 *     Data transfer is group-cooperative (threads split WQE construction).
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
  // call methods on a default-constructed instance — qpArray_ is null.
  P2pIbgdaTransportDevice() = default;

  /**
   * Construct a per-peer device transport handle.
   *
   * @param qpArray               GPU array of N primary QP pointers. WQEs for
   *                              a block/group are posted to
   *                              qpArray[group_id % numQps].
   * @param companionArray        GPU array of N companion QP pointers for
   *                              compound put+signal+counter ops. May be
   *                              nullptr if counters are not used.
   * @param numQps                Number of QPs in qpArray/companionArray.
   * @param sinkLkey              LKey of a scratch buffer used as the atomic
   *                              fetch-add response sink (value is discarded).
   * @param ownedRemoteSignalBuf  Remote-side signal outbox: writing here
   *                              targets the peer's local signal inbox. Used
   *                              by the slot-index signal API.
   * @param ownedLocalSignalBuf   Local signal inbox: receives signals from
   *                              the peer. Used by the slot-index
   *                              wait_signal/reset_signal/read_signal APIs.
   * @param ownedCounterBuf       Local counter buffer for compound
   *                              put+signal+counter and the slot-index
   *                              counter APIs. May be empty if not used.
   * @param discardSignalSlot     Remote uint64_t slot used as a "throwaway"
   *                              signal target for counter-only puts (see
   *                              put_impl for rationale). The peer never
   *                              reads this slot. Required when counter is
   *                              used; ignored otherwise.
   * @param numSignalSlots        Number of uint64_t slots in the owned signal
   *                              buffers. Used to bounds-check signalId
   *                              args. Zero disables the slot-index signal
   *                              API.
   * @param numCounterSlots       Number of uint64_t slots in the owned
   *                              counter buffer. Zero disables the
   *                              slot-index counter API.
   * @param sendRecvState         Optional pipelined send/recv protocol state.
   *                              When empty, send()/recv() are unavailable.
   */
  __host__ __device__ P2pIbgdaTransportDevice(
      DeviceSpan<doca_gpu_dev_verbs_qp*> qpArray,
      DeviceSpan<doca_gpu_dev_verbs_qp*> companionArray,
      NetworkLKey sinkLkey = NetworkLKey{},
      IbgdaRemoteBuffer ownedRemoteSignalBuf = {},
      IbgdaLocalBuffer ownedLocalSignalBuf = {},
      IbgdaLocalBuffer ownedCounterBuf = {},
      int numSignalSlots = 0,
      int numCounterSlots = 0,
      IbgdaRemoteBuffer discardSignalSlot = {},
      IbSendRecvState sendRecvState = {})
      : qpArray_(qpArray),
        companionArray_(companionArray),
        sinkLkey_(sinkLkey),
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
   *                 then __threadfence_system().
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
   * read_signal (slot-index) - Non-blocking volatile read of a local signal
   * inbox slot.
   *
   * @param signalId Slot index into the local signal inbox.
   * @return         Current value of the slot.
   */
  __device__ uint64_t read_signal(int signalId) const {
    return read_signal(local_signal_slot(signalId));
  }

  /**
   * read_counter (slot-index) - Non-blocking volatile read of a local counter
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
   * put (group-scope) - Group-cooperative RDMA Write with optional signal /
   * counter.
   *
   * All threads in the group must call. Data transfer adapts to group size:
   *   group_size == 1: single thread posts one WQE
   *   group_size > 1: threads cooperatively construct WQEs (one per thread)
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
   *                   atomic. Counter-only: fence + GPU atomicAdd.
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
   *                  then __threadfence_system().
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
   * read_signal - Non-blocking volatile read of a local signal slot.
   *
   * @param signalBuf Pre-resolved local signal slot.
   * @return          Current value of *signalBuf.
   */
  __device__ uint64_t read_signal(const IbgdaLocalBuffer& signalBuf) const {
    volatile uint64_t* sig = static_cast<volatile uint64_t*>(signalBuf.ptr);
    return *sig;
  }

  /**
   * read_counter - Non-blocking volatile read of a local counter slot.
   *
   * @param counterBuf Pre-resolved local counter slot.
   * @return           Current value of *counterBuf.
   */
  __device__ uint64_t read_counter(const IbgdaLocalBuffer& counterBuf) const {
    volatile uint64_t* ctr = static_cast<volatile uint64_t*>(counterBuf.ptr);
    return *ctr;
  }

  // =========================================================================
  // Private: _impl methods + internal building blocks
  // =========================================================================

 private:
  // --- put_impl: always group-cooperative data transfer ---

  __device__ void put_impl(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal,
      const IbgdaLocalBuffer& counterBuf,
      uint64_t counterVal) {
    bool hasSignal = signalBuf.ptr != nullptr;
    bool hasCounter = counterBuf.ptr != nullptr;

    // Step 1: ALWAYS group-cooperative data transfer
    put_cooperative(group, localBuf, remoteBuf, nbytes);

    // Step 2: Leader posts signal/counter WQE(s).
    //
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

  // --- wait_signal_impl ---
  //
  // The trailing __threadfence_system() is the standard "acquire fence after
  // observing a flag": it ensures payload writes (e.g. data the NIC RDMA'd
  // alongside the signal) are visible to subsequent loads on this thread.

  __device__ void wait_signal_impl(
      ThreadGroup& group,
      const IbgdaLocalBuffer& signalBuf,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    if (group.is_leader()) {
      volatile uint64_t* sig = static_cast<volatile uint64_t*>(signalBuf.ptr);
      while (*sig < expected) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "wait_signal: expected>=%llu, current=%llu",
            static_cast<unsigned long long>(expected),
            static_cast<unsigned long long>(*sig));
      }
      __threadfence_system();
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
      volatile uint64_t* ctr = static_cast<volatile uint64_t*>(counterBuf.ptr);
      while (*ctr < expected) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "wait_counter: expected>=%llu, current=%llu",
            static_cast<unsigned long long>(expected),
            static_cast<unsigned long long>(*ctr));
      }
      __threadfence_system();
    }
    group.sync();
  }

  // --- reset_local_impl: zero a local 64-bit slot ---
  //
  // The volatile store + group.sync() is sufficient for intra-group ordering.
  // __threadfence() (device scope) is a cheap belt-and-suspenders so that
  // threads in OTHER blocks observing the slot via volatile reads see the
  // reset. We deliberately do NOT use __threadfence_system() here: nothing
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

  // --- put_cooperative: group-collaborative WQE construction ---

  __device__ void put_cooperative(
      ThreadGroup& group,
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

    if (group.group_size == 1) {
      put_single_impl(group.group_id, laneBuf, laneRemoteBuf, laneBytes);
      return;
    }

    // Guard: group_size must fit within QP send queue depth
    if (group.is_leader()) {
      const uint16_t qp_depth = __ldg(&active_qp(group.group_id)->sq_wqe_num);
      if (group.group_size > qp_depth) {
        printf(
            "[PIPES] FATAL: put group_size (%u) > QP depth (%u). "
            "Set NCCL_CTRAN_IBGDA_QP_DEPTH >= %u to avoid deadlock.\n",
            group.group_size,
            qp_depth,
            group.group_size);
        __trap();
      }
    }

    // Leader reserves WQE slots for all threads
    uint64_t base_wqe_idx = 0;
    if (group.is_leader()) {
      base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
          active_qp(group.group_id), group.group_size);
    }
    base_wqe_idx = group.broadcast<uint64_t>(base_wqe_idx);

    // Each thread prepares its WQE
    uint64_t wqe_idx = base_wqe_idx + group.thread_id_in_group;
    struct doca_gpu_dev_verbs_wqe* wqe_ptr =
        doca_gpu_dev_verbs_get_wqe_ptr(active_qp(group.group_id), wqe_idx);

    doca_gpu_dev_verbs_wqe_prepare_write(
        active_qp(group.group_id),
        wqe_ptr,
        static_cast<uint16_t>(wqe_idx),
        DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        0,
        reinterpret_cast<uint64_t>(laneRemoteBuf.ptr),
        laneRemoteBuf.rkey.value,
        reinterpret_cast<uint64_t>(laneBuf.ptr),
        laneBuf.lkey.value,
        static_cast<uint32_t>(laneBytes));

    group.sync();

    // Leader marks ready and rings doorbell
    if (group.is_leader()) {
      doca_gpu_dev_verbs_mark_wqes_ready<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
          active_qp(group.group_id),
          base_wqe_idx,
          base_wqe_idx + group.group_size - 1);
      doca_gpu_dev_verbs_submit<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
          DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
          active_qp(group.group_id), base_wqe_idx + group.group_size);
    }

    group.sync();
  }

  // --- put_single_impl: one thread, one WQE ---

  __device__ void put_single_impl(
      uint32_t group_id,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes) {
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_addr localAddr = {
        .addr = reinterpret_cast<uint64_t>(localBuf.ptr),
        .key = localBuf.lkey.value};
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(remoteBuf.ptr),
        .key = remoteBuf.rkey.value};

    doca_gpu_dev_verbs_put<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
        DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>(
        active_qp(group_id), remoteAddr, localAddr, nbytes, &ticket);
  }

  // --- signal_fenced: atomic fetch-add with NIC FENCE (always fenced) ---

  __device__ void signal_fenced(
      uint32_t group_id,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal) {
    doca_gpu_dev_verbs_qp* qp = active_qp(group_id);
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(signalBuf.ptr),
        .key = signalBuf.rkey.value};
    doca_gpu_dev_verbs_addr sinkAddr = {.addr = 0, .key = sinkLkey_.value};

    uint64_t wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, 1);

    struct doca_gpu_dev_verbs_wqe* wqe_ptr =
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
    doca_gpu_dev_verbs_addr sigRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(signalBuf.ptr),
        .key = signalBuf.rkey.value};
    doca_gpu_dev_verbs_addr sigSinkAddr = {.addr = 0, .key = sinkLkey_.value};

    doca_gpu_dev_verbs_addr counterRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(counterBuf.ptr),
        .key = counterBuf.lkey.value};
    doca_gpu_dev_verbs_addr counterSinkAddr = {
        .addr = 0, .key = sinkLkey_.value};

    doca_gpu_dev_verbs_signal_counter<
        DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD,
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
        active_qp(group_id),
        sigRemoteAddr,
        sigSinkAddr,
        signalVal,
        active_companion_qp(group_id),
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
        ownedRemoteSignalBuf_.rkey);
  }

  __device__ IbgdaLocalBuffer local_signal_slot(int id) const {
    IBGDA_CHECK_SLOT_ID(id, numSignalSlots_, "signal");
    return IbgdaLocalBuffer(
        static_cast<uint64_t*>(ownedLocalSignalBuf_.ptr) + id,
        ownedLocalSignalBuf_.lkey);
  }

  __device__ IbgdaLocalBuffer counter_slot(int id) const {
    IBGDA_CHECK_SLOT_ID(id, numCounterSlots_, "counter");
    return IbgdaLocalBuffer(
        static_cast<uint64_t*>(ownedCounterBuf_.ptr) + id,
        ownedCounterBuf_.lkey);
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
  //   sub-chunk        = one signaled piece within a perBlockSlot. When
  //                      max_signal_bytes == 0, a sub-chunk is the whole
  //                      perBlockSlot. Otherwise:
  //                        chunkSize = floor16(min(perBlockSlot,
  //                                             max_signal_bytes))
  //                        chunksPerSlot = perBlockSlot / chunkSize
  //   stepState        = persistent sub-chunk cursor. It advances by one per
  //                      DATA_READY / SLOT_FREE / NIC_DONE event, not one per
  //                      whole slot.
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
   *   SLOT_FREE  — receiver increments per sub-chunk (symmetric with
   *                DATA_READY). send waits before overwriting recvStaging.
   *   DATA_READY — sender increments per sub-chunk, piggybacked on put.
   *                recv waits on this before reading recvStaging.
   *
   * stepState persists across calls, so send() resumes the staging-ring cursor
   * and protocol sequence numbers on each invocation. This allows callers to
   * pipeline across repeated send() calls without a separate drain.
   *
   * The caller must keep the staging layout stable while a sequence is in
   * flight. If active_blocks, max_signal_bytes, or any other parameter that
   * changes the slot/sub-chunk mapping is modified, both sides must perform a
   * higher-level barrier/quiescence step before issuing the next send()/recv().
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
  __device__ __forceinline__ void send(
      ThreadGroup& group,
      void* __restrict__ src,
      std::size_t nbytes,
      int active_blocks = 0,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout()) {
#ifndef __CUDA_ARCH__
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
      __trap();
    }
    if (groupId >= effActive) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: send group_id=%u >= active_blocks=%d\n",
            groupId,
            effActive);
      }
      __trap();
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
      __trap();
    }

    std::size_t chunkSize =
        (max_signal_bytes > 0 && max_signal_bytes < perBlockSlot)
        ? (max_signal_bytes & ~15ULL)
        : perBlockSlot;
    if (chunkSize == 0) {
      chunkSize = perBlockSlot;
    }
    const std::size_t chunksPerSlot = perBlockSlot / chunkSize;
    const std::size_t totalChunks = (nbytes + chunkSize - 1) / chunkSize;

    const int64_t baseStep = sendRecvState_.stepState[groupId];
    const int pipelineDepth = sendRecvState_.pipelineDepth;
    const std::size_t dataBufferSize = sendRecvState_.dataBufferSize;
    const int maxGroups = sendRecvState_.maxGroups;
    const int64_t chunksPerSlot64 = static_cast<int64_t>(chunksPerSlot);
    const int64_t pipelineChunks =
        static_cast<int64_t>(pipelineDepth) * chunksPerSlot64;

    for (std::size_t s = 0; s < totalChunks; ++s) {
      const int64_t chunkStep = baseStep + static_cast<int64_t>(s);
      const int64_t slotStep = chunkStep / chunksPerSlot64;
      const int64_t subStep = chunkStep % chunksPerSlot64;
      const int slot = static_cast<int>(slotStep % pipelineDepth);
      const std::size_t slotOff = slot * dataBufferSize;
      const std::size_t chunkOff =
          static_cast<std::size_t>(subStep) * chunkSize;
      const std::size_t stagingOff =
          slotOff + groupId * perBlockSlot + chunkOff;
      const std::size_t dataOff = s * chunkSize;
      const std::size_t bytesThis =
          (dataOff + chunkSize <= nbytes) ? chunkSize : (nbytes - dataOff);

      // (1) Wait for NIC to finish with this slot's local sendStaging.
      if (chunkStep >= pipelineChunks) {
        wait_counter(
            group,
            sendRecvState_.localCounterBuf.subBuffer(
                groupId * sizeof(uint64_t)),
            static_cast<uint64_t>(chunkStep - pipelineChunks + 1),
            timeout);
      }

      // (2) Cooperative memcpy: src → local sendStaging.
      memcpy_vectorized(
          sendRecvState_.sendStagingPtr + stagingOff,
          static_cast<char*>(src) + dataOff,
          bytesThis,
          group);
      group.sync();

      // (3) Backpressure: wait for receiver to free this sub-chunk's
      //     recvStaging offset. Symmetric with DATA_READY (per sub-chunk).
      if (chunkStep >= pipelineChunks) {
        wait_signal(
            group,
            sendRecvState_.localSignalBuf.subBuffer(
                (maxGroups + groupId) * sizeof(uint64_t)),
            static_cast<uint64_t>(chunkStep - pipelineChunks + 1),
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
            1ULL,
            sendRecvState_.localCounterBuf.subBuffer(
                groupId * sizeof(uint64_t)),
            1ULL);
      }
      group.sync();
    }

    if (group.is_leader()) {
      sendRecvState_.stepState[groupId] =
          baseStep + static_cast<int64_t>(totalChunks);
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
   *   DATA_READY — sender increments per sub-chunk after RDMA put completes.
   *                recv waits on this before copying from recvStaging.
   *   SLOT_FREE  — recv increments per sub-chunk (symmetric with DATA_READY)
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
  __device__ __forceinline__ void recv(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      int active_blocks = 0,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout()) {
#ifndef __CUDA_ARCH__
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
      __trap();
    }
    if (groupId >= effActive) {
      if (group.is_leader()) {
        printf(
            "[PIPES] FATAL: recv group_id=%u >= active_blocks=%d\n",
            groupId,
            effActive);
      }
      __trap();
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
      __trap();
    }

    std::size_t chunkSize =
        (max_signal_bytes > 0 && max_signal_bytes < perBlockSlot)
        ? (max_signal_bytes & ~15ULL)
        : perBlockSlot;
    if (chunkSize == 0) {
      chunkSize = perBlockSlot;
    }
    const std::size_t chunksPerSlot = perBlockSlot / chunkSize;
    const std::size_t totalChunks = (nbytes + chunkSize - 1) / chunkSize;

    const int64_t baseStep =
        sendRecvState_.stepState[sendRecvState_.maxGroups + groupId];
    const int pipelineDepth = sendRecvState_.pipelineDepth;
    const std::size_t dataBufferSize = sendRecvState_.dataBufferSize;
    const int maxGroups = sendRecvState_.maxGroups;
    const int64_t chunksPerSlot64 = static_cast<int64_t>(chunksPerSlot);

    for (std::size_t s = 0; s < totalChunks; ++s) {
      const int64_t chunkStep = baseStep + static_cast<int64_t>(s);
      const int64_t slotStep = chunkStep / chunksPerSlot64;
      const int64_t subStep = chunkStep % chunksPerSlot64;
      const int slot = static_cast<int>(slotStep % pipelineDepth);
      const std::size_t slotOff = slot * dataBufferSize;
      const std::size_t chunkOff =
          static_cast<std::size_t>(subStep) * chunkSize;
      const std::size_t stagingOff =
          slotOff + groupId * perBlockSlot + chunkOff;
      const std::size_t dataOff = s * chunkSize;
      const std::size_t bytesThis =
          (dataOff + chunkSize <= nbytes) ? chunkSize : (nbytes - dataOff);

      // (1) Wait for sender's DATA_READY signal.
      wait_signal(
          group,
          sendRecvState_.localSignalBuf.subBuffer(groupId * sizeof(uint64_t)),
          static_cast<uint64_t>(chunkStep + 1),
          timeout);

      // (2) Cooperative memcpy: local recvStaging → dst.
      memcpy_vectorized(
          static_cast<char*>(dst) + dataOff,
          sendRecvState_.recvStagingPtr + stagingOff,
          bytesThis,
          group);
      group.sync();

      // (3) Signal SLOT_FREE to sender — per sub-chunk (symmetric with
      //     DATA_READY). Sender waits per sub-chunk before reusing remote
      //     recvStaging at the same offset.
      signal(
          group,
          sendRecvState_.remoteSignalBuf.subBuffer(
              (maxGroups + groupId) * sizeof(uint64_t)),
          1ULL);
    }

    if (group.is_leader()) {
      sendRecvState_.stepState[sendRecvState_.maxGroups + groupId] =
          baseStep + static_cast<int64_t>(totalChunks);
    }
    group.sync();
#endif
  }

  // Send/recv state accessors

  __host__ __device__ const IbSendRecvState& send_recv_state() const {
    return sendRecvState_;
  }

 private:
  /**
   * active_qp - Select the QP for the calling group
   *
   * Maps group_id → QP via group_id % numQps_. Every leaf helper takes a
   * group_id argument so that all WQEs belonging to one logical operation
   * (e.g. put + signal_fenced inside put_impl) land on the same QP — this
   * is required for the FENCE bit to actually order them. Thread-scope
   * public methods pass 0.
   */
  __device__ doca_gpu_dev_verbs_qp* active_qp(uint32_t group_id) const {
    if (qpArray_.empty()) {
      return nullptr;
    }
    return qpArray_[group_id % qpArray_.size()];
  }

  __device__ doca_gpu_dev_verbs_qp* active_companion_qp(
      uint32_t group_id) const {
    if (companionArray_.empty()) {
      return nullptr;
    }
    return companionArray_[group_id % companionArray_.size()];
  }

  // --- Members ---
  DeviceSpan<doca_gpu_dev_verbs_qp*> qpArray_{};
  DeviceSpan<doca_gpu_dev_verbs_qp*> companionArray_{};
  NetworkLKey sinkLkey_{};

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
