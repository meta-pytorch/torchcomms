// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>

#include <device/doca_gpunetio_dev_verbs_counter.cuh>
#include <device/doca_gpunetio_dev_verbs_onesided.cuh>

#include "comms/pipes/DocaVerbsUtils.cuh"
#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {

/**
 * IbgdaWork - Wrapper for DOCA GPU verbs operation handle
 *
 * Wraps the raw doca_gpu_dev_verbs_ticket_t to provide type safety
 * and a cleaner interface for tracking RDMA operation completion.
 *
 * The work handle represents a pending RDMA operation and can be used
 * with wait_local() to synchronize on local NIC completion.
 */
struct IbgdaWork {
  doca_gpu_dev_verbs_ticket_t value{0};

  IbgdaWork() = default;

  __device__ explicit IbgdaWork(doca_gpu_dev_verbs_ticket_t ticket)
      : value(ticket) {}
};

/**
 * P2pIbgdaTransportDevice - Device-side per-peer RDMA transport handle
 *
 * Provides GPU-initiated RDMA operations using DOCA GPUNetIO high-level APIs.
 * Each instance represents a connection to a single peer and contains a
 * GPU QP handle for issuing RDMA operations.
 *
 * EXECUTION SCOPE:
 * ================
 * Thread-Level APIs:
 *   put(), wait_local(), fence()
 *   - Each thread posts its own independent RDMA operation
 *   - Supports multi-chunk transfers (size >
 * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE)
 *
 * Group-Level APIs (public):
 *   Group-local (data already partitioned per group):
 *     put_group_local()
 *     - Accept a ThreadGroup and partition a single group's data chunk across
 *       group threads
 *     - All ThreadGroup sizes are supported (WARP, MULTIWARP, BLOCK, etc.)
 *     - group_size == 1: falls back to thread-level put()
 *     - group_size > 1: uses put_group_impl() with manual WQE construction
 *
 *   Group-global (data shared across all groups):
 *     put_group_global()
 *     - Accept a ThreadGroup and a global data buffer shared by all groups
 *     - First partitions data across groups (last group picks up remainder),
 *       then calls the group-local API on each group's chunk
 *
 * Private building blocks:
 *   put_group_impl()
 *   - Generic group-collaborative RDMA write using manual WQE construction
 *   - Works for any group size via low-level DOCA verbs APIs
 *   - Leader reserves WQE slots, broadcasts base index, all threads prepare
 *     WQEs, leader marks ready and rings doorbell
 */
// Default timeout for internal synchronous waits (e.g., reset_signal).
// 10 billion cycles ≈ 5-7 seconds on typical GPU clocks (~1.5-1.8 GHz).
inline constexpr uint64_t kDefaultDeviceTimeoutCycles = 10'000'000'000ULL;

class P2pIbgdaTransportDevice {
 public:
  P2pIbgdaTransportDevice() = default;

  /**
   * Constructor
   *
   * @param qp GPU QP handle for RDMA operations
   */
  __host__ __device__ P2pIbgdaTransportDevice(
      doca_gpu_dev_verbs_qp* qp,
      doca_gpu_dev_verbs_qp* companionQp = nullptr,
      NetworkLKey sinkLkey = NetworkLKey{})
      : qp_(qp), companionQp_(companionQp), sinkLkey_(sinkLkey) {}

  /**
   * put - RDMA Write without signal (non-blocking)
   *
   * Performs an RDMA Write from local buffer to remote buffer.
   * Returns immediately with a work handle for optional completion tracking.
   *
   * @param localBuf Source buffer in local GPU memory
   * @param remoteBuf Destination buffer in remote GPU memory
   * @param nbytes Number of bytes to transfer
   *
   * @return IbgdaWork for tracking local completion via wait_local()
   */
  __device__ IbgdaWork
  put(const IbgdaLocalBuffer& localBuf,
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
        qp_, remoteAddr, localAddr, nbytes, &ticket);

    return IbgdaWork(ticket);
  }

  /**
   * put_group_local - Group-collaborative RDMA Write (group-local data)
   *
   * Accepts a ThreadGroup and a single data chunk that belongs to this group,
   * partitions the data across group threads, and issues RDMA writes.
   *
   * All ThreadGroup sizes are supported:
   * - group_size == 1: falls back to thread-level put()
   * - group_size > 1: uses put_group_impl() with manual WQE construction
   *
   * REQUIREMENTS:
   * - All threads in the group must call this function collectively
   *
   * @param group ThreadGroup describing the calling group
   * @param localBuf Source buffer in local GPU memory (this group's chunk)
   * @param remoteBuf Destination buffer in remote GPU memory (this group's
   * chunk)
   * @param nbytes Number of bytes to transfer (partitioned across lanes)
   *
   * @return IbgdaWork for tracking local completion via wait_local() (per-lane)
   */
  __device__ IbgdaWork put_group_local(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes) {
    std::size_t chunkSize = nbytes / group.group_size;
    std::size_t offset = group.thread_id_in_group * chunkSize;
    // Last thread picks up any remainder bytes
    std::size_t laneBytes = (group.thread_id_in_group == group.group_size - 1)
        ? (nbytes - offset)
        : chunkSize;

    IbgdaLocalBuffer laneBuf = localBuf.subBuffer(offset);
    IbgdaRemoteBuffer laneRemoteBuf = remoteBuf.subBuffer(offset);

    if (group.group_size == 1) {
      return put(laneBuf, laneRemoteBuf, laneBytes);
    }
    return put_group_impl(group, laneBuf, laneRemoteBuf, laneBytes);
  }

  /**
   * put_group_global - Group-collaborative RDMA Write (global data)
   *
   * Accepts a ThreadGroup and a global data buffer shared by all groups.
   * Partitions the data across groups (last group picks up remainder),
   * then calls put_group_local() on each group's chunk.
   *
   * REQUIREMENTS:
   * - All threads in the group must call this function collectively
   *
   * @param group ThreadGroup describing the calling group
   * @param localBuf Source buffer in local GPU memory (global, shared by all
   *   groups)
   * @param remoteBuf Destination buffer in remote GPU memory (global, shared
   *   by all groups)
   * @param nbytes Total number of bytes to transfer (partitioned across groups,
   *   then across lanes within each group)
   *
   * @return IbgdaWork for tracking local completion via wait_local() (per-lane)
   */
  __device__ IbgdaWork put_group_global(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes) {
    // Partition across groups; last group picks up remainder
    std::size_t chunkPerGroup = nbytes / group.total_groups;
    std::size_t groupOffset = group.group_id * chunkPerGroup;
    std::size_t groupBytes = (group.group_id == group.total_groups - 1)
        ? (nbytes - groupOffset)
        : chunkPerGroup;

    IbgdaLocalBuffer groupLocalBuf = localBuf.subBuffer(groupOffset);
    IbgdaRemoteBuffer groupRemoteBuf = remoteBuf.subBuffer(groupOffset);

    return put_group_local(group, groupLocalBuf, groupRemoteBuf, groupBytes);
  }

  // ===========================================================================
  // Compound Put + Signal APIs (caller-provided signal buffers)
  // ===========================================================================

  /**
   * put_signal - RDMA Write with fenced atomic signal (adaptive routing safe)
   *
   * Compound operation: data write + fenced atomic signal in a single call.
   * The NIC fence ensures the data write completes before the signal is sent.
   * No wait_local() is needed between put and signal.
   *
   * @param localBuf Source buffer in local GPU memory
   * @param remoteBuf Destination buffer in remote GPU memory
   * @param nbytes Number of bytes to transfer
   * @param remoteSignalBuf Remote signal buffer (caller-owned)
   * @param signalId Index into the remote signal buffer (uint64_t units)
   * @param signalVal Value to atomically add to remote signal buffer
   * @return IbgdaWork for tracking signal completion via wait_local()
   */
  __device__ IbgdaWork put_signal(
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int signalId,
      uint64_t signalVal) {
    put(localBuf, remoteBuf, nbytes);
    return signal_remote_with_fence(remoteSignalBuf, signalId, signalVal);
  }

  /**
   * put_signal_group_local - Group-collaborative RDMA Write with fenced signal
   * (group-local data, adaptive routing safe)
   *
   * Partitions data across group threads, issues collaborative RDMA writes,
   * then the leader issues a fenced atomic signal. The signal ticket is
   * broadcast to all threads.
   *
   * @param group ThreadGroup describing the calling group
   * @param localBuf Source buffer in local GPU memory (this group's chunk)
   * @param remoteBuf Destination buffer in remote GPU memory (this group's
   * chunk)
   * @param nbytes Number of bytes to transfer (partitioned across lanes)
   * @param remoteSignalBuf Remote signal buffer (caller-owned)
   * @param signalId Index into the remote signal buffer
   * @param signalVal Value to atomically add to remote signal buffer
   * @return IbgdaWork for tracking signal completion via wait_local()
   */
  __device__ IbgdaWork put_signal_group_local(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int signalId,
      uint64_t signalVal) {
    std::size_t chunkSize = nbytes / group.group_size;
    std::size_t offset = group.thread_id_in_group * chunkSize;
    std::size_t laneBytes = (group.thread_id_in_group == group.group_size - 1)
        ? (nbytes - offset)
        : chunkSize;

    IbgdaLocalBuffer laneBuf = localBuf.subBuffer(offset);
    IbgdaRemoteBuffer laneRemoteBuf = remoteBuf.subBuffer(offset);

    if (group.group_size == 1) {
      put(laneBuf, laneRemoteBuf, laneBytes);
      return signal_remote_with_fence(remoteSignalBuf, signalId, signalVal);
    }

    put_group_impl(group, laneBuf, laneRemoteBuf, laneBytes);

    uint64_t signalTicket = 0;
    if (group.is_leader()) {
      IbgdaWork signalWork =
          signal_remote_with_fence(remoteSignalBuf, signalId, signalVal);
      signalTicket = signalWork.value;
    }
    signalTicket = group.broadcast<uint64_t>(signalTicket);
    return IbgdaWork(signalTicket);
  }

  /**
   * put_signal_group_global - Group-collaborative RDMA Write with fenced signal
   * (global data, adaptive routing safe)
   *
   * Partitions data across groups, then calls put_signal_group_local().
   * Each group issues an atomic fetch-add signal, so the total accumulated
   * signal is (total_groups * signalVal).
   *
   * @param group ThreadGroup describing the calling group
   * @param localBuf Source buffer (global, shared by all groups)
   * @param remoteBuf Destination buffer (global, shared by all groups)
   * @param nbytes Total bytes to transfer (partitioned across groups)
   * @param remoteSignalBuf Remote signal buffer (caller-owned)
   * @param signalId Index into the remote signal buffer
   * @param signalVal Value to atomically add per group
   * @return IbgdaWork for tracking signal completion via wait_local()
   */
  __device__ IbgdaWork put_signal_group_global(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int signalId,
      uint64_t signalVal) {
    std::size_t chunkPerGroup = nbytes / group.total_groups;
    std::size_t groupOffset = group.group_id * chunkPerGroup;
    std::size_t groupBytes = (group.group_id == group.total_groups - 1)
        ? (nbytes - groupOffset)
        : chunkPerGroup;

    IbgdaLocalBuffer groupLocalBuf = localBuf.subBuffer(groupOffset);
    IbgdaRemoteBuffer groupRemoteBuf = remoteBuf.subBuffer(groupOffset);

    return put_signal_group_local(
        group,
        groupLocalBuf,
        groupRemoteBuf,
        groupBytes,
        remoteSignalBuf,
        signalId,
        signalVal);
  }

  // ===========================================================================
  // Local Signal Operations (caller-provided local signal buffer)
  // ===========================================================================

  /**
   * wait_signal - Wait for remote signal arrival
   *
   * Spin-waits on a local signal buffer at signalId until (value >= expected).
   * Provides "acquire" semantics — once the signal is seen, all prior remote
   * writes are visible.
   *
   * @param localSignalBuf Local signal buffer (caller-owned)
   * @param signalId Index into the signal buffer (uint64_t units)
   * @param expected Value to wait for (uses >= comparison)
   * @param timeout Optional timeout (default: disabled, infinite wait)
   */
  __device__ void wait_signal(
      const IbgdaLocalBuffer& localSignalBuf,
      int signalId,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    volatile uint64_t* sig =
        static_cast<volatile uint64_t*>(localSignalBuf.ptr) + signalId;
    while (*sig < expected) {
      TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
          timeout,
          "wait_signal(GE): signalId=%d, expected>=%llu, current=%llu",
          signalId,
          static_cast<unsigned long long>(expected),
          static_cast<unsigned long long>(*sig));
    }
    __threadfence_system();
  }

  /**
   * read_signal - Read current signal value (non-blocking)
   *
   * @param localSignalBuf Local signal buffer (caller-owned)
   * @param signalId Index into the signal buffer (uint64_t units)
   * @return Current signal value
   */
  __device__ uint64_t
  read_signal(const IbgdaLocalBuffer& localSignalBuf, int signalId) const {
    volatile uint64_t* sig =
        static_cast<volatile uint64_t*>(localSignalBuf.ptr) + signalId;
    return *sig;
  }

  /**
   * reset_signal - Reset a remote signal slot to zero via RDMA inline write
   *
   * Uses RDMA inline write to set the remote signal to zero. Includes a
   * fence before the write to ensure all prior RDMA operations have been
   * sent, and waits for the write to complete before returning.
   *
   * @param remoteSignalBuf Remote signal buffer (caller-owned)
   * @param signalId Index into the signal buffer (uint64_t units)
   */
  __device__ void reset_signal(
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int signalId) {
    fence();

    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(remoteSignalBuf.ptr) + signalId),
        .key = remoteSignalBuf.rkey.value};

    doca_gpu_dev_verbs_p<
        uint64_t,
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
        qp_, remoteAddr, static_cast<uint64_t>(0), &ticket);

    Timeout timeout(kDefaultDeviceTimeoutCycles);
    timeout.start();
    wait_local(IbgdaWork(ticket), timeout);
  }

  // ===========================================================================
  // Remote Signal / Counter Operations (for window-owned buffers)
  // ===========================================================================
  //
  // These methods use this transport's QP but caller-provided buffer info.
  // The window owns the signal/counter buffers; the transport provides the QP.

  /**
   * signal_remote - RDMA atomic to a caller-provided remote signal buffer
   *
   * Uses this transport's main QP to post an RDMA atomic fetch-add to
   * an arbitrary remote buffer. The caller provides the remote buffer info
   * (rkey + addr) — typically from the window's IBGDA signal inbox.
   *
   * @param remoteBuf Remote signal buffer (window-owned, RDMA-registered)
   * @param signalId Index into the remote signal buffer (uint64_t units)
   * @param value Value to atomically add
   * @return IbgdaWork for tracking local completion
   */
  __device__ IbgdaWork signal_remote(
      const IbgdaRemoteBuffer& remoteBuf,
      int signalId,
      uint64_t value) {
    doca_gpu_dev_verbs_ticket_t ticket;

    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(remoteBuf.ptr) + signalId),
        .key = remoteBuf.rkey.value};
    doca_gpu_dev_verbs_addr sinkAddr = {.addr = 0, .key = sinkLkey_.value};

    doca_gpu_dev_verbs_signal<
        DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD,
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
        qp_, remoteAddr, sinkAddr, value, &ticket);

    return IbgdaWork(ticket);
  }

  /**
   * signal_remote_with_fence - RDMA atomic with NIC-level fence to remote
   * buffer
   *
   * Same as signal_remote() but with IBV_SEND_FENCE flag on the WQE.
   * The NIC will complete all prior WQEs before processing this atomic.
   *
   * @param remoteBuf Remote signal buffer (window-owned, RDMA-registered)
   * @param signalId Index into the remote signal buffer (uint64_t units)
   * @param value Value to atomically add
   * @return IbgdaWork for tracking local completion
   */
  __device__ IbgdaWork signal_remote_with_fence(
      const IbgdaRemoteBuffer& remoteBuf,
      int signalId,
      uint64_t value) {
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(remoteBuf.ptr) + signalId),
        .key = remoteBuf.rkey.value};
    doca_gpu_dev_verbs_addr sinkAddr = {.addr = 0, .key = sinkLkey_.value};

    uint64_t wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp_, 1);

    struct doca_gpu_dev_verbs_wqe* wqe_ptr =
        doca_gpu_dev_verbs_get_wqe_ptr(qp_, wqe_idx);

    doca_gpu_dev_verbs_wqe_prepare_atomic(
        qp_,
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
        value,
        0);

    doca_gpu_dev_verbs_mark_wqes_ready<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp_, wqe_idx, wqe_idx);

    doca_gpu_dev_verbs_submit<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp_, wqe_idx + 1);

    return IbgdaWork(wqe_idx);
  }

  /**
   * put_signal_counter_remote - Data write + remote signal + local counter
   *
   * Compound operation using main QP (data + signal) and companion QP
   * (counter):
   * 1. Main QP: RDMA Write data to remote buffer
   * 2. Main QP: RDMA Atomic fetch-add to remote signal buffer
   * 3. Companion QP: WAIT on main QP completion, then RDMA Atomic fetch-add
   *    to LOCAL counter buffer (loopback for NIC completion tracking)
   *
   * All buffer addresses are caller-provided (window-owned).
   *
   * @param localDataBuf Source data buffer (local GPU memory)
   * @param remoteDataBuf Destination data buffer (remote GPU memory)
   * @param nbytes Number of data bytes to transfer
   * @param remoteSignalBuf Remote signal buffer (window-owned)
   * @param signalId Signal slot index
   * @param signalVal Signal value to atomically add
   * @param localCounterBuf Local counter buffer (window-owned)
   * @param counterId Counter slot index
   * @param counterVal Counter value to atomically add (typically 1)
   */
  __device__ void put_signal_counter_remote(
      const IbgdaLocalBuffer& localDataBuf,
      const IbgdaRemoteBuffer& remoteDataBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int signalId,
      uint64_t signalVal,
      const IbgdaLocalBuffer& localCounterBuf,
      int counterId,
      uint64_t counterVal) {
    doca_gpu_dev_verbs_addr laddr = {
        .addr = reinterpret_cast<uint64_t>(localDataBuf.ptr),
        .key = localDataBuf.lkey.value};
    doca_gpu_dev_verbs_addr raddr = {
        .addr = reinterpret_cast<uint64_t>(remoteDataBuf.ptr),
        .key = remoteDataBuf.rkey.value};

    doca_gpu_dev_verbs_addr sigRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(remoteSignalBuf.ptr) + signalId),
        .key = remoteSignalBuf.rkey.value};
    doca_gpu_dev_verbs_addr sigSinkAddr = {.addr = 0, .key = sinkLkey_.value};

    doca_gpu_dev_verbs_addr counterRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(localCounterBuf.ptr) + counterId),
        .key = localCounterBuf.lkey.value};
    doca_gpu_dev_verbs_addr counterSinkAddr = {
        .addr = 0, .key = sinkLkey_.value};

    doca_gpu_dev_verbs_put_signal_counter<
        DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD,
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
        qp_,
        raddr,
        laddr,
        nbytes,
        sigRemoteAddr,
        sigSinkAddr,
        signalVal,
        companionQp_,
        counterRemoteAddr,
        counterSinkAddr,
        counterVal);
  }

  /**
   * signal_counter_remote - Remote signal + local counter (no data write)
   *
   * Compound operation: signal a remote peer + track local completion via
   * counter. Same as put_signal_counter_remote but without the data write.
   *
   * @param remoteSignalBuf Remote signal buffer (window-owned)
   * @param signalId Signal slot index
   * @param signalVal Signal value to atomically add
   * @param localCounterBuf Local counter buffer (window-owned)
   * @param counterId Counter slot index
   * @param counterVal Counter value to atomically add (typically 1)
   */
  __device__ void signal_counter_remote(
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int signalId,
      uint64_t signalVal,
      const IbgdaLocalBuffer& localCounterBuf,
      int counterId,
      uint64_t counterVal) {
    doca_gpu_dev_verbs_addr sigRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(remoteSignalBuf.ptr) + signalId),
        .key = remoteSignalBuf.rkey.value};
    doca_gpu_dev_verbs_addr sigSinkAddr = {.addr = 0, .key = sinkLkey_.value};

    doca_gpu_dev_verbs_addr counterRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(localCounterBuf.ptr) + counterId),
        .key = localCounterBuf.lkey.value};
    doca_gpu_dev_verbs_addr counterSinkAddr = {
        .addr = 0, .key = sinkLkey_.value};

    doca_gpu_dev_verbs_signal_counter<
        DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD,
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
        qp_,
        sigRemoteAddr,
        sigSinkAddr,
        signalVal,
        companionQp_,
        counterRemoteAddr,
        counterSinkAddr,
        counterVal);
  }

  /**
   * wait_local - Wait for local completion of an RDMA operation
   *
   * Blocks until the RDMA operation identified by the work handle has completed
   * locally. This means the data has been handed off to the NIC, but does NOT
   * guarantee arrival at the remote HBM.
   *
   * Unlike fence(), this polls the CQ directly at the work handle's WQE index
   * without posting a NOP WQE, making it cheaper for single-operation waits.
   *
   * Supports an optional timeout to prevent infinite hangs. When a timeout
   * is provided and expires, the kernel traps with an error message.
   *
   * @param work Work handle returned from put(), signal_remote(), etc.
   * @param timeout Optional timeout (default: no timeout, infinite wait)
   */
  __device__ void wait_local(
      const IbgdaWork& work,
      Timeout timeout = Timeout()) {
    if (!timeout.isEnabled()) {
      doca_gpu_dev_verbs_wait<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp_, work.value);
    } else {
      int status;
      do {
        status = doca_gpu_dev_verbs_poll_one_cq_at<
            DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            doca_gpu_dev_verbs_qp_get_cq_sq(qp_), work.value);
        if (status == EBUSY) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "P2pIbgdaTransportDevice::wait_local timed out "
              "(ticket=%llu)",
              static_cast<unsigned long long>(work.value));
        }
      } while (status == EBUSY);
    }
  }

  /**
   * fence - Wait for all pending RDMA operations to complete at the NIC
   *
   * Issues a NOP WQE and waits for it to complete. Since WQEs are processed
   * in order by the NIC, when the NOP completes, all prior WQEs have been
   * processed.
   *
   * Note: This only ensures local NIC completion, not remote arrival.
   * For remote completion guarantees, use signal-based synchronization.
   */
  __device__ void fence() {
    doca_fence<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp_);
  }

  __host__ __device__ doca_gpu_dev_verbs_qp* getQp() const {
    return qp_;
  }

 private:
  /**
   * put_group_impl - Generic group-collaborative RDMA Write
   *
   * Uses manual WQE construction with low-level DOCA verbs APIs to support
   * any group size (not just warp). The leader reserves WQE slots for all
   * threads, broadcasts the base index, each thread prepares its WQE,
   * then the leader marks all WQEs ready and rings the doorbell.
   *
   * Per-lane parameters (laneBuf, laneRemoteBuf, laneBytes) should already
   * be computed by the caller (put_group_local).
   * Per-lane size constraint: laneBytes <=
   * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE.
   *
   * @param group ThreadGroup (must have group_size > 1)
   * @param laneBuf Source buffer for this thread's chunk
   * @param laneRemoteBuf Destination buffer for this thread's chunk
   * @param laneBytes Number of bytes for this thread's chunk
   *
   * @return IbgdaWork for tracking local completion via wait_local()
   */
  __device__ IbgdaWork put_group_impl(
      ThreadGroup& group,
      const IbgdaLocalBuffer& laneBuf,
      const IbgdaRemoteBuffer& laneRemoteBuf,
      std::size_t laneBytes) {
    // 1. Leader reserves group_size WQE slots
    uint64_t base_wqe_idx = 0;
    if (group.is_leader()) {
      base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp_, group.group_size);
    }

    // 2. Broadcast base index to all threads
    base_wqe_idx = group.broadcast<uint64_t>(base_wqe_idx);

    // 3. Each thread prepares its WQE
    uint64_t wqe_idx = base_wqe_idx + group.thread_id_in_group;
    struct doca_gpu_dev_verbs_wqe* wqe_ptr =
        doca_gpu_dev_verbs_get_wqe_ptr(qp_, wqe_idx);

    doca_gpu_dev_verbs_wqe_prepare_write(
        qp_,
        wqe_ptr,
        static_cast<uint16_t>(wqe_idx),
        DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        0, // immediate
        reinterpret_cast<uint64_t>(laneRemoteBuf.ptr),
        laneRemoteBuf.rkey.value,
        reinterpret_cast<uint64_t>(laneBuf.ptr),
        laneBuf.lkey.value,
        static_cast<uint32_t>(laneBytes));

    // 4. Sync — all WQEs prepared
    group.sync();

    // 5. Leader marks ready and rings doorbell
    if (group.is_leader()) {
      doca_gpu_dev_verbs_mark_wqes_ready<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
          qp_, base_wqe_idx, base_wqe_idx + group.group_size - 1);
      doca_gpu_dev_verbs_submit<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
          DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
          qp_, base_wqe_idx + group.group_size);
    }

    // 6. Sync — ensure submit done before threads proceed
    group.sync();

    return IbgdaWork(wqe_idx);
  }

  doca_gpu_dev_verbs_qp* qp_{nullptr};
  doca_gpu_dev_verbs_qp* companionQp_{nullptr};
  NetworkLKey sinkLkey_{};
};

} // namespace comms::pipes
