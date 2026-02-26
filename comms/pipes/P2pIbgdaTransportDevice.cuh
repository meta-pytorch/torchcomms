// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include <device/doca_gpunetio_dev_verbs_onesided.cuh>

#include "comms/pipes/DocaVerbsUtils.cuh"
#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {

// IbgdaSignalOp and IbgdaCmpOp are defined in IbgdaBuffer.h

/**
 * IbgdaWork - Wrapper for DOCA GPU verbs operation handle
 *
 * Wraps the raw doca_gpu_dev_verbs_ticket_t to provide type safety
 * and a cleaner interface for tracking RDMA operation completion.
 *
 * The work handle represents a pending RDMA operation and can be used
 * with wait_local() to synchronize on local completion.
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
 * Each instance represents a connection to a single peer and contains:
 * - GPU QP handle for issuing RDMA operations
 * - Local and remote signal buffer arrays for synchronization
 *
 * SIGNAL ID-BASED API:
 * ====================
 * All signal operations use a signal_id (integer index) to identify which
 * signal slot to operate on. This design is consistent with torchcomms
 * device API and allows multiple independent signal channels per peer.
 *
 * Signal buffer layout:
 * - localSignalBuf_: Base pointer to array of uint64_t signals
 * - remoteSignalBuf_: Base pointer to peer's signal array
 * - Each signal_id indexes into these arrays: buf[signal_id]
 *
 * EXECUTION SCOPE:
 * ================
 * Thread-Level APIs:
 *   put(), put_signal(), put_signal_non_adaptive(),
 *   signal(), signal_with_fence(), wait_local()
 *   - Each thread posts its own independent RDMA operation
 *   - Supports multi-chunk transfers (size >
 * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE)
 *
 * Group-Level APIs (public):
 *   Group-local (data already partitioned per group):
 *     put_group_local(), put_signal_group_local()
 *     - Accept a ThreadGroup and partition a single group's data chunk across
 *       group threads
 *     - All ThreadGroup sizes are supported (WARP, MULTIWARP, BLOCK, etc.)
 *     - group_size == 1: falls back to thread-level put() / put_signal()
 *     - group_size > 1: uses put_group_impl() with manual WQE construction
 *     - The leader issues the fenced signal and broadcasts the ticket
 *
 *   Group-global (data shared across all groups):
 *     put_group_global(), put_signal_group_global()
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
class P2pIbgdaTransportDevice {
 public:
  P2pIbgdaTransportDevice() = default;

  /**
   * Constructor
   *
   * @param qp GPU QP handle for RDMA operations
   * @param localSignalBuf Base pointer to local signal buffer array
   * @param remoteSignalBuf Base pointer to remote signal buffer array
   * @param numSignals Number of signal slots in the buffer arrays
   */
  __host__ __device__ P2pIbgdaTransportDevice(
      doca_gpu_dev_verbs_qp* qp,
      const IbgdaLocalBuffer& localSignalBuf,
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int numSignals = 1)
      : qp_(qp),
        localSignalBuf_(localSignalBuf),
        remoteSignalBuf_(remoteSignalBuf),
        numSignals_(numSignals) {
    // Sanity check: numSignals must be positive
    if (numSignals <= 0) {
#ifdef __CUDA_ARCH__
      printf(
          "P2pIbgdaTransportDevice: invalid numSignals (%d), must be > 0\n",
          numSignals);
      __trap();
#endif
    }
  }

  /**
   * put_signal - RDMA Write with fenced atomic signal (adaptive routing safe)
   *
   * Performs an RDMA Write from local buffer to remote buffer, followed by
   * an atomic fetch-add on the remote signal buffer at signal_id with the
   * IBV_SEND_FENCE flag set on the signal's WQE. The fence flag instructs
   * the NIC to complete all prior WQEs (the data write) before processing
   * the signal WQE, providing adaptive-routing safety without GPU-side CQ
   * polling overhead — the fence is handled entirely in NIC hardware.
   *
   * MEMORY ORDERING:
   * The IBV_SEND_FENCE flag (DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FENCE) on the
   * signal WQE ensures the NIC does not begin processing the atomic signal
   * until all prior posted WQEs have completed. This is a NIC-level
   * ordering guarantee that avoids the need for GPU-side CQ polling.
   *
   * PERFORMANCE NOTE:
   * This avoids the GPU-side CQ polling overhead of wait_local(), making it
   * faster than the old put + wait_local + signal approach while maintaining
   * the same correctness guarantees. Use put_signal_non_adaptive() only for
   * networks with deterministic routing where no fence is needed.
   *
   * @param localBuf Source buffer in local GPU memory
   * @param remoteBuf Destination buffer in remote GPU memory
   * @param nbytes Number of bytes to transfer
   * @param signalId Index into the signal buffer array
   * @param signalVal Value to atomically add to remote signal buffer
   *
   * @return IbgdaWork for tracking signal completion via wait_local()
   */
  __device__ IbgdaWork put_signal(
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      int signalId,
      uint64_t signalVal) {
    checkSignalId(signalId, "put_signal");
    put(localBuf, remoteBuf, nbytes);
    return signal_with_fence(signalId, signalVal);
  }

  /**
   * put_signal_non_adaptive - RDMA Write with atomic signal as single operation
   *
   * Performs an RDMA Write from local buffer to remote buffer, followed by
   * an atomic fetch-add on the remote signal buffer at signal_id as a single
   * fused operation. Returns immediately with a ticket for completion tracking.
   *
   * WARNING - ADAPTIVE ROUTING:
   * On networks with adaptive routing, the data and signal may take different
   * paths and the signal could arrive before the data, causing the receiver
   * to read stale data. Use put_signal() for networks with adaptive routing.
   *
   * MEMORY ORDERING:
   * Relies on the NIC's internal ordering guarantees for compound operations.
   * The atomic signal is issued after the data write at the sender NIC, but
   * arrival order at the receiver depends on network path consistency.
   *
   * @param localBuf Source buffer in local GPU memory
   * @param remoteBuf Destination buffer in remote GPU memory
   * @param nbytes Number of bytes to transfer
   * @param signalId Index into the signal buffer array
   * @param signalVal Value to atomically add to remote signal buffer
   *
   * @return IbgdaWork for tracking local completion via wait_local()
   */
  __device__ IbgdaWork put_signal_non_adaptive(
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      int signalId,
      uint64_t signalVal) {
    checkSignalId(signalId, "put_signal_non_adaptive");
    doca_gpu_dev_verbs_ticket_t ticket;

    doca_gpu_dev_verbs_addr localAddr = {
        .addr = reinterpret_cast<uint64_t>(localBuf.ptr),
        .key = localBuf.lkey.value};
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(remoteBuf.ptr),
        .key = remoteBuf.rkey.value};
    doca_gpu_dev_verbs_addr localSignalAddr = {
        .addr = reinterpret_cast<uint64_t>(getLocalSignalPtr(signalId)),
        .key = localSignalBuf_.lkey.value};
    doca_gpu_dev_verbs_addr remoteSignalAddr = {
        .addr = reinterpret_cast<uint64_t>(getRemoteSignalPtr(signalId)),
        .key = remoteSignalBuf_.rkey.value};

    doca_gpu_dev_verbs_put_signal<
        DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD,
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
        DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>(
        qp_,
        remoteAddr,
        localAddr,
        nbytes,
        remoteSignalAddr,
        localSignalAddr,
        signalVal,
        &ticket);

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
   * put_signal_group_local - Group-collaborative RDMA Write with fenced signal
   *                          (group-local data, adaptive routing safe)
   *
   * Accepts a ThreadGroup and a single data chunk that belongs to this group,
   * partitions the data across group threads, issues collaborative RDMA
   * writes, and the leader issues a fenced atomic signal. The signal ticket
   * is broadcast to all threads via group.broadcast<uint64_t>().
   *
   * All ThreadGroup sizes are supported:
   * - group_size == 1: falls back to thread-level put() + signal_with_fence()
   * - group_size > 1: uses put_group_impl(), leader issues fenced signal,
   *   and broadcasts the ticket to all threads
   *
   * REQUIREMENTS:
   * - All threads in the group must call this function collectively
   *
   * MEMORY ORDERING:
   * The leader uses signal_with_fence() which sets IBV_SEND_FENCE on the
   * signal WQE, ensuring the NIC completes all prior data WQEs before
   * processing the signal. This is adaptive routing safe.
   *
   * @param group ThreadGroup describing the calling group
   * @param localBuf Source buffer in local GPU memory (this group's chunk)
   * @param remoteBuf Destination buffer in remote GPU memory (this group's
   * chunk)
   * @param nbytes Number of bytes to transfer (partitioned across lanes)
   * @param signalId Index into the signal buffer array
   * @param signalVal Value to atomically add to remote signal buffer
   *
   * @return IbgdaWork for tracking signal completion via wait_local()
   *         (same ticket broadcast to all threads in group)
   */
  __device__ IbgdaWork put_signal_group_local(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      int signalId,
      uint64_t signalVal) {
    checkSignalId(signalId, "put_signal_group_local");

    std::size_t chunkSize = nbytes / group.group_size;
    std::size_t offset = group.thread_id_in_group * chunkSize;
    // Last thread picks up any remainder bytes
    std::size_t laneBytes = (group.thread_id_in_group == group.group_size - 1)
        ? (nbytes - offset)
        : chunkSize;

    IbgdaLocalBuffer laneBuf = localBuf.subBuffer(offset);
    IbgdaRemoteBuffer laneRemoteBuf = remoteBuf.subBuffer(offset);

    if (group.group_size == 1) {
      put(laneBuf, laneRemoteBuf, laneBytes);
      return signal_with_fence(signalId, signalVal);
    }

    // Group-collaborative put (put_group_impl already syncs at the end)
    put_group_impl(group, laneBuf, laneRemoteBuf, laneBytes);

    // Leader issues fenced signal, broadcast ticket to all threads
    uint64_t signalTicket = 0;
    if (group.is_leader()) {
      IbgdaWork signalWork = signal_with_fence(signalId, signalVal);
      signalTicket = signalWork.value;
    }
    signalTicket = group.broadcast<uint64_t>(signalTicket);

    return IbgdaWork(signalTicket);
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

  /**
   * put_signal_group_global - Group-collaborative RDMA Write with fenced signal
   *                           (global data, adaptive routing safe)
   *
   * Accepts a ThreadGroup and a global data buffer shared by all groups.
   * Partitions the data across groups (last group picks up remainder),
   * then calls put_signal_group_local() on each group's chunk.
   *
   * Each group's call to put_signal_group_local() issues an atomic fetch-add
   * signal, so the total accumulated signal is (total_groups * signalVal).
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
   * @param signalId Index into the signal buffer array
   * @param signalVal Value to atomically add to remote signal buffer (per
   * group)
   *
   * @return IbgdaWork for tracking signal completion via wait_local()
   *         (same ticket broadcast to all threads in group)
   */
  __device__ IbgdaWork put_signal_group_global(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      int signalId,
      uint64_t signalVal) {
    // Partition across groups; last group picks up remainder
    std::size_t chunkPerGroup = nbytes / group.total_groups;
    std::size_t groupOffset = group.group_id * chunkPerGroup;
    std::size_t groupBytes = (group.group_id == group.total_groups - 1)
        ? (nbytes - groupOffset)
        : chunkPerGroup;

    IbgdaLocalBuffer groupLocalBuf = localBuf.subBuffer(groupOffset);
    IbgdaRemoteBuffer groupRemoteBuf = remoteBuf.subBuffer(groupOffset);

    return put_signal_group_local(
        group, groupLocalBuf, groupRemoteBuf, groupBytes, signalId, signalVal);
  }

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
   * signal - Send atomic signal only
   *
   * Performs an atomic operation on the remote signal buffer at the
   * specified signal_id. Useful for pure synchronization.
   *
   * @param signalId Index into the signal buffer array
   * @param signalVal Value to use for the atomic operation
   * @param op Signal operation type (ADD or SET). Defaults to ADD.
   *           Note: SET is not yet supported by DOCA GPUNetIO.
   *
   * @return IbgdaWork for tracking local completion via wait_local()
   */
  __device__ IbgdaWork signal(
      int signalId,
      uint64_t signalVal,
      IbgdaSignalOp op = IbgdaSignalOp::ADD) {
    checkSignalId(signalId, "signal");
    // Only ADD is supported by DOCA GPUNetIO currently.
    // Trap if caller passes SET (or any future unsupported operation).
    if (op != IbgdaSignalOp::ADD) {
      printf(
          "P2pIbgdaTransportDevice::signal: unsupported IbgdaSignalOp (%d), only ADD is supported\n",
          static_cast<int>(op));
      __trap();
    }

    doca_gpu_dev_verbs_ticket_t ticket;

    doca_gpu_dev_verbs_addr localSignalAddr = {
        .addr = reinterpret_cast<uint64_t>(getLocalSignalPtr(signalId)),
        .key = localSignalBuf_.lkey.value};
    doca_gpu_dev_verbs_addr remoteSignalAddr = {
        .addr = reinterpret_cast<uint64_t>(getRemoteSignalPtr(signalId)),
        .key = remoteSignalBuf_.rkey.value};

    doca_gpu_dev_verbs_signal<
        DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD,
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
        qp_, remoteSignalAddr, localSignalAddr, signalVal, &ticket);

    return IbgdaWork(ticket);
  }

  /**
   * signal_with_fence - Send atomic signal with NIC-level fence
   *
   * Performs an atomic fetch-add on the remote signal buffer at the
   * specified signal_id, with the IBV_SEND_FENCE flag set on the WQE.
   * The fence flag instructs the NIC to complete all prior posted WQEs
   * before processing this atomic operation.
   *
   * This provides the same ordering guarantee as wait_local() + signal()
   * but avoids the GPU-side CQ polling overhead. The fence is handled
   * entirely by the NIC hardware.
   *
   * @param signalId Index into the signal buffer array
   * @param signalVal Value to use for the atomic operation
   *
   * @return IbgdaWork for tracking local completion via wait_local()
   */
  __device__ IbgdaWork signal_with_fence(int signalId, uint64_t signalVal) {
    checkSignalId(signalId, "signal_with_fence");

    doca_gpu_dev_verbs_ticket_t ticket;

    doca_gpu_dev_verbs_addr localSignalAddr = {
        .addr = reinterpret_cast<uint64_t>(getLocalSignalPtr(signalId)),
        .key = localSignalBuf_.lkey.value};
    doca_gpu_dev_verbs_addr remoteSignalAddr = {
        .addr = reinterpret_cast<uint64_t>(getRemoteSignalPtr(signalId)),
        .key = remoteSignalBuf_.rkey.value};

    // Reserve a WQE slot and prepare an atomic fetch-add with fence flag
    uint64_t wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp_, 1);

    struct doca_gpu_dev_verbs_wqe* wqe_ptr =
        doca_gpu_dev_verbs_get_wqe_ptr(qp_, wqe_idx);

    // Use FENCE flag: NIC will complete all prior WQEs before this one
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        qp_,
        wqe_ptr,
        static_cast<uint16_t>(wqe_idx),
        DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        static_cast<doca_gpu_dev_verbs_wqe_ctrl_flags>(
            DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE |
            DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FENCE),
        remoteSignalAddr.addr,
        remoteSignalAddr.key,
        localSignalAddr.addr,
        localSignalAddr.key,
        sizeof(uint64_t),
        signalVal,
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
   * wait_local - Wait for local completion of an RDMA operation
   *
   * Blocks until the RDMA operation identified by the work handle has completed
   * locally. This means the data has been handed off to the remote NIC, but
   * does NOT guarantee arrival at the remote HBM.
   *
   * For remote completion guarantee, use wait_signal() on the receiver side.
   *
   * @param work Work handle returned from put_signal(), put(), or signal()
   */
  __device__ void wait_local(const IbgdaWork& work) {
    doca_gpu_dev_verbs_wait<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp_, work.value);
  }

  /**
   * wait_signal - Wait for remote signal arrival
   *
   * Spin-waits on the local signal buffer at signal_id until the comparison
   * condition is satisfied. This provides "acquire" semantics - once the
   * signal is seen, all prior remote writes are visible.
   *
   * An optional Timeout parameter controls how long to wait before trapping.
   * The default Timeout() (disabled) waits indefinitely with zero overhead.
   * When enabled, the timeout adds one well-predicted branch per spin
   * iteration. On expiry, prints a diagnostic message and calls __trap().
   *
   * IMPORTANT: The caller must call timeout.start() before calling this
   * method. The Timeout object captures the GPU clock at start() and
   * checks against the precomputed deadline in each spin iteration.
   *
   * @param signalId Index into the signal buffer array
   * @param cmp Comparison operation to use
   * @param value Value to compare against
   * @param timeout Timeout config (default: disabled, infinite wait)
   */
  __device__ void wait_signal(
      int signalId,
      IbgdaCmpOp cmp,
      uint64_t value,
      const Timeout& timeout = Timeout()) {
    checkSignalId(signalId, "wait_signal");
    volatile uint64_t* sig =
        reinterpret_cast<volatile uint64_t*>(getLocalSignalPtr(signalId));

    switch (cmp) {
      case IbgdaCmpOp::EQ:
        while (*sig != value) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "wait_signal(EQ): signalId=%d, expected=%llu, current=%llu",
              signalId,
              static_cast<unsigned long long>(value),
              static_cast<unsigned long long>(*sig));
        }
        break;
      case IbgdaCmpOp::NE:
        while (*sig == value) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "wait_signal(NE): signalId=%d, unwanted=%llu, current=%llu",
              signalId,
              static_cast<unsigned long long>(value),
              static_cast<unsigned long long>(*sig));
        }
        break;
      case IbgdaCmpOp::LT:
        while (*sig >= value) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "wait_signal(LT): signalId=%d, threshold=%llu, current=%llu",
              signalId,
              static_cast<unsigned long long>(value),
              static_cast<unsigned long long>(*sig));
        }
        break;
      case IbgdaCmpOp::LE:
        while (*sig > value) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "wait_signal(LE): signalId=%d, threshold=%llu, current=%llu",
              signalId,
              static_cast<unsigned long long>(value),
              static_cast<unsigned long long>(*sig));
        }
        break;
      case IbgdaCmpOp::GT:
        while (*sig <= value) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "wait_signal(GT): signalId=%d, threshold=%llu, current=%llu",
              signalId,
              static_cast<unsigned long long>(value),
              static_cast<unsigned long long>(*sig));
        }
        break;
      case IbgdaCmpOp::GE:
        while (*sig < value) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "wait_signal(GE): signalId=%d, expected>=%llu, current=%llu",
              signalId,
              static_cast<unsigned long long>(value),
              static_cast<unsigned long long>(*sig));
        }
        break;
    }
    __threadfence_system();
  }

  /**
   * read_signal - Read current signal value
   *
   * Non-blocking read of the local signal buffer value at signal_id.
   *
   * @param signalId Index into the signal buffer array
   * @return Current signal value
   */
  __device__ uint64_t read_signal(int signalId) const {
    checkSignalId(signalId, "read_signal");
    volatile uint64_t* sig =
        reinterpret_cast<volatile uint64_t*>(getLocalSignalPtr(signalId));
    return *sig;
  }

  /**
   * reset_signal - Reset remote peer's signal buffer to zero
   *
   * Performs an RDMA write to reset the remote signal at signal_id to zero.
   * This is a sender-side operation - only the sender should reset the signal
   * after the receiver has consumed the data.
   *
   * ORDERING GUARANTEES:
   * This function inserts fences before and after the reset to ensure correct
   * ordering with other RDMA operations:
   * - Pre-fence: Ensures all prior operations (e.g., put_signal) are processed
   *   by the NIC before the reset is issued
   * - Post-fence: Ensures the reset completes before any subsequent operations
   *
   * This prevents packet reordering issues where a reset could overtake prior
   * operations on the network and arrive at the remote peer first.
   *
   * Typical flow:
   * 1. Sender: put_signal() - write data and signal receiver
   * 2. Receiver: wait_signal() - wait for signal and read data
   * 3. Sender: reset_signal() - reset for next iteration (fenced)
   *
   * @param signalId Index into the signal buffer array
   */
  __device__ void reset_signal(int signalId) {
    checkSignalId(signalId, "reset_signal");

    // Fence before reset: ensure all prior operations are processed by NIC
    fence();

    // Prepare local signal value to write (0)
    volatile uint64_t* localSig =
        reinterpret_cast<volatile uint64_t*>(getLocalSignalPtr(signalId));
    *localSig = 0;
    __threadfence_system();

    // Issue the reset RDMA write
    doca_gpu_dev_verbs_ticket_t ticket;

    doca_gpu_dev_verbs_addr localSignalAddr = {
        .addr = reinterpret_cast<uint64_t>(getLocalSignalPtr(signalId)),
        .key = localSignalBuf_.lkey.value};
    doca_gpu_dev_verbs_addr remoteSignalAddr = {
        .addr = reinterpret_cast<uint64_t>(getRemoteSignalPtr(signalId)),
        .key = remoteSignalBuf_.rkey.value};

    doca_gpu_dev_verbs_put<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
        DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>(
        qp_, remoteSignalAddr, localSignalAddr, sizeof(uint64_t), &ticket);

    // Wait for reset to complete locally
    doca_gpu_dev_verbs_wait<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp_, ticket);

    // Fence after reset: ensure reset is processed before subsequent operations
    fence();
  }

  /**
   * fence - Wait for all pending RDMA operations to complete at the NIC
   *
   * Issues a NOP WQE and waits for it to complete. Since WQEs are processed
   * in order by the NIC, when the NOP completes, all prior WQEs have been
   * processed. This is useful before reset_signal to ensure prior operations
   * have been sent to the remote peer before the reset.
   *
   * Note: This only ensures local NIC completion, not remote arrival.
   * For remote completion guarantees, use signal-based synchronization.
   */
  __device__ void fence() {
    doca_fence<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp_);
  }

  // Getters for buffer info (useful for advanced operations)
  __host__ __device__ const IbgdaLocalBuffer& getLocalSignalBuffer() const {
    return localSignalBuf_;
  }

  __host__ __device__ const IbgdaRemoteBuffer& getRemoteSignalBuffer() const {
    return remoteSignalBuf_;
  }

  __host__ __device__ doca_gpu_dev_verbs_qp* getQp() const {
    return qp_;
  }

  __host__ __device__ int getNumSignals() const {
    return numSignals_;
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
   * be computed by the caller (put_group_local / put_signal_group_local).
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

  /**
   * Check signalId bounds and trap if out of range.
   * Only active in device code for better debuggability.
   */
  __device__ void checkSignalId(int signalId, const char* funcName) const {
    if (signalId < 0 || signalId >= numSignals_) {
      printf(
          "P2pIbgdaTransportDevice::%s: signalId (%d) out of range [0, %d)\n",
          funcName,
          signalId,
          numSignals_);
      __trap();
    }
  }

  /**
   * Get pointer to local signal at index
   */
  __host__ __device__ __forceinline__ void* getLocalSignalPtr(
      int signalId) const {
    return static_cast<uint64_t*>(localSignalBuf_.ptr) + signalId;
  }

  /**
   * Get pointer to remote signal at index
   */
  __host__ __device__ __forceinline__ void* getRemoteSignalPtr(
      int signalId) const {
    return static_cast<uint64_t*>(remoteSignalBuf_.ptr) + signalId;
  }

  doca_gpu_dev_verbs_qp* qp_{nullptr};
  IbgdaLocalBuffer localSignalBuf_;
  IbgdaRemoteBuffer remoteSignalBuf_;
  int numSignals_{1};
};

} // namespace comms::pipes
