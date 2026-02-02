// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include <device/doca_gpunetio_dev_verbs_onesided.cuh>

#include "comms/pipes/IbgdaBuffer.h"

namespace comms::pipes {

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

  __device__ IbgdaWork() = default;

  __device__ explicit IbgdaWork(doca_gpu_dev_verbs_ticket_t ticket)
      : value(ticket) {}
};

/**
 * P2pIbgdaTransportDevice - Device-side per-peer RDMA transport handle
 *
 * Provides GPU-initiated RDMA operations using DOCA GPUNetIO high-level APIs.
 * Each instance represents a connection to a single peer and contains:
 * - GPU QP handle for issuing RDMA operations
 * - Local and remote signal buffers for synchronization
 *
 * OPERATION SEMANTICS:
 * ====================
 *
 * put_signal() - Non-blocking RDMA Write + Atomic FetchAdd
 *   Issues an RDMA Write from local buffer to remote buffer, followed by
 *   an atomic increment of the remote signal buffer. Returns a work handle
 *   for tracking local completion.
 *
 *   Use case: Sender writes data to receiver's buffer and signals arrival.
 *
 * wait_local() - Wait for local completion
 *   Blocks until the RDMA operation identified by the work handle has completed
 *   locally (data has been sent to the NIC). This does NOT guarantee the
 *   data has arrived at the remote side.
 *
 * wait_signal() - Wait for remote signal
 *   Spin-waits on the local signal buffer until it reaches the expected value.
 *   This is used by the receiver to wait for data arrival.
 *
 * EXECUTION SCOPE:
 * ================
 * Operations default to thread-level scope where each thread posts its own
 * RDMA operation. For large transfers, consider using warp-level scope where
 * all 32 threads in a warp collaborate on a single operation.
 *
 * Template parameters control execution behavior:
 * - resource_sharing_mode: GPU (default) for GPU-owned QP resources
 * - nic_handler: AUTO (default) lets DOCA choose optimal doorbell handling
 * - exec_scope: THREAD or WARP level operation granularity
 */
class P2pIbgdaTransportDevice {
 public:
  __host__ __device__ P2pIbgdaTransportDevice() = default;

  __host__ __device__ P2pIbgdaTransportDevice(
      doca_gpu_dev_verbs_qp* qp,
      const IbgdaLocalBuffer& localSignalBuf,
      const IbgdaRemoteBuffer& remoteSignalBuf)
      : qp_(qp),
        localSignalBuf_(localSignalBuf),
        remoteSignalBuf_(remoteSignalBuf) {}

  /**
   * put_signal - RDMA Write with atomic signal (non-blocking)
   *
   * Performs an RDMA Write from local buffer to remote buffer, followed by
   * an atomic fetch-add on the remote signal buffer. Returns immediately
   * with a ticket for optional completion tracking.
   *
   * MEMORY ORDERING:
   * The atomic signal is guaranteed to arrive after the data write,
   * providing a "release" semantic on the sender side.
   *
   * @param localBuf Source buffer in local GPU memory
   * @param remoteBuf Destination buffer in remote GPU memory
   * @param nbytes Number of bytes to transfer
   * @param signalVal Value to atomically add to remote signal buffer
   *
   * @return IbgdaWork for tracking local completion via wait_local()
   */
  template <
      doca_gpu_dev_verbs_exec_scope exec_scope =
          DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
  __device__ IbgdaWork put_signal(
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      uint64_t signalVal) {
    doca_gpu_dev_verbs_ticket_t ticket;

    doca_gpu_dev_verbs_addr localAddr = {
        .addr = reinterpret_cast<uint64_t>(localBuf.ptr), .key = localBuf.lkey};
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(remoteBuf.ptr),
        .key = remoteBuf.rkey};
    doca_gpu_dev_verbs_addr localSignalAddr = {
        .addr = reinterpret_cast<uint64_t>(localSignalBuf_.ptr),
        .key = localSignalBuf_.lkey};
    doca_gpu_dev_verbs_addr remoteSignalAddr = {
        .addr = reinterpret_cast<uint64_t>(remoteSignalBuf_.ptr),
        .key = remoteSignalBuf_.rkey};

    doca_gpu_dev_verbs_put_signal<
        DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD,
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
        exec_scope>(
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
  template <
      doca_gpu_dev_verbs_exec_scope exec_scope =
          DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
  __device__ IbgdaWork
  put(const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes) {
    doca_gpu_dev_verbs_ticket_t ticket;

    doca_gpu_dev_verbs_addr localAddr = {
        .addr = reinterpret_cast<uint64_t>(localBuf.ptr), .key = localBuf.lkey};
    doca_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(remoteBuf.ptr),
        .key = remoteBuf.rkey};

    doca_gpu_dev_verbs_put<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
        exec_scope>(qp_, remoteAddr, localAddr, nbytes, &ticket);

    return IbgdaWork(ticket);
  }

  /**
   * signal - Send atomic signal only (non-blocking)
   *
   * Performs an atomic fetch-add on the remote signal buffer without
   * any data transfer. Useful for pure synchronization.
   *
   * @param signalVal Value to atomically add to remote signal buffer
   *
   * @return IbgdaWork for tracking local completion via wait_local()
   */
  __device__ IbgdaWork signal(uint64_t signalVal) {
    doca_gpu_dev_verbs_ticket_t ticket;

    doca_gpu_dev_verbs_addr localSignalAddr = {
        .addr = reinterpret_cast<uint64_t>(localSignalBuf_.ptr),
        .key = localSignalBuf_.lkey};
    doca_gpu_dev_verbs_addr remoteSignalAddr = {
        .addr = reinterpret_cast<uint64_t>(remoteSignalBuf_.ptr),
        .key = remoteSignalBuf_.rkey};

    doca_gpu_dev_verbs_signal<
        DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD,
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
        qp_, remoteSignalAddr, localSignalAddr, signalVal, &ticket);

    return IbgdaWork(ticket);
  }

  /**
   * wait_local - Wait for local completion of an RDMA operation
   *
   * Blocks until the RDMA operation identified by the work handle has completed
   * locally. This means the data has been handed off to the NIC, but does
   * NOT guarantee arrival at the remote side.
   *
   * For remote completion guarantee, use wait_signal() on the receiver side.
   *
   * @param work Work handle returned from put_signal(), put(), or signal()
   */
  __device__ void wait_local(IbgdaWork work) {
    doca_gpu_dev_verbs_wait<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp_, work.value);
  }

  /**
   * wait_signal - Wait for remote signal arrival
   *
   * Spin-waits on the local signal buffer until its value reaches or
   * exceeds the expected value. This provides "acquire" semantics -
   * once the signal is seen, all prior remote writes are visible.
   *
   * @param expectedVal Signal value to wait for (uses >= comparison)
   */
  __device__ void wait_signal(uint64_t expectedVal) {
    volatile uint64_t* sig =
        reinterpret_cast<volatile uint64_t*>(localSignalBuf_.ptr);
    while (*sig < expectedVal) {
      // Spin
    }
    __threadfence_system();
  }

  /**
   * read_signal - Read current signal value
   *
   * Non-blocking read of the local signal buffer value.
   *
   * @return Current signal value
   */
  __device__ uint64_t read_signal() const {
    volatile uint64_t* sig =
        reinterpret_cast<volatile uint64_t*>(localSignalBuf_.ptr);
    return *sig;
  }

  /**
   * reset_signal - Reset local signal buffer to zero
   *
   * Resets the local signal counter. Should only be called when no
   * operations are in flight and both sides have synchronized.
   */
  __device__ void reset_signal() {
    volatile uint64_t* sig =
        reinterpret_cast<volatile uint64_t*>(localSignalBuf_.ptr);
    *sig = 0;
    __threadfence_system();
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

 private:
  doca_gpu_dev_verbs_qp* qp_{nullptr};
  IbgdaLocalBuffer localSignalBuf_;
  IbgdaRemoteBuffer remoteSignalBuf_;
};

} // namespace comms::pipes
