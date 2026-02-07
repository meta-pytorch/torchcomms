// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#include <device/doca_gpunetio_dev_verbs_onesided.cuh>

namespace comms::pipes {

/**
 * DocaVerbsUtils - Utility functions for DOCA GPU Verbs operations
 *
 * Provides higher-level abstractions over low-level DOCA GPUNetIO APIs,
 * encapsulating WQE management and common patterns.
 */

/**
 * doca_fence - Wait for all pending RDMA operations to complete at the NIC
 *
 * Issues a NOP WQE and waits for it to complete. Since WQEs are processed
 * in order by the NIC, when the NOP completes, all prior WQEs have been
 * processed.
 *
 * Use this to ensure ordering between RDMA operations when packet reordering
 * on the network could cause issues (e.g., before reset_signal to ensure
 * prior put_signal operations have been sent).
 *
 * Note: This only ensures local NIC completion, not remote arrival.
 * For remote completion guarantees, use signal-based synchronization.
 *
 * @param qp GPU QP handle
 */
template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_nic_handler nic_handler =
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ __forceinline__ void doca_fence(doca_gpu_dev_verbs_qp* qp) {
  // Issue a NOP WQE with CQ update
  uint64_t wqe_idx =
      doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, 1);

  struct doca_gpu_dev_verbs_wqe* wqe_ptr =
      doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

  doca_gpu_dev_verbs_wqe_prepare_nop(
      qp,
      wqe_ptr,
      static_cast<uint16_t>(wqe_idx),
      DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);

  doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(
      qp, wqe_idx, wqe_idx);

  doca_gpu_dev_verbs_submit<
      resource_sharing_mode,
      DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
      nic_handler>(qp, wqe_idx + 1);

  // Wait for the NOP to complete
  doca_gpu_dev_verbs_wait<resource_sharing_mode, nic_handler>(qp, wqe_idx);
}

/**
 * doca_put_fenced - RDMA Write with pre-fence for ordering
 *
 * Issues a fence to ensure all prior operations complete, then performs
 * an RDMA write and waits for local completion.
 *
 * @param qp GPU QP handle
 * @param raddr Remote address descriptor
 * @param laddr Local address descriptor
 * @param size Number of bytes to transfer
 */
template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_nic_handler nic_handler =
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
    enum doca_gpu_dev_verbs_exec_scope exec_scope =
        DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
__device__ __forceinline__ void doca_put_fenced(
    doca_gpu_dev_verbs_qp* qp,
    doca_gpu_dev_verbs_addr raddr,
    doca_gpu_dev_verbs_addr laddr,
    size_t size) {
  // Fence before: ensure all prior operations are processed
  doca_fence<resource_sharing_mode, nic_handler>(qp);

  // Issue the put
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put<resource_sharing_mode, nic_handler, exec_scope>(
      qp, raddr, laddr, size, &ticket);

  // Wait for local completion
  doca_gpu_dev_verbs_wait<resource_sharing_mode, nic_handler>(qp, ticket);

  // Fence after: ensure this operation is processed before subsequent ones
  doca_fence<resource_sharing_mode, nic_handler>(qp);
}

} // namespace comms::pipes
