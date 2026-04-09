// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/uniflow/Result.h"
#include "comms/uniflow/drivers/ibverbs/IbvCore.h"
#include "comms/uniflow/drivers/ibverbs/Mlx5Core.h" // IWYU pragma: keep

namespace uniflow {

/// Mockable wrapper around libibverbs.
/// All methods are virtual for testability via MockIbvApi.
/// Library is loaded via dlopen("libibverbs.so.1") on first use.
/// Hot-path ops (pollCq, postSend, postRecv) use the ibv_context ops vtable
/// for zero-overhead dispatch, following the NCCLX pattern.
class IbvApi {
 public:
  virtual ~IbvApi() = default;

  // --- Library initialization ---

  /// Load libibverbs.so.1 and resolve symbols. Thread-safe (call_once).
  virtual Status init();

  // --- Device management ---

  virtual Result<ibv_device**> getDeviceList(int* numDevices);
  virtual Status freeDeviceList(ibv_device** list);
  virtual Result<const char*> getDeviceName(ibv_device* device);
  virtual Result<ibv_context*> openDevice(ibv_device* device);
  virtual Status closeDevice(ibv_context* context);

  // --- Protection domain ---

  virtual Result<ibv_pd*> allocPd(ibv_context* context);
  virtual Status deallocPd(ibv_pd* pd);

  // --- Memory registration ---

  virtual Result<bool> isDmaBufSupported(ibv_pd* pd);

  virtual Result<ibv_mr*>
  regMr(ibv_pd* pd, void* addr, size_t length, int access);

  virtual Result<ibv_mr*> regDmabufMr(
      ibv_pd* pd,
      uint64_t offset,
      size_t length,
      uint64_t iova,
      int fd,
      int access);

  virtual Status deregMr(ibv_mr* mr);

  // --- Completion queue ---

  virtual Result<ibv_cq*> createCq(
      ibv_context* context,
      int cqe,
      void* cqContext,
      ibv_comp_channel* channel,
      int compVector);
  virtual Status destroyCq(ibv_cq* cq);

  /// Poll CQ for completions. Returns number of completions polled.
  /// Uses ops vtable for zero overhead.
  virtual Result<int> pollCq(ibv_cq* cq, int numEntries, ibv_wc* wc);

  // --- Queue pair ---

  virtual Result<ibv_qp*> createQp(ibv_pd* pd, ibv_qp_init_attr* attr);
  virtual Status destroyQp(ibv_qp* qp);
  virtual Status modifyQp(ibv_qp* qp, ibv_qp_attr* attr, int attrMask);

  // --- Data path (hot path — uses ops vtable) ---

  virtual Status postSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** badWr);
  virtual Status postRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** badWr);

  // --- Query ---

  virtual Status queryDevice(ibv_context* context, ibv_device_attr* attr);
  virtual Status
  queryPort(ibv_context* context, uint8_t portNum, ibv_port_attr* attr);
  virtual Status
  queryGid(ibv_context* context, uint8_t portNum, int index, ibv_gid* gid);

  // --- MLX5 direct verbs ---

  /// Check if an IB device supports mlx5 direct verbs.
  /// Returns false if libmlx5 is not available.
  virtual Result<bool> mlx5dvIsSupported(ibv_device* device);

  /// Query the Data Direct sysfs path for a device.
  virtual Status
  mlx5dvGetDataDirectSysfsPath(ibv_context* context, char* buf, size_t bufLen);

  /// Register a DMA-BUF memory region with mlx5 access flags.
  virtual Result<ibv_mr*> mlx5dvRegDmabufMr(
      ibv_pd* pd,
      uint64_t offset,
      size_t length,
      uint64_t iova,
      int fd,
      int access,
      int mlx5Access);
};

} // namespace uniflow
