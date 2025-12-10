// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/ibverbx/IbvCq.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/ibverbx/Mlx5dv.h"

namespace {

bool sanityCheckCqInitialization(void* cq_buf) {
  const uint32_t ncqes = 256; // CQ size (must be power of 2)

  // Check all CQE entries to ensure opcode is initialized to MLX5_CQE_INVALID
  size_t cqeSize = sizeof(ibverbx::mlx5_cqe64); // 64 bytes per CQE

  // Quick scan of all CQEs for overall statistics
  for (int i = 8; i < (int)ncqes; i++) {
    ibverbx::mlx5_cqe64* cqe =
        (ibverbx::mlx5_cqe64*)((uint8_t*)cq_buf + i * cqeSize);
    uint8_t opown = cqe->op_own; // Use direct read on host
    uint8_t opcode = opown >> 4;

    if (opcode != ibverbx::MLX5_CQE_INVALID) { // 0xf
      return false;
    }
  }
  return true;
}

} // namespace

namespace ibverbx {

extern IbvSymbols ibvSymbols;

/*** IbvCq ***/

IbvCq::IbvCq(ibv_cq* cq, int32_t deviceId) : cq_(cq), deviceId_(deviceId) {}

IbvCq::~IbvCq() {
  if (cq_) {
    int rc = ibvSymbols.ibv_internal_destroy_cq(cq_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to destroy cq rc: {}, {}", rc, strerror(errno));
    }
  }
}

IbvCq::IbvCq(IbvCq&& other) noexcept {
  cq_ = other.cq_;
  deviceId_ = other.deviceId_;
  other.cq_ = nullptr;
  other.deviceId_ = -1;
}

IbvCq& IbvCq::operator=(IbvCq&& other) noexcept {
  cq_ = other.cq_;
  deviceId_ = other.deviceId_;
  other.cq_ = nullptr;
  other.deviceId_ = -1;
  return *this;
}

ibv_cq* IbvCq::cq() const {
  return cq_;
}

int32_t IbvCq::getDeviceId() const {
  return deviceId_;
}

folly::Expected<folly::Unit, Error> IbvCq::reqNotifyCq(
    int solicited_only) const {
  int rc = cq_->context->ops.req_notify_cq(cq_, solicited_only);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

folly::Expected<struct device_cq, Error> IbvCq::getDeviceCq() const noexcept {
#if defined(__HIP_PLATFORM_AMD__)
  throw std::runtime_error("getDeviceQp() is not supported on AMD GPUs");
#else
  struct device_cq deviceCq{};

  // create mlx5dv_cq
  ibverbx::mlx5dv_obj obj = {};
  ibverbx::mlx5dv_cq mlx5_cq{};
  obj.cq.in = cq();
  obj.cq.out = &mlx5_cq;
  {
    auto ret = Mlx5dv::initObj(&obj, ibverbx::MLX5DV_OBJ_CQ);
    if (ret.hasError()) {
      return folly::makeUnexpected(ret.error());
    }
  }

  // sanity check cqe are initialied to 0xff
  if (!sanityCheckCqInitialization(mlx5_cq.buf)) {
    return folly::makeUnexpected(Error(1, "CQE not initialized to 0xff"));
  }

  // Map CQ buffer
  {
    auto ret = cudaHostGetDevicePointer(&deviceCq.cq_buf, mlx5_cq.buf, 0);
    if (ret != cudaSuccess) {
      return folly::makeUnexpected(Error(
          static_cast<int>(ret),
          fmt::format(
              "Mapping CQ to GPU buffer failed: {}", cudaGetErrorString(ret))));
    }
    XLOGF(
        INFO,
        "Mapped CQ buffer host={} device={} ncqes={} cq_dbrec={}",
        fmt::ptr(mlx5_cq.buf),
        fmt::ptr(deviceCq.cq_buf),
        mlx5_cq.cqe_cnt,
        fmt::ptr(mlx5_cq.dbrec));
  }

  // TODO: log2(mlx5_cq.cqe_cnt)
  deviceCq.ncqes = mlx5_cq.cqe_cnt;
  deviceCq.cq_dbrec = mlx5_cq.dbrec;

  return deviceCq;
#endif
}

} // namespace ibverbx
