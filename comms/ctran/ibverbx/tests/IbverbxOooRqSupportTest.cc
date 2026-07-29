// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "comms/ctran/ibverbx/IbvQpUtils.h"
#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/ibverbx/Mlx5dv.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

// OOO_RQ (MLX5DV_QP_CREATE_OOO_DP) is mlx5-only.
const std::string kNicPrefix("mlx5_");

class IbverbxOooRqTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
#if defined(__HIP_PLATFORM_AMD__)
    GTEST_SKIP()
        << "Skipping OOO_RQ test on AMD platform: mlx5dv not supported";
#else
    ASSERT_TRUE(ibvInit());
    // Filter to mlx5-family NICs — OOO_RQ is mlx5-only. Skips on hosts where
    // the first device would otherwise be e.g. bnxt_re, which would exercise
    // the mlx5dv APIs against a non-mlx5 provider (returns ENOTSUP but
    // defeats the intent of validating the mlx5 path).
    auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
    ASSERT_TRUE(devices);
    if (devices->empty()) {
      GTEST_SKIP() << "No mlx5 devices found on host";
    }
    devicesHolder_ = std::move(*devices);
    device_ = &devicesHolder_[0];

    // With an mlx5 device present, these symbols MUST resolve — production
    // wraps null-symbol into EOPNOTSUPP, silently disabling OOO_RQ fleet-wide.
    ASSERT_NE(ibvSymbols.mlx5dv_internal_query_device, nullptr)
        << "mlx5dv_query_device symbol failed to resolve on host with mlx5 "
           "device present — production dlopen/dlsym path is broken. Every "
           "OOO_RQ deployment silently disables the feature.";
    ASSERT_NE(ibvSymbols.mlx5dv_internal_create_qp, nullptr)
        << "mlx5dv_create_qp symbol failed to resolve on host with mlx5 "
           "device present — production dlopen/dlsym path is broken. Every "
           "OOO_RQ deployment silently disables the feature.";
#endif
  }

  // Held to keep device_ pointing at a live object across the test body.
  std::vector<IbvDevice> devicesHolder_;
  IbvDevice* device_{nullptr};
};

// Basic query: mlx5dv_query_device with MLX5DV_CONTEXT_MASK_OOO_RECV_WRS.
// Three outcomes:
//   1. Query API unsupported (older libmlx5 / no cap category)
//       → returns ENOTSUP / EOPNOTSUPP → skip.
//   2. Query works, HW doesn't support the feature (max_rc == 0)
//       → skip (caps struct legitimately populated with zeros).
//   3. Query works, HW supports the feature (max_rc > 0)
//       → pass.
TEST_F(IbverbxOooRqTestFixture, QueryOooRecvWrsCaps) {
  auto maybeCtx =
      Mlx5dv::queryDevice(device_->context(), MLX5DV_CONTEXT_MASK_OOO_RECV_WRS);
  if (maybeCtx.hasError() &&
      (maybeCtx.error().errNum == ENOTSUP ||
       maybeCtx.error().errNum == EOPNOTSUPP)) {
    GTEST_SKIP() << "mlx5dv_query_device not available: "
                 << maybeCtx.error().errStr;
  }
  ASSERT_TRUE(maybeCtx) << "queryDevice failed: " << maybeCtx.error().errStr;

  // Driver populated the caps struct but HW/firmware reports max_rc == 0.
  // The query API works; the underlying feature is just absent on this host.
  if (maybeCtx->ooo_recv_wrs_caps.max_rc == 0) {
    GTEST_SKIP() << "OOO_RECV_WRS caps struct populated but HW reports "
                    "max_rc == 0 (feature not supported on this host)";
  }
}

// Create an RC QP with MLX5DV_QP_CREATE_OOO_DP.
//
// Precheck the HW/firmware capability via mlx5dv_query_device and skip only
// when OOO_RQ is genuinely unsupported (query API absent, or reported max_rc
// below what we run with).
TEST_F(IbverbxOooRqTestFixture, CreateRcQpWithOooDp) {
  constexpr uint32_t kRequiredOooMaxRc = 128;

  auto maybeCtx =
      Mlx5dv::queryDevice(device_->context(), MLX5DV_CONTEXT_MASK_OOO_RECV_WRS);
  if (maybeCtx.hasError() &&
      (maybeCtx.error().errNum == ENOTSUP ||
       maybeCtx.error().errNum == EOPNOTSUPP)) {
    GTEST_SKIP() << "mlx5dv_query_device not available: "
                 << maybeCtx.error().errStr;
  }
  ASSERT_TRUE(maybeCtx) << "queryDevice failed: " << maybeCtx.error().errStr;
  if (maybeCtx->ooo_recv_wrs_caps.max_rc < kRequiredOooMaxRc) {
    GTEST_SKIP() << "OOO_RECV_WRS not supported on this host (max_rc="
                 << maybeCtx->ooo_recv_wrs_caps.max_rc << " < "
                 << kRequiredOooMaxRc << ")";
  }

  auto pd = device_->allocPd();
  ASSERT_TRUE(pd);

  auto cq = device_->createCq(1024, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);

  auto qp = createRcQpWithOooDp(
      &(*pd),
      cq->cq(),
      /*maxSendWr=*/256,
      /*maxRecvWr=*/128,
      /*oooDp=*/true);
  ASSERT_TRUE(qp)
      << "createRcQpWithOooDp(oooDp=true) failed on OOO-capable device "
      << "(errno=" << qp.error().errNum << " " << qp.error().errStr
      << "). Likely a bug in this diff's QP init attrs / comp_mask / "
      << "create_flags wiring rather than a hardware-support issue.";
  ASSERT_NE(qp->qp(), nullptr);
}

// Sanity: the same helper with oooDp=false still yields a valid RC QP via the
// standard createRcQp fallback (mlx5dv is not exercised on this path).
TEST_F(IbverbxOooRqTestFixture, CreateRcQpWithoutOooDp) {
  auto pd = device_->allocPd();
  ASSERT_TRUE(pd);

  auto cq = device_->createCq(1024, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);

  auto qp = createRcQpWithOooDp(&(*pd), cq->cq(), 256, 128, /*oooDp=*/false);
  ASSERT_TRUE(qp) << "createRcQpWithOooDp(oooDp=false) failed: "
                  << qp.error().errStr;
  ASSERT_NE(qp->qp(), nullptr);
}

} // namespace ibverbx
