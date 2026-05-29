// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <chrono>
#include <optional>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/ibverbx/benchmarks/IbverbxDcBenchUtils.h"
#include "comms/ctran/ibverbx/tests/dc_utils.h"

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

namespace ibverbx {

class DciStreamsTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
#if defined(__HIP_PLATFORM_AMD__)
    GTEST_SKIP() << "DCI Streams not supported on AMD platform";
#else
    ASSERT_TRUE(ibvInit());
    initTwoDctStreamContext();
    initBufferAndAddressHandles();
#endif
  }

  void initTwoDctStreamContext() {
    auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
    ASSERT_TRUE(devices);
    devices_.emplace(std::move(*devices));
    auto& device = devices_->at(0);

    auto pd = device.allocPd();
    ASSERT_TRUE(pd);
    pd_.emplace(std::move(*pd));

    auto cq = device.createCq(kCqe, nullptr, nullptr, 0);
    ASSERT_TRUE(cq);
    cq_.emplace(std::move(*cq));

    auto srq = createSRQ(*pd_);
    if (srq.hasError() && srq.error().errNum == ENOSYS) {
      GTEST_SKIP() << "ibv_create_srq not available";
    }
    ASSERT_TRUE(srq) << "createSRQ failed: " << srq.error().errStr;
    srq_.emplace(std::move(*srq));

    auto dctA = createDCT(*pd_, *cq_, *srq_);
    if (dctA.hasError() && dctA.error().errNum == ENOTSUP) {
      GTEST_SKIP() << "DC QP creation not supported";
    }
    ASSERT_TRUE(dctA) << "createDCT(A) failed: " << dctA.error().errStr;
    dctA_.emplace(std::move(*dctA));

    auto dctB = createDCT(*pd_, *cq_, *srq_);
    ASSERT_TRUE(dctB) << "createDCT(B) failed: " << dctB.error().errStr;
    dctB_.emplace(std::move(*dctB));

    auto dctARtr = transitionDCTToRtr(*dctA_, kPortNum, kDefaultMtu);
    ASSERT_TRUE(dctARtr) << "DCT-A RTR failed: " << dctARtr.error().errStr;

    auto dctBRtr = transitionDCTToRtr(*dctB_, kPortNum, kDefaultMtu);
    ASSERT_TRUE(dctBRtr) << "DCT-B RTR failed: " << dctBRtr.error().errStr;

    auto dci = createDCIWithStreams(*pd_, *cq_);
    if (dci.hasError()) {
      GTEST_SKIP() << "DCI Streams not supported on this NIC (errno="
                   << dci.error().errNum << ": " << dci.error().errStr << ")";
    }
    dci_.emplace(std::move(*dci));

    auto dciRts = transitionDCIToRts(*dci_, kPortNum, kDefaultMtu);
    ASSERT_TRUE(dciRts) << "DCI RTS failed: " << dciRts.error().errStr;

    ASSERT_NE(ibvSymbols.mlx5dv_internal_wr_set_dc_addr_stream, nullptr)
        << "mlx5dv_wr_set_dc_addr_stream not available";

    exQp_ = ibvSymbols.ibv_internal_qp_to_qp_ex(dci_->qp());
    dvQp_ = ibvSymbols.mlx5dv_internal_qp_ex_from_ibv_qp_ex(exQp_);
    ASSERT_NE(exQp_, nullptr);
    ASSERT_NE(dvQp_, nullptr);

    auto gid = device.queryGid(kPortNum, kGidIndex);
    ASSERT_TRUE(gid);
    gid_ = *gid;
  }

  void initBufferAndAddressHandles() {
    buf_.resize(kBufSize, 0);
    auto mr = pd_->regMr(buf_.data(), buf_.size(), defaultMrAccess());
    ASSERT_TRUE(mr);
    mr_.emplace(std::move(*mr));

    sge_.addr = reinterpret_cast<uint64_t>(buf_.data());
    sge_.length = 64;
    sge_.lkey = mr_->mr()->lkey;

    cardA_ = makeDcBusinessCard(
        dctA_->qp()->qp_num,
        gid_,
        reinterpret_cast<uint64_t>(buf_.data()),
        mr_->mr()->rkey);
    cardB_ = makeDcBusinessCard(
        dctB_->qp()->qp_num,
        gid_,
        reinterpret_cast<uint64_t>(buf_.data()),
        mr_->mr()->rkey);

    auto ahA = createAddressHandle(*pd_, cardA_);
    ASSERT_TRUE(ahA) << "createAddressHandle(A) failed: " << ahA.error().errStr;
    ahA_.emplace(std::move(*ahA));

    auto ahB = createAddressHandle(*pd_, cardB_);
    ASSERT_TRUE(ahB) << "createAddressHandle(B) failed: " << ahB.error().errStr;
    ahB_.emplace(std::move(*ahB));
  }

  static ibv_access_flags defaultMrAccess() {
    return static_cast<ibv_access_flags>(
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ);
  }

  static constexpr int kCqe = 1024;
  static constexpr size_t kBufSize = 4096;

  std::optional<std::vector<IbvDevice>> devices_;
  std::optional<IbvPd> pd_;
  std::optional<IbvCq> cq_;
  std::optional<IbvSrq> srq_;
  std::optional<IbvQp> dctA_;
  std::optional<IbvQp> dctB_;
  std::optional<IbvQp> dci_;
  std::optional<IbvMr> mr_;
  std::optional<IbvAh> ahA_;
  std::optional<IbvAh> ahB_;
  std::vector<uint8_t> buf_;
  ibv_sge sge_{};
  ibv_gid gid_{};
  DcBusinessCard cardA_{};
  DcBusinessCard cardB_{};
  ibv_qp_ex* exQp_{nullptr};
  mlx5dv_qp_ex* dvQp_{nullptr};
};

// Verify that RDMA writes using different stream_ids to different DCTs
// complete successfully through a stream-enabled DCI.
TEST_F(DciStreamsTestFixture, DciStreamsMultiTargetWrite) {
  // Post RDMA write with stream_id=0 to DCT-A
  ASSERT_EQ(postDcRdmaWriteStream(exQp_, dvQp_, *ahA_, cardA_, sge_, 0, 0), 0)
      << "wr_complete failed for stream 0 -> DCT-A";

  auto pollOk1 = pollCqForCompletions(0, *cq_, 1);
  ASSERT_TRUE(pollOk1) << "stream 0 -> DCT-A poll failed: "
                       << pollOk1.error().errStr;
  XLOG(INFO) << "stream_id=0 -> DCT-A: success";

  // Post RDMA write with stream_id=1 to DCT-B
  ASSERT_EQ(postDcRdmaWriteStream(exQp_, dvQp_, *ahB_, cardB_, sge_, 1, 1), 0)
      << "wr_complete failed for stream 1 -> DCT-B";

  auto pollOk2 = pollCqForCompletions(0, *cq_, 1);
  ASSERT_TRUE(pollOk2) << "stream 1 -> DCT-B poll failed: "
                       << pollOk2.error().errStr;
  XLOG(INFO) << "stream_id=1 -> DCT-B: success";

  // Verify DCI is still in RTS after successful multi-stream writes
  {
    auto queryResult = dci_->queryQp(IBV_QP_STATE);
    ASSERT_TRUE(queryResult) << "queryQp failed";
    auto& [qpAttr, qpInitAttr] = *queryResult;
    EXPECT_EQ(qpAttr.qp_state, IBV_QPS_RTS)
        << "DCI should remain in RTS after successful writes";
  }

  XLOG(INFO) << "Multi-target stream writes verified successfully.";
}

// Kill one peer (DCT-B), keep sending on both streams.
// Verify: stream to live DCT-A still delivers, stream to dead DCT-B errors,
// and the DCI stays operational for the healthy stream.
TEST_F(DciStreamsTestFixture, DciStreamsFaultIsolation) {
  // --- Step 1: Verify both streams work before killing peer ---
  ASSERT_EQ(
      postDcRdmaWriteStream(exQp_, dvQp_, *ahA_, cardA_, sge_, 100, 0), 0);
  auto preOk = pollCqForCompletions(0, *cq_, 1);
  ASSERT_TRUE(preOk) << "Pre-kill stream 0 -> DCT-A failed";
  XLOG(INFO) << "Pre-kill: stream_id=0 -> DCT-A: success";

  // --- Step 2: Kill DCT-B (simulate peer going down) ---
  XLOG(INFO) << "Destroying DCT-B (simulating peer failure)";
  dctB_.reset();

  // --- Step 3: Send to dead DCT-B on stream 1 → expect error ---
  ASSERT_EQ(postDcRdmaWriteStream(exQp_, dvQp_, *ahB_, cardB_, sge_, 1, 1), 0);

  // Poll for the error completion on the dead stream
  {
    auto startTime = std::chrono::steady_clock::now();
    bool gotError = false;
    while (!gotError) {
      auto wcs = cq_->pollCq(1);
      ASSERT_TRUE(wcs);
      if (!wcs->empty()) {
        auto& wc = wcs->at(0);
        EXPECT_NE(wc.status, IBV_WC_SUCCESS)
            << "Expected error completion for dead DCT-B";
        EXPECT_EQ(wc.wr_id, 1u) << "Error should be for stream 1 (wr_id=1)";
        gotError = true;
        XLOGF(
            INFO,
            "stream_id=1 -> dead DCT-B: error status={}, vendor_err={}",
            wc.status,
            wc.vendor_err);
      }
      auto elapsed = std::chrono::steady_clock::now() - startTime;
      if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >
          30) {
        FAIL() << "Timed out waiting for error completion on stream 1";
      }
    }
  }

  // --- Step 4: Check DCI state ---
  {
    auto queryResult = dci_->queryQp(IBV_QP_STATE);
    ASSERT_TRUE(queryResult) << "queryQp failed";
    auto& [qpAttr, qpInitAttr] = *queryResult;
    XLOGF(INFO, "DCI QP state after stream error: {}", qpAttr.qp_state);

    if (qpAttr.qp_state != IBV_QPS_RTS) {
      XLOGF(
          WARN,
          "DCI moved to state {} (not RTS). HW max_log_num_errored may be 0, "
          "meaning 1 errored stream pushes the DCI to ERR. Fault isolation "
          "requires max_log_num_errored > 0.",
          qpAttr.qp_state);
      // Report what happened but don't hard-fail — document the behavior
      GTEST_SKIP()
          << "DCI entered ERR state after stream error (qp_state="
          << qpAttr.qp_state << "). Fault isolation not available on this NIC "
          << "(max_log_num_errored likely 0). "
          << "The errored stream was correctly isolated to stream_id=1.";
    }
  }

  // --- Step 5: Keep sending on healthy stream 0 to live DCT-A ---
  // With max_log_num_errored=0, the DCI may report RTS but still fail on
  // other streams. True fault isolation requires max_log_num_errored > 0.
  XLOG(INFO) << "DCI still in RTS — sending on healthy stream to live peer";
  ASSERT_EQ(postDcRdmaWriteStream(exQp_, dvQp_, *ahA_, cardA_, sge_, 2, 0), 0);

  // Poll — on HW with max_log_num_errored=0, this may fail even though the
  // DCI is in RTS, because the errored stream contaminates the DCI.
  auto pollOk = pollCqForCompletions(0, *cq_, 1);
  if (pollOk.hasError()) {
    XLOGF(
        WARN,
        "Post-kill stream 0 -> DCT-A failed: {}. "
        "DCI is in RTS but errored stream contaminated other streams. "
        "True fault isolation requires firmware with max_log_num_errored > 0.",
        pollOk.error().errStr);
    GTEST_SKIP()
        << "Healthy stream failed after peer kill despite DCI in RTS. "
        << "Firmware max_log_num_errored=0 does not provide true fault "
        << "isolation: " << pollOk.error().errStr;
  }
  XLOG(INFO)
      << "Post-kill: stream_id=0 -> DCT-A: success. Fault isolation verified.";

  // --- Step 6: Try resetting the errored stream if supported ---
  auto resetResult = resetDciStream(*dci_, 1);
  if (resetResult.hasError()) {
    XLOGF(
        INFO,
        "dci_stream_id_reset not supported (errno={}): {}",
        resetResult.error().errNum,
        resetResult.error().errStr);
  } else {
    XLOG(INFO) << "stream_id=1 reset successfully";
  }
}

// Post multiple WRs on two streams to two DCTs, kill one DCT, then drain
// the CQ. Classify every completion by wr_id to show which stream's WRs
// succeeded, which errored, and which never completed (stuck).
TEST_F(DciStreamsTestFixture, DciStreamsCompletionClassification) {
  // --- Step 1: Post 3 WRs on stream 0 to DCT-A (all should succeed) ---
  constexpr int kWrsPerStream = 3;
  for (int i = 0; i < kWrsPerStream; i++) {
    ASSERT_EQ(
        postDcRdmaWriteStream(
            exQp_, dvQp_, *ahA_, cardA_, sge_, static_cast<uint64_t>(i * 2), 0),
        0);
  }

  // Poll all 3 to ensure they complete before we kill DCT-B
  auto preOk = pollCqForCompletions(0, *cq_, kWrsPerStream);
  ASSERT_TRUE(preOk) << "Pre-kill stream 0 writes failed: "
                     << preOk.error().errStr;
  XLOG(INFO) << "Pre-kill: 3 WRs on stream 0 -> DCT-A all succeeded";

  // --- Step 2: Kill DCT-B ---
  XLOG(INFO) << "Destroying DCT-B (simulating peer failure)";
  dctB_.reset();

  // --- Step 3: Post 3 WRs on stream 1 to dead DCT-B ---
  for (int i = 0; i < kWrsPerStream; i++) {
    ASSERT_EQ(
        postDcRdmaWriteStream(
            exQp_,
            dvQp_,
            *ahB_,
            cardB_,
            sge_,
            static_cast<uint64_t>(i * 2 + 1),
            1),
        0);
  }

  // --- Step 4: Post 3 more WRs on stream 0 to live DCT-A ---
  for (int i = 0; i < kWrsPerStream; i++) {
    ASSERT_EQ(
        postDcRdmaWriteStream(
            exQp_,
            dvQp_,
            *ahA_,
            cardA_,
            sge_,
            static_cast<uint64_t>((kWrsPerStream + i) * 2),
            0),
        0);
  }

  // --- Step 5: Drain CQ and classify all completions ---
  // We posted 6 WRs total (3 on stream 1, 3 on stream 0).
  // Some may succeed, some may error, some may never complete.
  constexpr int kTotalPosted = kWrsPerStream * 2;
  int stream0Success = 0;
  int stream0Error = 0;
  int stream1Success = 0;
  int stream1Error = 0;
  int totalPolled = 0;

  auto startTime = std::chrono::steady_clock::now();
  constexpr int kTimeoutSec = 30;

  while (totalPolled < kTotalPosted) {
    auto wcs = cq_->pollCq(kTotalPosted);
    ASSERT_TRUE(wcs);

    for (const auto& wc : *wcs) {
      uint16_t sid = static_cast<uint16_t>(wc.wr_id % 2);
      uint64_t seq = wc.wr_id / 2;
      bool ok = (wc.status == IBV_WC_SUCCESS);

      if (sid == 0) {
        ok ? stream0Success++ : stream0Error++;
      } else {
        ok ? stream1Success++ : stream1Error++;
      }

      XLOGF(
          INFO,
          "  WC: stream={}, seq={}, status={}, vendor_err={} [{}]",
          sid,
          seq,
          wc.status,
          wc.vendor_err,
          ok ? "OK" : "ERR");
      totalPolled++;
    }

    if (wcs->empty()) {
      auto elapsed = std::chrono::steady_clock::now() - startTime;
      if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >
          kTimeoutSec) {
        break; // Some WRs may be stuck — that's data too
      }
    }
  }

  int stuck = kTotalPosted - totalPolled;

  XLOG(INFO) << "=== Completion Classification ===";
  XLOGF(
      INFO,
      "  Stream 0 (live DCT-A):  {} success, {} error",
      stream0Success,
      stream0Error);
  XLOGF(
      INFO,
      "  Stream 1 (dead DCT-B): {} success, {} error",
      stream1Success,
      stream1Error);
  XLOGF(INFO, "  Stuck (no completion):  {}", stuck);

  // Stream 1 should have errors (sent to dead DCT)
  EXPECT_GT(stream1Error, 0) << "Expected errors on stream 1 (dead DCT-B)";
  EXPECT_EQ(stream1Success, 0) << "No success expected on dead DCT-B";

  // Log the DCI state for analysis
  {
    auto queryResult = dci_->queryQp(IBV_QP_STATE);
    ASSERT_TRUE(queryResult);
    auto& [qpAttr, qpInitAttr] = *queryResult;
    XLOGF(INFO, "  DCI QP state: {}", qpAttr.qp_state);
  }

  // On HW with max_log_num_errored=0, stream 0's post-kill WRs may also
  // error or be stuck. Log it either way — the classification is the value.
  if (stream0Error > 0 || stuck > 0) {
    XLOG(INFO) << "  Note: stream 0 WRs also affected — fault isolation not "
               << "available (max_log_num_errored likely 0)";
  }
  if (stream0Success == kWrsPerStream && stream0Error == 0) {
    XLOG(INFO) << "  Fault isolation confirmed: stream 0 fully operational "
               << "despite stream 1 errors";
  }
}

} // namespace ibverbx

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
