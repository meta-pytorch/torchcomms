// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/gpe/tests/CtranGpeFaultToleranceUTBase.h"
#include "comms/ctran/profiler/GpeProfilerReport.h"
#include "comms/ctran/profiler/IGpeProfilerReporter.h"
#include "comms/ctran/profiler/tests/MockGpeProfilerReporter.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::fttesting {

// ---------------------------------------------------------------------------
// GpeProfiler integration tests (T270778705).
//
// These tests inject a MockGpeProfilerReporter through the 3-arg CtranGpe ctor
// and assert on the per-tracepoint Scuba rows the GPE thread emits eagerly
// from inside ctran::GpeProfiler::mark(). Per the universal in-order backfill
// rule, ITER_START + CMD_DEQUEUE only emit when a downstream mark (sampled
// row, ALGO_ABORTED marker, or always-on TERMINATE_CMD) triggers a
// backfill; WAIT_KERNEL/HOST_ALGO/TERMINATE_KERNEL respect sampling;
// ALGO_ABORTED and TERMINATE_CMD are always-on.
// ---------------------------------------------------------------------------

// NCCL_CTRAN_GPE_PROFILING_ENABLE defaults to false in production, which
// nulls the reporter at the CtranGpe integration layer regardless of caller
// injection. The integration tests here inject a mock reporter and would
// see zero report() calls without flipping the cvar. SetUp/TearDown override
// the cvar around the test body and restore it after.
class GpeProfilerIntegrationTest : public CtranGpeFaultToleranceTestBase {
 protected:
  void SetUp() override {
    SetUpInternal(/*abortEnabled=*/true);
    savedProfilingEnable_ = NCCL_CTRAN_GPE_PROFILING_ENABLE;
    NCCL_CTRAN_GPE_PROFILING_ENABLE = true;
  }
  void TearDown() override {
    NCCL_CTRAN_GPE_PROFILING_ENABLE = savedProfilingEnable_;
    CtranGpeFaultToleranceTestBase::TearDown();
  }

 private:
  bool savedProfilingEnable_{false};
};

// Drives the abort path (HostDetectedTimeout style) and asserts that the
// reporter sees the anchor pair (ITER_START + CMD_DEQUEUE) and the
// force-emitted ALGO_ABORTED row carrying aborted=true and reason="timeout".
TEST_F(GpeProfilerIntegrationTest, GpeProfiler_RecordsAlgoAbortedOnAbort) {
  ASSERT_TRUE(ctranComm->abortEnabled());

  auto mockReporter = std::make_unique<::ctran::MockGpeProfilerReporter>();
  auto* mockPtr = mockReporter.get();

  std::vector<::ctran::GpeProfilerReport> rows;
  EXPECT_CALL(*mockPtr, report(::testing::_))
      .WillRepeatedly(
          [&](const ::ctran::GpeProfilerReport& r) { rows.push_back(r); });

  auto gpe = std::make_unique<CtranGpe>(
      cudaDev, ctranComm.get(), std::move(mockReporter));

  FtTestSync sync;
  sync.setTimeout();
  this->launchKernelFn(
      gpe.get(),
      (void*)CtranGpeTestFtBaseKernel,
      stream,
      &sync,
      /*timeout=*/kHostAlgoFnWait - std::chrono::milliseconds(500));

  // Wait for kernel + abort handling to complete.
  tryQueryStreamFor(stream, kHostAlgoFnWait + std::chrono::milliseconds(2000));
  ASSERT_TRUE(ctranComm->testAbort());

  // Tear down gpe — ~Impl joins the GPE thread.
  gpe.reset();

  ASSERT_FALSE(rows.empty()) << "expected at least the anchor pair";

  // Anchor pair always emits.
  bool sawIterStart = false;
  bool sawCmdDequeue = false;
  bool sawAlgoAborted = false;
  for (const auto& r : rows) {
    if (r.tracePoint == ::ctran::GpeTracePoint::ITER_START) {
      sawIterStart = true;
      EXPECT_EQ(r.iterUs, 0UL);
      EXPECT_EQ(r.durationUs, 0UL);
      EXPECT_FALSE(r.aborted);
    } else if (r.tracePoint == ::ctran::GpeTracePoint::CMD_DEQUEUE) {
      sawCmdDequeue = true;
      EXPECT_FALSE(r.aborted);
    } else if (r.tracePoint == ::ctran::GpeTracePoint::ALGO_ABORTED) {
      sawAlgoAborted = true;
      EXPECT_TRUE(r.aborted);
      EXPECT_EQ(r.message, std::string_view{"timeout"});
    }
  }
  EXPECT_TRUE(sawIterStart);
  EXPECT_TRUE(sawCmdDequeue);
  EXPECT_TRUE(sawAlgoAborted);
}

// Happy-path iter at default samplingWeight=1000 with opCount=100 is NOT
// sampled, so the user iter emits nothing (universal backfill rule — no
// downstream emit fires, so ITER_START + CMD_DEQUEUE stay deferred and
// WAIT_KERNEL/HOST_ALGO/TERMINATE_KERNEL drop). The trailing TERMINATE
// iter fires its always-on TERMINATE_CMD, which backfills that iter's
// ITER_START + CMD_DEQUEUE. So the expected emit set is exactly
// {ITER_START, CMD_DEQUEUE, TERMINATE_CMD} — no collective-phase rows.
TEST_F(
    GpeProfilerIntegrationTest,
    GpeProfiler_UnsampledHappyPathEmitsAnchorPairOnly) {
  ASSERT_TRUE(ctranComm->abortEnabled());

  auto mockReporter = std::make_unique<::ctran::MockGpeProfilerReporter>();
  auto* mockPtr = mockReporter.get();

  std::vector<::ctran::GpeProfilerReport> rows;
  EXPECT_CALL(*mockPtr, report(::testing::_))
      .WillRepeatedly(
          [&](const ::ctran::GpeProfilerReport& r) { rows.push_back(r); });

  auto gpe = std::make_unique<CtranGpe>(
      cudaDev, ctranComm.get(), std::move(mockReporter));

  // Run a happy-path iteration (no abort).
  FtTestSync sync;
  this->launchKernelFn(
      gpe.get(), (void*)CtranGpeTestFtEnabledOobTerminateKernel, stream, &sync);
  *oobKernelTerminateFlag = true;
  sync.signal();
  tryQueryStreamFor(stream, kHostAlgoFnWait + std::chrono::milliseconds(1000));

  gpe.reset();

  EXPECT_FALSE(ctranComm->testAbort());
  bool sawCollectivePhaseRow = false;
  for (const auto& r : rows) {
    EXPECT_FALSE(r.aborted);
    switch (r.tracePoint) {
      case ::ctran::GpeTracePoint::ITER_START:
      case ::ctran::GpeTracePoint::CMD_DEQUEUE:
      case ::ctran::GpeTracePoint::TERMINATE_CMD:
        // Anchor pair (backfilled by TERMINATE_CMD) + the TERMINATE_CMD
        // itself. All expected on a happy unsampled-then-terminate
        // sequence.
        break;
      default:
        sawCollectivePhaseRow = true;
        break;
    }
  }
  EXPECT_FALSE(sawCollectivePhaseRow)
      << "unsampled iter should not emit WAIT_KERNEL/HOST_ALGO/TERMINATE_KERNEL/ALGO_ABORTED";
}

// TERMINATE-only sequence (no real cmds). The TERMINATE iter triggers
// the always-on TERMINATE_CMD mark, which backfills the preceding
// ITER_START + CMD_DEQUEUE per the universal in-order backfill rule.
// Expected emits: exactly {ITER_START, CMD_DEQUEUE, TERMINATE_CMD}. No
// ALGO_ABORTED.
TEST_F(GpeProfilerIntegrationTest, GpeProfiler_TerminateIterEmitsAnchorPair) {
  ASSERT_TRUE(ctranComm->abortEnabled());

  auto mockReporter = std::make_unique<::ctran::MockGpeProfilerReporter>();
  auto* mockPtr = mockReporter.get();

  std::vector<::ctran::GpeProfilerReport> rows;
  EXPECT_CALL(*mockPtr, report(::testing::_))
      .WillRepeatedly(
          [&](const ::ctran::GpeProfilerReport& r) { rows.push_back(r); });

  auto gpe = std::make_unique<CtranGpe>(
      cudaDev, ctranComm.get(), std::move(mockReporter));

  // No collectives submitted; immediately tear down. ~Impl enqueues a
  // TERMINATE cmd and joins the thread.
  gpe.reset();
  EXPECT_FALSE(ctranComm->testAbort());

  for (const auto& r : rows) {
    EXPECT_FALSE(r.aborted);
    EXPECT_TRUE(
        r.tracePoint == ::ctran::GpeTracePoint::ITER_START ||
        r.tracePoint == ::ctran::GpeTracePoint::CMD_DEQUEUE ||
        r.tracePoint == ::ctran::GpeTracePoint::TERMINATE_CMD)
        << "TERMINATE iter emitted unexpected tracePoint";
  }
}

// External abort before TERMINATE: gpe is destroyed while the comm's
// abort flag is set but no collective ran. The SCOPE_EXIT on the
// TERMINATE iter sees testAbort() == true and logs ("now TERMINATE"
// flavor) but must NOT stamp ALGO_ABORTED — that would land later in
// wall-clock than the already-stamped TERMINATE_CMD. The abort flag
// must remain set across cancelTimeout().
TEST_F(
    GpeProfilerIntegrationTest,
    GpeProfiler_TerminateAfterAbortSkipsAlgoAborted) {
  ASSERT_TRUE(ctranComm->abortEnabled());

  auto mockReporter = std::make_unique<::ctran::MockGpeProfilerReporter>();
  auto* mockPtr = mockReporter.get();

  std::vector<::ctran::GpeProfilerReport> rows;
  EXPECT_CALL(*mockPtr, report(::testing::_))
      .WillRepeatedly(
          [&](const ::ctran::GpeProfilerReport& r) { rows.push_back(r); });

  auto gpe = std::make_unique<CtranGpe>(
      cudaDev, ctranComm.get(), std::move(mockReporter));

  // Externally abort the comm before teardown — TERMINATE will be the
  // only cmd the GPE thread sees.
  ctranComm->setAbort();
  ASSERT_TRUE(ctranComm->testAbort());

  gpe.reset();

  // Abort state must persist across cancelTimeout().
  EXPECT_TRUE(ctranComm->testAbort());

  // No ALGO_ABORTED row — TERMINATE branch skips the marker.
  bool sawAlgoAborted = false;
  for (const auto& r : rows) {
    if (r.tracePoint == ::ctran::GpeTracePoint::ALGO_ABORTED) {
      sawAlgoAborted = true;
    }
    EXPECT_TRUE(
        r.tracePoint == ::ctran::GpeTracePoint::ITER_START ||
        r.tracePoint == ::ctran::GpeTracePoint::CMD_DEQUEUE ||
        r.tracePoint == ::ctran::GpeTracePoint::TERMINATE_CMD)
        << "TERMINATE-after-abort iter emitted unexpected tracePoint";
  }
  EXPECT_FALSE(sawAlgoAborted)
      << "ALGO_ABORTED must be skipped on TERMINATE to avoid out-of-order stamp";
}

} // namespace ctran::fttesting
