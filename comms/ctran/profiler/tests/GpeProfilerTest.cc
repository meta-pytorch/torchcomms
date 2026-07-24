// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/profiler/GpeProfiler.h"

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/ctran/profiler/GpeProfilerReport.h"
#include "comms/ctran/profiler/tests/MockGpeProfilerReporter.h"

using namespace ::testing;

namespace ctran {

namespace {

// Wires the profiler to a captured-row vector via MockGpeProfilerReporter.
// Returned mock pointer remains valid until the profiler is destroyed.
struct ProfilerHarness {
  std::unique_ptr<GpeProfiler> profiler;
  MockGpeProfilerReporter* mockPtr{nullptr};
  std::vector<GpeProfilerReport>* rowsPtr{nullptr};
  std::shared_ptr<::comms::fault_tolerance::Abort> abort;
};

ProfilerHarness makeHarness(
    std::vector<GpeProfilerReport>& rows,
    int samplingWeight,
    std::shared_ptr<::comms::fault_tolerance::Abort> abort = nullptr) {
  auto mock = std::make_unique<MockGpeProfilerReporter>();
  auto* mockPtr = mock.get();
  EXPECT_CALL(*mockPtr, report(_))
      .WillRepeatedly([&](const GpeProfilerReport& r) { rows.push_back(r); });
  auto profiler = std::make_unique<GpeProfiler>(
      /*logMetaData=*/nullptr,
      /*rank=*/0,
      /*commHash=*/0xabc,
      samplingWeight,
      std::move(mock),
      abort);
  return {std::move(profiler), mockPtr, &rows, std::move(abort)};
}

} // namespace

class GpeProfilerTest : public ::testing::Test {};

// mark(ITER_START) DEFERS its Scuba emit. The row is only written
// when a later sampled / aborted emit triggers the backfill.
TEST_F(GpeProfilerTest, IterStartDefersEmitUntilBackfill) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/1);

  h.profiler->mark(GpeTracePoint::ITER_START);
  EXPECT_TRUE(rows.empty()); // deferred — no row yet

  // First sampled emit triggers backfill of ITER_START + then CMD_DEQUEUE.
  h.profiler->injectMetadata({.opCount = 1, .opType = 0});
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);

  ASSERT_EQ(rows.size(), 2u);
  EXPECT_EQ(rows[0].tracePoint, GpeTracePoint::ITER_START);
  EXPECT_EQ(rows[0].iterUs, 0u);
  EXPECT_EQ(rows[0].durationUs, 0u);
  EXPECT_FALSE(rows[0].aborted);
}

// Sampled iter (samplingWeight=1, opCount=1): all 5 happy-path tracepoints
// land in chronological order. ITER_START gets backfilled by CMD_DEQUEUE.
TEST_F(GpeProfilerTest, SampledIterEmitsAllFiveRowsInOrder) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/1);

  h.profiler->mark(GpeTracePoint::ITER_START);
  h.profiler->injectMetadata({.opCount = 1, .opType = 7});
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::WAIT_KERNEL);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::HOST_ALGO);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::TERMINATE_KERNEL);

  ASSERT_EQ(rows.size(), 5u);
  EXPECT_EQ(rows[0].tracePoint, GpeTracePoint::ITER_START);
  EXPECT_EQ(rows[1].tracePoint, GpeTracePoint::CMD_DEQUEUE);
  EXPECT_EQ(rows[2].tracePoint, GpeTracePoint::WAIT_KERNEL);
  EXPECT_EQ(rows[3].tracePoint, GpeTracePoint::HOST_ALGO);
  EXPECT_EQ(rows[4].tracePoint, GpeTracePoint::TERMINATE_KERNEL);

  EXPECT_EQ(rows[0].iterUs, 0u);
  EXPECT_EQ(rows[0].durationUs, 0u);
  // Identity fields pass through from makeHarness ctor args
  // (commHash=0xabc, rank=0, logMetaData=nullptr) to every row.
  EXPECT_EQ(rows[0].commHash, 0xabcULL);
  EXPECT_EQ(rows[0].rank, 0);
  EXPECT_EQ(rows[0].logMetaData, nullptr);
  for (size_t i = 1; i < rows.size(); ++i) {
    EXPECT_EQ(rows[i].durationUs, rows[i].iterUs - rows[i - 1].iterUs);
    EXPECT_GT(rows[i].iterUs, rows[i - 1].iterUs);
    EXPECT_EQ(rows[i].opCount, 1u);
    EXPECT_EQ(rows[i].opType, 7);
    EXPECT_FALSE(rows[i].aborted);
  }
}

// Unsampled, non-aborted iter (weight=1000, opCount=1): the gate rejects
// every emit, so the reporter sees zero rows. (Replaces the old
// "anchor pair always emits" semantic.)
TEST_F(GpeProfilerTest, UnsampledNonAbortedIterEmitsZeroRows) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/1000);

  h.profiler->mark(GpeTracePoint::ITER_START);
  h.profiler->injectMetadata({.opCount = 1, .opType = 1}); // 1 % 1000 != 0
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);
  h.profiler->mark(GpeTracePoint::WAIT_KERNEL);
  h.profiler->mark(GpeTracePoint::HOST_ALGO);
  h.profiler->mark(GpeTracePoint::TERMINATE_KERNEL);

  EXPECT_TRUE(rows.empty());
}

// samplingWeight=0 disables happy-path tracing entirely. Without an abort
// marker or live abort, no rows are emitted.
TEST_F(GpeProfilerTest, SamplingWeightZeroEmitsZeroRowsWithoutAbort) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/0);

  h.profiler->mark(GpeTracePoint::ITER_START);
  h.profiler->injectMetadata({.opCount = 1, .opType = 0});
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);
  h.profiler->mark(GpeTracePoint::WAIT_KERNEL);
  h.profiler->mark(GpeTracePoint::HOST_ALGO);

  EXPECT_TRUE(rows.empty());
}

// ALGO_ABORTED marker on an unsampled iter triggers backfill of all
// stamped predecessors, then the marker itself. Marker carries the
// reason in `message` and `aborted=true`; backfilled predecessors
// have `aborted=false` and empty message.
TEST_F(GpeProfilerTest, AbortMarkerBackfillsAllStampedPredecessors) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/1000); // unsampled

  h.profiler->mark(GpeTracePoint::ITER_START);
  h.profiler->injectMetadata({.opCount = 1, .opType = 1});
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::WAIT_KERNEL);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::HOST_ALGO);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::TERMINATE_KERNEL);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::ALGO_ABORTED, "timeout");

  // 5 backfilled predecessors + ALGO_ABORTED marker = 6 rows.
  ASSERT_EQ(rows.size(), 6u);
  EXPECT_EQ(rows[0].tracePoint, GpeTracePoint::ITER_START);
  EXPECT_EQ(rows[1].tracePoint, GpeTracePoint::CMD_DEQUEUE);
  EXPECT_EQ(rows[2].tracePoint, GpeTracePoint::WAIT_KERNEL);
  EXPECT_EQ(rows[3].tracePoint, GpeTracePoint::HOST_ALGO);
  EXPECT_EQ(rows[4].tracePoint, GpeTracePoint::TERMINATE_KERNEL);
  EXPECT_EQ(rows[5].tracePoint, GpeTracePoint::ALGO_ABORTED);

  // Marker carries reason + aborted=true.
  EXPECT_TRUE(rows[5].aborted);
  EXPECT_EQ(rows[5].message, std::string_view{"timeout"});
  // Backfilled predecessors carry empty message + aborted=false.
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_FALSE(rows[i].aborted);
    EXPECT_EQ(rows[i].message, std::string_view{});
  }
  // iterUs strictly monotonic.
  EXPECT_EQ(rows[0].iterUs, 0u);
  for (size_t i = 1; i < rows.size(); ++i) {
    EXPECT_GT(rows[i].iterUs, rows[i - 1].iterUs);
    EXPECT_EQ(rows[i].durationUs, rows[i].iterUs - rows[i - 1].iterUs);
  }
}

// Backfill skips slots that were never stamped (e.g., abort fires before
// some kernel-side phases were marked). The resulting waterfall is
// "every tracepoint that did get stamped, in enum order".
TEST_F(GpeProfilerTest, BackfillSkipsUnstampedSlots) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/1000); // unsampled

  h.profiler->mark(GpeTracePoint::ITER_START);
  h.profiler->injectMetadata({.opCount = 1, .opType = 0});
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);
  // Skip WAIT_KERNEL entirely — stamped_[WAIT_KERNEL] stays unset.
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::HOST_ALGO);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::ALGO_ABORTED, "external");

  // ITER_START + CMD_DEQUEUE + HOST_ALGO + ALGO_ABORTED = 4 rows.
  ASSERT_EQ(rows.size(), 4u);
  EXPECT_EQ(rows[0].tracePoint, GpeTracePoint::ITER_START);
  EXPECT_EQ(rows[1].tracePoint, GpeTracePoint::CMD_DEQUEUE);
  EXPECT_EQ(rows[2].tracePoint, GpeTracePoint::HOST_ALGO);
  EXPECT_EQ(rows[3].tracePoint, GpeTracePoint::ALGO_ABORTED);
  // No WAIT_KERNEL row — its slot was never stamped.
}

// Live abort: when the comm's Abort handle is set mid-iter, the next
// mark() bypasses the sampling gate AND backfills any stamped predecessors.
TEST_F(GpeProfilerTest, LiveAbortBypassesSamplingMidIter) {
  std::vector<GpeProfilerReport> rows;
  auto abort = ::comms::fault_tolerance::createAbort(/*enabled=*/true);
  auto h = makeHarness(rows, /*samplingWeight=*/1000, abort);

  h.profiler->mark(GpeTracePoint::ITER_START);
  h.profiler->injectMetadata({.opCount = 1, .opType = 0}); // unsampled
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::WAIT_KERNEL);

  EXPECT_TRUE(rows.empty()); // pre-abort + unsampled — nothing emitted

  // Abort flips. Next mark must emit + backfill all 3 predecessors.
  abort->Set();
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::HOST_ALGO);

  ASSERT_EQ(rows.size(), 4u);
  EXPECT_EQ(rows[0].tracePoint, GpeTracePoint::ITER_START);
  EXPECT_EQ(rows[1].tracePoint, GpeTracePoint::CMD_DEQUEUE);
  EXPECT_EQ(rows[2].tracePoint, GpeTracePoint::WAIT_KERNEL);
  EXPECT_EQ(rows[3].tracePoint, GpeTracePoint::HOST_ALGO);

  // Subsequent marks emit normally (live abort still set), no backfill
  // (already caught up).
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::TERMINATE_KERNEL);
  ASSERT_EQ(rows.size(), 5u);
  EXPECT_EQ(rows[4].tracePoint, GpeTracePoint::TERMINATE_KERNEL);
}

// markAborted before any mark(ITER_START) is silently dropped (very early
// shutdown / pre-iter state).
TEST_F(GpeProfilerTest, MarkAbortedBeforeIterStartIsDropped) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/1);

  h.profiler->mark(GpeTracePoint::ALGO_ABORTED, "too_early");
  EXPECT_TRUE(rows.empty());
}

// mark(ALGO_ABORTED) with an empty message is still the abort marker —
// it always emits (and triggers backfill) regardless of sampling. The
// `message` field on the resulting row is just empty.
TEST_F(GpeProfilerTest, AlgoAbortedWithEmptyMessageStillTriggersEmit) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/1000); // unsampled

  h.profiler->mark(GpeTracePoint::ITER_START);
  h.profiler->injectMetadata({.opCount = 1, .opType = 0});
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  // No message — still treated as marker, still triggers backfill.
  h.profiler->mark(GpeTracePoint::ALGO_ABORTED);

  // ITER_START + CMD_DEQUEUE (both backfilled) + ALGO_ABORTED = 3 rows.
  ASSERT_EQ(rows.size(), 3u);
  EXPECT_EQ(rows[0].tracePoint, GpeTracePoint::ITER_START);
  EXPECT_EQ(rows[1].tracePoint, GpeTracePoint::CMD_DEQUEUE);
  EXPECT_EQ(rows[2].tracePoint, GpeTracePoint::ALGO_ABORTED);
  EXPECT_TRUE(rows[2].aborted);
  EXPECT_EQ(rows[2].message, std::string_view{}); // empty
}

// TERMINATE_CMD is the GPE-exit signal: always emits regardless of
// sampling, backfills ITER_START + CMD_DEQUEUE so the final iter
// timeline lands in Scuba even when the comm was never sampled.
// aborted=false on the row (TERMINATE is a normal exit, not an abort).
TEST_F(GpeProfilerTest, TerminateCmdAlwaysEmitsAndBackfills) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/1000); // unsampled

  h.profiler->mark(GpeTracePoint::ITER_START);
  // No injectMetadata — TERMINATE cmd has no opCount/opType.
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::TERMINATE_CMD);

  // ITER_START + CMD_DEQUEUE (both backfilled) + TERMINATE_CMD = 3 rows.
  ASSERT_EQ(rows.size(), 3u);
  EXPECT_EQ(rows[0].tracePoint, GpeTracePoint::ITER_START);
  EXPECT_EQ(rows[1].tracePoint, GpeTracePoint::CMD_DEQUEUE);
  EXPECT_EQ(rows[2].tracePoint, GpeTracePoint::TERMINATE_CMD);
  EXPECT_FALSE(rows[2].aborted);
}

// Marks before mark(ITER_START) are silently dropped.
TEST_F(GpeProfilerTest, NonAnchorMarksBeforeIterStartAreDropped) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/1);

  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE); // dropped
  h.profiler->mark(GpeTracePoint::HOST_ALGO); // dropped
  EXPECT_TRUE(rows.empty());
}

// debugString walks the per-iter stamped_ timeline regardless of sampling
// or backfill. Reset by mark(ITER_START).
TEST_F(GpeProfilerTest, DebugStringWalksStampedTimeline) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/1);

  EXPECT_EQ(h.profiler->debugString(), "");

  h.profiler->mark(GpeTracePoint::ITER_START);
  EXPECT_EQ(h.profiler->debugString(), "");

  h.profiler->injectMetadata({.opCount = 1, .opType = 0});
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);
  const auto s1 = h.profiler->debugString();
  EXPECT_NE(s1.find("cmd_dequeueUs="), std::string::npos);
  EXPECT_EQ(s1.find("wait_kernelUs="), std::string::npos);

  // Next iter resets the timeline.
  h.profiler->mark(GpeTracePoint::ITER_START);
  EXPECT_EQ(h.profiler->debugString(), "");
}

// debugString lifecycle: full waterfall is preserved on aborted iters
// (regardless of sampling), since stamped_[] is updated on every mark.
TEST_F(GpeProfilerTest, DebugString_Lifecycle_UnsampledAbortLate) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/1000);

  h.profiler->mark(GpeTracePoint::ITER_START);
  h.profiler->injectMetadata({.opCount = 1, .opType = 0});
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::WAIT_KERNEL);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::HOST_ALGO);
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::TERMINATE_KERNEL);
  h.profiler->mark(GpeTracePoint::ALGO_ABORTED, "timeout");

  // debugString MUST contain every enum-adjacent delta.
  const auto s = h.profiler->debugString();
  EXPECT_NE(s.find("cmd_dequeueUs="), std::string::npos);
  EXPECT_NE(s.find("wait_kernelUs="), std::string::npos);
  EXPECT_NE(s.find("host_algoUs="), std::string::npos);
  EXPECT_NE(s.find("terminate_kernelUs="), std::string::npos);
  EXPECT_NE(s.find("algo_abortedUs="), std::string::npos);
}

// Two iters in sequence: each ITER_START resets the Watch + reported state.
// In sampled iters, the second ITER_START is backfilled from the second
// CMD_DEQUEUE just like the first.
TEST_F(GpeProfilerTest, SecondIterResetsState) {
  std::vector<GpeProfilerReport> rows;
  auto h = makeHarness(rows, /*samplingWeight=*/1);

  h.profiler->mark(GpeTracePoint::ITER_START);
  h.profiler->injectMetadata({.opCount = 1, .opType = 0});
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);

  // Second iter.
  h.profiler->mark(GpeTracePoint::ITER_START);
  h.profiler->injectMetadata({.opCount = 2, .opType = 0});
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  h.profiler->mark(GpeTracePoint::CMD_DEQUEUE);

  ASSERT_EQ(rows.size(), 4u);
  EXPECT_EQ(rows[0].tracePoint, GpeTracePoint::ITER_START);
  EXPECT_EQ(rows[0].iterUs, 0u);
  EXPECT_EQ(rows[1].tracePoint, GpeTracePoint::CMD_DEQUEUE);
  EXPECT_EQ(rows[2].tracePoint, GpeTracePoint::ITER_START);
  EXPECT_EQ(rows[2].iterUs, 0u);
  EXPECT_EQ(rows[3].tracePoint, GpeTracePoint::CMD_DEQUEUE);
  EXPECT_EQ(rows[1].durationUs, rows[1].iterUs);
  EXPECT_EQ(rows[3].durationUs, rows[3].iterUs);
}

} // namespace ctran
