// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/profiler/GpeProfiler.h"

#include <utility>

#include <fmt/format.h>
#include <glog/logging.h>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/ctran/profiler/GpeProfilerReport.h"

namespace ctran {

GpeProfiler::GpeProfiler(
    const CommLogData* logMetaData,
    int rank,
    uint64_t commHash,
    int samplingWeight,
    std::unique_ptr<IGpeProfilerReporter> reporter,
    std::shared_ptr<::comms::fault_tolerance::Abort> abort)
    : logMetaData_(logMetaData),
      rank_(rank),
      commHash_(commHash),
      samplingWeight_(samplingWeight),
      reporter_(std::move(reporter)),
      abort_(std::move(abort)) {}

void GpeProfiler::mark(GpeTracePoint p, std::string_view message) {
  if (p == GpeTracePoint::ITER_START) {
    // Iter anchor: reset Watch + per-iter timeline. Stamp ITER_START
    // at iterUs=0 but DEFER its Scuba emit — the first sampled or
    // aborted emit later in this iter will backfill it via the
    // universal in-order backfill rule below.
    watch_.reset();
    lastStampUs_ = 0;
    iterAnchored_ = true;
    shouldTrace_ = false;
    for (auto& s : stamped_) {
      s.reset();
    }
    for (auto& r : reported_) {
      r = false;
    }
    stamped_[static_cast<size_t>(p)] = 0;
    return;
  }
  if (!iterAnchored_) {
    // Marks before ITER_START are silently dropped.
    return;
  }

  const uint64_t iterUs = static_cast<uint64_t>(watch_.elapsed().count());

  // Always stamp the timeline so debugString() and the backfill walk
  // reflect every mark regardless of sampling.
  stamped_[static_cast<size_t>(p)] = iterUs;

  // Single gate decision: do we emit anything for this mark?
  //   - isAlwaysOn: tracepoint bypasses sampling unconditionally
  //     (ALGO_ABORTED carries an abort reason; TERMINATE_CMD signals
  //     GPE-thread exit and emits at most once per comm lifetime)
  //   - liveAborted: comm is in abort state at this mark — bypass
  //     sampling so post-abort tracepoints always reach Scuba
  //   - shouldTrace_: per-iter sampling verdict from injectMetadata
  const bool isAlwaysOn =
      (p == GpeTracePoint::ALGO_ABORTED) || (p == GpeTracePoint::TERMINATE_CMD);
  const bool liveAborted = (abort_ != nullptr) && abort_->Test();
  const bool shouldEmit = isAlwaysOn || liveAborted || shouldTrace_;
  if (!shouldEmit) {
    return;
  }

  // Universal in-order backfill: emit any stamped-but-not-reported
  // predecessor rows in this iter so the Scuba waterfall lands in
  // chronological order. stamped_[] values are monotonically
  // increasing (assigned from a monotonic Watch), so the loop walks
  // them in true chronological order.
  for (size_t i = 0; i < static_cast<size_t>(p); ++i) {
    if (stamped_[i].has_value() && !reported_[i]) {
      const uint64_t backfillUs = stamped_[i].value();
      DCHECK_GE(backfillUs, lastStampUs_)
          << "GpeTracePoint enum order must match chronological mark() "
          << "order in gpeThreadFn; out-of-order stamp at "
          << tracePointName(static_cast<GpeTracePoint>(i));
      const uint64_t backfillDur = backfillUs - lastStampUs_;
      // Backfilled rows carry empty message (they're not the marker).
      if (reportRaw(
              static_cast<GpeTracePoint>(i),
              backfillUs,
              backfillDur,
              std::string_view{})) {
        reported_[i] = true;
        lastStampUs_ = backfillUs;
      }
    }
  }

  // Emit current row with durationUs recomputed against the new
  // lastStampUs_ (which the backfill may have advanced).
  DCHECK_GE(iterUs, lastStampUs_)
      << "GpeTracePoint enum order must match chronological mark() "
      << "order in gpeThreadFn; out-of-order stamp at " << tracePointName(p);
  const uint64_t durationUs = iterUs - lastStampUs_;
  if (reportRaw(p, iterUs, durationUs, message)) {
    reported_[static_cast<size_t>(p)] = true;
    lastStampUs_ = iterUs;
  }
}

void GpeProfiler::injectMetadata(const GpeIterMetadata& md) {
  opCount_ = md.opCount;
  opType_ = md.opType;
  shouldTrace_ = (samplingWeight_ > 0) && (md.opCount % samplingWeight_ == 0);
}

bool GpeProfiler::reportRaw(
    GpeTracePoint p,
    uint64_t iterUs,
    uint64_t durationUs,
    std::string_view message) {
  if (!reporter_) {
    return false;
  }
  const bool isAbortMarker = (p == GpeTracePoint::ALGO_ABORTED);
  const GpeProfilerReport r{
      .logMetaData = logMetaData_,
      .commHash = commHash_,
      .rank = rank_,
      .opCount = opCount_,
      .opType = opType_,
      .tracePoint = p,
      .iterUs = iterUs,
      .durationUs = durationUs,
      .aborted = isAbortMarker,
      .message = message,
  };
  reporter_->report(r);
  return true;
}

std::string GpeProfiler::debugString() const {
  // For each tracepoint i in [1, NUM_TRACE_POINTS), emit
  // "<name(i)>Us=<v>" where v = stamped_[i] - stamped_[i-1]. A
  // duration is omitted if either endpoint wasn't stamped this iter.
  std::string out;
  for (size_t i = 1; i < kNumPoints; ++i) {
    const auto& curr = stamped_[i];
    const auto& prev = stamped_[i - 1];
    if (curr.has_value() && prev.has_value()) {
      fmt::format_to(
          std::back_inserter(out),
          "{}Us={} ",
          tracePointName(static_cast<GpeTracePoint>(i)),
          *curr - *prev);
    }
  }
  if (!out.empty() && out.back() == ' ') {
    out.pop_back();
  }
  return out;
}

} // namespace ctran
