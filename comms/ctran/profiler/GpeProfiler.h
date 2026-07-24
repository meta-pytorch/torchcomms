// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/ctran/profiler/GpeProfilerReport.h"
#include "comms/ctran/profiler/IGpeProfilerReporter.h"
#include "comms/ctran/utils/StopWatch.h"

struct CommLogData;

namespace ctran {

// Per-iteration metadata supplied by the caller after cmdDequeue
// returns and before mark(CMD_DEQUEUE). Drives the per-iter sampling
// decision (opCount % samplingWeight).
struct GpeIterMetadata {
  uint64_t opCount{0};
  int opType{-1};
};

// Single-threaded per-CtranGpe::Impl tracepoint stamper for the
// gpeThreadFn loop. Writes one Scuba row per mark() call eagerly —
// no buffering — so a crash mid-iter leaves all completed marks
// visible in Scuba.
//
// Lifecycle: constructed in CtranGpe ctor once per GPE thread.
// Used by gpeThreadFn directly — no locking required because only the
// single GPE thread ever touches it.
//
// CONTRACT for callers (gpeThreadFn):
//   1. mark(GpeTracePoint::ITER_START) is the iteration anchor.
//      Resets the internal Watch and per-iter timeline. Stamps
//      ITER_START at iterUs=0 but DEFERS the Scuba emit; the row
//      will be backfilled by the first sampled / aborted emit
//      later in this iter (see "Universal in-order backfill"
//      below). If the iter never reaches a sampled / aborted
//      emit, no Scuba rows are written for it.
//   2. injectMetadata(GpeIterMetadata{...}) MUST be called after
//      cmdDequeue returns and before mark(CMD_DEQUEUE). Sets opCount
//      / opType for downstream rows AND decides this iter's sampling
//      verdict.
//   3. mark(WAIT_KERNEL / HOST_ALGO / TERMINATE_KERNEL / CMD_DEQUEUE):
//      emits iff (a) the iter was sampled by injectMetadata, OR
//      (b) `abort->Test()` is true at mark time (live-abort bypass),
//      OR (c) this is the ALGO_ABORTED marker. The per-iter timeline
//      is stamped on every mark regardless of sampling so
//      debugString() and the backfill walk see the full waterfall.
//   4. To record an abort, call mark(ALGO_ABORTED, reason) from the
//      per-iteration SCOPE_EXIT when comm->testAbort() is true. The
//      ALGO_ABORTED tracepoint always forces an emit regardless of
//      sampling. `message` (if non-empty) lands on the row for
//      downstream consumers (e.g., the McclScubaSink writes it to
//      the `message` column); empty `message` is allowed and just
//      means no reason annotation. Reason string_view lifetime
//      must outlast the call.
//
// Universal in-order backfill: whenever a mark decides to emit
// (sampled, live-aborted, or marker), it first emits any earlier
// stamped-but-not-reported tracepoints from the same iter. This
// guarantees the Scuba waterfall lands in chronological order:
//   - Sampled iter: CMD_DEQUEUE backfills ITER_START.
//   - Live-aborted iter (abort flips mid-iter): the post-abort mark
//     backfills all skipped predecessors.
//   - SCOPE_EXIT-only abort: the ALGO_ABORTED marker backfills all
//     5 stamped predecessors.
//
// Sampling: 1-in-N (based on opCount). Set samplingWeight=0 to
// disable happy-path Scuba rows entirely (only abort-marker rows
// + their backfill reach the reporter). debugString() always
// reflects every stamped tracepoint regardless of sampling.
class GpeProfiler {
 public:
  using Watch =
      utils::StopWatch<std::chrono::steady_clock, std::chrono::microseconds>;

  // reporter may be nullptr — mark() then no-ops the report() calls.
  // abort may be nullptr — disables the live-abort sampling bypass; the
  // gate then only fires for sampled iters and the ALGO_ABORTED marker.
  // Production callers obtain abort via `comm->getAbort()`.
  GpeProfiler(
      const CommLogData* logMetaData,
      int rank,
      uint64_t commHash,
      int samplingWeight,
      std::unique_ptr<IGpeProfilerReporter> reporter = nullptr,
      std::shared_ptr<::comms::fault_tolerance::Abort> abort = nullptr);

  ~GpeProfiler() = default;

  GpeProfiler(const GpeProfiler&) = delete;
  GpeProfiler& operator=(const GpeProfiler&) = delete;
  GpeProfiler(GpeProfiler&&) = delete;
  GpeProfiler& operator=(GpeProfiler&&) = delete;

  // Stamp a tracepoint and (if sampled / always-on / aborted) emit
  // one Scuba row immediately. mark(ITER_START) resets per-iter
  // state. Pass a non-empty `message` on mark(ALGO_ABORTED, ...) to
  // force-emit an abort row. Empty `message` means "no message"
  // (default).
  void mark(GpeTracePoint p, std::string_view message = {});

  // Set per-iteration metadata + decide sampling. Call once per
  // iter, after cmdDequeue and before mark(CMD_DEQUEUE).
  void injectMetadata(const GpeIterMetadata& md);

  // Format the current iter's per-phase latencies as
  // "<name1>Us=<v1> <name2>Us=<v2> ..." for inclusion in the
  // SCOPE_EXIT abort log line. Reflects every stamped tracepoint,
  // independent of sampling. Only adjacent-pair durations whose
  // BOTH endpoints were stamped this iter are emitted; missing
  // endpoints (e.g., abort-early before kernel start) are skipped.
  // Returns "" when no adjacent pair is available.
  std::string debugString() const;

 private:
  // Build the GpeProfilerReport and dispatch it to reporter_.
  // No gate — the caller (mark()) decides whether to emit. Used
  // both for the current row and for backfilled predecessors.
  // Returns true iff dispatched (false only when reporter_ is null).
  bool reportRaw(
      GpeTracePoint p,
      uint64_t iterUs,
      uint64_t durationUs,
      std::string_view message);

  static constexpr size_t kNumPoints =
      static_cast<size_t>(GpeTracePoint::NUM_TRACE_POINTS);

  const CommLogData* const logMetaData_;
  const int rank_;
  const uint64_t commHash_;
  const int samplingWeight_;
  const std::unique_ptr<IGpeProfilerReporter> reporter_;
  // Source of the live abort flag for the per-mark "is the comm
  // aborted right now?" check that bypasses the sampling gate.
  // nullptr disables the bypass (only sampled / marker rows emit).
  const std::shared_ptr<::comms::fault_tolerance::Abort> abort_;

  Watch watch_; // Reset on mark(ITER_START) to set iter t0.

  // True between mark(ITER_START) and the next mark(ITER_START).
  // Non-anchor marks before ITER_START are silently dropped.
  bool iterAnchored_{false};
  // Sampling verdict for the current iter, set by injectMetadata.
  bool shouldTrace_{false};
  // Iter-relative µs of the last *reported* mark. durationUs of the
  // next reported mark = currentIterUs - lastStampUs_. Not updated
  // for marks that bypass the reporter (sampling-gated tracepoints).
  uint64_t lastStampUs_{0};

  // Per-iter timeline, indexed by GpeTracePoint. Populated in
  // mark() unconditionally (regardless of sampling or reporter
  // state); consumed by debugString() for the SCOPE_EXIT abort log
  // line. Reset on mark(ITER_START).
  std::array<std::optional<uint64_t>, kNumPoints> stamped_{};

  // Per-iter "did we already emit this tracepoint to Scuba?" flag.
  // Used by the universal in-order backfill in mark() to avoid
  // double-emitting any tracepoint. Reset on mark(ITER_START).
  std::array<bool, kNumPoints> reported_{};

  // Iter metadata captured at injectMetadata time, stamped on every
  // emitted row of the current iter.
  uint64_t opCount_{0};
  int opType_{-1};
};

} // namespace ctran
