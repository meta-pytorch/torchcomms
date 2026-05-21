// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <string_view>

struct CommLogData;

namespace ctran {

// One of the GPE thread's instrumented tracepoints. Stamped by
// gpeThreadFn at the corresponding boundary.
//
// INVARIANT: enum declaration order must match the chronological
// order of mark() calls in CtranGpe::Impl::gpeThreadFn — enforced
// by DCHECK_GE in GpeProfiler::mark().
enum class GpeTracePoint : int {
  ITER_START, // Top of the GPE main loop iter (anchor; always emits).
  CMD_DEQUEUE, // cmdDequeue returned + injectMetadata ran (always emits).
  WAIT_KERNEL, // KERNEL_STARTED observed (kernel started executing).
  HOST_ALGO, // cmd->coll.func returned (or skip-aborted bookkeeping done).
  TERMINATE_KERNEL, // setFlagPerGroup() returned (HOST_ABORT or TERMINATE).
  ALGO_ABORTED, // markAborted() called from SCOPE_EXIT (abort-only; forced).
  // TERMINATE_CMD intentionally last (just before NUM_TRACE_POINTS): it
  // only co-occurs with ITER_START + CMD_DEQUEUE in TERMINATE iters, so
  // its placement at the end avoids creating a stamp gap in the middle
  // of the enum that would skip slots in debugString's adjacent-pair
  // walk.
  TERMINATE_CMD, // Dequeued cmd is TERMINATE; gpeThread is about to exit
                 // (always emits, bypasses sampling).
  NUM_TRACE_POINTS,
};

// Stable string names for the tracepoints, used as the Scuba
// `trace_point` column value.
constexpr std::string_view tracePointName(GpeTracePoint p) {
  switch (p) {
    case GpeTracePoint::ITER_START:
      return "iter_start";
    case GpeTracePoint::CMD_DEQUEUE:
      return "cmd_dequeue";
    case GpeTracePoint::WAIT_KERNEL:
      return "wait_kernel";
    case GpeTracePoint::HOST_ALGO:
      return "host_algo";
    case GpeTracePoint::TERMINATE_KERNEL:
      return "terminate_kernel";
    case GpeTracePoint::ALGO_ABORTED:
      return "algo_aborted";
    case GpeTracePoint::TERMINATE_CMD:
      return "terminate_cmd";
    case GpeTracePoint::NUM_TRACE_POINTS:
      return "<invalid>";
  }
  return "<invalid>";
}

// One Scuba row per stamped tracepoint, written eagerly at mark() time.
// iterUs is microseconds since the iteration's ITER_START (so ITER_START
// itself always has iterUs == 0). durationUs is microseconds since the
// previous stamped tracepoint in the same iter (0 for ITER_START).
// `aborted` and `message` are populated only on the ALGO_ABORTED row;
// `message` carries the abort reason ("timeout" / "explicit" /
// "abnormal_exit") and lands in the Scuba "message" column.
struct GpeProfilerReport {
  const CommLogData* logMetaData{nullptr};
  uint64_t opCount{0};
  int opType{-1};
  GpeTracePoint tracePoint{GpeTracePoint::ITER_START};
  uint64_t iterUs{0};
  uint64_t durationUs{0};
  bool aborted{false};
  std::string_view message{};
};

} // namespace ctran
