// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef NCCLX_META_TUNER_INT64_RANGE_H_
#define NCCLX_META_TUNER_INT64_RANGE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ncclx::tuner {

// A self-contained integer interval matcher used by the meta tuner to match the
// byte size and topology (nNodes / nLocalRanks) of a collective against a rule.
// It deliberately has NO NCCL/folly dependency so it can be unit-tested with a
// plain cpp_unittest while also being compiled into libnccl via the
// meta/tuner/*.cc glob.
//
// An Int64Range is the half-open/closed interval [lo, hi] where each bound may
// be inclusive, exclusive, or unbounded. A fully unbounded range (both bounds
// unbounded) matches any value (the "*" wildcard). int64_t is used so the
// range covers GB-scale byte values.
struct Int64Range {
  int64_t lo{0};
  int64_t hi{0};
  bool loInclusive{true};
  bool hiInclusive{true};
  bool loBounded{false};
  bool hiBounded{false};

  // Returns true when n lies within the interval, honoring bound inclusivity
  // and unboundedness on each side.
  bool matches(int64_t n) const;

  // Renders the interval back to its grammar form (e.g. "[0,1048576]", "(1,)",
  // "*") for logging.
  std::string toString() const;
};

// Parses a single interval expression into an Int64Range, returning
// std::nullopt on any parse error. Grammar (leading/trailing whitespace
// tolerated):
//   *             -> wildcard, fully unbounded (matches any)
//   N             -> exact [N, N]
//   [a,b]         -> a <= n <= b
//   (a,b)         -> a <  n <  b
//   (a,b] / [a,b) -> half-open
//   (a,)  / [a,)  -> lower-bounded only (n > a / n >= a)
//   (,b)  / (,b]  -> upper-bounded only (n < b / n <= b)
// Parse errors (-> nullopt): empty string, unbalanced/missing brackets,
// non-numeric bound, missing comma in an interval, and lo > hi (e.g. "(2,1]").
std::optional<Int64Range> parseInt64Range(std::string_view value);

} // namespace ncclx::tuner

#endif // NCCLX_META_TUNER_INT64_RANGE_H_
