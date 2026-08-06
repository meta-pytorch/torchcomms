// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <type_traits>

/*
 * Folly-free replacements for the two rate limiters the CLOGF macros rely on:
 * `folly::detail::xlogFirstNExactImpl` and
 * `folly::logging::IntervalRateLimiter`.
 *
 * These are the last folly primitives reachable from the CLOGF rate-limiting
 * macros, and the folly one lives in a private `detail` namespace. Both are
 * header-only so the macros can keep instantiating per-call-site state via
 * function-local statics, exactly as before.
 */

namespace meta::comms::logger {

/*
 * Returns true for exactly the first `n` invocations and false thereafter.
 *
 * State is per `Tag` type, so each call site must supply a distinct (typically
 * lambda-local) tag to get its own counter. Thread-safe and exact: concurrent
 * callers never collectively observe more than `n` true results.
 */
template <typename Tag>
bool firstNExact(uint64_t n) {
  static std::atomic<uint64_t> count{0};
  uint64_t cur = count.load(std::memory_order_relaxed);
  while (cur < n) {
    if (count.compare_exchange_weak(
            cur,
            cur + 1,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      return true;
    }
    // cur was refreshed by the failed exchange; loop re-tests it against n.
  }
  return false;
}

/*
 * Allows at most `maxPerInterval` successful checks per `interval`, matching
 * folly::logging::IntervalRateLimiter's contract.
 *
 * Intervals are lazily advanced on the first check() that observes the previous
 * one as expired, rather than on a timer, so an idle limiter costs nothing.
 */
class IntervalRateLimiter {
 public:
  using clock = std::chrono::steady_clock;

  IntervalRateLimiter(uint64_t maxPerInterval, clock::duration interval)
      : maxPerInterval_(maxPerInterval), interval_(interval) {}

  bool check() {
    const auto now = clock::now().time_since_epoch().count();
    const auto intervalEnd = timestamp_.load(std::memory_order_acquire);
    if (now < intervalEnd &&
        count_.load(std::memory_order_relaxed) >= maxPerInterval_) {
      return false;
    }

    const auto origCount = count_.fetch_add(1, std::memory_order_acq_rel);
    if (origCount < maxPerInterval_) {
      return true;
    }
    return checkSlow(now);
  }

 private:
  static_assert(
      std::is_signed_v<clock::rep>,
      "Need signed time point to represent initial time");
  static constexpr auto kInitialTimestamp =
      std::numeric_limits<clock::rep>::min();

  bool checkSlow(clock::rep now) {
    auto intervalEnd = timestamp_.load(std::memory_order_acquire);
    if (now < intervalEnd) {
      return false;
    }

    const auto newEnd = now + interval_.count();
    if (!timestamp_.compare_exchange_strong(
            intervalEnd,
            newEnd,
            std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      return false;
    }

    if (intervalEnd == kInitialTimestamp) {
      // The increment in check() wrapped count_ to zero. Re-increment so we
      // do not overwrite increments from callers that arrived concurrently.
      const auto origCount = count_.fetch_add(1, std::memory_order_acq_rel);
      return origCount < maxPerInterval_;
    }

    count_.store(1, std::memory_order_release);
    return maxPerInterval_ > 0;
  }

  const uint64_t maxPerInterval_;
  const clock::duration interval_;
  std::atomic<uint64_t> count_{std::numeric_limits<uint64_t>::max()};
  std::atomic<clock::rep> timestamp_{kInitialTimestamp};
};

} // namespace meta::comms::logger
