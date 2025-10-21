// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>

namespace ctran::utils {

template <typename Clock, typename Duration = typename Clock::duration>
class StopWatch {
 public:
  using clock_type = Clock;
  using duration = Duration;
  using time_point = std::chrono::time_point<clock_type, duration>;

  static_assert(
      std::ratio_less_equal<
          typename clock_type::duration::period,
          typename duration::period>::value,
      "clock must be at least as precise as the requested duration");

  StopWatch() : checkpoint_(Clock::now()) {}
  StopWatch(const StopWatch&) = delete;
  StopWatch& operator=(const StopWatch&) = delete;
  StopWatch(StopWatch&&) = default;
  StopWatch& operator=(StopWatch&&) = default;

  /**
   * Returns the current checkpoint
   */
  typename clock_type::time_point getCheckpoint() const noexcept {
    return checkpoint_;
  }

  /**
   * Tells the elapsed time since the last update, and updates the checkpoint
   * to the current time.
   */
  duration lap() {
    auto lastCheckpoint = checkpoint_;

    checkpoint_ = Clock::now();

    return std::chrono::duration_cast<duration>(checkpoint_ - lastCheckpoint);
  }

  /**
   * Tells the elapsed time since the last update.
   * The stop watch's checkpoint remains unchanged.
   */
  [[nodiscard]] duration elapsed() const {
    return std::chrono::duration_cast<duration>(Clock::now() - checkpoint_);
  }

  /**
   * Updates the stop watch checkpoint to the current time.
   */
  void reset() {
    checkpoint_ = Clock::now();
  }

 private:
  typename clock_type::time_point checkpoint_;
};

} // namespace ctran::utils
