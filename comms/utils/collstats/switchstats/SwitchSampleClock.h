// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <chrono>
#include <cstdint>

/* The monotonic clock, for the numbers that are durations rather than instants.
 *
 * This feature produces both and they cannot share a clock. A timestamp has to
 * name the instant the rest of the host's logs would name, which only the wall
 * clock does, so `unixNs` stays on the wall clock. A duration must not come
 * from that clock: NTP steps it, and a step landing between the two reads of an
 * interval reports either a stall or -- once the unsigned subtraction wraps --
 * six hundred years. Neither is distinguishable from something the fabric did.
 *
 * This is deliberately the same shape as the NIC sampler's NicSampleClock.h
 * rather than a dependency on it, because the two features must be able to land
 * and be reverted separately. Once both are in, the pair is worth folding into
 * one header under comms/utils.
 */

namespace meta::comms::switchstats {

// Against an unspecified epoch, so this is only ever meaningful as a
// difference.
inline uint64_t nowSteadyNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// The wall clock, for the one field that has to be comparable with other logs.
inline uint64_t nowUnixNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

/* Saturating, because the operands are unsigned. A pair in the wrong order --
 * including a stamp left at its default -- would wrap to near 2^64 and be
 * published as a field that looks like a measurement; zero reads as one that
 * was never taken. */
constexpr uint64_t switchElapsedNs(uint64_t fromSteadyNs, uint64_t toSteadyNs) {
  return toSteadyNs > fromSteadyNs ? toSteadyNs - fromSteadyNs : 0;
}

} // namespace meta::comms::switchstats
