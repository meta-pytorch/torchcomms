// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <chrono>
#include <cstdint>
#include <utility>

namespace meta::comms::colltrace {

// Wall-clock source for colltrace timestamps. Backed by fbclock when the
// PTP daemon is reachable (giving PTP-aligned, fleet-comparable time with
// a known window of uncertainty); falls back to std::chrono::system_clock
// when fbclock_init fails or NCCL_USE_PTP=0.
//
// All entry points are thread-safe and noexcept. Initialization happens
// lazily on first use and is performed exactly once.
//
// API mirrors the historical ncclFbGet*Time* surface (D58118422) so
// ncclx engineers see a familiar shape.

// Returns the midpoint of the fbclock truetime window as a system_clock
// time_point, or std::chrono::system_clock::now() when in fallback.
std::chrono::system_clock::time_point precisionNow() noexcept;

// Returns {earliest_ns, latest_ns} since the UNIX epoch. The two values
// bracket true time when PTP is active; they are equal in fallback mode.
std::pair<uint64_t, uint64_t> precisionNowRangeNs() noexcept;

// Single-point nanoseconds since the UNIX epoch. Midpoint of the fbclock
// window when PTP is active; system_clock::now() converted to ns otherwise.
uint64_t precisionNowNs() noexcept;

// Window-of-uncertainty in nanoseconds (latest_ns - earliest_ns).
// Returns 0 in fallback mode.
uint64_t precisionErrorNs() noexcept;

// True when fbclock_init succeeded and NCCL_USE_PTP is not disabled.
// Determined once at first call and cached for the process lifetime.
bool precisionUsingPtp() noexcept;

} // namespace meta::comms::colltrace
