// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <string>

#include "comms/utils/collstats/switchstats/SwitchCounterSpec.h"

// One reading of one switch's counters, and the interface that produces it.
//
// The interface exists so the differencing above it is testable without a
// switch: everything about how an interval is measured is decided here in terms
// of values, and the Thrift client is one implementation of a two-method
// contract.

namespace meta::comms::switchstats {

// Why a counter has the value it has. Four outcomes rather than a value and a
// bool, because three of them are things a consumer must not average.
enum class SwitchCounterStatus : uint8_t {
  // The spec did not ask for it. The default, so a half-filled array reads as
  // unasked rather than as zero.
  NotRequested,
  // Asked for, and the switch does not export it. Distinct from Ok-with-zero,
  // which is what an idle link legitimately reports: a queue that moved no
  // bytes and a queue that does not exist are not the same observation, and a
  // platform whose queues are numbered differently produces the second.
  Unsupported,
  // Asked for, exported, and the read did not yield a usable number.
  ReadFailed,
  Ok,
};

// Paired with any status other than Ok. Not zero, which is a legitimate
// reading, and not a small number either: it is chosen to be obviously wrong if
// it ever reaches a consumer that ignored the status.
constexpr int64_t kSwitchCounterNoValue = std::numeric_limits<int64_t>::min();

// Which fb303 method actually served a reading. Every reading carries the one
// that ran, set by the code that ran it, because a reading naming the method
// that was requested would describe a call that did not happen -- a regex form
// and a whole-counter-map fallback do not cost the switch the same thing. Only
// the one mechanism there is a producer for is listed; the rest belong with the
// fallback that issues them.
enum class SwitchReadMechanism : uint8_t {
  Fb303GetRegexCounters,
};

// One port's measurements at one instant. Indexed by measurementIndex().
struct SwitchPortReading {
  std::array<int64_t, kNumMeasurements> values{};
  std::array<SwitchCounterStatus, kNumMeasurements> status{};

  SwitchPortReading() {
    values.fill(kSwitchCounterNoValue);
    status.fill(SwitchCounterStatus::NotRequested);
  }
};

// Every port the switch answered for, at one instant.
struct SwitchReading {
  // False when the call itself did not complete -- unreachable agent, timeout,
  // a Thrift error. `ports` is then empty, and an interval built from it
  // produces no measurement rather than one measured against nothing.
  bool ok{false};
  SwitchReadMechanism mechanism{SwitchReadMechanism::Fb303GetRegexCounters};
  // Wall clock, for joining against other logs.
  uint64_t unixNs{0};
  // Monotonic, taken immediately before the call. Difference these to measure
  // an interval, never `unixNs`. Carrying it here rather than letting the layer
  // above stamp around `read()` keeps the call's own latency out of the
  // interval: that latency is neither constant nor small next to a step, so
  // stamping outside would charge each interval a share of it.
  uint64_t steadyNs{0};
  // Monotonic, and the cost of this one call to the switch. The agent serves
  // getRegexCounters on its event-base thread, so this is worth watching.
  uint64_t lagNs{0};
  std::map<std::string, SwitchPortReading> ports;
};

class ISwitchCounterSource {
 public:
  virtual ~ISwitchCounterSource() = default;

  // One round trip. Must not throw: a switch that cannot be reached is an
  // ordinary outcome on a fabric this size, and it is reported as
  // `ok == false`.
  virtual SwitchReading read() noexcept = 0;

  // Which switch these readings describe, for the record and for logs.
  virtual const std::string& name() const noexcept = 0;
};

// Fills a reading from a counter map as getRegexCounters returns it. Separate
// from the call that produced it so the mapping -- which is where the
// zero/unsupported distinction is made -- is testable without a switch.
//
// A key the spec did not ask for is ignored. A measurement absent from a port
// that did answer is Unsupported, not zero: the regex asked for all of them, so
// the switch not naming one means it does not export it. Zero is what an idle
// link reports, and the two must not collapse.
// `mechanism` is supplied by the caller that made the call, because only it
// knows what actually ran. Inferring it here would name the method this mapper
// happens to be written for, which is not the same claim.
SwitchReading readingFromCounters(
    const SwitchCounterSpec& spec,
    const std::map<std::string, int64_t>& counters,
    SwitchReadMechanism mechanism);

} // namespace meta::comms::switchstats
