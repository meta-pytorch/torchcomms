// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

// The switch-side counter vocabulary: which of the switch agent's counters this
// feature reads, the regex that asks for exactly those, and the parser that
// turns a returned key back into (port, measurement).
//
// The regex and the parser are both generated from the one table below. They
// are each other's inverse, and a switch reply is matched against the same
// strings that requested it, so a measurement cannot be fetched under one name
// and read back under another.
//
// Only the bare `.sum` counters appear here. Those are lifetime cumulative, so
// an interval is two reads differenced and its resolution is bounded by the
// gap between them. The agent also exports `.rate` and windowed `.60`/`.600`/
// `.3600` forms of each; taking any of those would pin resolution to the
// window instead, which is the limit this feature exists to get out from
// under.

namespace meta::comms::switchstats {

// What a counter is scoped to, which is what decides where its name sits
// relative to the port.
enum class SwitchCounterScope : uint8_t {
  // Whole port: `<port>.<leaf>.sum`
  Port,
  // The lossless queue: `<port>.<queue>.<leaf>.sum`
  LosslessQueue,
  // The lossless priority: `<port>.<leaf>.priority<N>.sum`
  LosslessPriority,
};

enum class SwitchMeasurement : uint8_t {
  OutBytes,
  OutPkts,
  OutCongestionDiscards,
  OutCongestionDiscardBytes,
  OutEcnCounter,
  WredDroppedPackets,
  InPfcFrames,
  OutPfcFrames,
  InDiscards,
  InErrors,
};

constexpr uint32_t kNumMeasurements = 10;

// Indexes every per-measurement array in this feature, including
// `kMeasurements`, whose rows the static_assert below holds to enum order.
constexpr uint32_t measurementIndex(SwitchMeasurement m) {
  return static_cast<uint32_t>(m);
}

struct SwitchMeasurementEntry {
  SwitchMeasurement measurement;
  // The counter's own name, without port, queue or priority.
  std::string_view leaf;
  SwitchCounterScope scope;
};

// The starting set: about ten per port, chosen so a slow step can be attributed
// to the fabric. Six describe what the training queue sent and what the switch
// dropped or marked on it; the PFC pair says whether the lossless class was
// paused; the last two are the port's own error and discard counts.
constexpr std::array<SwitchMeasurementEntry, kNumMeasurements> kMeasurements{{
    {SwitchMeasurement::OutBytes,
     "out_bytes",
     SwitchCounterScope::LosslessQueue},
    {SwitchMeasurement::OutPkts, "out_pkts", SwitchCounterScope::LosslessQueue},
    {SwitchMeasurement::OutCongestionDiscards,
     "out_congestion_discards",
     SwitchCounterScope::LosslessQueue},
    {SwitchMeasurement::OutCongestionDiscardBytes,
     "out_congestion_discards_bytes",
     SwitchCounterScope::LosslessQueue},
    {SwitchMeasurement::OutEcnCounter,
     "out_ecn_counter",
     SwitchCounterScope::LosslessQueue},
    {SwitchMeasurement::WredDroppedPackets,
     "wred_dropped_packets",
     SwitchCounterScope::LosslessQueue},
    {SwitchMeasurement::InPfcFrames, "in_pfc_frames", SwitchCounterScope::Port},
    {SwitchMeasurement::OutPfcFrames,
     "out_pfc_frames",
     SwitchCounterScope::LosslessPriority},
    {SwitchMeasurement::InDiscards, "in_discards", SwitchCounterScope::Port},
    {SwitchMeasurement::InErrors, "in_errors", SwitchCounterScope::Port},
}};

// Rows are looked up by `measurementIndex`, so each must sit at its own enum
// value and none may be left default-filled. Otherwise a swapped row would
// still compile and tag counters with the wrong measurement.
constexpr bool measurementsAreDense() {
  for (uint32_t i = 0; i < kNumMeasurements; ++i) {
    if (measurementIndex(kMeasurements[i].measurement) != i ||
        kMeasurements[i].leaf.empty()) {
      return false;
    }
  }
  return true;
}
static_assert(measurementsAreDense());

// `in_pfc_frames` is deliberately whole-port. The agent does export
// `in_pfc_frames.priority0` through `.priority7`, but on the platform this was
// checked against all eight are byte-identical to the aggregate, so the
// receive-side per-priority split carries no information. Reading the aggregate
// says the same thing without implying a breakdown that is not there.

// Where the lossless class lives, and what a port is called.
//
// These are the values observed on the switches this was built against, not
// anything the agent guarantees. They are fields rather than constants because
// a platform that numbers or names its queues differently must be given the
// right ones instead of quietly reading a queue that carries no training
// traffic. A wrong value here yields a counter the switch never exports, which
// surfaces as Unsupported -- not as zero, which is what an idle link reports.
struct SwitchCounterSpec {
  // Matches a port name. Ports look like `eth1/5/1`.
  std::string portPattern{"eth[0-9/]+"};
  // The queue carrying RDMA. `queue2` is named `rdma`, and the agent spells
  // both in the counter name.
  std::string losslessQueueSegment{"queue2.rdma"};
  // The lossless priority. Priority 2 is the only one with non-zero
  // `out_pfc_frames`.
  uint8_t losslessPriority{2};
};

// The name of one counter under `spec`, minus the port: everything the agent
// puts between the port and the trailing `.sum`.
std::string counterSuffix(
    const SwitchCounterSpec& spec,
    SwitchMeasurement measurement);

// A PCRE selecting exactly the counters in `kMeasurements`, for
// `getRegexCounters`. Anchored at both ends: a switch exports around forty
// thousand counters and an unanchored pattern would both over-match and make
// the agent walk more of them than it has to.
std::string countersRegex(const SwitchCounterSpec& spec);

struct ParsedSwitchCounter {
  std::string port;
  SwitchMeasurement measurement;
};

// Splits a returned counter key back into port and measurement, or nothing if
// the key is not one this spec asked for. Nothing is the right answer for an
// unexpected key: the alternative is charging some port's measurement with a
// number that was never about it.
std::optional<ParsedSwitchCounter> parseCounterName(
    const SwitchCounterSpec& spec,
    std::string_view counterName);

} // namespace meta::comms::switchstats
