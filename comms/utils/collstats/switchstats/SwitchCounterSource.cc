// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/switchstats/SwitchCounterSource.h"

namespace meta::comms::switchstats {

SwitchReading readingFromCounters(
    const SwitchCounterSpec& spec,
    const std::map<std::string, int64_t>& counters,
    SwitchReadMechanism mechanism) {
  SwitchReading reading;
  reading.ok = true;
  reading.mechanism = mechanism;

  for (const auto& [key, value] : counters) {
    const auto parsed = parseCounterName(spec, key);
    if (!parsed.has_value()) {
      continue;
    }
    SwitchPortReading& port = reading.ports[parsed->port];
    port.values[measurementIndex(parsed->measurement)] = value;
    port.status[measurementIndex(parsed->measurement)] =
        SwitchCounterStatus::Ok;
  }

  for (auto& [portName, port] : reading.ports) {
    for (uint32_t i = 0; i < kNumMeasurements; ++i) {
      if (port.status[i] == SwitchCounterStatus::NotRequested) {
        port.status[i] = SwitchCounterStatus::Unsupported;
      }
    }
  }
  return reading;
}

} // namespace meta::comms::switchstats
