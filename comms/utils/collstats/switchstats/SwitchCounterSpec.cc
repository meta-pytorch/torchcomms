// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/switchstats/SwitchCounterSpec.h"

namespace meta::comms::switchstats {

namespace {

// Every counter this feature reads is a cumulative lifetime total, which the
// agent exports under this suffix.
constexpr std::string_view kSumSuffix = ".sum";

// Leaf and queue names are `[a-z0-9_]` and `.`, so the dot is the only
// metacharacter that can appear in an interpolated name.
std::string escapeDots(std::string_view text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (const char c : text) {
    if (c == '.') {
      escaped += '\\';
    }
    escaped += c;
  }
  return escaped;
}

bool endsWith(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() &&
      text.substr(text.size() - suffix.size()) == suffix;
}

} // namespace

std::string counterSuffix(
    const SwitchCounterSpec& spec,
    SwitchMeasurement measurement) {
  const SwitchMeasurementEntry& entry =
      kMeasurements[measurementIndex(measurement)];
  switch (entry.scope) {
    case SwitchCounterScope::Port:
      return std::string(entry.leaf);
    case SwitchCounterScope::LosslessQueue:
      return spec.losslessQueueSegment + "." + std::string(entry.leaf);
    case SwitchCounterScope::LosslessPriority:
      return std::string(entry.leaf) + ".priority" +
          std::to_string(static_cast<uint32_t>(spec.losslessPriority));
  }
  return std::string(entry.leaf);
}

std::string countersRegex(const SwitchCounterSpec& spec) {
  std::string alternation;
  for (const SwitchMeasurementEntry& entry : kMeasurements) {
    if (!alternation.empty()) {
      alternation += '|';
    }
    alternation += escapeDots(counterSuffix(spec, entry.measurement));
  }
  // Grouped so an alternation inside the port pattern stays inside it, rather
  // than splitting the whole expression in two.
  return "^(?:" + spec.portPattern + ")\\.(" + alternation + ")\\" +
      std::string(kSumSuffix) + "$";
}

std::optional<ParsedSwitchCounter> parseCounterName(
    const SwitchCounterSpec& spec,
    std::string_view counterName) {
  if (!endsWith(counterName, kSumSuffix)) {
    return std::nullopt;
  }
  const std::string_view withoutSum =
      counterName.substr(0, counterName.size() - kSumSuffix.size());

  for (const SwitchMeasurementEntry& entry : kMeasurements) {
    // The dot is required, so a port cannot be matched away to nothing and a
    // longer leaf ending in a shorter one cannot be taken for the shorter.
    const std::string tail = "." + counterSuffix(spec, entry.measurement);
    if (!endsWith(withoutSum, tail)) {
      continue;
    }
    const std::string_view port =
        withoutSum.substr(0, withoutSum.size() - tail.size());
    if (port.empty()) {
      return std::nullopt;
    }
    return ParsedSwitchCounter{std::string(port), entry.measurement};
  }
  return std::nullopt;
}

} // namespace meta::comms::switchstats
