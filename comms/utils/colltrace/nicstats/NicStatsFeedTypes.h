// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <cstdint>
#include <optional>

/* NIC counter telemetry is not part of the open-source build. These are the
 * types the open-source code names, with no counters and no sources: the
 * stand-ins in this directory never produce a reading. */

namespace meta::comms::colltrace::nicstats {

enum class NicCounterId : uint8_t {};

inline constexpr uint32_t kNumCounters = 0;

enum class NicCounterStatus : uint8_t {
  Ok,
  Unsupported,
  ReadFailed,
  NotRequested
};

inline constexpr uint64_t kNicCounterNoValue = ~uint64_t{0};

enum class NicStatsSource : uint8_t {};

inline constexpr uint32_t kNumNicStatsSources = 0;

struct NicStatsEndpoint {
  uint64_t unixNs{0};
  uint64_t lagNs{0};
  uint64_t samplePeriodNs{0};
  std::array<uint64_t, kNumCounters> values{};
  std::array<NicCounterStatus, kNumCounters> status{};
};

inline constexpr uint32_t kNicRailNameMax = 32;

struct NicRailId {
  std::array<char, kNicRailNameMax> device{};
  uint8_t port{0};
};

struct NicStatsReading {
  uint64_t sequence{0};
  NicStatsEndpoint counters;
  NicStatsSource source{};
  std::array<NicStatsSource, kNumCounters> counterSource{};
  NicRailId rail;
  std::array<uint64_t, kNumNicStatsSources> sourceReadNs{};
  std::optional<uint8_t> priority;
};

} // namespace meta::comms::colltrace::nicstats
