// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "comms/utils/colltrace/common/GraphCollTraceEvent.h"
#include "comms/utils/colltrace/v2/CollTraceV2.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBufferReader.h"

namespace meta::comms::colltracev2 {

using Ring = ::hrdw_ring_buffer::HRDWRingBuffer<colltrace::GraphCollTraceEvent>;
using Reader =
    ::hrdw_ring_buffer::HRDWRingBufferReader<colltrace::GraphCollTraceEvent>;

// collId layout: 16 bits of communicator index, 48 bits of per-comm counter.
constexpr int kCommIndexBits = 16;
constexpr int kLocalBits = 64 - kCommIndexBits;
constexpr uint64_t kLocalMask = (uint64_t{1} << kLocalBits) - 1;
constexpr uint32_t kMaxCommIndex = (1u << kCommIndexBits) - 1;

constexpr uint64_t makeCollId(uint32_t commIndex, uint64_t local) {
  return (static_cast<uint64_t>(commIndex) << kLocalBits) |
      (local & kLocalMask);
}

// One registered communicator, shared by the controller and the CollEvents.
// mu guards every field declared after it.
struct CommState {
  CommState(std::string desc, uint32_t idx, std::unique_ptr<Ring> r)
      : commDesc(std::move(desc)),
        index(idx),
        ring(std::move(r)),
        reader(*ring) {}

  ~CommState() {
    // Captured graphs may still write into the ring; never free it.
    (void)ring.release();
  }

  CommState(const CommState&) = delete;
  CommState& operator=(const CommState&) = delete;
  CommState(CommState&&) = delete;
  CommState& operator=(CommState&&) = delete;

  const std::string commDesc;
  const uint32_t index;
  std::unique_ptr<Ring> ring;

  std::mutex mu;

  // Controller: id allocation and per-collective kernel counts.
  uint64_t nextLocalCollId{0};
  uint64_t latestCollId{COLLTRACE_INVALID_COLLID};
  // Kernel count for the most recent COLLTRACE_COLL_HISTORY collectives,
  // indexed by local id minus firstCountedLocal.
  uint64_t firstCountedLocal{0};
  std::deque<uint32_t> kernelCounts;

  uint32_t* kernelCountFor(uint64_t local) {
    if (local < firstCountedLocal ||
        local - firstCountedLocal >= kernelCounts.size()) {
      return nullptr;
    }
    return &kernelCounts[local - firstCountedLocal];
  }

  // CollEvents: events pulled from the ring and not yet handed out.
  Reader reader;
  std::deque<Event> events;
  uint64_t dropped{0};
  uint64_t lost{0};
};

} // namespace meta::comms::colltracev2
