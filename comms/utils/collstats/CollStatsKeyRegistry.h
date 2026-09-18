// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "comms/utils/collstats/CollStatsTypes.h"

// Host-side key registry: maps a CollStatKey to the dense value slot the device
// accumulates into.
//
// Every field of the key — op, algorithm, protocol, datatype and the size class
// derived from the message size — is known on the enqueue thread before the
// collective launches, so the mapping is resolved here and the resolved id is
// handed to the kernel. The device then indexes rather than searches: no CAS,
// no probing, no saturation policy in device code.
//
// Ids are dense and allocated in first-touch order, which is what lets the
// reader copy only the occupied prefix of the value bank instead of its whole
// capacity. Ids are never recycled, so a key keeps its id for the life of the
// communicator and the id space only grows.
//
// Keys beyond `capacity` all resolve to `catchAllId()`, the trailing value
// slot, and are counted so saturation is visible rather than silent.

namespace meta::comms::collstats {

inline bool operator==(const CollStatKey& a, const CollStatKey& b) {
  return a.op == b.op && a.algorithm == b.algorithm &&
      a.protocol == b.protocol && a.dtype == b.dtype &&
      a.sizeClass == b.sizeClass;
}

class CollStatsKeyRegistry {
 public:
  explicit CollStatsKeyRegistry(uint32_t capacity) : capacity_(capacity) {
    byId_.reserve(capacity);
  }

  // The value slot shared by every key that did not fit. The value bank holds
  // capacity + 1 slots so this one always exists.
  uint32_t catchAllId() const {
    return capacity_;
  }

  // Resolve `key`, assigning the next dense id on first sight. Returns
  // catchAllId() once `capacity` distinct keys have been assigned. Called from
  // the enqueue thread on every instrumented collective.
  uint32_t resolve(const CollStatKey& key);

  // Number of ids assigned so far, i.e. the occupied prefix length of the value
  // bank. Monotonic.
  uint32_t size() const;

  // Keys indexed by id, for attributing a readout window. Index i holds the key
  // that owns value slot i.
  std::vector<CollStatKey> keys() const;

  // Collectives that resolved to the catch-all because the registry was full.
  uint64_t catchAllCount() const;

 private:
  struct Hash {
    std::size_t operator()(const CollStatKey& k) const noexcept {
      // FNV-1a over the five fields. Host-only: the device indexes a slot the
      // host already resolved, so there is no device-side hash for this to
      // agree with.
      uint64_t h = 1469598103934665603ull;
      for (uint32_t f :
           {static_cast<uint32_t>(k.op),
            static_cast<uint32_t>(k.algorithm),
            static_cast<uint32_t>(k.protocol),
            static_cast<uint32_t>(k.dtype),
            static_cast<uint32_t>(k.sizeClass)}) {
        for (int b = 0; b < 4; ++b) {
          h ^= static_cast<uint64_t>((f >> (b * 8)) & 0xFF);
          h *= 1099511628211ull;
        }
      }
      return static_cast<std::size_t>(h);
    }
  };

  mutable std::mutex mu_;
  uint32_t capacity_;
  std::unordered_map<CollStatKey, uint32_t, Hash> ids_;
  std::vector<CollStatKey> byId_;
  uint64_t catchAllCount_{0};
};

} // namespace meta::comms::collstats
