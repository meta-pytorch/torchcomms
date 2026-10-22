/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace rcclx::relay::hipipc {
inline constexpr uint32_t kAbiVersion = 1;
inline constexpr size_t kMaxCapacityBytes = 144ULL << 20;
inline constexpr size_t kRingBytes = 4ULL << 20;
using Owner = std::shared_ptr<void>;
using IpcHandle = std::array<unsigned char, 64>;
using BusId = std::array<char, 32>;

struct RegistrationId {
  uint64_t identity{};
  uint64_t generation{};
  bool operator==(const RegistrationId& other) const {
    return identity == other.identity && generation == other.generation;
  }
  bool operator<(const RegistrationId& other) const {
    return std::make_pair(identity, generation) <
        std::make_pair(other.identity, other.generation);
  }
};

struct WorkspaceDescriptor {
  uint32_t version{kAbiVersion};
  uint32_t helperCount{};
  uint64_t generation{};
  uint64_t capacity{};
  BusId activeBus{};
  std::array<BusId, 7> helperBus{};
  IpcHandle input{};
  IpcHandle flags{};
  std::array<IpcHandle, 7> rings{};
};

struct RegistrationDescriptor {
  uint32_t version{kAbiVersion};
  uint32_t reserved{};
  uint64_t workspaceGeneration{};
  uint64_t peerWorkspaceGeneration{};
  RegistrationId registration{};
  uint64_t allocationGeneration{};
  uint64_t allocationBytes{};
  uint64_t offset{};
  uint64_t bytes{};
  IpcHandle base{};
};

inline void
validateRange(uint64_t allocation, uint64_t offset, uint64_t bytes) {
  if (!bytes || offset % 16 || bytes % 16 || offset > allocation ||
      bytes > allocation - offset) {
    throw std::invalid_argument("invalid aligned allocation range");
  }
}
inline bool rangesOverlap(
    uint64_t first,
    uint64_t firstBytes,
    uint64_t second,
    uint64_t secondBytes) {
  return first <= second ? second - first < firstBytes
                         : first - second < secondBytes;
}
template <class Record>
std::vector<uint8_t> encodeRecord(const Record& record) {
  static_assert(std::is_trivially_copyable_v<Record>);
  std::vector<uint8_t> result(sizeof(Record));
  std::memcpy(result.data(), &record, sizeof(Record));
  return result;
}
template <class Record>
Record decodeRecord(const std::vector<uint8_t>& bytes) {
  static_assert(std::is_trivially_copyable_v<Record>);
  if (bytes.size() != sizeof(Record)) {
    throw std::invalid_argument("binary record has wrong length");
  }
  Record result{};
  std::memcpy(&result, bytes.data(), sizeof(result));
  if (result.version != kAbiVersion) {
    throw std::invalid_argument("binary record has wrong ABI version");
  }
  return result;
}

namespace detail {
struct BaseRecord {
  size_t bytes{};
  uint64_t generation{};
  IpcHandle handle{};
  size_t refs{};
  std::vector<Owner> owners;
};
struct InputRecord {
  RegistrationDescriptor local{};
  RegistrationDescriptor peer{};
  const void* input{};
  void* base{};
  bool bound{};
};

// Used by the native state and CPU tests; HIP mappings and locks live in state.
struct RegistrationRecords {
  std::map<void*, BaseRecord> bases;
  std::map<uint64_t, InputRecord> inputs;

  InputRecord& get(RegistrationId id) {
    auto found = inputs.find(id.identity);
    if (found == inputs.end() || !(found->second.local.registration == id)) {
      throw std::invalid_argument("stale registration identity/generation");
    }
    return found->second;
  }
  void add(const InputRecord& input, const Owner& owner) {
    auto found = bases.find(input.base);
    if (found != bases.end() &&
        (found->second.bytes != input.local.allocationBytes ||
         found->second.generation != input.local.allocationGeneration ||
         found->second.handle != input.local.base)) {
      throw std::invalid_argument("live allocation identity changed");
    }
    if (inputs.count(input.local.registration.identity)) {
      throw std::invalid_argument("duplicate registration identity");
    }
    auto& base = bases[input.base];
    base.bytes = input.local.allocationBytes;
    base.generation = input.local.allocationGeneration;
    base.handle = input.local.base;
    // The call's owner remains strong outside the native lock on any failure.
    base.owners.push_back(owner);
    inputs.emplace(input.local.registration.identity, input);
    ++base.refs;
  }
  std::vector<Owner> erase(RegistrationId id) {
    const auto baseAddress = get(id).base;
    auto& base = bases.at(baseAddress);
    std::vector<Owner> owners;
    inputs.erase(id.identity);
    if (--base.refs == 0) {
      owners = std::move(base.owners);
      bases.erase(baseAddress);
    }
    return owners;
  }
};
} // namespace detail
} // namespace rcclx::relay::hipipc
