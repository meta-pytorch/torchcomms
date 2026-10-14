/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace rcclx::relay {

constexpr int kRegisteredRelayThreads = 512;
constexpr int kRegisteredRelayMaxHelpers = 7;
constexpr int kRegisteredRelayMaxLanes = 16;
constexpr int kRegisteredRelayMaxSlots = 8;
constexpr size_t kRegisteredRelayVectorBytes = 16;

struct alignas(256) RegisteredRelayWord {
  uint32_t value;
};

// One rank's relay flags. start/done/full/freed are written by the peer;
// calls, ticket and the per-lane sequence counters are this rank's own. All
// per-call values derive from these device-resident words, so a launch can be
// captured into a graph and replayed.
struct RegisteredRelayFlags {
  RegisteredRelayWord start;
  RegisteredRelayWord done;
  RegisteredRelayWord calls;
  RegisteredRelayWord ticket;
  uint32_t full[kRegisteredRelayMaxHelpers][kRegisteredRelayMaxLanes]
               [kRegisteredRelayMaxSlots];
  uint32_t freed[kRegisteredRelayMaxHelpers][kRegisteredRelayMaxLanes]
                [kRegisteredRelayMaxSlots];
  uint32_t pushSeq[kRegisteredRelayMaxHelpers][kRegisteredRelayMaxLanes];
  uint32_t reduceSeq[kRegisteredRelayMaxHelpers][kRegisteredRelayMaxLanes];
};

// A two-rank BF16 SUM of `vectors` 16-byte vectors. Slice s goes to the direct
// path for the first directWeight positions of each (directWeight + helpers)
// cycle and to helper paths for the rest. The direct path reads the peer's
// registered input; helper path h stages this rank's slices in its own ring on
// helper GPU h (push) and reads the peer's ring there (pull).
struct RegisteredRelayArgs {
  const void* mine;
  const void* peer;
  void* output;
  size_t vectors;
  size_t sliceVectors;
  size_t slotVectors;
  int directWeight;
  int helpers;
  int directLanes;
  int lanes;
  int slots;
  int rank;
  void* push[kRegisteredRelayMaxHelpers];
  const void* pull[kRegisteredRelayMaxHelpers];
  RegisteredRelayFlags* myFlags;
  RegisteredRelayFlags* peerFlags;
};

struct RegisteredRelayLaunchPolicy {
  size_t minSliceBytes;
  size_t maxSliceBytes;
  int directWeight;
  int directLanes;
  int lanes;
  int slots;
};

inline size_t registeredRelaySliceBytes(
    size_t bytes,
    int helpers,
    const RegisteredRelayLaunchPolicy& policy) {
  assert(helpers >= 0 && helpers <= kRegisteredRelayMaxHelpers);
  assert(policy.lanes > 0 && policy.lanes <= kRegisteredRelayMaxLanes);
  assert(
      policy.minSliceBytes > 0 && policy.minSliceBytes <= policy.maxSliceBytes);
  const size_t target =
      std::max<size_t>(bytes / ((helpers + 1) * policy.lanes), 1);
  size_t sliceBytes = 1;
  while (sliceBytes <= target / 2) {
    sliceBytes *= 2;
  }
  return std::clamp(sliceBytes, policy.minSliceBytes, policy.maxSliceBytes);
}

// Keep kernel-contract field initialization shared by all host launch paths.
// New argument fields need an explicit policy here and matching regression
// tests; value-initialization alone does not guarantee a valid new field value.
inline RegisteredRelayArgs makeRegisteredRelayArgs(
    const void* mine,
    const void* peer,
    void* output,
    size_t bytes,
    int rank,
    int helpers,
    const RegisteredRelayLaunchPolicy& policy,
    void* const* push,
    void* const* pull,
    RegisteredRelayFlags* myFlags,
    RegisteredRelayFlags* peerFlags) {
  assert(policy.directWeight > 0);
  assert(policy.directLanes > 0);
  assert(policy.slots > 0 && policy.slots <= kRegisteredRelayMaxSlots);
  assert(helpers == 0 || (push != nullptr && pull != nullptr));
  RegisteredRelayArgs args{};
  args.mine = mine;
  args.peer = peer;
  args.output = output;
  args.vectors = bytes / kRegisteredRelayVectorBytes;
  args.sliceVectors = registeredRelaySliceBytes(bytes, helpers, policy) /
      kRegisteredRelayVectorBytes;
  args.slotVectors = policy.maxSliceBytes / kRegisteredRelayVectorBytes;
  args.directWeight = policy.directWeight;
  args.helpers = helpers;
  args.directLanes = policy.directLanes;
  args.lanes = policy.lanes;
  args.slots = policy.slots;
  args.rank = rank;
  for (int h = 0; h < helpers; ++h) {
    args.push[h] = push[h];
    args.pull[h] = pull[h];
  }
  args.myFlags = myFlags;
  args.peerFlags = peerFlags;
  return args;
}

} // namespace rcclx::relay
