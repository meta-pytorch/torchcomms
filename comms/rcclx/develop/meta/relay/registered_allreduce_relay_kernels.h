/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace rcclx::relay {

constexpr int kRegisteredRelayThreads = 512;
constexpr int kRegisteredRelayMaxHelpers = 7;
constexpr int kRegisteredRelayMaxLanes = 16;
constexpr int kRegisteredRelayMaxSlots = 8;

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

// Enqueues exactly one kernel of directLanes + 2 * helpers * lanes CTAs.
hipError_t launchRegisteredRelayAllReduce(
    const RegisteredRelayArgs& args,
    hipStream_t stream);

} // namespace rcclx::relay
