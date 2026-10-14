/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "comms/rcclx/develop/meta/relay/registered_allreduce_relay_args.h"

#include <gtest/gtest.h>

#include <array>

namespace rcclx::relay {
namespace {

constexpr size_t kKiB = 1024;
constexpr size_t kMiB = 1024 * kKiB;
constexpr RegisteredRelayLaunchPolicy
    kAdaptive{16 * kKiB, 256 * kKiB, 1, 8, 8, 2};
constexpr RegisteredRelayLaunchPolicy
    kFixed{256 * kKiB, 256 * kKiB, 1, 8, 8, 2};

struct Buffers {
  int mine{};
  int peer{};
  int output{};
  std::array<int, kRegisteredRelayMaxHelpers> local{};
  std::array<int, kRegisteredRelayMaxHelpers> remote{};
  std::array<void*, kRegisteredRelayMaxHelpers> push{};
  std::array<void*, kRegisteredRelayMaxHelpers> pull{};
  RegisteredRelayFlags myFlags{};
  RegisteredRelayFlags peerFlags{};

  Buffers() {
    for (int h = 0; h < kRegisteredRelayMaxHelpers; ++h) {
      push[h] = &local[h];
      pull[h] = &remote[h];
    }
  }

  RegisteredRelayArgs args(
      size_t bytes,
      int rank,
      int helpers,
      const RegisteredRelayLaunchPolicy& policy) {
    return makeRegisteredRelayArgs(
        &mine,
        &peer,
        &output,
        bytes,
        rank,
        helpers,
        policy,
        push.data(),
        pull.data(),
        &myFlags,
        &peerFlags);
  }
};

void expectFields(
    const RegisteredRelayArgs& args,
    Buffers& buffers,
    size_t bytes,
    size_t sliceBytes,
    int rank,
    int helpers,
    const RegisteredRelayLaunchPolicy& policy) {
  EXPECT_EQ(args.mine, &buffers.mine);
  EXPECT_EQ(args.peer, &buffers.peer);
  EXPECT_EQ(args.output, &buffers.output);
  EXPECT_EQ(args.vectors, bytes / 16);
  EXPECT_EQ(args.sliceVectors, sliceBytes / 16);
  EXPECT_EQ(args.slotVectors, policy.maxSliceBytes / 16);
  EXPECT_EQ(args.directWeight, policy.directWeight);
  EXPECT_EQ(args.helpers, helpers);
  EXPECT_EQ(args.directLanes, policy.directLanes);
  EXPECT_EQ(args.lanes, policy.lanes);
  EXPECT_EQ(args.slots, policy.slots);
  EXPECT_EQ(args.rank, rank);
  EXPECT_EQ(args.myFlags, &buffers.myFlags);
  EXPECT_EQ(args.peerFlags, &buffers.peerFlags);
  for (int h = 0; h < kRegisteredRelayMaxHelpers; ++h) {
    EXPECT_EQ(args.push[h], h < helpers ? buffers.push[h] : nullptr);
    EXPECT_EQ(args.pull[h], h < helpers ? buffers.pull[h] : nullptr);
  }
}

TEST(RegisteredRelayArgsTest, PrefillPoliciesPreserveAllFields) {
  Buffers buffers;
  for (size_t bytes : {72 * kMiB, 144 * kMiB}) {
    for (int rank : {0, 1}) {
      for (const auto& policy : {kAdaptive, kFixed}) {
        SCOPED_TRACE(bytes);
        SCOPED_TRACE(rank);
        expectFields(
            buffers.args(bytes, rank, 6, policy),
            buffers,
            bytes,
            256 * kKiB,
            rank,
            6,
            policy);
      }
    }
  }
}

TEST(RegisteredRelayArgsTest, DistinctPolicyFieldsAndHelperCounts) {
  Buffers buffers;
  const RegisteredRelayLaunchPolicy policy{64 * kKiB, 64 * kKiB, 5, 3, 8, 4};
  for (int helpers : {0, 1, 6, kRegisteredRelayMaxHelpers}) {
    SCOPED_TRACE(helpers);
    expectFields(
        buffers.args(72 * kMiB, 1, helpers, policy),
        buffers,
        72 * kMiB,
        64 * kKiB,
        1,
        helpers,
        policy);
  }
}

TEST(RegisteredRelayArgsTest, ZeroHelpersDoNotRequireRingArrays) {
  const auto args = makeRegisteredRelayArgs(
      nullptr,
      nullptr,
      nullptr,
      72 * kMiB,
      0,
      0,
      kFixed,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
  for (int h = 0; h < kRegisteredRelayMaxHelpers; ++h) {
    EXPECT_EQ(args.push[h], nullptr);
    EXPECT_EQ(args.pull[h], nullptr);
  }
}

TEST(RegisteredRelayArgsTest, AdaptiveSliceClampsAndRoundsDown) {
  EXPECT_EQ(registeredRelaySliceBytes(16, 6, kAdaptive), 16 * kKiB);
  EXPECT_EQ(registeredRelaySliceBytes(2 * kMiB, 6, kAdaptive), 32 * kKiB);
  EXPECT_EQ(registeredRelaySliceBytes(2 * kMiB + 16, 6, kAdaptive), 32 * kKiB);
  EXPECT_EQ(registeredRelaySliceBytes(4 * kMiB, 6, kAdaptive), 64 * kKiB);
  EXPECT_EQ(registeredRelaySliceBytes(14 * kMiB, 6, kAdaptive), 256 * kKiB);
  EXPECT_EQ(registeredRelaySliceBytes(144 * kMiB, 6, kAdaptive), 256 * kKiB);
}

TEST(RegisteredRelayArgsTest, FixedSliceRemainsFixedAtSmallerSizes) {
  for (size_t bytes : {16UL, 2 * kMiB, 4 * kMiB, 14 * kMiB}) {
    EXPECT_EQ(registeredRelaySliceBytes(bytes, 6, kFixed), 256 * kKiB);
  }
}

TEST(RegisteredRelayArgsTest, AdaptiveSliceDoesNotShrinkAllocatedSlots) {
  Buffers buffers;
  const auto args = buffers.args(2 * kMiB, 0, 6, kAdaptive);
  expectFields(args, buffers, 2 * kMiB, 32 * kKiB, 0, 6, kAdaptive);
  EXPECT_LT(args.sliceVectors, args.slotVectors);
}

} // namespace
} // namespace rcclx::relay
