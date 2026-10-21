/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "comms/rcclx/develop/meta/relay/hipipc/hipipc_relay.h"

#include <gtest/gtest.h>
#include <limits>

namespace rcclx::relay::hipipc {
namespace {
detail::InputRecord inputRecord(void* base, RegistrationId id) {
  detail::InputRecord r;
  r.base = base;
  r.input = base;
  r.local.registration = id;
  r.local.allocationGeneration = 29;
  r.local.workspaceGeneration = 17;
  r.local.peerWorkspaceGeneration = 19;
  r.local.allocationBytes = (8ULL << 20) + 8192;
  r.local.offset = 4112;
  r.local.bytes = 2ULL << 20;
  r.local.base[0] = 7;
  return r;
}
TEST(HipIpcWorkspaceRecordTest, DescriptorRoundTripChecksLengthAndVersion) {
  int storage = 0;
  const auto d = inputRecord(&storage, {23, 31}).local;
  auto bytes = encodeRecord(d);
  const auto decoded = decodeRecord<RegistrationDescriptor>(bytes);
  EXPECT_EQ(decoded.offset, d.offset);
  EXPECT_EQ(decoded.registration, d.registration);
  EXPECT_EQ(decoded.base, d.base);
  bytes.pop_back();
  EXPECT_THROW(
      decodeRecord<RegistrationDescriptor>(bytes), std::invalid_argument);
  bytes.push_back(0);
  bytes.push_back(0);
  EXPECT_THROW(
      decodeRecord<RegistrationDescriptor>(bytes), std::invalid_argument);
  auto wrong = d;
  ++wrong.version;
  EXPECT_THROW(
      decodeRecord<RegistrationDescriptor>(encodeRecord(wrong)),
      std::invalid_argument);
  WorkspaceDescriptor workspace;
  workspace.capacity = 2ULL << 20;
  workspace.helperCount = 2;
  workspace.generation = 17;
  EXPECT_EQ(
      decodeRecord<WorkspaceDescriptor>(encodeRecord(workspace)).capacity,
      workspace.capacity);
}
TEST(HipIpcWorkspaceRecordTest, RangeIncludesAllocatorAndLogicalDisplacement) {
  EXPECT_NO_THROW(validateRange(4ULL << 20, 4112, 2ULL << 20));
  EXPECT_THROW(
      validateRange(4ULL << 20, (2ULL << 20) + 16, 2ULL << 20),
      std::invalid_argument);
  EXPECT_THROW(validateRange(4ULL << 20, 1, 16), std::invalid_argument);
  EXPECT_THROW(validateRange(4ULL << 20, 0, 0), std::invalid_argument);
  const auto max = std::numeric_limits<uint64_t>::max();
  EXPECT_THROW(validateRange(max, max - 15, 32), std::invalid_argument);
  EXPECT_FALSE(rangesOverlap(4096, 4096, 8192, 4096));
  EXPECT_TRUE(rangesOverlap(4096, 4096, 8176, 4096));
  EXPECT_FALSE(rangesOverlap(0, 16, max - 15, 16));
}
TEST(HipIpcWorkspaceRecordTest, SharedBaseRetainsIndependentAliasOwners) {
  detail::RegistrationRecords records;
  int storage = 0;
  auto firstOwner = std::make_shared<int>(1);
  auto secondOwner = std::make_shared<int>(2);
  std::weak_ptr<int> firstWeak = firstOwner, secondWeak = secondOwner;
  const auto first = inputRecord(&storage, {23, 31});
  auto second = inputRecord(&storage, {24, 32});
  second.local.offset += second.local.bytes;
  records.add(first, firstOwner);
  records.add(second, secondOwner);
  EXPECT_EQ(records.bases.size(), 1);
  EXPECT_EQ(records.bases.at(&storage).refs, 2);
  firstOwner.reset();
  secondOwner.reset();
  EXPECT_TRUE(records.erase(first.local.registration).empty());
  EXPECT_FALSE(firstWeak.expired());
  EXPECT_FALSE(secondWeak.expired());
  auto detached = records.erase(second.local.registration);
  EXPECT_TRUE(records.bases.empty());
  EXPECT_FALSE(firstWeak.expired());
  detached.clear();
  EXPECT_TRUE(firstWeak.expired());
  EXPECT_TRUE(secondWeak.expired());
}
TEST(HipIpcWorkspaceRecordTest, StaleTokensAndLiveBaseChangesAreRejected) {
  detail::RegistrationRecords records;
  int storage = 0;
  auto owner = std::make_shared<int>(1);
  auto first = inputRecord(&storage, {23, 31});
  records.add(first, owner);
  EXPECT_THROW(records.get({23, 30}), std::invalid_argument);
  auto changed = inputRecord(&storage, {24, 32});
  ++changed.local.allocationGeneration;
  EXPECT_THROW(records.add(changed, owner), std::invalid_argument);
  records.erase(first.local.registration);
  EXPECT_THROW(records.get(first.local.registration), std::invalid_argument);
  auto reused = inputRecord(&storage, {23, 33});
  ++reused.local.allocationGeneration;
  records.add(reused, owner);
  EXPECT_THROW(records.get(first.local.registration), std::invalid_argument);
  EXPECT_EQ(
      records.get(reused.local.registration).local.allocationGeneration,
      reused.local.allocationGeneration);
  EXPECT_THROW(records.add(reused, owner), std::invalid_argument);
}
} // namespace
} // namespace rcclx::relay::hipipc
