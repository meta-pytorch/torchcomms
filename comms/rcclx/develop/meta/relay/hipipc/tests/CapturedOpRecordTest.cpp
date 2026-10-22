/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>
#include "comms/rcclx/develop/meta/relay/hipipc/hipipc_relay.h"

namespace rcclx::relay::hipipc {
namespace {
CapturedDescriptor staged() {
  CapturedDescriptor result;
  result.input.workspaceGeneration = 17;
  result.input.peerWorkspaceGeneration = 19;
  result.input.registration = {23, 29};
  result.input.bytes = 2ULL << 20;
  return result;
}
TEST(CapturedOpRecordTest, StagedRoleHasNoProducerIpcMetadata) {
  auto descriptor = staged();
  EXPECT_NO_THROW(validateCapturedDescriptor(descriptor));
  descriptor.input.base[0] = 1;
  EXPECT_THROW(validateCapturedDescriptor(descriptor), std::invalid_argument);
  descriptor = staged();
  descriptor.input.allocationGeneration = 31;
  EXPECT_THROW(validateCapturedDescriptor(descriptor), std::invalid_argument);
  descriptor = staged();
  descriptor.role = static_cast<CaptureRole>(2);
  EXPECT_THROW(validateCapturedDescriptor(descriptor), std::invalid_argument);
}
TEST(CapturedOpRecordTest, DirectRolePreservesBaseOffsetAndGeneration) {
  auto descriptor = staged();
  descriptor.role = CaptureRole::Direct;
  descriptor.input.allocationGeneration = 31;
  descriptor.input.allocationBytes = 4ULL << 20;
  descriptor.input.offset = 4112;
  descriptor.input.base[0] = 1;
  const auto decoded =
      decodeRecord<CapturedDescriptor>(encodeRecord(descriptor));
  EXPECT_NO_THROW(validateCapturedDescriptor(decoded));
  EXPECT_EQ(decoded.input.offset, 4112);
  EXPECT_EQ(decoded.input.registration, descriptor.input.registration);
  descriptor.input.offset = descriptor.input.allocationBytes;
  EXPECT_THROW(validateCapturedDescriptor(descriptor), std::invalid_argument);
}
TEST(CapturedOpRecordTest, OneAbiExactLengthAndNonOwningToken) {
  auto descriptor = staged();
  auto bytes = encodeRecord(descriptor);
  bytes.pop_back();
  EXPECT_THROW(decodeRecord<CapturedDescriptor>(bytes), std::invalid_argument);
  ++descriptor.version;
  EXPECT_THROW(
      decodeRecord<CapturedDescriptor>(encodeRecord(descriptor)),
      std::invalid_argument);
  descriptor = staged();
  descriptor.input.registration.generation = 0;
  EXPECT_THROW(validateCapturedDescriptor(descriptor), std::invalid_argument);
  static_assert(sizeof(CapturedOp) == sizeof(RegistrationId));
  static_assert(std::is_trivially_copyable_v<CapturedOp>);
}
} // namespace
} // namespace rcclx::relay::hipipc
