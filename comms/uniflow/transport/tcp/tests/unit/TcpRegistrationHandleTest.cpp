// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/tcp/TcpRegistrationHandle.h"

#include <gtest/gtest.h>

namespace uniflow {

TEST(TcpRegistrationHandleTest, SerializeDeserializeRoundTrip) {
  TcpRegistrationHandle local(7, 4096);

  auto remote =
      TcpRemoteRegistrationHandle::deserialize(4096, local.serialize());

  ASSERT_TRUE(remote.hasValue()) << remote.error().message();
  EXPECT_EQ(remote.value()->transportType(), TransportType::TCP);
  EXPECT_EQ(remote.value()->segId(), 7);
  EXPECT_EQ(remote.value()->len(), 4096);
}

TEST(TcpRegistrationHandleTest, DeserializeRejectsLengthMismatch) {
  TcpRegistrationHandle local(7, 4096);

  auto remote =
      TcpRemoteRegistrationHandle::deserialize(2048, local.serialize());

  ASSERT_TRUE(remote.hasError());
  EXPECT_EQ(remote.error().code(), ErrCode::InvalidArgument);
}

} // namespace uniflow
