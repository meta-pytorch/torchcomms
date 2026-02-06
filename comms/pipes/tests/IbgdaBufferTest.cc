// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <endian.h>

#include "comms/pipes/IbgdaBuffer.h"

namespace comms::pipes::tests {

// =============================================================================
// Key Conversion Tests
// =============================================================================

TEST(IbgdaBufferTest, KeyImplicitConversion) {
  // Test implicit conversion from HostLKey to NetworkLKey
  HostLKey hostLKey(0x12345678);
  NetworkLKey networkLKey = hostLKey; // Implicit conversion
  EXPECT_EQ(networkLKey.value, htobe32(0x12345678));
  EXPECT_EQ(be32toh(networkLKey.value), hostLKey.value);

  // Test implicit conversion from HostRKey to NetworkRKey
  HostRKey hostRKey(0xABCDEF01);
  NetworkRKey networkRKey = hostRKey; // Implicit conversion
  EXPECT_EQ(networkRKey.value, htobe32(0xABCDEF01));
  EXPECT_EQ(be32toh(networkRKey.value), hostRKey.value);
}

TEST(IbgdaBufferTest, KeyImplicitConversionInBufferConstructor) {
  // Test that implicit conversion works when constructing buffer descriptors
  char data[64];
  HostLKey hostLKey(0x1234);
  HostRKey hostRKey(0x5678);

  // IbgdaLocalBuffer should accept HostLKey via implicit conversion
  IbgdaLocalBuffer localBuf(data, hostLKey);
  EXPECT_EQ(localBuf.ptr, data);
  EXPECT_EQ(localBuf.lkey.value, htobe32(0x1234));

  // IbgdaRemoteBuffer should accept HostRKey via implicit conversion
  IbgdaRemoteBuffer remoteBuf(data, hostRKey);
  EXPECT_EQ(remoteBuf.ptr, data);
  EXPECT_EQ(remoteBuf.rkey.value, htobe32(0x5678));
}

// =============================================================================
// Buffer Tests
// =============================================================================

TEST(IbgdaBufferTest, LocalBufferOperations) {
  char data[64];
  NetworkLKey lkey(0x1234);

  // Construction
  IbgdaLocalBuffer buf(data, lkey);
  EXPECT_EQ(buf.ptr, data);
  EXPECT_EQ(buf.lkey, lkey);

  // SubBuffer with offset
  auto sub = buf.subBuffer(16);
  EXPECT_EQ(sub.ptr, data + 16);
  EXPECT_EQ(sub.lkey, lkey);
}

TEST(IbgdaBufferTest, RemoteBufferOperations) {
  char data[64];
  NetworkRKey rkey(0x5678);

  // Construction
  IbgdaRemoteBuffer buf(data, rkey);
  EXPECT_EQ(buf.ptr, data);
  EXPECT_EQ(buf.rkey, rkey);

  // SubBuffer with offset
  auto sub = buf.subBuffer(32);
  EXPECT_EQ(sub.ptr, data + 32);
  EXPECT_EQ(sub.rkey, rkey);
}

} // namespace comms::pipes::tests
