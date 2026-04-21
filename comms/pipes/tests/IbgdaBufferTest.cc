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

// =============================================================================
// Multi-NIC Buffer Tests
// =============================================================================

TEST(IbgdaBufferTest, LocalBufferMultiKeyConstruction) {
  // Multi-key constructor takes the per-NIC lkeys array by reference.
  char data[64];
  NetworkLKey keys[kMaxIbgdaNics]{NetworkLKey(0x1111), NetworkLKey(0x2222)};
  IbgdaLocalBuffer buf(data, keys);

  EXPECT_EQ(buf.ptr, data);
  EXPECT_EQ(buf.lkeys[0].value, 0x1111u);
  EXPECT_EQ(buf.lkeys[1].value, 0x2222u);

  // Union aliasing: legacy `.lkey` member reads the same memory as `.lkeys[0]`,
  // so existing call sites using `.lkey.value` continue to work unchanged.
  EXPECT_EQ(buf.lkey.value, 0x1111u);
}

TEST(IbgdaBufferTest, RemoteBufferMultiKeyConstruction) {
  char data[64];
  NetworkRKey keys[kMaxIbgdaNics]{NetworkRKey(0x3333), NetworkRKey(0x4444)};
  IbgdaRemoteBuffer buf(data, keys);

  EXPECT_EQ(buf.ptr, data);
  EXPECT_EQ(buf.rkeys[0].value, 0x3333u);
  EXPECT_EQ(buf.rkeys[1].value, 0x4444u);
  EXPECT_EQ(buf.rkey.value, 0x3333u); // union alias
}

TEST(IbgdaBufferTest, LocalBufferSubBufferPropagatesAllKeys) {
  // subBuffer must preserve BOTH lkeys[0] AND lkeys[1] (loop in ctor).
  char data[64];
  NetworkLKey keys[kMaxIbgdaNics]{NetworkLKey(0xAAAA), NetworkLKey(0xBBBB)};
  IbgdaLocalBuffer buf(data, keys);

  auto sub = buf.subBuffer(16);
  EXPECT_EQ(sub.ptr, data + 16);
  EXPECT_EQ(sub.lkeys[0].value, 0xAAAAu);
  EXPECT_EQ(sub.lkeys[1].value, 0xBBBBu);
}

TEST(IbgdaBufferTest, RemoteBufferSubBufferPropagatesAllKeys) {
  char data[64];
  NetworkRKey keys[kMaxIbgdaNics]{NetworkRKey(0xCCCC), NetworkRKey(0xDDDD)};
  IbgdaRemoteBuffer buf(data, keys);

  auto sub = buf.subBuffer(32);
  EXPECT_EQ(sub.ptr, data + 32);
  EXPECT_EQ(sub.rkeys[0].value, 0xCCCCu);
  EXPECT_EQ(sub.rkeys[1].value, 0xDDDDu);
}

TEST(IbgdaBufferTest, DefaultConstructorZeroInitsAllKeys) {
  // Default-constructed buffers must have all per-NIC keys zeroed.
  IbgdaLocalBuffer localBuf;
  EXPECT_EQ(localBuf.ptr, nullptr);
  for (int n = 0; n < kMaxIbgdaNics; n++) {
    EXPECT_EQ(localBuf.lkeys[n].value, 0u);
  }

  IbgdaRemoteBuffer remoteBuf;
  EXPECT_EQ(remoteBuf.ptr, nullptr);
  for (int n = 0; n < kMaxIbgdaNics; n++) {
    EXPECT_EQ(remoteBuf.rkeys[n].value, 0u);
  }
}

} // namespace comms::pipes::tests
