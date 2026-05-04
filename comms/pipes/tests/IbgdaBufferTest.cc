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
  // HostLKey converts implicitly to NetworkLKey during the slot assignment.
  char data[64];
  HostLKey hostLKey(0x1234);
  HostRKey hostRKey(0x5678);

  NetworkLKeys lkeys(1);
  lkeys[0] = hostLKey;
  IbgdaLocalBuffer localBuf(data, lkeys);
  EXPECT_EQ(localBuf.ptr, data);
  EXPECT_EQ(localBuf.lkey_per_device[0].value, htobe32(0x1234));

  NetworkRKeys rkeys(1);
  rkeys[0] = hostRKey;
  IbgdaRemoteBuffer remoteBuf(data, rkeys);
  EXPECT_EQ(remoteBuf.ptr, data);
  EXPECT_EQ(remoteBuf.rkey_per_device[0].value, htobe32(0x5678));
}

// =============================================================================
// Buffer Tests
// =============================================================================

TEST(IbgdaBufferTest, LocalBufferOperations) {
  char data[64];
  NetworkLKey lkey(0x1234);

  NetworkLKeys keys(1);
  keys[0] = lkey;
  IbgdaLocalBuffer buf(data, keys);
  EXPECT_EQ(buf.ptr, data);
  EXPECT_EQ(buf.lkey_per_device[0], lkey);

  // SubBuffer with offset
  auto sub = buf.subBuffer(16);
  EXPECT_EQ(sub.ptr, data + 16);
  EXPECT_EQ(sub.lkey_per_device[0], lkey);
}

TEST(IbgdaBufferTest, RemoteBufferOperations) {
  char data[64];
  NetworkRKey rkey(0x5678);

  NetworkRKeys keys(1);
  keys[0] = rkey;
  IbgdaRemoteBuffer buf(data, keys);
  EXPECT_EQ(buf.ptr, data);
  EXPECT_EQ(buf.rkey_per_device[0], rkey);

  // SubBuffer with offset
  auto sub = buf.subBuffer(32);
  EXPECT_EQ(sub.ptr, data + 32);
  EXPECT_EQ(sub.rkey_per_device[0], rkey);
}

// =============================================================================
// Multi-NIC Buffer Tests
// =============================================================================

TEST(IbgdaBufferTest, LocalBufferMultiKeyConstruction) {
  char data[64];
  NetworkLKeys keys(2);
  keys[0] = NetworkLKey(0x1111);
  keys[1] = NetworkLKey(0x2222);
  IbgdaLocalBuffer buf(data, keys);

  EXPECT_EQ(buf.ptr, data);
  EXPECT_EQ(buf.lkey_per_device[0].value, 0x1111u);
  EXPECT_EQ(buf.lkey_per_device[1].value, 0x2222u);
  EXPECT_EQ(buf.lkey_per_device.size, 2);
}

TEST(IbgdaBufferTest, RemoteBufferMultiKeyConstruction) {
  char data[64];
  NetworkRKeys keys(2);
  keys[0] = NetworkRKey(0x3333);
  keys[1] = NetworkRKey(0x4444);
  IbgdaRemoteBuffer buf(data, keys);

  EXPECT_EQ(buf.ptr, data);
  EXPECT_EQ(buf.rkey_per_device[0].value, 0x3333u);
  EXPECT_EQ(buf.rkey_per_device[1].value, 0x4444u);
  EXPECT_EQ(buf.rkey_per_device.size, 2);
}

TEST(IbgdaBufferTest, LocalBufferSubBufferPropagatesAllKeys) {
  // subBuffer must preserve both lkeys[0] AND lkeys[1].
  char data[64];
  NetworkLKeys keys(2);
  keys[0] = NetworkLKey(0xAAAA);
  keys[1] = NetworkLKey(0xBBBB);
  IbgdaLocalBuffer buf(data, keys);

  auto sub = buf.subBuffer(16);
  EXPECT_EQ(sub.ptr, data + 16);
  EXPECT_EQ(sub.lkey_per_device[0].value, 0xAAAAu);
  EXPECT_EQ(sub.lkey_per_device[1].value, 0xBBBBu);
}

TEST(IbgdaBufferTest, RemoteBufferSubBufferPropagatesAllKeys) {
  char data[64];
  NetworkRKeys keys(2);
  keys[0] = NetworkRKey(0xCCCC);
  keys[1] = NetworkRKey(0xDDDD);
  IbgdaRemoteBuffer buf(data, keys);

  auto sub = buf.subBuffer(32);
  EXPECT_EQ(sub.ptr, data + 32);
  EXPECT_EQ(sub.rkey_per_device[0].value, 0xCCCCu);
  EXPECT_EQ(sub.rkey_per_device[1].value, 0xDDDDu);
}

TEST(IbgdaBufferTest, DefaultConstructorZeroInitsAllKeys) {
  // Default-constructed buffers must have size=0 and storage zero.
  IbgdaLocalBuffer localBuf;
  EXPECT_EQ(localBuf.ptr, nullptr);
  EXPECT_EQ(localBuf.lkey_per_device.size, 0);
  for (int n = 0; n < kMaxNicsPerGpu; n++) {
    EXPECT_EQ(localBuf.lkey_per_device.values[n].value, 0u);
  }

  IbgdaRemoteBuffer remoteBuf;
  EXPECT_EQ(remoteBuf.ptr, nullptr);
  EXPECT_EQ(remoteBuf.rkey_per_device.size, 0);
  for (int n = 0; n < kMaxNicsPerGpu; n++) {
    EXPECT_EQ(remoteBuf.rkey_per_device.values[n].value, 0u);
  }
}

} // namespace comms::pipes::tests
