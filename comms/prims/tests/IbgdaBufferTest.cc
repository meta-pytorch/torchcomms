// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <endian.h>

#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace comms::prims::tests {

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

TEST(IbgdaBufferTest, MakeChannelSlicesEveryChannelResource) {
  constexpr int kMaxChannels = 3;
  constexpr int kNumLanes = 2;
  constexpr int kChannel = 2;
  constexpr std::size_t kChannelBytes = 64;
  constexpr int kPipelineDepth = 4;

  char sendStaging[kMaxChannels * kChannelBytes]{};
  char recvStaging[kMaxChannels * kChannelBytes]{};
  char remoteRecvStaging[kMaxChannels * kChannelBytes]{};
  char localSignals
      [(kNumLanes * kMaxChannels + kMaxChannels) * kSendRecvSignalSlotStride]{};
  char remoteSignals[sizeof(localSignals)]{};
  char localCounters[kMaxChannels * kSendRecvSignalSlotStride]{};
  char localCompletions[kMaxChannels * kSendRecvSignalSlotStride]{};
  IbSendCompletionSlot completionSlots[kPipelineDepth]{};

  NetworkLKeys localKeys(2);
  localKeys[0] = NetworkLKey(0x1111);
  localKeys[1] = NetworkLKey(0x2222);
  NetworkRKeys remoteKeys(2);
  remoteKeys[0] = NetworkRKey(0x3333);
  remoteKeys[1] = NetworkRKey(0x4444);

  const IbChannelLayout layout{
      .sendStagingBuf = IbgdaLocalBuffer(sendStaging, localKeys),
      .recvStagingBuf = IbgdaRemoteBuffer(remoteRecvStaging, remoteKeys),
      .recvStagingPtr = recvStaging,
      .localSignalBuf = IbgdaLocalBuffer(localSignals, localKeys),
      .remoteSignalBuf = IbgdaRemoteBuffer(remoteSignals, remoteKeys),
      .localCounterBuf = IbgdaLocalBuffer(localCounters, localKeys),
      .localCounterCompletionBuf =
          IbgdaLocalBuffer(localCompletions, localKeys),
      .maxChannels = kMaxChannels,
      .numLanes = kNumLanes,
      .pipelineDepth = kPipelineDepth,
      .perChannelBufferSize = kChannelBytes,
  };

  const IbChannel channel = makeIbChannel(layout, kChannel, completionSlots);
  const IbChannel previousChannel = makeIbChannel(layout, kChannel - 1);
  const std::size_t stagingOffset = kChannel * kChannelBytes;
  const int dataReadySlot = kChannel * kNumLanes;
  const int slotFreeSlot = kNumLanes * kMaxChannels + kChannel;

  EXPECT_EQ(channel.sendStaging.ptr, sendStaging + stagingOffset);
  EXPECT_EQ(channel.recvStaging, recvStaging + stagingOffset);
  EXPECT_EQ(channel.remoteRecvStaging.ptr, remoteRecvStaging + stagingOffset);
  EXPECT_EQ(
      channel.dataReady.ptr,
      localSignals + sendRecvSignalSlotOffset(dataReadySlot));
  EXPECT_EQ(
      channel.remoteDataReady.ptr,
      remoteSignals + sendRecvSignalSlotOffset(dataReadySlot));
  EXPECT_EQ(
      channel.slotFree.ptr,
      localSignals + sendRecvSignalSlotOffset(slotFreeSlot));
  EXPECT_EQ(
      channel.remoteSlotFree.ptr,
      remoteSignals + sendRecvSignalSlotOffset(slotFreeSlot));
  EXPECT_EQ(
      channel.nicDoneWait.ptr,
      localCounters + sendRecvSignalSlotOffset(kChannel));
  EXPECT_EQ(
      channel.nicDoneCompletion.ptr,
      localCompletions + sendRecvSignalSlotOffset(kChannel));
  EXPECT_EQ(channel.sendCompletionSlots, completionSlots);
  EXPECT_EQ(
      channel.sendStaging.ptr,
      static_cast<char*>(previousChannel.sendStaging.ptr) + kChannelBytes);
  EXPECT_EQ(channel.recvStaging, previousChannel.recvStaging + kChannelBytes);
  EXPECT_NE(channel.dataReady.ptr, previousChannel.dataReady.ptr);
  EXPECT_NE(channel.slotFree.ptr, previousChannel.slotFree.ptr);
  EXPECT_NE(channel.nicDoneWait.ptr, previousChannel.nicDoneWait.ptr);

  EXPECT_EQ(channel.sendStaging.lkey_per_device.size, 2);
  EXPECT_EQ(channel.sendStaging.lkey_per_device[0], localKeys[0]);
  EXPECT_EQ(channel.sendStaging.lkey_per_device[1], localKeys[1]);
  EXPECT_EQ(channel.remoteRecvStaging.rkey_per_device.size, 2);
  EXPECT_EQ(channel.remoteRecvStaging.rkey_per_device[0], remoteKeys[0]);
  EXPECT_EQ(channel.remoteRecvStaging.rkey_per_device[1], remoteKeys[1]);
  EXPECT_EQ(channel.remoteDataReady.rkey_per_device[0], remoteKeys[0]);
  EXPECT_EQ(channel.remoteSlotFree.rkey_per_device[1], remoteKeys[1]);
}

TEST(IbgdaBufferTest, QpSlotWithinNicUsesChannelDirectionLaneOrder) {
  constexpr uint32_t kDirectionCount = 2;
  constexpr uint32_t kQpsPerConnection = 2;

  EXPECT_EQ(
      ibQpSlotWithinNic(
          0, IbDirection::Send, kDirectionCount, kQpsPerConnection, 0),
      0);
  EXPECT_EQ(
      ibQpSlotWithinNic(
          0, IbDirection::Send, kDirectionCount, kQpsPerConnection, 1),
      1);
  EXPECT_EQ(
      ibQpSlotWithinNic(
          0, IbDirection::Recv, kDirectionCount, kQpsPerConnection, 0),
      2);
  EXPECT_EQ(
      ibQpSlotWithinNic(
          1, IbDirection::Send, kDirectionCount, kQpsPerConnection, 0),
      4);
  EXPECT_EQ(
      ibQpSlotWithinNic(
          1, IbDirection::Recv, kDirectionCount, kQpsPerConnection, 1),
      7);
  EXPECT_EQ(
      ibQpSlotWithinNic(
          1,
          IbDirection::Send,
          /*directionCount=*/1,
          kQpsPerConnection,
          1),
      3);

  EXPECT_EQ(
      ibCommandQueueSlot(
          1,
          IbDirection::Recv,
          kDirectionCount,
          kQpsPerConnection,
          1,
          /*numNics=*/3,
          /*nicId=*/2),
      23);
}

TEST(IbgdaBufferTest, MakeChannelAllowsDisabledSendRecvLayout) {
  IbSendCompletionSlot completionSlots[2]{};

  const IbChannel channel =
      makeIbChannel(IbChannelLayout{}, 3, completionSlots);

  EXPECT_EQ(channel.sendStaging.ptr, nullptr);
  EXPECT_EQ(channel.recvStaging, nullptr);
  EXPECT_EQ(channel.remoteRecvStaging.ptr, nullptr);
  EXPECT_EQ(channel.dataReady.ptr, nullptr);
  EXPECT_EQ(channel.remoteDataReady.ptr, nullptr);
  EXPECT_EQ(channel.slotFree.ptr, nullptr);
  EXPECT_EQ(channel.remoteSlotFree.ptr, nullptr);
  EXPECT_EQ(channel.sendCompletionSlots, completionSlots);
}

} // namespace comms::prims::tests
