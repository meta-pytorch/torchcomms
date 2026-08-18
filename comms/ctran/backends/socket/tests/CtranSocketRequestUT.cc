// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <array>

#include <gtest/gtest.h>

#include "comms/ctran/backends/socket/CtranSocketBase.h"

class CtranSocketRequestTest : public ::testing::Test {
 public:
  CtranSocketRequestTest() = default;

 protected:
  void SetUp() override {}
};

TEST_F(CtranSocketRequestTest, Complete) {
  CtranSocketRequest req;
  auto res = req.complete();
  EXPECT_EQ(res, commSuccess);
  EXPECT_TRUE(req.isComplete());
}

// A maximum-size raw payload must survive the frame round trip. wireSize is
// checked by WireSizeTracksPayload -- at max size it cannot distinguish
// "tracks the payload" from "sends the whole buffer", so it is not asserted
// here.
TEST_F(CtranSocketRequestTest, MaxPayloadRoundTrip) {
  std::array<unsigned char, CTRAN_CTRL_MAX_PAYLOAD_SIZE> input{};
  std::array<unsigned char, CTRAN_CTRL_MAX_PAYLOAD_SIZE> output{};
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<unsigned char>(i % 251);
  }

  SocketCtrlPacket packet;
  EXPECT_TRUE(packet.copyFrom(input.data(), input.size()));
  EXPECT_EQ(packet.payloadSize, input.size());
  EXPECT_TRUE(packet.copyTo(output.data(), output.size()));
  EXPECT_EQ(output, input);
}

// A short payload must not put the full buffer on the wire.
TEST_F(CtranSocketRequestTest, WireSizeTracksPayload) {
  SocketCtrlPacket packet;
  const unsigned char value = 7;
  ASSERT_TRUE(packet.copyFrom(&value, sizeof(value)));
  EXPECT_EQ(
      packet.wireSize(), offsetof(SocketCtrlPacket, payload) + sizeof(value));
  EXPECT_LT(packet.wireSize(), sizeof(SocketCtrlPacket));
}

TEST_F(CtranSocketRequestTest, PayloadValidation) {
  std::array<unsigned char, CTRAN_CTRL_MAX_PAYLOAD_SIZE + 1> oversized{};
  SocketCtrlPacket packet;
  EXPECT_FALSE(packet.copyFrom(oversized.data(), oversized.size()));
  EXPECT_FALSE(packet.copyFrom(nullptr, 1));

  const unsigned char value = 42;
  unsigned char output = 0;
  ASSERT_TRUE(packet.copyFrom(&value, sizeof(value)));
  // Size mismatch on receive must be rejected, not silently truncated.
  EXPECT_FALSE(packet.copyTo(&output, 0));
  EXPECT_FALSE(packet.copyTo(nullptr, sizeof(output)));
  EXPECT_TRUE(packet.copyTo(&output, sizeof(output)));
  EXPECT_EQ(output, value);
}

TEST_F(CtranSocketRequestTest, EmptyPayload) {
  SocketCtrlPacket packet;
  EXPECT_TRUE(packet.copyFrom(nullptr, 0));
  EXPECT_EQ(packet.wireSize(), offsetof(SocketCtrlPacket, payload));
  EXPECT_TRUE(packet.copyTo(nullptr, 0));
}
