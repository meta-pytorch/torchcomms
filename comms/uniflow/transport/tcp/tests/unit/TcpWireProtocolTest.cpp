// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/tcp/TcpWireProtocol.h"

#include <gtest/gtest.h>

namespace uniflow {

TEST(TcpWireProtocolTest, HeaderRoundTrip) {
  TcpMsgHeader header{
      .version = kTcpWireVersion,
      .op = static_cast<uint8_t>(TcpOp::Write),
      .flags = 3,
      .rsvd = 0,
      .reqId = 11,
      .segId = 22,
      .offset = 33,
      .len = 44,
  };

  auto parsed = deserializeTcpHeader(serializeTcpHeader(header));

  ASSERT_TRUE(parsed.hasValue()) << parsed.error().message();
  EXPECT_EQ(parsed.value().version, kTcpWireVersion);
  EXPECT_EQ(parsed.value().op, static_cast<uint8_t>(TcpOp::Write));
  EXPECT_EQ(parsed.value().flags, 3);
  EXPECT_EQ(parsed.value().reqId, 11);
  EXPECT_EQ(parsed.value().segId, 22);
  EXPECT_EQ(parsed.value().offset, 33);
  EXPECT_EQ(parsed.value().len, 44);
}

TEST(TcpWireProtocolTest, RejectsTruncatedHeader) {
  std::vector<uint8_t> data(sizeof(TcpMsgHeader) - 1);

  auto parsed = deserializeTcpHeader(data);

  ASSERT_TRUE(parsed.hasError());
  EXPECT_EQ(parsed.error().code(), ErrCode::InvalidArgument);
}

TEST(TcpWireProtocolTest, PayloadClassification) {
  EXPECT_TRUE(tcpOpHasPayload(TcpOp::Write));
  EXPECT_TRUE(tcpOpHasPayload(TcpOp::ReadReply));
  EXPECT_FALSE(tcpOpHasPayload(TcpOp::Ack));
  EXPECT_FALSE(tcpOpHasPayload(TcpOp::Notification));
}

} // namespace uniflow
