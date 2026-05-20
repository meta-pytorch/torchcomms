// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/controller/TcpController.h"

#include <sys/socket.h>
#include <unistd.h>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comms/uniflow/executor/ScopedEventBaseThread.h"

using namespace uniflow;
using namespace uniflow::controller;

// ---------------------------------------------------------------------------
// Helper: create a connected socket pair, wrap in TcpConns.
// One gets an EventBase (async-capable), the other is sync-only.
// The magic handshake must run concurrently on both sides.
// ---------------------------------------------------------------------------

struct AsyncSyncPair {
  std::unique_ptr<TcpConn<AsyncIO>> asyncConn;
  std::unique_ptr<TcpConn<SyncIO>> syncConn;
};

static AsyncSyncPair makeConnectedPair(EventBase& evb) {
  int fds[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) < 0) {
    return {};
  }

  // Magic handshake blocks until both sides have exchanged — run concurrently.
  std::unique_ptr<TcpConn<AsyncIO>> asyncConn;
  std::thread t([&]() { asyncConn = TcpConn<AsyncIO>::create(fds[0], evb); });
  auto syncConn = TcpConn<SyncIO>::create(fds[1]);
  t.join();

  if (!asyncConn || !syncConn) {
    return {};
  }

  return {std::move(asyncConn), std::move(syncConn)};
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class TcpAsyncSendRecvTest : public ::testing::Test {
 protected:
  ScopedEventBaseThread evbThread_{"async-sendrecv"};
};

// ========================== Span Send Tests ==========================

TEST_F(TcpAsyncSendRecvTest, SpanSendSyncRecv) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  std::vector<uint8_t> data = {'P', 'I', 'N', 'G'};
  auto future = asyncConn->send(std::span<const uint8_t>(data));

  std::vector<uint8_t> recv;
  auto recvResult = syncConn->recv(recv).get();
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(recv, data);

  auto sendResult = future.get();
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();
  EXPECT_EQ(sendResult.value(), 4u);
}

TEST_F(TcpAsyncSendRecvTest, VectorSendSyncRecv) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  std::vector<uint8_t> data = {'P', 'I', 'N', 'G'};
  std::vector<uint8_t> dataCopy = data;
  auto future = asyncConn->send(std::span<const uint8_t>(dataCopy));

  std::vector<uint8_t> recv;
  auto recvResult = syncConn->recv(recv).get();
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(recv, dataCopy);

  auto sendResult = future.get();
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();
  EXPECT_EQ(sendResult.value(), 4u);
}

// ========================== Alloc Recv Tests ==========================

TEST_F(TcpAsyncSendRecvTest, SyncSendAllocRecv) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  std::vector<uint8_t> data = {'P', 'O', 'N', 'G'};

  std::vector<uint8_t> recvData;
  auto future = asyncConn->recv(recvData);

  auto sendResult = syncConn->send(data).get();
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();

  auto recvResult = future.get();
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(recvData, data);
}

// ========================== Span Recv Tests ==========================

TEST_F(TcpAsyncSendRecvTest, SyncSendSpanRecv) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  std::vector<uint8_t> data = {'P', 'O', 'N', 'G'};

  std::vector<uint8_t> recvBuf(64);
  auto future = asyncConn->recv(std::span<uint8_t>(recvBuf));

  auto sendResult = syncConn->send(data).get();
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();

  auto recvResult = future.get();
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(recvResult.value(), data.size());
  recvBuf.resize(recvResult.value());
  EXPECT_EQ(recvBuf, data);
}

// ========================== Cross-Overload Tests ==========================

TEST_F(TcpAsyncSendRecvTest, SpanSendAllocRecv) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  std::vector<uint8_t> sendData = {'H', 'E', 'L', 'L', 'O'};
  auto sendFuture = asyncConn->send(std::span<const uint8_t>(sendData));

  std::vector<uint8_t> recv1;
  ASSERT_TRUE(syncConn->recv(recv1).get().hasValue());
  EXPECT_EQ(recv1, sendData);
  ASSERT_TRUE(sendFuture.get().hasValue());

  std::vector<uint8_t> replyData = {'W', 'O', 'R', 'L', 'D'};
  std::vector<uint8_t> recvData;
  auto recvFuture = asyncConn->recv(recvData);
  ASSERT_TRUE(syncConn->send(replyData).get().hasValue());

  auto recvResult = recvFuture.get();
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(recvData, replyData);
}

// ========================== Large Data Tests ==========================

TEST_F(TcpAsyncSendRecvTest, SpanSendLargeData) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  constexpr size_t kSize = 1 << 20; // 1MB
  std::vector<uint8_t> sendData(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    sendData[i] = static_cast<uint8_t>(i & 0xFF);
  }

  auto sendFuture = asyncConn->send(std::span<const uint8_t>(sendData));

  std::vector<uint8_t> recv;
  auto recvResult = syncConn->recv(recv).get();
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(recv.size(), kSize);
  EXPECT_EQ(recv, sendData);

  auto sendResult = sendFuture.get();
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();
  EXPECT_EQ(sendResult.value(), kSize);
}

TEST_F(TcpAsyncSendRecvTest, SpanRecvLargeData) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  constexpr size_t kSize = 1 << 20; // 1MB
  std::vector<uint8_t> sendData(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    sendData[i] = static_cast<uint8_t>(i & 0xFF);
  }

  std::vector<uint8_t> recvBuf(kSize);
  auto recvFuture = asyncConn->recv(std::span<uint8_t>(recvBuf));

  auto sendResult = syncConn->send(sendData).get();
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();

  auto recvResult = recvFuture.get();
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(recvResult.value(), kSize);
  EXPECT_EQ(recvBuf, sendData);
}

// ========================== Empty Message Tests ==========================

TEST_F(TcpAsyncSendRecvTest, SpanSendEmptyMessage) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  std::span<const uint8_t> empty;
  auto sendFuture = asyncConn->send(std::span<const uint8_t>(empty));

  std::vector<uint8_t> recv;
  auto recvResult = syncConn->recv(recv).get();
  ASSERT_TRUE(recvResult.hasValue());
  EXPECT_TRUE(recv.empty());

  auto sendResult = sendFuture.get();
  ASSERT_TRUE(sendResult.hasValue());
  EXPECT_EQ(sendResult.value(), 0u);
}

TEST_F(TcpAsyncSendRecvTest, AllocRecvEmptyMessage) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  std::vector<uint8_t> recvData;
  auto recvFuture = asyncConn->recv(recvData);

  std::vector<uint8_t> empty;
  ASSERT_TRUE(syncConn->send(empty).get().hasValue());

  auto recvResult = recvFuture.get();
  ASSERT_TRUE(recvResult.hasValue());
  EXPECT_EQ(recvData.size(), 0u);
}

// ========================== Error Cases ==========================

TEST_F(TcpAsyncSendRecvTest, SpanRecvTooSmall) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  std::vector<uint8_t> data(10, 0xAA);
  std::vector<uint8_t> recvBuf(4);
  auto recvFuture = asyncConn->recv(std::span<uint8_t>(recvBuf));

  ASSERT_TRUE(syncConn->send(data).get().hasValue());

  auto recvResult = recvFuture.get();
  EXPECT_TRUE(recvResult.hasError())
      << "asyncRecv(span) should fail when payload exceeds buffer";
}

// ========================== Simultaneous Send/Recv ==========================

TEST_F(TcpAsyncSendRecvTest, SimultaneousAsyncSendAndRecv) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  // Exercises the EPOLLOUT|EPOLLIN combined registration path.
  std::vector<uint8_t> sendData = {'S', 'E', 'N', 'D'};
  auto sendFuture = asyncConn->send(std::span<const uint8_t>(sendData));
  std::vector<uint8_t> recvData;
  auto recvFuture = asyncConn->recv(recvData);

  std::vector<uint8_t> recv;
  auto recvResult = syncConn->recv(recv).get();
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(recv, sendData);

  std::vector<uint8_t> replyData = {'R', 'E', 'C', 'V'};
  ASSERT_TRUE(syncConn->send(replyData).get().hasValue());

  auto sendResult = sendFuture.get();
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();
  EXPECT_EQ(sendResult.value(), 4u);

  auto asyncRecvResult = recvFuture.get();
  ASSERT_TRUE(asyncRecvResult.hasValue()) << asyncRecvResult.error().toString();
  EXPECT_EQ(recvData, replyData);
}

// ========================== Peer Closed Tests ==========================

TEST_F(TcpAsyncSendRecvTest, AllocRecvPeerClosed) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  std::vector<uint8_t> recvData;
  auto recvFuture = asyncConn->recv(recvData);

  // Barrier: ensure asyncRecv's dispatch has registered the fd before
  // closing the peer, otherwise EPOLLHUP races with registration.
  evbThread_.getEventBase()->dispatchAndWait([]() noexcept {});

  syncConn.reset();

  auto recvResult = recvFuture.get();
  EXPECT_TRUE(recvResult.hasError())
      << "asyncRecv should fail when peer closes";
}

TEST_F(TcpAsyncSendRecvTest, SpanRecvPeerClosed) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  std::vector<uint8_t> recvBuf(64);
  auto recvFuture = asyncConn->recv(std::span<uint8_t>(recvBuf));

  // Barrier: ensure recv's dispatch has registered the fd before
  // closing the peer, otherwise EPOLLHUP races with registration.
  evbThread_.getEventBase()->dispatchAndWait([]() noexcept {});

  syncConn.reset();

  auto recvResult = recvFuture.get();
  EXPECT_TRUE(recvResult.hasError())
      << "recv(span) should fail when peer closes";
}

// ========================== Concurrent Operation Tests
// ==========================

TEST_F(TcpAsyncSendRecvTest, ConcurrentSendReturnsResourceExhausted) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  constexpr size_t kSize = 1 << 20; // 1MB
  std::vector<uint8_t> bigData(kSize, 0xCC);
  auto firstSend = asyncConn->send(std::span<const uint8_t>(bigData));

  // Barrier: ensure the first send's dispatch has installed sendState.
  evbThread_.getEventBase()->dispatchAndWait([]() noexcept {});
  std::vector<uint8_t> small = {'X'};
  auto secondSend = asyncConn->send(std::span<const uint8_t>(small));

  auto secondResult = secondSend.get();
  EXPECT_TRUE(secondResult.hasError());
  EXPECT_EQ(secondResult.error().code(), ErrCode::ResourceExhausted);

  // Drain the first send so cleanup doesn't hang.
  std::vector<uint8_t> recv;
  syncConn->recv(recv).get();
  firstSend.get();
}

TEST_F(TcpAsyncSendRecvTest, ConcurrentRecvReturnsResourceExhausted) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  std::vector<uint8_t> recvData;
  auto firstRecv = asyncConn->recv(recvData);

  // Barrier: ensure the first recv's dispatch has installed recvState.
  evbThread_.getEventBase()->dispatchAndWait([]() noexcept {});
  auto secondRecv = asyncConn->recv(recvData);

  auto secondResult = secondRecv.get();
  EXPECT_TRUE(secondResult.hasError());
  EXPECT_EQ(secondResult.error().code(), ErrCode::ResourceExhausted);

  // Complete the first recv so cleanup doesn't hang.
  std::vector<uint8_t> msg = {'Y'};
  syncConn->send(msg).get();
  firstRecv.get();
}

TEST_F(TcpAsyncSendRecvTest, AsyncSendPeerClosed) {
  auto [asyncConn, syncConn] = makeConnectedPair(*evbThread_.getEventBase());
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  syncConn.reset();

  constexpr size_t kSize = 1 << 20; // 1MB — too large to fit in socket buffer
  std::vector<uint8_t> bigData(kSize, 0xDD);
  auto sendFuture = asyncConn->send(std::span<const uint8_t>(bigData));

  auto sendResult = sendFuture.get();
  EXPECT_TRUE(sendResult.hasError())
      << "asyncSend should fail when peer is closed";
}

TEST_F(TcpAsyncSendRecvTest, DestructionDuringInflightSend) {
  auto evb = evbThread_.getEventBase();
  auto [asyncConn, syncConn] = makeConnectedPair(*evb);
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  constexpr size_t kSize = 1 << 20; // 1MB
  std::vector<uint8_t> bigData(kSize, 0xEE);
  auto sendFuture = asyncConn->send(std::span<const uint8_t>(bigData));

  // Barrier: ensure the send's dispatch has installed sendState.
  evb->dispatchAndWait([]() noexcept {});

  // Destroy the TcpConn while send is in flight — should not hang.
  asyncConn.reset();

  // Don't assert success/error — just that the future resolves without hanging.
  auto sendResult = sendFuture.get();
}

TEST_F(TcpAsyncSendRecvTest, DestructionDuringInflightRecv) {
  auto evb = evbThread_.getEventBase();
  auto [asyncConn, syncConn] = makeConnectedPair(*evb);
  ASSERT_NE(asyncConn, nullptr);
  ASSERT_NE(syncConn, nullptr);

  std::vector<uint8_t> recvData;
  auto recvFuture = asyncConn->recv(recvData);

  // Barrier: ensure the recv's dispatch has installed recvState.
  evb->dispatchAndWait([]() noexcept {});

  // Destroy the TcpConn while recv is in flight — should not hang.
  asyncConn.reset();

  auto recvResult = recvFuture.get();
  EXPECT_TRUE(recvResult.hasError())
      << "asyncRecv should fail when TcpConn is destroyed";
}
