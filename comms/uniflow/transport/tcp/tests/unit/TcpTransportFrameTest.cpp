// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/tcp/TcpTransport.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/tcp/TcpWireProtocol.h"

namespace uniflow {

// handleFrame() acts on peer-supplied segId/offset/len before touching a
// registered buffer, so its bounds checks are this transport's memory-safety
// boundary. They are correct as written but nothing exercised them, meaning a
// refactor could drop a clause silently. These tests drive frames straight into
// handleFrame() and assert two things per case: the registered buffer is left
// untouched, and an Error frame is queued back to the peer.
//
// The last test covers the other buffer handleFrame() writes into: a READ_REPLY
// targets a *caller-owned* get() destination, which the caller may free as soon
// as the op's future resolves.
/// A connection whose send() always fails; recv() is never used by these tests.
class FailingConn : public controller::Conn {
 public:
  std::future<Result<size_t>> send(std::span<const uint8_t>) override {
    return make_ready_future<Result<size_t>>(
        Err(ErrCode::ConnectionFailed, "test: send failed"));
  }
  std::future<Result<size_t>> recv(std::vector<uint8_t>&) override {
    return make_ready_future<Result<size_t>>(
        Err(ErrCode::ConnectionFailed, "test: recv failed"));
  }
  std::future<Result<size_t>> recv(std::span<uint8_t>) override {
    return make_ready_future<Result<size_t>>(
        Err(ErrCode::ConnectionFailed, "test: recv failed"));
  }
  void close() override {
    closed = true;
  }
  bool closed{false};
};

class TcpTransportFrameTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kSegId = 42;
  static constexpr size_t kSegLen = 256;

  void SetUp() override {
    evbThread_ = std::make_unique<ScopedEventBaseThread>("tcp-frame-test");
    registry_ = std::make_shared<TcpSegmentRegistry>();
    cudaApi_ = std::make_shared<::testing::NiceMock<MockCudaApi>>();
    // Enough for CudaDeviceGuard; the VRAM copies themselves are stubbed
    // per-test.
    ON_CALL(*cudaApi_, getDevice())
        .WillByDefault(::testing::Return(Result<int>(0)));
    ON_CALL(*cudaApi_, setDevice(::testing::_))
        .WillByDefault(::testing::Return(Ok()));
    ON_CALL(*cudaApi_, streamSynchronize(::testing::_))
        .WillByDefault(::testing::Return(Ok()));

    transport_ = std::make_unique<TcpTransport>(
        /*deviceId=*/-1,
        evbThread_->getEventBase(),
        registry_,
        controller::TcpSocketConfig{},
        /*host=*/"127.0.0.1",
        cudaApi_);

    // A registered DRAM segment filled with a known pattern, so any
    // out-of-contract write is detectable.
    segment_.assign(kSegLen, std::byte{0xAB});
    registry_->add(
        kSegId, segment_.data(), kSegLen, MemoryType::DRAM, /*deviceId=*/-1);
    pristine_ = segment_;
  }

  void TearDown() override {
    if (transport_) {
      transport_->shutdown();
      transport_.reset();
    }
    evbThread_.reset();
  }

  /// Build one framed message: header followed by `payload` bytes.
  static std::vector<uint8_t> makeFrame(
      TcpOp op,
      uint64_t segId,
      uint64_t offset,
      uint64_t len,
      size_t payloadBytes) {
    TcpMsgHeader header;
    header.op = static_cast<uint8_t>(op);
    header.reqId = 1;
    header.segId = segId;
    header.offset = offset;
    header.len = len;

    std::vector<uint8_t> frame(sizeof(TcpMsgHeader) + payloadBytes);
    std::memcpy(frame.data(), &header, sizeof(header));
    // Distinct from the segment pattern so a stray write is unmistakable.
    std::fill(frame.begin() + sizeof(TcpMsgHeader), frame.end(), uint8_t{0xCD});
    return frame;
  }

  void feed(const std::vector<uint8_t>& frame) {
    transport_->handleFrame(frame);
  }

  /// True if the transport queued at least one Error frame back to the peer.
  bool queuedErrorFrame() const {
    std::lock_guard<std::mutex> lk(transport_->outMu_);
    for (const auto& item : transport_->outQueue_) {
      auto header = deserializeTcpHeader(item.bytes);
      if (header.hasValue() &&
          static_cast<TcpOp>(header.value().op) == TcpOp::Error) {
        return true;
      }
    }
    return false;
  }

  bool segmentUntouched() const {
    return segment_ == pristine_;
  }

  /// Registers `chunks` in-flight READ chunks sharing one op state, the way a
  /// multi-chunk get() does. Chunk reqId 1 targets `dst`; the rest stand in for
  /// siblings whose replies have not arrived yet. VRAM so the copy runs through
  /// the mocked CudaApi. Returns the caller's future.
  std::future<Status> postReadChunks(void* dst, size_t len, size_t chunks) {
    auto state = std::make_shared<TcpOpState>();
    state->remaining = chunks;
    auto future = state->promise.get_future();
    std::lock_guard<std::mutex> lk(transport_->inflightMu_);
    for (size_t i = 0; i < chunks; ++i) {
      transport_->inflight_[i + 1] = TcpInflight{
          state,
          i == 0 ? dst : nullptr,
          len,
          /*isRead=*/true,
          MemoryType::VRAM,
          /*deviceId=*/0,
          /*stream=*/nullptr};
    }
    return future;
  }

  void failAllPending() {
    transport_->failAllPending("test: connection failed");
  }

  size_t inflightCount() {
    std::lock_guard<std::mutex> lk(transport_->inflightMu_);
    return transport_->inflight_.size();
  }

  Status admitInflight(uint64_t reqId, TcpInflight entry) {
    return transport_->admitInflight(reqId, std::move(entry));
  }

  bool enqueueReaderFrame(size_t bytes) {
    return transport_->enqueueFrame(
        std::vector<uint8_t>(bytes, 0), /*mayBlock=*/false);
  }

  size_t outQueueBytes() {
    std::lock_guard<std::mutex> lk(transport_->outMu_);
    return transport_->outBytes_;
  }

  static constexpr size_t outQueueCap() {
    return TcpTransport::kMaxOutQueueBytes;
  }

  static constexpr size_t inflightCap() {
    return TcpTransport::kMaxInflightRequests;
  }

  /// Installs a connection whose send() always fails, so senderLoop() can be
  /// driven down its error path without a peer.
  void installFailingConn() {
    auto conn = std::make_unique<FailingConn>();
    failingConn_ = conn.get();
    transport_->dataConn_ = std::move(conn);
  }

  bool connClosed() const {
    return failingConn_ != nullptr && failingConn_->closed;
  }

  size_t unmatchedBytes() {
    std::lock_guard<std::mutex> lk(transport_->recvMu_);
    return transport_->unmatchedBytes_;
  }

  static constexpr size_t unmatchedCap() {
    return TcpTransport::kMaxUnmatchedSendBytes;
  }

  void runSenderLoop() {
    transport_->senderLoop();
  }

  bool enqueueFrame(std::vector<uint8_t> frame) {
    return transport_->enqueueFrame(std::move(frame), /*mayBlock=*/false);
  }

  void enqueueSendFrame(
      std::vector<uint8_t> frame,
      std::shared_ptr<TcpOpState> onSent) {
    transport_->enqueueSendFrame(std::move(frame), std::move(onSent));
  }

  size_t outQueueDepth() {
    std::lock_guard<std::mutex> lk(transport_->outMu_);
    return transport_->outQueue_.size();
  }

  /// send()/recv() are gated on the Connected state; these tests drive the
  /// staging path directly rather than standing up a real peer.
  void setConnected() {
    transport_->state_ = TransportState::Connected;
  }

  /// Buffers an inbound SEND payload so the next recv() takes its
  /// already-matched fast path instead of posting a pending recv.
  void bufferInboundSend(size_t len) {
    std::lock_guard<std::mutex> lk(transport_->recvMu_);
    transport_->unmatchedSends_.emplace_back(len, uint8_t{0xCD});
  }

  std::unique_ptr<ScopedEventBaseThread> evbThread_;
  std::shared_ptr<TcpSegmentRegistry> registry_;
  std::shared_ptr<::testing::NiceMock<MockCudaApi>> cudaApi_;
  std::unique_ptr<TcpTransport> transport_;
  std::vector<std::byte> segment_;
  std::vector<std::byte> pristine_;
  FailingConn* failingConn_{nullptr};
};

TEST_F(TcpTransportFrameTest, WriteToUnknownSegIdIsRejected) {
  feed(makeFrame(TcpOp::Write, kSegId + 1, /*offset=*/0, /*len=*/8, 8));

  EXPECT_TRUE(segmentUntouched()) << "an unknown segId must touch no buffer";
  EXPECT_TRUE(queuedErrorFrame());
}

TEST_F(TcpTransportFrameTest, WritePastSegmentEndIsRejected) {
  // offset is inside the segment but offset + len runs past the end.
  feed(makeFrame(TcpOp::Write, kSegId, /*offset=*/kSegLen - 4, /*len=*/8, 8));

  EXPECT_TRUE(segmentUntouched()) << "a write past the end must be rejected";
  EXPECT_TRUE(queuedErrorFrame());
}

TEST_F(TcpTransportFrameTest, WriteWithLenExceedingSegmentIsRejected) {
  feed(makeFrame(
      TcpOp::Write, kSegId, /*offset=*/0, /*len=*/kSegLen + 1, kSegLen + 1));

  EXPECT_TRUE(segmentUntouched());
  EXPECT_TRUE(queuedErrorFrame());
}

TEST_F(TcpTransportFrameTest, WriteWithPayloadLengthMismatchIsRejected) {
  // header.len says 32 but only 8 payload bytes arrived. Without the
  // payload.size() == header.len clause this would read past the frame.
  feed(makeFrame(TcpOp::Write, kSegId, /*offset=*/0, /*len=*/32, 8));

  EXPECT_TRUE(segmentUntouched())
      << "a header.len/payload.size() mismatch must be rejected";
  EXPECT_TRUE(queuedErrorFrame());
}

TEST_F(TcpTransportFrameTest, ReadRequestPastSegmentEndIsRejected) {
  feed(makeFrame(
      TcpOp::ReadRequest, kSegId, /*offset=*/kSegLen - 4, /*len=*/64, 0));

  EXPECT_TRUE(queuedErrorFrame())
      << "an out-of-range ReadRequest must not return segment contents";
}

TEST_F(TcpTransportFrameTest, InBoundsWriteIsApplied) {
  // Control: the checks must not reject a legitimate frame.
  feed(makeFrame(TcpOp::Write, kSegId, /*offset=*/0, /*len=*/8, 8));

  EXPECT_FALSE(segmentUntouched()) << "an in-bounds write must be applied";
  EXPECT_FALSE(queuedErrorFrame());
}

// A READ_REPLY copies its payload into the caller's get() destination after
// dropping inflightMu_. Resolving the op's future is what releases the caller
// to free that destination, so the copy and the resolution must be one atomic
// step. failAllPending() (send error, peer disconnect, shutdown) can resolve
// the op mid-copy via a *sibling* chunk of the same multi-chunk get(), which
// shares the op state -- a silent write-after-free.
TEST_F(TcpTransportFrameTest, ReadReplyCopyKeepsTheGetUnresolvedWhileItRuns) {
  constexpr size_t kLen = 64;
  std::vector<uint8_t> dst(kLen, 0);
  // Chunk 2 never replies, so failAllPending() has a sibling to fail.
  auto future = postReadChunks(dst.data(), kLen, /*chunks=*/2);

  // Park the H2D mid-copy so the fail is guaranteed to land while the copy into
  // dst is in flight.
  std::promise<void> copyStarted;
  std::promise<void> releaseCopy;
  auto copyStartedFuture = copyStarted.get_future();
  auto releaseCopyFuture = releaseCopy.get_future();
  EXPECT_CALL(
      *cudaApi_,
      memcpyAsync(
          ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
      .WillOnce([&](void* d, const void* s, size_t n, auto, auto) -> Status {
        copyStarted.set_value();
        releaseCopyFuture.wait();
        std::memcpy(d, s, n);
        return Ok();
      });

  std::thread reader([&]() {
    feed(makeFrame(TcpOp::ReadReply, kSegId, /*offset=*/0, kLen, kLen));
  });
  copyStartedFuture.wait();
  std::thread failer([&]() { failAllPending(); });

  // Generous: a resolution here would be immediate, so the full window is only
  // ever waited out when the op is correctly held open.
  const bool resolvedDuringCopy =
      future.wait_for(std::chrono::milliseconds(500)) ==
      std::future_status::ready;

  releaseCopy.set_value();
  reader.join();
  failer.join();

  EXPECT_FALSE(resolvedDuringCopy)
      << "the get() future resolved while a READ_REPLY copy into the caller's "
         "destination was still running; the caller may free it there";

  const std::vector<uint8_t> expected(kLen, 0xCD);
  EXPECT_EQ(dst, expected) << "the in-flight copy must still complete";
  EXPECT_TRUE(future.get().hasError())
      << "failAllPending() must still fail the op";
}

// put()/get() forward options.stream into their VRAM host-staging copies, but
// all four send()/recv() overloads used to discard RequestOptions, so the
// D2H/H2D ran on the null stream whatever the caller passed. That can transmit
// stale device data on send, and let the caller launch kernels against an
// unfilled buffer on recv, with no diagnostic. These tests pin both halves of
// the fix: the stream now reaches every staging site, and a VRAM transfer with
// no stream is refused rather than guessed at (matching RdmaTransport).

TEST_F(TcpTransportFrameTest, VramSendWithoutAnExplicitStreamIsRejected) {
  setConnected();
  std::vector<uint8_t> buf(64, 0xEE);
  Segment seg(buf.data(), buf.size(), MemoryType::VRAM, /*deviceId=*/0);
  // It must refuse before staging anything, not stage on the null stream.
  EXPECT_CALL(
      *cudaApi_,
      memcpyAsync(
          ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
      .Times(0);

  // Bounded: with the guard removed this queues a frame no sender drains, so an
  // unbounded get() would hang instead of failing.
  auto future = transport_->send(seg.span(size_t{0}, buf.size()));
  ASSERT_EQ(future.wait_for(std::chrono::seconds{5}), std::future_status::ready)
      << "a rejected VRAM transfer must fail immediately, not queue work";
  const auto status = future.get();

  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_F(TcpTransportFrameTest, VramRecvWithoutAnExplicitStreamIsRejected) {
  setConnected();
  std::vector<uint8_t> buf(64, 0);
  Segment seg(buf.data(), buf.size(), MemoryType::VRAM, /*deviceId=*/0);
  EXPECT_CALL(
      *cudaApi_,
      memcpyAsync(
          ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
      .Times(0);

  // Bounded: with the guard removed this queues a frame no sender drains, so an
  // unbounded get() would hang instead of failing.
  auto future = transport_->recv(seg.span(size_t{0}, buf.size()));
  ASSERT_EQ(future.wait_for(std::chrono::seconds{5}), std::future_status::ready)
      << "a rejected VRAM transfer must fail immediately, not queue work";
  const auto status = future.get();

  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_F(TcpTransportFrameTest, VramSendStagesOnTheCallersStream) {
  setConnected();
  std::vector<uint8_t> buf(64, 0xEE);
  Segment seg(buf.data(), buf.size(), MemoryType::VRAM, /*deviceId=*/0);
  auto* const callerStream = reinterpret_cast<void*>(0xF00D);
  RequestOptions options;
  options.stream = callerStream;

  void* staged = nullptr;
  EXPECT_CALL(
      *cudaApi_,
      memcpyAsync(
          ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
      .WillOnce([&](void*, const void*, size_t, auto, auto s) -> Status {
        staged = static_cast<void*>(s);
        return Ok();
      });

  (void)transport_->send(seg.span(size_t{0}, buf.size()), options);

  EXPECT_EQ(staged, callerStream) << "the D2H must run on the caller's stream";
}

TEST_F(
    TcpTransportFrameTest,
    VramRecvMatchedImmediatelyStagesOnTheCallersStream) {
  setConnected();
  constexpr size_t kLen = 64;
  bufferInboundSend(kLen);
  std::vector<uint8_t> buf(kLen, 0);
  Segment seg(buf.data(), kLen, MemoryType::VRAM, /*deviceId=*/0);
  auto* const callerStream = reinterpret_cast<void*>(0xBEEF);
  RequestOptions options;
  options.stream = callerStream;

  void* staged = nullptr;
  EXPECT_CALL(
      *cudaApi_,
      memcpyAsync(
          ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
      .WillOnce([&](void*, const void*, size_t, auto, auto s) -> Status {
        staged = static_cast<void*>(s);
        return Ok();
      });

  (void)transport_->recv(seg.span(size_t{0}, kLen), options);

  EXPECT_EQ(staged, callerStream);
}

// The case that needs TcpPendingRecv to carry the stream: a posted recv is
// completed by the reader thread when the SEND eventually arrives, long after
// the caller's RequestOptions is gone. Without the retained field this stages
// on the null stream.
TEST_F(TcpTransportFrameTest, PostedVramRecvRetainsTheStreamForALaterSend) {
  setConnected();
  constexpr size_t kLen = 64;
  std::vector<uint8_t> buf(kLen, 0);
  Segment seg(buf.data(), kLen, MemoryType::VRAM, /*deviceId=*/0);
  auto* const callerStream = reinterpret_cast<void*>(0xCAFE);
  RequestOptions options;
  options.stream = callerStream;

  // No buffered send, so this posts a pending recv and returns unresolved.
  auto future = transport_->recv(seg.span(size_t{0}, kLen), options);

  void* staged = nullptr;
  EXPECT_CALL(
      *cudaApi_,
      memcpyAsync(
          ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
      .WillOnce([&](void*, const void*, size_t, auto, auto s) -> Status {
        staged = static_cast<void*>(s);
        return Ok();
      });

  feed(makeFrame(TcpOp::Send, kSegId, /*offset=*/0, kLen, kLen));

  EXPECT_EQ(staged, callerStream)
      << "a posted recv must remember the stream it was given";
  EXPECT_FALSE(future.get().hasError());
}

// The admission race mahi flagged: a put()/get() that passes the entry check,
// then has failAllPending() sweep inflight_ before its own insert lands, used
// to leave an entry nothing would ever resolve -- enqueueFrame() drops the
// frame silently once the out queue is closed, and no later sweep runs, so the
// caller blocked in future.get() forever. This diff previously made that worse:
// the one-shot shutdown_ guard removed the accidental second failAllPending()
// that the MultiTransport-then-destructor double shutdown used to provide.
//
// Tested at the admission primitive rather than end-to-end: the window is
// between the entry check and the insert, and there is no hook in put()/get()
// to park a thread inside it, so an end-to-end test would either pass for the
// wrong reason (the entry check rejects first) or be a flaky thread race.
TEST_F(TcpTransportFrameTest, AdmissionIsRefusedOnceTeardownHasSwept) {
  setConnected();
  // Teardown has run: the sweep is done and the transport is marked broken.
  failAllPending();

  auto state = std::make_shared<TcpOpState>();
  state->remaining = 1;
  auto future = state->promise.get_future();

  const Status admitted = admitInflight(
      /*reqId=*/99, TcpInflight{state, nullptr, /*len=*/64, /*isRead=*/false});

  EXPECT_TRUE(admitted.hasError())
      << "admitting after the sweep orphans the entry and hangs the caller";
  EXPECT_EQ(inflightCount(), 0u) << "no orphaned inflight entry may remain";
  // The caller is expected to fail the op on refusal; nothing else can.
  EXPECT_NE(future.wait_for(std::chrono::seconds{0}), std::future_status::ready)
      << "admitInflight must not resolve the promise itself";
}

TEST_F(TcpTransportFrameTest, AdmissionSucceedsOnAHealthyTransport) {
  setConnected();

  auto state = std::make_shared<TcpOpState>();
  state->remaining = 1;

  EXPECT_FALSE(admitInflight(
                   /*reqId=*/7,
                   TcpInflight{state, nullptr, /*len=*/64, /*isRead=*/false})
                   .hasError());
  EXPECT_EQ(inflightCount(), 1u);
}

// A dead sender used to leave the out queue open: enqueueFrame() gates only on
// outClosed_, so the still-running reader thread kept appending Ack/Error/
// ReadReply frames (up to 4 MiB each) to a queue nothing drained, and
// failAllPending() clears it exactly once. Unbounded, peer-driven growth.
TEST_F(TcpTransportFrameTest, DeadSenderClosesTheOutQueue) {
  setConnected();
  installFailingConn();

  // One item for the sender to pick up and fail on.
  auto state = std::make_shared<TcpOpState>();
  state->remaining = 1;
  auto future = state->promise.get_future();
  enqueueSendFrame(std::vector<uint8_t>(16, 0), state);
  ASSERT_EQ(outQueueDepth(), 1u);

  runSenderLoop(); // returns once send() fails

  EXPECT_TRUE(future.get().hasError()) << "the in-flight op must be failed";
  // The reader thread would keep producing replies after the sender is gone.
  enqueueFrame(std::vector<uint8_t>(4096, 0));
  enqueueFrame(std::vector<uint8_t>(4096, 0));
  EXPECT_EQ(outQueueDepth(), 0u)
      << "a dead sender must close the out queue, not let it grow unbounded";
}

// A ReadRequest whose reply would exceed the wire-frame cap must be refused per
// request. The controller refuses an oversized send and senderLoop treats that
// as fatal, so without this bound one oversized request from a version-skewed
// peer (built with a larger chunk size) kills the sender and fails every
// unrelated in-flight transfer. The segment here is deliberately large enough
// that the offset/len bounds check passes, so only the cap can reject it.
TEST_F(TcpTransportFrameTest, OversizedReadRequestIsRefusedPerRequest) {
  constexpr uint64_t kBigSegId = kSegId + 100;
  const size_t kBigLen = kMaxFrameSize;
  std::vector<std::byte> big(kBigLen, std::byte{0x11});
  registry_->add(
      kBigSegId, big.data(), kBigLen, MemoryType::DRAM, /*deviceId=*/-1);

  feed(makeFrame(
      TcpOp::ReadRequest,
      kBigSegId,
      /*offset=*/0,
      kBigLen,
      /*payloadBytes=*/0));

  EXPECT_TRUE(queuedErrorFrame())
      << "an oversized read must get a per-request Error rather than a "
         "ReadReply the controller will refuse and the sender will treat as "
         "fatal";
}

// unmatchedSends_ buffers every inbound Send that finds no posted recv. The
// reader never stops draining the socket -- that is what avoids the mutual-READ
// deadlock -- so nothing upstream slows the peer down and the deque would
// otherwise grow without limit.
TEST_F(TcpTransportFrameTest, UnmatchedSendsOverflowRefusesTheConnection) {
  setConnected();
  installFailingConn();

  constexpr size_t kChunk = 4UL * 1024 * 1024;
  const size_t maxFrames = (unmatchedCap() / kChunk) + 4;
  size_t frames = 0;
  while (!connClosed() && frames < maxFrames) {
    feed(makeFrame(TcpOp::Send, kSegId, /*offset=*/0, kChunk, kChunk));
    ++frames;
  }

  EXPECT_TRUE(connClosed())
      << "unmatched sends past the cap must refuse the connection instead of "
         "buffering without bound";
  EXPECT_LE(unmatchedBytes(), unmatchedCap());
}

// A get of N bytes arrives at the responder as N/kMaxChunkSize ReadRequests and
// legitimately needs up to N bytes of ReadReply queued, with N bounded only by
// the segment size. Capping the reader path therefore rejects honest traffic:
// applying kMaxOutQueueBytes here made every get >= 128 MiB kill the
// connection, failing all unrelated in-flight transfers with it. The reader
// must accept past the cap -- it also cannot wait, since ceasing to drain the
// socket is the mutual-READ deadlock the reader/sender split exists to avoid.
TEST_F(
    TcpTransportFrameTest,
    ReaderRepliesAreNotCappedAndDoNotKillTheConnection) {
  setConnected();
  installFailingConn();

  // Mimic serving a 128 MiB get: 32 x 4 MiB replies, i.e. 2x the queue cap.
  constexpr size_t kChunk = 4UL * 1024 * 1024;
  const size_t frames = (2 * outQueueCap()) / kChunk;
  for (size_t i = 0; i < frames; ++i) {
    ASSERT_TRUE(enqueueReaderFrame(kChunk))
        << "the reader path must keep accepting replies past the queue cap, "
        << "refused at frame " << i;
  }

  EXPECT_GT(outQueueBytes(), outQueueCap())
      << "the reply path is deliberately uncapped";
  EXPECT_FALSE(connClosed())
      << "serving a large get must not refuse the connection";
}

TEST_F(TcpTransportFrameTest, InflightAdmissionIsCappedNotUnbounded) {
  setConnected();
  auto state = std::make_shared<TcpOpState>();
  state->remaining = 1;

  for (size_t i = 0; i < inflightCap(); ++i) {
    ASSERT_FALSE(
        admitInflight(
            i + 1, TcpInflight{state, nullptr, /*len=*/8, /*isRead=*/false})
            .hasError())
        << "admission below the cap must succeed, failed at " << i;
  }
  ASSERT_EQ(inflightCount(), inflightCap());

  const Status overflow = admitInflight(
      inflightCap() + 1, TcpInflight{state, nullptr, /*len=*/8, false});

  ASSERT_TRUE(overflow.hasError()) << "inflight_ must not grow past its cap";
  EXPECT_EQ(overflow.error().code(), ErrCode::ResourceExhausted);
  EXPECT_EQ(inflightCount(), inflightCap());
}

// The registry's mutex protects the map, not the lifetime of the application
// buffer an entry points at. Nothing stopped the reader thread from copying an
// entry out, the owner deregistering and freeing the buffer, and the reader
// then writing into freed memory. erase() now drains outstanding leases first.
//
// These exercise the registry directly rather than through handleFrame(): the
// race needs the reader parked between the lookup and the copy, and there is no
// hook to park it there, so an end-to-end version would be a flaky thread race
// that passes for the wrong reason most runs.
class TcpSegmentRegistryTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kSegId = 7;

  void SetUp() override {
    buf_.assign(64, std::byte{0});
    registry_.add(kSegId, buf_.data(), buf_.size(), MemoryType::DRAM, -1);
  }

  TcpSegmentRegistry registry_;
  std::vector<std::byte> buf_;
};

TEST_F(TcpSegmentRegistryTest, EraseWaitsForAnOutstandingLease) {
  auto lease = registry_.find(kSegId);
  ASSERT_TRUE(lease) << "a registered segment must yield a lease";
  EXPECT_EQ(lease->ptr, buf_.data());
  EXPECT_EQ(lease->len, buf_.size());

  std::atomic<bool> erased{false};
  std::thread eraser([this, &erased]() {
    registry_.erase(kSegId);
    erased.store(true);
  });

  // Deregistration must not complete while a reader could still be mid-copy.
  // Without the drain this flips to true almost immediately.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(erased.load())
      << "erase() returned while a lease was outstanding: the owner is now free "
         "to release memory the reader may still write into";

  lease.reset();
  eraser
      .join(); // Hangs here if releasing the last lease fails to wake erase().
  EXPECT_TRUE(erased.load());
}

TEST_F(TcpSegmentRegistryTest, DrainingSegmentHandsOutNoNewLeases) {
  auto lease = registry_.find(kSegId);
  ASSERT_TRUE(lease);

  std::atomic<bool> erased{false};
  std::thread eraser([this, &erased]() {
    registry_.erase(kSegId);
    erased.store(true);
  });

  // Let erase() mark the segment dying and start waiting. If find() kept
  // issuing leases here, a steady stream of inbound frames for this segment
  // could hold the count above zero and starve the drain indefinitely.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  // Both halves matter: the segment must still be mid-drain (not already gone)
  // AND refusing leases. Checking only the refusal would also pass against an
  // erase() that removed the entry outright, which is the bug.
  ASSERT_FALSE(erased.load()) << "erase() completed before the drain finished";
  EXPECT_FALSE(registry_.find(kSegId))
      << "a segment being torn down must not yield further leases";

  lease.reset();
  eraser.join();
  EXPECT_FALSE(registry_.find(kSegId));
}

TEST_F(TcpSegmentRegistryTest, FindNeverBlocksBehindADrain) {
  // The reader thread calls find() and must never be parked behind a
  // deregistration: it is the thread that releases leases, and blocking it
  // behind an app thread waiting on it would deadlock both.
  auto lease = registry_.find(kSegId);
  ASSERT_TRUE(lease);
  std::thread eraser([this]() { registry_.erase(kSegId); });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto refused = std::async(std::launch::async, [this]() {
    return static_cast<bool>(registry_.find(kSegId));
  });
  ASSERT_EQ(
      refused.wait_for(std::chrono::seconds(5)), std::future_status::ready)
      << "find() blocked while a drain was in progress";
  EXPECT_FALSE(refused.get());

  lease.reset();
  eraser.join();
}

TEST_F(TcpSegmentRegistryTest, EraseWithNoLeasesOrUnknownSegmentReturns) {
  registry_.erase(kSegId); // No leases outstanding: must not wait.
  EXPECT_FALSE(registry_.find(kSegId));
  registry_.erase(kSegId); // Already gone.
  registry_.erase(kSegId + 1000); // Never registered.
}

} // namespace uniflow
