// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/tcp/TcpTransport.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "comms/uniflow/Segment.h"
#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/tcp/TcpRegistrationHandle.h"
#include "comms/uniflow/transport/tcp/TcpWireProtocol.h"

namespace uniflow {

/// Builds RegisteredSegment / RemoteRegisteredSegment without a factory,
/// through the `friend class SegmentTest` seam in Segment.h. The name has to be
/// exactly this to match that declaration.
class SegmentTest {
 public:
  static RegisteredSegment
  makeRegistered(void* buf, size_t len, MemoryType memType, int deviceId) {
    return RegisteredSegment(buf, len, memType, deviceId);
  }

  static RemoteRegisteredSegment makeRemote(
      void* buf,
      size_t len,
      std::unique_ptr<RemoteRegistrationHandle> handle) {
    RemoteRegisteredSegment remote(buf, len);
    remote.handles_.push_back(std::move(handle));
    return remote;
  }
};

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
    ++closeCount;
  }
  bool closed{false};
  int closeCount{0};
};

/// A connection that accepts every send and counts them, for the tests that
/// need the sender to actually drain the queue.
class CountingConn : public controller::Conn {
 public:
  std::future<Result<size_t>> send(std::span<const uint8_t> data) override {
    sendCount_.fetch_add(1, std::memory_order_release);
    return make_ready_future<Result<size_t>>(Result<size_t>(data.size()));
  }
  std::future<Result<size_t>> recv(std::vector<uint8_t>&) override {
    return make_ready_future<Result<size_t>>(
        Err(ErrCode::ConnectionFailed, "test: recv failed"));
  }
  std::future<Result<size_t>> recv(std::span<uint8_t>) override {
    return make_ready_future<Result<size_t>>(
        Err(ErrCode::ConnectionFailed, "test: recv failed"));
  }
  void close() override {}

  int sendCount() const {
    return sendCount_.load(std::memory_order_acquire);
  }

 private:
  std::atomic<int> sendCount_{0};
};

/// A connection whose send() parks until the test lets it through, so the
/// window in which the sender thread is mid-send -- and must still own the
/// frame's storage -- can be inspected.
class GatedConn : public controller::Conn {
 public:
  std::future<Result<size_t>> send(std::span<const uint8_t> data) override {
    std::unique_lock<std::mutex> lk(mu_);
    sentPointer_ = data.data();
    sentSize_ = data.size();
    ++sendCount_;
    entered_.notify_all();
    // Parked inside send() rather than by returning an unfulfilled future: the
    // sender thread calls .get() on the result either way, so this is the same
    // wait a real socket write imposes.
    gate_.wait(lk, [this]() { return released_; });
    return make_ready_future<Result<size_t>>(Result<size_t>(sentSize_));
  }
  std::future<Result<size_t>> recv(std::vector<uint8_t>&) override {
    return make_ready_future<Result<size_t>>(
        Err(ErrCode::ConnectionFailed, "test: recv failed"));
  }
  std::future<Result<size_t>> recv(std::span<uint8_t>) override {
    return make_ready_future<Result<size_t>>(
        Err(ErrCode::ConnectionFailed, "test: recv failed"));
  }
  void close() override {}

  void waitForSend() {
    std::unique_lock<std::mutex> lk(mu_);
    entered_.wait(lk, [this]() { return sendCount_ > 0; });
  }

  void releaseSend() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      released_ = true;
    }
    gate_.notify_all();
  }

  const uint8_t* sentPointer() {
    std::lock_guard<std::mutex> lk(mu_);
    return sentPointer_;
  }

  size_t sentSize() {
    std::lock_guard<std::mutex> lk(mu_);
    return sentSize_;
  }

 private:
  std::mutex mu_;
  std::condition_variable entered_;
  std::condition_variable gate_;
  const uint8_t* sentPointer_{nullptr};
  size_t sentSize_{0};
  int sendCount_{0};
  bool released_{false};
};

/// A VRAM put of `len` bytes, held together so the segments outlive the request
/// that borrows them. The source is never read -- memcpyAsync is mocked -- but
/// it has to be a real allocation of the full length, because put() chunks by
/// size.
class VramPut {
 public:
  explicit VramPut(size_t len)
      : src_(len),
        local_(
            SegmentTest::makeRegistered(
                src_.data(),
                len,
                MemoryType::VRAM,
                /*deviceId=*/0)),
        remote_(
            SegmentTest::makeRemote(
                // The peer's address is never dereferenced here: TCP sends the
                // segId and offset and lets the peer resolve them.
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                reinterpret_cast<void*>(0x100000),
                len,
                std::make_unique<TcpRemoteRegistrationHandle>(
                    kRemoteSegId,
                    len))),
        request_(TransferRequest{local_.span(), remote_.span()}) {}

  std::span<const TransferRequest> requests() const {
    return {&request_, 1};
  }

 private:
  static constexpr uint64_t kRemoteSegId = 7;
  std::vector<uint8_t> src_;
  RegisteredSegment local_;
  RemoteRegisteredSegment remote_;
  TransferRequest request_;
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
        .WillByDefault([this](auto) -> Status {
          syncCount_.fetch_add(1, std::memory_order_release);
          std::unique_lock<std::mutex> lk(syncMu_);
          syncReleased_.wait(lk, [this]() { return !syncsGated_; });
          return Ok();
        });
    // Real host memory behind the staging pool: the pool hands out pointers the
    // transport builds frames in and copies into, and the tests check that a
    // copy landed inside this memory rather than in a vector.
    ON_CALL(*cudaApi_, hostAlloc(::testing::_, ::testing::_))
        .WillByDefault([this](size_t size, unsigned int) -> Result<void*> {
          // Default-initialised rather than zeroed: this is 64 MiB per pool.
          auto buf = std::unique_ptr<uint8_t[]>(new uint8_t[size]);
          void* ptr = buf.get();
          std::lock_guard<std::mutex> lk(allocMu_);
          hostAllocs_.emplace_back(std::move(buf), size);
          return ptr;
        });
    ON_CALL(*cudaApi_, hostFree(::testing::_))
        .WillByDefault([this](void* ptr) -> Status {
          std::lock_guard<std::mutex> lk(allocMu_);
          std::erase_if(hostAllocs_, [ptr](const auto& alloc) {
            return alloc.first.get() == ptr;
          });
          return Ok();
        });

    transport_ = std::make_unique<TcpTransport>(
        /*deviceId=*/-1,
        evbThread_->getEventBase(),
        registry_,
        controller::TcpSocketConfig{},
        /*host=*/"127.0.0.1",
        cudaApi_);

    // These tests drive the outbound path without connecting a peer, so give
    // the transport the single lane that path indexes.
    lane0();

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

  void feed(std::span<const uint8_t> frame, TcpPinnedSlab slab) {
    transport_->handleFrame(frame, std::move(slab));
  }

  std::shared_ptr<TcpPinnedSlabPool> receivePool() {
    return transport_->ensureReceivePool();
  }

  /// True if the transport queued at least one Error frame back to the peer.
  bool queuedErrorFrame() {
    std::lock_guard<std::mutex> lk(lane0().mu);
    for (const auto& item : lane0().queue) {
      auto header = deserializeTcpHeader(item.frame.bytes());
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

  bool connBroken() const {
    return transport_->connBroken_.load(std::memory_order_acquire);
  }

  bool readerRunning() const {
    return transport_->running_.load(std::memory_order_acquire);
  }

  /// Registers `chunks` in-flight READ chunks sharing one op state, the way a
  /// multi-chunk get() does. Chunk reqId 1 targets `dst`; the rest stand in for
  /// siblings whose replies have not arrived yet. VRAM so the copy runs through
  /// the mocked CudaApi. Returns the caller's future.
  std::future<Status>
  postReadChunks(void* dst, size_t len, size_t chunks, void* stream = nullptr) {
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
          stream};
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

  Status admitInflightBulk(std::span<PlannedChunk> chunks) {
    return transport_->admitInflightBulk(chunks);
  }

  void abandonInflight(std::span<const PlannedChunk> chunks, size_t fromIdx) {
    transport_->abandonInflight(chunks, fromIdx);
  }

  /// Occupies `n` admission slots so the cap can be reached without moving the
  /// gigabytes of payload a real put() would need to get there.
  void fillInflight(size_t n) {
    auto state = std::make_shared<TcpOpState>();
    for (size_t i = 0; i < n; ++i) {
      ASSERT_FALSE(transport_
                       ->admitInflight(
                           kFillReqIdBase + i,
                           TcpInflight{state, nullptr, /*len=*/1, false})
                       .hasError());
    }
  }

  static std::vector<PlannedChunk> makeChunks(
      const std::shared_ptr<TcpOpState>& state,
      size_t count,
      uint64_t baseReqId) {
    std::vector<PlannedChunk> chunks;
    chunks.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      chunks.push_back(
          PlannedChunk{
              baseReqId + i,
              /*len=*/64,
              TcpInflight{state, nullptr, /*len=*/64, false}});
    }
    return chunks;
  }

  bool enqueueReaderFrame(size_t bytes) {
    return transport_->enqueueFrame(
        std::vector<uint8_t>(bytes, 0), /*mayBlock=*/false);
  }

  size_t outQueueBytes() {
    std::lock_guard<std::mutex> lk(lane0().mu);
    return lane0().bytes;
  }

  static constexpr size_t outQueueCap() {
    return TcpTransport::kMaxOutQueueBytes;
  }

  static constexpr size_t inflightCap() {
    return TcpTransport::kMaxInflightRequests;
  }

  // Far from the reqIds the tests pick by hand, so filler cannot collide.
  static constexpr uint64_t kFillReqIdBase = 1'000'000;

  /// Lane 0, created on demand. These tests all run single-lane, so lane 0 is
  /// the entire outbound queue and its cap equals kMaxOutQueueBytes.
  TcpTransport::TcpLane& lane0() {
    if (transport_->lanes_.empty()) {
      transport_->lanes_.push_back(std::make_unique<TcpTransport::TcpLane>());
    }
    return *transport_->lanes_[0];
  }

  /// Installs `conn` as lane 0, creating the lane when the transport was never
  /// connected.
  void installLaneConn(std::unique_ptr<controller::Conn> conn) {
    lane0().conn = std::move(conn);
  }

  /// Installs a connection whose send() always fails, so senderLoop() can be
  /// driven down its error path without a peer.
  void installFailingConn() {
    auto conn = std::make_unique<FailingConn>();
    failingConn_ = conn.get();
    installLaneConn(std::move(conn));
  }

  bool connClosed() const {
    return failingConn_ != nullptr && failingConn_->closed;
  }

  int connCloseCount() const {
    return failingConn_ == nullptr ? 0 : failingConn_->closeCount;
  }

  size_t unmatchedBytes() {
    std::lock_guard<std::mutex> lk(transport_->recvMu_);
    return transport_->unmatchedBytes_;
  }

  static constexpr size_t unmatchedCap() {
    return TcpTransport::kMaxUnmatchedSendBytes;
  }

  void runSenderLoop() {
    transport_->senderLoop(0);
  }

  /// Installs a connection whose send() parks until released, so the sender
  /// thread can be caught mid-send.
  GatedConn* installGatedConn() {
    auto conn = std::make_unique<GatedConn>();
    auto* raw = conn.get();
    installLaneConn(std::move(conn));
    return raw;
  }

  /// Installs a connection that accepts every send, for the tests that need the
  /// queue actually drained.
  CountingConn& installCountingConn() {
    auto conn = std::make_unique<CountingConn>();
    auto& ref = *conn;
    installLaneConn(std::move(conn));
    return ref;
  }

  static size_t deferredCap() {
    return TcpTransport::kMaxInflightRequests;
  }

  static size_t slabPayloadCap() {
    return TcpTransport::kMaxChunkSize;
  }

  /// Fills the deferred queue with entries holding no lease, so the overflow
  /// boundary can be reached without the millions of requests a real peer would
  /// have to send to get there.
  void fillDeferredReplies(size_t n) {
    std::lock_guard<std::mutex> lk(transport_->stagingMu_);
    for (size_t i = 0; i < n; ++i) {
      transport_->deferredReplies_.push_back(DeferredReadReply{});
    }
  }

  /// Ends senderLoop() the way shutdown() does, without tearing the rest of the
  /// transport down.
  void closeOutQueue() {
    {
      std::lock_guard<std::mutex> lk(lane0().mu);
      lane0().outClosed = true;
    }
    lane0().cv.notify_all();
  }

  /// A standalone pool, for the frame-ownership tests. Backed by the fixture's
  /// hostAlloc stub, so its slabs are real memory.
  std::shared_ptr<TcpPinnedSlabPool> makeSlabPool(
      size_t slabCount,
      size_t reserved) {
    auto pool =
        TcpPinnedSlabPool::create(cudaApi_, kTestSlabSize, slabCount, reserved);
    EXPECT_TRUE(pool.hasValue());
    return pool.hasValue() ? pool.value() : nullptr;
  }

  /// The transport's own staging pool, created on the first call exactly as a
  /// VRAM read would create it.
  std::shared_ptr<TcpPinnedSlabPool> transportStagingPool() {
    auto pool = transport_->stagingPool();
    EXPECT_TRUE(pool.hasValue());
    return pool.hasValue() ? pool.value() : nullptr;
  }

  /// Installs the stubs a staged VRAM read needs, with the copy held unfinished
  /// until `releaseStagingCopies()`. Every copy destination is recorded, which
  /// is how a test tells staging into the pinned pool from staging into a
  /// vector.
  void stubStagingCopies() {
    ON_CALL(
        *cudaApi_,
        memcpyAsync(
            ::testing::_,
            ::testing::_,
            ::testing::_,
            ::testing::_,
            ::testing::_))
        .WillByDefault(
            [this](void* dst, const void*, size_t, auto, auto) -> Status {
              std::lock_guard<std::mutex> lk(copyMu_);
              copyDsts_.push_back(dst);
              if (failCopyAt_ != 0 && copyDsts_.size() == failCopyAt_) {
                return Err(ErrCode::DriverError, "test: staging copy failed");
              }
              return Ok();
            });
    ON_CALL(*cudaApi_, eventCreate(::testing::_))
        .WillByDefault([](auto* event) -> Status {
          // Spelled with auto because this TU is not hipified: cudaEvent_t is
          // ihipEvent_t* here, and naming it does not compile under ROCm.
          *event = {};
          return Ok();
        });
    ON_CALL(*cudaApi_, eventRecord(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(Ok()));
    ON_CALL(*cudaApi_, eventDestroy(::testing::_))
        .WillByDefault(::testing::Return(Ok()));
    ON_CALL(*cudaApi_, eventQuery(::testing::_))
        .WillByDefault([this](auto) -> Result<bool> {
          return Result<bool>(copyDone_.load(std::memory_order_acquire));
        });
  }

  void releaseStagingCopies() {
    copyDone_.store(true, std::memory_order_release);
  }

  /// Fails the copy issued after `succeeding` others, so a wave can fail with
  /// copies already in flight -- the case where releasing slabs early would let
  /// a later copy write into a buffer the GPU is still filling.
  void failCopyAfter(size_t succeeding) {
    std::lock_guard<std::mutex> lk(copyMu_);
    failCopyAt_ = succeeding + 1;
  }

  /// Holds every staging wait open, so the window between "all copies issued"
  /// and "all copies finished" can be inspected. Without it that window closes
  /// too fast to assert anything about.
  void gateStagingWaits() {
    std::lock_guard<std::mutex> lk(syncMu_);
    syncsGated_ = true;
  }

  void releaseStagingWaits() {
    {
      std::lock_guard<std::mutex> lk(syncMu_);
      syncsGated_ = false;
    }
    syncReleased_.notify_all();
  }

  size_t stagedCopyCount() {
    std::lock_guard<std::mutex> lk(copyMu_);
    return copyDsts_.size();
  }

  /// True if every recorded copy destination lies inside memory the pool
  /// allocated. A copy into a frame's vector storage fails this.
  bool allCopiesLandedInPinnedMemory() {
    std::lock_guard<std::mutex> lk(copyMu_);
    std::lock_guard<std::mutex> allocLk(allocMu_);
    for (const auto* dst : copyDsts_) {
      const bool inside = std::any_of(
          hostAllocs_.begin(), hostAllocs_.end(), [dst](const auto& alloc) {
            const auto* base = alloc.first.get();
            if (base == nullptr) {
              return false;
            }
            return dst >= base && dst < base + alloc.second;
          });
      if (!inside) {
        return false;
      }
    }
    return true;
  }

  int syncCount() const {
    return syncCount_.load(std::memory_order_acquire);
  }

  size_t deferredReplyCount() {
    std::lock_guard<std::mutex> lk(transport_->stagingMu_);
    return transport_->deferredReplies_.size();
  }

  std::future<Status> put(std::span<const TransferRequest> requests) {
    return transport_->put(requests, RequestOptions{});
  }

  static constexpr size_t waveCap() {
    return TcpTransport::kMaxPutWaveChunks;
  }

  /// How many slabs the pool will still hand out. Drains with tryAcquire rather
  /// than a bulk acquire so a test asserting "nothing was stranded" reports the
  /// shortfall instead of waiting for slabs that are never coming back.
  static size_t freeSlabs(TcpPinnedSlabPool& pool) {
    std::vector<TcpPinnedSlab> held;
    while (auto slab = pool.tryAcquire(/*allowReserved=*/true)) {
      held.push_back(std::move(slab));
    }
    return held.size();
  }

  /// Builds a header-only frame in `slab`, so a queued frame can be traced back
  /// to the slab it was staged in.
  static TcpFrame slabFrame(TcpPinnedSlab slab, TcpOp op, uint64_t reqId) {
    TcpMsgHeader header;
    header.op = static_cast<uint8_t>(op);
    header.reqId = reqId;
    auto bytes = serializeTcpHeader(header);
    uint8_t* dst = slab.data();
    if (dst == nullptr) {
      return TcpFrame{};
    }
    std::copy(bytes.begin(), bytes.end(), dst);
    return TcpFrame{std::move(slab), bytes.size()};
  }

  bool enqueueFrames(std::vector<TcpFrame> frames, bool mayBlock) {
    return transport_->enqueueFrames(std::move(frames), mayBlock);
  }

  bool enqueueStagedFrame(TcpFrame frame) {
    return transport_->enqueueFrame(std::move(frame), /*mayBlock=*/false);
  }

  /// reqIds of the frames currently queued, in queue order.
  std::vector<uint64_t> queuedReqIds() {
    std::vector<uint64_t> ids;
    std::lock_guard<std::mutex> lk(lane0().mu);
    for (const auto& item : lane0().queue) {
      auto header = deserializeTcpHeader(item.frame.bytes());
      ids.push_back(header.hasValue() ? header.value().reqId : 0);
    }
    return ids;
  }

  /// Spins until `predicate` holds or the deadline passes. Used where the state
  /// under test is changed by the sender thread and there is nothing to signal
  /// on -- a slab returning to its pool has no observer hook.
  template <typename Fn>
  static bool waitFor(Fn predicate) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      if (predicate()) {
        return true;
      }
      std::this_thread::yield();
    }
    return predicate();
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
    std::lock_guard<std::mutex> lk(lane0().mu);
    return lane0().queue.size();
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
  static constexpr size_t kTestSlabSize = 256;
  std::atomic<bool> copyDone_{false};
  std::atomic<int> syncCount_{0};
  std::mutex copyMu_;
  std::vector<const void*> copyDsts_;
  std::mutex allocMu_;
  std::vector<std::pair<std::unique_ptr<uint8_t[]>, size_t>> hostAllocs_;
  std::mutex syncMu_;
  std::condition_variable syncReleased_;
  bool syncsGated_{false};
  size_t failCopyAt_{0};
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

TEST(TcpOpStateTest, FailureWaitsForReservedWriteToRetire) {
  TcpOpState state;
  state.remaining = 1;
  auto future = state.promise.get_future();

  ASSERT_TRUE(state.tryBeginWrite());
  state.fail(Err(ErrCode::ConnectionFailed, "test failure"));
  EXPECT_EQ(
      future.wait_for(std::chrono::milliseconds(0)),
      std::future_status::timeout);

  state.endWrite(Ok());
  ASSERT_EQ(
      future.wait_for(std::chrono::milliseconds(0)), std::future_status::ready);
  const Status status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::ConnectionFailed);
}

TEST(TcpOpStateTest, SuccessfulReservedWritesCompleteTheOperation) {
  TcpOpState state;
  state.remaining = 2;
  auto future = state.promise.get_future();

  ASSERT_TRUE(state.tryBeginWrite());
  ASSERT_TRUE(state.tryBeginWrite());
  state.endWrite(Ok());
  EXPECT_EQ(
      future.wait_for(std::chrono::milliseconds(0)),
      std::future_status::timeout);

  state.endWrite(Ok());
  EXPECT_FALSE(future.get().hasError());
  EXPECT_FALSE(state.tryBeginWrite());
}

TEST(TcpOpStateTest, ReservedWriteFailureWinsAndResolvesExactlyOnce) {
  TcpOpState state;
  state.remaining = 2;
  auto future = state.promise.get_future();

  ASSERT_TRUE(state.tryBeginWrite());
  ASSERT_TRUE(state.tryBeginWrite());
  state.endWrite(Err(ErrCode::DriverError, "copy failed"));
  EXPECT_EQ(
      future.wait_for(std::chrono::milliseconds(0)),
      std::future_status::timeout);

  state.fail(Err(ErrCode::ConnectionFailed, "later failure"));
  state.endWrite(Ok());
  const Status status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::DriverError);
}

TEST(TcpOpStateTest, WriteReservationIsRefusedAfterFailure) {
  TcpOpState state;
  state.remaining = 1;
  auto future = state.promise.get_future();

  state.fail(Err(ErrCode::ConnectionFailed, "test failure"));
  EXPECT_FALSE(state.tryBeginWrite());
  EXPECT_TRUE(future.get().hasError());
}

TEST(TcpOpStateTest, ThrowingSynchronousWriteRetiresItsReservation) {
  TcpOpState state;
  state.remaining = 1;
  auto future = state.promise.get_future();

  EXPECT_THROW(
      state.writeAndComplete(
          []() -> Status { throw std::runtime_error("test exception"); }),
      std::runtime_error);

  ASSERT_EQ(
      future.wait_for(std::chrono::milliseconds(0)), std::future_status::ready);
  const Status status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::TransportError);
  EXPECT_FALSE(state.tryBeginWrite());
}

TEST_F(TcpTransportFrameTest, AsyncReadReplyFailureWaitsForEventRetirement) {
  constexpr size_t kLen = 64;
  std::vector<uint8_t> dst(kLen, 0);
  auto* const callerStream = reinterpret_cast<void*>(0xCA11);
  // Chunk 2 remains in inflight_ so failAllPending() can latch an error while
  // chunk 1's asynchronous destination write is still reserved.
  auto future = postReadChunks(dst.data(), kLen, /*chunks=*/2, callerStream);

  auto pool = receivePool();
  ASSERT_NE(pool, nullptr);
  auto slab = pool->tryAcquire(/*allowReserved=*/true);
  ASSERT_TRUE(slab);
  auto frame = makeFrame(TcpOp::ReadReply, kSegId, /*offset=*/0, kLen, kLen);
  ASSERT_LE(frame.size(), slab.capacity());
  std::memcpy(slab.data(), frame.data(), frame.size());
  const auto bytes = std::span<const uint8_t>{slab.data(), frame.size()};

  std::atomic<bool> copyDone{false};
  EXPECT_CALL(
      *cudaApi_,
      memcpyAsync(
          dst.data(),
          ::testing::_,
          kLen,
          kMockMemcpyH2D,
          static_cast<MockStream>(callerStream)))
      .WillOnce([&](void* d, const void* s, size_t n, auto, auto) -> Status {
        std::memcpy(d, s, n);
        return Ok();
      });
  EXPECT_CALL(*cudaApi_, eventCreate(::testing::_))
      .WillOnce([](auto* event) -> Status {
        *event = {};
        return Ok();
      });
  EXPECT_CALL(
      *cudaApi_,
      eventRecord(::testing::_, static_cast<MockStream>(callerStream)))
      .WillOnce(::testing::Return(Ok()));
  EXPECT_CALL(*cudaApi_, eventQuery(::testing::_))
      .WillRepeatedly([&](auto) -> Result<bool> {
        return copyDone.load(std::memory_order_acquire);
      });
  EXPECT_CALL(*cudaApi_, eventDestroy(::testing::_))
      .WillOnce(::testing::Return(Ok()));
  EXPECT_CALL(*cudaApi_, streamSynchronize(::testing::_)).Times(0);

  feed(bytes, std::move(slab));
  auto onlyFreeSlab = pool->tryAcquire(/*allowReserved=*/true);
  ASSERT_TRUE(onlyFreeSlab);
  failAllPending();

  EXPECT_EQ(
      future.wait_for(std::chrono::milliseconds(100)),
      std::future_status::timeout)
      << "the failure must remain latched until the async destination write "
         "retires";
  EXPECT_FALSE(pool->tryAcquire(/*allowReserved=*/true));

  copyDone.store(true, std::memory_order_release);
  ASSERT_EQ(
      future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_TRUE(future.get().hasError());
  EXPECT_EQ(dst, std::vector<uint8_t>(kLen, uint8_t{0xCD}));
}

// A vector-backed READ_REPLY still copies synchronously. Resolving the op's
// future is what releases the caller to free its destination, so that fallback
// copy and completion remain one atomic lifetime reservation.
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

// put() reserves a slot per chunk. Reserving them one at a time inside the send
// loop means a put larger than the cap allows -- above ~16 GiB at 4 MiB chunks
// and 4096 slots -- delivers chunks until the slots run out and only then
// fails, so the caller is told the whole put failed while most of it landed in
// the peer's segment with nothing to tell the peer about it. A put that cannot
// fit must be refused before any chunk is staged.
TEST_F(TcpTransportFrameTest, BulkAdmissionIsAllOrNothingAtTheCap) {
  setConnected();
  auto state = std::make_shared<TcpOpState>();

  constexpr size_t kHeadroom = 3;
  fillInflight(inflightCap() - kHeadroom);
  ASSERT_EQ(inflightCount(), inflightCap() - kHeadroom);

  auto tooMany = makeChunks(state, kHeadroom + 1, /*baseReqId=*/9000);
  EXPECT_TRUE(admitInflightBulk(tooMany).hasError())
      << "a put that does not fit must be refused whole";
  EXPECT_EQ(inflightCount(), inflightCap() - kHeadroom)
      << "a refused put must leave no partial reservation, because every slot it "
         "did take is a chunk already on its way to the peer";

  auto exactFit = makeChunks(state, kHeadroom, /*baseReqId=*/9100);
  EXPECT_FALSE(admitInflightBulk(exactFit).hasError())
      << "a put that exactly fills the remaining slots must be admitted";
  EXPECT_EQ(inflightCount(), inflightCap());
}

// When staging or the out queue fails partway through the send loop, the chunks
// already handed to enqueueFrame keep their reservations so their Acks still
// match an entry; only the ones that never reached the wire are dropped.
TEST_F(TcpTransportFrameTest, AbandonInflightDropsOnlyTheUnsentTail) {
  setConnected();
  auto state = std::make_shared<TcpOpState>();

  auto chunks = makeChunks(state, 5, /*baseReqId=*/700);
  ASSERT_FALSE(admitInflightBulk(chunks).hasError());
  ASSERT_EQ(inflightCount(), 5u);

  abandonInflight(chunks, /*fromIdx=*/2);

  EXPECT_EQ(inflightCount(), 2u)
      << "only the unsent tail may be dropped; dropping a sent chunk's entry "
         "would leave its Ack unmatched";
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

// A VRAM ReadRequest used to be answered with a blocking device copy on the
// reader thread, so the reader stopped draining the socket for the length of an
// application GPU step. That re-arms the mutual-READ deadlock the reader/sender
// split exists to prevent, and the registry lease held across the copy stalled
// any concurrent erase() for just as long.
//
// The copy is staged instead: issued into a pinned slab and left to run, with
// the reply queued by the EventBase once it signals. Pinned matters -- the same
// copy into a frame's vector storage is pageable, which CUDA specifies as
// completing synchronously, so the reader would still be parked for the whole
// transfer while looking asynchronous.
TEST_F(TcpTransportFrameTest, AStagedVramReadReplyDoesNotBlockTheReader) {
  setConnected();
  stubStagingCopies();

  constexpr uint64_t kVramSegId = kSegId + 200;
  constexpr size_t kLen = 64;
  std::vector<std::byte> vram(kLen, std::byte{0x77});
  registry_->add(
      kVramSegId, vram.data(), kLen, MemoryType::VRAM, /*deviceId=*/0);

  feed(makeFrame(
      TcpOp::ReadRequest, kVramSegId, /*offset=*/0, kLen, /*payloadBytes=*/0));

  EXPECT_EQ(outQueueDepth(), 0u)
      << "a reply whose payload is still being copied must not be queued; the "
         "peer would receive an unfilled buffer";
  EXPECT_EQ(stagedCopyCount(), 1u);
  EXPECT_TRUE(allCopiesLandedInPinnedMemory())
      << "the copy went somewhere other than a pinned slab; a device-to-host "
         "copy into pageable memory blocks the thread that issues it, so the "
         "reader is parked for the whole transfer";
  EXPECT_EQ(syncCount(), 0)
      << "the reader must not synchronize: waiting on the device is exactly "
         "what staging exists to avoid";

  // The reader is not parked, so a request arriving behind the staged one is
  // answered while that copy is still running.
  feed(makeFrame(
      TcpOp::ReadRequest, kSegId, /*offset=*/0, kLen, /*payloadBytes=*/0));
  EXPECT_EQ(outQueueDepth(), 1u)
      << "the reader is blocked behind the VRAM copy: a request queued after it "
         "was not answered until the device finished";

  releaseStagingCopies();

  EXPECT_TRUE(waitFor([this]() { return outQueueDepth() >= 2u; }))
      << "the staged reply must be queued once its copy signals";
  EXPECT_EQ(outQueueDepth(), 2u);
}

// The pool is finite, so a read can arrive with nothing to stage into. It is
// recorded and answered later rather than waited on: waiting is the
// head-of-line block the whole staging path exists to remove, and it would be
// worse here because the thread that frees the next slab is the sender, which
// the reader would be holding up.
TEST_F(TcpTransportFrameTest, AVramReadWithNoFreeSlabIsDeferredNotBlocked) {
  setConnected();
  stubStagingCopies();

  constexpr uint64_t kVramSegId = kSegId + 210;
  constexpr size_t kLen = 64;
  std::vector<std::byte> vram(kLen, std::byte{0x33});
  registry_->add(
      kVramSegId, vram.data(), kLen, MemoryType::VRAM, /*deviceId=*/0);

  // Every slab, the reserved one included: nothing is left for the responder.
  auto pool = transportStagingPool();
  ASSERT_NE(pool, nullptr);
  std::vector<TcpPinnedSlab> held;
  held.reserve(pool->slabCount());
  for (size_t i = 0; i < pool->slabCount(); ++i) {
    held.push_back(pool->tryAcquire(/*allowReserved=*/true));
    ASSERT_TRUE(static_cast<bool>(held.back()));
  }

  feed(makeFrame(
      TcpOp::ReadRequest, kVramSegId, /*offset=*/0, kLen, /*payloadBytes=*/0));

  EXPECT_EQ(deferredReplyCount(), 1u)
      << "the read must be recorded for later, not dropped: the peer is "
         "blocked waiting for this reply";
  EXPECT_EQ(stagedCopyCount(), 0u)
      << "no slab was free, so no copy may have been issued -- a fallback into "
         "pageable memory would block the reader, which is the bug this path "
         "exists to avoid";
  EXPECT_EQ(syncCount(), 0)
      << "the reader must not wait on the device to make room";

  // And the reader is still serving: a DRAM read behind the deferred one is
  // answered immediately.
  feed(makeFrame(
      TcpOp::ReadRequest, kSegId, /*offset=*/0, kLen, /*payloadBytes=*/0));
  EXPECT_EQ(outQueueDepth(), 1u)
      << "a deferred VRAM read must not hold up unrelated requests";
}

// A deferral must be refused once the connection is swept. failAllPending()
// empties the deferred queue exactly so a lease is not pinned for the lifetime
// of the transport; a read that lands afterwards and is deferred anyway puts
// one straight back, and nothing drains it again until teardown. Written with
// the pool exhausted so the read has no choice but the deferral path, which is
// the only way to reach the check.
TEST_F(TcpTransportFrameTest, ADeferralIsRefusedOnceTheConnectionIsSwept) {
  setConnected();
  stubStagingCopies();

  constexpr uint64_t kVramSegId = kSegId + 230;
  constexpr size_t kLen = 64;
  std::vector<std::byte> vram(kLen, std::byte{0x44});
  registry_->add(
      kVramSegId, vram.data(), kLen, MemoryType::VRAM, /*deviceId=*/0);

  auto pool = transportStagingPool();
  ASSERT_NE(pool, nullptr);
  std::vector<TcpPinnedSlab> held;
  held.reserve(pool->slabCount());
  for (size_t i = 0; i < pool->slabCount(); ++i) {
    held.push_back(pool->tryAcquire(/*allowReserved=*/true));
    ASSERT_TRUE(static_cast<bool>(held.back()));
  }

  failAllPending();
  ASSERT_EQ(deferredReplyCount(), 0u)
      << "precondition: the sweep left the deferred queue empty";

  feed(makeFrame(
      TcpOp::ReadRequest, kVramSegId, /*offset=*/0, kLen, /*payloadBytes=*/0));

  EXPECT_EQ(deferredReplyCount(), 0u)
      << "a deferral after the sweep pins its segment lease until the transport "
         "is destroyed, which is what the sweep clears the queue to avoid";
  EXPECT_EQ(stagedCopyCount(), 0u)
      << "nothing may be staged for a read on a broken connection";
}

// The other half of deferral: a slab coming back has to actually restart the
// deferred read. The release that matters is the sender's, after the frame it
// was transmitting is gone, so this drives it through senderLoop rather than
// releasing a slab by hand.
TEST_F(TcpTransportFrameTest, AReleasedSlabStartsTheDeferredRead) {
  setConnected();
  stubStagingCopies();

  constexpr uint64_t kVramSegId = kSegId + 220;
  constexpr size_t kLen = 64;
  std::vector<std::byte> vram(kLen, std::byte{0x55});
  registry_->add(
      kVramSegId, vram.data(), kLen, MemoryType::VRAM, /*deviceId=*/0);

  // All but one slab held, so the first read takes the last one and the second
  // has nowhere to go.
  auto pool = transportStagingPool();
  ASSERT_NE(pool, nullptr);
  std::vector<TcpPinnedSlab> held;
  held.reserve(pool->slabCount());
  for (size_t i = 0; i + 1 < pool->slabCount(); ++i) {
    held.push_back(pool->tryAcquire(/*allowReserved=*/true));
    ASSERT_TRUE(static_cast<bool>(held.back()));
  }

  feed(makeFrame(
      TcpOp::ReadRequest, kVramSegId, /*offset=*/0, kLen, /*payloadBytes=*/0));
  feed(makeFrame(
      TcpOp::ReadRequest, kVramSegId, /*offset=*/0, kLen, /*payloadBytes=*/0));
  ASSERT_EQ(stagedCopyCount(), 1u);
  ASSERT_EQ(deferredReplyCount(), 1u);

  auto& conn = installCountingConn();
  std::thread sender([this]() { runSenderLoop(); });

  // The first copy signals, its reply is queued and sent, and only then is its
  // slab free for the deferred read.
  releaseStagingCopies();
  EXPECT_TRUE(waitFor([&]() { return conn.sendCount() >= 2; }))
      << "the deferred read was never started, so its reply never went out: a "
         "slab returning to the pool must wake it";
  EXPECT_EQ(deferredReplyCount(), 0u);
  EXPECT_EQ(stagedCopyCount(), 2u);
  EXPECT_TRUE(allCopiesLandedInPinnedMemory());

  closeOutQueue();
  sender.join();
}

// A deferred read holds its lease with no copy yet issued, and that lease has
// to keep blocking deregistration: the copy will read that buffer, just later.
// Releasing the lease at deferral time and re-finding the segment later would
// mean answering a read out of memory the owner had already freed.
TEST_F(TcpTransportFrameTest, ADeferredReadStillBlocksDeregistration) {
  setConnected();
  stubStagingCopies();

  constexpr uint64_t kVramSegId = kSegId + 230;
  constexpr size_t kLen = 64;
  std::vector<std::byte> vram(kLen, std::byte{0x66});
  registry_->add(
      kVramSegId, vram.data(), kLen, MemoryType::VRAM, /*deviceId=*/0);

  auto pool = transportStagingPool();
  ASSERT_NE(pool, nullptr);
  std::vector<TcpPinnedSlab> held;
  held.reserve(pool->slabCount());
  for (size_t i = 0; i < pool->slabCount(); ++i) {
    held.push_back(pool->tryAcquire(/*allowReserved=*/true));
  }

  feed(makeFrame(
      TcpOp::ReadRequest, kVramSegId, /*offset=*/0, kLen, /*payloadBytes=*/0));
  ASSERT_EQ(deferredReplyCount(), 1u);

  std::atomic<bool> erased{false};
  std::thread eraser([&]() {
    registry_->erase(kVramSegId);
    erased.store(true, std::memory_order_release);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(erased.load(std::memory_order_acquire))
      << "erase() returned while a deferred read still held a lease: the owner "
         "is now free to free a buffer this transport will still copy out of";

  // Teardown discards what is deferred, which releases the lease.
  transport_->shutdown();
  eraser.join();
  EXPECT_TRUE(erased.load(std::memory_order_acquire));
  EXPECT_EQ(stagedCopyCount(), 0u)
      << "a discarded deferred read must not have started a copy into memory "
         "that is being torn down";
}

// A thread parked in the staging pool's blocking acquire() must be released by
// shutdown(). acquire() waits on `freed_` with no deadline and `closed_` as its
// only escape, and it runs on the application's own put() thread -- which
// shutdown() never joins and nothing else wakes. Before shutdown() closed this
// pool, a put() in flight across teardown parked forever, because the sender
// that would have returned a staging slab was joined away first.
//
// Written against the pool directly rather than through put(): reaching the
// blocking acquire() through put() needs a live peer and a full wave, and the
// property under test belongs to shutdown()'s ordering, not to the put path.
TEST_F(TcpTransportFrameTest, ShutdownReleasesAThreadParkedInStagingAcquire) {
  setConnected();
  stubStagingCopies();

  auto pool = transportStagingPool();
  ASSERT_NE(pool, nullptr);

  // Hold every slab so the acquire below cannot be satisfied.
  std::vector<TcpPinnedSlab> held;
  held.reserve(pool->slabCount());
  for (size_t i = 0; i < pool->slabCount(); ++i) {
    held.push_back(pool->tryAcquire(/*allowReserved=*/true));
    ASSERT_TRUE(static_cast<bool>(held.back()));
  }

  std::atomic<bool> returned{false};
  std::thread parked([&]() {
    // Blocks: nothing is free, and this test never releases `held`.
    auto leases = pool->acquire(1);
    // Either outcome is fine; the point is that it *returns*.
    (void)leases;
    returned.store(true, std::memory_order_release);
  });

  // Give it time to reach the wait rather than racing the assertion below.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ASSERT_FALSE(returned.load(std::memory_order_acquire))
      << "precondition: the acquire must actually be parked";

  transport_->shutdown();
  parked.join();
  EXPECT_TRUE(returned.load(std::memory_order_acquire))
      << "shutdown() must close the staging pool; otherwise a put() in flight "
         "across teardown never comes back";
}

// A read the pool cannot serve and the deferred queue cannot hold is answered
// with an Error rather than dropped or blocked on. Dropping it leaves the peer
// waiting forever; blocking stops the reader; failing the connection punishes
// every unrelated transfer on it.
TEST_F(TcpTransportFrameTest, DeferredQueueOverflowFailsOnlyThatRequest) {
  setConnected();
  stubStagingCopies();

  constexpr uint64_t kVramSegId = kSegId + 240;
  constexpr size_t kLen = 64;
  std::vector<std::byte> vram(kLen, std::byte{0x11});
  registry_->add(
      kVramSegId, vram.data(), kLen, MemoryType::VRAM, /*deviceId=*/0);

  auto pool = transportStagingPool();
  ASSERT_NE(pool, nullptr);
  std::vector<TcpPinnedSlab> held;
  held.reserve(pool->slabCount());
  for (size_t i = 0; i < pool->slabCount(); ++i) {
    held.push_back(pool->tryAcquire(/*allowReserved=*/true));
  }
  fillDeferredReplies(deferredCap());

  feed(makeFrame(
      TcpOp::ReadRequest, kVramSegId, /*offset=*/0, kLen, /*payloadBytes=*/0));

  EXPECT_FALSE(connClosed())
      << "one unservable read must not take the connection down";
  EXPECT_TRUE(queuedErrorFrame())
      << "the peer must be told this read failed; silence leaves it waiting for "
         "a reply that will never come";
  EXPECT_EQ(deferredReplyCount(), deferredCap())
      << "the refused read must not have been queued anyway";
}

// A VRAM read larger than a staging slab cannot be served at all: our own get()
// chunks at kMaxChunkSize, so this is the version-skew case, and it has to be a
// per-request error rather than a fatal one for the same reason the wire-frame
// cap is.
TEST_F(TcpTransportFrameTest, AVramReadLargerThanASlabIsRefusedPerRequest) {
  setConnected();
  stubStagingCopies();

  constexpr uint64_t kVramSegId = kSegId + 250;
  const size_t oversized = slabPayloadCap() + 1;
  // Registered, but never read from: the request is refused before any copy.
  std::vector<std::byte> vram(1, std::byte{0x99});
  registry_->add(
      kVramSegId, vram.data(), oversized, MemoryType::VRAM, /*deviceId=*/0);

  feed(makeFrame(
      TcpOp::ReadRequest,
      kVramSegId,
      /*offset=*/0,
      oversized,
      /*payloadBytes=*/0));

  EXPECT_FALSE(connClosed())
      << "an oversized read must not be fatal to the connection";
  EXPECT_TRUE(queuedErrorFrame()) << "the peer must be told this read failed";
  EXPECT_EQ(stagedCopyCount(), 0u)
      << "nothing may be copied for a read that cannot fit a slab";
  EXPECT_EQ(deferredReplyCount(), 0u)
      << "an unservable read must be refused, not deferred forever";
}

// The copy is issued before the event exists, so any failure after that point
// leaves a D2H copy running into `frame` -- and returning an error unwinds
// `frame` and its lease. The GPU is then writing into memory the allocator has
// taken back, and reading from a segment a waiting erase() is now free to
// deregister. drainPendingReadReplies() waits per-device for exactly this
// reason; these error paths did not.
//
// The wait is the only observable evidence available: MockCudaApi mocks
// memcpyAsync, so the out-of-bounds write itself cannot be seen from a unit
// test. What is pinned is that nothing releases the buffer without waiting.
TEST_F(TcpTransportFrameTest, AReadReplyWhoseEventFailsWaitsForTheIssuedCopy) {
  setConnected();

  constexpr uint64_t kVramSegId = kSegId + 300;
  constexpr size_t kLen = 64;
  std::vector<std::byte> vram(kLen, std::byte{0x77});
  registry_->add(
      kVramSegId, vram.data(), kLen, MemoryType::VRAM, /*deviceId=*/0);

  std::atomic<int> syncs{0};
  ON_CALL(
      *cudaApi_,
      memcpyAsync(
          ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
      .WillByDefault(::testing::Return(Ok()));
  // Fails only after memcpyAsync has already reported success, which is what
  // leaves a copy in flight with no event to track it.
  ON_CALL(*cudaApi_, eventCreate(::testing::_))
      .WillByDefault(
          ::testing::Return(Err(ErrCode::DriverError, "test: no event")));
  ON_CALL(*cudaApi_, eventDestroy(::testing::_))
      .WillByDefault(::testing::Return(Ok()));
  ON_CALL(*cudaApi_, streamSynchronize(::testing::_))
      .WillByDefault([&](auto) -> Status {
        syncs.fetch_add(1, std::memory_order_release);
        return Ok();
      });

  feed(makeFrame(
      TcpOp::ReadRequest, kVramSegId, /*offset=*/0, kLen, /*payloadBytes=*/0));

  EXPECT_GE(syncs.load(std::memory_order_acquire), 1)
      << "the staging frame was released with a copy still in flight: the event "
         "failed after memcpyAsync succeeded, and nothing waited for the device "
         "before the frame and its lease went out of scope";
  EXPECT_TRUE(queuedErrorFrame())
      << "the failed read must still be answered per-request";
}

// eventQuery failing says nothing about the copy -- it may still be running --
// but the record was popped and its frame destroyed anyway, which is the same
// use-after-free as the staging path above.
//
// The Error frame is queued after the wait, so waiting for the frame and then
// asserting on the count is what makes this deterministic: observing the frame
// means the barrier already ran.
TEST_F(TcpTransportFrameTest, AFailedEventQueryWaitsBeforeDroppingTheReply) {
  setConnected();

  constexpr uint64_t kVramSegId = kSegId + 301;
  constexpr size_t kLen = 64;
  std::vector<std::byte> vram(kLen, std::byte{0x77});
  registry_->add(
      kVramSegId, vram.data(), kLen, MemoryType::VRAM, /*deviceId=*/0);

  std::atomic<int> syncs{0};
  ON_CALL(
      *cudaApi_,
      memcpyAsync(
          ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
      .WillByDefault(::testing::Return(Ok()));
  ON_CALL(*cudaApi_, eventCreate(::testing::_))
      .WillByDefault([](auto* event) -> Status {
        // Spelled with auto because this TU is not hipified: cudaEvent_t is
        // ihipEvent_t* here, and naming it does not compile under ROCm.
        *event = {};
        return Ok();
      });
  ON_CALL(*cudaApi_, eventRecord(::testing::_, ::testing::_))
      .WillByDefault(::testing::Return(Ok()));
  ON_CALL(*cudaApi_, eventDestroy(::testing::_))
      .WillByDefault(::testing::Return(Ok()));
  ON_CALL(*cudaApi_, eventQuery(::testing::_))
      .WillByDefault([](auto) -> Result<bool> {
        return Result<bool>(Err(ErrCode::DriverError, "test: query failed"));
      });
  ON_CALL(*cudaApi_, streamSynchronize(::testing::_))
      .WillByDefault([&](auto) -> Status {
        syncs.fetch_add(1, std::memory_order_release);
        return Ok();
      });

  feed(makeFrame(
      TcpOp::ReadRequest, kVramSegId, /*offset=*/0, kLen, /*payloadBytes=*/0));

  // The poll runs on the EventBase, so this waits for its observable result.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!queuedErrorFrame() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(queuedErrorFrame())
      << "the reply whose event query failed must still be answered "
         "per-request";
  EXPECT_GE(syncs.load(std::memory_order_acquire), 1)
      << "the reply was dropped without waiting for the copy, so the GPU may "
         "still be writing into the frame that was just freed";
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

// The reader refuses the connection on unmatched-send overflow, and shutdown()
// closes it too, from an application thread. Conn::close() tests the fd, closes
// it, then clears it with no synchronisation, so two callers that both see it
// open both ::close() it -- and the second reaps whatever descriptor another
// thread in the process has since been handed. Only one close may reach the
// connection for the lifetime of the transport.
TEST_F(TcpTransportFrameTest, TheDataConnectionIsClosedAtMostOnce) {
  setConnected();
  installFailingConn();

  constexpr size_t kChunk = 4UL * 1024 * 1024;
  const size_t maxFrames = (unmatchedCap() / kChunk) + 4;
  size_t frames = 0;
  while (!connClosed() && frames < maxFrames) {
    feed(makeFrame(TcpOp::Send, kSegId, /*offset=*/0, kChunk, kChunk));
    ++frames;
  }
  ASSERT_EQ(connCloseCount(), 1) << "the reader must refuse the connection";

  // The path that races the reader in production. Sequential here because the
  // race is a lost check-then-act, so a second close arriving at any time is
  // the defect; running them concurrently would only make the test flaky about
  // observing it.
  transport_->shutdown();

  EXPECT_EQ(connCloseCount(), 1)
      << "shutdown() must not close a connection the reader already closed";
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

// handleFrame() is noexcept, so an exception escaping it is std::terminate --
// i.e. a peer-triggerable process abort. The VRAM staging path runs
// CudaDeviceGuard, which throws when setDevice fails, and the deviceId it uses
// comes from whatever the application passed to registerSegment (Segment
// defaults it to -1). So this is reachable without a hostile peer at all.
TEST_F(TcpTransportFrameTest, VramStagingThrowDoesNotAbortTheReader) {
  constexpr uint64_t kVramSegId = kSegId + 500;
  std::vector<std::byte> vram(kSegLen, std::byte{0});
  registry_->add(
      kVramSegId, vram.data(), kSegLen, MemoryType::VRAM, /*deviceId=*/-1);
  EXPECT_CALL(*cudaApi_, setDevice(-1))
      .WillRepeatedly(
          ::testing::Return(
              Err(ErrCode::InvalidArgument, "test: no such device")));

  // Without the try/catch boundary in handleFrame this call terminates the
  // process, so the failure mode of this test is an abort, not an assertion.
  feed(makeFrame(TcpOp::Write, kVramSegId, /*offset=*/0, /*len=*/8, 8));

  // Failed rather than silently dropped: an exception can land midway through a
  // staging copy, so protocol state is no longer known to be sound.
  EXPECT_TRUE(connBroken())
      << "a throw inside handleFrame must fail the connection";
  EXPECT_FALSE(readerRunning())
      << "the reader must stop rather than keep serving frames";
}

// A frame staged in a pinned slab is transmitted straight out of that slab, and
// the slab is on loan for the whole send. Returning it when the item is popped
// would let the next staging copy start writing into bytes the socket has not
// read yet -- silent corruption, and only under load.
TEST_F(TcpTransportFrameTest, ASlabFrameIsSentInPlaceAndHeldUntilTheSendEnds) {
  setConnected();
  auto pool = makeSlabPool(/*slabCount=*/1, /*reserved=*/0);
  ASSERT_NE(pool, nullptr);
  auto slab = pool->tryAcquire(/*allowReserved=*/false);
  ASSERT_TRUE(static_cast<bool>(slab));
  const uint8_t* slabData = slab.data();

  ASSERT_TRUE(enqueueStagedFrame(
      slabFrame(std::move(slab), TcpOp::ReadReply, /*reqId=*/11)));

  auto* conn = installGatedConn();
  std::thread sender([this]() { runSenderLoop(); });
  conn->waitForSend();

  EXPECT_EQ(conn->sentPointer(), slabData)
      << "the span handed to the socket must point into the slab; a copy on the "
         "way out would defeat staging there at all";
  EXPECT_EQ(conn->sentSize(), sizeof(TcpMsgHeader));
  EXPECT_FALSE(static_cast<bool>(pool->tryAcquire(/*allowReserved=*/true)))
      << "the slab must still be on loan while the send is outstanding";

  conn->releaseSend();
  EXPECT_TRUE(waitFor([&]() {
    return static_cast<bool>(pool->tryAcquire(/*allowReserved=*/true));
  })) << "the slab must go back once the send resolves, or the pool leaks a "
         "slab per staged reply";

  closeOutQueue();
  sender.join();
}

// The slab has to come back on the error path too. A send failure tears the
// connection down but not the transport, so a slab stranded here is one the
// pool never sees again.
TEST_F(TcpTransportFrameTest, ASlabFrameIsReleasedAfterAFailedSend) {
  setConnected();
  installFailingConn();
  auto pool = makeSlabPool(/*slabCount=*/1, /*reserved=*/0);
  ASSERT_NE(pool, nullptr);
  auto slab = pool->tryAcquire(/*allowReserved=*/false);
  ASSERT_TRUE(static_cast<bool>(slab));

  ASSERT_TRUE(enqueueStagedFrame(
      slabFrame(std::move(slab), TcpOp::ReadReply, /*reqId=*/12)));

  runSenderLoop(); // returns once send() fails

  EXPECT_TRUE(static_cast<bool>(pool->tryAcquire(/*allowReserved=*/true)))
      << "a failed send must still return the slab";
}

// Clearing the queue is the other way a frame dies. failAllPending() drops
// every queued item, and each one owning its slab is what keeps that from
// leaking the pool dry on a connection failure.
TEST_F(TcpTransportFrameTest, ClearingTheOutQueueReleasesSlabFrames) {
  setConnected();
  auto pool = makeSlabPool(/*slabCount=*/2, /*reserved=*/0);
  ASSERT_NE(pool, nullptr);
  auto slabs = pool->acquire(2);
  ASSERT_TRUE(slabs.hasValue());
  std::vector<TcpFrame> frames;
  frames.push_back(
      slabFrame(std::move(slabs.value()[0]), TcpOp::ReadReply, /*reqId=*/21));
  frames.push_back(
      slabFrame(std::move(slabs.value()[1]), TcpOp::ReadReply, /*reqId=*/22));
  ASSERT_TRUE(enqueueFrames(std::move(frames), /*mayBlock=*/false));
  ASSERT_EQ(outQueueDepth(), 2u);

  failAllPending();

  EXPECT_EQ(outQueueDepth(), 0u);
  auto reclaimed = pool->acquire(2);
  EXPECT_TRUE(reclaimed.hasValue())
      << "both slabs must be back in the pool after the queue is cleared";
}

// enqueueFrames exists so a put wave lands in the queue as one step. Enqueuing
// chunk by chunk lets the sender start transmitting a group this side may still
// fail to finish staging, which is a partial remote write the caller is told
// nothing about.
TEST_F(TcpTransportFrameTest, EnqueueFramesQueuesTheWholeGroupInOrder) {
  setConnected();
  std::vector<TcpFrame> frames;
  for (uint64_t reqId = 31; reqId <= 33; ++reqId) {
    TcpMsgHeader header;
    header.op = static_cast<uint8_t>(TcpOp::Write);
    header.reqId = reqId;
    frames.emplace_back(serializeTcpHeader(header));
  }
  ASSERT_TRUE(enqueueFrames(std::move(frames), /*mayBlock=*/false));

  EXPECT_THAT(queuedReqIds(), ::testing::ElementsAre(31u, 32u, 33u))
      << "the group must be queued whole and in order";
  EXPECT_EQ(outQueueBytes(), 3 * sizeof(TcpMsgHeader))
      << "every frame in the group must be accounted against the queue cap";
}

// A closed queue must take none of the group. Half a wave queued and half
// refused is the partial write the batch enqueue exists to prevent.
TEST_F(TcpTransportFrameTest, EnqueueFramesQueuesNothingWhenTheQueueIsClosed) {
  setConnected();
  closeOutQueue();

  auto pool = makeSlabPool(/*slabCount=*/2, /*reserved=*/0);
  ASSERT_NE(pool, nullptr);
  auto slabs = pool->acquire(2);
  ASSERT_TRUE(slabs.hasValue());
  std::vector<TcpFrame> frames;
  frames.push_back(
      slabFrame(std::move(slabs.value()[0]), TcpOp::Write, /*reqId=*/41));
  frames.push_back(
      slabFrame(std::move(slabs.value()[1]), TcpOp::Write, /*reqId=*/42));

  EXPECT_FALSE(enqueueFrames(std::move(frames), /*mayBlock=*/false));
  EXPECT_EQ(outQueueDepth(), 0u);
  EXPECT_EQ(outQueueBytes(), 0u);
  auto reclaimed = pool->acquire(2);
  EXPECT_TRUE(reclaimed.hasValue())
      << "a refused group must not strand the slabs it was built in";
}

// A put stages every chunk of a wave and only then queues them, so a staging
// failure partway through leaves the peer untouched instead of holding the
// chunks that happened to copy first. Before this the commit loop queued each
// chunk as its own copy finished, and the sender could already have flushed
// them: the caller was told the put failed while some of it had landed, at
// offsets nobody could name.
//
// This is the whole-wave case, which is what nearly every put is.
TEST_F(TcpTransportFrameTest, AFullWaveReachesTheQueueOnlyOnceEveryCopyIsDone) {
  setConnected();
  stubStagingCopies();
  gateStagingWaits();

  const size_t chunks = waveCap();
  VramPut transfer(chunks * slabPayloadCap());
  std::thread putter([&]() { (void)put(transfer.requests()); });

  // Every copy in the wave is in flight...
  EXPECT_TRUE(waitFor([&]() { return stagedCopyCount() == chunks; }))
      << "the wave must issue all of its copies before waiting on any of them; "
         "waiting per chunk is what serialised staging against itself";
  // ...and nothing is queued, because none of them has finished.
  EXPECT_EQ(outQueueDepth(), 0u)
      << "a Write reached the queue before its wave had finished staging; the "
         "sender may already have put a partial transfer on the wire";
  EXPECT_TRUE(allCopiesLandedInPinnedMemory());

  releaseStagingWaits();

  EXPECT_TRUE(waitFor([&]() { return outQueueDepth() == chunks; }))
      << "the whole wave must be queued once its copies are done";
  auto ids = queuedReqIds();
  ASSERT_EQ(ids.size(), chunks);
  EXPECT_TRUE(std::is_sorted(ids.begin(), ids.end()))
      << "the wave must keep the caller's chunk order";
  putter.join();
}

// A staging failure inside a wave must queue nothing at all, and must not hand
// a slab back while the GPU is still writing into it. Copies already launched
// keep running after a later one fails; a slab released underneath one of them
// goes straight to the next staging copy, and the two then race over the same
// bytes.
TEST_F(TcpTransportFrameTest, AFailedWaveQueuesNothingAndDrainsWhatItLaunched) {
  setConnected();
  stubStagingCopies();
  failCopyAfter(2);

  const size_t chunks = waveCap();
  VramPut transfer(chunks * slabPayloadCap());
  auto future = put(transfer.requests());

  ASSERT_EQ(
      future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_TRUE(future.get().hasError()) << "the caller must be told it failed";
  EXPECT_EQ(outQueueDepth(), 0u)
      << "a wave that failed to stage must queue nothing; anything queued is a "
         "partial write the peer will apply and nobody will hear about";
  EXPECT_GE(syncCount(), 1)
      << "the copies that were launched must be waited for before their slabs "
         "go back to the pool, or the next staging copy writes into a buffer "
         "the GPU is still filling";

  auto pool = transportStagingPool();
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(freeSlabs(*pool), pool->slabCount())
      << "a failed wave must strand none of its slabs, and must leave none of "
         "them held by a frame it queued";
}

// Above one wave the guarantee is per wave, not per put, and that boundary is
// documented on put(). This pins it: with 16 chunks the first 15 are queued --
// and so may reach the peer -- while the 16th has not been staged at all.
// Whatever this test asserts is what callers are promised, so it is worth being
// explicit that the promise stops here.
TEST_F(TcpTransportFrameTest, APutLargerThanOneWaveIsAtomicOnlyPerWave) {
  setConnected();
  stubStagingCopies();
  releaseStagingCopies();

  const size_t chunks = waveCap() + 1;
  VramPut transfer(chunks * slabPayloadCap());
  // On its own thread because it is expected to park: nothing drains the queue,
  // so the first wave's slabs stay on loan and the second wave has nothing to
  // stage into.
  auto putResult = std::async(
      std::launch::async, [&]() { return put(transfer.requests()).get(); });

  EXPECT_TRUE(waitFor([&]() { return outQueueDepth() == waveCap(); }));
  EXPECT_EQ(outQueueDepth(), waveCap())
      << "the documented boundary moved: a put above one wave is expected to "
         "queue its first wave before the rest has been staged";
  EXPECT_EQ(stagedCopyCount(), waveCap())
      << "the second wave must not be staged until slabs come back";

  // And teardown must not strand the parked put. It is released by the queue
  // being dropped, which is what hands its slabs back; a caller thread is not
  // one shutdown() can join its way out of, so nothing else would.
  transport_->shutdown();
  ASSERT_EQ(
      putResult.wait_for(std::chrono::seconds(5)), std::future_status::ready)
      << "teardown left a put parked waiting for staging slabs";
  EXPECT_TRUE(putResult.get().hasError());
}

// Two waves, a queue whose cap (64 MiB) is only just above one wave (60 MiB),
// and a pool that only refills as the sender drains. Every one of those can
// block a put, and they depend on each other: the wave waits for slabs, the
// slabs wait for the sender, and the sender needs the queue mutex the wave must
// therefore not be holding.
TEST_F(TcpTransportFrameTest, BackToBackWavesDrainWithoutDeadlock) {
  setConnected();
  stubStagingCopies();
  releaseStagingCopies();
  auto& conn = installCountingConn();
  std::thread sender([this]() { runSenderLoop(); });

  const size_t chunks = 2 * waveCap();
  VramPut transfer(chunks * slabPayloadCap());
  // The future stays open -- nothing Acks these writes -- so the frames
  // reaching the connection are what says both waves got through.
  auto future = put(transfer.requests());

  EXPECT_TRUE(waitFor([&]() {
    return static_cast<size_t>(conn.sendCount()) == chunks;
  })) << "both waves must make it out; a wave holding slabs while it waits for "
         "queue room, or holding outMu_ while it waits for slabs, deadlocks here";

  closeOutQueue();
  sender.join();
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
