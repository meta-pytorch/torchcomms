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
#include <span>
#include <vector>

#include "comms/uniflow/Segment.h"
#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/tcp/TcpRegistrationHandle.h"
#include "comms/uniflow/transport/tcp/TcpWireProtocol.h"

namespace uniflow {

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

namespace {

class ScriptedRecvConn final : public controller::Conn {
 public:
  explicit ScriptedRecvConn(std::vector<uint8_t> frame)
      : frame_(std::move(frame)) {}

  std::future<Result<size_t>> send(std::span<const uint8_t> data) override {
    return make_ready_future<Result<size_t>>(Result<size_t>(data.size()));
  }

  std::future<Result<size_t>> recv(std::vector<uint8_t>& data) override {
    ++vectorRecvs_;
    return next([&]() { data = frame_; });
  }

  std::future<Result<size_t>> recv(std::span<uint8_t> data) override {
    ++spanRecvs_;
    return next([&]() {
      ASSERT_GE(data.size(), frame_.size());
      std::memcpy(data.data(), frame_.data(), frame_.size());
    });
  }

  void close() override {}

  int spanRecvs() const {
    return spanRecvs_;
  }

  int vectorRecvs() const {
    return vectorRecvs_;
  }

 private:
  template <typename Fill>
  std::future<Result<size_t>> next(Fill&& fill) {
    if (delivered_) {
      return make_ready_future<Result<size_t>>(
          Err(ErrCode::ConnectionFailed, "test: end of script"));
    }
    delivered_ = true;
    fill();
    return make_ready_future<Result<size_t>>(Result<size_t>(frame_.size()));
  }

  std::vector<uint8_t> frame_;
  bool delivered_{false};
  int spanRecvs_{0};
  int vectorRecvs_{0};
};

} // namespace

class TcpReceivePoolTest : public ::testing::Test {
 protected:
  static constexpr size_t kLen = 64;
  static constexpr uint64_t kRemoteSegId = 17;

  void SetUp() override {
    evbThread_ =
        std::make_unique<ScopedEventBaseThread>("tcp-receive-pool-test");
    cudaApi_ = std::make_shared<::testing::NiceMock<MockCudaApi>>();
    ON_CALL(*cudaApi_, getDevice())
        .WillByDefault(::testing::Return(Result<int>(0)));
    ON_CALL(*cudaApi_, setDevice(::testing::_))
        .WillByDefault(::testing::Return(Ok()));
    ON_CALL(*cudaApi_, hostAlloc(::testing::_, ::testing::_))
        .WillByDefault([this](size_t size, unsigned int) -> Result<void*> {
          auto memory = std::make_unique<uint8_t[]>(size);
          void* result = memory.get();
          std::lock_guard<std::mutex> lk(allocMu_);
          allocations_.push_back(std::move(memory));
          return result;
        });
    ON_CALL(*cudaApi_, hostFree(::testing::_))
        .WillByDefault([this](void* ptr) -> Status {
          std::lock_guard<std::mutex> lk(allocMu_);
          std::erase_if(allocations_, [ptr](const auto& allocation) {
            return allocation.get() == ptr;
          });
          return Ok();
        });
  }

  void TearDown() override {
    if (transport_) {
      transport_->shutdown();
      transport_.reset();
    }
    evbThread_.reset();
  }

  void makeTransport(bool asyncGetH2d = true) {
    TcpTransportConfig config;
    config.asyncGetH2d = asyncGetH2d;
    transport_ = std::make_unique<TcpTransport>(
        /*deviceId=*/-1,
        evbThread_->getEventBase(),
        std::make_shared<TcpSegmentRegistry>(),
        config,
        /*host=*/"127.0.0.1",
        cudaApi_);
  }

  std::future<Status> issueGet(size_t len, MemoryType memType) {
    localBytes_.resize(len == 0 ? 1 : len);
    auto local = SegmentTest::makeRegistered(
        localBytes_.data(), len, memType, memType == MemoryType::VRAM ? 0 : -1);
    auto remote = SegmentTest::makeRemote(
        reinterpret_cast<void*>(0x100000),
        len,
        std::make_unique<TcpRemoteRegistrationHandle>(kRemoteSegId, len));
    const TransferRequest request{local.span(), remote.span()};
    transport_->state_ = TransportState::Connected;
    return transport_->get(std::span<const TransferRequest>{&request, 1});
  }

  static std::vector<uint8_t> ackFrame() {
    TcpMsgHeader header;
    header.op = static_cast<uint8_t>(TcpOp::Ack);
    header.reqId = 999;
    return serializeTcpHeader(header);
  }

  static std::vector<uint8_t> readReplyFrame(uint64_t reqId, size_t len) {
    TcpMsgHeader header;
    header.op = static_cast<uint8_t>(TcpOp::ReadReply);
    header.reqId = reqId;
    header.len = len;
    std::vector<uint8_t> frame(sizeof(header) + len, uint8_t{0xCD});
    std::memcpy(frame.data(), &header, sizeof(header));
    return frame;
  }

  std::shared_ptr<TcpPinnedSlabPool> receivePool() {
    return transport_->receivePoolIfCreated();
  }

  std::shared_ptr<TcpPinnedSlabPool> createReceivePool() {
    return transport_->ensureReceivePool();
  }

  void failPending() {
    transport_->failAllPending("test cleanup");
  }

  void runReader(std::unique_ptr<ScriptedRecvConn> conn) {
    scriptedConn_ = conn.get();
    transport_->dataConn_ = std::move(conn);
    transport_->running_.store(true, std::memory_order_release);
    transport_->readerLoop();
  }

  std::future<Status>
  postVramRead(void* destination, void* stream = nullptr, size_t len = kLen) {
    auto state = std::make_shared<TcpOpState>();
    state->remaining = 1;
    auto future = state->promise.get_future();
    transport_->inflight_[1] = TcpInflight{
        state,
        destination,
        len,
        /*isRead=*/true,
        MemoryType::VRAM,
        /*deviceId=*/0,
        stream};
    return future;
  }

  void handleFrame(std::span<const uint8_t> frame, TcpPinnedSlab slab) {
    transport_->handleFrame(frame, std::move(slab));
  }

  std::unique_ptr<ScopedEventBaseThread> evbThread_;
  std::shared_ptr<::testing::NiceMock<MockCudaApi>> cudaApi_;
  std::unique_ptr<TcpTransport> transport_;
  ScriptedRecvConn* scriptedConn_{nullptr};
  std::mutex allocMu_;
  std::vector<std::unique_ptr<uint8_t[]>> allocations_;
  std::vector<uint8_t> localBytes_;
};

TEST_F(TcpReceivePoolTest, PoolIsLazyAndOnlyCreatedForEnabledNonZeroVramGet) {
  makeTransport();
  EXPECT_EQ(receivePool(), nullptr);

  auto dram = issueGet(kLen, MemoryType::DRAM);
  EXPECT_EQ(receivePool(), nullptr);
  failPending();
  EXPECT_TRUE(dram.get().hasError());

  transport_->shutdown();
  transport_.reset();
  makeTransport();
  auto vram = issueGet(kLen, MemoryType::VRAM);
  auto pool = receivePool();
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->slabSize(), kMaxFrameSize);
  EXPECT_EQ(pool->slabCount(), 2);
  failPending();
  EXPECT_TRUE(vram.get().hasError());

  transport_->shutdown();
  transport_.reset();
  makeTransport(/*asyncGetH2d=*/false);
  auto disabled = issueGet(kLen, MemoryType::VRAM);
  EXPECT_EQ(receivePool(), nullptr);
  failPending();
  EXPECT_TRUE(disabled.get().hasError());
}

TEST_F(TcpReceivePoolTest, AllocationFailureFallsBackWithoutRetrying) {
  makeTransport();
  EXPECT_CALL(*cudaApi_, hostAlloc(::testing::_, ::testing::_))
      .Times(1)
      .WillOnce(
          ::testing::Return(
              Result<void*>(Err(
                  ErrCode::DriverError, "test: pinned allocation failed"))));

  auto first = issueGet(kLen, MemoryType::VRAM);
  EXPECT_EQ(receivePool(), nullptr);
  auto second = issueGet(kLen, MemoryType::VRAM);
  EXPECT_EQ(receivePool(), nullptr);

  runReader(std::make_unique<ScriptedRecvConn>(ackFrame()));
  EXPECT_EQ(scriptedConn_->spanRecvs(), 0);
  EXPECT_EQ(scriptedConn_->vectorRecvs(), 2);
  EXPECT_TRUE(first.get().hasError());
  EXPECT_TRUE(second.get().hasError());
}

TEST_F(TcpReceivePoolTest, ReaderUsesSlabAndFallsBackToVectorWhenPoolIsBusy) {
  makeTransport();
  auto pool = createReceivePool();
  ASSERT_NE(pool, nullptr);

  runReader(std::make_unique<ScriptedRecvConn>(ackFrame()));
  EXPECT_EQ(scriptedConn_->spanRecvs(), 2);
  EXPECT_EQ(scriptedConn_->vectorRecvs(), 0);

  auto first = pool->tryAcquire(/*allowReserved=*/true);
  auto second = pool->tryAcquire(/*allowReserved=*/true);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  runReader(std::make_unique<ScriptedRecvConn>(ackFrame()));
  EXPECT_EQ(scriptedConn_->spanRecvs(), 0);
  EXPECT_EQ(scriptedConn_->vectorRecvs(), 2);
}

TEST_F(TcpReceivePoolTest, VramReadReplyKeepsSlabUntilEventCompletes) {
  makeTransport();
  auto pool = createReceivePool();
  ASSERT_NE(pool, nullptr);
  auto slab = pool->tryAcquire(/*allowReserved=*/true);
  ASSERT_TRUE(slab);
  auto frame = readReplyFrame(/*reqId=*/1, kLen);
  ASSERT_LE(frame.size(), slab.capacity());
  std::memcpy(slab.data(), frame.data(), frame.size());
  const auto bytes = std::span<const uint8_t>{slab.data(), frame.size()};

  std::vector<uint8_t> destination(kLen);
  auto* const callerStream = reinterpret_cast<void*>(0xCA11);
  auto future = postVramRead(destination.data(), callerStream);

  std::mutex queryMu;
  std::condition_variable queryCv;
  bool queried = false;
  std::atomic<bool> copyDone{false};
  EXPECT_CALL(
      *cudaApi_,
      memcpyAsync(
          destination.data(),
          ::testing::_,
          kLen,
          kMockMemcpyH2D,
          static_cast<MockStream>(callerStream)))
      .WillOnce([&](void* dst, const void* src, size_t len, auto, auto) {
        std::memcpy(dst, src, len);
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
        {
          std::lock_guard<std::mutex> lk(queryMu);
          queried = true;
        }
        queryCv.notify_all();
        return copyDone.load(std::memory_order_acquire);
      });
  EXPECT_CALL(*cudaApi_, eventDestroy(::testing::_))
      .WillOnce(::testing::Return(Ok()));
  EXPECT_CALL(*cudaApi_, streamSynchronize(::testing::_)).Times(0);

  handleFrame(bytes, std::move(slab));
  {
    std::unique_lock<std::mutex> lk(queryMu);
    ASSERT_TRUE(queryCv.wait_for(
        lk, std::chrono::seconds(5), [&]() { return queried; }));
  }

  auto onlyFreeSlab = pool->tryAcquire(/*allowReserved=*/true);
  ASSERT_TRUE(onlyFreeSlab);
  EXPECT_FALSE(pool->tryAcquire(/*allowReserved=*/true));
  EXPECT_EQ(
      future.wait_for(std::chrono::milliseconds(0)),
      std::future_status::timeout);

  copyDone.store(true, std::memory_order_release);
  ASSERT_EQ(
      future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_TRUE(future.get().hasValue());
  onlyFreeSlab.reset();
  EXPECT_TRUE(pool->tryAcquire(/*allowReserved=*/true));
}

TEST_F(
    TcpReceivePoolTest,
    EventRecordFailureCompletesGetWhenStreamSynchronizationSucceeds) {
  makeTransport();
  auto pool = createReceivePool();
  ASSERT_NE(pool, nullptr);
  auto slab = pool->tryAcquire(/*allowReserved=*/true);
  ASSERT_TRUE(slab);
  auto frame = readReplyFrame(/*reqId=*/1, kLen);
  ASSERT_LE(frame.size(), slab.capacity());
  std::memcpy(slab.data(), frame.data(), frame.size());
  const auto bytes = std::span<const uint8_t>{slab.data(), frame.size()};

  std::vector<uint8_t> destination(kLen);
  auto* const callerStream = reinterpret_cast<void*>(0xCA11);
  auto future = postVramRead(destination.data(), callerStream);

  EXPECT_CALL(
      *cudaApi_,
      memcpyAsync(
          destination.data(),
          ::testing::_,
          kLen,
          kMockMemcpyH2D,
          static_cast<MockStream>(callerStream)))
      .WillOnce([&](void* dst, const void* src, size_t len, auto, auto) {
        std::memcpy(dst, src, len);
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
      .WillOnce(
          ::testing::Return(
              Err(ErrCode::DriverError, "test: event record failed")));
  EXPECT_CALL(
      *cudaApi_, streamSynchronize(static_cast<MockStream>(callerStream)))
      .WillOnce(::testing::Return(Ok()));
  EXPECT_CALL(*cudaApi_, eventDestroy(::testing::_))
      .WillOnce(::testing::Return(Ok()));
  EXPECT_CALL(*cudaApi_, eventQuery(::testing::_)).Times(0);

  handleFrame(bytes, std::move(slab));

  ASSERT_EQ(
      future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_TRUE(future.get().hasValue());
  EXPECT_TRUE(
      std::all_of(destination.begin(), destination.end(), [](uint8_t b) {
        return b == uint8_t{0xCD};
      }));
  auto returnedSlabs = pool->acquire(2);
  EXPECT_TRUE(returnedSlabs.hasValue());
}

TEST_F(
    TcpReceivePoolTest,
    EventQueryFailureCompletesGetWhenStreamSynchronizationSucceeds) {
  makeTransport();
  auto pool = createReceivePool();
  ASSERT_NE(pool, nullptr);
  auto slab = pool->tryAcquire(/*allowReserved=*/true);
  ASSERT_TRUE(slab);
  auto frame = readReplyFrame(/*reqId=*/1, kLen);
  ASSERT_LE(frame.size(), slab.capacity());
  std::memcpy(slab.data(), frame.data(), frame.size());
  const auto bytes = std::span<const uint8_t>{slab.data(), frame.size()};

  std::vector<uint8_t> destination(kLen);
  auto* const callerStream = reinterpret_cast<void*>(0xCA11);
  auto future = postVramRead(destination.data(), callerStream);

  EXPECT_CALL(
      *cudaApi_,
      memcpyAsync(
          destination.data(),
          ::testing::_,
          kLen,
          kMockMemcpyH2D,
          static_cast<MockStream>(callerStream)))
      .WillOnce([&](void* dst, const void* src, size_t len, auto, auto) {
        std::memcpy(dst, src, len);
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
      .WillOnce([](auto) -> Result<bool> {
        return Err(ErrCode::DriverError, "test: event query failed");
      });
  EXPECT_CALL(
      *cudaApi_, streamSynchronize(static_cast<MockStream>(callerStream)))
      .WillOnce(::testing::Return(Ok()));
  EXPECT_CALL(*cudaApi_, eventDestroy(::testing::_))
      .WillOnce(::testing::Return(Ok()));

  handleFrame(bytes, std::move(slab));

  ASSERT_EQ(
      future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_TRUE(future.get().hasValue());
  EXPECT_TRUE(
      std::all_of(destination.begin(), destination.end(), [](uint8_t b) {
        return b == uint8_t{0xCD};
      }));
  auto returnedSlabs = pool->acquire(2);
  EXPECT_TRUE(returnedSlabs.hasValue());
}

TEST_F(TcpReceivePoolTest, VectorBackedVramReadReplyRemainsSynchronous) {
  makeTransport();
  auto frame = readReplyFrame(/*reqId=*/1, kLen);
  std::vector<uint8_t> destination(kLen);
  auto* const callerStream = reinterpret_cast<void*>(0xCA11);
  auto future = postVramRead(destination.data(), callerStream);

  EXPECT_CALL(*cudaApi_, eventCreate(::testing::_)).Times(0);
  EXPECT_CALL(
      *cudaApi_,
      memcpyAsync(
          destination.data(),
          ::testing::_,
          kLen,
          kMockMemcpyH2D,
          static_cast<MockStream>(callerStream)))
      .WillOnce([&](void* dst, const void* src, size_t len, auto, auto) {
        std::memcpy(dst, src, len);
        return Ok();
      });
  EXPECT_CALL(
      *cudaApi_, streamSynchronize(static_cast<MockStream>(callerStream)))
      .WillOnce(::testing::Return(Ok()));

  handleFrame(frame, {});

  ASSERT_EQ(
      future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_TRUE(future.get().hasValue());
  EXPECT_EQ(destination, std::vector<uint8_t>(kLen, uint8_t{0xCD}));
}

} // namespace uniflow
