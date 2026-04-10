// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/Segment.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <gtest/gtest.h>

namespace uniflow {

class SegmentTest : public ::testing::Test {
 protected:
  template <typename TSegment>
  TSegment createSegment(
      void* buf,
      size_t len,
      MemoryType memType = MemoryType::DRAM,
      int deviceId = -1) {
    return TSegment(buf, len, memType, deviceId);
  }

  using remoteHandleT = RemoteRegisteredSegment::remoteHandleT;
  template <typename... TArgs>
  Result<RemoteRegisteredSegment> createRemoteRegisteredSegment(
      TArgs&&... args) {
    return RemoteRegisteredSegment::from(std::forward<TArgs>(args)...);
  }

  static void addHandle(
      RegisteredSegment& seg,
      std::unique_ptr<RegistrationHandle> handle) {
    seg.handles_.push_back(std::move(handle));
  }

  static const auto& remoteHandles(const RemoteRegisteredSegment& seg) {
    return seg.handles_;
  }
};

template <typename T>
class SegmentTypedTest : public SegmentTest {};

using SegmentTypes =
    ::testing::Types<Segment, RegisteredSegment, RemoteRegisteredSegment>;
TYPED_TEST_SUITE(SegmentTypedTest, SegmentTypes);

// --- Span tests ---

TYPED_TEST(SegmentTypedTest, FullSpan) {
  uint8_t buf[64];
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  auto s = seg.span();
  EXPECT_EQ(s.data(), buf);
  EXPECT_EQ(s.size(), sizeof(buf));
}

TYPED_TEST(SegmentTypedTest, SubSpanWithOffsetAndLength) {
  uint8_t buf[64];
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  auto s = seg.span(10, 20);
  EXPECT_EQ(s.data(), buf + 10);
  EXPECT_EQ(s.size(), 20u);
}

TYPED_TEST(SegmentTypedTest, SubSpanFromPointerAndLength) {
  uint8_t buf[64];
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  auto s = seg.span(buf + 5, 10);
  EXPECT_EQ(s.data(), buf + 5);
  EXPECT_EQ(s.size(), 10u);
}

TYPED_TEST(SegmentTypedTest, ZeroLengthSpanAtStart) {
  uint8_t buf[64];
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  auto s = seg.span(size_t{0}, size_t{0});
  EXPECT_EQ(s.data(), buf);
  EXPECT_EQ(s.size(), 0u);
}

TYPED_TEST(SegmentTypedTest, ZeroLengthSpanAtEnd) {
  uint8_t buf[64];
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  auto s = seg.span(64, 0);
  EXPECT_EQ(s.data(), buf + 64);
  EXPECT_EQ(s.size(), 0u);
}

TYPED_TEST(SegmentTypedTest, SpanCoversEntireBuffer) {
  uint8_t buf[64];
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  auto s = seg.span(size_t{0}, size_t{64});
  EXPECT_EQ(s.data(), buf);
  EXPECT_EQ(s.size(), 64u);
}

TYPED_TEST(SegmentTypedTest, MutableData) {
  uint8_t buf[4] = {0, 0, 0, 0};
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  auto s = seg.span();
  std::memset(s.mutable_data(), 0xAB, s.size());
  EXPECT_EQ(buf[0], 0xAB);
  EXPECT_EQ(buf[3], 0xAB);
}

// --- Bounds checking ---

TYPED_TEST(SegmentTypedTest, OffsetBeyondLengthThrows) {
  uint8_t buf[64];
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  EXPECT_THROW(seg.span(65, 0), std::invalid_argument);
}

TYPED_TEST(SegmentTypedTest, LengthExceedsBoundsThrows) {
  uint8_t buf[64];
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  EXPECT_THROW(seg.span(60, 10), std::invalid_argument);
}

TYPED_TEST(SegmentTypedTest, OffsetPlusLengthOverflowThrows) {
  uint8_t buf[64];
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  EXPECT_THROW(seg.span(1, SIZE_MAX), std::invalid_argument);
}

TYPED_TEST(SegmentTypedTest, PointerSpanOutOfBoundsThrows) {
  uint8_t buf[64];
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  EXPECT_THROW(seg.span(buf + 60, 10), std::invalid_argument);
}

// --- Zero-length buffer ---

TYPED_TEST(SegmentTypedTest, ZeroLengthBuffer) {
  uint8_t buf[1];
  auto seg = this->template createSegment<TypeParam>(buf, 0);
  auto s = seg.span();
  EXPECT_EQ(s.size(), 0u);
}

TYPED_TEST(SegmentTypedTest, ZeroLengthBufferRejectsNonZeroSpan) {
  uint8_t buf[1];
  auto seg = this->template createSegment<TypeParam>(buf, 0);
  EXPECT_THROW(seg.span(static_cast<size_t>(0), 1), std::invalid_argument);
}

// --- Span survives after segment is moved ---

TYPED_TEST(SegmentTypedTest, SpanValidAfterSegmentMoved) {
  uint8_t buf[64];
  std::memset(buf, 0x42, sizeof(buf));
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  auto s = seg.span(10, 20);

  // Move the segment — span should still be valid since it captures buf/len
  [[maybe_unused]] TypeParam moved = std::move(seg);

  EXPECT_EQ(s.data(), buf + 10);
  EXPECT_EQ(s.size(), 20u);
  EXPECT_EQ(static_cast<const uint8_t*>(s.data())[0], 0x42);
}

// --- memType ---

TYPED_TEST(SegmentTypedTest, GetMemTypeDefault) {
  uint8_t buf[4];
  auto seg = this->template createSegment<TypeParam>(buf, sizeof(buf));
  EXPECT_EQ(seg.memType(), MemoryType::DRAM);
}

TYPED_TEST(SegmentTypedTest, GetMemTypeVRAM) {
  uint8_t buf[4];
  auto seg = this->template createSegment<TypeParam>(
      buf, sizeof(buf), MemoryType::VRAM, 0);
  EXPECT_EQ(seg.memType(), MemoryType::VRAM);
}

// --- exportId / RemoteRegisteredSegment::from tests ---

/// Simple test handle for exportId/from round-trip testing.
class TestRegistrationHandle : public RegistrationHandle {
 public:
  explicit TestRegistrationHandle(TransportType type, std::vector<uint8_t> data)
      : type_(type), data_(std::move(data)) {}

  TransportType transportType() const noexcept override {
    return type_;
  }
  std::vector<uint8_t> serialize() const override {
    return data_;
  }

 private:
  TransportType type_;
  std::vector<uint8_t> data_;
};

class TestRemoteRegistrationHandle : public RemoteRegistrationHandle {
 public:
  explicit TestRemoteRegistrationHandle(
      TransportType type,
      std::vector<uint8_t> data)
      : type_(type), data_(std::move(data)) {}

  TransportType transportType() const noexcept override {
    return type_;
  }
  const std::vector<uint8_t>& data() const {
    return data_;
  }

 private:
  TransportType type_;
  std::vector<uint8_t> data_;
};

class ExportImportTest : public SegmentTest {};

TEST_F(ExportImportTest, RoundTripSingleHandle) {
  uint8_t buf[256];
  auto seg = createSegment<RegisteredSegment>(buf, sizeof(buf));
  addHandle(
      seg,
      std::make_unique<TestRegistrationHandle>(
          TransportType::RDMA, std::vector<uint8_t>{0x10, 0x20, 0x30}));

  auto exportResult = seg.exportId();
  ASSERT_TRUE(exportResult.hasValue());
  auto& exported = exportResult.value();
  ASSERT_FALSE(exported.empty());

  auto result = createRemoteRegisteredSegment(
      exported,
      [](TransportType type,
         size_t,
         std::span<const uint8_t> payload) -> remoteHandleT {
        return std::make_unique<TestRemoteRegistrationHandle>(
            type, std::vector<uint8_t>(payload.begin(), payload.end()));
      });

  ASSERT_TRUE(result.hasValue());
  auto& remote = result.value();
  EXPECT_EQ(remote.data(), buf);
  EXPECT_EQ(remote.len(), sizeof(buf));
  EXPECT_EQ(remote.memType(), MemoryType::DRAM);
  ASSERT_EQ(remoteHandles(remote).size(), 1u);
  EXPECT_EQ(remoteHandles(remote)[0]->transportType(), TransportType::RDMA);
  auto* rh = dynamic_cast<TestRemoteRegistrationHandle*>(
      remoteHandles(remote)[0].get());
  ASSERT_NE(rh, nullptr);
  const std::vector<uint8_t> expectedPayload{0x10, 0x20, 0x30};
  EXPECT_EQ(rh->data(), expectedPayload);
}

TEST_F(ExportImportTest, RoundTripMultipleHandles) {
  uint8_t buf[128];
  auto seg =
      createSegment<RegisteredSegment>(buf, sizeof(buf), MemoryType::VRAM, 3);
  addHandle(
      seg,
      std::make_unique<TestRegistrationHandle>(
          TransportType::RDMA, std::vector<uint8_t>{0xAA}));
  addHandle(
      seg,
      std::make_unique<TestRegistrationHandle>(
          TransportType::NVLink, std::vector<uint8_t>{0xBB, 0xCC}));

  auto exportResult = seg.exportId();
  ASSERT_TRUE(exportResult.hasValue());
  auto& exported = exportResult.value();

  auto result = createRemoteRegisteredSegment(
      exported,
      [](TransportType type,
         size_t,
         std::span<const uint8_t> payload) -> remoteHandleT {
        return std::make_unique<TestRemoteRegistrationHandle>(
            type, std::vector<uint8_t>(payload.begin(), payload.end()));
      });

  ASSERT_TRUE(result.hasValue());
  auto& remote = result.value();
  EXPECT_EQ(remote.data(), buf);
  EXPECT_EQ(remote.len(), sizeof(buf));
  EXPECT_EQ(remote.memType(), MemoryType::VRAM);
  EXPECT_EQ(remote.deviceId(), 3);
  ASSERT_EQ(remoteHandles(remote).size(), 2u);
  EXPECT_EQ(remoteHandles(remote)[0]->transportType(), TransportType::RDMA);
  EXPECT_EQ(remoteHandles(remote)[1]->transportType(), TransportType::NVLink);
}

TEST_F(ExportImportTest, RoundTripNoHandles) {
  uint8_t buf[64];
  auto seg = createSegment<RegisteredSegment>(buf, sizeof(buf));
  // No handles added.

  auto result = seg.exportId();
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(ExportImportTest, ExportIdRejectsEmptyHandleData) {
  uint8_t buf[64];
  auto seg = createSegment<RegisteredSegment>(buf, sizeof(buf));
  addHandle(
      seg,
      std::make_unique<TestRegistrationHandle>(
          TransportType::TCP, std::vector<uint8_t>{}));

  auto result = seg.exportId();
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(ExportImportTest, FromRejectsTooShort) {
  std::vector<uint8_t> tooShort(10, 0);
  auto result = createRemoteRegisteredSegment(
      tooShort,
      [](TransportType, size_t, std::span<const uint8_t>) -> remoteHandleT {
        return Err(ErrCode::InvalidArgument, "should not be called");
      });

  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(ExportImportTest, FromRejectsBadVersion) {
  uint8_t buf[64];
  auto seg = createSegment<RegisteredSegment>(buf, sizeof(buf));
  addHandle(
      seg,
      std::make_unique<TestRegistrationHandle>(
          TransportType::RDMA, std::vector<uint8_t>{0x01}));
  auto exportResult = seg.exportId();
  ASSERT_TRUE(exportResult.hasValue());
  auto exported = std::move(exportResult).value();
  exported[0] = 0xFF; // corrupt version

  auto result = createRemoteRegisteredSegment(
      exported,
      [](TransportType, size_t, std::span<const uint8_t>) -> remoteHandleT {
        return Err(ErrCode::InvalidArgument, "should not be called");
      });

  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(ExportImportTest, FromRejectsTruncatedHandleHeader) {
  uint8_t buf[64];
  auto seg = createSegment<RegisteredSegment>(buf, sizeof(buf));
  addHandle(
      seg,
      std::make_unique<TestRegistrationHandle>(
          TransportType::RDMA, std::vector<uint8_t>{0x01, 0x02}));

  auto exportResult = seg.exportId();
  ASSERT_TRUE(exportResult.hasValue());
  auto exported = std::move(exportResult).value();
  // Truncate so handle header is incomplete.
  exported.resize(exported.size() - 5);

  auto result = createRemoteRegisteredSegment(
      exported,
      [](TransportType type,
         size_t,
         std::span<const uint8_t> payload) -> remoteHandleT {
        return std::make_unique<TestRemoteRegistrationHandle>(
            type, std::vector<uint8_t>(payload.begin(), payload.end()));
      });

  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(ExportImportTest, FromRejectsTruncatedHandleData) {
  uint8_t buf[64];
  auto seg = createSegment<RegisteredSegment>(buf, sizeof(buf));
  addHandle(
      seg,
      std::make_unique<TestRegistrationHandle>(
          TransportType::RDMA, std::vector<uint8_t>{0x01, 0x02, 0x03}));

  auto exportResult = seg.exportId();
  ASSERT_TRUE(exportResult.hasValue());
  auto exported = std::move(exportResult).value();
  // Truncate last byte of handle data.
  exported.resize(exported.size() - 1);

  auto result = createRemoteRegisteredSegment(
      exported,
      [](TransportType type,
         size_t,
         std::span<const uint8_t> payload) -> remoteHandleT {
        return std::make_unique<TestRemoteRegistrationHandle>(
            type, std::vector<uint8_t>(payload.begin(), payload.end()));
      });

  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(ExportImportTest, FromPropagatesGetHandleError) {
  uint8_t buf[64];
  auto seg = createSegment<RegisteredSegment>(buf, sizeof(buf));
  addHandle(
      seg,
      std::make_unique<TestRegistrationHandle>(
          TransportType::RDMA, std::vector<uint8_t>{0x01}));

  auto exportResult = seg.exportId();
  ASSERT_TRUE(exportResult.hasValue());
  auto& exported = exportResult.value();

  auto result = createRemoteRegisteredSegment(
      exported,
      [](TransportType, size_t, std::span<const uint8_t>) -> remoteHandleT {
        return Err(ErrCode::NotImplemented, "unsupported transport");
      });

  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::NotImplemented);
}

} // namespace uniflow
