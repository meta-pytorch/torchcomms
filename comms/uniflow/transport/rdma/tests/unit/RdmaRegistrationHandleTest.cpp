// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/rdma/RdmaRegistrationHandle.h"
#include "comms/uniflow/transport/rdma/RdmaTransport.h"

#include "comms/uniflow/drivers/cuda/mock/MockCudaDriverApi.h"
#include "comms/uniflow/drivers/ibverbs/mock/MockIbvApi.h"
#include "comms/uniflow/executor/LockFreeEventBase.h"

#include <unistd.h>
#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace uniflow;
using ::testing::_;
using ::testing::Exactly;
using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Return;

// --- RdmaRegistrationHandle tests ---

class RdmaRegistrationHandleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ibv_ = std::make_shared<NiceMock<MockIbvApi>>();
    ON_CALL(*ibv_, deregMr(_)).WillByDefault(Return(Ok()));
  }

  std::shared_ptr<NiceMock<MockIbvApi>> ibv_;
};

TEST_F(RdmaRegistrationHandleTest, TransportTypeIsRdma) {
  ibv_mr mr{};
  RdmaRegistrationHandle handle({&mr}, ibv_, 42);
  EXPECT_EQ(handle.transportType(), TransportType::RDMA);
}

TEST_F(RdmaRegistrationHandleTest, SingleMrExposesAccessors) {
  char buf[4096]{};
  ibv_mr mr{};
  mr.addr = buf;
  mr.length = sizeof(buf);
  mr.lkey = 0xAAAA;
  mr.rkey = 0xBBBB;
  RdmaRegistrationHandle handle({&mr}, ibv_, 42);

  EXPECT_EQ(handle.lkey(0), 0xAAAAu);
  EXPECT_EQ(handle.rkey(0), 0xBBBBu);
  EXPECT_EQ(handle.numMrs(), 1u);
}

TEST_F(RdmaRegistrationHandleTest, MultipleMrsHaveDifferentKeys) {
  char buf[1024]{};
  ibv_mr mr0{};
  mr0.addr = buf;
  mr0.length = sizeof(buf);
  mr0.lkey = 0x1111;
  mr0.rkey = 0x2222;

  ibv_mr mr1{};
  mr1.addr = buf;
  mr1.length = sizeof(buf);
  mr1.lkey = 0x3333;
  mr1.rkey = 0x4444;

  RdmaRegistrationHandle handle({&mr0, &mr1}, ibv_, 42);
  EXPECT_EQ(handle.numMrs(), 2u);
  EXPECT_EQ(handle.lkey(0), 0x1111u);
  EXPECT_EQ(handle.rkey(0), 0x2222u);
  EXPECT_EQ(handle.lkey(1), 0x3333u);
  EXPECT_EQ(handle.rkey(1), 0x4444u);
}

TEST_F(RdmaRegistrationHandleTest, SerializeSingleMr) {
  char buf[1024]{};
  ibv_mr mr{};
  mr.addr = buf;
  mr.length = sizeof(buf);
  mr.rkey = 0x5555;
  RdmaRegistrationHandle handle({&mr}, ibv_, 42);

  auto serialized = handle.serialize();
  size_t expected =
      RdmaRegistrationHandle::kPayloadHeaderSize + sizeof(uint32_t);
  ASSERT_EQ(serialized.size(), expected);

  RdmaRegistrationHandle::Header header;
  std::memcpy(&header, serialized.data(), sizeof(header));
  EXPECT_EQ(header.domainId, 42u);
  EXPECT_EQ(header.numMrs, 1u);

  uint32_t rkey;
  std::memcpy(
      &rkey,
      serialized.data() + RdmaRegistrationHandle::kPayloadHeaderSize,
      sizeof(rkey));
  EXPECT_EQ(rkey, 0x5555u);
}

TEST_F(RdmaRegistrationHandleTest, SerializeMultipleMrs) {
  char buf[512]{};
  ibv_mr mr0{};
  mr0.addr = buf;
  mr0.length = sizeof(buf);
  mr0.rkey = 0xAA;

  ibv_mr mr1{};
  mr1.addr = buf;
  mr1.length = sizeof(buf);
  mr1.rkey = 0xBB;

  RdmaRegistrationHandle handle({&mr0, &mr1}, ibv_, 42);
  auto serialized = handle.serialize();

  size_t expected =
      RdmaRegistrationHandle::kPayloadHeaderSize + 2 * sizeof(uint32_t);
  ASSERT_EQ(serialized.size(), expected);

  RdmaRegistrationHandle::Header header;
  std::memcpy(&header, serialized.data(), sizeof(header));
  EXPECT_EQ(header.numMrs, 2u);

  uint32_t rkeys[2];
  std::memcpy(
      rkeys,
      serialized.data() + RdmaRegistrationHandle::kPayloadHeaderSize,
      2 * sizeof(uint32_t));
  EXPECT_EQ(rkeys[0], 0xAAu);
  EXPECT_EQ(rkeys[1], 0xBBu);
}

TEST_F(RdmaRegistrationHandleTest, DestructorDeregsAllMrs) {
  ibv_mr mr0{};
  ibv_mr mr1{};
  EXPECT_CALL(*ibv_, deregMr(&mr0)).Times(Exactly(1));
  EXPECT_CALL(*ibv_, deregMr(&mr1)).Times(Exactly(1));
  {
    RdmaRegistrationHandle handle({&mr0, &mr1}, ibv_, 42);
  }
}

// --- RdmaRemoteRegistrationHandle tests ---

TEST(RdmaRemoteRegistrationHandleTest, StoresPerNicRkeys) {
  RdmaRemoteRegistrationHandle handle({0xDEAD, 0xBEEF}, 42);
  EXPECT_EQ(handle.numMrs(), 2u);
  EXPECT_EQ(handle.rkey(0), 0xDEADu);
  EXPECT_EQ(handle.rkey(1), 0xBEEFu);
}

// --- Round-trip tests ---

TEST_F(RdmaRegistrationHandleTest, SerializeDeserializeRoundTrip) {
  char buf[2048]{};
  ibv_mr mr0{};
  mr0.addr = buf;
  mr0.length = sizeof(buf);
  mr0.rkey = 0x1111;

  ibv_mr mr1{};
  mr1.addr = buf;
  mr1.length = sizeof(buf);
  mr1.rkey = 0x2222;

  RdmaRegistrationHandle handle({&mr0, &mr1}, ibv_, 42);
  auto serialized = handle.serialize();

  // Deserialize header.
  RdmaRegistrationHandle::Header header;
  std::memcpy(&header, serialized.data(), sizeof(header));

  // Deserialize rkeys.
  std::vector<uint32_t> rkeys(header.numMrs);
  std::memcpy(
      rkeys.data(),
      serialized.data() + RdmaRegistrationHandle::kPayloadHeaderSize,
      header.numMrs * sizeof(uint32_t));

  // addr and length are provided by the caller, not from the wire.
  RdmaRemoteRegistrationHandle remote(std::move(rkeys), header.domainId);

  EXPECT_EQ(remote.numMrs(), handle.numMrs());
  EXPECT_EQ(remote.rkey(0), handle.rkey(0));
  EXPECT_EQ(remote.rkey(1), handle.rkey(1));
}

// --- Factory registerSegment/importSegment tests ---

class RdmaFactoryRegistrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ibv_ = std::make_shared<NiceMock<MockIbvApi>>();
    cudaDriver_ = std::make_shared<NiceMock<MockCudaDriverApi>>();

    // Factory constructor opens devices — set up mocks for that.
    EXPECT_CALL(*ibv_, getDeviceList(_)).WillOnce([this](int* n) {
      *n = 1;
      return Result<ibv_device**>(fakeDeviceList_);
    });
    EXPECT_CALL(*ibv_, openDevice(&fakeDevice_))
        .WillOnce(Return(Result<ibv_context*>(&fakeCtx_)));
    EXPECT_CALL(*ibv_, getDeviceName(&fakeDevice_))
        .WillOnce(Return(Result<const char*>("mlx5_0")));
    EXPECT_CALL(*ibv_, queryDevice(&fakeCtx_, _))
        .WillOnce([](ibv_context*, ibv_device_attr* attr) {
          attr->phys_port_cnt = 1;
          return Ok();
        });
    EXPECT_CALL(*ibv_, allocPd(&fakeCtx_))
        .WillOnce(Return(Result<ibv_pd*>(&fakePd_)));
    EXPECT_CALL(*ibv_, isDmaBufSupported(&fakePd_))
        .WillOnce(Return(Result<bool>(true)));
    EXPECT_CALL(*ibv_, queryPort(&fakeCtx_, 1, _))
        .WillRepeatedly([](ibv_context*, uint8_t, ibv_port_attr* attr) {
          std::memset(attr, 0, sizeof(*attr));
          attr->lid = 1;
          attr->active_mtu = IBV_MTU_4096;
          attr->link_layer = IBV_LINK_LAYER_ETHERNET;
          attr->state = IBV_PORT_ACTIVE;
          return Ok();
        });
    EXPECT_CALL(*ibv_, queryGid(&fakeCtx_, 1, 3, _)).WillOnce(Return(Ok()));
    EXPECT_CALL(*ibv_, freeDeviceList(_)).WillOnce(Return(Ok()));
    // Cleanup on destruction.
    EXPECT_CALL(*ibv_, deallocPd(&fakePd_)).WillOnce(Return(Ok()));
    EXPECT_CALL(*ibv_, closeDevice(&fakeCtx_)).WillOnce(Return(Ok()));

    ON_CALL(*ibv_, deregMr(_)).WillByDefault(Return(Ok()));

    factory_ = std::make_unique<RdmaTransportFactory>(
        std::vector<std::string>{"mlx5_0"},
        &evb_,
        RdmaTransportConfig{},
        ibv_,
        cudaDriver_);
  }

  std::shared_ptr<NiceMock<MockIbvApi>> ibv_;
  std::shared_ptr<NiceMock<MockCudaDriverApi>> cudaDriver_;
  uniflow::LockFreeEventBase evb_;
  std::unique_ptr<RdmaTransportFactory> factory_;

  ibv_device fakeDevice_{};
  ibv_device* fakeDeviceList_[2] = {&fakeDevice_, nullptr};
  ibv_context fakeCtx_{};
  ibv_pd fakePd_{};
  ibv_mr fakeMr_{};
};

TEST_F(RdmaFactoryRegistrationTest, RegisterDramSegmentUsesRegMr) {
  char buf[4096]{};
  Segment segment(buf, sizeof(buf), MemoryType::DRAM);

  fakeMr_.addr = buf;
  fakeMr_.length = sizeof(buf);
  fakeMr_.lkey = 0x1111;
  fakeMr_.rkey = 0x2222;

  EXPECT_CALL(*ibv_, regMr(&fakePd_, buf, sizeof(buf), _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr_)));
  // regDmabufMr should NOT be called for DRAM.
  EXPECT_CALL(*ibv_, regDmabufMr(_, _, _, _, _, _)).Times(Exactly(0));

  auto result = factory_->registerSegment(segment);
  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result.value()->transportType(), TransportType::RDMA);
}

TEST_F(RdmaFactoryRegistrationTest, RegisterVramSegmentUsesDmabufMr) {
  char buf[4096]{};
  Segment segment(buf, sizeof(buf), MemoryType::VRAM, 0);

  fakeMr_.addr = buf;
  fakeMr_.length = sizeof(buf);
  fakeMr_.lkey = 0x3333;
  fakeMr_.rkey = 0x4444;

  // Use a real disposable fd to avoid closing an unrelated fd in the test
  // process when FdGuard destructor runs.
  EXPECT_CALL(*cudaDriver_, isDmaBufSupported(0))
      .WillOnce(Return(Result<bool>(true)));
  EXPECT_CALL(*cudaDriver_, cuMemGetHandleForAddressRange(_, _, _, _, _))
      .WillOnce([](void* handle,
                   CUdeviceptr,
                   size_t,
                   CUmemRangeHandleType,
                   unsigned long long) {
        *static_cast<int*>(handle) = ::dup(STDERR_FILENO);
        return Ok();
      });
  EXPECT_CALL(*ibv_, regDmabufMr(&fakePd_, _, sizeof(buf), _, _, _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr_)));
  // regMr should NOT be called for VRAM with DMA-BUF support.
  EXPECT_CALL(*ibv_, regMr(_, _, _, _)).Times(Exactly(0));

  auto result = factory_->registerSegment(segment);
  ASSERT_TRUE(result.hasValue());
}

TEST_F(RdmaFactoryRegistrationTest, VramFallsBackToRegMrWhenDmaBufUnsupported) {
  char buf[4096]{};
  Segment segment(buf, sizeof(buf), MemoryType::VRAM, 0);

  fakeMr_.addr = buf;
  fakeMr_.length = sizeof(buf);
  fakeMr_.lkey = 0x5555;
  fakeMr_.rkey = 0x6666;

  // CUDA reports DMA-BUF not supported → should fall back to regMr.
  EXPECT_CALL(*cudaDriver_, isDmaBufSupported(0))
      .WillOnce(Return(Result<bool>(false)));
  EXPECT_CALL(*cudaDriver_, cuMemGetHandleForAddressRange(_, _, _, _, _))
      .Times(Exactly(0));
  EXPECT_CALL(*ibv_, regDmabufMr(_, _, _, _, _, _)).Times(Exactly(0));
  EXPECT_CALL(*ibv_, regMr(&fakePd_, buf, sizeof(buf), _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr_)));

  auto result = factory_->registerSegment(segment);
  ASSERT_TRUE(result.hasValue());
}

TEST_F(RdmaFactoryRegistrationTest, VramFallsBackToRegMrWhenGetHandleFails) {
  char buf[4096]{};
  Segment segment(buf, sizeof(buf), MemoryType::VRAM, 0);

  fakeMr_.addr = buf;
  fakeMr_.length = sizeof(buf);
  fakeMr_.lkey = 0x7777;
  fakeMr_.rkey = 0x8888;

  // DMA-BUF is supported but cuMemGetHandleForAddressRange fails →
  // should fall back to regMr.
  EXPECT_CALL(*cudaDriver_, isDmaBufSupported(0))
      .WillOnce(Return(Result<bool>(true)));
  EXPECT_CALL(*cudaDriver_, cuMemGetHandleForAddressRange(_, _, _, _, _))
      .WillOnce(Return(Err(ErrCode::DriverError, "simulated failure")));
  EXPECT_CALL(*ibv_, regDmabufMr(_, _, _, _, _, _)).Times(Exactly(0));
  EXPECT_CALL(*ibv_, regMr(&fakePd_, buf, sizeof(buf), _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr_)));

  auto result = factory_->registerSegment(segment);
  ASSERT_TRUE(result.hasValue());
}

TEST_F(RdmaFactoryRegistrationTest, ImportSegmentSucceeds) {
  RdmaRegistrationHandle::Header header{
      .domainId = 99,
      .numMrs = 1,
  };
  uint32_t rkey = 0xAAAA;
  std::vector<uint8_t> data(sizeof(header) + sizeof(rkey));
  std::memcpy(data.data(), &header, sizeof(header));
  std::memcpy(data.data() + sizeof(header), &rkey, sizeof(rkey));

  char fakeBuf[8192]{};
  auto result = factory_->importSegment(sizeof(fakeBuf), data);
  ASSERT_TRUE(result.hasValue());

  auto* remote =
      dynamic_cast<RdmaRemoteRegistrationHandle*>(result.value().get());
  ASSERT_NE(remote, nullptr);
  EXPECT_EQ(remote->rkey(0), 0xAAAAu);
}

TEST_F(RdmaFactoryRegistrationTest, ImportSegmentRejectsTooSmallPayload) {
  std::vector<uint8_t> tooSmall(4);
  char fakeBuf[4096]{};
  auto result = factory_->importSegment(sizeof(fakeBuf), tooSmall);
  EXPECT_TRUE(result.hasError());
}

TEST_F(RdmaFactoryRegistrationTest, ImportSegmentRejectsSizeMismatch) {
  // Header says numMrs=3 but payload only contains 1 rkey.
  RdmaRegistrationHandle::Header header{
      .domainId = 99,
      .numMrs = 3,
  };
  uint32_t rkey = 0xAAAA;
  std::vector<uint8_t> data(sizeof(header) + sizeof(rkey));
  std::memcpy(data.data(), &header, sizeof(header));
  std::memcpy(data.data() + sizeof(header), &rkey, sizeof(rkey));

  auto result = factory_->importSegment(4096, data);
  EXPECT_TRUE(result.hasError());
}

TEST_F(RdmaFactoryRegistrationTest, ImportSegmentWithZeroMrs) {
  RdmaRegistrationHandle::Header header{
      .domainId = 99,
      .numMrs = 0,
  };
  std::vector<uint8_t> data(sizeof(header));
  std::memcpy(data.data(), &header, sizeof(header));

  auto result = factory_->importSegment(4096, data);
  EXPECT_TRUE(result.hasError());
}

TEST_F(RdmaFactoryRegistrationTest, RegisterAndImportRoundTrip) {
  char buf[2048]{};
  Segment segment(buf, sizeof(buf), MemoryType::DRAM);

  fakeMr_.addr = buf;
  fakeMr_.length = sizeof(buf);
  fakeMr_.lkey = 0x5555;
  fakeMr_.rkey = 0x6666;

  ON_CALL(*ibv_, regMr(_, _, _, _))
      .WillByDefault(Return(Result<ibv_mr*>(&fakeMr_)));

  auto regResult = factory_->registerSegment(segment);
  ASSERT_TRUE(regResult.hasValue());

  auto serialized = regResult.value()->serialize();

  auto importResult = factory_->importSegment(sizeof(buf), serialized);
  ASSERT_TRUE(importResult.hasValue());

  auto* remote =
      dynamic_cast<RdmaRemoteRegistrationHandle*>(importResult.value().get());
  ASSERT_NE(remote, nullptr);

  auto* local = dynamic_cast<RdmaRegistrationHandle*>(regResult.value().get());
  ASSERT_NE(local, nullptr);
  EXPECT_EQ(remote->rkey(0), local->rkey(0));
  EXPECT_EQ(remote->domainId(), local->domainId());
}

// --- Multi-NIC partial failure tests ---

class RdmaFactoryMultiNicTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ibv_ = std::make_shared<NiceMock<MockIbvApi>>();
    cudaDriver_ = std::make_shared<NiceMock<MockCudaDriverApi>>();

    // Set up 2 devices for the factory constructor.
    EXPECT_CALL(*ibv_, getDeviceList(_)).WillOnce([this](int* n) {
      *n = 2;
      return Result<ibv_device**>(fakeDeviceList_);
    });
    EXPECT_CALL(*ibv_, openDevice(&fakeDevice0_))
        .WillOnce(Return(Result<ibv_context*>(&fakeCtx0_)));
    EXPECT_CALL(*ibv_, openDevice(&fakeDevice1_))
        .WillOnce(Return(Result<ibv_context*>(&fakeCtx1_)));
    EXPECT_CALL(*ibv_, getDeviceName(&fakeDevice0_))
        .WillRepeatedly(Return(Result<const char*>("mlx5_0")));
    EXPECT_CALL(*ibv_, getDeviceName(&fakeDevice1_))
        .WillRepeatedly(Return(Result<const char*>("mlx5_1")));
    EXPECT_CALL(*ibv_, queryDevice(_, _))
        .WillRepeatedly([](ibv_context*, ibv_device_attr* attr) {
          attr->phys_port_cnt = 1;
          return Ok();
        });
    EXPECT_CALL(*ibv_, allocPd(&fakeCtx0_))
        .WillOnce(Return(Result<ibv_pd*>(&fakePd0_)));
    EXPECT_CALL(*ibv_, allocPd(&fakeCtx1_))
        .WillOnce(Return(Result<ibv_pd*>(&fakePd1_)));
    EXPECT_CALL(*ibv_, isDmaBufSupported(&fakePd0_))
        .WillOnce(Return(Result<bool>(true)));
    EXPECT_CALL(*ibv_, isDmaBufSupported(&fakePd1_))
        .WillOnce(Return(Result<bool>(true)));
    EXPECT_CALL(*ibv_, queryPort(_, 1, _))
        .WillRepeatedly([](ibv_context*, uint8_t, ibv_port_attr* attr) {
          std::memset(attr, 0, sizeof(*attr));
          attr->lid = 1;
          attr->active_mtu = IBV_MTU_4096;
          attr->link_layer = IBV_LINK_LAYER_ETHERNET;
          attr->state = IBV_PORT_ACTIVE;
          return Ok();
        });
    EXPECT_CALL(*ibv_, queryGid(_, 1, 3, _)).WillRepeatedly(Return(Ok()));
    EXPECT_CALL(*ibv_, freeDeviceList(_)).WillOnce(Return(Ok()));
    EXPECT_CALL(*ibv_, deallocPd(&fakePd0_)).WillOnce(Return(Ok()));
    EXPECT_CALL(*ibv_, deallocPd(&fakePd1_)).WillOnce(Return(Ok()));
    EXPECT_CALL(*ibv_, closeDevice(&fakeCtx0_)).WillOnce(Return(Ok()));
    EXPECT_CALL(*ibv_, closeDevice(&fakeCtx1_)).WillOnce(Return(Ok()));

    ON_CALL(*ibv_, deregMr(_)).WillByDefault(Return(Ok()));

    factory_ = std::make_unique<RdmaTransportFactory>(
        std::vector<std::string>{"mlx5_0", "mlx5_1"},
        &evb_,
        RdmaTransportConfig{},
        ibv_,
        cudaDriver_);
  }

  std::shared_ptr<NiceMock<MockIbvApi>> ibv_;
  std::shared_ptr<NiceMock<MockCudaDriverApi>> cudaDriver_;
  uniflow::LockFreeEventBase evb_;
  std::unique_ptr<RdmaTransportFactory> factory_;

  ibv_device fakeDevice0_{};
  ibv_device fakeDevice1_{};
  ibv_device* fakeDeviceList_[3] = {&fakeDevice0_, &fakeDevice1_, nullptr};
  ibv_context fakeCtx0_{};
  ibv_context fakeCtx1_{};
  ibv_pd fakePd0_{};
  ibv_pd fakePd1_{};
};

TEST_F(RdmaFactoryMultiNicTest, PartialRegFailureCleansUpFirstMr) {
  char buf[4096]{};
  Segment segment(buf, sizeof(buf), MemoryType::DRAM);

  ibv_mr fakeMr0{};
  fakeMr0.addr = buf;
  fakeMr0.length = sizeof(buf);
  fakeMr0.lkey = 0x1111;
  fakeMr0.rkey = 0x2222;

  // First NIC succeeds, second NIC fails.
  EXPECT_CALL(*ibv_, regMr(&fakePd0_, buf, sizeof(buf), _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr0)));
  EXPECT_CALL(*ibv_, regMr(&fakePd1_, buf, sizeof(buf), _))
      .WillOnce(
          Return(Err(ErrCode::DriverError, "simulated NIC1 reg failure")));

  // The first MR must be cleaned up on partial failure.
  EXPECT_CALL(*ibv_, deregMr(&fakeMr0)).Times(Exactly(1));

  auto result = factory_->registerSegment(segment);
  EXPECT_TRUE(result.hasError());
}

TEST_F(RdmaFactoryMultiNicTest, AllNicsRegisterSuccessfully) {
  char buf[4096]{};
  Segment segment(buf, sizeof(buf), MemoryType::DRAM);

  ibv_mr fakeMr0{};
  fakeMr0.addr = buf;
  fakeMr0.length = sizeof(buf);
  fakeMr0.lkey = 0x1111;
  fakeMr0.rkey = 0x2222;

  ibv_mr fakeMr1{};
  fakeMr1.addr = buf;
  fakeMr1.length = sizeof(buf);
  fakeMr1.lkey = 0x3333;
  fakeMr1.rkey = 0x4444;

  EXPECT_CALL(*ibv_, regMr(&fakePd0_, buf, sizeof(buf), _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr0)));
  EXPECT_CALL(*ibv_, regMr(&fakePd1_, buf, sizeof(buf), _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr1)));

  auto result = factory_->registerSegment(segment);
  ASSERT_TRUE(result.hasValue());

  auto* handle = dynamic_cast<RdmaRegistrationHandle*>(result.value().get());
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->numMrs(), 2u);
  EXPECT_EQ(handle->rkey(0), 0x2222u);
  EXPECT_EQ(handle->rkey(1), 0x4444u);
}
