// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/rdma/RdmaTransport.h"
#include "comms/uniflow/transport/rdma/RdmaRegistrationHandle.h"

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/drivers/ibverbs/mock/MockIbvApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"

#include <cstring>
#include <queue>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace uniflow;
using ::testing::_;
using ::testing::Return;

// Use default config values for test constants.
static const RdmaTransportConfig kDefaultConfig{};
// must match RdmaTransportConfig default
constexpr size_t kChunkSize = 512 * 1024;
// must match RdmaTransportConfig default
constexpr size_t kDefaultMaxWr = 128;

namespace uniflow {

/// Friend-class wrapper to construct RegisteredSegment /
/// RemoteRegisteredSegment with handles for testing.
class SegmentTest {
 public:
  static RegisteredSegment makeRegistered(
      Segment& segment,
      std::unique_ptr<RegistrationHandle> handle) {
    RegisteredSegment reg(segment);
    reg.handles_.push_back(std::move(handle));
    return reg;
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

} // namespace uniflow

// --- Serialization tests ---

TEST(RdmaTransportInfoTest, SerializeDeserializeRoundTrip) {
  RdmaTransportInfo info;
  info.header.version = kRdmaVersion;
  info.header.numQps = 4;
  info.header.numNics = 2;
  info.nicInfos = {
      {.lid = 0x1234,
       .linkLayer = IBV_LINK_LAYER_ETHERNET,
       .mtu = IBV_MTU_4096,
       .gid = {}},
      {.lid = 0x5678,
       .linkLayer = IBV_LINK_LAYER_ETHERNET,
       .mtu = IBV_MTU_4096,
       .gid = {}},
  };
  info.nicInfos[0].gid.global.subnet_prefix = 0xfe80000000000000ULL;
  info.nicInfos[0].gid.global.interface_id = 0x0001020304050607ULL;
  info.nicInfos[1].gid.global.subnet_prefix = 0xfe80000000000000ULL;
  info.nicInfos[1].gid.global.interface_id = 0x0008090a0b0c0d0eULL;
  info.qpInfos = {
      {.qpNum = 100, .psn = 200},
      {.qpNum = 300, .psn = 400},
      {.qpNum = 500, .psn = 600},
      {.qpNum = 700, .psn = 800},
  };
  info.ctrl = {.addr = 0x1000, .length = 64, .rkeys = {0xAA, 0xBB}};
  info.slab = {.addr = 0x2000, .length = 4096, .rkeys = {0xCC, 0xDD}};

  auto data = info.serialize();
  auto result = RdmaTransportInfo::deserialize(data);
  ASSERT_TRUE(result.hasValue());

  auto& out = result.value();
  EXPECT_EQ(out.header.version, kRdmaVersion);
  EXPECT_EQ(out.header.numQps, 4);
  EXPECT_EQ(out.header.numNics, 2);

  ASSERT_EQ(out.nicInfos.size(), 2);
  EXPECT_EQ(out.nicInfos[0].lid, 0x1234);
  EXPECT_EQ(out.nicInfos[1].lid, 0x5678);
  EXPECT_EQ(
      out.nicInfos[0].gid.global.interface_id,
      info.nicInfos[0].gid.global.interface_id);

  ASSERT_EQ(out.qpInfos.size(), 4);
  EXPECT_EQ(out.qpInfos[0].qpNum, 100);
  EXPECT_EQ(out.qpInfos[3].qpNum, 700);

  EXPECT_EQ(out.ctrl.addr, 0x1000);
  EXPECT_EQ(out.ctrl.length, 64);
  ASSERT_EQ(out.ctrl.rkeys.size(), 2);
  EXPECT_EQ(out.ctrl.rkeys[0], 0xAA);
  EXPECT_EQ(out.ctrl.rkeys[1], 0xBB);
  EXPECT_EQ(out.slab.addr, 0x2000);
  EXPECT_EQ(out.slab.length, 4096);
  ASSERT_EQ(out.slab.rkeys.size(), 2);
  EXPECT_EQ(out.slab.rkeys[0], 0xCC);
  EXPECT_EQ(out.slab.rkeys[1], 0xDD);
}

TEST(RdmaTransportInfoTest, DeserializeTooSmallForHeader) {
  TransportInfo data(4, 0);
  auto result = RdmaTransportInfo::deserialize(data);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST(RdmaTransportInfoTest, DeserializeBadVersion) {
  RdmaTransportInfo info;
  info.header.version = 99;
  info.header.numQps = 0;
  info.header.numNics = 0;
  auto data = info.serialize();

  auto result = RdmaTransportInfo::deserialize(data);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST(RdmaTransportInfoTest, DeserializeTruncatedHeader) {
  // Provide fewer bytes than sizeof(Header) — should fail at header check.
  TransportInfo data = {1, 3, 2};

  auto result = RdmaTransportInfo::deserialize(data);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST(RdmaTransportInfoTest, DeserializeTruncatedPayload) {
  // Provide a full header but no NicInfo/QpInfo payload.
  RdmaTransportInfo info;
  info.header.version = 1;
  info.header.numQps = 3;
  info.header.numNics = 2;

  TransportInfo data(sizeof(RdmaTransportInfo::Header));
  std::memcpy(data.data(), &info.header, sizeof(info.header));

  auto result = RdmaTransportInfo::deserialize(data);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST(RdmaTransportInfoTest, SerializeZeroQps) {
  RdmaTransportInfo info;
  info.header.version = 1;
  info.header.numQps = 0;
  info.header.numNics = 1;
  info.nicInfos = {{.lid = 1}};
  info.ctrl.rkeys = {0};
  info.slab.rkeys = {0};

  auto data = info.serialize();
  auto result = RdmaTransportInfo::deserialize(data);
  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result.value().header.numQps, 0);
  EXPECT_TRUE(result.value().qpInfos.empty());
  ASSERT_EQ(result.value().nicInfos.size(), 1);
}

// --- Helpers ---

NicResources makeNic(
    ibv_device* dev,
    ibv_context* ctx,
    ibv_pd* pd,
    std::shared_ptr<testing::NiceMock<MockIbvApi>> mockApi,
    uint16_t lid = 0,
    ibv_gid gid = {},
    uint8_t port = 1) {
  ON_CALL(*mockApi, openDevice(dev))
      .WillByDefault(Return(Result<ibv_context*>(ctx)));
  ON_CALL(*mockApi, allocPd(ctx)).WillByDefault(Return(Result<ibv_pd*>(pd)));
  ON_CALL(*mockApi, isDmaBufSupported(pd))
      .WillByDefault(Return(Result<bool>(false)));
  ON_CALL(*mockApi, queryPort(ctx, port, _))
      .WillByDefault([lid, gid](ibv_context*, uint8_t, ibv_port_attr* attr) {
        attr->lid = lid;
        attr->active_mtu = IBV_MTU_4096;
        attr->link_layer = IBV_LINK_LAYER_ETHERNET;
        attr->state = IBV_PORT_ACTIVE;
        return Ok();
      });
  ON_CALL(*mockApi, queryGid(ctx, port, _, _))
      .WillByDefault([gid](ibv_context*, uint8_t, int, ibv_gid* out) {
        *out = gid;
        return Ok();
      });
  ON_CALL(*mockApi, queryDevice(ctx, _))
      .WillByDefault([port](ibv_context*, ibv_device_attr* attr) {
        attr->phys_port_cnt = port;
        return Ok();
      });
  ON_CALL(*mockApi, deallocPd(pd)).WillByDefault(Return(Ok()));
  ON_CALL(*mockApi, closeDevice(ctx)).WillByDefault(Return(Ok()));

  return NicResources(dev, mockApi, 3, port);
}

// --- Mock-based transport tests ---

class RdmaTransportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mockApi_ = std::make_shared<testing::NiceMock<MockIbvApi>>();
    mockCudaApi_ = std::make_shared<testing::NiceMock<MockCudaApi>>();
    ON_CALL(*mockApi_, regMr(_, _, _, _))
        .WillByDefault(Return(Result<ibv_mr*>(&fakeSlabMr_)));
    ON_CALL(*mockApi_, deregMr(_)).WillByDefault(Return(Ok()));
    ON_CALL(*mockCudaApi_, hostAlloc(_, _))
        .WillByDefault([](size_t size, unsigned int) {
          return Result<void*>(std::calloc(1, size));
        });
    ON_CALL(*mockCudaApi_, hostFree(_)).WillByDefault([](void* ptr) {
      std::free(ptr);
      return Ok();
    });
  }

  std::shared_ptr<testing::NiceMock<MockIbvApi>> mockApi_;
  std::shared_ptr<testing::NiceMock<MockCudaApi>> mockCudaApi_;
  ScopedEventBaseThread evbThread_;

  ibv_device fakeDev0_{};
  ibv_device fakeDev1_{};
  ibv_context fakeCtx0_{.device = &fakeDev0_};
  ibv_context fakeCtx1_{.device = &fakeDev1_};
  ibv_pd fakePd0_{};
  ibv_pd fakePd1_{};
  ibv_cq fakeCq0_{};
  ibv_cq fakeCq1_{};
  ibv_qp fakeQp0_{};
  ibv_qp fakeQp1_{};
  ibv_qp fakeQp2_{};
  ibv_qp fakeQp3_{};
  ibv_mr fakeSlabMr_{};
};

TEST_F(RdmaTransportTest, SingleNicBindCreatesQPs) {
  fakeQp0_.qp_num = 42;
  fakeQp1_.qp_num = 43;

  EXPECT_CALL(*mockApi_, createCq(&fakeCtx0_, _, _, _, _))
      .WillOnce(Return(Result<ibv_cq*>(&fakeCq0_)));
  EXPECT_CALL(*mockApi_, createQp(&fakePd0_, _))
      .WillOnce(Return(Result<ibv_qp*>(&fakeQp0_)))
      .WillOnce(Return(Result<ibv_qp*>(&fakeQp1_)));
  EXPECT_CALL(*mockApi_, modifyQp(_, _, _)).WillRepeatedly(Return(Ok()));

  ibv_gid gid{};
  auto nics = std::make_shared<std::vector<NicResources>>();
  nics->push_back(
      makeNic(&fakeDev0_, &fakeCtx0_, &fakePd0_, mockApi_, 0x1234, gid));
  RdmaTransportConfig config{.numQps = 2};
  RdmaTransport transport(
      mockApi_,
      mockCudaApi_,
      evbThread_.getEventBase(),
      nics,
      0,
      nullptr,
      config);

  auto data = transport.bind();
  ASSERT_FALSE(data.empty());

  auto result = RdmaTransportInfo::deserialize(data);
  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result.value().header.numQps, 2);
  EXPECT_EQ(result.value().qpInfos[0].qpNum, 42);
  EXPECT_EQ(result.value().qpInfos[1].qpNum, 43);
}

TEST_F(RdmaTransportTest, MultiNicBindDistributesQPsRoundRobin) {
  fakeQp0_.qp_num = 10;
  fakeQp1_.qp_num = 20;
  fakeQp2_.qp_num = 30;
  fakeQp3_.qp_num = 40;

  EXPECT_CALL(*mockApi_, createCq(&fakeCtx0_, _, _, _, _))
      .WillOnce(Return(Result<ibv_cq*>(&fakeCq0_)));
  EXPECT_CALL(*mockApi_, createCq(&fakeCtx1_, _, _, _, _))
      .WillOnce(Return(Result<ibv_cq*>(&fakeCq1_)));

  EXPECT_CALL(*mockApi_, createQp(&fakePd0_, _))
      .WillOnce(Return(Result<ibv_qp*>(&fakeQp0_)))
      .WillOnce(Return(Result<ibv_qp*>(&fakeQp2_)));
  EXPECT_CALL(*mockApi_, createQp(&fakePd1_, _))
      .WillOnce(Return(Result<ibv_qp*>(&fakeQp1_)))
      .WillOnce(Return(Result<ibv_qp*>(&fakeQp3_)));

  EXPECT_CALL(*mockApi_, modifyQp(_, _, _)).WillRepeatedly(Return(Ok()));

  ibv_gid gid0{}, gid1{};
  auto nics = std::make_shared<std::vector<NicResources>>();
  nics->push_back(
      makeNic(&fakeDev0_, &fakeCtx0_, &fakePd0_, mockApi_, 0x1111, gid0));
  nics->push_back(
      makeNic(&fakeDev1_, &fakeCtx1_, &fakePd1_, mockApi_, 0x2222, gid1));
  RdmaTransportConfig config{.numQps = 4};
  RdmaTransport transport(
      mockApi_,
      mockCudaApi_,
      evbThread_.getEventBase(),
      nics,
      0,
      nullptr,
      config);

  auto data = transport.bind();
  ASSERT_FALSE(data.empty());

  auto result = RdmaTransportInfo::deserialize(data);
  ASSERT_TRUE(result.hasValue());

  auto& out = result.value();
  EXPECT_EQ(out.header.numQps, 4);
  EXPECT_EQ(out.header.numNics, 2);
  ASSERT_EQ(out.nicInfos.size(), 2);
  EXPECT_EQ(out.nicInfos[0].lid, 0x1111);
  EXPECT_EQ(out.nicInfos[1].lid, 0x2222);
  EXPECT_EQ(out.qpInfos[0].qpNum, 10);
  EXPECT_EQ(out.qpInfos[1].qpNum, 20);
  EXPECT_EQ(out.qpInfos[2].qpNum, 30);
  EXPECT_EQ(out.qpInfos[3].qpNum, 40);
}

TEST_F(RdmaTransportTest, BindReturnsEmptyOnCqFailure) {
  EXPECT_CALL(*mockApi_, createCq(_, _, _, _, _))
      .WillOnce(Return(Result<ibv_cq*>(ErrCode::ResourceExhausted)));

  ibv_gid gid{};
  auto nics = std::make_shared<std::vector<NicResources>>();
  nics->push_back(makeNic(&fakeDev0_, &fakeCtx0_, &fakePd0_, mockApi_, 0, gid));
  RdmaTransport transport(
      mockApi_, mockCudaApi_, evbThread_.getEventBase(), nics, 0);

  auto data = transport.bind();
  EXPECT_TRUE(data.empty());
  EXPECT_EQ(transport.state(), TransportState::Error);
}

TEST_F(RdmaTransportTest, BindReturnsEmptyOnQpFailure) {
  EXPECT_CALL(*mockApi_, createCq(_, _, _, _, _))
      .WillOnce(Return(Result<ibv_cq*>(&fakeCq0_)));
  EXPECT_CALL(*mockApi_, createQp(_, _))
      .WillOnce(Return(Result<ibv_qp*>(ErrCode::ResourceExhausted)));

  ibv_gid gid{};
  auto nics = std::make_shared<std::vector<NicResources>>();
  nics->push_back(makeNic(&fakeDev0_, &fakeCtx0_, &fakePd0_, mockApi_, 0, gid));
  RdmaTransport transport(
      mockApi_, mockCudaApi_, evbThread_.getEventBase(), nics, 0);

  auto data = transport.bind();
  EXPECT_TRUE(data.empty());
  EXPECT_EQ(transport.state(), TransportState::Error);
}

TEST_F(RdmaTransportTest, ConnectTransitionsQPs) {
  fakeQp0_.qp_num = 100;

  EXPECT_CALL(*mockApi_, createCq(_, _, _, _, _))
      .WillOnce(Return(Result<ibv_cq*>(&fakeCq0_)));
  EXPECT_CALL(*mockApi_, createQp(_, _))
      .WillOnce(Return(Result<ibv_qp*>(&fakeQp0_)));
  EXPECT_CALL(*mockApi_, modifyQp(_, _, _)).WillRepeatedly(Return(Ok()));

  ibv_gid gid{};
  auto nics = std::make_shared<std::vector<NicResources>>();
  nics->push_back(
      makeNic(&fakeDev0_, &fakeCtx0_, &fakePd0_, mockApi_, 0x1234, gid));
  RdmaTransport transport(
      mockApi_, mockCudaApi_, evbThread_.getEventBase(), nics, 0);
  transport.bind();

  RdmaTransportInfo remoteInfo;
  remoteInfo.header.version = 1;
  remoteInfo.header.numQps = 1;
  remoteInfo.header.numNics = 1;
  remoteInfo.nicInfos = {{.lid = 0x5678}};
  remoteInfo.qpInfos = {{.qpNum = 200, .psn = 300}};
  remoteInfo.ctrl.rkeys = {0};
  remoteInfo.slab.rkeys = {0};

  auto status = transport.connect(remoteInfo.serialize());
  ASSERT_FALSE(status.hasError());
  EXPECT_EQ(transport.state(), TransportState::Connected);
}

TEST_F(RdmaTransportTest, ConnectRejectsQpCountMismatch) {
  fakeQp0_.qp_num = 100;

  EXPECT_CALL(*mockApi_, createCq(_, _, _, _, _))
      .WillOnce(Return(Result<ibv_cq*>(&fakeCq0_)));
  EXPECT_CALL(*mockApi_, createQp(_, _))
      .WillOnce(Return(Result<ibv_qp*>(&fakeQp0_)));
  EXPECT_CALL(*mockApi_, modifyQp(_, _, _)).WillRepeatedly(Return(Ok()));

  ibv_gid gid{};
  auto nics = std::make_shared<std::vector<NicResources>>();
  nics->push_back(makeNic(&fakeDev0_, &fakeCtx0_, &fakePd0_, mockApi_, 0, gid));
  RdmaTransport transport(
      mockApi_, mockCudaApi_, evbThread_.getEventBase(), nics, 0);
  transport.bind();

  RdmaTransportInfo remoteInfo;
  remoteInfo.header.version = 1;
  remoteInfo.header.numQps = 2;
  remoteInfo.header.numNics = 1;
  remoteInfo.nicInfos = {{.lid = 1}};
  remoteInfo.qpInfos = {{.qpNum = 200, .psn = 300}, {.qpNum = 201, .psn = 301}};
  remoteInfo.ctrl.rkeys = {0};
  remoteInfo.slab.rkeys = {0};

  auto status = transport.connect(remoteInfo.serialize());
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_F(RdmaTransportTest, ConnectWithoutBindFails) {
  ibv_gid gid{};
  auto nics = std::make_shared<std::vector<NicResources>>();
  nics->push_back(makeNic(&fakeDev0_, &fakeCtx0_, &fakePd0_, mockApi_, 0, gid));
  RdmaTransport transport(
      mockApi_, mockCudaApi_, evbThread_.getEventBase(), nics, 0);

  // Don't call bind() - go straight to connect()
  RdmaTransportInfo remoteInfo;
  remoteInfo.header.version = 1;
  remoteInfo.header.numQps = 1;
  remoteInfo.header.numNics = 1;
  remoteInfo.nicInfos = {{.lid = 1}};
  remoteInfo.qpInfos = {{.qpNum = 200, .psn = 300}};
  remoteInfo.ctrl.rkeys = {0};
  remoteInfo.slab.rkeys = {0};

  auto status = transport.connect(remoteInfo.serialize());
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::NotConnected);
}

TEST_F(RdmaTransportTest, ShutdownDestroysResources) {
  fakeQp0_.qp_num = 100;

  EXPECT_CALL(*mockApi_, createCq(_, _, _, _, _))
      .WillOnce(Return(Result<ibv_cq*>(&fakeCq0_)));
  EXPECT_CALL(*mockApi_, createQp(_, _))
      .WillOnce(Return(Result<ibv_qp*>(&fakeQp0_)));
  EXPECT_CALL(*mockApi_, modifyQp(_, _, _)).WillRepeatedly(Return(Ok()));

  EXPECT_CALL(*mockApi_, destroyQp(&fakeQp0_)).WillOnce(Return(Ok()));
  EXPECT_CALL(*mockApi_, destroyCq(&fakeCq0_)).WillOnce(Return(Ok()));

  ibv_gid gid{};
  auto nics = std::make_shared<std::vector<NicResources>>();
  nics->push_back(makeNic(&fakeDev0_, &fakeCtx0_, &fakePd0_, mockApi_, 0, gid));
  RdmaTransport transport(
      mockApi_, mockCudaApi_, evbThread_.getEventBase(), nics, 0);
  transport.bind();

  transport.shutdown();
  EXPECT_EQ(transport.state(), TransportState::Disconnected);
}

// --- RdmaTransportFactory tests ---

class RdmaTransportFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mockApi_ = std::make_shared<testing::NiceMock<MockIbvApi>>();
  }

  std::shared_ptr<testing::NiceMock<MockIbvApi>> mockApi_;
  ScopedEventBaseThread evbThread_;

  ibv_device fakeDevice0_{};
  ibv_device* fakeDeviceList_[2] = {&fakeDevice0_, nullptr};
  ibv_context fakeCtx0_{.device = &fakeDevice0_};
  ibv_pd fakePd0_{};

  void setupSingleDevice() {
    EXPECT_CALL(*mockApi_, getDeviceList(_)).WillOnce([this](int* numDevices) {
      *numDevices = 1;
      return Result<ibv_device**>(fakeDeviceList_);
    });
    EXPECT_CALL(*mockApi_, openDevice(&fakeDevice0_))
        .WillOnce(Return(Result<ibv_context*>(&fakeCtx0_)));
    EXPECT_CALL(*mockApi_, getDeviceName(&fakeDevice0_))
        .WillOnce(Return(Result<const char*>("mlx5_0")));
    EXPECT_CALL(*mockApi_, queryDevice(&fakeCtx0_, _))
        .WillOnce([](ibv_context*, ibv_device_attr* attr) {
          attr->phys_port_cnt = 1;
          return Ok();
        });
    EXPECT_CALL(*mockApi_, allocPd(&fakeCtx0_))
        .WillOnce(Return(Result<ibv_pd*>(&fakePd0_)));
    EXPECT_CALL(*mockApi_, isDmaBufSupported(&fakePd0_))
        .WillOnce(Return(Result<bool>(true)));
    EXPECT_CALL(*mockApi_, queryPort(&fakeCtx0_, 1, _))
        .WillRepeatedly([](ibv_context*, uint8_t, ibv_port_attr* attr) {
          attr->lid = 0x1234;
          attr->active_mtu = IBV_MTU_4096;
          attr->link_layer = IBV_LINK_LAYER_ETHERNET;
          attr->state = IBV_PORT_ACTIVE;
          return Ok();
        });
    EXPECT_CALL(*mockApi_, queryGid(&fakeCtx0_, 1, 3, _))
        .WillOnce(Return(Ok()));
    EXPECT_CALL(*mockApi_, freeDeviceList(_)).WillOnce(Return(Ok()));
    EXPECT_CALL(*mockApi_, deallocPd(&fakePd0_)).WillOnce(Return(Ok()));
    EXPECT_CALL(*mockApi_, closeDevice(&fakeCtx0_)).WillOnce(Return(Ok()));
  }
};

TEST_F(RdmaTransportFactoryTest, ConstructorThrowsOnEmptyDeviceNames) {
  EXPECT_THROW(
      RdmaTransportFactory({}, evbThread_.getEventBase(), {}, mockApi_),
      std::runtime_error);
}

TEST_F(RdmaTransportFactoryTest, ConstructorByNameInitializesDevices) {
  setupSingleDevice();
  RdmaTransportFactory factory(
      {"mlx5_0"}, evbThread_.getEventBase(), {}, mockApi_);
}

// --- Topology and version tests ---

TEST_F(RdmaTransportFactoryTest, GetTopologyReturnsNonEmpty) {
  setupSingleDevice();
  RdmaTransportFactory factory(
      {"mlx5_0"}, evbThread_.getEventBase(), {}, mockApi_);
  auto topo = factory.getTopology();
  EXPECT_FALSE(topo.empty());
  EXPECT_EQ(topo[0], kRdmaVersion);
}

TEST_F(RdmaTransportFactoryTest, CreateTransportAcceptsValidTopology) {
  setupSingleDevice();
  RdmaTransportFactory factory(
      {"mlx5_0"}, evbThread_.getEventBase(), {}, mockApi_);
  auto topo = factory.getTopology();
  auto result = factory.createTransport(topo);
  EXPECT_TRUE(result.hasValue());
}

TEST_F(RdmaTransportFactoryTest, CreateTransportRejectsEmptyTopology) {
  setupSingleDevice();
  RdmaTransportFactory factory(
      {"mlx5_0"}, evbThread_.getEventBase(), {}, mockApi_);
  auto result = factory.createTransport({});
  EXPECT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(RdmaTransportFactoryTest, CreateTransportRejectsWrongVersion) {
  setupSingleDevice();
  RdmaTransportFactory factory(
      {"mlx5_0"}, evbThread_.getEventBase(), {}, mockApi_);
  std::vector<uint8_t> badTopo = {99};
  auto result = factory.createTransport(badTopo);
  EXPECT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

TEST_F(RdmaTransportFactoryTest, CreateTransportRejectsOversizedTopology) {
  setupSingleDevice();
  RdmaTransportFactory factory(
      {"mlx5_0"}, evbThread_.getEventBase(), {}, mockApi_);
  std::vector<uint8_t> bigTopo = {1, 0, 0, 0};
  auto result = factory.createTransport(bigTopo);
  EXPECT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

// --- supported() tests ---

TEST_F(RdmaTransportFactoryTest, SupportedReturnsTrueWithActiveDevice) {
  EXPECT_CALL(*mockApi_, getDeviceList(_)).WillOnce([this](int* numDevices) {
    *numDevices = 1;
    return Result<ibv_device**>(fakeDeviceList_);
  });
  EXPECT_CALL(*mockApi_, openDevice(&fakeDevice0_))
      .WillOnce(Return(Result<ibv_context*>(&fakeCtx0_)));
  EXPECT_CALL(*mockApi_, queryDevice(&fakeCtx0_, _))
      .WillOnce([](ibv_context*, ibv_device_attr* attr) {
        attr->phys_port_cnt = 1;
        return Ok();
      });
  EXPECT_CALL(*mockApi_, queryPort(&fakeCtx0_, 1, _))
      .WillOnce([](ibv_context*, uint8_t, ibv_port_attr* attr) {
        attr->state = IBV_PORT_ACTIVE;
        return Ok();
      });
  EXPECT_CALL(*mockApi_, closeDevice(&fakeCtx0_)).WillOnce(Return(Ok()));
  EXPECT_CALL(*mockApi_, freeDeviceList(_)).WillOnce(Return(Ok()));

  EXPECT_TRUE(RdmaTransportFactory::supported(mockApi_));
}

TEST_F(RdmaTransportFactoryTest, SupportedReturnsFalseWhenInitFails) {
  EXPECT_CALL(*mockApi_, init())
      .WillOnce(Return(Err(ErrCode::DriverError, "no libibverbs")));

  EXPECT_FALSE(RdmaTransportFactory::supported(mockApi_));
}

TEST_F(RdmaTransportFactoryTest, SupportedReturnsFalseWhenNoDevices) {
  EXPECT_CALL(*mockApi_, getDeviceList(_)).WillOnce([](int* numDevices) {
    *numDevices = 0;
    return Result<ibv_device**>(nullptr);
  });

  EXPECT_FALSE(RdmaTransportFactory::supported(mockApi_));
}

TEST_F(RdmaTransportFactoryTest, SupportedReturnsFalseWhenNoActivePort) {
  EXPECT_CALL(*mockApi_, getDeviceList(_)).WillOnce([this](int* numDevices) {
    *numDevices = 1;
    return Result<ibv_device**>(fakeDeviceList_);
  });
  EXPECT_CALL(*mockApi_, openDevice(&fakeDevice0_))
      .WillOnce(Return(Result<ibv_context*>(&fakeCtx0_)));
  EXPECT_CALL(*mockApi_, queryDevice(&fakeCtx0_, _))
      .WillOnce([](ibv_context*, ibv_device_attr* attr) {
        attr->phys_port_cnt = 1;
        return Ok();
      });
  EXPECT_CALL(*mockApi_, queryPort(&fakeCtx0_, 1, _))
      .WillOnce([](ibv_context*, uint8_t, ibv_port_attr* attr) {
        attr->state = IBV_PORT_DOWN;
        return Ok();
      });
  EXPECT_CALL(*mockApi_, closeDevice(&fakeCtx0_)).WillOnce(Return(Ok()));
  EXPECT_CALL(*mockApi_, freeDeviceList(_)).WillOnce(Return(Ok()));

  auto status = RdmaTransportFactory::supported(mockApi_);
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::ResourceExhausted);
}

TEST_F(RdmaTransportFactoryTest, SupportedReturnsFalseWhenOpenDeviceFails) {
  EXPECT_CALL(*mockApi_, getDeviceList(_)).WillOnce([this](int* numDevices) {
    *numDevices = 1;
    return Result<ibv_device**>(fakeDeviceList_);
  });
  EXPECT_CALL(*mockApi_, openDevice(&fakeDevice0_))
      .WillOnce(Return(Result<ibv_context*>(ErrCode::DriverError)));
  EXPECT_CALL(*mockApi_, freeDeviceList(_)).WillOnce(Return(Ok()));

  auto status = RdmaTransportFactory::supported(mockApi_);
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::DriverError);
}

TEST_F(RdmaTransportFactoryTest, SupportedClosesDeviceWhenQueryDeviceFails) {
  EXPECT_CALL(*mockApi_, getDeviceList(_)).WillOnce([this](int* numDevices) {
    *numDevices = 1;
    return Result<ibv_device**>(fakeDeviceList_);
  });
  EXPECT_CALL(*mockApi_, openDevice(&fakeDevice0_))
      .WillOnce(Return(Result<ibv_context*>(&fakeCtx0_)));
  EXPECT_CALL(*mockApi_, queryDevice(&fakeCtx0_, _))
      .WillOnce(Return(Err(ErrCode::DriverError, "query failed")));
  EXPECT_CALL(*mockApi_, closeDevice(&fakeCtx0_)).WillOnce(Return(Ok()));
  EXPECT_CALL(*mockApi_, freeDeviceList(_)).WillOnce(Return(Ok()));

  auto status = RdmaTransportFactory::supported(mockApi_);
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::ResourceExhausted);
}

TEST_F(RdmaTransportFactoryTest, ConstructorByNameThrowsOnUnknownDevice) {
  EXPECT_CALL(*mockApi_, getDeviceList(_)).WillOnce([this](int* numDevices) {
    *numDevices = 1;
    return Result<ibv_device**>(fakeDeviceList_);
  });
  EXPECT_CALL(*mockApi_, freeDeviceList(_)).WillOnce(Return(Ok()));

  EXPECT_THROW(
      RdmaTransportFactory(
          {"mlx5_99"}, evbThread_.getEventBase(), {}, mockApi_),
      std::runtime_error);
}

// --- Data-path unit tests ---

// Helper to create a connected RdmaTransport with mocked QPs.
// Uses ScopedEventBaseThread so dispatched work executes automatically.
class RdmaTransportDataPathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mockApi_ = std::make_shared<testing::NiceMock<MockIbvApi>>();
    mockCudaApi_ = std::make_shared<testing::NiceMock<MockCudaApi>>();
    evbThread_ = std::make_unique<ScopedEventBaseThread>();

    fakeQp_.qp_num = 100;

    ON_CALL(*mockApi_, regMr(_, _, _, _))
        .WillByDefault(Return(Result<ibv_mr*>(&fakeDataMr_)));
    ON_CALL(*mockApi_, deregMr(_)).WillByDefault(Return(Ok()));
    ON_CALL(*mockCudaApi_, hostAlloc(_, _))
        .WillByDefault([](size_t size, unsigned int) {
          return Result<void*>(std::calloc(1, size));
        });
    ON_CALL(*mockCudaApi_, hostFree(_)).WillByDefault([](void* ptr) {
      std::free(ptr);
      return Ok();
    });

    EXPECT_CALL(*mockApi_, createCq(&fakeCtx_, _, _, _, _))
        .WillOnce(Return(Result<ibv_cq*>(&fakeCq_)));
    EXPECT_CALL(*mockApi_, createQp(&fakePd_, _))
        .WillOnce(Return(Result<ibv_qp*>(&fakeQp_)));
    EXPECT_CALL(*mockApi_, modifyQp(_, _, _)).WillRepeatedly(Return(Ok()));

    ibv_gid gid{};
    auto nics = std::make_shared<std::vector<NicResources>>();
    nics->push_back(
        makeNic(&fakeDev_, &fakeCtx_, &fakePd_, mockApi_, 0x1234, gid));

    transport_ = std::make_unique<RdmaTransport>(
        mockApi_,
        mockCudaApi_,
        evbThread_->getEventBase(),
        nics,
        kLocalDomainId);
    transport_->bind();

    RdmaTransportInfo remoteInfo;
    remoteInfo.header.version = 1;
    remoteInfo.header.numQps = 1;
    remoteInfo.header.numNics = 1;
    remoteInfo.header.domainId = kRemoteDomainId;
    remoteInfo.nicInfos = {{.lid = 0x5678}};
    remoteInfo.qpInfos = {{.qpNum = 200, .psn = 300}};
    remoteInfo.ctrl.rkeys = {0};
    remoteInfo.slab.rkeys = {0};
    transport_->connect(remoteInfo.serialize());
  }

  void TearDown() override {
    if (transport_) {
      transport_->shutdown();
      transport_.reset();
    }
    evbThread_.reset();
  }

  /// Reusable local + remote segment pair with matching handles.
  struct TestSegments {
    std::vector<char> localBuf;
    std::vector<char> remoteBuf;
    Segment localSeg;
    ibv_mr fakeMr{};
    RegisteredSegment localReg;
    RemoteRegisteredSegment remoteReg;

    TestSegments(
        size_t size,
        std::shared_ptr<IbvApi> ibvApi,
        uint64_t localDomainId,
        uint64_t remoteDomainId)
        : localBuf(size),
          remoteBuf(size),
          localSeg(localBuf.data(), size, MemoryType::DRAM),
          localReg(
              SegmentTest::makeRegistered(
                  localSeg,
                  [&]() {
                    fakeMr.addr = localBuf.data();
                    fakeMr.length = size;
                    fakeMr.lkey = 0x1111;
                    fakeMr.rkey = 0x2222;
                    return std::make_unique<RdmaRegistrationHandle>(
                        std::vector<ibv_mr*>{&fakeMr}, ibvApi, localDomainId);
                  }())),
          remoteReg(
              SegmentTest::makeRemote(
                  remoteBuf.data(),
                  size,
                  std::make_unique<RdmaRemoteRegistrationHandle>(
                      std::vector<uint32_t>{0x3333},
                      remoteDomainId))) {}
  };

  std::unique_ptr<TestSegments> makeTestSegments(size_t size = 4096) {
    return std::make_unique<TestSegments>(
        size, mockApi_, kLocalDomainId, kRemoteDomainId);
  }

  static constexpr uint64_t kLocalDomainId = 42;
  static constexpr uint64_t kRemoteDomainId = 99;

  std::shared_ptr<testing::NiceMock<MockIbvApi>> mockApi_;
  std::shared_ptr<testing::NiceMock<MockCudaApi>> mockCudaApi_;
  std::unique_ptr<ScopedEventBaseThread> evbThread_;
  std::unique_ptr<RdmaTransport> transport_;

  ibv_device fakeDev_{};
  ibv_context fakeCtx_{.device = &fakeDev_};
  ibv_pd fakePd_{};
  ibv_cq fakeCq_{};
  ibv_qp fakeQp_{};
  ibv_mr fakeDataMr_{};
};

// --- Opcode-only parameterized fixture (put vs get) ---

struct OpcodeParam {
  ibv_wr_opcode opcode;
  std::string name;
};

std::string opcodeParamName(const ::testing::TestParamInfo<OpcodeParam>& info) {
  return info.param.name;
}

class RdmaTransportOpcodeTest
    : public RdmaTransportDataPathTest,
      public ::testing::WithParamInterface<OpcodeParam> {
 protected:
  std::future<Status> transfer(std::vector<TransferRequest>& reqs) {
    return (GetParam().opcode == IBV_WR_RDMA_WRITE) ? transport_->put(reqs)
                                                    : transport_->get(reqs);
  }
};

// --- Error-path tests ---

TEST_P(RdmaTransportOpcodeTest, DisconnectedTransportReturnsError) {
  transport_->shutdown();
  ASSERT_EQ(transport_->state(), TransportState::Disconnected);

  auto ts = makeTestSegments();
  TransferRequest req{
      .local = ts->localReg.span(0ul, ts->localBuf.size()),
      .remote = ts->remoteReg.span(0ul, ts->remoteBuf.size()),
  };
  std::vector<TransferRequest> reqs = {req};
  auto status = transfer(reqs).get();
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::NotConnected);
}

TEST_P(RdmaTransportOpcodeTest, EmptyRequestsReturnsOk) {
  std::vector<TransferRequest> reqs;
  auto status = transfer(reqs).get();
  EXPECT_FALSE(status.hasError());
}

TEST_P(RdmaTransportOpcodeTest, MismatchedSpanSizesReturnsError) {
  auto ts = makeTestSegments();
  TransferRequest req{
      .local = ts->localReg.span(0ul, 1024),
      .remote = ts->remoteReg.span(0ul, 2048),
  };
  std::vector<TransferRequest> reqs = {req};
  auto status = transfer(reqs).get();
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_P(RdmaTransportOpcodeTest, NoMatchingHandleReturnsError) {
  char localBuf[4096]{};
  char remoteBuf[4096]{};
  Segment localSeg(localBuf, sizeof(localBuf), MemoryType::DRAM);

  ibv_mr fakeMr{};
  fakeMr.lkey = 0x1111;
  fakeMr.rkey = 0x2222;

  // Wrong domainIds → preprocessRequest fails.
  auto localHandle = std::make_unique<RdmaRegistrationHandle>(
      std::vector<ibv_mr*>{&fakeMr}, mockApi_, 9999);
  auto remoteHandle = std::make_unique<RdmaRemoteRegistrationHandle>(
      std::vector<uint32_t>{0x3333}, 8888);

  auto localReg = SegmentTest::makeRegistered(localSeg, std::move(localHandle));
  auto remoteReg = SegmentTest::makeRemote(
      remoteBuf, sizeof(remoteBuf), std::move(remoteHandle));

  TransferRequest req{
      .local = localReg.span(0ul, sizeof(localBuf)),
      .remote = remoteReg.span(0ul, sizeof(remoteBuf)),
  };
  std::vector<TransferRequest> reqs = {req};
  auto status = transfer(reqs).get();
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_P(RdmaTransportOpcodeTest, NicCountMismatchReturnsError) {
  char localBuf[4096]{};
  char remoteBuf[4096]{};
  Segment localSeg(localBuf, sizeof(localBuf), MemoryType::DRAM);

  ibv_mr fakeMr0{};
  ibv_mr fakeMr1{};
  fakeMr0.lkey = 0x1111;
  fakeMr1.lkey = 0x2222;

  // 2 MRs but transport has 1 NIC → NIC count mismatch.
  auto localHandle = std::make_unique<RdmaRegistrationHandle>(
      std::vector<ibv_mr*>{&fakeMr0, &fakeMr1}, mockApi_, kLocalDomainId);
  auto remoteHandle = std::make_unique<RdmaRemoteRegistrationHandle>(
      std::vector<uint32_t>{0x3333}, kRemoteDomainId);

  auto localReg = SegmentTest::makeRegistered(localSeg, std::move(localHandle));
  auto remoteReg = SegmentTest::makeRemote(
      remoteBuf, sizeof(remoteBuf), std::move(remoteHandle));

  TransferRequest req{
      .local = localReg.span(0ul, sizeof(localBuf)),
      .remote = remoteReg.span(0ul, sizeof(remoteBuf)),
  };
  std::vector<TransferRequest> reqs = {req};
  auto status = transfer(reqs).get();
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_P(RdmaTransportOpcodeTest, WcErrorFailsTask) {
  auto ts = makeTestSegments();

  EXPECT_CALL(*mockApi_, postSend(&fakeQp_, _, _)).WillOnce(Return(Ok()));

  EXPECT_CALL(*mockApi_, pollCq(&fakeCq_, _, _))
      .WillOnce([](ibv_cq*, int, ibv_wc* wcs) {
        wcs[0].status = IBV_WC_REM_ACCESS_ERR;
        wcs[0].qp_num = 100;
        wcs[0].wr_id = (0ULL << 32) | 1;
        return Result<int>(1);
      })
      .WillRepeatedly(Return(Result<int>(0)));

  TransferRequest req{
      .local = ts->localReg.span(0ul, ts->localBuf.size()),
      .remote = ts->remoteReg.span(0ul, ts->remoteBuf.size()),
  };
  std::vector<TransferRequest> reqs = {req};
  auto status = transfer(reqs).get();
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::DriverError);
}

TEST_P(RdmaTransportOpcodeTest, RetriesWhenQpIsFull) {
  constexpr size_t kMaxWr = 128; // must match RdmaTransportConfig default
  constexpr size_t kNumWr = kMaxWr + 1; // 129 WRs
  constexpr size_t kBufSize = kNumWr * kChunkSize; // 129 chunks

  auto ts = makeTestSegments(kBufSize);

  // Each spray() call posts one signaled chain via postSend. Record
  // the signaled WR count from each chain so pollCq can return the
  // matching CQE at the right time.
  std::queue<uint32_t> pendingCqes; // signaled counts awaiting completion
  uint32_t postSendCallCount = 0;

  EXPECT_CALL(*mockApi_, postSend(&fakeQp_, _, _))
      .WillRepeatedly([&pendingCqes, &postSendCallCount](
                          ibv_qp*, ibv_send_wr* wr, ibv_send_wr**) {
        // The signaled count is encoded in the last WR's wr_id lower 32 bits.
        ibv_send_wr* last = wr;
        while (last->next != nullptr) {
          last = last->next;
        }
        uint32_t count = static_cast<uint32_t>(last->wr_id & 0xffffffff);
        pendingCqes.push(count);
        ++postSendCallCount;
        return Ok();
      });

  // Return one CQE per signaled chain, in order. Each CQE carries
  // the exact WR count that spray() encoded in the signaled WR.
  EXPECT_CALL(*mockApi_, pollCq(&fakeCq_, _, _))
      .WillRepeatedly([&pendingCqes](ibv_cq*, int, ibv_wc* wcs) {
        if (!pendingCqes.empty()) {
          uint32_t count = pendingCqes.front();
          pendingCqes.pop();
          wcs[0].status = IBV_WC_SUCCESS;
          wcs[0].qp_num = 100;
          wcs[0].wr_id = (0ULL << 32) | count;
          return Result<int>(1);
        }
        return Result<int>(0);
      });

  TransferRequest req{
      .local = ts->localReg.span(0ul, kBufSize),
      .remote = ts->remoteReg.span(0ul, kBufSize),
  };
  std::vector<TransferRequest> reqs = {req};
  auto status = transfer(reqs).get();
  EXPECT_FALSE(status.hasError()) << status.error().message();

  EXPECT_GE(postSendCallCount, 2u)
      << "Expected at least 2 postSend calls (QP full → retry)";
}

TEST_P(RdmaTransportOpcodeTest, PartialPostDrainsCqeBeforeCompleting) {
  // Verify that when postSend partially fails (some WRs consumed before the
  // bad WR), the flush WR's CQE is polled before the task completes.
  constexpr size_t kBufSize = kChunkSize * 2; // 2 chunks → 2 WRs in chain

  auto ts = makeTestSegments(kBufSize);

  int postSendCallCount = 0;
  bool pollCqCalled = false;

  EXPECT_CALL(*mockApi_, postSend(&fakeQp_, _, _))
      .WillOnce(
          [&postSendCallCount](ibv_qp*, ibv_send_wr* wr, ibv_send_wr** badWr) {
            // Partial failure: first WR consumed, second WR is "bad".
            ++postSendCallCount;
            *badWr = wr->next;
            return Err(ErrCode::DriverError, "partial post failure");
          })
      .WillOnce([&postSendCallCount](ibv_qp*, ibv_send_wr*, ibv_send_wr**) {
        // Flush WR succeeds.
        ++postSendCallCount;
        return Ok();
      });

  // The poll chain must be scheduled to drain the flush CQE.
  // flushCount = consumed(1) + 1 = 2
  EXPECT_CALL(*mockApi_, pollCq(&fakeCq_, _, _))
      .WillOnce([&pollCqCalled](ibv_cq*, int, ibv_wc* wcs) {
        pollCqCalled = true;
        wcs[0].status = IBV_WC_SUCCESS;
        wcs[0].qp_num = 100;
        wcs[0].wr_id = (0ULL << 32) | 2; // flushCount = 2
        return Result<int>(1);
      })
      .WillRepeatedly(Return(Result<int>(0)));

  TransferRequest req{
      .local = ts->localReg.span(0ul, kBufSize),
      .remote = ts->remoteReg.span(0ul, kBufSize),
  };
  std::vector<TransferRequest> reqs = {req};
  auto status = transfer(reqs).get();

  // Task should complete with an error (not hang).
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::DriverError);

  // The future is resolved before the poll chain runs on the event base.
  // shutdown() blocks until all inflight tasks are drained, which requires
  // the poll chain to poll the flush CQE. If the poll chain was never
  // scheduled, pollCq() will never be called.
  transport_->shutdown();

  EXPECT_TRUE(pollCqCalled) << "Poll chain was not scheduled for flush CQE";
  EXPECT_EQ(postSendCallCount, 2)
      << "Expected 2 postSend calls: original (partial fail) + flush";
}

TEST_P(RdmaTransportOpcodeTest, ShutdownDrainsInflightTransfer) {
  // Verify that shutdown() waits for the poll chain to drain in-flight
  // CQEs before destroying resources. The pollCq mock delays returning
  // the CQE to simulate a transfer that is still in progress when
  // shutdown() is called.
  auto ts = makeTestSegments();

  EXPECT_CALL(*mockApi_, postSend(&fakeQp_, _, _)).WillOnce(Return(Ok()));

  EXPECT_CALL(*mockApi_, pollCq(&fakeCq_, _, _))
      .WillOnce(Return(Result<int>(0)))
      .WillOnce(Return(Result<int>(0)))
      .WillOnce([](ibv_cq*, int, ibv_wc* wcs) {
        wcs[0].status = IBV_WC_SUCCESS;
        wcs[0].qp_num = 100;
        wcs[0].wr_id = (0ULL << 32) | 1; // numWrs = 1
        return Result<int>(1);
      })
      .WillRepeatedly(Return(Result<int>(0)));

  TransferRequest req{
      .local = ts->localReg.span(0ul, ts->localBuf.size()),
      .remote = ts->remoteReg.span(0ul, ts->remoteBuf.size()),
  };
  std::vector<TransferRequest> reqs = {req};

  // Get the future but don't block on it yet — let the transfer start.
  auto future = transfer(reqs);

  // shutdown() blocks until the poll chain drains the CQE. If the drain
  // logic is broken, this would hang or fail.
  transport_->shutdown();

  auto status = future.get();
  EXPECT_FALSE(status.hasError()) << status.error().message();
}

TEST_P(RdmaTransportOpcodeTest, SubsequentTransferAfterWcError) {
  // Verify that a WC error on one transfer does not prevent subsequent
  // transfers from succeeding — the error is per-task, not transport-wide.
  auto ts = makeTestSegments(kChunkSize * kDefaultMaxWr);

  std::vector<uint64_t> capturedWrIds;
  EXPECT_CALL(*mockApi_, postSend(&fakeQp_, _, _))
      .Times(2)
      .WillRepeatedly(
          [&capturedWrIds](ibv_qp*, ibv_send_wr* wr, ibv_send_wr**) {
            ibv_send_wr* last = wr;
            while (last->next) {
              last = last->next;
            }
            capturedWrIds.push_back(last->wr_id);
            return Ok();
          });

  size_t cqeIdx = 0;
  EXPECT_CALL(*mockApi_, pollCq(&fakeCq_, _, _))
      .WillRepeatedly([&capturedWrIds, &cqeIdx](ibv_cq*, int, ibv_wc* wcs) {
        if (cqeIdx < capturedWrIds.size()) {
          wcs[0].qp_num = 100;
          wcs[0].wr_id = capturedWrIds[cqeIdx];
          wcs[0].status =
              (cqeIdx == 0) ? IBV_WC_REM_ACCESS_ERR : IBV_WC_SUCCESS;
          ++cqeIdx;
          return Result<int>(1);
        }
        return Result<int>(0);
      });

  TransferRequest req{
      .local = ts->localReg.span(0ul, ts->localBuf.size()),
      .remote = ts->remoteReg.span(0ul, ts->remoteBuf.size()),
  };
  std::vector<TransferRequest> reqs = {req};

  // First transfer fails with WC error.
  auto status1 = transfer(reqs).get();
  EXPECT_TRUE(status1.hasError());
  EXPECT_EQ(status1.error().code(), ErrCode::DriverError);

  // Transport should still be connected.
  EXPECT_EQ(transport_->state(), TransportState::Connected);

  // Second transfer succeeds.
  auto status2 = transfer(reqs).get();
  EXPECT_FALSE(status2.hasError()) << status2.error().message();
}

TEST_P(RdmaTransportOpcodeTest, PollCqErrorFailsAllSubsequentTasks) {
  // Verify that when pollCq itself returns an API error, the transport
  // transitions to Error state and all inflight tasks get TransportError.
  auto ts1 = makeTestSegments();
  auto ts2 = makeTestSegments();

  EXPECT_CALL(*mockApi_, postSend(&fakeQp_, _, _)).WillRepeatedly(Return(Ok()));

  // pollCq returns an API error on the first call.
  EXPECT_CALL(*mockApi_, pollCq(&fakeCq_, _, _))
      .WillRepeatedly(
          Return(Result<int>(Err(ErrCode::DriverError, "pollCq failed"))));

  TransferRequest req1{
      .local = ts1->localReg.span(0ul, ts1->localBuf.size()),
      .remote = ts1->remoteReg.span(0ul, ts1->remoteBuf.size()),
  };
  TransferRequest req2{
      .local = ts2->localReg.span(0ul, ts2->localBuf.size()),
      .remote = ts2->remoteReg.span(0ul, ts2->remoteBuf.size()),
  };
  std::vector<TransferRequest> reqs1 = {req1};
  std::vector<TransferRequest> reqs2 = {req2};

  // Launch two transfers concurrently (don't block on futures yet).
  auto future1 = transfer(reqs1);
  auto future2 = transfer(reqs2);

  // Both tasks should fail with TransportError.
  auto status1 = future1.get();
  auto status2 = future2.get();

  EXPECT_TRUE(status1.hasError());
  EXPECT_EQ(status1.error().code(), ErrCode::TransportError);
  EXPECT_TRUE(status2.hasError());
  EXPECT_EQ(status2.error().code(), ErrCode::TransportError);
  EXPECT_EQ(transport_->state(), TransportState::Error);
}

TEST_P(RdmaTransportOpcodeTest, UnsignaledCqeDoesNotOverDecrementPendingCqe) {
  // When a QP flushes (e.g., on error), the HCA generates CQEs for ALL
  // pending WRs, including unsignaled ones. Unsignaled CQEs have numWrs=0
  // in their wr_id. Before the fix (D99868453), pollCompletions decremented
  // numPendingCqe_ by the total polled count n (including unsignaled CQEs).
  // This over-decrement made numPendingCqe_ go negative, causing subsequent
  // pollCompletions calls to skip the polling loop (expected <= 0) and hang.
  //
  // This test verifies the fix: only CQEs with numWrs > 0 (signaled WRs)
  // decrement numPendingCqe_, keeping it non-negative and allowing
  // subsequent transfers to poll correctly.
  constexpr size_t kNumChunks = 3;
  constexpr size_t kBufSize = kChunkSize * kNumChunks;

  auto ts1 = makeTestSegments(kBufSize);
  auto ts2 = makeTestSegments(kChunkSize);

  // Capture the signaled WR's wr_id from each postSend chain.
  std::vector<uint64_t> signaledWrIds;
  EXPECT_CALL(*mockApi_, postSend(&fakeQp_, _, _))
      .WillRepeatedly(
          [&signaledWrIds](ibv_qp*, ibv_send_wr* wr, ibv_send_wr**) {
            ibv_send_wr* last = wr;
            while (last->next) {
              last = last->next;
            }
            signaledWrIds.push_back(last->wr_id);
            return Ok();
          });

  // pollCq mock: for the first transfer, return 3 CQEs (2 unsignaled + 1
  // signaled) simulating a QP flush. For the second transfer, return a
  // normal single CQE.
  size_t nextCqe = 0;
  EXPECT_CALL(*mockApi_, pollCq(&fakeCq_, _, _))
      .WillRepeatedly([&](ibv_cq*, int, ibv_wc* wcs) {
        if (nextCqe < signaledWrIds.size()) {
          auto wrId = signaledWrIds[nextCqe];
          uint32_t taskId = static_cast<uint32_t>(wrId >> 32);

          if (nextCqe == 0) {
            // First transfer: inject 2 unsignaled CQEs (numWrs=0) before
            // the real signaled CQE, simulating QP error flush.
            wcs[0].status = IBV_WC_WR_FLUSH_ERR;
            wcs[0].qp_num = 100;
            wcs[0].wr_id = (static_cast<uint64_t>(taskId) << 32); // numWrs=0

            wcs[1].status = IBV_WC_WR_FLUSH_ERR;
            wcs[1].qp_num = 100;
            wcs[1].wr_id = (static_cast<uint64_t>(taskId) << 32); // numWrs=0

            wcs[2].status = IBV_WC_SUCCESS;
            wcs[2].qp_num = 100;
            wcs[2].wr_id = wrId; // signaled: numWrs=3

            ++nextCqe;
            return Result<int>(3);
          }

          // Subsequent transfers: normal single CQE.
          wcs[0].status = IBV_WC_SUCCESS;
          wcs[0].qp_num = 100;
          wcs[0].wr_id = wrId;
          ++nextCqe;
          return Result<int>(1);
        }
        return Result<int>(0);
      });

  // --- First transfer (3 chunks) ---
  // pollCq returns 2 unsignaled CQEs (numWrs=0) + 1 signaled CQE (numWrs=3).
  // With the fix: numPendingCqe_ decremented by 1 (only for signaled CQE).
  // Without fix: numPendingCqe_ decremented by 3 → goes to -2.
  TransferRequest req1{
      .local = ts1->localReg.span(0ul, kBufSize),
      .remote = ts1->remoteReg.span(0ul, kBufSize),
  };
  std::vector<TransferRequest> reqs1 = {req1};
  auto status1 = transfer(reqs1).get();

  EXPECT_TRUE(status1.hasError());
  EXPECT_EQ(status1.error().code(), ErrCode::DriverError);
  EXPECT_EQ(transport_->state(), TransportState::Connected);

  // --- Second transfer (1 chunk) ---
  // If numPendingCqe_ was over-decremented (went to -2), then after postSend
  // increments it by 1, it would be -1. pollCompletions would see
  // expected = -1 <= 0, skip the polling loop, and this transfer would hang.
  TransferRequest req2{
      .local = ts2->localReg.span(0ul, kChunkSize),
      .remote = ts2->remoteReg.span(0ul, kChunkSize),
  };
  std::vector<TransferRequest> reqs2 = {req2};
  auto status2 = transfer(reqs2).get();
  EXPECT_FALSE(status2.hasError()) << status2.error().message();
}

INSTANTIATE_TEST_SUITE_P(
    Opcode,
    RdmaTransportOpcodeTest,
    ::testing::Values(
        OpcodeParam{IBV_WR_RDMA_WRITE, "Put"},
        OpcodeParam{IBV_WR_RDMA_READ, "Get"}),
    opcodeParamName);

// --- Parameterized SGE/WR verification tests ---

struct PutGetParam {
  size_t bufSize;
  size_t numRequests;
  ibv_wr_opcode opcode;
  std::string name;
};

std::string putGetParamName(const ::testing::TestParamInfo<PutGetParam>& info) {
  return info.param.name;
}

class RdmaTransportPutGetTest
    : public RdmaTransportDataPathTest,
      public ::testing::WithParamInterface<PutGetParam> {
 protected:
  std::future<Status> transfer(std::vector<TransferRequest>& reqs) {
    const auto& param = GetParam();
    return (param.opcode == IBV_WR_RDMA_WRITE) ? transport_->put(reqs)
                                               : transport_->get(reqs);
  }
};

TEST_P(RdmaTransportPutGetTest, VerifySgeAndWrFields) {
  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;

  auto ts = makeTestSegments(totalSize);

  // Calculate expected number of chunks across all requests.
  size_t expectedChunks = 0;
  for (size_t r = 0; r < numRequests; ++r) {
    expectedChunks += (bufSize + kChunkSize - 1) / kChunkSize;
  }

  // Capture WRs passed to postSend for verification.
  struct CapturedWr {
    uint64_t sgeAddr;
    uint32_t sgeLen;
    uint32_t sgeLkey;
    uint64_t remoteAddr;
    uint32_t remoteRkey;
    ibv_wr_opcode opcode;
    int numSge;
    uint32_t sendFlags;
  };
  std::vector<CapturedWr> capturedWrs;

  EXPECT_CALL(*mockApi_, postSend(&fakeQp_, _, _))
      .WillOnce([&capturedWrs](ibv_qp*, ibv_send_wr* wr, ibv_send_wr**) {
        for (auto* w = wr; w != nullptr; w = w->next) {
          capturedWrs.push_back({
              .sgeAddr = w->sg_list->addr,
              .sgeLen = w->sg_list->length,
              .sgeLkey = w->sg_list->lkey,
              .remoteAddr = w->wr.rdma.remote_addr,
              .remoteRkey = w->wr.rdma.rkey,
              .opcode = w->opcode,
              .numSge = w->num_sge,
              .sendFlags = static_cast<uint32_t>(w->send_flags),
          });
        }
        return Ok();
      });

  // Return successful CQEs for all chunks (encoded as single signaled WR).
  EXPECT_CALL(*mockApi_, pollCq(&fakeCq_, _, _))
      .WillOnce([&expectedChunks](ibv_cq*, int, ibv_wc* wcs) {
        wcs[0].status = IBV_WC_SUCCESS;
        wcs[0].qp_num = 100;
        wcs[0].wr_id = (0ULL << 32) | static_cast<uint32_t>(expectedChunks);
        return Result<int>(1);
      })
      .WillRepeatedly(Return(Result<int>(0)));

  // Build transfer requests.
  std::vector<TransferRequest> reqs;
  reqs.reserve(numRequests);
  for (size_t r = 0; r < numRequests; ++r) {
    reqs.push_back(
        TransferRequest{
            .local = ts->localReg.span(r * bufSize, bufSize),
            .remote = ts->remoteReg.span(r * bufSize, bufSize),
        });
  }

  auto status = transfer(reqs).get();
  EXPECT_FALSE(status.hasError()) << status.error().message();

  // Verify captured WRs.
  ASSERT_EQ(capturedWrs.size(), expectedChunks)
      << "Expected " << expectedChunks << " chunks, got " << capturedWrs.size();

  // Walk through each request's chunks and verify SGE/WR fields.
  size_t wrIdx = 0;
  for (size_t r = 0; r < numRequests; ++r) {
    size_t reqOffset = r * bufSize;
    size_t remaining = bufSize;
    size_t chunkOffset = 0;
    while (remaining > 0) {
      size_t chunkLen = std::min(remaining, kChunkSize);
      ASSERT_LT(wrIdx, capturedWrs.size());
      auto& cwr = capturedWrs[wrIdx];

      // SGE addr should point to local buffer + offset.
      EXPECT_EQ(
          cwr.sgeAddr,
          reinterpret_cast<uint64_t>(ts->localBuf.data()) + reqOffset +
              chunkOffset)
          << "WR " << wrIdx << ": sge.addr mismatch";
      EXPECT_EQ(cwr.sgeLen, static_cast<uint32_t>(chunkLen))
          << "WR " << wrIdx << ": sge.length mismatch";
      EXPECT_EQ(cwr.sgeLkey, 0x1111u)
          << "WR " << wrIdx << ": sge.lkey mismatch";

      // Remote addr should point to remote buffer + offset.
      EXPECT_EQ(
          cwr.remoteAddr,
          reinterpret_cast<uint64_t>(ts->remoteBuf.data()) + reqOffset +
              chunkOffset)
          << "WR " << wrIdx << ": wr.rdma.remote_addr mismatch";
      EXPECT_EQ(cwr.remoteRkey, 0x3333u)
          << "WR " << wrIdx << ": wr.rdma.rkey mismatch";

      EXPECT_EQ(cwr.opcode, param.opcode)
          << "WR " << wrIdx << ": opcode mismatch";
      EXPECT_EQ(cwr.numSge, 1) << "WR " << wrIdx << ": num_sge mismatch";

      remaining -= chunkLen;
      chunkOffset += chunkLen;
      ++wrIdx;
    }
  }

  // Last WR should be signaled; others should not.
  for (size_t i = 0; i + 1 < capturedWrs.size(); ++i) {
    EXPECT_EQ(capturedWrs[i].sendFlags & IBV_SEND_SIGNALED, 0u)
        << "WR " << i << " should NOT be signaled";
  }
  if (!capturedWrs.empty()) {
    EXPECT_NE(capturedWrs.back().sendFlags & IBV_SEND_SIGNALED, 0u)
        << "Last WR should be signaled";
  }
}

INSTANTIATE_TEST_SUITE_P(
    PutGet,
    RdmaTransportPutGetTest,
    ::testing::Values(
        // Single request, fits in one chunk (4KB < 512KB).
        PutGetParam{4096, 1, IBV_WR_RDMA_WRITE, "Put_4KB_x1"},
        PutGetParam{4096, 1, IBV_WR_RDMA_READ, "Get_4KB_x1"},
        // Single request, exactly one chunk (512KB).
        PutGetParam{512 * 1024, 1, IBV_WR_RDMA_WRITE, "Put_512KB_x1"},
        PutGetParam{512 * 1024, 1, IBV_WR_RDMA_READ, "Get_512KB_x1"},
        // Single request, multi-chunk (1MB = 2 chunks of 512KB).
        PutGetParam{1024 * 1024, 1, IBV_WR_RDMA_WRITE, "Put_1MB_x1"},
        PutGetParam{1024 * 1024, 1, IBV_WR_RDMA_READ, "Get_1MB_x1"},
        // Multiple requests, single chunk each.
        PutGetParam{4096, 3, IBV_WR_RDMA_WRITE, "Put_4KB_x3"},
        PutGetParam{4096, 3, IBV_WR_RDMA_READ, "Get_4KB_x3"},
        // Multiple requests, multi-chunk each (1MB x 2 = 4 chunks total).
        PutGetParam{1024 * 1024, 2, IBV_WR_RDMA_WRITE, "Put_1MB_x2"},
        PutGetParam{1024 * 1024, 2, IBV_WR_RDMA_READ, "Get_1MB_x2"},
        // Multiple requests, multi-chunk each, unaligned buffer size.
        PutGetParam{
            1024 * 1024 + 157,
            3,
            IBV_WR_RDMA_WRITE,
            "Put_unaligned_x3"},
        PutGetParam{
            1024 * 1024 + 157,
            3,
            IBV_WR_RDMA_READ,
            "Get_unaligned_x3"}),
    putGetParamName);
