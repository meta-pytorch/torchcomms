// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/topology/NicTopologyDiscovery.h"

#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/uniflow/drivers/ibverbs/mock/MockIbvApi.h"
#include "comms/uniflow/drivers/sysfs/mock/MockSysfsApi.h"

using namespace uniflow;
using ::testing::_;
using ::testing::EndsWith;
using ::testing::NiceMock;
using ::testing::Return;

namespace {

class NicTopologyDiscoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ibv_ = std::make_shared<NiceMock<MockIbvApi>>();
    sysfs_ = std::make_shared<NiceMock<MockSysfsApi>>();
    ON_CALL(*sysfs_, resolvePath(_))
        .WillByDefault(Return(Err(ErrCode::InvalidArgument, "no sysfs")));
    ON_CALL(*sysfs_, readFile(_)).WillByDefault(Return(std::string{}));
    ON_CALL(*sysfs_, listDir("/sys/devices/system/node", "node"))
        .WillByDefault(Return(std::vector<std::string>{"node0", "node1"}));
    ON_CALL(*ibv_, init()).WillByDefault(Return(Ok()));
    ON_CALL(*ibv_, freeDeviceList(_)).WillByDefault(Return(Ok()));
    ON_CALL(*ibv_, closeDevice(_)).WillByDefault(Return(Ok()));
    ON_CALL(*ibv_, getDeviceList(_))
        .WillByDefault([](int* count) -> Result<ibv_device**> {
          *count = 0;
          return static_cast<ibv_device**>(nullptr);
        });
  }

  void setupNic() {
    std::strncpy(
        device_.ibdev_path, kIbdevPath, sizeof(device_.ibdev_path) - 1);
    ON_CALL(*ibv_, getDeviceList(_))
        .WillByDefault([this](int* count) -> Result<ibv_device**> {
          *count = 1;
          return deviceList_;
        });
    ON_CALL(*ibv_, getDeviceName(&device_)).WillByDefault(Return("mlx5_0"));
    ON_CALL(*ibv_, openDevice(&device_)).WillByDefault(Return(&context_));
    ON_CALL(*ibv_, queryDevice(&context_, _))
        .WillByDefault([](ibv_context*, ibv_device_attr* attr) {
          attr->phys_port_cnt = 2;
          return Ok();
        });
    ON_CALL(*ibv_, queryPort(&context_, _, _))
        .WillByDefault([](ibv_context*, uint8_t port, ibv_port_attr* attr) {
          attr->state = IBV_PORT_ACTIVE;
          attr->active_speed = port == 1 ? 1 : 64;
          attr->active_width = 2;
          return Ok();
        });
    ON_CALL(*sysfs_, resolvePath(std::string(kIbdevPath) + "/device"))
        .WillByDefault(Return(Result<std::string>(kPciPath)));
    ON_CALL(*sysfs_, readFile(std::string(kPciPath) + "/numa_node"))
        .WillByDefault(Return("1"));
    ON_CALL(*sysfs_, readFile(EndsWith("max_link_speed")))
        .WillByDefault(Return("16 GT/s"));
    ON_CALL(*sysfs_, readFile(EndsWith("max_link_width")))
        .WillByDefault(Return("16"));
    ON_CALL(*sysfs_, listDir(std::string(kIbdevPath) + "/device/net", _))
        .WillByDefault(Return(std::vector<std::string>{"beth0"}));
  }

  static constexpr const char* kIbdevPath = "/sys/class/infiniband/mlx5_0";
  static constexpr const char* kPciPath =
      "/sys/devices/pci0000:00/0000:00:01.0/0000:02:00.0";
  ibv_device device_{};
  ibv_device* deviceList_[1]{&device_};
  ibv_context context_{};
  std::shared_ptr<NiceMock<MockIbvApi>> ibv_;
  std::shared_ptr<NiceMock<MockSysfsApi>> sysfs_;
};

} // namespace

TEST_F(
    NicTopologyDiscoveryTest,
    BuildsOnlyCpuAndNicNodesWithFastestActivePort) {
  setupNic();
  EXPECT_CALL(*ibv_, closeDevice(&context_)).Times(1);
  EXPECT_CALL(*ibv_, freeDeviceList(deviceList_)).Times(1);
  Topology topology;

  ASSERT_TRUE(NicTopologyDiscovery(ibv_, sysfs_).discover(topology));

  ASSERT_TRUE(topology.available());
  EXPECT_EQ(topology.gpuCount(), 0u);
  EXPECT_EQ(topology.numaNodeCount(), 2u);
  ASSERT_EQ(topology.nicCount(), 1u);
  const auto& nic = topology.getNicNode(0);
  const auto& data = std::get<TopoNode::NicData>(nic.data);
  EXPECT_EQ(nic.name, "mlx5_0");
  EXPECT_EQ(data.port, 2);
  EXPECT_EQ(data.portSpeedMbps, 200000u);
  EXPECT_EQ(data.bdf, "0000:02:00.0");
  EXPECT_EQ(data.netdevName, "beth0");
  EXPECT_EQ(topology.nicNumaNode("mlx5_0"), 1);
  const auto& path = topology.getPath(nic.id, topology.getCpuNode(1).id);
  EXPECT_EQ(path.type, PathType::PHB);
  EXPECT_EQ(path.bw, 24000u);
  EXPECT_EQ(
      topology.selectCpuNics(NicFilter{}, "beth"),
      std::vector<std::string>{"mlx5_0"});
}

TEST_F(NicTopologyDiscoveryTest, MissingVerbsStillProvidesNumaGraph) {
  EXPECT_CALL(*ibv_, init())
      .WillOnce(Return(Err(ErrCode::NotImplemented, "verbs unavailable")));
  EXPECT_CALL(*ibv_, getDeviceList(_)).Times(0);
  Topology topology;

  ASSERT_TRUE(NicTopologyDiscovery(ibv_, sysfs_).discover(topology));

  EXPECT_TRUE(topology.available());
  EXPECT_EQ(topology.gpuCount(), 0u);
  EXPECT_EQ(topology.nicCount(), 0u);
  EXPECT_EQ(topology.numaNodeCount(), 2u);
  EXPECT_EQ(
      topology.getPath(topology.getCpuNode(0).id, topology.getCpuNode(1).id)
          .type,
      PathType::SYS);
}

TEST_F(NicTopologyDiscoveryTest, VirtualRdmaKeepsHostPathAndDefaultSpeed) {
  setupNic();
  ON_CALL(*sysfs_, resolvePath(kIbdevPath))
      .WillByDefault(
          Return(Result<std::string>("/sys/devices/virtual/infiniband/rxe0")));
  ON_CALL(*ibv_, getDeviceName(&device_)).WillByDefault(Return("rxe0"));
  ON_CALL(*ibv_, queryPort(&context_, _, _))
      .WillByDefault([](ibv_context*, uint8_t, ibv_port_attr* attr) {
        attr->state = IBV_PORT_ACTIVE;
        attr->active_speed = 0;
        attr->active_width = 0;
        return Ok();
      });
  Topology topology;

  ASSERT_TRUE(NicTopologyDiscovery(ibv_, sysfs_).discover(topology));

  ASSERT_EQ(topology.nicCount(), 1u);
  const auto& nic = topology.getNicNode(0);
  const auto& data = std::get<TopoNode::NicData>(nic.data);
  EXPECT_EQ(data.bdf, "virtual");
  EXPECT_EQ(data.numaNode, 0);
  EXPECT_EQ(data.portSpeedMbps, 10000u);
  EXPECT_EQ(topology.getPath(nic.id, topology.getCpuNode(0).id).bw, 1250u);
}

TEST_F(NicTopologyDiscoveryTest, InactivePortsDoNotCreateNicNodes) {
  setupNic();
  ON_CALL(*ibv_, queryPort(&context_, _, _))
      .WillByDefault([](ibv_context*, uint8_t, ibv_port_attr* attr) {
        attr->state = IBV_PORT_DOWN;
        return Ok();
      });
  EXPECT_CALL(*ibv_, closeDevice(&context_)).Times(1);
  EXPECT_CALL(*ibv_, freeDeviceList(deviceList_)).Times(1);
  Topology topology;

  ASSERT_TRUE(NicTopologyDiscovery(ibv_, sysfs_).discover(topology));

  EXPECT_EQ(topology.nicCount(), 0u);
  EXPECT_EQ(topology.gpuCount(), 0u);
}
