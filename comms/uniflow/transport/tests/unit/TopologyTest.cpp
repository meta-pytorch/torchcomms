// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/Topology.h"

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/drivers/ibverbs/mock/MockIbvApi.h"
#include "comms/uniflow/drivers/nvml/mock/MockNvmlApi.h"
#include "comms/uniflow/drivers/sysfs/mock/MockSysfsApi.h"

#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::AtLeast;
using ::testing::Exactly;
using ::testing::NiceMock;
using ::testing::Return;

namespace uniflow {

// --- Data structure tests ---

TEST(PathTypeTest, OrderingIsCorrect) {
  EXPECT_LT(PathType::NVL, PathType::C2C);
  EXPECT_LT(PathType::C2C, PathType::PIX);
  EXPECT_LT(PathType::PIX, PathType::PXB);
  EXPECT_LT(PathType::PXB, PathType::PXN);
  EXPECT_LT(PathType::PXN, PathType::PHB);
  EXPECT_LT(PathType::PHB, PathType::SYS);
  EXPECT_LT(PathType::SYS, PathType::DIS);
}

TEST(PathTypeTest, ToStringReturnsValidNames) {
  EXPECT_STREQ(pathTypeToString(PathType::NVL), "NVL");
  EXPECT_STREQ(pathTypeToString(PathType::C2C), "C2C");
  EXPECT_STREQ(pathTypeToString(PathType::PIX), "PIX");
  EXPECT_STREQ(pathTypeToString(PathType::PXB), "PXB");
  EXPECT_STREQ(pathTypeToString(PathType::PXN), "PXN");
  EXPECT_STREQ(pathTypeToString(PathType::PHB), "PHB");
  EXPECT_STREQ(pathTypeToString(PathType::SYS), "SYS");
  EXPECT_STREQ(pathTypeToString(PathType::DIS), "DIS");
}

TEST(PcieLinkInfoTest, BandwidthCalculation) {
  PcieLinkInfo gen4x16{.speedMbpsPerLane = 12000, .width = 16};
  EXPECT_EQ(gen4x16.bandwidthMBps(), 24000u);

  PcieLinkInfo gen5x16{.speedMbpsPerLane = 24000, .width = 16};
  EXPECT_EQ(gen5x16.bandwidthMBps(), 48000u);

  PcieLinkInfo zero{};
  EXPECT_EQ(zero.bandwidthMBps(), 0u);
}

TEST(TopoPathTest, DefaultIsDisconnected) {
  TopoPath path;
  EXPECT_EQ(path.type, PathType::DIS);
  EXPECT_EQ(path.bw, 0u);
  EXPECT_FALSE(path.proxyNode.has_value());
}

TEST(TopoNodeTest, VariantHoldsCorrectTypes) {
  TopoNode gpuNode{
      .type = NodeType::GPU,
      .data = TopoNode::GpuData{.cudaDeviceId = 0, .bdf = "0000:07:00.0"},
  };
  EXPECT_TRUE(std::holds_alternative<TopoNode::GpuData>(gpuNode.data));
  EXPECT_EQ(std::get<TopoNode::GpuData>(gpuNode.data).cudaDeviceId, 0);

  TopoNode cpuNode{
      .type = NodeType::CPU,
      .data = TopoNode::CpuData{.numaId = 1},
  };
  EXPECT_TRUE(std::holds_alternative<TopoNode::CpuData>(cpuNode.data));
  EXPECT_EQ(std::get<TopoNode::CpuData>(cpuNode.data).numaId, 1);

  TopoNode nicNode{
      .type = NodeType::NIC,
      .name = "mlx5_0",
      .data = TopoNode::NicData{.port = 1},
  };
  EXPECT_TRUE(std::holds_alternative<TopoNode::NicData>(nicNode.data));
  EXPECT_EQ(nicNode.name, "mlx5_0");
  EXPECT_EQ(std::get<TopoNode::NicData>(nicNode.data).port, 1);
}

TEST(PathFilterTest, DefaultDisablesC2CAndPxn) {
  PathFilter filter;
  EXPECT_FALSE(filter.allowC2C);
  EXPECT_FALSE(filter.allowPxn);
}

// --- Topology tests with mocked hardware ---

// Friend class allows direct construction bypassing singleton.

class TopologyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cuda_ = std::make_shared<NiceMock<MockCudaApi>>();
    nvml_ = std::make_shared<NiceMock<MockNvmlApi>>();
    ibv_ = std::make_shared<NiceMock<MockIbvApi>>();
    sysfs_ = std::make_shared<NiceMock<MockSysfsApi>>();

    // Default sysfs: resolvePath fails (no real sysfs), readFile returns
    // empty, listDir returns 1 NUMA node.
    ON_CALL(*sysfs_, resolvePath(_))
        .WillByDefault(Return(Err(ErrCode::InvalidArgument, "no sysfs")));
    ON_CALL(*sysfs_, readFile(_)).WillByDefault(Return(std::string()));
    ON_CALL(*sysfs_, listDir("/sys/devices/system/node", "node"))
        .WillByDefault(Return(std::vector<std::string>{"node0"}));

    // Default: no NVLink, no C2C.
    ON_CALL(*nvml_, nvmlDeviceGetNvLinkCapability(_, _, _, _))
        .WillByDefault(Return(Err(ErrCode::NotImplemented)));
    ON_CALL(*nvml_, nvmlDeviceGetFieldValues(_, _, _))
        .WillByDefault(Return(Err(ErrCode::NotImplemented)));

    // Default: ibverbs init succeeds with no devices.
    ON_CALL(*ibv_, init()).WillByDefault(Return(Ok()));
    ON_CALL(*ibv_, getDeviceList(_))
        .WillByDefault([](int* n) -> Result<ibv_device**> {
          *n = 0;
          return static_cast<ibv_device**>(nullptr);
        });
    ON_CALL(*ibv_, freeDeviceList(_)).WillByDefault(Return(Ok()));

    // Default: CUDA device save/restore.
    ON_CALL(*cuda_, getDevice()).WillByDefault(Return(Result<int>(0)));
    ON_CALL(*cuda_, setDevice(_)).WillByDefault(Return(Ok()));
  }

  void setupGpus(int count) {
    ON_CALL(*nvml_, deviceCount()).WillByDefault(Return(Result<int>(count)));
    for (int i = 0; i < count; ++i) {
      NvmlApi::DeviceInfo info;
      info.handle =
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(i + 1));
      info.computeCapabilityMajor = 9;
      info.computeCapabilityMinor = 0;
      ON_CALL(*nvml_, deviceInfo(i))
          .WillByDefault(Return(Result<NvmlApi::DeviceInfo>(info)));

      std::string bdf = "0000:0" + std::to_string(i) + ":00.0";
      ON_CALL(*cuda_, getDevicePCIBusId(_, _, i))
          .WillByDefault([bdf](char* buf, int len, int) {
            strncpy(buf, bdf.c_str(), len);
            return Ok();
          });

      ON_CALL(*cuda_, deviceCanAccessPeer(i, _))
          .WillByDefault(Return(Result<bool>(true)));
    }
  }

  std::unique_ptr<Topology> createTopology() {
    return std::unique_ptr<Topology>(new Topology(cuda_, nvml_, ibv_, sysfs_));
  }

  std::shared_ptr<NiceMock<MockCudaApi>> cuda_;
  std::shared_ptr<NiceMock<MockNvmlApi>> nvml_;
  std::shared_ptr<NiceMock<MockIbvApi>> ibv_;
  std::shared_ptr<NiceMock<MockSysfsApi>> sysfs_;
};

// --- discover() tests ---

TEST_F(TopologyTest, DiscoverWithZeroGpusSucceeds) {
  setupGpus(0);
  // Verify IbvApi lifecycle: init, getDeviceList called.
  // freeDeviceList is guarded by a null check (unique_ptr deleter),
  // so it's NOT called when getDeviceList returns nullptr.
  EXPECT_CALL(*ibv_, init()).Times(Exactly(1));
  EXPECT_CALL(*ibv_, getDeviceList(_)).Times(Exactly(1));
  EXPECT_CALL(*ibv_, freeDeviceList(_)).Times(Exactly(0));
  auto topo = createTopology();
  EXPECT_TRUE(topo->available());
  EXPECT_EQ(topo->gpuCount(), 0u);
  EXPECT_EQ(topo->nicCount(), 0u);
  EXPECT_GE(topo->numaNodeCount(), 1u);
}

TEST_F(TopologyTest, DiscoverWithTwoGpusCreatesNodes) {
  setupGpus(2);
  // Verify CUDA context is initialized per GPU before querying PCI bus ID.
  EXPECT_CALL(*cuda_, setDevice(0)).Times(AtLeast(1));
  EXPECT_CALL(*cuda_, setDevice(1)).Times(AtLeast(1));
  EXPECT_CALL(*cuda_, getDevicePCIBusId(_, _, 0)).Times(Exactly(1));
  EXPECT_CALL(*cuda_, getDevicePCIBusId(_, _, 1)).Times(Exactly(1));
  // Verify original CUDA device is saved and restored.
  EXPECT_CALL(*cuda_, getDevice()).Times(AtLeast(1));
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  EXPECT_EQ(topo->gpuCount(), 2u);

  const auto& gpu0 = topo->getGpuNode(0);
  EXPECT_EQ(gpu0.type, NodeType::GPU);
  EXPECT_EQ(std::get<TopoNode::GpuData>(gpu0.data).cudaDeviceId, 0);
  EXPECT_EQ(std::get<TopoNode::GpuData>(gpu0.data).sm, 90);

  const auto& gpu1 = topo->getGpuNode(1);
  EXPECT_EQ(std::get<TopoNode::GpuData>(gpu1.data).cudaDeviceId, 1);
}

TEST_F(TopologyTest, DiscoverFailsWhenNvmlFails) {
  EXPECT_CALL(*nvml_, deviceCount())
      .WillOnce(Return(Err(ErrCode::DriverError, "nvml failed")));
  // CUDA should not be called if NVML fails first.
  EXPECT_CALL(*cuda_, getDevicePCIBusId(_, _, _)).Times(Exactly(0));
  auto topo = createTopology();
  EXPECT_FALSE(topo->available());
}

// --- P2P matrix tests ---

TEST_F(TopologyTest, P2PGpuAccessIsTrue) {
  setupGpus(2);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  EXPECT_TRUE(topo->canGpuAccess(0, 1));
  EXPECT_TRUE(topo->canGpuAccess(1, 0));
}

TEST_F(TopologyTest, P2PDisabledBetweenGpus) {
  setupGpus(2);
  // Verify Topology queries P2P for both directions.
  EXPECT_CALL(*cuda_, deviceCanAccessPeer(0, 1))
      .WillRepeatedly(Return(Result<bool>(false)));
  EXPECT_CALL(*cuda_, deviceCanAccessPeer(1, 0))
      .WillRepeatedly(Return(Result<bool>(false)));
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  EXPECT_FALSE(topo->canGpuAccess(0, 1));
  EXPECT_FALSE(topo->canGpuAccess(1, 0));
}

TEST_F(TopologyTest, P2POutOfBoundsReturnsFalse) {
  setupGpus(1);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  EXPECT_FALSE(topo->canGpuAccess(-1, 0));
  EXPECT_FALSE(topo->canGpuAccess(0, 999));
  EXPECT_FALSE(topo->canGpuAccess(999, 999));
}

// --- getPath() tests ---

TEST_F(TopologyTest, GetPathOutOfBoundsReturnsDisconnected) {
  setupGpus(1);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  EXPECT_EQ(topo->getPath(-1, 0).type, PathType::DIS);
  EXPECT_EQ(topo->getPath(0, 999).type, PathType::DIS);
  EXPECT_EQ(topo->getPath(999, 999).type, PathType::DIS);
}

TEST_F(TopologyTest, SelfPathIsOptimal) {
  setupGpus(1);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  int gpuNodeId = topo->getGpuNode(0).id;
  const auto& path = topo->getPath(gpuNodeId, gpuNodeId);
  EXPECT_EQ(path.type, PathType::NVL);
  EXPECT_GT(path.bw, 0u);
}

TEST_F(TopologyTest, GpuToCpuPathExists) {
  setupGpus(1);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  int gpuNodeId = topo->getGpuNode(0).id;
  int cpuNodeId = topo->getCpuNode(0).id;
  const auto& path = topo->getPath(gpuNodeId, cpuNodeId);
  EXPECT_NE(path.type, PathType::DIS);
}

TEST_F(TopologyTest, PathIsSymmetric) {
  setupGpus(2);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  int n0 = topo->getGpuNode(0).id;
  int n1 = topo->getGpuNode(1).id;
  const auto& fwd = topo->getPath(n0, n1);
  const auto& rev = topo->getPath(n1, n0);
  EXPECT_EQ(fwd.type, rev.type);
  EXPECT_EQ(fwd.bw, rev.bw);
}

// --- getPath() filter tests ---

TEST_F(TopologyTest, GetPathDefaultFilterExcludesC2CAndPxn) {
  setupGpus(1);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  // With default filter, C2C and PXN overrides should not be returned.
  // Since mocks don't produce C2C/PXN, baseline path should be returned.
  int gpuNodeId = topo->getGpuNode(0).id;
  int cpuNodeId = topo->getCpuNode(0).id;
  const auto& path = topo->getPath(gpuNodeId, cpuNodeId);
  EXPECT_NE(path.type, PathType::C2C);
  EXPECT_NE(path.type, PathType::PXN);
}

// --- Node access tests ---

TEST_F(TopologyTest, GetNodeThrowsOnInvalidIndex) {
  setupGpus(1);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  EXPECT_THROW(topo->getNode(-1), std::runtime_error);
  EXPECT_THROW(topo->getNode(9999), std::runtime_error);
}

TEST_F(TopologyTest, GetGpuNodeThrowsOnInvalidIndex) {
  setupGpus(1);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  EXPECT_NO_THROW(topo->getGpuNode(0));
  EXPECT_THROW(topo->getGpuNode(-1), std::runtime_error);
  EXPECT_THROW(topo->getGpuNode(999), std::runtime_error);
}

TEST_F(TopologyTest, GetNicNodeThrowsOnInvalidIndex) {
  setupGpus(0);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  // No NICs configured in mock.
  EXPECT_THROW(topo->getNicNode(0), std::runtime_error);
}

TEST_F(TopologyTest, GetCpuNodeThrowsOnInvalidIndex) {
  setupGpus(0);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  EXPECT_NO_THROW(topo->getCpuNode(0));
  EXPECT_THROW(topo->getCpuNode(999), std::runtime_error);
}

TEST_F(TopologyTest, CpuNodeHasCorrectNumaId) {
  setupGpus(0);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  const auto& cpu = topo->getCpuNode(0);
  EXPECT_EQ(cpu.type, NodeType::CPU);
  EXPECT_EQ(std::get<TopoNode::CpuData>(cpu.data).numaId, 0);
}

// --- Graph structure tests ---

TEST_F(TopologyTest, GpuNodeHasLinks) {
  setupGpus(1);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  const auto& gpu = topo->getGpuNode(0);
  // GPU should have at least a PHB link to its CPU node.
  EXPECT_FALSE(gpu.links.empty());
}

TEST_F(TopologyTest, CpuNodeHasLinks) {
  setupGpus(1);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  const auto& cpu = topo->getCpuNode(0);
  // CPU should have at least a link to the GPU.
  EXPECT_FALSE(cpu.links.empty());
}

TEST_F(TopologyTest, LinksAreBidirectional) {
  setupGpus(1);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  const auto& gpu = topo->getGpuNode(0);
  for (const auto& link : gpu.links) {
    const auto& peer = topo->getNode(link.peerNodeId);
    bool foundReverse = false;
    for (const auto& reverseLink : peer.links) {
      if (reverseLink.peerNodeId == gpu.id) {
        foundReverse = true;
        EXPECT_EQ(reverseLink.type, link.type);
        EXPECT_EQ(reverseLink.bw, link.bw);
        break;
      }
    }
    EXPECT_TRUE(foundReverse) << "Missing reverse link from node "
                              << link.peerNodeId << " to GPU " << gpu.id;
  }
}

// --- Multiple GPU tests ---

TEST_F(TopologyTest, FourGpusAllHaveNodes) {
  setupGpus(4);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  EXPECT_EQ(topo->gpuCount(), 4u);
  for (int i = 0; i < 4; ++i) {
    const auto& gpu = topo->getGpuNode(i);
    EXPECT_EQ(std::get<TopoNode::GpuData>(gpu.data).cudaDeviceId, i);
  }
}

TEST_F(TopologyTest, AllGpuPairsHavePaths) {
  setupGpus(4);
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      int ni = topo->getGpuNode(i).id;
      int nj = topo->getGpuNode(j).id;
      const auto& path = topo->getPath(ni, nj);
      EXPECT_NE(path.type, PathType::DIS)
          << "GPU " << i << " to GPU " << j << " is disconnected";
    }
  }
}

// --- IbvApi failure is non-fatal ---

TEST_F(TopologyTest, IbvInitFailureIsNonFatal) {
  setupGpus(1);
  EXPECT_CALL(*ibv_, init()).WillOnce(Return(Err(ErrCode::NotImplemented)));
  EXPECT_CALL(*ibv_, getDeviceList(_)).Times(Exactly(0));
  auto topo = createTopology();
  EXPECT_TRUE(topo->available());
  EXPECT_EQ(topo->gpuCount(), 1u);
  EXPECT_EQ(topo->nicCount(), 0u);
}

// --- Cross-NUMA SYS path tests ---

TEST_F(TopologyTest, TwoNumaNodesProducesSysPath) {
  setupGpus(2);
  ON_CALL(*sysfs_, listDir("/sys/devices/system/node", "node"))
      .WillByDefault(Return(std::vector<std::string>{"node0", "node1"}));

  // GPU0 on NUMA 0, GPU1 on NUMA 1.
  ON_CALL(*sysfs_, readFile(testing::HasSubstr("0000:00:00.0/numa_node")))
      .WillByDefault(Return("0"));
  ON_CALL(*sysfs_, readFile(testing::HasSubstr("0000:01:00.0/numa_node")))
      .WillByDefault(Return("1"));

  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  EXPECT_EQ(topo->numaNodeCount(), 2u);

  // GPUs on different NUMA nodes should have SYS path.
  int n0 = topo->getGpuNode(0).id;
  int n1 = topo->getGpuNode(1).id;
  const auto& path = topo->getPath(n0, n1);
  EXPECT_EQ(path.type, PathType::SYS);
}

// --- Sysfs-based PCI hierarchy tests ---

class TopologyPciTest : public TopologyTest {
 protected:
  void SetUp() override {
    TopologyTest::SetUp();
    setupGpus(2);

    // Build a sysfs hierarchy where GPU0 and GPU1 share a PCIe switch:
    //   /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0  (GPU0)
    //   /sys/devices/pci0000:00/0000:00:01.0/0000:02:00.0  (GPU1)
    // Common ancestor = 0000:00:01.0 (one switch → PIX)
    const std::string gpu0Sysfs =
        "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0";
    const std::string gpu1Sysfs =
        "/sys/devices/pci0000:00/0000:00:01.0/0000:02:00.0";
    const std::string switchSysfs = "/sys/devices/pci0000:00/0000:00:01.0";
    const std::string rootSysfs = "/sys/devices/pci0000:00";

    // resolvePath: PCI device symlinks → real sysfs paths
    ON_CALL(*sysfs_, resolvePath("/sys/bus/pci/devices/0000:00:00.0"))
        .WillByDefault(Return(Result<std::string>(gpu0Sysfs)));
    ON_CALL(*sysfs_, resolvePath("/sys/bus/pci/devices/0000:01:00.0"))
        .WillByDefault(Return(Result<std::string>(gpu1Sysfs)));

    // Ancestor chain: GPU0 → switch → root (not PCI BDF)
    ON_CALL(*sysfs_, resolvePath(gpu0Sysfs + "/.."))
        .WillByDefault(Return(Result<std::string>(switchSysfs)));
    ON_CALL(*sysfs_, resolvePath(gpu1Sysfs + "/.."))
        .WillByDefault(Return(Result<std::string>(switchSysfs)));
    ON_CALL(*sysfs_, resolvePath(switchSysfs + "/.."))
        .WillByDefault(Return(Result<std::string>(rootSysfs)));

    // PCIe link info for bandwidth computation.
    ON_CALL(*sysfs_, readFile(gpu0Sysfs + "/max_link_speed"))
        .WillByDefault(Return("16 GT/s"));
    ON_CALL(*sysfs_, readFile(gpu0Sysfs + "/max_link_width"))
        .WillByDefault(Return("16"));
    ON_CALL(*sysfs_, readFile(gpu1Sysfs + "/max_link_speed"))
        .WillByDefault(Return("16 GT/s"));
    ON_CALL(*sysfs_, readFile(gpu1Sysfs + "/max_link_width"))
        .WillByDefault(Return("16"));
    ON_CALL(*sysfs_, readFile(switchSysfs + "/max_link_speed"))
        .WillByDefault(Return("16 GT/s"));
    ON_CALL(*sysfs_, readFile(switchSysfs + "/max_link_width"))
        .WillByDefault(Return("16"));
    // Upstream port link info (parent of each GPU).
    ON_CALL(*sysfs_, readFile(gpu0Sysfs + "/../max_link_speed"))
        .WillByDefault(Return("16 GT/s"));
    ON_CALL(*sysfs_, readFile(gpu0Sysfs + "/../max_link_width"))
        .WillByDefault(Return("16"));
    ON_CALL(*sysfs_, readFile(gpu1Sysfs + "/../max_link_speed"))
        .WillByDefault(Return("16 GT/s"));
    ON_CALL(*sysfs_, readFile(gpu1Sysfs + "/../max_link_width"))
        .WillByDefault(Return("16"));

    // NUMA node for both GPUs.
    ON_CALL(*sysfs_, readFile(gpu0Sysfs + "/numa_node"))
        .WillByDefault(Return("0"));
    ON_CALL(*sysfs_, readFile(gpu1Sysfs + "/numa_node"))
        .WillByDefault(Return("0"));
  }
};

TEST_F(TopologyPciTest, SameSwitchGpusGetPixPath) {
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());

  int n0 = topo->getGpuNode(0).id;
  int n1 = topo->getGpuNode(1).id;
  const auto& path = topo->getPath(n0, n1);
  EXPECT_EQ(path.type, PathType::PIX);
  EXPECT_GT(path.bw, 0u);
}

TEST_F(TopologyPciTest, PcieBandwidthIsCorrect) {
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());

  int n0 = topo->getGpuNode(0).id;
  int n1 = topo->getGpuNode(1).id;
  const auto& path = topo->getPath(n0, n1);
  // Gen4 x16: 12000 Mbps/lane * 16 lanes / 8 = 24000 MB/s
  EXPECT_EQ(path.bw, 24000u);
}

// --- NVLink path tests ---

class TopologyNvLinkTest : public TopologyTest {
 protected:
  void SetUp() override {
    TopologyTest::SetUp();
    setupGpus(2);

    // Enable NVLink: each GPU has 18 NVSwitch-connected links (H100-like).
    // Override nvmlDeviceGetFieldValues to handle both NVLink state queries
    // and C2C queries. NVLink state → ENABLED, C2C count → 0 (no C2C).
    ON_CALL(*nvml_, nvmlDeviceGetFieldValues(_, _, _))
        .WillByDefault([](nvmlDevice_t, int count, nvmlFieldValue_t* fvs) {
          for (int i = 0; i < count; ++i) {
            fvs[i].nvmlReturn = NVML_SUCCESS;
            if (fvs[i].fieldId == NVML_FI_DEV_NVLINK_GET_STATE) {
              fvs[i].value.uiVal = NVML_FEATURE_ENABLED;
            }
            // C2C fields: uiVal stays 0 (zero-initialized by caller).
          }
          return Ok();
        });

    for (int gpu = 0; gpu < 2; ++gpu) {
      auto info = nvml_->deviceInfo(gpu);
      ASSERT_TRUE(info.hasValue());
      nvmlDevice_t handle = info.value().handle;

      ON_CALL(
          *nvml_,
          nvmlDeviceGetNvLinkCapability(
              handle, _, NVML_NVLINK_CAP_P2P_SUPPORTED, _))
          .WillByDefault([](nvmlDevice_t,
                            unsigned int,
                            nvmlNvLinkCapability_t,
                            unsigned int* output) {
            *output = 1;
            return Ok();
          });

      // Fallback path for CUDART_VERSION < 11080.
      ON_CALL(*nvml_, nvmlDeviceGetNvLinkState(handle, _, _))
          .WillByDefault(
              [](nvmlDevice_t, unsigned int, nvmlEnableState_t* isActive) {
                *isActive = NVML_FEATURE_ENABLED;
                return Ok();
              });

      // Remote PCI info: return sentinel BDF → NVSwitch.
      ON_CALL(*nvml_, nvmlDeviceGetNvLinkRemotePciInfo(handle, _, _))
          .WillByDefault([](nvmlDevice_t, unsigned int, nvmlPciInfo_t* pci) {
            strncpy(pci->busId, "fffffff:ffff:ff", sizeof(pci->busId));
            return Ok();
          });
    }
  }
};

TEST_F(TopologyNvLinkTest, NvSwitchConnectedGpusGetNvlPath) {
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());

  int n0 = topo->getGpuNode(0).id;
  int n1 = topo->getGpuNode(1).id;
  const auto& path = topo->getPath(n0, n1);
  EXPECT_EQ(path.type, PathType::NVL);
  EXPECT_GT(path.bw, 0u);
}

TEST_F(TopologyNvLinkTest, NvLinkBandwidthIsReasonable) {
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());

  int n0 = topo->getGpuNode(0).id;
  int n1 = topo->getGpuNode(1).id;
  const auto& path = topo->getPath(n0, n1);
  // SM 90 with 18 NVSwitch links: 18 * 20600 = 370800 MB/s.
  EXPECT_EQ(path.bw, 370800u);
}

TEST_F(TopologyNvLinkTest, NvLinkPathIsPreferredOverPcie) {
  // Also set up a sysfs hierarchy so both PCIe and NVLink paths exist.
  const std::string gpu0Sysfs =
      "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0";
  const std::string gpu1Sysfs =
      "/sys/devices/pci0000:00/0000:00:01.0/0000:02:00.0";
  const std::string switchSysfs = "/sys/devices/pci0000:00/0000:00:01.0";
  const std::string rootSysfs = "/sys/devices/pci0000:00";

  ON_CALL(*sysfs_, resolvePath("/sys/bus/pci/devices/0000:00:00.0"))
      .WillByDefault(Return(Result<std::string>(gpu0Sysfs)));
  ON_CALL(*sysfs_, resolvePath("/sys/bus/pci/devices/0000:01:00.0"))
      .WillByDefault(Return(Result<std::string>(gpu1Sysfs)));
  ON_CALL(*sysfs_, resolvePath(gpu0Sysfs + "/.."))
      .WillByDefault(Return(Result<std::string>(switchSysfs)));
  ON_CALL(*sysfs_, resolvePath(gpu1Sysfs + "/.."))
      .WillByDefault(Return(Result<std::string>(switchSysfs)));
  ON_CALL(*sysfs_, resolvePath(switchSysfs + "/.."))
      .WillByDefault(Return(Result<std::string>(rootSysfs)));

  auto topo = createTopology();
  ASSERT_TRUE(topo->available());

  int n0 = topo->getGpuNode(0).id;
  int n1 = topo->getGpuNode(1).id;
  const auto& path = topo->getPath(n0, n1);
  // NVLink should win over PIX since NVL < PIX in the enum ordering.
  EXPECT_EQ(path.type, PathType::NVL);
}

// --- NicFilter tests ---

TEST(NicFilterTest, EmptyFilterMatchesEverything) {
  NicFilter filter;
  EXPECT_TRUE(filter.empty());
  EXPECT_TRUE(filter.matches("mlx5_0"));
  EXPECT_TRUE(filter.matches("bnxt_re0"));
  EXPECT_TRUE(filter.matches("anything"));
}

TEST(NicFilterTest, PrefixInclude) {
  NicFilter filter("mlx5");
  EXPECT_TRUE(filter.matches("mlx5_0"));
  EXPECT_TRUE(filter.matches("mlx5_10"));
  EXPECT_FALSE(filter.matches("bnxt_re0"));
}

TEST(NicFilterTest, PrefixIncludeMultipleEntries) {
  NicFilter filter("mlx5_0,mlx5_3");
  EXPECT_TRUE(filter.matches("mlx5_0"));
  EXPECT_TRUE(filter.matches("mlx5_3"));
  EXPECT_FALSE(filter.matches("mlx5_1"));
  EXPECT_FALSE(filter.matches("bnxt_re0"));
}

TEST(NicFilterTest, ExactInclude) {
  NicFilter filter("=mlx5_0");
  EXPECT_TRUE(filter.matches("mlx5_0"));
  EXPECT_FALSE(filter.matches("mlx5_0_extra"));
  EXPECT_FALSE(filter.matches("mlx5_1"));
}

TEST(NicFilterTest, PrefixExclude) {
  NicFilter filter("^bnxt_re");
  EXPECT_TRUE(filter.matches("mlx5_0"));
  EXPECT_TRUE(filter.matches("mlx5_1"));
  EXPECT_FALSE(filter.matches("bnxt_re0"));
  EXPECT_FALSE(filter.matches("bnxt_re1"));
}

TEST(NicFilterTest, ExactExclude) {
  NicFilter filter("^=mlx5_1");
  EXPECT_TRUE(filter.matches("mlx5_0"));
  EXPECT_TRUE(filter.matches("mlx5_1_extra"));
  EXPECT_FALSE(filter.matches("mlx5_1"));
}

TEST(NicFilterTest, PortFiltering) {
  NicFilter filter("mlx5_0:1");
  EXPECT_TRUE(filter.matches("mlx5_0", 1));
  EXPECT_FALSE(filter.matches("mlx5_0", 2));
  // Port -1 means "don't care" — matches any port in the entry.
  EXPECT_TRUE(filter.matches("mlx5_0", -1));
  EXPECT_FALSE(filter.matches("mlx5_1", 1));
}

TEST(NicFilterTest, WhitespaceIsTrimmed) {
  NicFilter filter("  mlx5_0 , mlx5_1 ");
  EXPECT_TRUE(filter.matches("mlx5_0"));
  EXPECT_TRUE(filter.matches("mlx5_1"));
}

// --- Topology with NIC tests ---

// PCI hierarchy for NIC selection tests:
//   root (pci0000:00)
//   ├── switch0 (0000:00:01.0)
//   │   ├── GPU0 (0000:01:00.0)        ← PIX to nic0, nic1
//   │   ├── nic0/mlx5_0 (0000:02:00.0) ← PIX to GPU0
//   │   └── nic1/mlx5_1 (0000:03:00.0) ← PIX to GPU0
//   └── switch1 (0000:00:02.0)
//       └── nic2/mlx5_2 (0000:04:00.0) ← PXB to GPU0
class TopologyNicTest : public TopologyTest {
 protected:
  void SetUp() override {
    TopologyTest::SetUp();
    setupGpus(1);

    // PCI sysfs paths.
    gpu0_ = "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0";
    nic0_ = "/sys/devices/pci0000:00/0000:00:01.0/0000:02:00.0";
    nic1_ = "/sys/devices/pci0000:00/0000:00:01.0/0000:03:00.0";
    nic2_ = "/sys/devices/pci0000:00/0000:00:02.0/0000:04:00.0";
    sw0_ = "/sys/devices/pci0000:00/0000:00:01.0";
    sw1_ = "/sys/devices/pci0000:00/0000:00:02.0";
    root_ = "/sys/devices/pci0000:00";

    // GPU0 sysfs resolve.
    ON_CALL(*sysfs_, resolvePath("/sys/bus/pci/devices/0000:00:00.0"))
        .WillByDefault(Return(Result<std::string>(gpu0_)));

    // Ancestor chains.
    for (const auto& dev : {gpu0_, nic0_, nic1_}) {
      ON_CALL(*sysfs_, resolvePath(dev + "/.."))
          .WillByDefault(Return(Result<std::string>(sw0_)));
    }
    ON_CALL(*sysfs_, resolvePath(nic2_ + "/.."))
        .WillByDefault(Return(Result<std::string>(sw1_)));
    ON_CALL(*sysfs_, resolvePath(sw0_ + "/.."))
        .WillByDefault(Return(Result<std::string>(root_)));
    ON_CALL(*sysfs_, resolvePath(sw1_ + "/.."))
        .WillByDefault(Return(Result<std::string>(root_)));

    // Link info for all PCI devices (Gen4 x16).
    for (const auto& dev : {gpu0_, nic0_, nic1_, nic2_, sw0_, sw1_}) {
      ON_CALL(*sysfs_, readFile(dev + "/max_link_speed"))
          .WillByDefault(Return("16 GT/s"));
      ON_CALL(*sysfs_, readFile(dev + "/max_link_width"))
          .WillByDefault(Return("16"));
      ON_CALL(*sysfs_, readFile(dev + "/../max_link_speed"))
          .WillByDefault(Return("16 GT/s"));
      ON_CALL(*sysfs_, readFile(dev + "/../max_link_width"))
          .WillByDefault(Return("16"));
      ON_CALL(*sysfs_, readFile(dev + "/numa_node")).WillByDefault(Return("0"));
    }
  }

  void setupNics(const std::vector<std::pair<std::string, std::string>>& nics) {
    // nics = {{"mlx5_0", nicSysfsPath}, ...}
    nicDevices_.resize(nics.size());
    nicDevicePtrs_.resize(nics.size());
    nicNames_.clear();

    for (size_t i = 0; i < nics.size(); ++i) {
      std::memset(&nicDevices_[i], 0, sizeof(ibv_device));
      std::string ibdevPath = "/sys/class/infiniband/" + nics[i].first;
      std::strncpy(
          nicDevices_[i].ibdev_path,
          ibdevPath.c_str(),
          sizeof(nicDevices_[i].ibdev_path) - 1);
      nicDevicePtrs_[i] = &nicDevices_[i];
      nicNames_.push_back(nics[i].first);

      // Resolve ibdev_path/device → PCI device path.
      ON_CALL(*sysfs_, resolvePath(ibdevPath + "/device"))
          .WillByDefault(Return(Result<std::string>(nics[i].second)));
    }

    int n = static_cast<int>(nics.size());
    ON_CALL(*ibv_, getDeviceList(_))
        .WillByDefault([this, n](int* numDevices) -> Result<ibv_device**> {
          *numDevices = n;
          return nicDevicePtrs_.data();
        });

    ON_CALL(*ibv_, freeDeviceList(_)).WillByDefault(Return(Ok()));

    for (size_t i = 0; i < nics.size(); ++i) {
      ON_CALL(*ibv_, getDeviceName(&nicDevices_[i]))
          .WillByDefault([this, i](ibv_device*) -> Result<const char*> {
            return nicNames_[i].c_str();
          });

      ON_CALL(*ibv_, openDevice(&nicDevices_[i]))
          .WillByDefault([i](ibv_device*) -> Result<ibv_context*> {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            return reinterpret_cast<ibv_context*>(
                static_cast<uintptr_t>(0x100 + i));
          });
    }

    ON_CALL(*ibv_, queryDevice(_, _))
        .WillByDefault([](ibv_context*, ibv_device_attr* attr) {
          attr->phys_port_cnt = 1;
          return Ok();
        });

    ON_CALL(*ibv_, queryPort(_, _, _))
        .WillByDefault([](ibv_context*, uint8_t, ibv_port_attr* attr) {
          attr->state = IBV_PORT_ACTIVE;
          attr->active_speed = 128; // NDR (bit 7)
          attr->active_width = 2; // 4x (bit 1)
          return Ok();
        });

    ON_CALL(*ibv_, closeDevice(_)).WillByDefault(Return(Ok()));
  }

  std::string gpu0_, nic0_, nic1_, nic2_, sw0_, sw1_, root_;
  std::vector<ibv_device> nicDevices_;
  std::vector<ibv_device*> nicDevicePtrs_;
  std::vector<std::string> nicNames_;
};

TEST_F(TopologyNicTest, NicNodesAreDiscovered) {
  setupNics({{"mlx5_0", nic0_}, {"mlx5_1", nic1_}});
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  EXPECT_EQ(topo->nicCount(), 2u);
  EXPECT_EQ(topo->getNicNode(0).name, "mlx5_0");
  EXPECT_EQ(topo->getNicNode(1).name, "mlx5_1");
}

TEST_F(TopologyNicTest, ZeroNicsIsValid) {
  // Default mock: no IB devices. GPU-only topology should work.
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  EXPECT_EQ(topo->gpuCount(), 1u);
  EXPECT_EQ(topo->nicCount(), 0u);
}

TEST_F(TopologyNicTest, SameSwitchNicGetPixPath) {
  setupNics({{"mlx5_0", nic0_}});
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());

  int gpuNodeId = topo->getGpuNode(0).id;
  int nicNodeId = topo->getNicNode(0).id;
  const auto& path = topo->getPath(gpuNodeId, nicNodeId);
  EXPECT_EQ(path.type, PathType::PIX);
  EXPECT_GT(path.bw, 0u);
}

TEST_F(TopologyNicTest, DifferentSwitchNicGetPhbPath) {
  // nic2 is under a different PCIe switch that shares no PCI BDF ancestor
  // with GPU0's switch. Path goes through the CPU node → PHB.
  setupNics({{"mlx5_2", nic2_}});
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());

  int gpuNodeId = topo->getGpuNode(0).id;
  int nicNodeId = topo->getNicNode(0).id;
  const auto& path = topo->getPath(gpuNodeId, nicNodeId);
  EXPECT_EQ(path.type, PathType::PHB);
}

TEST_F(TopologyNicTest, CloserNicHasBetterPath) {
  // nic0 under same switch as GPU (PIX), nic2 under different switch (PXB).
  setupNics({{"mlx5_0", nic0_}, {"mlx5_2", nic2_}});
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  ASSERT_EQ(topo->nicCount(), 2u);

  int gpuNodeId = topo->getGpuNode(0).id;
  const auto& pathToNic0 = topo->getPath(gpuNodeId, topo->getNicNode(0).id);
  const auto& pathToNic2 = topo->getPath(gpuNodeId, topo->getNicNode(1).id);

  // PIX < PHB — closer NIC has a better (lower) path type.
  EXPECT_LT(pathToNic0.type, pathToNic2.type);
  EXPECT_EQ(pathToNic0.type, PathType::PIX);
  EXPECT_EQ(pathToNic2.type, PathType::PHB);
}

TEST_F(TopologyNicTest, MultiRailNicsAtSameDistance) {
  // Two NICs under the same switch as GPU → both PIX, same bandwidth.
  setupNics({{"mlx5_0", nic0_}, {"mlx5_1", nic1_}});
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  ASSERT_EQ(topo->nicCount(), 2u);

  int gpuNodeId = topo->getGpuNode(0).id;
  const auto& path0 = topo->getPath(gpuNodeId, topo->getNicNode(0).id);
  const auto& path1 = topo->getPath(gpuNodeId, topo->getNicNode(1).id);

  EXPECT_EQ(path0.type, PathType::PIX);
  EXPECT_EQ(path1.type, PathType::PIX);
  EXPECT_EQ(path0.bw, path1.bw);
}

TEST_F(TopologyNicTest, FilterNicByPrefix) {
  setupNics({{"mlx5_0", nic0_}, {"bnxt_re0", nic1_}});
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  ASSERT_EQ(topo->nicCount(), 2u);

  NicFilter mlxOnly("mlx5");
  EXPECT_TRUE(topo->filterNic(0, mlxOnly));
  EXPECT_FALSE(topo->filterNic(1, mlxOnly));

  NicFilter excludeMlx("^mlx5");
  EXPECT_FALSE(topo->filterNic(0, excludeMlx));
  EXPECT_TRUE(topo->filterNic(1, excludeMlx));
}

TEST_F(TopologyNicTest, NodeNamesAreSet) {
  setupNics({{"mlx5_0", nic0_}});
  auto topo = createTopology();
  ASSERT_TRUE(topo->available());

  EXPECT_EQ(topo->getGpuNode(0).name, "cuda:0");
  EXPECT_EQ(topo->getCpuNode(0).name, "cpu:0");
  EXPECT_EQ(topo->getNicNode(0).name, "mlx5_0");
}

TEST_F(TopologyNicTest, PortSpeedIsCapturedFromIbverbs) {
  setupNics({{"mlx5_0", nic0_}, {"mlx5_1", nic1_}});

  // mlx5_0 (ctx 0x100): NDR 4x = 400000 Mbps
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto* ctx0 = reinterpret_cast<ibv_context*>(static_cast<uintptr_t>(0x100));
  ON_CALL(*ibv_, queryPort(ctx0, _, _))
      .WillByDefault([](ibv_context*, uint8_t, ibv_port_attr* attr) {
        attr->state = IBV_PORT_ACTIVE;
        attr->active_speed = 128; // NDR (bit 7)
        attr->active_width = 2; // 4x (bit 1)
        return Ok();
      });

  // mlx5_1 (ctx 0x101): HDR 4x = 200000 Mbps
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto* ctx1 = reinterpret_cast<ibv_context*>(static_cast<uintptr_t>(0x101));
  ON_CALL(*ibv_, queryPort(ctx1, _, _))
      .WillByDefault([](ibv_context*, uint8_t, ibv_port_attr* attr) {
        attr->state = IBV_PORT_ACTIVE;
        attr->active_speed = 64; // HDR (bit 6)
        attr->active_width = 2; // 4x (bit 1)
        return Ok();
      });

  auto topo = createTopology();
  ASSERT_TRUE(topo->available());
  ASSERT_EQ(topo->nicCount(), 2u);

  auto& nic0Data = std::get<TopoNode::NicData>(topo->getNicNode(0).data);
  auto& nic1Data = std::get<TopoNode::NicData>(topo->getNicNode(1).data);

  EXPECT_EQ(nic0Data.portSpeedMbps, 400000u); // NDR 4x
  EXPECT_EQ(nic1Data.portSpeedMbps, 200000u); // HDR 4x
  EXPECT_GT(nic0Data.portSpeedMbps, nic1Data.portSpeedMbps);
}

} // namespace uniflow
