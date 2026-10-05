// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/Topology.h"

#include "comms/uniflow/drivers/TopologyDiscovery.h"
#include "comms/uniflow/drivers/cuda/CudaTopologyDiscovery.h"
#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
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

    // Default: CUDA device save/restore.
    ON_CALL(*cuda_, getDevice()).WillByDefault(Return(Result<int>(0)));
    ON_CALL(*cuda_, setDevice(_)).WillByDefault(Return(Ok()));
  }

  void setupGpus(int count) {
    ON_CALL(*nvml_, deviceCount()).WillByDefault(Return(Result<int>(count)));
    ON_CALL(*cuda_, getDeviceCount()).WillByDefault(Return(Result<int>(count)));
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

      // Resolve NVML handle by PCI bus ID (matches discoverGpus logic).
      ON_CALL(*nvml_, nvmlDeviceGetHandleByPciBusId(_, _))
          .WillByDefault([](const char*, nvmlDevice_t* dev) {
            // Return a non-null handle for any valid PCI bus ID.
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            *dev = reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(1));
            return Ok();
          });

      ON_CALL(*nvml_, nvmlDeviceGetCudaComputeCapability(_, _, _))
          .WillByDefault([](nvmlDevice_t, int* major, int* minor) {
            *major = 9;
            *minor = 0;
            return Ok();
          });

      ON_CALL(*cuda_, deviceCanAccessPeer(i, _))
          .WillByDefault(Return(Result<bool>(true)));
    }
  }

  std::unique_ptr<Topology> createTopology() {
    auto topo = std::make_unique<Topology>();
    CudaTopologyDiscovery(cuda_, nvml_, nullptr, sysfs_).discover(*topo);
    return topo;
  }

  std::shared_ptr<NiceMock<MockCudaApi>> cuda_;
  std::shared_ptr<NiceMock<MockNvmlApi>> nvml_;
  std::shared_ptr<NiceMock<MockSysfsApi>> sysfs_;
};

// --- discover() tests ---

TEST_F(TopologyTest, DiscoverWithZeroGpusSucceeds) {
  setupGpus(0);
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

TEST_F(TopologyTest, DiscoverMapsReorderedCudaOrdinalsToNvmlByPciBusId) {
  ON_CALL(*cuda_, getDeviceCount()).WillByDefault(Return(Result<int>(2)));
  ON_CALL(*cuda_, deviceCanAccessPeer(_, _))
      .WillByDefault(Return(Result<bool>(true)));

  ON_CALL(*cuda_, getDevicePCIBusId(_, _, 0))
      .WillByDefault([](char* buf, int len, int) {
        strncpy(buf, "0000:19:00.0", len);
        return Ok();
      });
  ON_CALL(*cuda_, getDevicePCIBusId(_, _, 1))
      .WillByDefault([](char* buf, int len, int) {
        strncpy(buf, "0000:09:00.0", len);
        return Ok();
      });

  // CUDA ordinal order can differ from physical NVML index order when
  // CUDA_VISIBLE_DEVICES filters or reorders GPUs. Discovery must enrich each
  // CUDA-visible GPU through its PCI bus ID instead of assuming ordinal == NVML
  // index.
  EXPECT_CALL(*nvml_, deviceCount()).Times(Exactly(0));
  EXPECT_CALL(*nvml_, nvmlDeviceGetHandleByPciBusId(_, _))
      .Times(Exactly(2))
      .WillRepeatedly([](const char* busId, nvmlDevice_t* dev) -> Status {
        if (strcmp(busId, "0000:19:00.0") == 0) {
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          *dev = reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(19));
          return Ok();
        }
        if (strcmp(busId, "0000:09:00.0") == 0) {
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          *dev = reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(9));
          return Ok();
        }
        return Err(ErrCode::InvalidArgument, "unexpected PCI bus ID");
      });
  EXPECT_CALL(*nvml_, nvmlDeviceGetCudaComputeCapability(_, _, _))
      .Times(Exactly(2))
      .WillRepeatedly([](nvmlDevice_t dev, int* major, int* minor) {
        auto handle = reinterpret_cast<uintptr_t>(dev);
        *major = handle == 19 ? 9 : 8;
        *minor = 0;
        return Ok();
      });

  auto topo = createTopology();

  ASSERT_TRUE(topo->available());
  ASSERT_EQ(topo->gpuCount(), 2u);
  const auto& cudaOrdinal0 =
      std::get<TopoNode::GpuData>(topo->getGpuNode(0).data);
  EXPECT_EQ(cudaOrdinal0.cudaDeviceId, 0);
  EXPECT_EQ(cudaOrdinal0.bdf, "0000:19:00.0");
  EXPECT_EQ(cudaOrdinal0.sm, 90);

  const auto& cudaOrdinal1 =
      std::get<TopoNode::GpuData>(topo->getGpuNode(1).data);
  EXPECT_EQ(cudaOrdinal1.cudaDeviceId, 1);
  EXPECT_EQ(cudaOrdinal1.bdf, "0000:09:00.0");
  EXPECT_EQ(cudaOrdinal1.sm, 80);
}

TEST_F(TopologyTest, DiscoverContinuesWhenNvmlFails) {
  setupGpus(1);

  // GPU enumeration is driven by CUDA_VISIBLE_DEVICES-aware CUDA APIs. NVML is
  // only used to enrich each CUDA-visible GPU with NVLink/C2C details, so an
  // NVML handle lookup failure must not discard the CUDA-visible GPU.
  EXPECT_CALL(*nvml_, deviceCount()).Times(Exactly(0));
  EXPECT_CALL(*nvml_, nvmlDeviceGetHandleByPciBusId(_, _))
      .WillOnce(Return(Err(ErrCode::DriverError, "nvml failed")));
  EXPECT_CALL(*cuda_, getDevicePCIBusId(_, _, 0)).Times(Exactly(1));
  auto topo = createTopology();
  // NVML failure is non-fatal; topology is still available with the
  // CUDA-visible GPU and without NVML-derived link metadata.
  EXPECT_TRUE(topo->available());
  EXPECT_EQ(topo->gpuCount(), 1u);
}

TEST_F(TopologyTest, DiscoverContinuesWhenCudaDeviceCountFails) {
  EXPECT_CALL(*cuda_, getDeviceCount())
      .WillOnce(Return(Err(ErrCode::DriverError, "cuda failed")));
  EXPECT_CALL(*cuda_, getDevicePCIBusId(_, _, _)).Times(Exactly(0));
  auto topo = createTopology();
  // GPU failure is non-fatal; topology is still available with zero GPUs.
  EXPECT_TRUE(topo->available());
  EXPECT_EQ(topo->gpuCount(), 0u);
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

} // namespace uniflow
