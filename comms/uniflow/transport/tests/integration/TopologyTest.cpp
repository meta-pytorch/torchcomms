// Copyright (c) Meta Platforms, Inc. and affiliates.

/// Integration test for Topology on H100 systems.
/// Requires real GPUs, NVML, and ibverbs — not for CI without GPU hardware.

#include "comms/uniflow/transport/Topology.h"

#include <gtest/gtest.h>

using namespace uniflow;

class TopologyIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    topo_ = &Topology::get();
    ASSERT_TRUE(topo_->available());
  }

  Topology& topo() {
    return *topo_;
  }

 private:
  Topology* topo_{nullptr};
};

TEST_F(TopologyIntegrationTest, DiscoverFindsGpus) {
  EXPECT_GT(topo().gpuCount(), 0u);
}

TEST_F(TopologyIntegrationTest, DiscoverFindsNumaNodes) {
  EXPECT_GT(topo().numaNodeCount(), 0u);
}

TEST_F(TopologyIntegrationTest, GpuNodesAreValid) {
  for (size_t i = 0; i < topo().gpuCount(); ++i) {
    const auto& node = topo().getGpuNode(i);
    EXPECT_EQ(node.type, NodeType::GPU);
    const auto& gpuData = std::get<TopoNode::GpuData>(node.data);
    EXPECT_EQ(gpuData.cudaDeviceId, static_cast<int>(i));
    EXPECT_FALSE(gpuData.bdf.empty());
  }
}

TEST_F(TopologyIntegrationTest, CpuNodesAreValid) {
  for (size_t i = 0; i < topo().numaNodeCount(); ++i) {
    const auto& node = topo().getCpuNode(i);
    EXPECT_EQ(node.type, NodeType::CPU);
    EXPECT_EQ(
        std::get<TopoNode::CpuData>(node.data).numaId, static_cast<int>(i));
  }
}

TEST_F(TopologyIntegrationTest, NicNodesAreValid) {
  for (size_t i = 0; i < topo().nicCount(); ++i) {
    const auto& node = topo().getNicNode(i);
    EXPECT_EQ(node.type, NodeType::NIC);
    EXPECT_FALSE(node.name.empty());
    const auto& nicData = std::get<TopoNode::NicData>(node.data);
    EXPECT_NE(nicData.port, -1);
  }
}

TEST_F(TopologyIntegrationTest, GpuSelfPathIsOptimal) {
  for (size_t i = 0; i < topo().gpuCount(); ++i) {
    int nodeId = topo().getGpuNode(i).id;
    const auto& path = topo().getPath(nodeId, nodeId);
    EXPECT_EQ(path.type, PathType::NVL);
    EXPECT_GT(path.bw, 0u);
  }
}

TEST_F(TopologyIntegrationTest, GpuToGpuPathExists) {
  if (topo().gpuCount() < 2) {
    GTEST_SKIP() << "Need at least 2 GPUs";
  }
  int node0 = topo().getGpuNode(0).id;
  int node1 = topo().getGpuNode(1).id;
  const auto& path = topo().getPath(node0, node1);
  EXPECT_NE(path.type, PathType::DIS) << "GPU 0 and GPU 1 are disconnected";
  EXPECT_GT(path.bw, 0u);
}

TEST_F(TopologyIntegrationTest, P2PMatrixIsSymmetric) {
  for (size_t i = 0; i < topo().gpuCount(); ++i) {
    EXPECT_TRUE(topo().canGpuAccess(i, i))
        << "GPU " << i << " cannot self-access";
    for (size_t j = i + 1; j < topo().gpuCount(); ++j) {
      EXPECT_EQ(topo().canGpuAccess(i, j), topo().canGpuAccess(j, i))
          << "P2P asymmetry between GPU " << i << " and " << j;
    }
  }
}

TEST_F(TopologyIntegrationTest, PathSymmetry) {
  for (size_t i = 0; i < topo().gpuCount(); ++i) {
    for (size_t j = i + 1; j < topo().gpuCount(); ++j) {
      int ni = topo().getGpuNode(i).id;
      int nj = topo().getGpuNode(j).id;
      const auto& fwd = topo().getPath(ni, nj);
      const auto& rev = topo().getPath(nj, ni);
      EXPECT_EQ(fwd.type, rev.type)
          << "Path type asymmetry between GPU " << i << " and " << j;
      EXPECT_EQ(fwd.bw, rev.bw)
          << "Path BW asymmetry between GPU " << i << " and " << j;
    }
  }
}

TEST_F(TopologyIntegrationTest, NodeAccessThrowsOnInvalidIndex) {
  EXPECT_THROW(topo().getNode(-1), std::runtime_error);
  EXPECT_THROW(topo().getGpuNode(999), std::runtime_error);
  EXPECT_THROW(topo().getNicNode(999), std::runtime_error);
  EXPECT_THROW(topo().getCpuNode(999), std::runtime_error);
}

// --- H100-specific tests (8 GPUs, NVSwitch, NICs) ---

class H100TopologyTest : public TopologyIntegrationTest {
 protected:
  void SetUp() override {
    TopologyIntegrationTest::SetUp();
    if (topo().gpuCount() != 8) {
      GTEST_SKIP() << "Not an 8-GPU system (got " << topo().gpuCount() << ")";
    }
    const auto& gpuData =
        std::get<TopoNode::GpuData>(topo().getGpuNode(0).data);
    if (gpuData.sm < 90) {
      GTEST_SKIP() << "Not H100 (SM " << gpuData.sm << ")";
    }
  }
};

TEST_F(H100TopologyTest, AllGpuPairsHaveNvlinkPath) {
  for (size_t i = 0; i < topo().gpuCount(); ++i) {
    for (size_t j = i + 1; j < topo().gpuCount(); ++j) {
      int ni = topo().getGpuNode(i).id;
      int nj = topo().getGpuNode(j).id;
      const auto& path = topo().getPath(ni, nj);
      EXPECT_EQ(path.type, PathType::NVL)
          << "GPU " << i << " to " << j << " is not NVL (got "
          << pathTypeToString(path.type) << ")";
    }
  }
}

TEST_F(H100TopologyTest, AllGpuPairsCanP2P) {
  for (size_t i = 0; i < topo().gpuCount(); ++i) {
    for (size_t j = 0; j < topo().gpuCount(); ++j) {
      EXPECT_TRUE(topo().canGpuAccess(i, j))
          << "GPU " << i << " cannot access GPU " << j;
    }
  }
}

TEST_F(H100TopologyTest, NicsArePresent) {
  EXPECT_GT(topo().nicCount(), 0u) << "H100 system should have NICs";
}

TEST_F(H100TopologyTest, GpuToNicPathIsNotDisconnected) {
  if (topo().nicCount() == 0) {
    GTEST_SKIP() << "No NICs available";
  }
  for (size_t g = 0; g < topo().gpuCount(); ++g) {
    for (size_t n = 0; n < topo().nicCount(); ++n) {
      int gi = topo().getGpuNode(g).id;
      int ni = topo().getNicNode(n).id;
      const auto& path = topo().getPath(gi, ni);
      EXPECT_NE(path.type, PathType::DIS)
          << "GPU " << g << " to NIC " << n << " is disconnected";
    }
  }
}

TEST_F(H100TopologyTest, NvlinkBandwidthIsReasonable) {
  int n0 = topo().getGpuNode(0).id;
  int n1 = topo().getGpuNode(1).id;
  const auto& path = topo().getPath(n0, n1);
  EXPECT_GE(path.bw, 100000u) << "NVLink BW too low: " << path.bw << " MB/s";
  EXPECT_LE(path.bw, 500000u) << "NVLink BW too high: " << path.bw << " MB/s";
}
