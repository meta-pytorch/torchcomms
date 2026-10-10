// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <queue>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "comms/ctran/algos/topo/CtranTreeBuilder.h"

namespace ctran::algos::topo {

class CtranTreeBuilderTest : public ::testing::Test {};

// Verify that BFS from the unique root reaches all nodes exactly once.
static void verifyConnectivity(
    const std::vector<TreeNeighbors>& nodes,
    int expectedRoot = -1) {
  const int numNodes = static_cast<int>(nodes.size());
  int rootCount = 0;
  int rootIdx = -1;
  for (int i = 0; i < numNodes; i++) {
    if (nodes[i].isRoot()) {
      rootCount++;
      rootIdx = i;
    }
  }
  ASSERT_EQ(rootCount, 1) << "Exactly one root expected";
  if (expectedRoot >= 0) {
    ASSERT_EQ(rootIdx, expectedRoot);
  }

  std::set<int> visited;
  std::queue<int> bfs;
  bfs.push(rootIdx);
  visited.insert(rootIdx);

  while (!bfs.empty()) {
    int cur = bfs.front();
    bfs.pop();
    for (int c = 0; c < nodes[cur].numChildren; c++) {
      int child = nodes[cur].children[c];
      ASSERT_GE(child, 0);
      ASSERT_LT(child, numNodes);
      ASSERT_EQ(visited.count(child), 0)
          << "Node " << child << " visited twice (cycle or shared child)";
      visited.insert(child);
      bfs.push(child);
    }
  }

  ASSERT_EQ(static_cast<int>(visited.size()), numNodes)
      << "BFS did not reach all nodes. Reached " << visited.size() << " out of "
      << numNodes;
}

// Verify that BFS from root reaches all N nodes exactly once.
static void verifyConnectivity(int numNodes, int fanOut) {
  std::vector<TreeNeighbors> nodes(numNodes);
  for (int i = 0; i < numNodes; i++) {
    nodes[i] = buildKaryTree(numNodes, i, fanOut);
  }

  verifyConnectivity(nodes, 0);
}

// Verify parent-child consistency: if A lists B as child, B must list A as
// parent.
static void verifyParentChildConsistency(int numNodes, int fanOut) {
  std::vector<TreeNeighbors> nodes(numNodes);
  for (int i = 0; i < numNodes; i++) {
    nodes[i] = buildKaryTree(numNodes, i, fanOut);
  }

  for (int i = 0; i < numNodes; i++) {
    for (int c = 0; c < nodes[i].numChildren; c++) {
      int child = nodes[i].children[c];
      ASSERT_GE(child, 0);
      ASSERT_LT(child, numNodes);
      EXPECT_EQ(nodes[child].parent, i)
          << "Node " << i << " lists " << child
          << " as child, but child's parent is " << nodes[child].parent;
    }
    if (nodes[i].parent >= 0) {
      int par = nodes[i].parent;
      bool found = false;
      for (int c = 0; c < nodes[par].numChildren; c++) {
        if (nodes[par].children[c] == i) {
          found = true;
          break;
        }
      }
      EXPECT_TRUE(found) << "Node " << i << " has parent " << par
                         << " but parent does not list it as a child";
    }
  }
}

// Verify leaf/root roles are derived from parent and child count.
static void verifyDerivedRoles(int numNodes, int fanOut) {
  std::vector<TreeNeighbors> nodes(numNodes);
  for (int i = 0; i < numNodes; i++) {
    nodes[i] = buildKaryTree(numNodes, i, fanOut);
  }

  for (int i = 0; i < numNodes; i++) {
    EXPECT_EQ(nodes[i].isLeaf(), nodes[i].numChildren == 0)
        << "Node " << i << ": isLeaf inconsistent with numChildren";
    EXPECT_EQ(nodes[i].isRoot(), nodes[i].parent == -1)
        << "Node " << i << ": isRoot inconsistent with parent";
  }
}

static void verifyUnusedChildSlotsAreSentinels(
    const TreeNeighbors& node,
    int nodeRank) {
  for (int c = node.numChildren; c < kMaxTreeChildren; c++) {
    EXPECT_EQ(node.children[c], -1) << "Node " << nodeRank << " child slot "
                                    << c << " should keep the sentinel value";
  }
}

static int expectedTree1Root(int numNodes, int fanOut) {
  if (numNodes <= 1) {
    return 0;
  }
  if (fanOut == 2) {
    return numNodes % 2 == 0 ? numNodes - 1 : 1;
  }
  return numNodes / 2;
}

// --- Single K-ary Tree Tests ---

class KaryTreeTest : public CtranTreeBuilderTest,
                     public ::testing::WithParamInterface<std::pair<int, int>> {
};

TEST_P(KaryTreeTest, Connectivity) {
  auto [numNodes, fanOut] = GetParam();
  verifyConnectivity(numNodes, fanOut);
}

TEST_P(KaryTreeTest, ParentChildConsistency) {
  auto [numNodes, fanOut] = GetParam();
  verifyParentChildConsistency(numNodes, fanOut);
}

TEST_P(KaryTreeTest, FlagConsistency) {
  auto [numNodes, fanOut] = GetParam();
  verifyDerivedRoles(numNodes, fanOut);
}

TEST_P(KaryTreeTest, MaxChildrenRespected) {
  auto [numNodes, fanOut] = GetParam();
  for (int i = 0; i < numNodes; i++) {
    auto node = buildKaryTree(numNodes, i, fanOut);
    EXPECT_LE(node.numChildren, fanOut);
    EXPECT_LE(node.numChildren, kMaxTreeChildren);
  }
}

TEST_P(KaryTreeTest, UnusedChildSlotsAreSentinels) {
  auto [numNodes, fanOut] = GetParam();
  for (int i = 0; i < numNodes; i++) {
    verifyUnusedChildSlotsAreSentinels(buildKaryTree(numNodes, i, fanOut), i);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BinaryAndTernary,
    KaryTreeTest,
    ::testing::Values(
        // (numNodes, fanOut)
        std::make_pair(1, 2),
        std::make_pair(2, 2),
        std::make_pair(3, 2),
        std::make_pair(4, 2),
        std::make_pair(7, 2),
        std::make_pair(8, 2),
        std::make_pair(15, 2),
        std::make_pair(16, 2),
        std::make_pair(32, 2),
        std::make_pair(64, 2),
        std::make_pair(1, 3),
        std::make_pair(2, 3),
        std::make_pair(3, 3),
        std::make_pair(4, 3),
        std::make_pair(7, 3),
        std::make_pair(8, 3),
        std::make_pair(9, 3),
        std::make_pair(15, 3),
        std::make_pair(16, 3),
        std::make_pair(27, 3),
        std::make_pair(32, 3),
        std::make_pair(64, 3)));

// --- Dual Tree Tests ---

class DualTreeTest : public CtranTreeBuilderTest,
                     public ::testing::WithParamInterface<std::pair<int, int>> {
};

TEST_P(DualTreeTest, BothTreesConnected) {
  auto [numNodes, fanOut] = GetParam();
  for (int r = 0; r < numNodes; r++) {
    auto [t0, t1] = buildDualKaryTree(numNodes, r, fanOut);
    // Both trees should produce valid nodes
    EXPECT_GE(t0.parent, -1);
    EXPECT_GE(t1.parent, -1);
    EXPECT_LE(t0.numChildren, fanOut);
    EXPECT_LE(t1.numChildren, fanOut);
  }

  // Verify full connectivity for both trees independently
  std::vector<TreeNeighbors> t0Nodes(numNodes), t1Nodes(numNodes);
  for (int i = 0; i < numNodes; i++) {
    auto [t0, t1] = buildDualKaryTree(numNodes, i, fanOut);
    t0Nodes[i] = t0;
    t1Nodes[i] = t1;
  }
  verifyConnectivity(t0Nodes, 0);
  verifyConnectivity(t1Nodes);
}

TEST_P(DualTreeTest, Tree1Connectivity) {
  auto [numNodes, fanOut] = GetParam();
  std::vector<TreeNeighbors> t1Nodes(numNodes);
  for (int i = 0; i < numNodes; i++) {
    auto [t0, t1] = buildDualKaryTree(numNodes, i, fanOut);
    t1Nodes[i] = t1;
  }

  verifyConnectivity(t1Nodes, expectedTree1Root(numNodes, fanOut));
}

TEST_P(DualTreeTest, UnusedChildSlotsAreSentinels) {
  auto [numNodes, fanOut] = GetParam();
  for (int i = 0; i < numNodes; i++) {
    auto [t0, t1] = buildDualKaryTree(numNodes, i, fanOut);
    verifyUnusedChildSlotsAreSentinels(t0, i);
    verifyUnusedChildSlotsAreSentinels(t1, i);
  }
}

TEST_P(DualTreeTest, ParentChildConsistencyBothTrees) {
  auto [numNodes, fanOut] = GetParam();

  // Verify Tree-0 consistency
  std::vector<TreeNeighbors> t0Nodes(numNodes), t1Nodes(numNodes);
  for (int i = 0; i < numNodes; i++) {
    auto [t0, t1] = buildDualKaryTree(numNodes, i, fanOut);
    t0Nodes[i] = t0;
    t1Nodes[i] = t1;
  }

  for (int i = 0; i < numNodes; i++) {
    for (int c = 0; c < t0Nodes[i].numChildren; c++) {
      int child = t0Nodes[i].children[c];
      EXPECT_EQ(t0Nodes[child].parent, i)
          << "Tree-0 inconsistency at node " << i;
    }
    for (int c = 0; c < t1Nodes[i].numChildren; c++) {
      int child = t1Nodes[i].children[c];
      EXPECT_EQ(t1Nodes[child].parent, i)
          << "Tree-1 inconsistency at node " << i;
    }
  }
}

TEST_P(DualTreeTest, DifferentRoots) {
  auto [numNodes, fanOut] = GetParam();
  if (numNodes <= 1) {
    return; // Single node — both trees are trivially the same
  }

  int root0 = -1, root1 = -1;
  for (int i = 0; i < numNodes; i++) {
    auto [t0, t1] = buildDualKaryTree(numNodes, i, fanOut);
    if (t0.isRoot())
      root0 = i;
    if (t1.isRoot())
      root1 = i;
  }
  EXPECT_NE(root0, root1) << "Dual trees should have different roots";
}

// Phase-complementary property: leaves in tree-0 are internal in tree-1.
// This is only guaranteed for binary (k=2) dual trees.
class BinaryDualTreeComplementaryTest
    : public CtranTreeBuilderTest,
      public ::testing::WithParamInterface<int> {};

TEST_P(BinaryDualTreeComplementaryTest, LeavesSwapWithInternal) {
  int numNodes = GetParam();
  if (numNodes <= 2) {
    return; // Too small to have meaningful leaf/internal distinction
  }

  std::vector<TreeNeighbors> t0Nodes(numNodes), t1Nodes(numNodes);
  for (int i = 0; i < numNodes; i++) {
    auto [t0, t1] = buildDualKaryTree(numNodes, i, 2);
    t0Nodes[i] = t0;
    t1Nodes[i] = t1;
  }

  for (int i = 0; i < numNodes; i++) {
    bool isLeaf0 = t0Nodes[i].isLeaf();
    bool isLeaf1 = t1Nodes[i].isLeaf();
    bool isRoot0 = t0Nodes[i].isRoot();
    bool isRoot1 = t1Nodes[i].isRoot();

    // Skip roots — root in one tree can be anything in the other
    if (isRoot0 || isRoot1) {
      continue;
    }

    EXPECT_NE(isLeaf0, isLeaf1)
        << "Node " << i
        << ": leaf in tree-0 should be internal in tree-1 (or vice versa). "
        << "tree-0 isLeaf=" << isLeaf0 << " tree-1 isLeaf=" << isLeaf1;
  }
}

// Phase-complementary property is only guaranteed for even N.
// For odd N, one node must be leaf in both trees (unequal leaf/internal
// counts).
INSTANTIATE_TEST_SUITE_P(
    BinaryComplementary,
    BinaryDualTreeComplementaryTest,
    ::testing::Values(4, 8, 16, 32, 64));

INSTANTIATE_TEST_SUITE_P(
    AllFanOuts,
    DualTreeTest,
    ::testing::Values(
        std::make_pair(1, 2),
        std::make_pair(2, 2),
        std::make_pair(3, 2),
        std::make_pair(4, 2),
        std::make_pair(7, 2),
        std::make_pair(8, 2),
        std::make_pair(15, 2),
        std::make_pair(16, 2),
        std::make_pair(32, 2),
        std::make_pair(64, 2),
        std::make_pair(2, 3),
        std::make_pair(3, 3),
        std::make_pair(4, 3),
        std::make_pair(8, 3),
        std::make_pair(9, 3),
        std::make_pair(27, 3),
        std::make_pair(64, 3)));

// --- Edge Case Tests ---

TEST_F(CtranTreeBuilderTest, SingleNode) {
  auto node = buildKaryTree(1, 0, 2);
  EXPECT_TRUE(node.isRoot());
  EXPECT_TRUE(node.isLeaf());
  EXPECT_EQ(node.numChildren, 0);
  EXPECT_EQ(node.parent, -1);
}

TEST_F(CtranTreeBuilderTest, TwoNodesBinary) {
  auto n0 = buildKaryTree(2, 0, 2);
  auto n1 = buildKaryTree(2, 1, 2);

  EXPECT_TRUE(n0.isRoot());
  EXPECT_FALSE(n0.isLeaf());
  EXPECT_EQ(n0.numChildren, 1);
  EXPECT_EQ(n0.children[0], 1);

  EXPECT_FALSE(n1.isRoot());
  EXPECT_TRUE(n1.isLeaf());
  EXPECT_EQ(n1.parent, 0);
}

TEST_F(CtranTreeBuilderTest, DualSingleNode) {
  auto [t0, t1] = buildDualKaryTree(1, 0, 2);
  EXPECT_TRUE(t0.isRoot());
  EXPECT_TRUE(t0.isLeaf());
  EXPECT_TRUE(t1.isRoot());
  EXPECT_TRUE(t1.isLeaf());
}

TEST_F(CtranTreeBuilderTest, InvalidFanOutRejected) {
  EXPECT_DEATH(buildKaryTree(8, 0, 0), "fanOut");
  EXPECT_DEATH(buildKaryTree(8, 0, kMaxTreeChildren + 1), "fanOut");
  EXPECT_DEATH(buildDualKaryTree(8, 0, 0), "fanOut");
  EXPECT_DEATH(buildDualKaryTree(8, 0, kMaxTreeChildren + 1), "fanOut");
}

} // namespace ctran::algos::topo
