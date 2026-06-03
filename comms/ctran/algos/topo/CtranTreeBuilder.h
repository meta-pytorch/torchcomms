// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <algorithm>
#include <array>
#include <utility>

#include <glog/logging.h>

#include "comms/ctran/algos/topo/TreeConstants.h"

namespace ctran::algos::topo {

struct TreeNeighbors {
  int parent{-1};
  std::array<int, kMaxTreeChildren> children;
  int numChildren{0};

  TreeNeighbors() {
    children.fill(-1);
  }

  bool isRoot() const {
    return parent == -1;
  }

  bool isLeaf() const {
    return numChildren == 0;
  }
};

// Build a single k-ary tree over numNodes using BFS level-order assignment.
// fanOut must be in [1, kMaxTreeChildren], which keeps parent and child
// relationships representable by TreeNeighbors.
static inline TreeNeighbors
buildKaryTree(int numNodes, int nodeRank, int fanOut) {
  CHECK_GE(fanOut, 1);
  CHECK_LE(fanOut, kMaxTreeChildren);

  if (numNodes <= 1) {
    return TreeNeighbors{};
  }

  TreeNeighbors result;

  // BFS-order k-ary tree: node i has parent (i-1)/k, children k*i+1 .. k*i+k
  // This gives a complete k-ary tree with level-order node assignment.
  if (nodeRank == 0) {
    result.parent = -1;
  } else {
    result.parent = (nodeRank - 1) / fanOut;
  }

  for (int c = 0; c < fanOut && c < kMaxTreeChildren; c++) {
    int child = fanOut * nodeRank + 1 + c;
    if (child < numNodes) {
      result.children[result.numChildren++] = child;
    }
  }

  return result;
}

// Build dual complementary k-ary trees.
//
// For k=2 (binary):
//   Tree-0: standard binary tree in BFS order.
//   Tree-1 (even N): mirror — build tree over mapping r' = N-1-r, remap back.
//   Tree-1 (odd N): shift-by-1 — build tree over mapping r' = (r-1+N)%N, remap
//   back. Property: leaves in Tree-0 are internal nodes in Tree-1
//   (phase-complementary).
//
// For k>2:
//   Tree-0: standard k-ary tree.
//   Tree-1: k-ary tree with shifted root (root at node N/2).
//   Phase-complementary property is NOT guaranteed for k>2.
static inline std::pair<TreeNeighbors, TreeNeighbors>
buildDualKaryTree(int numNodes, int nodeRank, int fanOut = 2) {
  if (numNodes <= 1) {
    TreeNeighbors single;
    return {single, single};
  }

  // Tree-0: standard k-ary tree
  TreeNeighbors tree0 = buildKaryTree(numNodes, nodeRank, fanOut);

  // Tree-1: complementary tree
  TreeNeighbors tree1;

  if (fanOut == 2) {
    // Binary dual tree: mirror/shift construction
    int mappedRank;
    if (numNodes % 2 == 0) {
      // Even N: mirror r' = N-1-r
      mappedRank = numNodes - 1 - nodeRank;
    } else {
      // Odd N: shift r' = (r-1+N) % N
      mappedRank = (nodeRank - 1 + numNodes) % numNodes;
    }

    TreeNeighbors mapped = buildKaryTree(numNodes, mappedRank, fanOut);

    // Remap results back
    auto remap = [&](int r) -> int {
      if (r < 0)
        return -1;
      if (numNodes % 2 == 0) {
        return numNodes - 1 - r;
      } else {
        return (r + 1) % numNodes;
      }
    };

    tree1.parent = remap(mapped.parent);
    tree1.numChildren = mapped.numChildren;
    for (int c = 0; c < mapped.numChildren; c++) {
      tree1.children[c] = remap(mapped.children[c]);
    }
  } else {
    // k>2: shifted root construction
    int shift = numNodes / 2;
    int mappedRank = (nodeRank - shift + numNodes) % numNodes;
    TreeNeighbors mapped = buildKaryTree(numNodes, mappedRank, fanOut);

    auto remap = [&](int r) -> int {
      if (r < 0)
        return -1;
      return (r + shift) % numNodes;
    };

    tree1.parent = remap(mapped.parent);
    tree1.numChildren = mapped.numChildren;
    for (int c = 0; c < mapped.numChildren; c++) {
      tree1.children[c] = remap(mapped.children[c]);
    }
  }

  return {tree0, tree1};
}

} // namespace ctran::algos::topo
