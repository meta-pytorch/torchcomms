// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <map>
#include <vector>

#include <cuda_runtime.h>

// Inspects a CUDA graph's topology: nodes grouped by type and dependency edges.
// Used by tests to verify that captured graphs have the correct structure.
struct GraphTopology {
  std::vector<cudaGraphNode_t> allNodes;
  // Nodes grouped by type
  std::map<cudaGraphNodeType, std::vector<cudaGraphNode_t>> nodesByType;
  // edgesFrom[i] -> edgesTo[i] is a directed edge
  std::vector<cudaGraphNode_t> edgesFrom;
  std::vector<cudaGraphNode_t> edgesTo;

  // Returns the nodes of a given type, or an empty vector if none.
  const std::vector<cudaGraphNode_t>& nodesOfType(cudaGraphNodeType type) const;

  // Returns true if 'from' has a direct edge to 'to'.
  bool hasEdge(cudaGraphNode_t from, cudaGraphNode_t to) const;

  // Returns true if 'from' can reach 'to' via any sequence of edges (BFS).
  bool hasPath(cudaGraphNode_t from, cudaGraphNode_t to) const;
};

// Extracts the full topology of a captured CUDA graph.
GraphTopology getGraphTopology(cudaGraph_t graph);
