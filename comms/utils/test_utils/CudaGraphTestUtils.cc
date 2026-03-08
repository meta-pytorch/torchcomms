// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/test_utils/CudaGraphTestUtils.h"

#include <queue>
#include <unordered_set>

const std::vector<cudaGraphNode_t>& GraphTopology::nodesOfType(
    cudaGraphNodeType type) const {
  static const std::vector<cudaGraphNode_t> empty;
  auto it = nodesByType.find(type);
  return it != nodesByType.end() ? it->second : empty;
}

bool GraphTopology::hasEdge(cudaGraphNode_t from, cudaGraphNode_t to) const {
  for (size_t i = 0; i < edgesFrom.size(); i++) {
    if (edgesFrom[i] == from && edgesTo[i] == to) {
      return true;
    }
  }
  return false;
}

bool GraphTopology::hasPath(cudaGraphNode_t from, cudaGraphNode_t to) const {
  std::unordered_set<cudaGraphNode_t> visited;
  std::queue<cudaGraphNode_t> queue;
  queue.push(from);
  while (!queue.empty()) {
    auto cur = queue.front();
    queue.pop();
    if (cur == to) {
      return true;
    }
    if (!visited.insert(cur).second) {
      continue;
    }
    for (size_t i = 0; i < edgesFrom.size(); i++) {
      if (edgesFrom[i] == cur) {
        queue.push(edgesTo[i]);
      }
    }
  }
  return false;
}

GraphTopology getGraphTopology(cudaGraph_t graph) {
  GraphTopology topo;

  size_t numNodes = 0;
  cudaGraphGetNodes(graph, nullptr, &numNodes);
  topo.allNodes.resize(numNodes);
  cudaGraphGetNodes(graph, topo.allNodes.data(), &numNodes);

  for (auto& node : topo.allNodes) {
    cudaGraphNodeType type;
    cudaGraphNodeGetType(node, &type);
    topo.nodesByType[type].push_back(node);
  }

  size_t numEdges = 0;
  cudaGraphGetEdges(graph, nullptr, nullptr, &numEdges);
  topo.edgesFrom.resize(numEdges);
  topo.edgesTo.resize(numEdges);
  cudaGraphGetEdges(
      graph, topo.edgesFrom.data(), topo.edgesTo.data(), &numEdges);

  return topo;
}
