// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/Topology.h"

#include <algorithm>
#include <queue>
#include <string_view>
#include <tuple>
#include <utility>

#include "comms/uniflow/logging/Logger.h"

namespace uniflow {

namespace {

/// Returns true if path @p a is better than path @p b.
/// Lower type is better; for equal types, higher bandwidth is better.
bool isBetterPath(const TopoPath& a, const TopoPath& b) {
  return std::tie(a.type, b.bw) < std::tie(b.type, a.bw);
}

/// Shared NIC-selection core. Ranks NICs by the path returned from @p pathFor
/// and returns all NICs tied for the best (lowest) path type and bandwidth.
///
/// When @p netdevPrefix is non-empty AND the topology has known netdev names,
/// NICs are first restricted to those whose backing netdev name starts with the
/// prefix; if that yields nothing, it logs a warning and falls back to
/// filter-only selection (the legacy behavior). Topologies that do not populate
/// netdev names skip the prefix predicate entirely (no warning).
template <typename PathFor>
std::vector<std::string> selectNicsImpl(
    const Topology& topo,
    const NicFilter& filter,
    std::string_view netdevPrefix,
    PathFor&& pathFor) {
  const auto pick = [&](bool requirePrefix) {
    std::vector<std::string> nics;
    PathType bestType = PathType::DIS;
    uint32_t maxBw = 0;
    const int count = static_cast<int>(topo.nicCount());
    for (int i = 0; i < count; ++i) {
      if (!topo.filterNic(i, filter)) {
        continue;
      }
      if (requirePrefix && !topo.matchesNetdevPrefix(i, netdevPrefix)) {
        continue;
      }
      const auto& nicNode = topo.getNicNode(i);
      const auto& path = pathFor(nicNode);
      if (path.type < bestType || (path.type == bestType && path.bw > maxBw)) {
        nics.clear();
        nics.push_back(nicNode.name);
        bestType = path.type;
        maxBw = path.bw;
      } else if (path.type == bestType && path.bw == maxBw) {
        nics.push_back(nicNode.name);
      }
    }
    return nics;
  };

  if (!netdevPrefix.empty() && topo.hasNetdevNames()) {
    auto preferred = pick(/*requirePrefix=*/true);
    if (!preferred.empty()) {
      return preferred;
    }
    UNIFLOW_LOG_WARN(
        "No NIC matched netdev prefix '{}'; falling back to filter-only NIC selection",
        netdevPrefix);
  }
  return pick(/*requirePrefix=*/false);
}

} // namespace

// --- PathType ---

const char* pathTypeToString(PathType p) noexcept {
  switch (p) {
    case PathType::NVL:
      return "NVL";
    case PathType::C2C:
      return "C2C";
    case PathType::PIX:
      return "PIX";
    case PathType::PXB:
      return "PXB";
    case PathType::PXN:
      return "PXN";
    case PathType::PHB:
      return "PHB";
    case PathType::SYS:
      return "SYS";
    case PathType::DIS:
      return "DIS";
  }
  return "UNKNOWN";
}

// --- PcieLinkInfo ---

uint32_t PcieLinkInfo::bandwidthMBps() const noexcept {
  // speedMbpsPerLane already includes encoding overhead.
  // Total Mbps = speedMbpsPerLane * width. Divide by 8 for MB/s.
  return speedMbpsPerLane * width / 8;
}

// --- Topology: mutation API ---

void Topology::clear() {
  status = ErrCode::TopologyDisconnect;
  nodes_.clear();
  paths_.clear();
  c2cPaths_.clear();
  pxnPaths_.clear();
  gpuNodeIds_.clear();
  nicNodeIds_.clear();
  nicNameToNuma_.clear();
  cpuNodeIds_.clear();
  p2pMatrix_.clear();
}

int Topology::addNode(TopoNode node) {
  int id = static_cast<int>(nodes_.size());
  node.id = id;
  nodes_.push_back(std::move(node));
  return id;
}

void Topology::addLink(int srcId, int dstId, PathType type, uint32_t bw) {
  nodes_[srcId].links.push_back({type, bw, dstId});
  nodes_[dstId].links.push_back({type, bw, srcId});
}

void Topology::registerGpuNode(int cudaDeviceId, int nodeId) {
  if (static_cast<int>(gpuNodeIds_.size()) <= cudaDeviceId) {
    gpuNodeIds_.resize(cudaDeviceId + 1, -1);
  }
  gpuNodeIds_[cudaDeviceId] = nodeId;
}

void Topology::registerCpuNode(int numaId, int nodeId) {
  if (static_cast<int>(cpuNodeIds_.size()) <= numaId) {
    cpuNodeIds_.resize(numaId + 1, -1);
  }
  cpuNodeIds_[numaId] = nodeId;
}

void Topology::registerNicNode(int nicIndex, int nodeId) {
  if (static_cast<int>(nicNodeIds_.size()) <= nicIndex) {
    nicNodeIds_.resize(nicIndex + 1, -1);
  }
  nicNodeIds_[nicIndex] = nodeId;
  const auto& node = nodes_[nodeId];
  const auto& nicData = std::get<TopoNode::NicData>(node.data);
  nicNameToNuma_[node.name] = nicData.numaNode;
}

void Topology::setP2pMatrix(std::vector<std::vector<bool>> matrix) {
  p2pMatrix_ = std::move(matrix);
}

void Topology::recomputePaths() {
  computePaths();
  computeC2cPaths();
  computePxnPaths();
}

// --- Topology: path computation ---

void Topology::computePaths() {
  int n = static_cast<int>(nodes_.size());
  paths_.assign(n, std::vector<TopoPath>(n));

  for (int src = 0; src < n; ++src) {
    auto& srcPaths = paths_[src];

    // Self path: type is best possible, bandwidth is unlimited.
    srcPaths[src] = {PathType::NVL, UINT32_MAX, std::nullopt};

    // SPFA-style BFS with relaxation.
    std::vector<bool> inQueue(n, false);
    std::queue<int> queue;
    queue.push(src);
    inQueue[src] = true;

    while (!queue.empty()) {
      int cur = queue.front();
      queue.pop();
      inQueue[cur] = false;
      // Transit GPU restriction: if the current node is a GPU that is not
      // the source, restrict which edges can be followed.
      // - Source is GPU: allow NVLink only (for GPU→NVL→GPU paths).
      // - Source is not GPU: block all edges (prevents NIC→GPU→NVL→GPU;
      //   those routes are handled by PXN post-processing).
      bool isTransitGpu = cur != src && nodes_[cur].type == NodeType::GPU;
      bool srcIsGpu = nodes_[src].type == NodeType::GPU;

      for (const auto& link : nodes_[cur].links) {
        if (isTransitGpu && (!srcIsGpu || link.type != PathType::NVL)) {
          continue;
        }
        // Skip C2C edges in baseline BFS. C2C paths are stored separately.
        if (link.type == PathType::C2C) {
          continue;
        }
        int next = link.peerNodeId;
        PathType newType = std::max(srcPaths[cur].type, link.type);
        uint32_t newBw = std::min(srcPaths[cur].bw, link.bw);
        TopoPath candidate{newType, newBw, std::nullopt};

        if (isBetterPath(candidate, srcPaths[next])) {
          srcPaths[next] = candidate;
          if (!inQueue[next]) {
            queue.push(next);
            inQueue[next] = true;
          }
        }
      }
    }
  }
}

void Topology::computeC2cPaths() {
  // For each GPU with a C2C edge to a CPU, the C2C path to any destination
  // is: GPU→C2C→CPU→(baseline path from CPU to dest).
  // Store overrides where this is better than the baseline GPU→PHB→CPU path.
  int n = static_cast<int>(nodes_.size());

  for (auto gpuId : gpuNodeIds_) {
    for (const auto& link : nodes_[gpuId].links) {
      if (link.type != PathType::C2C) {
        continue;
      }
      int cpuId = link.peerNodeId;
      uint32_t c2cBw = link.bw;

      // Outbound: GPU → C2C → CPU → (baseline to dest)
      for (int dst = 0; dst < n; ++dst) {
        if (dst == gpuId) {
          continue;
        }
        const auto& cpuToDst = paths_[cpuId][dst];
        if (cpuToDst.type == PathType::DIS) {
          continue;
        }
        TopoPath candidate{
            std::max(PathType::C2C, cpuToDst.type),
            std::min(c2cBw, cpuToDst.bw),
            std::nullopt};
        if (isBetterPath(candidate, paths_[gpuId][dst])) {
          c2cPaths_[{gpuId, dst}] = candidate;
          c2cPaths_[{dst, gpuId}] = candidate; // Symmetric
        }
      }
    }
  }
}

void Topology::computePxnPaths() {
  // For each (GPU, NIC) pair, check if routing through an NVLink-connected
  // intermediate GPU gives a better path than the baseline BFS path.
  for (auto gpuId : gpuNodeIds_) {
    for (auto nicId : nicNodeIds_) {
      const auto& curPath = paths_[gpuId][nicId];

      // Only try PXN if current path is PHB or worse.
      if (curPath.type < PathType::PHB) {
        continue;
      }

      TopoPath bestPxn;
      for (auto proxyId : gpuNodeIds_) {
        if (gpuId == proxyId) {
          continue;
        }

        // Need NVLink path from source GPU to proxy GPU.
        const auto& gpuToProxy = paths_[gpuId][proxyId];
        if (gpuToProxy.type > PathType::NVL) {
          continue;
        }

        // Need close PCIe path from proxy GPU to NIC.
        const auto& proxyToNic = paths_[proxyId][nicId];
        if (proxyToNic.type > PathType::PXB) {
          continue;
        }

        uint32_t pxnBw = std::min(gpuToProxy.bw, proxyToNic.bw);
        TopoPath candidate{PathType::PXN, pxnBw, proxyId};

        if (isBetterPath(candidate, bestPxn)) {
          bestPxn = candidate;
        }
      }

      if (bestPxn.type != PathType::DIS && isBetterPath(bestPxn, curPath)) {
        pxnPaths_[{gpuId, nicId}] = bestPxn;
        pxnPaths_[{nicId, gpuId}] = bestPxn;
      }
    }
  }
}

// --- Query methods ---

const TopoPath& Topology::getPath(
    int srcNodeId,
    int dstNodeId,
    const PathFilter& filter) const {
  static const TopoPath kDisconnected;
  int n = static_cast<int>(nodes_.size());
  if (srcNodeId < 0 || srcNodeId >= n || dstNodeId < 0 || dstNodeId >= n) {
    return kDisconnected;
  }

  auto key = std::make_pair(srcNodeId, dstNodeId);
  const TopoPath& best = paths_[srcNodeId][dstNodeId];

  if (filter.allowC2C) {
    auto it = c2cPaths_.find(key);
    if (it != c2cPaths_.end() && isBetterPath(it->second, best)) {
      return it->second;
    }
  }

  if (filter.allowPxn) {
    auto it = pxnPaths_.find(key);
    if (it != pxnPaths_.end() && isBetterPath(it->second, best)) {
      return it->second;
    }
  }

  return best;
}

bool Topology::canGpuAccess(int device, int peerDevice) const {
  int n = static_cast<int>(gpuNodeIds_.size());
  if (device < 0 || device >= n || peerDevice < 0 || peerDevice >= n) {
    return false;
  }
  return p2pMatrix_[device][peerDevice];
}

const TopoNode& Topology::getNode(int nodeId) const {
  CHECK_THROW_EXCEPTION(
      nodeId >= 0 && nodeId < static_cast<int>(nodes_.size()),
      std::runtime_error);
  return nodes_[nodeId];
}

const TopoNode& Topology::getGpuNode(int cudaDeviceId) const {
  CHECK_THROW_EXCEPTION(
      cudaDeviceId >= 0 && cudaDeviceId < static_cast<int>(gpuNodeIds_.size()),
      std::runtime_error);
  return nodes_[gpuNodeIds_[cudaDeviceId]];
}

const TopoNode& Topology::getNicNode(int nicIndex) const {
  CHECK_THROW_EXCEPTION(
      nicIndex >= 0 && nicIndex < static_cast<int>(nicNodeIds_.size()),
      std::runtime_error);
  return nodes_[nicNodeIds_[nicIndex]];
}

const TopoNode& Topology::getCpuNode(int numaId) const {
  CHECK_THROW_EXCEPTION(
      numaId >= 0 && numaId < static_cast<int>(cpuNodeIds_.size()),
      std::runtime_error);
  return nodes_[cpuNodeIds_[numaId]];
}

int Topology::getCurrentCpuNumaNode() const {
  unsigned cpu = 0;
  unsigned node = 0;
  if (syscall(SYS_getcpu, &cpu, &node, nullptr) == 0) {
    return static_cast<int>(node);
  }
  return -1;
}

int Topology::nicNumaNode(const std::string& nicName) const {
  if (nicName.empty()) {
    return -1;
  }
  auto it = nicNameToNuma_.find(nicName);
  return it != nicNameToNuma_.end() ? it->second : -1;
}

bool Topology::filterNic(int nicIndex, const NicFilter& filter) const {
  return filter.matches(nodes_[nicNodeIds_[nicIndex]].name);
}

bool Topology::matchesNetdevPrefix(int nicIndex, std::string_view netdevPrefix)
    const {
  if (netdevPrefix.empty()) {
    return true;
  }
  const auto& nicData = std::get<TopoNode::NicData>(getNicNode(nicIndex).data);
  return std::string_view(nicData.netdevName).starts_with(netdevPrefix);
}

bool Topology::hasNetdevNames() const {
  const int count = static_cast<int>(nicCount());
  for (int i = 0; i < count; ++i) {
    const auto& nicData = std::get<TopoNode::NicData>(getNicNode(i).data);
    if (!nicData.netdevName.empty()) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> Topology::selectCpuNics(
    const NicFilter& filter,
    std::string_view netdevPrefix) const {
  return selectNicsImpl(
      *this,
      filter,
      netdevPrefix,
      [this](const TopoNode& nicNode) -> const TopoPath& {
        const auto& numaNode =
            getCpuNode(std::get<TopoNode::NicData>(nicNode.data).numaNode);
        return getPath(numaNode.id, nicNode.id, {.allowC2C = true});
      });
}

std::vector<std::string> Topology::selectGpuNics(
    int cudaDeviceId,
    const NicFilter& filter,
    std::string_view netdevPrefix) const {
  const auto& gpuNode = getGpuNode(cudaDeviceId);
  return selectNicsImpl(
      *this,
      filter,
      netdevPrefix,
      [this, &gpuNode](const TopoNode& nicNode) -> const TopoPath& {
        return getPath(gpuNode.id, nicNode.id, {.allowC2C = true});
      });
}

std::vector<std::string> Topology::selectCpuNicsForNuma(
    int numaId,
    const NicFilter& filter,
    size_t maxNics) const {
  if (numaId < 0 || numaId >= static_cast<int>(cpuNodeIds_.size()) ||
      cpuNodeIds_[numaId] < 0) {
    return {};
  }
  const auto& numaNode = getCpuNode(numaId);
  auto nics = selectNicsImpl(
      *this,
      filter,
      "",
      [this, &numaNode](const TopoNode& nicNode) -> const TopoPath& {
        return getPath(numaNode.id, nicNode.id, {.allowC2C = true});
      });
  if (maxNics > 0 && nics.size() > maxNics) {
    nics.resize(maxNics);
  }
  return nics;
}

std::vector<std::string> Topology::selectCpuNicsForNumaNodes(
    const NicFilter& filter,
    size_t maxNicsPerNuma) const {
  std::vector<std::string> nics;
  for (int numaId = 0; numaId < static_cast<int>(cpuNodeIds_.size());
       ++numaId) {
    auto numaNics = selectCpuNicsForNuma(numaId, filter, maxNicsPerNuma);
    for (auto& nic : numaNics) {
      if (std::find(nics.begin(), nics.end(), nic) == nics.end()) {
        nics.push_back(std::move(nic));
      }
    }
  }
  return nics;
}

// --- NicFilter ---

NicFilter::NicFilter(std::string_view filterStr) {
  if (filterStr.empty()) {
    return;
  }

  // Parse prefix modifiers.
  size_t pos = 0;
  if (filterStr.substr(pos, 2) == "^=") {
    mode_ = MatchMode::ExactExclude;
    pos += 2;
  } else if (filterStr[pos] == '^') {
    mode_ = MatchMode::PrefixExclude;
    pos += 1;
  } else if (filterStr[pos] == '=') {
    mode_ = MatchMode::ExactInclude;
    pos += 1;
  }

  // Split by ',' and parse each entry.
  std::string_view remaining = filterStr.substr(pos);
  while (!remaining.empty()) {
    auto comma = remaining.find(',');
    std::string_view token = (comma != std::string_view::npos)
        ? remaining.substr(0, comma)
        : remaining;

    // Trim whitespace.
    auto start = token.find_first_not_of(" \t");
    if (start != std::string_view::npos) {
      auto end = token.find_last_not_of(" \t");
      token = token.substr(start, end - start + 1);

      Entry entry;
      auto colon = token.find(':');
      if (colon != std::string_view::npos) {
        entry.name = std::string(token.substr(0, colon));
        entry.port = std::atoi(std::string(token.substr(colon + 1)).c_str());
      } else {
        entry.name = std::string(token);
      }
      entries_.push_back(std::move(entry));
    }

    if (comma == std::string_view::npos) {
      break;
    }
    remaining = remaining.substr(comma + 1);
  }
}

bool NicFilter::isExactMode() const noexcept {
  return mode_ == MatchMode::ExactInclude || mode_ == MatchMode::ExactExclude;
}

bool NicFilter::matches(const std::string& devName, int port) const {
  if (entries_.empty()) {
    return true;
  }

  bool found = false;
  for (const auto& entry : entries_) {
    bool nameMatch = isExactMode()
        ? (devName == entry.name)
        : (devName.compare(0, entry.name.size(), entry.name) == 0);

    if (nameMatch) {
      if (entry.port >= 0 && port >= 0 && entry.port != port) {
        continue;
      }
      found = true;
      break;
    }
  }

  return (mode_ == MatchMode::PrefixInclude || mode_ == MatchMode::ExactInclude)
      ? found
      : !found;
}

} // namespace uniflow
