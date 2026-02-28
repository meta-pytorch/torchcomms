// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/String.h>
#include <folly/logging/xlog.h>
#include <string>

#include "CommStateX.h"
#include "comms/ctran/commstate/Topology.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ncclx {

namespace {

// check helpers
#define CHECK_TOPO_SET(rankStates) \
  FB_CHECKABORT(rankStates.size() > 0, "Rank topology not set");

#define CHECK_RANKMAP_SET(commRanksToWorldRanks) \
  FB_CHECKABORT(                                 \
      commRanksToWorldRanks.size() > 0, "commRanksToWorldRanks not set");

#define CHECK_VALID_RANK(rank, maxSize) \
  FB_CHECKABORT(                        \
      rank >= 0 && rank < maxSize,      \
      "Invalid rank {}, maxSize: {}",   \
      rank,                             \
      maxSize);

#define CHECK_TOPO_AND_SET_RANK(rank, myRank, rankStates_) \
  do {                                                     \
    CHECK_TOPO_SET(rankStates_);                           \
    if (rank == -1) {                                      \
      rank = myRank;                                       \
    }                                                      \
    CHECK_VALID_RANK(rank, rankStates_.size());            \
  } while (0)

} // namespace

CommStateX::CommStateX(
    int rank,
    int nRanks,
    int cudaDev,
    int cudaArch,
    int64_t busId,
    uint64_t commHash,
    std::vector<RankTopology> rankTopologies,
    std::vector<int> commRanksToWorldRanks,
    const std::string& commDesc)
    : rank_(rank),
      nRanks_(nRanks),
      cudaDev_(cudaDev),
      cudaArch_(cudaArch),
      busId_(busId),
      commHash_(commHash),
      commDesc_(commDesc) {
  setRankStatesTopologies(std::move(rankTopologies));
  setCommRankToWorldRanks(std::move(commRanksToWorldRanks));
}

CommStateX::~CommStateX() {}

void CommStateX::initRankTopologyNolocal() {
  rankStates_.resize(nRanks_);
  nodeRanks_.resize(nRanks_);
  for (int r = 0; r < nRanks_; ++r) {
    auto& rankState = rankStates_.at(r);
    rankState.rank = r;
    rankState.nodeId = r;
    rankState.localRank = 0;
    rankState.nLocalRanks = 1;
    rankState.localRankToRanks.assign(1, r);
    const std::string nolocalHost("nolocal_node_" + std::to_string(r));
    rankState.host = nolocalHost;
    rankState.pid = r;
    hostToRanks_[nolocalHost].emplace_back(r);
    nodeRanks_[rankState.nodeId].emplace_back(rankState.rank);
  }
}

void CommStateX::initRankTopologyVnode(const int nLocalRanks) {
  rankStates_.resize(nRanks_);
  nodeRanks_.resize(nRanks_);
  for (int r = 0; r < nRanks_; ++r) {
    auto& rankState = rankStates_.at(r);
    rankState.nLocalRanks = nLocalRanks;
    rankState.rank = r;
    rankState.nodeId = r / rankState.nLocalRanks;
    rankState.localRank = r % rankState.nLocalRanks;
    rankState.localRankToRanks.assign(
        rankState.nLocalRanks, rankState.nodeId * rankState.nLocalRanks);
    for (int i = 0; i < rankState.nLocalRanks; i++) {
      rankState.localRankToRanks[i] += i;
    }
    const std::string vNodeHost(
        "vnode_node_" + std::to_string(rankState.nodeId));
    rankState.host = vNodeHost;
    rankState.pid = r;
    hostToRanks_[vNodeHost].emplace_back(r);
    nodeRanks_[rankState.nodeId].emplace_back(r);
  }
}

void CommStateX::initRankStatesTopology(
    ctran::bootstrap::IBootstrap* bootstrap) {
  auto myTopo = ctran::commstate::loadTopology(rank_, NCCL_TOPO_FILE_PATH);
  if (!myTopo) {
    FB_CHECKTHROW_EX(
        false,
        rank_,
        commHash_,
        commDesc_,
        fmt::format("Failed to load topology from {}", NCCL_TOPO_FILE_PATH));
  } else {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "load topology rank: {}, nRanks: {}, host: {}, dc: {} zone: {} rtsw: {} scaling unit: {} rackSerial: {}",
        rank_,
        nRanks_,
        myTopo->host,
        myTopo->dc,
        myTopo->zone,
        myTopo->rtsw,
        myTopo->su,
        myTopo->rackSerial);
  }

  // Gather topologies across all the ranks
  std::vector<ncclx::RankTopology> allTopos(nRanks_);
  allTopos.at(rank_) = *myTopo;
  auto resFuture = bootstrap->allGather(
      allTopos.data(), sizeof(ncclx::RankTopology), rank_, nRanks_);
  FB_COMMCHECKTHROW_EX(
      static_cast<commResult_t>(std::move(resFuture).get()),
      rank_,
      commHash_,
      commDesc_);

  // Create statex variable
  setRankStatesTopologies(std::move(allTopos));
}

void CommStateX::setNvlFabricTopos(
    std::vector<NvlFabricTopology> nvlFabricTopologies) {
  FB_CHECKABORT(
      nvlFabricTopologies.size() == nRanks_,
      "size of nvlFabricTopologies not equal to nRanks_: {}",
      nRanks_);
  nvlFabricTopos_ = std::move(nvlFabricTopologies);
  nvlFabricRankStates_.clear();
  nvlDomainRanks_.clear();
  nvlFabricEnabled_ = NCCL_CTRAN_NVL_FABRIC_ENABLE &&
      nvlFabricTopos_.at(rank_).supportNvlFabric;
  if (!nvlFabricEnabled_) {
    return;
  }
  nvlFabricCliqueEnabled_ = nvlFabricEnabled_ && NCCL_MNNVL_CLIQUE_SIZE > 0 &&
      NCCL_MNNVL_DETERMINISTIC_COLLECTIVE_ENABLE;
  std::unordered_map<std::string, int> clusterIdToNvlDomainIndex;
  // cliqueIds might not be contiguous, within the same communicator.
  // CliqueIndex is contiguous fron 0 to nCliqueIds - 1.
  std::unordered_map<int64_t, int> cliqueIdToCliqueIndex;

  nvlFabricRankStates_.resize(nRanks_);
  for (int i = 0; i < nRanks_; i++) {
    nvlFabricRankStates_.at(i).rank = i;
    if (nvlFabricTopos_.at(i).supportNvlFabric) {
      const auto& clusterId = nvlFabricTopos_.at(i).clusterId;
      const auto cliqueId = nvlFabricTopos_.at(i).cliqueId;
      // update nvl domain level info
      if (!clusterIdToNvlDomainIndex.contains(clusterId)) {
        // new domain found
        clusterIdToNvlDomainIndex[clusterId] = nvlDomainRanks_.size();
        nvlDomainRanks_.emplace_back();
      }
      int nvlDomainIndex = clusterIdToNvlDomainIndex.at(clusterId);
      int nvlDomainRank = nvlDomainRanks_.at(nvlDomainIndex).size();
      nvlDomainRanks_.at(nvlDomainIndex).emplace_back(i);

      nvlFabricRankStates_.at(i).nvlDomainIndex = nvlDomainIndex;
      nvlFabricRankStates_.at(i).clusterId = clusterId;
      nvlFabricRankStates_.at(i).nvlDomainRank = nvlDomainRank;
      if (nvlFabricCliqueEnabled_) {
        // update clique level info if clique is enabled
        if (!cliqueIdToCliqueIndex.contains(cliqueId)) {
          // new clique found
          cliqueIdToCliqueIndex[cliqueId] = cliqueRanks_.size();
          cliqueRanks_.emplace_back();
        }
        int cliqueIndex = cliqueIdToCliqueIndex.at(cliqueId);
        int cliqueRank = cliqueRanks_.at(cliqueIndex).size();
        cliqueRanks_.at(cliqueIndex).emplace_back(i);

        nvlFabricRankStates_.at(i).cliqueIndex = cliqueIndex;
        nvlFabricRankStates_.at(i).cliqueRank = cliqueRank;
        nvlFabricRankStates_.at(i).cliqueId = cliqueId;
      }
    }
  }
  // another loop to update the other stats of nvlFabricRankStates_
  for (int i = 0; i < nRanks_; i++) {
    if (nvlFabricTopos_.at(i).supportNvlFabric) {
      auto& state = nvlFabricRankStates_.at(i);
      const auto nvlDomainIndex = state.nvlDomainIndex;
      state.nNvlDomainRanks = nvlDomainRanks_.at(nvlDomainIndex).size();
      state.nvlDomainRankToRank = nvlDomainRanks_.at(nvlDomainIndex);
      if (nvlFabricCliqueEnabled_) {
        const auto cliqueIndex = state.cliqueIndex;
        state.nCliqueRanks = cliqueRanks_.at(cliqueIndex).size();
        state.cliqueRankToRank = cliqueRanks_.at(cliqueIndex);
      }
    }
  }

  // make a copy of myNvlFabricRankState_ for even faster query for
  // local rank state info
  myNvlFabricRankState_ = nvlFabricRankStates_.at(rank_);

  FB_CHECKABORT(
      clusterIdToNvlDomainIndex.size() == nvlDomainRanks_.size(),
      "size of clusterIdToNvlDomainIndex : {} not equal to size nvlDomainRanks_: {}",
      clusterIdToNvlDomainIndex.size(),
      nvlDomainRanks_.size());

  FB_CHECKABORT(
      cliqueIdToCliqueIndex.size() == cliqueRanks_.size(),
      "size of cliqueIdToCliqueIndex : {} not equal to size cliqueRanks_: {}",
      cliqueIdToCliqueIndex.size(),
      cliqueRanks_.size());
}

void CommStateX::setRankStatesTopologies(
    std::vector<RankTopology> rankTopologies) {
  rankStates_.clear();
  nodeRanks_.clear();
  hostToRanks_.clear();

  rankTopologies_ = rankTopologies;

  for (const auto& rankTopology : rankTopologies_) {
    std::string host(rankTopology.host);
    const std::string rtsw(rankTopology.rtsw);
    const std::string su(rankTopology.su);
    const std::string dc(rankTopology.dc);
    const std::string zone(rankTopology.zone);

    if (NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::nolocal) {
      host = "nolocal_node_" + std::to_string(rankTopology.rank);
    } else if (
        NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::vnode) {
      host =
          "vnode_node_" +
          std::to_string(
              rankTopology.rank / NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS);
    }
    hostToRanks_[host].emplace_back(rankTopology.rank);

    RankState state;
    state.rank = rankTopology.rank;
    state.pid = rankTopology.pid;
    state.host = host;
    state.rtsw = rtsw;
    state.su = su;
    state.dc = dc;
    state.zone = zone;
    state.rackSerial = rankTopology.rackSerial;
    if (dc.empty()) {
      // for non-DC scale clusters.
      // Example: "/dkl2.2B//rtsw223.c082.f00.dkl2". Here zone is "dkl2.2B"
      // When dc is empty, we extract the first token before the first '.' as
      // dc. In the example above, "dkl2" would be extracted as dc from
      // "dkl2.2B".
      std::vector<std::string> zoneTokens;
      folly::split('.', zone, zoneTokens);
      if (zoneTokens.size() > 1) {
        state.dc = std::move(zoneTokens[0]);
      }
    }

    state.nodeId = hostToRanks_.size() - 1;
    state.localRank = hostToRanks_.at(host).size() - 1;

    rankStates_.push_back(std::move(state));
  }

  // Populate nodeRanks_ after setup hostToRanks_ so we know how many nodes
  // there are to resize nodeRanks_
  nodeRanks_.resize(hostToRanks_.size());
  for (const auto& state : rankStates_) {
    nodeRanks_[state.nodeId].emplace_back(state.rank);
  }

  for (auto& state : rankStates_) {
    state.localRankToRanks = hostToRanks_.at(state.host);
    state.nLocalRanks = state.localRankToRanks.size();
  }

  XLOGF(
      INFO,
      "CommStateX: set rankTopology with {}",
      topoNameMap[NCCL_COMM_STATE_DEBUG_TOPO]);
}

void CommStateX::setCommRankToWorldRanks(
    std::vector<int> commRanksToWorldRanks) {
  commRanksToWorldRanks_ = std::move(commRanksToWorldRanks);
}

const std::vector<ncclx::RankTopology>& CommStateX::rankTopologiesRef() const {
  return rankTopologies_;
}

const std::vector<int>& CommStateX::commRanksToWorldRanksRef() const {
  return commRanksToWorldRanks_;
}

int CommStateX::rank() const {
  return rank_;
}

int CommStateX::nRanks() const {
  return nRanks_;
}

int CommStateX::cudaDev() const {
  return cudaDev_;
}

int CommStateX::cudaArch() const {
  return cudaArch_;
}

int64_t CommStateX::busId() const {
  return busId_;
}

uint64_t CommStateX::commHash() const {
  return commHash_;
}

std::string CommStateX::commDesc() const {
  return commDesc_;
}

int CommStateX::node(int rank) const {
  CHECK_TOPO_AND_SET_RANK(rank, rank_, rankStates_);
  // Note currently a node is defined as a host in H100, but a NVL domain in
  // GB200 cases.
  // TODO: if we don't need to differentiate the whether a node is a host or nvl
  // domain at alg layer, consider consolidating the NVL fabric
  // branch/data structures with non NVL fabric branches/data structures.
  if (nvlFabricEnabled_) {
    if (nvlFabricCliqueEnabled_) {
      if (rank == rank_) {
        return myNvlFabricRankState_.cliqueIndex;
      }
      return nvlFabricRankStates_.at(rank).cliqueIndex;
    }
    if (rank == rank_) {
      return myNvlFabricRankState_.nvlDomainIndex;
    }
    return nvlFabricRankStates_.at(rank).nvlDomainIndex;
  } else {
    return rankStates_.at(rank).nodeId;
  }
}

int CommStateX::nNodes() const {
  CHECK_TOPO_SET(rankStates_);
  if (nvlFabricEnabled_) {
    if (nvlFabricCliqueEnabled_) {
      return cliqueRanks_.size();
    }
    return nvlDomainRanks_.size();
  } else {
    return hostToRanks_.size();
  }
}

int CommStateX::localRank(int rank) const {
  CHECK_TOPO_AND_SET_RANK(rank, rank_, rankStates_);
  // When NVL fabric/MNNVL is enabled, override localRanK as the NVL domain rank
  // This is to keep consistency with the NCCL baseline implementation (in
  // src/graph/topo.cc), to avoid exposing the NVL domain rank to algorithm
  // layer.
  if (nvlFabricEnabled_) {
    // optimization for faster query
    if (nvlFabricCliqueEnabled_) {
      if (rank == rank_) {
        return myNvlFabricRankState_.cliqueRank;
      }
      return nvlFabricRankStates_.at(rank).cliqueRank;
    }
    if (rank == rank_) {
      return myNvlFabricRankState_.nvlDomainRank;
    }
    return nvlFabricRankStates_.at(rank).nvlDomainRank;
  } else {
    return rankStates_.at(rank).localRank;
  }
}

int CommStateX::nLocalRanks(int rank) const {
  CHECK_TOPO_AND_SET_RANK(rank, rank_, rankStates_);
  if (nvlFabricEnabled_) {
    if (nvlFabricCliqueEnabled_) {
      if (rank == rank_) {
        return myNvlFabricRankState_.nCliqueRanks;
      }
      return nvlFabricRankStates_.at(rank).nCliqueRanks;
    }
    if (rank == rank_) {
      return myNvlFabricRankState_.nNvlDomainRanks;
    }
    return nvlFabricRankStates_.at(rank).nNvlDomainRanks;
  } else {
    return rankStates_.at(rank).nLocalRanks;
  }
}

int CommStateX::localRankToRank(int localRank, int nodeId) const {
  CHECK_TOPO_SET(rankStates_);
  if (nvlFabricEnabled_) {
    if (nvlFabricCliqueEnabled_) {
      if (nodeId == -1) {
        CHECK_VALID_RANK(
            localRank, myNvlFabricRankState_.cliqueRankToRank.size());
        return myNvlFabricRankState_.cliqueRankToRank.at(localRank);
      } else {
        return cliqueRanks_.at(nodeId).at(localRank);
      }
    }
    if (nodeId == -1) {
      CHECK_VALID_RANK(
          localRank, myNvlFabricRankState_.nvlDomainRankToRank.size());
      return myNvlFabricRankState_.nvlDomainRankToRank.at(localRank);
    } else {
      return nvlDomainRanks_.at(nodeId).at(localRank);
    }
  }
  if (nodeId == -1) {
    const auto& state = rankStates_.at(rank_);
    CHECK_VALID_RANK(localRank, state.localRankToRanks.size());
    return state.localRankToRanks.at(localRank);
  } else {
    return nodeRanks_.at(nodeId).at(localRank);
  }
}

std::vector<int> CommStateX::localRankToRanks() const {
  if (nvlFabricEnabled_) {
    if (nvlFabricCliqueEnabled_) {
      return myNvlFabricRankState_.cliqueRankToRank;
    }
    return myNvlFabricRankState_.nvlDomainRankToRank;
  }

  CHECK_TOPO_SET(rankStates_);
  const auto& state = rankStates_.at(rank_);

  return state.localRankToRanks;
}

std::string CommStateX::host(int rank) const {
  CHECK_TOPO_AND_SET_RANK(rank, rank_, rankStates_);
  return rankStates_.at(rank).host;
}

std::string CommStateX::rtsw(int rank) const {
  CHECK_TOPO_AND_SET_RANK(rank, rank_, rankStates_);
  return rankStates_.at(rank).rtsw;
}

int CommStateX::gRank(int rank) const {
  CHECK_RANKMAP_SET(commRanksToWorldRanks_);
  if (rank == -1) {
    rank = rank_;
  }

  CHECK_VALID_RANK(rank, commRanksToWorldRanks_.size());
  return commRanksToWorldRanks_.at(rank);
}

std::string CommStateX::gPid(int rank) const {
  CHECK_TOPO_AND_SET_RANK(rank, rank_, rankStates_);
  return rankStates_.at(rank).host + ":" +
      std::to_string(rankStates_.at(rank).pid) + ":" + std::to_string(rank);
}

std::string CommStateX::dc(int rank) const {
  CHECK_TOPO_AND_SET_RANK(rank, rank_, rankStates_);
  return rankStates_.at(rank).dc;
}

std::string CommStateX::zone(int rank) const {
  CHECK_TOPO_AND_SET_RANK(rank, rank_, rankStates_);
  return rankStates_.at(rank).zone;
}

std::string CommStateX::su(int rank) const {
  CHECK_TOPO_AND_SET_RANK(rank, rank_, rankStates_);
  return rankStates_.at(rank).su;
}

int CommStateX::deviceRack(int rank) const {
  CHECK_TOPO_AND_SET_RANK(rank, rank_, rankStates_);
  return rankStates_.at(rank).rackSerial;
}

// helper functions
bool CommStateX::isSameNode(int myRank, int peer) const {
  return node(peer) == node(myRank);
}
bool CommStateX::isSameRack(int myRank, int peer) const {
  const auto& rtsw1 = rtsw(myRank);
  if (!rtsw1.empty()) {
    const auto& rtsw2 = rtsw(peer);
    return !rtsw2.empty() && rtsw1 == rtsw2;
  }

  // Check su only if rtsw check failed. Since PXN is enabled all ranks in the
  // same SU are 1 hop away.
  const auto& su1 = su(myRank);
  if (!su1.empty()) {
    const auto& su2 = su(peer);
    return !su2.empty() && su1 == su2;
  }

  return false;
}

bool CommStateX::isSameZone(int myRank, int peer) const {
  return zone(peer) == zone(myRank);
}
bool CommStateX::isSameDc(int myRank, int peer) const {
  return dc(peer) == dc(myRank);
}
bool CommStateX::isSameDeviceRack(int myRank, int peer) const {
  return deviceRack(myRank) == deviceRack(peer);
}
bool CommStateX::isSameNvlFabric(int myRank, int peer) const {
  if (!nvlFabricEnabled_) {
    return false;
  }
  auto toposSize = nvlFabricTopos_.size();
  CHECK_VALID_RANK(myRank, toposSize);
  CHECK_VALID_RANK(peer, toposSize);
  const auto& myNvlFabric = nvlFabricTopos_.at(myRank);
  const auto& peerNvlFabric = nvlFabricTopos_.at(peer);

  return myNvlFabric.supportNvlFabric && peerNvlFabric.supportNvlFabric &&
      myNvlFabric.clusterId == peerNvlFabric.clusterId &&
      !myNvlFabric.clusterId.empty();
}

bool CommStateX::nvlFabricEnabled() const {
  return nvlFabricEnabled_;
}

bool CommStateX::nvlFabricCliqueEnabled() const {
  return nvlFabricCliqueEnabled_;
}

void CommStateX::setupDev(::ctran::CommStateXDev& statexDev) {
  // Copy subset of statex field to device side
  statexDev.rank_ = this->rank();
  statexDev.pid_ = getpid();
  statexDev.localRank_ = this->localRank();
  statexDev.localRanks_ = this->nLocalRanks();
  statexDev.nRanks_ = this->nRanks();
  statexDev.nNodes_ = this->nNodes();
  statexDev.commHash_ = this->commHash();
}

} // namespace ncclx
