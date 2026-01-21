// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef COMM_STATE_X_H_INCLUDED
#define COMM_STATE_X_H_INCLUDED

#include <netinet/in.h>
#include <sys/socket.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "comms/ctran/commstate/CommStateXDev.h"
#include "comms/ctran/interfaces/IBootstrap.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ncclx {

namespace {
constexpr size_t kMaxNameLen = 64;
}

// We use sizeof(T) to determine the size for bootstrapAllGather, so have to use
// c style char[] to get accurate size
// TODO: merge with RankState, there is redundancy in these two structures
struct RankTopology {
  // rank-id
  int rank{-1};

  // process id
  int pid{-1};

  // host name e.g twshared0265.02.nha1
  char host[kMaxNameLen];

  // rtsw name e.g rtsw098.c084.f00.nha1
  char rtsw[kMaxNameLen];

  // Scaling unit info for rail based networks eg. atn3.z085.u001
  char su[kMaxNameLen];

  char zone[kMaxNameLen];

  char dc[kMaxNameLen];

  int rackSerial{-1};
};

// ncclx internal structure for NVL Fabric info
struct NvlFabricTopology {
  int rank{-1};
  bool supportNvlFabric{false};
  std::string clusterId;
  int64_t cliqueId{0};
};

namespace {
inline static std::unordered_map<enum NCCL_COMM_STATE_DEBUG_TOPO, std::string>
    topoNameMap{
        {NCCL_COMM_STATE_DEBUG_TOPO::system, "system"},
        {NCCL_COMM_STATE_DEBUG_TOPO::nolocal, "nolocal"},
        {NCCL_COMM_STATE_DEBUG_TOPO::vnode, "vnode"},
    };

}

// Use forward declaration
class CommStateX;
std::unique_ptr<CommStateX> createCommStateXFrom(void* comm);

// Communicator State, a Data class to hide internal state from users
class CommStateX {
 public:
  CommStateX(
      int rank,
      int nRanks,
      int cudaDev,
      int cudaArch,
      int64_t busId,
      uint64_t commHash,
      std::vector<RankTopology> rankTopologies,
      std::vector<int> commRanksToWorldRanks,
      const std::string& commDesc = "");

  ~CommStateX();

  std::shared_ptr<CommStateX> createCommStateXFromNcclComm(void* comm);

  void initRankTopologyNolocal();
  void initRankTopologyVnode(const int nLocalRanks);
  friend void initRankTopologyFrom(CommStateX* _CommStateX, void* _comm);
  void initRankStatesTopology(ctran::bootstrap::IBootstrap* bootstrap);

  /* Setters */
  void setRankTopologies(std::vector<RankTopology> rankTopologies);
  void setNvlFabricTopos(std::vector<NvlFabricTopology> nvlFabricTopologies);

  /* Getters */
  const std::vector<ncclx::RankTopology>& rankTopologiesRef() const;

  const std::vector<int>& commRanksToWorldRanksRef() const;

  // get my rank
  int rank() const;

  // get number of ranks
  int nRanks() const;

  // get cuda device index
  int cudaDev() const;

  int cudaArch() const;

  int64_t busId() const;

  // get comm hash
  uint64_t commHash() const;

  // get comm descriptor
  std::string commDesc() const;

  // get node Id for a given rank, default to current rank
  int node(int rank = -1) const;

  int nNodes() const;

  // get localRank for a given rank, default to current rank
  int localRank(int rank = -1) const;

  // get number of localRanks for a given rank, default to current rank
  int nLocalRanks(int rank = -1) const;

  // get rank for a given localRank and nodeId. If nodeId is not provided, use
  // current node.
  // if nvl fabric is enabled, nodeId actually means nvl domain id, if not
  // provided, use current nvl domain.
  int localRankToRank(int localRank, int nodeId = -1) const;

  // get localRank to Rank map
  std::vector<int> localRankToRanks() const;

  // get host name for a given rank, default to current rank
  std::string host(int rank = -1) const;

  // get rtsw/rack name for a given rank, default to current rank
  std::string rtsw(int rank = -1) const;

  // get scaling unit name for a given rank, default to current rank
  std::string su(int rank = -1) const;

  // get zone name for a given rank, default to current rank
  std::string zone(int rank = -1) const;

  // get DataCenter name for a given rank, default to current rank
  std::string dc(int rank = -1) const;

  int deviceRack(int rank) const;

  // get globalRank for a given rank, default to current rank
  int gRank(int rank = -1) const;

  // get global pid (hostname:pid) for a given rank, default to current rank
  // This serves as the global identifier for a rank, used when gRank is not
  // supported in non-eager-init mode
  std::string gPid(int rank = -1) const;

  // check if two ranks are on the same node
  bool isSameNode(int myRank, int peer) const;

  // check if two ranks are on the same rack/ (under the same rtsw)
  bool isSameRack(int myRank, int peer) const;

  // check if two ranks are on the same zone
  bool isSameZone(int myRank, int peer) const;

  // check if two ranks are on the same DC
  bool isSameDc(int myRank, int peer) const;

  // check if current rank has enabled NVL Fabric
  bool nvlFabricEnabled() const;

  // check if current rank has enabled NVL Fabric Clique
  bool nvlFabricCliqueEnabled() const;

  // check if two ranks are on the same NVL Fabric
  bool isSameNvlFabric(int myRank, int peer) const;

  // setup device side statex
  void setupDev(::ctran::CommStateXDev& statexDev);

  bool isSameDeviceRack(int myRank, int peer) const;

 private:
  /* Setters */
  void setRankStatesTopologies(std::vector<RankTopology> rankTopologies);
  void setCommRankToWorldRanks(std::vector<int> commRanksToWorldRanks);

  // TODO: merge with RankTopology, there is redundancy in these two structures
  struct RankState {
    int rank{-1};

    int pid{-1};

    std::string host;

    std::string rtsw;

    std::string su;

    std::string dc;

    std::string zone;

    int nodeId{-1};

    int localRank{-1};

    int nLocalRanks{-1};

    std::vector<int> localRankToRanks{};

    int rackSerial{-1};
  };

  // internal state related to NVL Fabric for quick one time lookup
  struct NvlFabricRankState {
    int rank{-1};
    // physical NVL domain level info
    int nvlDomainRank{-1};
    // clusterId is assigned by NVL Fabric, it's a unique id for a NVL domain
    std::string clusterId;
    // nvlDomainIndex an index from 0 to number of nvl domains - 1
    int nvlDomainIndex{-1};
    int nNvlDomainRanks{-1};
    std::vector<int> nvlDomainRankToRank{};
    // Logical clique level info, meaningful only when NCCL_MNNVL_CLIQUE_SIZE >
    // 0
    int cliqueRank{-1};
    // cliqueId is used by users to soft partition a NVL domain to multiple
    // cliques.
    int cliqueId{-1};
    // cliqueIds might not be contiguous, within the same communicator.
    // CliqueIndex is contiguous from 0 to number of cliques - 1.
    int cliqueIndex{-1};
    // number of ranks in my clique
    int nCliqueRanks{-1};
    std::vector<int> cliqueRankToRank{};
  };

  // TODO: clean fields withing StateX and RankState/RankTOpology
  const int rank_{0};
  const int nRanks_{0};
  const int cudaDev_{0};
  int cudaArch_{-1};
  int64_t busId_{-1};
  const uint64_t commHash_{0};
  const std::string commDesc_;

  std::vector<RankTopology> rankTopologies_{};
  // World ranks only exist in eager init when there's a default_pg
  // and an associated communicator.
  std::vector<int> commRanksToWorldRanks_{};

  // e.g map<host-name, localRankToRank>
  // e.g map<host1: [0, 1, 2, 3], host2: [4, 5, 6, 7]>
  std::unordered_map<std::string, std::vector<int>> hostToRanks_{};

  // similar to hostToRanks_ but access by nodeId
  // e.g vector<0: [0, 1, 2, 3], 1: [0, 1, 2, 3]>
  std::vector<std::vector<int>> nodeRanks_{};

  // similar to nodeRanks, but at nvlDomain level. e.g vector<0: [0, 1, 2, 3],
  // 1: [4, 5, 6, 7]>, index of the outside vector is nvlDomainIndex
  std::vector<std::vector<int>> nvlDomainRanks_{};

  // similar to nvlDomainRanks_, but at the clique level.
  std::vector<std::vector<int>> cliqueRanks_{};

  std::vector<RankState> rankStates_{};

  std::vector<NvlFabricRankState> nvlFabricRankStates_{};

  NvlFabricRankState myNvlFabricRankState_{};

  // flag to indicate if this rank has enabled NVL Fabric, if this flag is on,
  //  we assume all the nvl communication from/to this rank is through
  //  NVL Fabric
  bool nvlFabricEnabled_{false};

  // clique is used to soft partition a NVL domain to multiple cliques.
  bool nvlFabricCliqueEnabled_{false};

  std::vector<NvlFabricTopology> nvlFabricTopos_{};
};

} // namespace ncclx

#endif /* COMM_STATE_X_H_INCLUDED */
