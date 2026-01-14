#include "meta/commstate/FactoryCommStateX.h"
#include "checks.h"
#include "comm.h"
#include "comms/ctran/commstate/CommStateX.h"
#include "comms/ctran/utils/Checks.h"

#include "bootstrap.h"
#include "nvmlwrap.h"
#include "transport.h"

namespace ncclx {

void initRankTopologyFrom(CommStateX* _CommStateX, void* _comm) {
  auto comm = reinterpret_cast<ncclComm*>(_comm);
  _CommStateX->rankStates_.resize(comm->nRanks);
  _CommStateX->nodeRanks_.resize(comm->nNodes);
  for (int r = 0; r < comm->nRanks; ++r) {
    auto& rankState = _CommStateX->rankStates_.at(r);
    rankState.rank = r;
    rankState.nodeId = comm->rankToNode[r];
    rankState.localRank = comm->rankToLocalRank[r];
    rankState.nLocalRanks = comm->localRanks;
    rankState.localRankToRanks.assign(
        comm->localRankToRank, comm->localRankToRank + comm->localRanks);

    // Order of global ranks never changes, thus OK to assume global rank on
    // each node is already ordered by local rank
    // NOTE: two GPUs on the same node may be with different nodeId because
    // they don't have direct NVL access. To keep same nodeId in statex, we
    // use hostHash+nodeId to make it unique
    std::string host(
        std::to_string(comm->peerInfo[r].hostHash) + "_" +
        std::to_string(rankState.nodeId));
    _CommStateX->hostToRanks_[host].emplace_back(r);

    _CommStateX->nodeRanks_[rankState.nodeId].emplace_back(rankState.rank);
  }
}

std::unique_ptr<CommStateX> createCommStateXFromNcclComm(void* _comm) {
  auto comm = reinterpret_cast<ncclComm*>(_comm);
  CHECKABORT(comm->rankToNode, "rankToNode is nullptr");
  CHECKABORT(comm->localRankToRank, "localRankToRank is nullptr");

  auto _CommStateX = std::make_unique<CommStateX>(
      comm->rank,
      comm->nRanks,
      comm->cudaDev,
      comm->cudaArch,
      comm->busId,
      comm->commHash,
      std::vector<RankTopology>(), /* rankTopologies */
      std::vector<int>(), /* commRanksToWorldRanks */
      comm->config.commDesc);

  if (NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::nolocal) {
    // Fake topology with nLocalRanks=1
    _CommStateX->initRankTopologyNolocal();
  } else if (NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::vnode) {
    // Fake topology with
    // nLocalRanks=NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS
    _CommStateX->initRankTopologyVnode(
        NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS);
  } else {
    initRankTopologyFrom(_CommStateX.get(), _comm);
  }

  INFO(
      NCCL_INIT | NCCL_GRAPH,
      "CommStateX: initialize from ncclComm with %s",
      topoNameMap[NCCL_COMM_STATE_DEBUG_TOPO].c_str());

  return _CommStateX;
}

ncclResult_t getLocalGpuFabricInfo(
    ncclComm* ncclComm,
    nvmlGpuFabricInfoV_t& fabricInfo) {
  // we could get fabricInfo from ncclComm->peerInfo, but it could be overridden
  // by NCCL ENVs. we prefer to minimize the depenency on ncclComm and generate
  // the fabricInfo more independently for statex.
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  nvmlDevice_t nvmlDev;

  NCCLCHECK(int64ToBusId(ncclComm->busId, busId));
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));

  fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
  (void)ncclNvmlDeviceGetGpuFabricInfoV(nvmlDev, &fabricInfo);
  if (NCCL_MNNVL_DETERMINISTIC_COLLECTIVE_ENABLE &&
      NCCL_MNNVL_CLIQUE_SIZE > 0) {
    int cliqueId = -1;
    assignMnnvlCliqueIdBasedOnCliqueSize(&cliqueId);
    fabricInfo.cliqueId = cliqueId;
  } else if (NCCL_MNNVL_CLIQUE_ID != -1) {
    fabricInfo.cliqueId = NCCL_MNNVL_CLIQUE_ID;
  }

  return ncclSuccess;
}

ncclResult_t initNvlFabricTopologies(ncclComm* ncclComm, CtranComm* ctranComm) {
  // Get local fabric information
  nvmlGpuFabricInfoV_t localFabricInfo;
  NCCLCHECK(getLocalGpuFabricInfo(ncclComm, localFabricInfo));

  // Gather fabric info from all ranks
  std::vector<nvmlGpuFabricInfoV_t> allFabricInfos(ncclComm->nRanks);
  allFabricInfos.at(ncclComm->rank) = localFabricInfo;

  bootstrapAllGather(
      ncclComm->bootstrap, allFabricInfos.data(), sizeof(nvmlGpuFabricInfoV_t));

  // Create NVL fabric topologies for all ranks
  std::vector<ncclx::NvlFabricTopology> nvlFabricTopos;
  nvlFabricTopos.reserve(ncclComm->nRanks);
  for (int rank = 0; rank < ncclComm->nRanks; rank++) {
    const auto& fabricInfo_ = allFabricInfos.at(rank);
    ncclx::NvlFabricTopology topo;
    if (fabricInfo_.state != NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
      topo.supportNvlFabric = true;
      topo.rank = rank;
      topo.clusterId = fmt::format(
          "{:x}.{:x}",
          ((unsigned long*)&fabricInfo_.clusterUuid)[0],
          ((unsigned long*)&fabricInfo_.clusterUuid)[1]);
      topo.cliqueId = fabricInfo_.cliqueId;
    }
    nvlFabricTopos.emplace_back(std::move(topo));
  }
  ctranComm->statex_->setNvlFabricTopos(std::move(nvlFabricTopos));
  return ncclSuccess;
}

/**
 * Initialize CtranComm statex from NCCL communicator.
 * This function performs two main phases:
 * 1. Initialize rank states topology
 * 2. Initialize NVL fabric topologies by gathering fabric info from all ranks
 */
ncclResult_t initCtranCommStatexFromNcclComm(
    ncclComm* ncclComm,
    CtranComm* ctranComm) {
  if (!ncclComm || !ctranComm) {
    FB_ERRORRETURN(ncclInvalidArgument, "Invalid arguments provided");
  }

  try {
    ctranComm->statex_->initRankStatesTopology(ctranComm->bootstrap_.get());

    NCCLCHECK(initNvlFabricTopologies(ncclComm, ctranComm));

    return ncclSuccess;

  } catch (const std::exception& e) {
    FB_ERRORRETURN(
        ncclInternalError,
        "Failed to initialize CtranComm statex from ncclComm: {}",
        e.what());
  }
}

ncclResult_t assignMnnvlCliqueIdBasedOnCliqueSize(int* cliqueId) {
  XCHECK(NCCL_MNNVL_CLIQUE_SIZE > 0)
      << "NCCL_MNNVL_CLIQUE_SIZE must be positive";
  XCHECK(NCCL_MNNVL_CLIQUE_ID == -1)
      << "NCCL_MNNVL_CLIQUE_SIZE and NCCL_MNNVL_CLIQUE_ID can NOT be set at the same time";
  auto globalRank = RankUtil::getGlobalRank();
  auto worldSize = RankUtil::getWorldSize();
  XCHECK(globalRank.has_value()) << "RANK is not set";
  XCHECK(worldSize.has_value()) << "WORLD_SIZE is not set";
  XCHECK(worldSize.value() % NCCL_MNNVL_CLIQUE_SIZE == 0)
      << "WORLD_SIZE is not a multiple of NCCL_MNNVL_CLIQUE_SIZE";
  *cliqueId = globalRank.value() / NCCL_MNNVL_CLIQUE_SIZE;
  return ncclSuccess;
}

} // namespace ncclx
