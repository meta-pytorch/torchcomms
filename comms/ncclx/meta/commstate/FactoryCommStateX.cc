#include "meta/commstate/FactoryCommStateX.h"
#include "checks.h"
#include "comm.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/commstate/CommStateX.h"
#include "meta/NcclxConfig.h" // @manual
#include "meta/hints/CommHintConfig.h" // @manual

#include "nvmlwrap.h"
#include "transport.h"

namespace ncclx {
namespace {

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

// TODO: NVL fabric topology init should be handled within ctran/statex
// (e.g., as part of CommStateX initialization), not passed from ncclx.
ncclResult_t initNvlFabricTopologies(
    ncclComm* comm,
    CommStateX* statex,
    meta::comms::IBootstrap* bootstrap) {
  nvmlGpuFabricInfoV_t localFabricInfo;
  NCCLCHECK(getLocalGpuFabricInfo(comm, localFabricInfo));

  std::vector<nvmlGpuFabricInfoV_t> allFabricInfos(comm->nRanks);
  allFabricInfos.at(comm->rank) = localFabricInfo;

  auto resFuture = bootstrap->allGather(
      allFabricInfos.data(),
      sizeof(nvmlGpuFabricInfoV_t),
      comm->rank,
      comm->nRanks);
  const int res = std::move(resFuture).get();
  if (res != 0) {
    WARN("initNvlFabricTopologies: bootstrap allGather failed with %d", res);
    return ncclInternalError;
  }

  std::vector<NvlFabricTopology> nvlFabricTopos;
  nvlFabricTopos.reserve(comm->nRanks);
  for (int rank = 0; rank < comm->nRanks; rank++) {
    const auto& fabricInfo_ = allFabricInfos.at(rank);
    NvlFabricTopology topo;
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
  statex->setNvlFabricTopos(std::move(nvlFabricTopos), std::nullopt);
  return ncclSuccess;
}

} // namespace

ncclResult_t createCommStateXFromNcclComm(void* _comm, CtranComm* ctranComm) {
  auto comm = reinterpret_cast<ncclComm*>(_comm);
  CHECKABORT(comm->rankToNode, "rankToNode is nullptr");
  CHECKABORT(comm->localRankToRank, "localRankToRank is nullptr");

  auto* bootstrap = ctranComm->bootstrap_.get();

  const int vCliqueSize =
      commVCliqueSize(NCCLX_CONFIG_FIELD(comm->config, vCliqueSize));

  ctranComm->statex_ = std::make_unique<CommStateX>(
      comm->rank,
      comm->nRanks,
      comm->cudaDev,
      comm->cudaArch,
      comm->busId,
      comm->commHash,
      std::vector<RankTopology>(), /* rankTopologies */
      std::vector<int>(), /* commRanksToWorldRanks */
      NCCLX_CONFIG_FIELD(comm->config, commDesc),
      comm->noLocal_,
      vCliqueSize);

  try {
    ctranComm->statex_->initRankStatesTopology(bootstrap);
  } catch (const std::exception& e) {
    WARN(
        "createCommStateXFromNcclComm: initRankStatesTopology failed: %s",
        e.what());
    return ncclInternalError;
  }

  INFO(
      NCCL_INIT | NCCL_GRAPH,
      "CommStateX: set rankTopology with noLocal=%d, vCliqueSize=%d, commDesc=%s, commHash=0x%lx, rank=%d, nRanks=%d, nLocalRanks=%d",
      comm->noLocal_,
      vCliqueSize,
      NCCLX_CONFIG_FIELD(comm->config, commDesc).c_str(),
      comm->commHash,
      comm->rank,
      comm->nRanks,
      ctranComm->statex_->nLocalRanks());

  NCCLCHECK(initNvlFabricTopologies(comm, ctranComm->statex_.get(), bootstrap));

  return ncclSuccess;
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
