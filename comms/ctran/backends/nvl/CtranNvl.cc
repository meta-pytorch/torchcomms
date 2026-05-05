// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/nvl/CtranNvl.h"
#include "comms/ctran/backends/nvl/CtranNvlImpl.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

// Map a PCI bus ID to a locally visible CUDA device index.
// Returns -1 if the bus ID does not match any visible device
// (e.g. under CUDA_VISIBLE_DEVICES isolation).
// Mirrors ncclx p2p.cc busIdToCudaDev().
static int busIdToCudaDev(int64_t busId) {
  int ndev;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess) {
    return -1;
  }
  for (int i = 0; i < ndev; i++) {
    auto devBusId = ctran::utils::BusId::makeFrom(i).toInt64();
    if (busId == devBusId) {
      return i;
    }
  }
  return -1;
}

CtranNvl::CtranNvl(CtranComm* comm) {
  const auto statex = comm->statex_.get();
  int myRank = statex->rank();
  int myLocalRank = statex->localRank();
  int nLocalRanks = statex->nLocalRanks();
  // Exchange PCI bus IDs (hardware-level, process-independent) so we can
  // correctly map peers to local CUDA device indices even under
  // CUDA_VISIBLE_DEVICES isolation.
  std::vector<int64_t> peerBusIds(nLocalRanks, 0);
  std::vector<std::string> supportedInraHostRanksStr;
  std::vector<std::string> nvlFabricSupportedRanksStr;

  peerBusIds[myLocalRank] = statex->busId();
  auto resFuture = comm->bootstrap_->allGatherNvlDomain(
      peerBusIds.data(),
      sizeof(int64_t),
      myLocalRank,
      nLocalRanks,
      statex->localRankToRanks());
  FB_COMMCHECKTHROW_EX(
      static_cast<commResult_t>(std::move(resFuture).get()),
      comm->logMetaData_);

  this->pimpl_ = std::make_unique<Impl>();
  this->pimpl_->comm = comm;
  this->pimpl_->nvlRankSupportMode.resize(statex->nRanks());

  // Check IPC support for each peer
  for (int i = 0; i < nLocalRanks; ++i) {
    // if supported, update nvlFabric support mode for each peer
    if (statex->nvlFabricEnabled()) {
      if (NCCL_MNNVL_TRUNK_DISABLE) {
        bool p2pAccess = comm->statex_->isSameDeviceRack(
            comm->logMetaData_.rank, statex->localRankToRank(i));
        if (!p2pAccess) {
          CLOGF_SUBSYS(
              INFO,
              INIT,
              "NCCL_MNNVL_TRUNK_DISABLE set to True. P2P disabled between rank1: {} rank2: {} because rackserial mismatch",
              comm->logMetaData_.rank,
              statex->localRankToRank(i));
          continue;
        }
      }
      this->pimpl_->nvlRankSupportMode[statex->localRankToRank(i)].nvlFabric =
          true;
      nvlFabricSupportedRanksStr.push_back(
          std::to_string(statex->localRankToRank(i)));
    } else {
      if (myLocalRank == i) {
        this->pimpl_->nvlRankSupportMode[statex->localRankToRank(i)]
            .nvlIntraHost = true;
        supportedInraHostRanksStr.push_back(
            std::to_string(statex->localRankToRank(i)));
        continue;
      }
      // Map peer's bus ID to a locally visible CUDA device index.
      // Under CUDA_VISIBLE_DEVICES isolation (e.g. torchrun), peer GPUs
      // are not visible so this returns -1.
      int peerLocalDev = busIdToCudaDev(peerBusIds[i]);
      int canAccessPeer = 0;
      if (peerLocalDev >= 0) {
        FB_CUDACHECKTHROW_EX(
            cudaDeviceCanAccessPeer(
                &canAccessPeer, statex->cudaDev(), peerLocalDev),
            comm->logMetaData_);
      } else if (NCCL_CTRAN_ASSUME_HIDDEN_LOCAL_PEERS_NVL) {
        canAccessPeer = 1;
        CLOGF_SUBSYS(
            INFO,
            INIT,
            "CTRAN-NVL: Peer {} (local rank {} busId {:x}) not visible locally, assuming NVL (NCCL_CTRAN_ASSUME_HIDDEN_LOCAL_PEERS_NVL=1)",
            statex->localRankToRank(i),
            i,
            peerBusIds[i]);
      }
      if (canAccessPeer) {
        this->pimpl_->nvlRankSupportMode[statex->localRankToRank(i)]
            .nvlIntraHost = true;
        supportedInraHostRanksStr.push_back(
            std::to_string(statex->localRankToRank(i)));
      } else {
        CLOGF_SUBSYS(
            INFO,
            INIT,
            "CTRAN-NVL: Rank {} (local rank {} GPU {}) cannot access peer {} (local rank {} busId {:x}), disable NVL support",
            myRank,
            myLocalRank,
            statex->cudaDev(),
            statex->localRankToRank(i),
            i,
            peerBusIds[i]);
      }
    }
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-NVL: Initialized NVL backend on rank {} localRank {}, "
      "commHash {:x} commDesc {} nRanks {} nLocalRanks {} nNodes {}, "
      "supported intra-host peer ranks {}, supported NVL fabric ranks {}",
      myRank,
      myLocalRank,
      statex->commHash(),
      statex->commDesc(),
      statex->nRanks(),
      statex->nLocalRanks(),
      statex->nNodes(),
      vecToStr(supportedInraHostRanksStr).c_str(),
      vecToStr(nvlFabricSupportedRanksStr));

  return;
}

CtranNvl::~CtranNvl() {
  CLOGF_TRACE(
      INIT,
      "CTRAN-NVL: Destroyed NVL backend on rank {} localRank {}",
      this->pimpl_->comm->statex_->rank(),
      this->pimpl_->comm->statex_->localRank());
}

bool CtranNvl::isSupported(int rank) {
  FB_CHECKABORT(
      rank < this->pimpl_->nvlRankSupportMode.size(),
      "CTRAN-NVL : rank {} should be smaller than nvlRankSupportMode's size {}.",
      rank,
      this->pimpl_->nvlRankSupportMode.size());
  return this->pimpl_ &&
      (this->pimpl_->nvlRankSupportMode[rank].nvlFabric ||
       this->pimpl_->nvlRankSupportMode[rank].nvlIntraHost);
}

bool CtranNvl::isNvlFabric(int rank) const {
  FB_CHECKABORT(
      rank < this->pimpl_->nvlRankSupportMode.size(),
      "CTRAN-NVL : rank {} should be smaller than nvlRankSupportMode's size {}.",
      rank,
      this->pimpl_->nvlRankSupportMode.size());
  return this->pimpl_->nvlRankSupportMode[rank].nvlFabric;
}
