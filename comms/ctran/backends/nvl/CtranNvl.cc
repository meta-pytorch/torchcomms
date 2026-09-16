// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/nvl/CtranNvl.h"
#include "comms/ctran/backends/nvl/CtranNvlImpl.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CtranLogUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/utils/StrUtils.h"

CtranNvl::CtranNvl(CtranComm* comm) {
  const auto statex = comm->statex_.get();
  int myRank = statex->rank();
  int myLocalRank = statex->localRank();
  int nLocalRanks = statex->nLocalRanks();
  const bool precomputedTopology = statex->hasPrecomputedTopology();
  std::vector<std::string> supportedInraHostRanksStr;
  std::vector<std::string> nvlFabricSupportedRanksStr;

  std::vector<int> peerDevs;
  if (!precomputedTopology) {
    peerDevs.resize(nLocalRanks, 0);
    peerDevs[myLocalRank] = statex->cudaDev();
    auto resFuture = comm->bootstrap_->allGatherNvlDomain(
        peerDevs.data(),
        sizeof(int),
        myLocalRank,
        nLocalRanks,
        statex->localRankToRanks());
    FB_COMMCHECKTHROW_EX(
        static_cast<commResult_t>(std::move(resFuture).get()),
        comm->logMetaData_);
  }

  this->pimpl_ = std::make_unique<Impl>();
  this->pimpl_->comm = comm;
  this->pimpl_->nvlRankSupportMode.resize(statex->nRanks());

  // Check IPC support for each peer
  for (int i = 0; i < nLocalRanks; ++i) {
    const int peerRank = statex->localRankToRank(i);
    if (precomputedTopology) {
      const bool sameHost = statex->host(myRank) == statex->host(peerRank);
      if (sameHost) {
        this->pimpl_->nvlRankSupportMode[peerRank].nvlIntraHost = true;
        supportedInraHostRanksStr.push_back(std::to_string(peerRank));
      } else if (statex->nvlFabricEnabled()) {
        this->pimpl_->nvlRankSupportMode[peerRank].nvlFabric = true;
        nvlFabricSupportedRanksStr.push_back(std::to_string(peerRank));
      } else {
        FB_ERRORTHROW_EX(
            commInternalError,
            comm->logMetaData_,
            "CTRAN-NVL: precomputed non-fabric domain crosses hosts for rank {}",
            peerRank);
      }
      continue;
    }
    // if supported, update nvlFabric support mode for each peer
    if (statex->nvlFabricEnabled()) {
      if (NCCL_MNNVL_TRUNK_DISABLE) {
        bool p2pAccess =
            comm->statex_->isSameDeviceRack(comm->logMetaData_.rank, peerRank);
        if (!p2pAccess) {
          CTRAN_LOG_SUBSYS(
              INFO,
              INIT,
              "NCCL_MNNVL_TRUNK_DISABLE set to True. P2P disabled between rank1: {} rank2: {} because rackserial mismatch",
              comm->logMetaData_.rank,
              peerRank);
          continue;
        }
      }
      this->pimpl_->nvlRankSupportMode[peerRank].nvlFabric = true;
      nvlFabricSupportedRanksStr.push_back(std::to_string(peerRank));
    } else {
      if (myLocalRank == i) {
        this->pimpl_->nvlRankSupportMode[peerRank].nvlIntraHost = true;
        supportedInraHostRanksStr.push_back(std::to_string(peerRank));
        continue;
      }
      int canAccessPeer = 1;
      FB_CUDACHECKTHROW_EX(
          cudaDeviceCanAccessPeer(
              &canAccessPeer, statex->cudaDev(), peerDevs[i]),
          comm->logMetaData_);
      if (canAccessPeer) {
        this->pimpl_->nvlRankSupportMode[peerRank].nvlIntraHost = true;
        supportedInraHostRanksStr.push_back(std::to_string(peerRank));
      } else {
        CTRAN_LOG_SUBSYS(
            INFO,
            INIT,
            "CTRAN-NVL: Rank {} (local rank {} GPU {}) cannot access peer {} (local rank {} GPU {}), disable NVL support",
            myRank,
            myLocalRank,
            statex->cudaDev(),
            peerRank,
            i,
            peerDevs[i]);
      }
    }
  }

  CTRAN_LOG_SUBSYS(
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
  CTRAN_LOG_TRACE(
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
