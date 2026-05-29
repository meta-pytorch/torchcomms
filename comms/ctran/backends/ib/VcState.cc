// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/backends/ib/VcState.h"

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::ib {

void VcState::init(void* owner, const CommLogData& logData, int nRanks) {
  owner_ = owner;
  logData_ = &logData;
  // assign the lock-free vcStateMaps ptr for fast path access when applicable
  {
    auto locked = vcStateMaps_.wlock();
    vcStateMapsPtr_ = &(*locked);
  }
  if (nRanks > 0) {
    // initialize connection map before preConnect() is called by any algorithm
    connectedPeerMap_.resize(nRanks, false);
  }
}

void VcState::releaseAll() {
  { // Explicitly release all virtual connections before destroying cq.
    auto locked = vcStateMaps_.wlock();
    locked->qpToVcMap.clear();
    locked->rankToVcs.clear();
    vcStateMapsPtr_ = nullptr;
  }
  connectedPeerMap_.clear();
}

commResult_t VcState::setupAndPublishVc(
    std::vector<std::shared_ptr<CtranIbVirtualConn>> vcs,
    const std::vector<std::string>& remoteVcIdentifiers,
    int peerRank) {
  if (vcs.empty() || vcs.size() != remoteVcIdentifiers.size()) {
    CLOGF(
        ERR,
        "CTRAN-IB: setupAndPublishVc called with vcs.size()={} remoteVcIdentifiers.size()={} for peerRank {} in pimpl {} commHash {:x} commDesc {}",
        vcs.size(),
        remoteVcIdentifiers.size(),
        peerRank,
        owner_,
        logData_->commHash,
        logData_->commDesc);
    return commInternalError;
  }

  // Setup each VC under its own per-VC lock. The VC is not yet exposed
  // to local rank; the lock is taken to follow VC thread-safety semantics.
  for (size_t i = 0; i < vcs.size(); ++i) {
    const std::lock_guard<std::mutex> lock(vcs[i]->mutex);
    FB_COMMCHECKTHROW_EX(
        vcs[i]->setupVc((void*)remoteVcIdentifiers[i].data()), *logData_);
  }

  {
    auto locked = vcStateMaps_.wlock();
    if (locked->rankToVcs.find(peerRank) != locked->rankToVcs.end()) {
      CLOGF(
          ERR,
          "CTRAN-IB: VirtualConnection (VC) already exists for peerRank {} in pimpl {} commHash {:x}, commDesc {}. It likely indicates a COMM bug.",
          peerRank,
          owner_,
          logData_->commHash,
          logData_->commDesc);
      return commInternalError;
    }

    // Insert each VC's QPs into qpToVcMap. Key ctrl/notify/atomic QPs by
    // the VC's control device (NIC 0 for the legacy non-pinned VC, the
    // pinned NIC for multi-VC). Data QPs are keyed by getIbDevFromQpIdx().
    for (const auto& vc : vcs) {
      const int ctrlDevice = vc->getCtrlDevice();
      std::shared_ptr<CtranIbVirtualConn> vcRef = vc;
      QpUniqueId controlQpId =
          std::make_pair(vc->getControlQpNum(), ctrlDevice);
      if (checkAndInsertQpToVcMap(locked->qpToVcMap, controlQpId, vcRef) !=
          commSuccess) {
        return commInternalError;
      }
      QpUniqueId notifyQpId = std::make_pair(vc->getNotifyQpNum(), ctrlDevice);
      if (checkAndInsertQpToVcMap(locked->qpToVcMap, notifyQpId, vcRef) !=
          commSuccess) {
        return commInternalError;
      }
      QpUniqueId atomicQpId = std::make_pair(vc->getAtomicQpNum(), ctrlDevice);
      if (checkAndInsertQpToVcMap(locked->qpToVcMap, atomicQpId, vcRef) !=
          commSuccess) {
        return commInternalError;
      }
      std::vector<uint32_t> dataQps = vc->getDataQpNums();
      for (int qpIdx = 0; qpIdx < static_cast<int>(dataQps.size()); ++qpIdx) {
        int device = vc->getIbDevFromQpIdx(qpIdx);
        QpUniqueId qpId = std::make_pair(dataQps[qpIdx], device);
        if (checkAndInsertQpToVcMap(locked->qpToVcMap, qpId, vcRef) !=
            commSuccess) {
          return commInternalError;
        }
      }
    }

    // Log each established VC while we still own a stable reference.
    for (const auto& vc : vcs) {
      CLOGF_SUBSYS(
          INFO,
          INIT,
          "CTRAN-IB: Established connection: commHash {:x}, commDesc {}, "
          "vc {}, rank {}, peer {}, control qpn {}, notify qpn {}, atomic qpn {}, data qpns {}",
          logData_->commHash,
          logData_->commDesc,
          (void*)vc.get(),
          logData_->rank,
          peerRank,
          vc->getControlQpNum(),
          vc->getNotifyQpNum(),
          vc->getAtomicQpNum(),
          vecToStr(vc->getDataQpNums()));
    }

    locked->rankToVcs[peerRank] = std::move(vcs);
  }

  return commSuccess;
}

const std::vector<std::shared_ptr<CtranIbVirtualConn>>* VcState::tryGetVcs(
    int peerRank) {
  auto locked = vcStateMaps_.rlock();
  auto it = locked->rankToVcs.find(peerRank);
  if (it == locked->rankToVcs.end()) {
    return nullptr;
  }
  // Value-reference stability via std::unordered_map: the pointer
  // remains valid until releaseAll() / rankToVcs.erase(peerRank).
  return &it->second;
}

commResult_t VcState::checkAndInsertQpToVcMap(
    folly::F14FastMap<QpUniqueId, std::shared_ptr<CtranIbVirtualConn>>& map,
    QpUniqueId& qpId,
    std::shared_ptr<CtranIbVirtualConn>& vc) {
  if (map.find(qpId) != map.end()) {
    CLOGF(
        ERR,
        "CTRAN-IB: QP {} on device {} already exists in pimpl {} commHash {:x}, commDesc {}. It likely indicates a COMM bug.",
        qpId.first,
        qpId.second,
        owner_,
        logData_->commHash,
        logData_->commDesc);
    return commInternalError;
  }
  map.emplace(qpId, vc);
  return commSuccess;
}

} // namespace ctran::ib
