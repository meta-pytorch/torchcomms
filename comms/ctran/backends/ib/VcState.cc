// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/backends/ib/VcState.h"

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
    locked->rankToVcMap.clear();
    locked->qpToVcMap.clear();
    vcStateMapsPtr_ = nullptr;
  }
  connectedPeerMap_.clear();
}

commResult_t VcState::setupAndPublishVc(
    std::shared_ptr<CtranIbVirtualConn> vc,
    const std::string& remoteVcIdentifier,
    int peerRank) {
  {
    const std::lock_guard<std::mutex> lock(vc->mutex);
    FB_COMMCHECKTHROW_EX(
        vc->setupVc((void*)remoteVcIdentifier.data()), *logData_);
  }

  uint32_t controlQp = vc->getControlQpNum();
  uint32_t notifyQp = vc->getNotifyQpNum();
  uint32_t atomicQp = vc->getAtomicQpNum();
  std::vector<uint32_t> dataQps = vc->getDataQpNums();

  // Till now VC is not yet exposed to local rank. Local rank can use the VC
  // once updated the vcStateMaps_.
  FB_COMMCHECKTHROW_EX(updateVcState(vc, peerRank), *logData_);

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
      controlQp,
      notifyQp,
      atomicQp,
      vecToStr(dataQps));

  return commSuccess;
}

commResult_t VcState::updateVcState(
    std::shared_ptr<CtranIbVirtualConn> vc,
    int peerRank) {
  auto locked = vcStateMaps_.wlock();
  if (locked->rankToVcMap.find(peerRank) != locked->rankToVcMap.end()) {
    CLOGF(
        ERR,
        "CTRAN-IB: VirtualConnection (VC) already exists for peerRank {} in pimpl {} commHash {:x}, commDesc {}. It likely indicates a COMM bug.",
        peerRank,
        owner_,
        logData_->commHash,
        logData_->commDesc);
    return commInternalError;
  }

  locked->rankToVcMap[peerRank] = vc;
  QpUniqueId controlQpId = std::make_pair(vc->getControlQpNum(), 0);
  if (checkAndInsertQpToVcMap(locked->qpToVcMap, controlQpId, vc) !=
      commSuccess) {
    return commInternalError;
  }
  QpUniqueId notifyQpId = std::make_pair(vc->getNotifyQpNum(), 0);
  if (checkAndInsertQpToVcMap(locked->qpToVcMap, notifyQpId, vc) !=
      commSuccess) {
    return commInternalError;
  }
  QpUniqueId atomicQpId = std::make_pair(vc->getAtomicQpNum(), 0);
  if (checkAndInsertQpToVcMap(locked->qpToVcMap, atomicQpId, vc) !=
      commSuccess) {
    return commInternalError;
  }

  std::vector<uint32_t> dataQps = vc->getDataQpNums();
  for (int qpIdx = 0; qpIdx < dataQps.size(); qpIdx++) {
    int device = vc->getIbDevFromQpIdx(qpIdx);
    QpUniqueId qpId = std::make_pair(dataQps[qpIdx], device);
    if (checkAndInsertQpToVcMap(locked->qpToVcMap, qpId, vc) != commSuccess) {
      return commInternalError;
    }
  }

  return commSuccess;
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
