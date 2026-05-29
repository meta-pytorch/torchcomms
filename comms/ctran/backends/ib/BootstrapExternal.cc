// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/backends/ib/BootstrapExternal.h"

#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/ctran/backends/ib/VcState.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::ib {

BootstrapExternal::BootstrapExternal(
    VcState& vcState,
    std::vector<CtranIbDevice>& devices,
    CtranComm* comm,
    int cudaDev,
    uint64_t commHash,
    const std::string& commDesc,
    const CommLogData& logData,
    uint32_t trafficClass)
    : vcState_(vcState),
      devices_(devices),
      comm_(comm),
      cudaDev_(cudaDev),
      commHash_(commHash),
      commDesc_(commDesc),
      logData_(logData),
      trafficClass_(trafficClass) {}

std::string BootstrapExternal::getLocalVcId(const int peerRank) {
  // External bootstrap supports exactly one VC per peer that spans all
  // local NICs (no NIC pinning). This matches the legacy single-VC
  // semantics and avoids depending on the cvar-derived VcLayout used by
  // the internal bootstrap path.
  std::vector<int> activeDevices(devices_.size());
  std::iota(activeDevices.begin(), activeDevices.end(), 0);

  // Single VC per peer -> per-VC MAX_QPS == per-peer MAX_QPS. Value
  // depends on the peer's connection class, so resolve per call.
  int maxQpsPerVc =
      CtranIbVirtualConn::computeMaxQpsPerVc(comm_, peerRank, /*numVcs=*/1);

  auto vc = std::make_shared<CtranIbVirtualConn>(
      devices_,
      peerRank,
      comm_,
      trafficClass_,
      cudaDev_,
      activeDevices,
      maxQpsPerVc);

  std::string localBusCard;
  {
    const std::lock_guard<std::mutex> lock(vc->mutex);
    localBusCard.resize(vc->getBusCardSize());
    FB_COMMCHECKTHROW_EX(vc->getLocalBusCard(localBusCard.data()), logData_);
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = pendingVcs_.emplace(peerRank, std::move(vc));
    FB_CHECKABORT(
        inserted,
        "BootstrapExternal::getLocalVcId called twice for the same peerRank without a matching connectVc");
  }

  return localBusCard;
}

commResult_t BootstrapExternal::connectVc(
    const std::string& remoteVcIdentifier,
    const int peerRank) {
  std::shared_ptr<CtranIbVirtualConn> vc;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pendingVcs_.find(peerRank);
    if (it == pendingVcs_.end()) {
      CLOGF(
          ERR,
          "CTRAN-IB: BootstrapExternal::connectVc called for peerRank {} before "
          "getLocalVcId. commHash {:x}, commDesc {}",
          peerRank,
          commHash_,
          commDesc_);
      return commInternalError;
    }
    vc = std::move(it->second);
    pendingVcs_.erase(it);
  }

  // External bootstrap publishes a single VC per peer; wrap and delegate
  // to the unified setup+publish entry point, which runs setupVc under
  // vc->mutex.
  std::vector<std::shared_ptr<CtranIbVirtualConn>> vcs;
  vcs.push_back(std::move(vc));
  std::vector<std::string> remoteIds{remoteVcIdentifier};
  return vcState_.setupAndPublishVc(std::move(vcs), remoteIds, peerRank);
}

} // namespace ctran::ib
