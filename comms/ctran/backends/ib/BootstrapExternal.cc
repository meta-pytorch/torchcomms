// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/backends/ib/BootstrapExternal.h"

#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/ctran/backends/ib/VcState.h"
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
  auto vc = std::make_shared<CtranIbVirtualConn>(
      devices_, peerRank, comm_, trafficClass_, cudaDev_);

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

  return vcState_.setupAndPublishVc(
      std::move(vc), remoteVcIdentifier, peerRank);
}

} // namespace ctran::ib
