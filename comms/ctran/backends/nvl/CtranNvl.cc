// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/nvl/CtranNvl.h"
#include "comms/ctran/backends/nvl/CtranNvlImpl.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/logger/LogUtils.h"

CtranNvl::CtranNvl(CtranComm* comm) {
  const auto statex = comm->statex_.get();
  int myRank = statex->rank();
  int myLocalRank = statex->localRank();
  int nLocalRanks = statex->nLocalRanks();
  // Exchange device IDs used by each local rank
  std::vector<int> peerDevs(nLocalRanks, 0);
  std::vector<std::string> supportedInraHostRanksStr;
  std::vector<std::string> nvlFabricSupportedRanksStr;

  peerDevs[myLocalRank] = statex->cudaDev();
  auto resFuture = comm->bootstrap_->allGatherIntraNode(
      peerDevs.data(),
      sizeof(int),
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
      int canAccessPeer = 1;
      FB_CUDACHECKTHROW_EX(
          cudaDeviceCanAccessPeer(
              &canAccessPeer, statex->cudaDev(), peerDevs[i]),
          comm->logMetaData_);
      if (canAccessPeer) {
        this->pimpl_->nvlRankSupportMode[statex->localRankToRank(i)]
            .nvlIntraHost = true;
        supportedInraHostRanksStr.push_back(
            std::to_string(statex->localRankToRank(i)));
      } else {
        CLOGF_SUBSYS(
            INFO,
            INIT,
            "CTRAN-NVL: Rank {} (local rank {} GPU {}) cannot access peer {} (local rank {} GPU {}), disable NVL support",
            myRank,
            myLocalRank,
            statex->cudaDev(),
            statex->localRankToRank(i),
            i,
            peerDevs[i]);
      }
    }
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-NVL: Initialized NVL backend on rank {} localRank {}, supported "
      "intra-host peer ranks {}, supported NVL fabric ranks {}",
      myRank,
      myLocalRank,
      vecToStr(supportedInraHostRanksStr).c_str(),
      vecToStr(nvlFabricSupportedRanksStr));

  return;
}

CtranNvl::~CtranNvl() {
  this->pimpl_->clearAllRemReg();
  CLOGF_TRACE(
      INIT,
      "CTRAN-NVL: Destroyed NVL backend on rank {} localRank {}",
      this->pimpl_->comm->statex_->rank(),
      this->pimpl_->comm->statex_->localRank());
}

void CtranNvl::regCtrlCb(std::unique_ptr<CtranCtrlManager>& ctrlMgr) {
  // Register callback for NVL remote release ctrl msg
  ctrlMgr->regCb(
      ControlMsgType::NVL_RELEASE_MEM,
      releaseMemCb,
      this->pimpl_.get() /* ctx */);
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

commResult_t CtranNvl::regMem(
    const void* buf,
    const size_t len,
    const int cudaDev,
    void** nvlRegElem,
    bool shouldSupportCudaMalloc) {
  auto reg = new CtranNvlRegElem(buf, len, cudaDev);
  bool supported = false;

  FB_COMMCHECK(reg->tryLoad(supported, shouldSupportCudaMalloc));
  if (supported) {
    CLOGF_TRACE(COLL, "CTRAN-NVL: Registered memory {}", reg->toString());

    *nvlRegElem = reinterpret_cast<void*>(reg);
  } else {
    // Return nullptr to indicate unsupported memory type
    delete reg;
    *nvlRegElem = nullptr;
  }

  return commSuccess;
}

commResult_t CtranNvl::deregMem(void* nvlRegElem) {
  auto reg = reinterpret_cast<CtranNvlRegElem*>(nvlRegElem);

  CLOGF_TRACE(COLL, "CTRAN-NVL: Deregistered memory {}", reg->toString());
  // Memory handle release in ~CtranIpcMem()
  delete reg;
  return commSuccess;
}

commResult_t
CtranNvl::exportMem(const void* buf, void* nvlRegElem, ControlMsg& msg) {
  auto reg = reinterpret_cast<CtranNvlRegElem*>(nvlRegElem);

  // Fill ctrl msg content
  msg.setType(ControlMsgType::NVL_EXPORT_MEM);
  auto ipcMem = reg->ipcMem.wlock();
  FB_COMMCHECK(ipcMem->ipcExport(msg.nvlExp.ipcDesc));
  msg.nvlExp.offset = reinterpret_cast<size_t>(buf) -
      reinterpret_cast<size_t>(ipcMem->getBase());

  return commSuccess;
}

commResult_t CtranNvl::importMem(
    void** buf,
    struct CtranNvlRemoteAccessKey* key,
    int rank,
    const ControlMsg& msg) {
  void* basePtr = nullptr;
  FB_COMMCHECK(this->pimpl_->importRemReg(rank, msg.nvlExp.ipcDesc, &basePtr));

  // import from baseAddr of a remote segment, return buf at offset from
  // baseAddr
  *buf = reinterpret_cast<char*>(basePtr) + msg.nvlExp.offset;

  key->peerRank = rank;
  key->basePtr = msg.nvlExp.ipcDesc.base;

  CLOGF_TRACE(
      COLL,
      "CTRAN-NVL: Imported remote mem from rank {}: buf {} (base {} offset {})",
      rank,
      (void*)*buf,
      (void*)basePtr,
      msg.nvlExp.offset);
  return commSuccess;
}

commResult_t CtranNvl::remReleaseMem(void* nvlRegElem, ControlMsg& msg) {
  auto reg = reinterpret_cast<CtranNvlRegElem*>(nvlRegElem);
  msg.setType(ControlMsgType::NVL_RELEASE_MEM);
  msg.nvlRls.base = reg->ipcMem.rlock()->getBase();
  return commSuccess;
}

commResult_t CtranNvl::releaseMemCb(int rank, void* msgPtr, void* ctx) {
  auto nvlImpl = reinterpret_cast<CtranNvl::Impl*>(ctx);
  auto msg = reinterpret_cast<ControlMsg*>(msgPtr);
  auto& deregMsg = msg->nvlRls;
  CLOGF_TRACE(
      COLL,
      "CTRAN-NVL: Handle received CB ctrlmsg from rank {}: {}",
      rank,
      msg->toString());
  return nvlImpl->releaseRemReg(rank, deregMsg.base);
}

commResult_t CtranNvl::releaseMem(CtranNvlRemoteAccessKey* rkey) {
  CLOGF_TRACE(
      COLL,
      "CTRAN-NVL: Release imported memory from rank {} base {}",
      rkey->peerRank,
      (void*)rkey->basePtr);
  return pimpl_->releaseRemReg(rkey->peerRank, rkey->basePtr);
}

size_t CtranNvl::getNumRemMem(int peerRank) const {
  return pimpl_->getNumRemMem(peerRank);
}
