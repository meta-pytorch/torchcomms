// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <comms/ctran/commstate/CommStateX.h>
#include "comms/ctran/backends/nvl/CtranNvl.h"
#include "comms/ctran/backends/nvl/CtranNvlImpl.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/utils/logger/LogUtils.h"

commResult_t CtranNvl::Impl::importRemReg(
    const int peer,
    const ctran::utils::CtranIpcDesc& ipcDesc,
    void** mappedBase) {
  std::lock_guard<std::mutex> lock(remRegMapMutex);
  uint64_t base = reinterpret_cast<uint64_t>(ipcDesc.base);

  // Already mapped handle
  auto peerIt = this->remRegMap.find(peer);
  if (peerIt != this->remRegMap.end()) {
    auto baseIt = peerIt->second.find(base);
    if (peerIt->second.find(base) != peerIt->second.end()) {
      *mappedBase = baseIt->second->ipcRemMem.getBase();
      return commSuccess;
    }
  }

  std::unique_ptr<CtranNvlRemRegElem> reg = nullptr;
  try {
    reg = std::make_unique<CtranNvlRemRegElem>(
        ipcDesc, this->comm->statex_->cudaDev(), &this->comm->logMetaData_);
  } catch (std::exception& e) {
    CLOGF(
        WARN,
        "CTRAN-NVL: failed to import remote registration from peer {} ipcDesc {}, error {}",
        peer,
        ipcDesc.toString(),
        e.what());
    return ErrorStackTraceUtil::log(commInternalError);
  }

  CLOGF_TRACE(
      COLL,
      "CTRAN-NVL: cache remote registration rank:base=<{}:{}> {}",
      peer,
      reinterpret_cast<void*>(ipcDesc.base),
      reg->toString());

  *mappedBase = reg->ipcRemMem.getBase();
  this->remRegMap[peer][base] = std::move(reg); // cache

  return commSuccess;
}

commResult_t CtranNvl::Impl::releaseRemReg(int peer, void* basePtr) {
  std::lock_guard<std::mutex> lock(remRegMapMutex);
  uint64_t base = reinterpret_cast<uint64_t>(basePtr);

  if (this->remRegMap.find(peer) == this->remRegMap.end() ||
      this->remRegMap[peer].find(base) == this->remRegMap[peer].end()) {
    CLOGF(
        ERR,
        "Unknown remote memory registration from peer {} base {}",
        peer,
        basePtr);
    return ErrorStackTraceUtil::log(commInternalError);
  }

  CLOGF_TRACE(
      COLL,
      "CTRAN-NVL: remove remote registration from cache rank:base=<{}:{}> : {}",
      peer,
      basePtr,
      this->remRegMap[peer][base]->toString());

  // Remove from cache
  // Memory and handle will be released in ~CtranIpcRemMem()
  try {
    this->remRegMap[peer].erase(base);
  } catch (std::exception& e) {
    CLOGF(
        WARN,
        "CTRAN-NVL: failed to remove remote registration from cache rank:base=<{}:{}>, error {}",
        peer,
        basePtr,
        e.what());
    return ErrorStackTraceUtil::log(commInternalError);
  }

  return commSuccess;
}

void CtranNvl::Impl::clearAllRemReg() {
  std::lock_guard<std::mutex> lock(remRegMapMutex);

  // Release all remote registrations in case any of them are not deregistered
  // by user program
  for (auto& [peer, regs] : this->remRegMap) {
    CLOGF_TRACE(
        INIT,
        "CTRAN-NVL: clear all {} cached remote registrations from peer {}",
        regs.size(),
        peer);
  }

  // Memory and handle will be released in ~CtranIpcRemMem()
  this->remRegMap.clear();
}

size_t CtranNvl::Impl::getNumRemMem(int peerRank) const {
  std::lock_guard<std::mutex> lock(remRegMapMutex);

  auto it = remRegMap.find(peerRank);
  if (it != remRegMap.end()) {
    return it->second.size();
  } else {
    return 0;
  }
}
