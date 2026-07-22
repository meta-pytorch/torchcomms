// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/common/NvlUtils.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/logger/ScubaLogger.h"

namespace ctran::allgatherp {
inline void* getPtr(void* base, size_t offset) {
  return (void*)((uintptr_t)base + offset);
}

using ctran::algos::copyToSelf;
using ctran::algos::nvlBarrier;
using ctran::algos::nvlCeBcast;

// Exchange intra-node NVL IPC handles into pArgs and mark initialized. Owns
// the remote-buffer resize, the intraAllGatherCtrl + intraBarrier, per-peer
// IPC logging, and setting initState. Logs its timing breakdown on the GPE
// thread. INVARIANT: must run on the GPE thread (submitted via submitHost),
// which already holds the mapper epoch through the worker loop
// (CtranGpeImpl.cc); never call it inline on a user thread. Shared verbatim
// by the eager and graph-capture init paths.
inline commResult_t exchangeIpcReg(CtranComm* comm, PersistArgs& pArgs) {
  auto* mapper = comm->ctran_->mapper.get();

  const auto* const statex = comm->statex_.get();
  const int nRanks = statex->nRanks();
  pArgs.remoteRecvBuffs.assign(nRanks, nullptr);
  pArgs.remoteAccessKeys.assign(nRanks, CtranMapperRemoteAccessKey());

  CtranMapperTimer ctrlTimer;
  FB_COMMCHECK(mapper->intraAllGatherCtrl(
      pArgs.recvbuff,
      pArgs.recvHdl,
      pArgs.remoteRecvBuffs,
      pArgs.remoteAccessKeys,
      pArgs.remoteIpcRegHdls_));
  const double ctrlUs = ctrlTimer.durationUs();

  CtranMapperTimer barrierTimer;
  FB_COMMCHECK(mapper->intraBarrier());
  const double barrierUs = barrierTimer.durationUs();

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-AGP: Rank {} IPC exchange breakdown: intraAllGatherCtrl {} us, intraBarrier {} us, nLocalRanks {}",
      statex->rank(),
      ctrlUs,
      barrierUs,
      statex->nLocalRanks());

  NcclScubaEvent(
      std::make_unique<CommEvent>(
          &comm->logMetaData_,
          "AgpCreate/IpcExchange/IntraAllGatherCtrl",
          std::string(),
          ctrlUs / 1000.0))
      .record();
  NcclScubaEvent(
      std::make_unique<CommEvent>(
          &comm->logMetaData_,
          "AgpCreate/IpcExchange/IntraBarrier",
          std::string(),
          barrierUs / 1000.0))
      .record();

  const int myRank = statex->rank();
  const int nLocalRanks = statex->nLocalRanks();
  for (int i = 0; i < nLocalRanks; ++i) {
    const int peerRank = statex->localRankToRank(i);
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-AGP     Peer {}: addr {} ipcImport {}",
        i,
        pArgs.remoteRecvBuffs[peerRank],
        myRank == peerRank ? "(local)"
                           : pArgs.remoteIpcRegHdls_.at(i).toString());
  }

  pArgs.initState = InitState::kInitialized;
  return commSuccess;
}
} // namespace ctran::allgatherp
