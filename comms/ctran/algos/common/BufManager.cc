// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/common/BufManager.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/memory/memCacheAllocator.h"

namespace ctran::algos::bufmanager {
commResult_t commitBase(
    MemBase& base,
    const ncclx::CommStateX* statex,
    const CommLogData* logMetaData) {
  switch (base.type) {
    case MemType::kDevice:
      FB_COMMCHECK(
          ncclx::memory::memCacheAllocator::getInstance()->getCachedCuMemById(
              base.memKey,
              &base.ptr,
              /*cuHandle=*/nullptr,
              base.size,
              logMetaData,
              __func__));
      break;
    case MemType::kHostPinned:
      FB_CUDACHECK(cudaHostAlloc(&base.ptr, base.size, cudaHostAllocDefault));
      break;
    default:
      break;
  }

  CLOGF_TRACE(
      INIT,
      "Rank {} allocated {} bufBase: {}",
      statex->rank(),
      base.memKey,
      base.toString());
  return commSuccess;
}

commResult_t releaseBase(
    MemBase& base,
    const ncclx::CommStateX* statex,
    CtranMapper* mapper) {
  // release remote imported registration
  for (auto& rkey : base.remoteAccessKeys) {
    FB_COMMCHECK(mapper->deregRemReg(&rkey));
    // reset to avoid double release
    rkey.backend = CtranMapperBackend::UNSET;
  }
  // release local registration
  if (base.segHdl) {
    FB_COMMCHECK(mapper->deregMem(base.segHdl, true /* skipRemRelease */));
    // reset to avoid double release
    base.segHdl = nullptr;
  }

  CLOGF_TRACE(
      INIT,
      "Rank {} deregistered {} bufBase: {}",
      statex->rank(),
      base.memKey,
      base.toString());

  // release memory
  if (base.ptr) {
    switch (base.type) {
      case MemType::kDevice:
        FB_COMMCHECK(
            ncclx::memory::memCacheAllocator::getInstance()->release(
                {base.memKey}));
        break;
      case MemType::kHostPinned:
        FB_CUDACHECK(cudaFreeHost(base.ptr));
        break;
      default:
        break;
    }
    CLOGF_TRACE(
        INIT,
        "Rank {} released {} bufBase: {}",
        statex->rank(),
        base.memKey,
        base.toString());

    // reset to avoid double release
    base.ptr = nullptr;
  }
  return commSuccess;
}

commResult_t exchangeBase(
    MemBase& base,
    const std::vector<int>& peerRanks,
    const int maxNumRanks,
    const ncclx::CommStateX* statex,
    CtranMapper* mapper) {
  // Skip if base is not allocated
  if (base.size == 0 || !base.ptr) {
    return commSuccess;
  }

  CtranMapperEpochRAII epochRAII(mapper);

  // First register it locally
  FB_COMMCHECK(mapper->regMem(
      base.ptr,
      base.size,
      &base.segHdl,
      true /* forceRegist */,
      true /* ncclManaged */,
      &base.regHdl));
  CLOGF_TRACE(
      INIT,
      "Rank {} allocated {} bufBase: {}",
      statex->rank(),
      base.memKey,
      base.toString());

  // Exchange the registered buffer with peerRanks; reserve for max nRanks space
  // since peerRanks may contain any ranks in the comm
  base.remotePtrs.resize(maxNumRanks, nullptr);
  base.remoteAccessKeys.resize(maxNumRanks, CtranMapperRemoteAccessKey{});
  FB_COMMCHECK(mapper->allGatherCtrl(
      base.ptr,
      base.regHdl,
      peerRanks,
      base.remotePtrs,
      base.remoteAccessKeys));

  // Barrier to ensure all local ranks have finished NVL import before any rank
  // starts destroy
  FB_COMMCHECK(mapper->intraBarrier());
  CLOGF_TRACE(
      INIT,
      "Rank {} exchanged {} bufBase: {} with {} peers",
      statex->rank(),
      base.memKey,
      base.toString(),
      peerRanks.size());
  return commSuccess;
}
}; // namespace ctran::algos::bufmanager
