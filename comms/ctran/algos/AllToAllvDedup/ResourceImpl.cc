// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstddef>

#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"
#include "comms/ctran/algos/AllToAllvDedup/ResourceImpl.h"
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/algos/common/BufManager.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::alltoallvdedup {
using ::ctran::algos::bufmanager::MemType;

using algos::GpeKernelSync;
using algos::bufmanager::BasicBuf;
using algos::bufmanager::MemType;
using algos::bufmanager::RegBuf;
using algos::bufmanager::RemRegBuf;
using BufName = ResourceBufName;

namespace {
#define CHECK_BUF(bufMngr_, md, bufName)                           \
  FB_CHECKABORT(                                                   \
      bufMngr_->contains(md.memType, bufName),                     \
      "Invalid assignement! Memtype {} buf {} is not initialized", \
      algos::bufmanager::memTypeToStr(md.memType).c_str(),         \
      md.str);

#define SET_REM_REGBUF(bufMngr_, bufMdMap_, bufName, peerRanks)             \
  {                                                                         \
    const auto& md = bufMdMap_.at(bufName);                                 \
    /* Omit check since already checked in SET_REGBUF*/                     \
    auto& remRegBufVec = ref_.bufs.remRegBufs[(size_t)bufName];             \
    auto ret = bufMngr_->assignRemRegBuf(                                   \
        md.memType, bufName, peerRanks, remRegBufVec);                      \
    FB_CHECKABORT(                                                          \
        ret, "Failed to assign remote regbuf {} {}", md.str, (int)bufName); \
    for (int i = 0; i < peerRanks.size(); i++) {                            \
      CLOGF_TRACE(                                                          \
          INIT,                                                             \
          "Rank {} assigned {} remRegBufVec[{}] = {}",                      \
          statex_->rank(),                                                  \
          md.str,                                                           \
          i,                                                                \
          remRegBufVec[i].toString());                                      \
    }                                                                       \
  }

void prepareExchange(
    ncclx::CommStateX* statex,
    std::vector<int>& localPeers,
    std::vector<int>& railPeers,
    std::vector<int>& allPeers) {
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nNodes = statex->nNodes();
  const auto myNode = statex->node();
  const auto myLocalRank = statex->localRank();

  // Include intra-node peers and rail peers
  localPeers.reserve(nLocalRanks);
  railPeers.reserve(nNodes);
  allPeers.reserve(nLocalRanks + nNodes - 1);
  for (int i = 0; i < nLocalRanks; i++) {
    const auto rank = statex->localRankToRank(i, myNode);
    localPeers.push_back(rank);
    allPeers.push_back(rank);
  }
  for (int n = 0; n < nNodes; n++) {
    const auto rank = statex->localRankToRank(myLocalRank, n);
    railPeers.push_back(rank);
    if (n != myNode) {
      allPeers.push_back(rank);
    }
  }
}
} // namespace

commResult_t ResourceImpl::setKSync(cudaStream_t stream, const bool skipRem) {
  auto& kSync = ref_.kSync;
  const auto useReg = skipRem ? false : true;
  GET_RESOURCE_BUFPTR(&ref_, kWorkerGroupSync, useReg, kSync.wgSyncs);
  GET_RESOURCE_BUFPTR(&ref_, kIntraRedSync, useReg, kSync.intraRedSync);
  GET_RESOURCE_BUFPTR(&ref_, kFwdRecvSync, useReg, kSync.fwdRecvSyncs);
  GET_RESOURCE_BUFPTR(
      &ref_, kIntraFwdRecvSync, useReg, kSync.intraFwdRecvSyncs);
  GET_RESOURCE_BUFPTR(&ref_, kSendGKSyncs, useReg, kSync.sendGKSyncs);
  GET_RESOURCE_BUFPTR(&ref_, kRecvGKSyncs, useReg, kSync.recvGKSyncs);
  GET_RESOURCE_BUFPTR(&ref_, kIntraFwdGKSyncs, useReg, kSync.intraFwdGKSyncs);
  GET_RESOURCE_BUFPTR(&ref_, kRecvCopyGKSyncs, useReg, kSync.recvCopyGKSyncs);
  GET_RESOURCE_BUFPTR(
      &ref_, kIntraRecvCopyGKSyncs, useReg, kSync.intraRecvCopyGKSyncs);

  if (!skipRem) {
    GET_RESOURCE_REM_BUFPTRS(&ref_, kFwdRecvSync, kSync.remFwdRecvSyncs);
    GET_RESOURCE_REM_BUFPTRS(
        &ref_, kIntraFwdRecvSync, kSync.remIntraFwdRecvSyncs);
  }

  // Initialize fwdRecvSync only once at init; other kSync objects are reset
  // before exec()/combine().
  auto numSyncs =
      bufMdMap_.at(BufName::kFwdRecvSync).buflen / sizeof(FwdRecvSync);
  std::vector<FwdRecvSync> fwdRecvSyncH(numSyncs, FwdRecvSync());

  FB_CUDACHECK(cudaMemcpyAsync(
      kSync.fwdRecvSyncs,
      fwdRecvSyncH.data(),
      sizeof(FwdRecvSync) * numSyncs,
      cudaMemcpyHostToDevice,
      stream));
  numSyncs =
      bufMdMap_.at(BufName::kIntraFwdRecvSync).buflen / sizeof(FwdRecvSync);
  fwdRecvSyncH.clear();
  fwdRecvSyncH.resize(numSyncs, FwdRecvSync());
  FB_CUDACHECK(cudaMemcpyAsync(
      kSync.intraFwdRecvSyncs,
      fwdRecvSyncH.data(),
      sizeof(FwdRecvSync) * numSyncs,
      cudaMemcpyHostToDevice,
      stream));
  return commSuccess;
}

void ResourceImpl::setRef() {
  for (int i = 0; i < static_cast<int>(BufName::kNumBufsNames); i++) {
    const auto bname = static_cast<BufName>(i);
    const auto& md = bufMdMap_.at(bname);
    CHECK_BUF(bufMngr_, md, bname);
    auto& buf = ref_.bufs.bufs[(size_t)bname];
    auto ret = bufMngr_->assignBuf(md.memType, bname, buf);
    FB_CHECKABORT(ret, "Failed to assign buf {} {}", md.str, i);
    CLOGF_TRACE(
        INIT,
        "Rank {} assigned {} = {}",
        statex_->rank(),
        md.str,
        buf.toString());
  }
}

void ResourceImpl::setRegRef() {
  // Assign all local buffers as regBuf to simplify ptr query in algorithm
  for (int i = 0; i < static_cast<int>(BufName::kNumBufsNames); i++) {
    const auto bname = static_cast<BufName>(i);
    const auto& md = bufMdMap_.at(bname);
    CHECK_BUF(bufMngr_, md, bname);
    auto& regBuf = ref_.bufs.regBufs[(size_t)bname];
    auto ret = bufMngr_->assignRegBuf(md.memType, bname, regBuf);
    FB_CHECKABORT(ret, "Failed to assign regbuf {} {}", md.str, i);
    CLOGF_TRACE(
        INIT,
        "Rank {} assigned {} = {}",
        statex_->rank(),
        md.str,
        regBuf.toString());
  }
}

void ResourceImpl::setRemRef(
    std::vector<int>& localPeers,
    std::vector<int>& railPeers) {
  SET_REM_REGBUF(bufMngr_, bufMdMap_, BufName::kTmpFwdBuff, railPeers);
  SET_REM_REGBUF(bufMngr_, bufMdMap_, BufName::kTmpRecvBuff, localPeers);
  SET_REM_REGBUF(bufMngr_, bufMdMap_, BufName::kTmpIntraRecvBuff, localPeers);
  SET_REM_REGBUF(bufMngr_, bufMdMap_, BufName::kFwdRecvSync, localPeers);
  SET_REM_REGBUF(bufMngr_, bufMdMap_, BufName::kIntraFwdRecvSync, localPeers);
}

commResult_t ResourceImpl::initialize(
    const PersistArgs& args,
    const PersistConfig& config,
    cudaStream_t stream,
    const bool skipRem) {
  const auto nNodes = statex_->nNodes();
  const auto nLocalRanks = statex_->nLocalRanks();
  const auto nRanks = statex_->nRanks();

  SetCudaDevRAII setCudaDev(statex_->cudaDev());

  // initialize buffer metadata
  initBufMd(args, config, nNodes, nLocalRanks);

  for (int i = 0; i < static_cast<int>(BufName::kNumBufsNames); i++) {
    const auto bname = static_cast<BufName>(i);
    const auto md = bufMdMap_.at(bname);
    FB_COMMCHECK(bufMngr_->insert(md.memType, bname, md.buflen));
  }

  // allocate memory
  FB_COMMCHECK(bufMngr_->commit());
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "AllToAllvDedup::ResourceImpl: Rank {} commited buffer allocation",
      statex_->rank());

  if (!skipRem) {
    std::vector<int> localPeers, railPeers, allPeers;
    // register and exchange memory handles
    prepareExchange(statex_, localPeers, railPeers, allPeers);
    CLOGF_TRACE(
        INIT,
        "AllToAllvDedup::ResourceImpl: Rank {} exchanged with allPeers [{}]",
        statex_->rank(),
        folly::join(",", allPeers));
    FB_COMMCHECK(bufMngr_->exchange(allPeers, nRanks));

    // assign buffers from allocated memory and keep a const copy of it for
    // later algorithm to use; if not skip remote exchange, all buffers are
    // registered and remote exchanged
    setRegRef();
    setRemRef(localPeers, railPeers);
  } else {
    // if skip remote, set as basic buffer
    setRef();
  }

  FB_COMMCHECK(setKSync(stream, skipRem));
  return commSuccess;
}

commResult_t ResourceImpl::destroy() {
  FB_COMMCHECK(bufMngr_->release());

  return commSuccess;
}

ResourceImpl::~ResourceImpl() {
  FB_COMMCHECKIGNORE(destroy());
}

ResourceImpl::ResourceImpl(
    ncclx::CommStateX* statex,
    CtranMapper* mapper,
    CommLogData* logMetadata)
    : statex_(statex), mapper_(mapper), logMetaData_(logMetadata) {
  // memory pool requires unique key for each memory region allocation
  auto memKey = folly::sformat(
      "Ctran::AllToAllvDedup::ResourceImpl-{:#x}", statex->commHash());

  bufMngr_ = std::make_unique<
      ::ctran::algos::BufManager<BufName, BufName::kNumBufsNames>>(
      statex, mapper, logMetadata, memKey);
}

} // namespace ctran::alltoallvdedup
