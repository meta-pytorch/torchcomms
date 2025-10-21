// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstddef>

#include "comms/ctran/algos/AllToAllvDedup/ResourceImpl.h"
#include "comms/ctran/algos/common/BufManager.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/algos/common/MPSCTbSync.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::alltoallvdedup {
using ::ctran::algos::MPSCTbSync;
using ::ctran::algos::bufmanager::MemType;

using algos::GpeKernelSync;
using algos::bufmanager::BasicBuf;
using algos::bufmanager::MemType;
using algos::bufmanager::RegBuf;
using algos::bufmanager::RemRegBuf;

namespace {
#define CHECK_BUF(bufMngr_, memType, bufName)                      \
  FB_CHECKABORT(                                                   \
      bufMngr_->contains(memType, ResourceBufName::bufName),       \
      "Invalid assignement! Memtype {} buf {} is not initialized", \
      algos::bufmanager::memTypeToStr(memType).c_str(),            \
      ARGTOSTR(bufName));

#define SET_REGBUF(bufMngr_, memType, bufName)                             \
  {                                                                        \
    CHECK_BUF(bufMngr_, memType, bufName);                                 \
    auto& regBuf = ref_.bufs.regBufs[(size_t)ResourceBufName::bufName];    \
    auto ret =                                                             \
        bufMngr_->assignRegBuf(memType, ResourceBufName::bufName, regBuf); \
    FB_CHECKABORT(ret, "Failed to assign regbuf {}", ARGTOSTR(bufName));   \
    CLOGF_TRACE(                                                           \
        INIT,                                                              \
        "Rank {} assigned {} = {}",                                        \
        statex_->rank(),                                                   \
        ARGTOSTR(bufName),                                                 \
        regBuf.toString());                                                \
  }

#define SET_REM_REGBUF(bufMngr_, memType, bufName, peerRanks)         \
  {                                                                   \
    /* Omit check since already checked in SET_REGBUF*/               \
    auto& remRegBufVec =                                              \
        ref_.bufs.remRegBufs[(size_t)ResourceBufName::bufName];       \
    auto ret = bufMngr_->assignRemRegBuf(                             \
        memType, ResourceBufName::bufName, peerRanks, remRegBufVec);  \
    FB_CHECKABORT(                                                    \
        ret, "Failed to assign remote regbuf {}", ARGTOSTR(bufName)); \
    for (int i = 0; i < peerRanks.size(); i++) {                      \
      CLOGF_TRACE(                                                    \
          INIT,                                                       \
          "Rank {} assigned {} remRegBufVec[{}] = {}",                \
          statex_->rank(),                                            \
          ARGTOSTR(bufName),                                          \
          i,                                                          \
          remRegBufVec[i].toString());                                \
    }                                                                 \
  }

std::vector<int> prepareExchange(ncclx::CommStateX* statex) {
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nNodes = statex->nNodes();
  const auto myNode = statex->node();
  const auto myLocalRank = statex->localRank();

  // Include intra-node peers and rail peers
  std::vector<int> peers;
  peers.reserve(nLocalRanks + nNodes - 1);
  for (int i = 0; i < nLocalRanks; i++) {
    peers.push_back(statex->localRankToRank(i, myNode));
  }
  for (int n = 0; n < nNodes; n++) {
    // Skip myself since already included in above localRanks
    if (n == myNode) {
      continue;
    }
    peers.push_back(statex->localRankToRank(myLocalRank, n));
  }
  return peers;
}
} // namespace

commResult_t ResourceImpl::setRef(
    const PersistConfig& config,
    cudaStream_t stream) {
  const auto nLocalRanks = statex_->nLocalRanks();
  const auto nNodes = statex_->nNodes();
  const auto myNode = statex_->node();
  const auto myLocalRank = statex_->localRank();

  std::vector<int> localPeers(nLocalRanks);
  std::vector<int> railPeers(nNodes);
  for (int i = 0; i < nLocalRanks; i++) {
    localPeers[i] = statex_->localRankToRank(i, myNode);
  }
  for (int n = 0; n < nNodes; n++) {
    railPeers[n] = statex_->localRankToRank(myLocalRank, n);
  }

  // Assign all local buffers as regBuf to simplify ptr query in algorithm
  SET_REGBUF(bufMngr_, MemType::kDevice, kTmpSendBuff);

  SET_REGBUF(bufMngr_, MemType::kDevice, kTmpSendIdx);
  SET_REGBUF(bufMngr_, MemType::kDevice, kTmpIntraFwdIdx);

  SET_REGBUF(bufMngr_, MemType::kDevice, kTmpFwdBuff);
  SET_REM_REGBUF(bufMngr_, MemType::kDevice, kTmpFwdBuff, railPeers);

  SET_REGBUF(bufMngr_, MemType::kDevice, kTmpRecvBuff);
  SET_REM_REGBUF(bufMngr_, MemType::kDevice, kTmpRecvBuff, localPeers);

  SET_REGBUF(bufMngr_, MemType::kDevice, kLocalOutputSplits);
  SET_REM_REGBUF(bufMngr_, MemType::kDevice, kLocalOutputSplits, localPeers);
  SET_REGBUF(bufMngr_, MemType::kHostPinned, kLocalOutputSplitsH);

  SET_REGBUF(bufMngr_, MemType::kDevice, kRankBitmaps);
  SET_REM_REGBUF(bufMngr_, MemType::kDevice, kRankBitmaps, localPeers);
  SET_REGBUF(bufMngr_, MemType::kHostPinned, kRankBitmapsH);

  SET_REGBUF(bufMngr_, MemType::kHostPinned, kTmpNumSendBlocksBuffH);
  SET_REM_REGBUF(
      bufMngr_, MemType::kHostPinned, kTmpNumSendBlocksBuffH, railPeers);

  SET_REGBUF(bufMngr_, MemType::kDevice, kTmpNumRecvBlocksBuff);
  SET_REM_REGBUF(bufMngr_, MemType::kDevice, kTmpNumRecvBlocksBuff, localPeers);
  SET_REGBUF(bufMngr_, MemType::kHostPinned, kTmpNumRecvBlocksBuffH);

  SET_REGBUF(bufMngr_, MemType::kHostPinned, kBlockRecvBucketsH);
  SET_REM_REGBUF(bufMngr_, MemType::kHostPinned, kBlockRecvBucketsH, railPeers);
  SET_REGBUF(bufMngr_, MemType::kHostPinned, kNumForwardBlocksH);
  SET_REGBUF(bufMngr_, MemType::kDevice, kTmpRecvOffsets);

  SET_REGBUF(bufMngr_, MemType::kHostPinned, kGpeKernelSyncs);

  ResourceBufs& bufs = ref_.bufs;

  // Initialize sync objects used by host
  auto& gkSyncBuff = bufs.getRegBuf(ResourceBufName::kGpeKernelSyncs);

  ref_.sendCopyGKSyncs.resize(nNodes);
  ref_.recvFwdGKSyncs.resize(nNodes);
  ref_.recvCopyGKSyncs.resize(nLocalRanks);

  // - Initialize syncs for sendCopy and recvFwd which needs to be per node, and
  // each sync is watched by numWorkers thread blocks
  for (int n = 0; n < nNodes; n++) {
    ref_.sendCopyGKSyncs[n] =
        reinterpret_cast<GpeKernelSync*>(gkSyncBuff.ptr) + n;
    new (ref_.sendCopyGKSyncs[n]) GpeKernelSync(config.numSendWorkers);
  }
  for (int n = 0; n < nNodes; n++) {
    ref_.recvFwdGKSyncs[n] =
        reinterpret_cast<GpeKernelSync*>(gkSyncBuff.ptr) + nNodes + n;
    new (ref_.recvFwdGKSyncs[n]) GpeKernelSync(config.numFwdWorkers);
  }
  for (int n = 0; n < nLocalRanks; n++) {
    ref_.recvCopyGKSyncs[n] =
        reinterpret_cast<GpeKernelSync*>(gkSyncBuff.ptr) + 2 * nNodes + n;
    new (ref_.recvCopyGKSyncs[n]) GpeKernelSync(config.numRecvWorkers);
  }

  // - Single sync for prepare phase which is watched by all thread blocks
  ref_.prepareGKSync = reinterpret_cast<GpeKernelSync*>(gkSyncBuff.ptr) +
      nNodes * 2 + nLocalRanks;
  new (ref_.prepareGKSync) GpeKernelSync(config.numPrepareThreadBlocks);

  // Initialize sync objects used by device

  // - Copy GK sync objects that are already initialized for host
  ref_.kSync.prepareGKSync = ref_.prepareGKSync;
  ref_.kSync.sendCopyGKSyncs = ref_.sendCopyGKSyncs[0];
  ref_.kSync.recvFwdGKSyncs = ref_.recvFwdGKSyncs[0];
  ref_.kSync.recvCopyGKSyncs = ref_.recvCopyGKSyncs[0];

  // - Initialize sync objects used only by device
  SET_REGBUF(bufMngr_, MemType::kDevice, kFwdGroupSync);
  auto& fwdGroupSyncBuff = bufs.getRegBuf(ResourceBufName::kFwdGroupSync);
  ref_.kSync.fwdGroupSync =
      reinterpret_cast<FwdGroupSync*>(fwdGroupSyncBuff.ptr);

  SET_REGBUF(bufMngr_, MemType::kDevice, kWorkerSync);
  auto& workerSyncBuff = bufs.getRegBuf(ResourceBufName::kWorkerSync);
  ref_.kSync.workerSync = reinterpret_cast<WorkerSync*>(workerSyncBuff.ptr);

  SET_REGBUF(bufMngr_, MemType::kDevice, kFwdRecvSync);
  SET_REM_REGBUF(bufMngr_, MemType::kDevice, kFwdRecvSync, localPeers);
  auto& fwdRecvSyncBuff = bufs.getRegBuf(ResourceBufName::kFwdRecvSync);
  ref_.kSync.fwdRecvSyncs =
      reinterpret_cast<algos::MPSCTbSync<>*>(fwdRecvSyncBuff.ptr);

  // - Initialize fwdRecvSync only once at init
  // - fwdGroupSync will be initialized at each prepare
  std::vector<MPSCTbSync<>> fwdRecvSyncH(
      NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS * nLocalRanks,
      MPSCTbSync(config.numFwdWorkers));
  FB_CUDACHECK(cudaMemcpyAsync(
      fwdRecvSyncBuff.ptr,
      fwdRecvSyncH.data(),
      sizeof(MPSCTbSync<>) * fwdRecvSyncH.size(),
      cudaMemcpyHostToDevice,
      stream));

  auto& remFwdRecvSyncBuffs = bufs.getRemRegBufs(ResourceBufName::kFwdRecvSync);
  FB_CHECKABORT(
      remFwdRecvSyncBuffs.size() <= (size_t)CTRAN_MAX_NVL_PEERS,
      "Unexpected remFwdRecvSyncBuffs.size() {} > CTRAN_MAX_NVL_PEERS {}",
      nLocalRanks,
      CTRAN_MAX_NVL_PEERS);
  for (auto i = 0; i < remFwdRecvSyncBuffs.size(); i++) {
    ref_.kSync.remFwdRecvSyncs[i] =
        reinterpret_cast<algos::MPSCTbSync<>*>(remFwdRecvSyncBuffs[i].ptr);
  }

  return commSuccess;
}

commResult_t ResourceImpl::initialize(
    const PersistArgs& args,
    const PersistConfig& config,
    cudaStream_t stream) {
  const auto nNodes = statex_->nNodes();
  const auto nLocalRanks = statex_->nLocalRanks();
  const auto nRanks = statex_->nRanks();

  SetCudaDevRAII setCudaDev(statex_->cudaDev());

  // Cache all required tmp buffers for allocation preparation

  size_t perRankBufLen = config.tmpChunkSize * config.tmpNumChunks;
  size_t buflen = perRankBufLen * nNodes;

  FB_COMMCHECK(
      bufMngr_->insert(MemType::kDevice, ResourceBufName::kTmpFwdBuff, buflen));
  // same size as tmpFwdBuff
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kDevice, ResourceBufName::kTmpSendBuff, buflen));

  buflen = nNodes * args.totalNumSendBlocks * sizeof(int);
  FB_COMMCHECK(
      bufMngr_->insert(MemType::kDevice, ResourceBufName::kTmpSendIdx, buflen));

  buflen = nNodes * nLocalRanks * args.totalNumSendBlocks * sizeof(int);
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kDevice, ResourceBufName::kTmpIntraFwdIdx, buflen));

  // separate buffer for each local FWD ranks
  buflen = perRankBufLen * nLocalRanks;
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kDevice, ResourceBufName::kTmpRecvBuff, buflen));

  // used for pt metadata
  buflen = nLocalRanks * nLocalRanks * sizeof(int);
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kDevice, ResourceBufName::kLocalOutputSplits, buflen));
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kHostPinned, ResourceBufName::kLocalOutputSplitsH, buflen));
  buflen = nLocalRanks * nLocalRanks * nNodes * args.totalNumSendBlocks *
      sizeof(int);
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kDevice, ResourceBufName::kRankBitmaps, buflen));
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kHostPinned, ResourceBufName::kRankBitmapsH, buflen));

  buflen = nNodes * nNodes *
      (args.numRecvBuckets * nLocalRanks + nLocalRanks + 1) * sizeof(size_t);
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kHostPinned, ResourceBufName::kTmpNumSendBlocksBuffH, buflen));

  buflen = (args.numRecvBuckets * nRanks + nRanks) * sizeof(size_t);
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kDevice, ResourceBufName::kTmpNumRecvBlocksBuff, buflen));

  FB_COMMCHECK(bufMngr_->insert(
      MemType::kHostPinned, ResourceBufName::kTmpNumRecvBlocksBuffH, buflen));

  buflen = sizeof(algos::MPSCTbSync<>) * NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS *
      nLocalRanks;
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kDevice, ResourceBufName::kFwdRecvSync, buflen));

  // Max space needed to host dynamic arguments: blockRecvRanks
  // we do an allgather across nNodes rail peers to compute metadata for
  // external combine
  buflen =
      nNodes * args.totalNumSendBlocks * args.blockNumRecvBuckets * sizeof(int);
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kHostPinned, ResourceBufName::kBlockRecvBucketsH, buflen));

  buflen = nRanks * sizeof(size_t);
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kHostPinned, ResourceBufName::kNumForwardBlocksH, buflen));

  buflen = args.numRecvBuckets * nRanks * sizeof(size_t);
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kDevice, ResourceBufName::kTmpRecvOffsets, buflen));

  // nNodes number of syncs for each of sendCopy and recvFwd, single sync for
  // each of sendMetadata and recvMetadata
  buflen = (nNodes * 2 + nLocalRanks + 2) * sizeof(GpeKernelSync);
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kHostPinned, ResourceBufName::kGpeKernelSyncs, buflen));

  buflen = sizeof(FwdGroupSync);
  FB_COMMCHECK(bufMngr_->insert(
      MemType::kDevice, ResourceBufName::kFwdGroupSync, buflen));

  buflen = sizeof(WorkerSync);
  FB_COMMCHECK(
      bufMngr_->insert(MemType::kDevice, ResourceBufName::kWorkerSync, buflen));

  // allocate memory
  FB_COMMCHECK(bufMngr_->commit());
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "AllToAllvDedup::ResourceImpl: Rank {} commited buffer allocation",
      statex_->rank());

  // register and exchange memory handles
  auto peers = prepareExchange(statex_);
  CLOGF_TRACE(
      INIT,
      "AllToAllvDedup::ResourceImpl: Rank {} exchanged with peers [{}]",
      statex_->rank(),
      folly::join(",", peers));
  FB_COMMCHECK(bufMngr_->exchange(peers, nRanks));

  // assign buffers from allocated memory and keep a const copy of it for later
  // algorithm to use
  FB_COMMCHECK(setRef(config, stream));

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

  bufMngr_ = std::make_unique<::ctran::algos::BufManager<
      ResourceBufName,
      ResourceBufName::kNumBufsNames>>(statex, mapper, logMetadata, memKey);
}
} // namespace ctran::alltoallvdedup
