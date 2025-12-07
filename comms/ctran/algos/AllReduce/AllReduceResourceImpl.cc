// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllReduce/AllReduceResourceImpl.h"
#include <sstream>
#include "comms/ctran/algos/AllReduce/AllReduceDevTypes.h"
#include "comms/ctran/algos/AllReduce/AllReduceNetTypes.h"
#include "comms/ctran/mapper/CtranMapper.h"

#define SET_BUFF(bufMngr_, memType, bufName)                          \
  {                                                                   \
    auto& basicBuf =                                                  \
        ref_.bufs.bufs[(size_t)AllReduceResourceBufName::bufName];    \
    auto ret = bufMngr_->assignBuf(                                   \
        memType, AllReduceResourceBufName::bufName, basicBuf);        \
    FB_CHECKABORT(ret, "Failed to assign buf {}", ARGTOSTR(bufName)); \
    CLOGF_SUBSYS(                                                     \
        INFO,                                                         \
        INIT,                                                         \
        "Rank {} assigned {} = {}",                                   \
        statex_->rank(),                                              \
        ARGTOSTR(bufName),                                            \
        basicBuf.toString());                                         \
  }

#define SET_REGBUF(bufMngr_, memType, bufName)                           \
  {                                                                      \
    auto& regBuf =                                                       \
        ref_.bufs.regBufs[(size_t)AllReduceResourceBufName::bufName];    \
    auto ret = bufMngr_->assignRegBuf(                                   \
        memType, AllReduceResourceBufName::bufName, regBuf);             \
    FB_CHECKABORT(ret, "Failed to assign regbuf {}", ARGTOSTR(bufName)); \
    CLOGF_SUBSYS(                                                        \
        INFO,                                                            \
        INIT,                                                            \
        "Rank {} assigned {} = {}",                                      \
        statex_->rank(),                                                 \
        ARGTOSTR(bufName),                                               \
        regBuf.toString());                                              \
  }

#define SET_REM_REGBUF(bufMngr_, memType, bufName, peerRanks)                 \
  {                                                                           \
    /* Omit check since already checked in SET_REGBUF*/                       \
    auto& remRegBufVec =                                                      \
        ref_.bufs.remRegBufs[(size_t)AllReduceResourceBufName::bufName];      \
    auto ret = bufMngr_->assignRemRegBuf(                                     \
        memType, AllReduceResourceBufName::bufName, peerRanks, remRegBufVec); \
    FB_CHECKABORT(                                                            \
        ret, "Failed to assign remote regbuf {}", ARGTOSTR(bufName));         \
    for (int i = 0; i < peerRanks.size(); i++) {                              \
      CLOGF_SUBSYS(                                                           \
          INFO,                                                               \
          INIT,                                                               \
          "Rank {} assigned {} remRegBufVec[{}] = {}",                        \
          statex_->rank(),                                                    \
          ARGTOSTR(bufName),                                                  \
          i,                                                                  \
          remRegBufVec[i].toString());                                        \
    }                                                                         \
  }

namespace ctran::algos::allreduce {

AllReduceResourceImpl::AllReduceResourceImpl(
    ncclx::CommStateX* statex,
    CtranMapper* mapper,
    CommLogData* logMetadata)
    : statex_(statex), mapper_(mapper), logMetaData_(logMetadata) {
  // memory pool requires unique key for each memory region allocation
  std::stringstream ss;
  ss << "Ctran::AllReduceResource-0x" << std::hex << statex->commHash();
  auto memKey = ss.str();

  bufMngr_ = std::make_unique<::ctran::algos::BufManager<
      AllReduceResourceBufName,
      AllReduceResourceBufName::kNumBufsNames>>(
      statex, mapper, logMetadata, memKey);
}

commResult_t AllReduceResourceImpl::destroy() {
  FB_COMMCHECKIGNORE(bufMngr_->release());
  return commSuccess;
}
AllReduceResourceImpl::~AllReduceResourceImpl() {
  FB_COMMCHECKIGNORE(destroy());
}

commResult_t AllReduceResourceImpl::initAllReduceDirectResourceAsync(
    int nBlocks,
    cudaStream_t stream) {
  const auto nRanks = statex_->nRanks();
  const auto myRank = statex_->rank();
  const auto nLocalRanks = statex_->nLocalRanks();

  std::vector<int> peers(statex_->nRanks());
  for (int i = 0; i < statex_->nRanks(); i++) {
    peers[i] = i;
  }

  size_t buflen = CTRAN_ALLREDUCE_BUFF_SIZE * nBlocks;
  FB_COMMCHECK(bufMngr_->insert(
      ctran::algos::bufmanager::MemType::kDevice,
      AllReduceResourceBufName::kTmpbuff,
      buflen));

  // used for cross-block sync for intra-node
  buflen = sizeof(ctran::algos::MPSCTbSync<1>) * nBlocks * nLocalRanks;
  FB_COMMCHECK(bufMngr_->insert(
      ctran::algos::bufmanager::MemType::kDevice,
      AllReduceResourceBufName::kPostFlags,
      buflen));

  buflen = sizeof(ctran::algos::MPSCTbSync<1>) * nBlocks * nLocalRanks;
  FB_COMMCHECK(bufMngr_->insert(
      ctran::algos::bufmanager::MemType::kDevice,
      AllReduceResourceBufName::kCompleteFlags,
      buflen));

  buflen = nBlocks * statex_->nRanks() * sizeof(AllReduceDevConn*);
  FB_COMMCHECK(bufMngr_->insert(
      ctran::algos::bufmanager::MemType::kDevice,
      AllReduceResourceBufName::kLocalPeerStructures,
      buflen));

  buflen = nBlocks * statex_->nRanks() * sizeof(AllReduceDevConn);
  FB_COMMCHECK(bufMngr_->insert(
      ctran::algos::bufmanager::MemType::kDevice,
      AllReduceResourceBufName::kDevPeers,
      buflen));

  buflen = sizeof(AllReduceComm);
  FB_COMMCHECK(bufMngr_->insert(
      ctran::algos::bufmanager::MemType::kDevice,
      AllReduceResourceBufName::kReduceComm,
      buflen));

  // allocate memory
  FB_COMMCHECK(bufMngr_->commit());
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "AllReduceResourceImpl: Rank {} committed buffer allocation",
      statex_->rank());

  FB_COMMCHECK(bufMngr_->exchange(peers, statex_->nRanks()));
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "AllReduceResourceImpl: Rank {} exchanged with peer {}",
      statex_->rank(),
      folly::join(",", peers));

  // Initialize sync objects used by host
  AllReduceResourceBufs& bufs = ref_.bufs;
  SET_REGBUF(bufMngr_, ctran::algos::bufmanager::MemType::kDevice, kTmpbuff);
  SET_REM_REGBUF(
      bufMngr_, ctran::algos::bufmanager::MemType::kDevice, kTmpbuff, peers);
  auto& tmpbuff = bufs.getRegBuf(AllReduceResourceBufName::kTmpbuff);
  auto& remoteTmpbuff = bufs.getRemRegBufs(AllReduceResourceBufName::kTmpbuff);

  SET_REGBUF(bufMngr_, ctran::algos::bufmanager::MemType::kDevice, kPostFlags);
  SET_REM_REGBUF(
      bufMngr_, ctran::algos::bufmanager::MemType::kDevice, kPostFlags, peers);
  auto& localPostFlags = bufs.getRegBuf(AllReduceResourceBufName::kPostFlags);
  auto& remotePostFlags =
      bufs.getRemRegBufs(AllReduceResourceBufName::kPostFlags);

  SET_REGBUF(
      bufMngr_, ctran::algos::bufmanager::MemType::kDevice, kCompleteFlags);
  SET_REM_REGBUF(
      bufMngr_,
      ctran::algos::bufmanager::MemType::kDevice,
      kCompleteFlags,
      peers);
  auto& localCompleteFlags =
      bufs.getRegBuf(AllReduceResourceBufName::kCompleteFlags);
  auto& remoteCompleteFlags =
      bufs.getRemRegBufs(AllReduceResourceBufName::kCompleteFlags);

  SET_BUFF(bufMngr_, ctran::algos::bufmanager::MemType::kDevice, kDevPeers);
  auto& devPeerBuff = bufs.getBuf(AllReduceResourceBufName::kDevPeers);
  AllReduceDevConn* devPeers =
      reinterpret_cast<AllReduceDevConn*>(devPeerBuff.ptr);

  SET_BUFF(
      bufMngr_,
      ctran::algos::bufmanager::MemType::kDevice,
      kLocalPeerStructures);
  auto& localPeerBuff =
      bufs.getBuf(AllReduceResourceBufName::kLocalPeerStructures);
  AllReduceDevConn** localPeerStructures =
      reinterpret_cast<AllReduceDevConn**>(localPeerBuff.ptr);

  // Host mirror used only during bootstrap patch-up.
  auto commChannels = std::make_unique<AllReduceComm>();

  AllReducePeerHost hostPeer;
  hostPeer.peers.resize(nBlocks);

  for (int i = 0; i < nBlocks; i++) {
    /* host array, one entry per peer */
    std::vector<AllReduceDevConn> devConns(nLocalRanks);
    commChannels->blocks[i].rank = myRank;
    commChannels->blocks[i].nRanks = nRanks;
    for (int peer = 0; peer < nLocalRanks; peer++) {
      if (peer == myRank) {
        devConns[peer].send.step = 0;
        devConns[peer].recv.step = 0;
        continue;
      }

      auto localComplete =
          (ctran::algos::MPSCTbSync<1>*)localCompleteFlags.ptr +
          i * nLocalRanks + peer;
      auto tmpLocalComplete = std::make_unique<ctran::algos::MPSCTbSync<1>>(1);
      FB_CUDACHECK(cudaMemcpyAsync(
          localComplete,
          tmpLocalComplete.get(),
          sizeof(ctran::algos::MPSCTbSync<1>),
          cudaMemcpyHostToDevice,
          stream));

      auto remotePost =
          (ctran::algos::MPSCTbSync<1>*)remotePostFlags[peer].ptr +
          i * nLocalRanks + myRank;
      devConns[peer].send.buff =
          (char*)remoteTmpbuff[peer].ptr + i * CTRAN_ALLREDUCE_BUFF_SIZE;
      devConns[peer].send.complete = localComplete;
      devConns[peer].send.post = remotePost;

      ctran::algos::MPSCTbSync<1>* localPost =
          (ctran::algos::MPSCTbSync<1>*)localPostFlags.ptr + i * nLocalRanks +
          peer;
      auto tmpLocalPost = std::make_unique<ctran::algos::MPSCTbSync<1>>(1);
      FB_CUDACHECK(cudaMemcpyAsync(
          localPost,
          tmpLocalPost.get(),
          sizeof(ctran::algos::MPSCTbSync<1>),
          cudaMemcpyHostToDevice,
          stream));
      ctran::algos::MPSCTbSync<1>* remoteComplete =
          (ctran::algos::MPSCTbSync<1>*)remoteCompleteFlags[peer].ptr +
          i * nLocalRanks + myRank;
      devConns[peer].recv.buff =
          (char*)tmpbuff.ptr + i * CTRAN_ALLREDUCE_BUFF_SIZE;
      devConns[peer].recv.post = localPost;
      devConns[peer].recv.complete = remoteComplete;

      devConns[peer].send.step = 0;
      devConns[peer].send.stepSize =
          CTRAN_ALLREDUCE_BUFF_SIZE / CTRAN_ALLREDUCE_STEPS;
      devConns[peer].recv.step = 0;
      devConns[peer].recv.stepSize =
          CTRAN_ALLREDUCE_BUFF_SIZE / CTRAN_ALLREDUCE_STEPS;
    }
    AllReduceDevConn* currDevPeer = devPeers + i * statex_->nRanks();
    FB_CUDACHECK(cudaMemcpyAsync(
        currDevPeer,
        devConns.data(),
        devConns.size() * sizeof(AllReduceDevConn),
        cudaMemcpyHostToDevice,
        stream));
    std::vector<AllReduceDevConn*> hostPointerArray(nRanks);
    for (auto r = 0; r < nRanks; r++) {
      hostPointerArray[r] = currDevPeer + r;
    }

    AllReduceDevConn** currPeer = localPeerStructures + i * statex_->nRanks();

    FB_CUDACHECK(cudaMemcpyAsync(
        currPeer,
        hostPointerArray.data(),
        hostPointerArray.size() * sizeof(AllReduceDevConn*),
        cudaMemcpyHostToDevice,
        stream));
    commChannels->blocks[i].peers = currPeer;
  }

  SET_BUFF(bufMngr_, ctran::algos::bufmanager::MemType::kDevice, kReduceComm);
  auto& reduceCommBuf = bufs.getBuf(AllReduceResourceBufName::kReduceComm);
  ref_.allReduceComms = reinterpret_cast<AllReduceComm*>(reduceCommBuf.ptr);

  // Copy the CtranDevChannel structure directly to the channel field
  FB_CUDACHECK(cudaMemcpyAsync(
      ref_.allReduceComms,
      commChannels.get(),
      sizeof(AllReduceComm),
      cudaMemcpyHostToDevice,
      stream));

  return commSuccess;
}
} // namespace ctran::algos::allreduce
