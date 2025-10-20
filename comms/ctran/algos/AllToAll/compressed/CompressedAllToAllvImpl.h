// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef ENABLE_META_COMPRESSION
#pragma once

#include <folly/synchronization/CallOnce.h>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/CtranAux.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"

// Set to 16MB for, needed due to the max chunk size supported by nvcomp is
// 16MB, will update it in the future
constexpr size_t kMaxChunkSize = 1 << 24;

static inline size_t getCompChunkSize() {
  if (NCCL_CTRAN_COMPRESSED_ALLTOALLV_CHUNK_SIZE > kMaxChunkSize) {
    CLOGF(
        WARN,
        "Invalid NCCL_CTRAN_COMPRESSED_ALLTOALLV_CHUNK_SIZE value: {}. Defaulting to {} bytes.",
        NCCL_CTRAN_COMPRESSED_ALLTOALLV_CHUNK_SIZE,
        kMaxChunkSize);
    return kMaxChunkSize;
  }
  return NCCL_CTRAN_COMPRESSED_ALLTOALLV_CHUNK_SIZE;
}

#define PRUNE_COMP_SEND_BYTES(ibSendBytes, nRanks)     \
  do {                                                 \
    for (int i = 0; i < nRanks; i++) {                 \
      if (ibSendBytes[i] == 0) {                       \
        compression::CompressionManager::getInstance() \
            ->getHostCompSendBytes()[i] = 0;           \
      }                                                \
    }                                                  \
  } while (0)

static inline commResult_t compressedAllToAllvBootstrap(CtranComm* comm) {
  const auto& statex = comm->statex_;
  const int nRanks = statex->nRanks();
  const int myRank = statex->rank();
  const int myNode = statex->node();
  auto compressionManager = compression::CompressionManager::getInstance();

  for (int i = 0; i < nRanks; i++) {
    // Create unique tags for each pair of sender and receiver
    int sendTag = myRank * 1000 + i; // Unique tag for sending to rank i
    int recvTag = i * 1000 + myRank; // Unique tag for receiving from rank i
    // Send sendcounts to rank i
    const int peerNode = statex->node(i);
    if (myNode != peerNode) {
      auto resFuture = comm->bootstrap_->send(
          &compressionManager->getHostCompSendBytes()[i],
          sizeof(size_t),
          i,
          sendTag);
      FB_COMMCHECKTHROW(static_cast<commResult_t>(std::move(resFuture).get()));

      // Receive sendcounts from rank i into recvcounts
      resFuture = comm->bootstrap_->recv(
          &compressionManager->getHostCompRecvBytes()[i],
          sizeof(size_t),
          i,
          recvTag);
      FB_COMMCHECKTHROW(static_cast<commResult_t>(std::move(resFuture).get()));
    } else {
      // Set recvBytes from local peers always to 0
      compressionManager->getHostCompRecvBytes()[i] = 0;
    }
  }

  // Update comp rdispls
  size_t offset = 0;
  for (int i = 0; i < nRanks; i++) {
    size_t compRecvBytes = compressionManager->getHostCompRecvBytes()[i];
    compressionManager->getCompRdisplsBytes().at(i) = offset;
    offset += compRecvBytes;
  }

  return commSuccess;
}

// This function has similar function to the original
// compressedAllToAllvBootstrap the only difference is that it will not set the
// compressed buffer size to other IB peers through bootstrap. This is used to
// avoid the cost of performing blocking NCCL socket comms. Instead it assumes a
// fixed compressed buffer size for all peers, at of cost of having internal
// memory fragmentation in exchange for speed.
static inline commResult_t compressedAllToAllvFastRdisplsUpdate(
    CtranComm* comm) {
  const auto& statex = comm->statex_;
  const int nRanks = statex->nRanks();
  auto compressionManager = compression::CompressionManager::getInstance();

  // Update comp rdispls, it will use equal-sized chunks, regardless whether
  // an IB peer will send 0 bytes of data.
  size_t offset = 0;
  for (int i = 0; i < nRanks; i++) {
    compressionManager->getCompRdisplsBytes().at(i) = offset;
    offset += getCompChunkSize();
  }

  return commSuccess;
}

template <typename PerfConfig = DefaultPerfCollConfig>
commResult_t ctranCompressedAllToAllvBootstrapIbImpl(
    const void* sendbuff,
    std::vector<size_t>& sendCounts,
    std::vector<size_t>& sDispls,
    void* recvbuff,
    std::vector<size_t>& recvCounts,
    std::vector<size_t>& rDispls,
    commDataType_t datatype,
    CtranComm* comm,
    std::unique_ptr<CtranMapperTimestamp> timestamp) {
  const auto& statex = comm->statex_;
  const int myRank = statex->rank();
  const int nRanks = statex->nRanks();

  static const auto myAlgo = NCCL_ALLTOALLV_ALGO::bsCompCtran;
  const std::string algoName = allToAllvAlgoName(myAlgo);
  const bool useProfiler = NCCL_CTRAN_PROFILING != NCCL_CTRAN_PROFILING::none;

  std::vector<const void*> sendBuffs(nRanks);
  std::vector<void*> recvBuffs(nRanks);
  std::vector<void*> remoteRecvBuffs(nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(nRanks);

  std::vector<void*> sendMemHdl(nRanks);
  std::vector<void*> recvMemHdl(nRanks);
  std::vector<void*> tmpRegHdls;

  std::vector<int> ibRecvPeers, ibSendPeers;
  std::unordered_set<int> ibPeers;

  auto compressionManager = compression::CompressionManager::getInstance();

  // By default, the compressed buf might not be empty even if the input buf
  // size is 0, so we need to set the getHostCompSendBytes to 0 if the input buf
  // size is 0, or when the peer is in the same node.
  PRUNE_COMP_SEND_BYTES(sendCounts, nRanks);

  // Perform boostrapping
  commResult_t status = compressedAllToAllvBootstrap(comm);
  if (status != commSuccess) {
    CLOGF(WARN, "compressedAllToAllvBootstrap failed with ERROR: {}", status);
    return commSystemError;
  }

  // Save send/recv bytes
  std::vector<size_t> compSendBytes(nRanks, 0);
  std::vector<size_t> compRecvBytes(nRanks, 0);
  for (int i = 0; i < nRanks; i++) {
    compSendBytes[i] = compressionManager->getHostCompSendBytes()[i];
    compRecvBytes[i] = compressionManager->getHostCompRecvBytes()[i];
  }

  if (compSendBytes.size() > 0) {
    std::vector<size_t> sendSizes(nRanks, 0);
    for (int i = 0; i < nRanks; i++) {
      sendSizes[i] = compSendBytes[i];
    }
    std::vector<size_t> recvSizes(nRanks, 0);
    for (int i = 0; i < nRanks; i++) {
      recvSizes[i] = compRecvBytes[i];
    }
    CtranMapperContext context(algoName, sendSizes, recvSizes);
    comm->ctran_->mapper->setContext(std::move(context));
  }

  // Prepare buffers shifted with displacement, and set ctrl/put/notify
  // schedules. Try to schedule ctrl message and put sequence as rank i start
  // sending to rank i+1 to avoid congestion in potential all-to-one case.
  // Specified in putPeers, sendCtrlPeers.
  size_t contigSendBufSize = 0;
  size_t contigRecvBufSize = 0;
  uint8_t** compSendBuf = compressionManager->getHostCompPtrs();
  uint8_t* compRecvBuf = compressionManager->getCompRecvBuff();
  for (int i = 0; i < nRanks; i++) {
    int peer = (myRank + i) % nRanks;
    if (compSendBytes[peer]) {
      sendBuffs[peer] = reinterpret_cast<const char*>(compSendBuf[peer]);
      ibSendPeers.push_back(peer);
      ibPeers.insert(peer);
      contigSendBufSize = std::max(
          contigSendBufSize,
          compressionManager->getCompSdisplsBytes().at(peer) +
              compSendBytes[peer]);
    }
    if (compRecvBytes[peer]) {
      recvBuffs[peer] = reinterpret_cast<char*>(compRecvBuf) +
          compressionManager->getCompRdisplsBytes().at(peer);
      ibRecvPeers.push_back(peer);
      ibPeers.insert(peer);
      contigRecvBufSize = std::max(
          contigRecvBufSize,
          compressionManager->getCompRdisplsBytes().at(peer) +
              compRecvBytes[peer]);
    }
  }

  // pre-connect all peers
  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    comm->ctran_->mapper->preConnect(ibPeers);
  }

  // schedule IB ctrl messages
  std::vector<CtranMapperRequest> ibSendCtrlReqs(ibRecvPeers.size()),
      ibRecvCtrlReqs(ibSendPeers.size());
  std::vector<CtranMapperNotify> notifyVec(ibRecvPeers.size());

  void* tmpHdl = nullptr;
  // Search for the handle only when there are RecvPeers to avoid attempting to
  // search/register with a buffer size of 0.
  if (!ibRecvPeers.empty()) {
    FB_COMMCHECK(searchRegHandle(
        comm,
        compressionManager->getCompRecvBuff(),
        contigRecvBufSize,
        tmpHdl,
        tmpRegHdls));
  }

  FB_COMMCHECK(comm->ctran_->mapper->isendCtrlBatch<PerfConfig>(
      recvBuffs, tmpHdl, ibRecvPeers, ibSendCtrlReqs, CtranMapperBackend::IB));
  FB_COMMCHECK(comm->ctran_->mapper->initNotifyBatchIB(ibRecvPeers, notifyVec));

  tmpHdl = nullptr;
  // Search for the handle only when there are SendPeers to avoid attempting to
  // search/register with a buffer size of 0.
  if (!ibSendPeers.empty()) {
    FB_COMMCHECK(searchRegHandle(
        comm, compSendBuf[0], contigSendBufSize, tmpHdl, tmpRegHdls));
  }
  int idx = 0;
  for (auto peer : ibSendPeers) {
    sendMemHdl[peer] = tmpHdl;
    FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl<PerfConfig>(
        &remoteRecvBuffs[peer],
        &remoteAccessKeys[peer],
        peer,
        &ibRecvCtrlReqs[idx++]));
  }

  static auto alltoallvIbConfig =
      comm->ctran_->algo->getCollToVcConfig(CollType::ALLTOALL);

  // issue network puts:
  // - Sender puts data for peers, whenever received the remote recvbuff handle
  // - Exit until all peers' put have been issued (putPeers becomes empty)
  std::vector<CtranMapperRequest> ibPutReqs(ibSendPeers.size());
  idx = 0;
  for (auto& recvCtrlReq : ibRecvCtrlReqs) {
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest<PerfConfig>(&recvCtrlReq));
    const int peer = recvCtrlReq.peer;
    FB_COMMCHECK(comm->ctran_->mapper->iput<PerfConfig>(
        sendBuffs[peer],
        remoteRecvBuffs[peer],
        compSendBytes[peer],
        peer,
        CtranMapperConfig{
            .memHdl_ = sendMemHdl[peer],
            .remoteAccessKey_ = remoteAccessKeys[peer],
            .notify_ = true, /*notify*/
            .ibConfig_ = alltoallvIbConfig},
        &ibPutReqs[idx++]));
    if (useProfiler) {
      timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));
    }
  }

  // Wait for all puts to complete
  FB_COMMCHECK(comm->ctran_->mapper->waitAllRequests<PerfConfig>(
      ibPutReqs, useProfiler ? (&timestamp->putComplete) : nullptr));
  // Wait for all receives (i.e., remote IB puts) to complete
  FB_COMMCHECK(comm->ctran_->mapper->waitAllNotifies<PerfConfig>(notifyVec));

  // Always wait for all sendCtrlReqs to complete so that the memory can be
  // safely reused in next collective; otherwise, ibvc may complete the previous
  // request while the memory has already been assigned to a new request.
  FB_COMMCHECK(
      comm->ctran_->mapper->waitAllRequests<PerfConfig>(ibSendCtrlReqs));

  if (useProfiler) {
    comm->ctran_->mapper->timestamps.emplace_back(std::move(timestamp));
    comm->ctran_->mapper->reportProfiling();
  }

  /* deregister temporary registrations */
  // FIXME: let GPE kernel to finish then deregister to avoid race condition on
  // cuda context
  for (auto& hdl : tmpRegHdls) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(hdl));
  }

  return commSuccess;
}

// This function has similar function to ctranCompressedAllToAllvIbImpl.
// The only difference is that it will not perform boostrapping to exchange
// compressed buffer sizes. Instead it assumes a fixed compressed buffer size
// for all peers, at of cost of having internal memory fragmentation in exchange
template <typename PerfConfig = DefaultPerfCollConfig>
commResult_t ctranCompressedAllToAllvIbImpl(
    const void* sendbuff,
    std::vector<size_t>& sendCounts,
    std::vector<size_t>& sDispls,
    void* recvbuff,
    std::vector<size_t>& recvCounts,
    std::vector<size_t>& rDispls,
    commDataType_t datatype,
    CtranComm* comm,
    std::unique_ptr<CtranMapperTimestamp> timestamp) {
  const auto& statex = comm->statex_;
  const int myRank = statex->rank();
  const int nRanks = statex->nRanks();
  const int myNode = statex->node();

  static const auto myAlgo = NCCL_ALLTOALLV_ALGO::compCtran;
  const std::string algoName = allToAllvAlgoName(myAlgo);
  const bool useProfiler = NCCL_CTRAN_PROFILING != NCCL_CTRAN_PROFILING::none;

  std::vector<const void*> sendBuffs(nRanks);
  std::vector<void*> recvBuffs(nRanks);
  std::vector<void*> remoteRecvBuffs(nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(nRanks);

  std::vector<void*> sendMemHdl(nRanks);
  std::vector<void*> recvMemHdl(nRanks);
  std::vector<void*> tmpRegHdls;

  std::vector<int> ibRecvPeers, ibSendPeers;
  std::unordered_set<int> ibPeers;

  auto compressionManager = compression::CompressionManager::getInstance();

  // By default, the compressed buf might not be empty even if the input buf
  // size is 0, so we need to set the compSendBytes to 0 if the input buf size
  // is 0, or when the peer is in the same node.
  PRUNE_COMP_SEND_BYTES(sendCounts, nRanks);

  // Perform boostrapping
  commResult_t status = compressedAllToAllvFastRdisplsUpdate(comm);
  if (status != commSuccess) {
    CLOGF(
        WARN,
        "compressedAllToAllvFastRdisplsUpdate failed with EROR {}",
        status);
    return commSystemError;
  }

  // Save send/recv bytes
  std::vector<size_t> compSendBytes(nRanks, 0);
  for (int i = 0; i < nRanks; i++) {
    compSendBytes[i] = compressionManager->getHostCompSendBytes()[i];
  }

  if (compSendBytes.size() > 0) {
    std::vector<size_t> sendSizes(nRanks, 0);
    for (int i = 0; i < nRanks; i++) {
      sendSizes[i] = compSendBytes[i];
    }

    // This buffer is empty at this moment because isendCtrl/Recv
    // hasn't been issued yet.
    std::vector<size_t> recvSizes(nRanks, 0);

    CtranMapperContext context(algoName, sendSizes, recvSizes);
    comm->ctran_->mapper->setContext(std::move(context));
  }

  // Prepare buffers shifted with displacement, and set ctrl/put/notify
  // schedules. Try to schedule ctrl message and put sequence as rank i start
  // sending to rank i+1 to avoid congestion in potential all-to-one case.
  // Specified in putPeers, sendCtrlPeers.
  size_t contigSendBufSize = 0;
  size_t contigRecvBufSize = 0;
  uint8_t** compSendBuf = compressionManager->getHostCompPtrs();
  uint8_t* compRecvBuf = compressionManager->getCompRecvBuff();
  for (int i = 0; i < nRanks; i++) {
    // only use remote
    int peer = (myRank + i) % nRanks;
    const int peerNode = statex->node(peer);
    if (peerNode != myNode) {
      // configure send
      {
        sendBuffs[peer] = reinterpret_cast<const char*>(compSendBuf[peer]);
        ibSendPeers.push_back(peer);
        ibPeers.insert(peer);
        contigSendBufSize = std::max(
            contigSendBufSize,
            compressionManager->getCompSdisplsBytes()[peer] +
                compSendBytes[peer]);
      }

      // configure recv, using constant chunk size
      {
        recvBuffs[peer] = reinterpret_cast<char*>(compRecvBuf) +
            compressionManager->getCompRdisplsBytes()[peer];
        ibRecvPeers.push_back(peer);
        ibPeers.insert(peer);
        contigRecvBufSize = std::max(
            contigRecvBufSize,
            compressionManager->getCompRdisplsBytes()[peer] +
                getCompChunkSize());
      }
    }
  }

  std::vector<AuxData_t<DefaultAuxType>> auxSendCompBufSizes(nRanks, 0);
  {
    for (int i = 0; i < nRanks; i++) {
      auxSendCompBufSizes[i] =
          AuxData_t(compressionManager->getHostCompSendBytes()[i]);
    }
  }

  // pre-connect all peers
  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    comm->ctran_->mapper->preConnect(ibPeers);
  }

  // schedule IB ctrl messages
  std::vector<CtranMapperRequest> ibSendCtrlReqs(ibRecvPeers.size()),
      ibRecvCtrlReqs(ibSendPeers.size());
  std::vector<CtranMapperNotify> notifyVec(ibRecvPeers.size());

  void* tmpHdl = nullptr;
  // Search for the handle only when there are RecvPeers to avoid attempting to
  // search/register with a buffer size of 0.
  if (!ibRecvPeers.empty()) {
    FB_COMMCHECK(searchRegHandle(
        comm,
        compressionManager->getCompRecvBuff(),
        contigRecvBufSize,
        tmpHdl,
        tmpRegHdls));
  }

  // Assign auxSendCompBufSizes to ibSendCtrlReqs
  {
    if (ibSendCtrlReqs.size() != ibRecvPeers.size()) {
      CLOGF(
          WARN,
          "compressedAllToAllv failed because ibSendCtrlReqs.size() != ibRecvPeers.size()");
      return commSystemError;
    }

    for (int i = 0; i < ibSendCtrlReqs.size(); i++) {
      const int peer = ibRecvPeers[i];
      ibSendCtrlReqs[i].aux = auxSendCompBufSizes[peer];
    }
  }

  FB_COMMCHECK(comm->ctran_->mapper->isendCtrlBatch<PerfConfig>(
      recvBuffs, tmpHdl, ibRecvPeers, ibSendCtrlReqs, CtranMapperBackend::IB));
  FB_COMMCHECK(comm->ctran_->mapper->initNotifyBatchIB(ibRecvPeers, notifyVec));

  tmpHdl = nullptr;
  // Search for the handle only when there are SendPeers to avoid attempting to
  // search/register with a buffer size of 0.
  if (!ibSendPeers.empty()) {
    FB_COMMCHECK(searchRegHandle(
        comm, compSendBuf[0], contigSendBufSize, tmpHdl, tmpRegHdls));
  }
  std::vector<AuxData_t<DefaultAuxType>> auxRecvCompBufSizes(
      nRanks, AuxData_t(static_cast<uint64_t>(0)));

  int idx = 0;
  for (auto peer : ibSendPeers) {
    sendMemHdl[peer] = tmpHdl;
    FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl<PerfConfig>(
        &remoteRecvBuffs[peer],
        &remoteAccessKeys[peer],
        peer,
        &ibRecvCtrlReqs[idx++]));
  }

  static auto alltoallvIbConfig =
      comm->ctran_->algo->getCollToVcConfig(CollType::ALLTOALL);

  // issue network puts:
  // - Sender puts data for peers, whenever received the remote recvbuff handle
  // - Exit until all peers' put have been issued (putPeers becomes empty)
  std::vector<CtranMapperRequest> ibPutReqs(ibSendPeers.size());
  idx = 0;
  for (auto& recvCtrlReq : ibRecvCtrlReqs) {
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest<PerfConfig>(&recvCtrlReq));

    const int peer = recvCtrlReq.peer;
    auxRecvCompBufSizes[peer] = recvCtrlReq.recvCtrl.msg.aux;

    FB_COMMCHECK(comm->ctran_->mapper->iput<PerfConfig>(
        sendBuffs[peer],
        remoteRecvBuffs[peer],
        compSendBytes[peer],
        peer,
        CtranMapperConfig{
            .memHdl_ = sendMemHdl[peer],
            .remoteAccessKey_ = remoteAccessKeys[peer],
            .notify_ = true, /*notify*/
            .ibConfig_ = alltoallvIbConfig},
        &ibPutReqs[idx++]));
    if (useProfiler) {
      timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));
    }
  }

  // Wait for all puts to complete
  FB_COMMCHECK(comm->ctran_->mapper->waitAllRequests<PerfConfig>(
      ibPutReqs, useProfiler ? (&timestamp->putComplete) : nullptr));
  // Wait for all receives (i.e., remote IB puts) to complete
  FB_COMMCHECK(comm->ctran_->mapper->waitAllNotifies<PerfConfig>(notifyVec));

  // Always wait for all sendCtrlReqs to complete so that the memory can be
  // safely reused in next collective; otherwise, ibvc may complete the previous
  // request while the memory has already been assigned to a new request.
  FB_COMMCHECK(
      comm->ctran_->mapper->waitAllRequests<PerfConfig>(ibSendCtrlReqs));

  if (useProfiler) {
    comm->ctran_->mapper->timestamps.emplace_back(std::move(timestamp));
    comm->ctran_->mapper->reportProfiling();
  }

  // Check if the received compressed buffer size exceeds the maximum
  // compressed chunk size.
  {
    for (int i = 0; i < nRanks; i++) {
      if (auxRecvCompBufSizes[i].data > getCompChunkSize()) {
        CLOGF(
            WARN,
            "compressedAllToAllv failed because COMPRESSION_CHUNKSIZE is too big, consider change COMPRESSION_CHUNKSIZE");
        return commSystemError;
      }
    }
  }

  // Assign the received compressed buffer size to the corresponding rank.
  {
    for (int i = 0; i < nRanks; i++) {
      const int peerNode = statex->node(i);
      if (peerNode != myNode) {
        compressionManager->getHostCompRecvBytes()[i] =
            auxRecvCompBufSizes[i].data;
      } else {
        compressionManager->getHostCompRecvBytes()[i] = 0;
      }
    }
  }

  /* deregister temporary registrations */
  // FIXME: let GPE kernel to finish then deregister to avoid race condition on
  // cuda context
  for (auto& hdl : tmpRegHdls) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(hdl));
  }

  return commSuccess;
}
#endif // ENABLE_META_COMPRESSION
