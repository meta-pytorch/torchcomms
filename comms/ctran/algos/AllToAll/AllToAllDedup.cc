// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_fp16.h>

#include "comms/ctran/algos/AllToAll/AllToAllDedupImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/utils/Alloc.h"

#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

namespace {
void* alltoalldedupKerns[commNumTypes] = {
    (void*)ncclKernelAllToAllDedup<int8_t>,
    (void*)ncclKernelAllToAllDedup<uint8_t>,
    (void*)ncclKernelAllToAllDedup<int32_t>,
    (void*)ncclKernelAllToAllDedup<uint32_t>,
    (void*)ncclKernelAllToAllDedup<int64_t>,
    (void*)ncclKernelAllToAllDedup<uint64_t>,
    (void*)ncclKernelAllToAllDedup<half>,
    (void*)ncclKernelAllToAllDedup<float>,
    (void*)ncclKernelAllToAllDedup<double>,
#if defined(__CUDA_BF16_TYPES_EXIST__)
    (void*)ncclKernelAllToAllDedup<__nv_bfloat16>,
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    (void*)ncclKernelAllToAllDedup<__nv_fp8_e4m3>,
    (void*)ncclKernelAllToAllDedup<__nv_fp8_e5m2>,
#endif
};

inline commResult_t fwdCtrlExchange(CtranPersistentRequest* request) {
  auto comm = request->comm_;
  const auto myRank = comm->statex_->rank();
  const auto nRanks = comm->statex_->nRanks();
  auto recvbuff = request->op->alltoall_dedup.recvbuff;
  auto recvMemHdl = request->op->alltoall_dedup.recvHdl;
  auto& remoteRecvBuffs = request->op->alltoall_dedup.remoteRecvBuffs;
  auto& remoteAccessKeys = request->op->alltoall_dedup.remoteAccessKeys;

  std::vector<std::unique_ptr<CtranMapperRequest>> fwdSendCtrlReqs,
      fwdRecvCtrlReqs;
  std::vector<int> fwdRecvPeers, fwdSendPeers;

  CtranMapperEpochRAII epochRAII(comm->ctran_->mapper.get());

  fwdSendCtrlReqs.reserve(fwdRecvPeers.size());
  fwdRecvCtrlReqs.reserve(fwdSendPeers.size());

  for (int p = 1; p < nRanks; p++) {
    const int peer = (myRank + p) % nRanks;
    if (comm->statex_->isSameNode(myRank, peer) && peer != myRank) {
      CtranMapperRequest* req = nullptr;
      FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl(
          &remoteRecvBuffs[peer], &remoteAccessKeys[peer], peer, &req));
      fwdRecvCtrlReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));

      req = nullptr;
      FB_COMMCHECK(
          comm->ctran_->mapper->isendCtrl(recvbuff, recvMemHdl, peer, &req));
      fwdSendCtrlReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
    }
  }

  for (auto& req : fwdSendCtrlReqs) {
    comm->ctran_->mapper->waitRequest(req.get());
  }
  for (auto& req : fwdRecvCtrlReqs) {
    comm->ctran_->mapper->waitRequest(req.get());
  }
  return commSuccess;
}

inline commResult_t setupKernelConfig(
    const void* const sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    commDataType_t datatype,
    CtranComm* comm,
    CtranPersistentRequest* const request,
    KernelConfig& config) {
  // TODO: just pass in recvbuffs as kernel arg and write to pinned memory
  // instead of using kernel elem post
  std::vector<int> fwdPeers;
  const auto statex = comm->statex_.get();
  const auto numIbPeers = statex->nNodes();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto localRank = statex->localRank();
  const auto node = statex->node();

  config.numThreads = NCCL_CTRAN_ALLTOALL_DEDUP_THREAD_BLOCK_SIZE;
  int numBlocks =
      NCCL_CTRAN_ALLTOALL_DEDUP_NUM_THREAD_BLOCKS_PER_RAIL_P2P * numIbPeers;
  config.numBlocks = std::min(numBlocks, CTRAN_ALGO_MAX_THREAD_BLOCKS);

  config.args.devState_d = comm->ctran_->algo->getDevState();

  for (int i = 1; i < nLocalRanks; i++) {
    int peer = statex->localRankToRank((localRank + i) % nLocalRanks, node);
    fwdPeers.push_back(peer);
  }

  auto& remoteRecvBuffs = request->op->alltoall_dedup.remoteRecvBuffs;
  KernelElem* bcastElemList = nullptr;
  // alloc kernel elems for each exec since opElem destructor frees kernel elems
  // after each exec
  FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(
      numIbPeers,
      NCCL_CTRAN_ALLTOALL_DEDUP_NUM_THREAD_BLOCKS_PER_RAIL_P2P,
      &bcastElemList));
  // each kernel elem is in charge of one broadcast, set information for the
  // kernel via kernel elem
  KernelElem* curElem = bcastElemList;
  int curNode = statex->node();
  for (int i = 0; i < numIbPeers; i++) {
    // after network puts of splits, kernel broadcasts the split associated with
    // its local rank to its local peers
    if (i == curNode) {
      size_t disp = sdispls[i] * commTypeSize(datatype);
      curElem->bcast.count = sendcounts[i];
      curElem->bcast.src = (void*)((char*)sendbuff + disp);
    } else {
      size_t disp = rdispls[i] * commTypeSize(datatype);
      curElem->bcast.count = recvcounts[i];
      curElem->bcast.src = (void*)((char*)recvbuff + disp);
    }

    size_t displ = rdispls[i] * commTypeSize(datatype);
    curElem->bcast.dsts[localRank] = (char*)recvbuff + displ;
    for (auto fwdPeer : fwdPeers) {
      int p = fwdPeer % nLocalRanks;
      curElem->bcast.dsts[p] = (char*)remoteRecvBuffs[fwdPeer] + displ;
    }

    request->op->alltoall_dedup.bcastElemMap[i] = curElem;

    curElem = curElem->next;
  }
  config.args.collective.alltoall_dedup.bcastElemList = bcastElemList;
  config.args.collective.alltoall_dedup.numIbPeers = numIbPeers;

  return commSuccess;
}

inline commResult_t setupGpeOp(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    commDataType_t datatype,
    void* sendMemHdl,
    void* regHdl,
    CtranComm* comm,
    uint64_t opCount,
    std::unique_ptr<struct OpElem>& op) {
  const auto statex = comm->statex_.get();
  if (statex->nLocalRanks() < statex->nRanks()) {
    op = std::unique_ptr<struct OpElem>(
        new OpElem(OpElem::opType::ALLTOALL_DEDUP, comm, opCount));
    op->alltoall_dedup.sendbuff = sendbuff;
    op->alltoall_dedup.recvbuff = recvbuff;
    op->alltoall_dedup.datatype = datatype;
    op->alltoall_dedup.sendcounts = sendcounts;
    op->alltoall_dedup.sdispls = sdispls;
    op->alltoall_dedup.recvcounts = recvcounts;
    op->alltoall_dedup.rdispls = rdispls;
    op->alltoall_dedup.sendHdl = sendMemHdl;
    op->alltoall_dedup.recvHdl = regHdl;
  }
  return commSuccess;
}

inline commResult_t searchRegHandle(
    CtranComm*& comm,
    const void* buff,
    size_t bytes,
    void*& hdl,
    std::vector<void*>& tmpRegHdls) {
  bool localReg = false;
  FB_COMMCHECK(
      comm->ctran_->mapper->searchRegHandle(buff, bytes, &hdl, &localReg));
  if (localReg) {
    tmpRegHdls.push_back(hdl);
  }
  return commSuccess;
}

const std::string kAllToAllDedupAlgoName{"ctranAllToAllDedup"};
} // namespace

commResult_t ctranAllToAllDedupInit(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    const size_t maxSendCount,
    void*& recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    const size_t maxRecvCount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request) {
  if (maxSendCount == 0 || sendbuff == nullptr) {
    return commInvalidArgument;
  }

  void* sendMemHdl = nullptr;
  std::vector<void*> tmpRegHdls;

  SetCudaDevRAII setCudaDev(comm->statex_->cudaDev());
  // recvbuff is internally managed nccl memory and must be freed with
  // ctranAllToAllDedupDestroy
  size_t size = maxRecvCount * commTypeSize(datatype);
  FB_COMMCHECK(ctran::utils::commCudaMalloc(
      (char**)&recvbuff, size, &comm->logMetaData_, "AllToAllDedupInit"));

  void* segHdl{nullptr};
  void* regHdl{nullptr};
  FB_COMMCHECK(comm->ctran_->mapper->regMem(
      recvbuff,
      size,
      &segHdl,
      true, /* force registration */
      true, /* NCCL managed buffer */
      &regHdl));

  FB_COMMCHECK(searchRegHandle(
      comm,
      sendbuff,
      maxSendCount * commTypeSize(datatype),
      sendMemHdl,
      tmpRegHdls));

  std::unique_ptr<struct OpElem> op;
  FB_COMMCHECK(setupGpeOp(
      sendbuff,
      sendcounts,
      sdispls,
      recvbuff,
      recvcounts,
      rdispls,
      datatype,
      sendMemHdl,
      regHdl,
      comm,
      1,
      op));

  request = new CtranPersistentRequest(
      CtranPersistentRequest::Type::ALLTOALL_DEDUP,
      comm,
      std::move(op),
      stream,
      segHdl);
  FB_COMMCHECK(fwdCtrlExchange(request));
  return commSuccess;
}

commResult_t ctranAllToAllDedupExec(CtranPersistentRequest* request) {
  auto comm = request->comm_;
  auto stream = request->stream;
  const auto sendbuff = request->op->alltoall_dedup.sendbuff;
  auto recvbuff = request->op->alltoall_dedup.recvbuff;
  auto datatype = request->op->alltoall_dedup.datatype;
  auto sendcounts = request->op->alltoall_dedup.sendcounts;
  auto sdispls = request->op->alltoall_dedup.sdispls;
  auto recvcounts = request->op->alltoall_dedup.recvcounts;
  auto rdispls = request->op->alltoall_dedup.rdispls;
  auto opCount = comm->ctran_->getOpCount();
  const auto statex = comm->statex_.get();

  CTRAN_COLL_INFO(
      kAllToAllDedupAlgoName.c_str(),
      sendbuff,
      recvbuff,
      0UL,
      datatype,
      -1,
      comm,
      stream);

  for (int i = 0; i < statex->nNodes(); i++) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "{}: opCount {} - sendcounts[{}] {} sdispls[{}] {} recvcounts[{}] {} rdispls[{}] {}",
        kAllToAllDedupAlgoName,
        opCount,
        i,
        sendcounts[i],
        i,
        sdispls[i],
        i,
        recvcounts[i],
        i,
        rdispls[i]);
  }

  // prepare kernel config for NVL copies
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALL_DEDUP,
      stream,
      kAllToAllDedupAlgoName,
      opCount);
  FB_COMMCHECK(setupKernelConfig(
      sendbuff,
      sendcounts,
      sdispls,
      recvbuff,
      recvcounts,
      rdispls,
      datatype,
      comm,
      request,
      config));

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  std::unique_ptr<struct OpElem> op(new OpElem(request->op.get()));
  opGroup.push_back(std::move(op));

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      ctranAllToAllDedupExecImpl,
      config,
      reinterpret_cast<void*>(alltoalldedupKerns[datatype])));

  return commSuccess;
}

commResult_t ctranAllToAllDedupDestroy(CtranPersistentRequest* request) {
  auto comm = request->comm_;
  FB_COMMCHECK(comm->ctran_->mapper->deregMem(
      request->segHdl, true /* skipRemRelease */));
  FB_COMMCHECK(
      ctran::utils::commCudaFree((char*)request->op->alltoall_dedup.recvbuff));
  // Explicitly release remote registrations so that the memory can be back to
  // user rather than waiting till communicator destruction
  for (auto& rkey : request->op->alltoall_dedup.remoteAccessKeys) {
    FB_COMMCHECK(comm->ctran_->mapper->deregRemReg(&rkey));
  }
  request->op->alltoall_dedup.remoteAccessKeys.~vector();
  request->op->alltoall_dedup.remoteRecvBuffs.~vector();

  return commSuccess;
}

bool ctranAllToAllDedupSupport(CtranComm* comm) {
  bool ctranSupport = false;
  const auto statex = comm->statex_.get();
  if (ctranInitialized(comm)) {
    ctranSupport = true;
    for (int rank = 0; rank < statex->nRanks(); rank++) {
      if (comm->ctran_->mapper->getBackend(rank) == CtranMapperBackend::UNSET) {
        ctranSupport = false;
        break;
      }
    }
  }

  return ctranSupport;
}
