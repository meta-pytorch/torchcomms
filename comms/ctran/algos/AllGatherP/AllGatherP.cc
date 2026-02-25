// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/ctran/mapper/CtranMapper.h"

using ctran::algos::GpeKernelSync;
using ctran::allgatherp::AlgoImpl;

#define CHECK_VALID_PREQ(pReq)                                     \
  do {                                                             \
    if (pReq->type != CtranPersistentRequest::Type::ALLGATHER_P) { \
      FB_ERRORRETURN(                                              \
          commInvalidArgument,                                     \
          "Unexpected PersistentRequest type {} called into {}",   \
          pReq->type,                                              \
          __func__);                                               \
    }                                                              \
  } while (0)

namespace ctran::allgatherp {
const std::string algoInitName = "CtranAllGatherPInit";

commResult_t exchangeMemHdl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->allgatherp_init.pArgs);
  CtranComm* comm = opGroup.front()->comm_;
  auto mapper = comm->ctran_->mapper.get();

  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();

  CtranAlgoLogger logger(algoInitName, op->opCount, comm);

  pArgs->remoteAccessKeys.resize(nRanks, CtranMapperRemoteAccessKey());
  pArgs->remoteRecvBuffs.resize(nRanks, nullptr);
  FB_COMMCHECK(mapper->allGatherCtrl(
      pArgs->recvbuff,
      pArgs->recvHdl,
      pArgs->remoteRecvBuffs,
      pArgs->remoteAccessKeys));

  // Ensure all ranks have finished remote importing before return
  FB_COMMCHECK(mapper->barrier());

  if (NCCL_CTRAN_ENABLE_TRACE_LOG) {
    for (int i = 0; i < nRanks; i++) {
      CLOGF_TRACE(
          INIT,
          "    remoteRecvBuffs[{}]: {}, remoteAccessKey: {}",
          i,
          (void*)pArgs->remoteRecvBuffs[i],
          pArgs->remoteAccessKeys[i].toString());
    }
  }

  // Mark the remote registration as initialized, so that consequent execution
  // can schedule CE based NVL copy
  pArgs->initialized.store(true);
  return commSuccess;
}

extern __global__ void ncclKernelAllGatherPInit(
    int* flag,
    CtranAlgoDeviceState* devState);

commResult_t AlgoImpl::initialize() {
  auto opCount = comm_->ctran_->getOpCount();

  // Pre-allocate streams for parallel icopy in nvlCeBcast
  // (nLocalRanks - 1) local peers
  // --> the main stream + (nLocalRanks - 2) CE streams
  const auto nLocalRanks = comm_->statex_->nLocalRanks();
  if (nLocalRanks > 2) {
    ceStreams_.resize(nLocalRanks - 2);
    for (auto& s : ceStreams_) {
      FB_CUDACHECK(cudaStreamCreate(&s));
    }
  }

  CTRAN_COLL_INFO(
      algoInitName,
      nullptr,
      pArgs.recvbuff,
      pArgs.maxRecvCount,
      pArgs.datatype,
      -1,
      comm_,
      stream_);

  void* base = nullptr;
  FB_CUDACHECK(
      cudaHostAlloc(&base, sizeof(GpeKernelSync), cudaHostAllocDefault));

  resource_.pipeSync = reinterpret_cast<GpeKernelSync*>(base);
  new (resource_.pipeSync) GpeKernelSync(1 /* numWorkers */);

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERP_INIT,
      stream_,
      algoInitName,
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = comm_->ctran_->algo->getDevState();

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLGATHERP_INIT, stream_, comm_, opCount);
  op->allgatherp_init.pArgs = &pArgs;
  opGroup.push_back(std::move(op));

  FB_COMMCHECK(comm_->ctran_->gpe->submit(
      std::move(opGroup),
      exchangeMemHdl,
      config,
      reinterpret_cast<void*>(ncclKernelAllGatherPInit)));

  return commSuccess;
};

commResult_t AlgoImpl::destroy() {
  if (resource_.pipeSync) {
    FB_CUDACHECK(cudaFreeHost(resource_.pipeSync));
    resource_.pipeSync = nullptr;
  }

  // Destroy pre-allocated CE streams
  for (auto& s : ceStreams_) {
    if (s != nullptr) {
      FB_CUDACHECK(cudaStreamDestroy(s));
      s = nullptr;
    }
  }
  ceStreams_.clear();

  return commSuccess;
}
} // namespace ctran::allgatherp

namespace ctran {
bool allGatherPSupport(CtranComm* comm) {
  bool ctranSupport = false;
  const auto statex = comm->statex_.get();
  if (ctranInitialized(comm)) {
    ctranSupport = true;
    auto mapper = comm->ctran_->mapper.get();
    const auto myRank = statex->rank();
    // Check if all remote peers are supported by ctran
    for (auto rank = 0; rank < statex->nRanks(); rank++) {
      if (mapper->getBackend(rank) == CtranMapperBackend::UNSET &&
          rank != myRank) {
        ctranSupport = false;
        break;
      }
    }
  }

  return ctranSupport;
}

commResult_t allGatherPInit(
    void* recvbuff,
    const size_t maxRecvCount,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request) {
  AlgoImpl* algo = new AlgoImpl(comm, stream);
  if (!algo) {
    return commSystemError;
  }

  auto guard = folly::makeGuard([algo] {
    if (algo) {
      delete algo;
    }
  });

  auto mapper = comm->ctran_->mapper.get();
  const auto maxRecvSize = maxRecvCount * commTypeSize(datatype);
  void* recvHdl;
  bool localRegRecv;

  FB_COMMCHECK(
      mapper->searchRegHandle(recvbuff, maxRecvSize, &recvHdl, &localRegRecv));
  if (localRegRecv) {
    FB_COMMCHECK(mapper->deregDynamic(recvHdl));
    CLOGF(
        ERR,
        "recvbuff is not registered. Pointer: {} length: {}",
        (void*)recvbuff,
        maxRecvSize);
    return commInternalError;
  }

  // Set up persistent arguments
  algo->pArgs.recvHdl = recvHdl;
  algo->pArgs.recvbuff = recvbuff;
  algo->pArgs.maxRecvCount = maxRecvCount;
  algo->pArgs.datatype = datatype;
  algo->pArgs.initialized.store(false);

  // Initialize algo internal resource and schedule handle exchange on GPE
  // thread
  FB_COMMCHECK(algo->initialize());

  request = new CtranPersistentRequest(
      CtranPersistentRequest::Type::ALLGATHER_P, comm, stream);
  if (!request) {
    return commSystemError;
  }
  request->algo = algo;

  guard.dismiss();
  return commSuccess;
}

commResult_t allGatherPExec(
    const void* sendbuff,
    const size_t count,
    commDataType_t datatype,
    CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  auto algo = reinterpret_cast<AlgoImpl*>(request->algo);
  const auto nRanks = request->comm_->statex_->nRanks();
  if (count * nRanks * commTypeSize(datatype) >
      algo->pArgs.maxRecvCount * commTypeSize(algo->pArgs.datatype)) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "AllGatherP invalid sendbuff count {} * nRanks {} * sizeof datatype {} exceeds maxRecvCount {} * sizeof datatype {}.",
        count,
        nRanks,
        datatype,
        algo->pArgs.maxRecvCount,
        algo->pArgs.datatype);
  }

  switch (NCCL_ALLGATHER_P_ALGO) {
    case NCCL_ALLGATHER_P_ALGO::ctdirect:
      return algo->execDirect(sendbuff, count, datatype);
    case NCCL_ALLGATHER_P_ALGO::ctpipeline:
      return algo->execPipeline(sendbuff, count, datatype);
    default:
      return ErrorStackTraceUtil::log(commInternalError);
  }
}

commResult_t allGatherPDestroy(CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  // No need to dereg handles now since user should call explicit
  // commDeregister before buffer is freed
  auto algo = reinterpret_cast<AlgoImpl*>(request->algo);
  // Destroy algo internal resource
  FB_COMMCHECK(algo->destroy());

  delete algo;
  request->algo = nullptr;

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "allGatherPDestroy: rank {} destroyed request {}",
      request->comm_->statex_->rank(),
      (void*)request);

  return commSuccess;
}
} // namespace ctran
