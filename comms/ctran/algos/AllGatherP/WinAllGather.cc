// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ctran::allgatherp::AlgoImpl;
using ctran::allgatherp::PersistArgs;

#define CHECK_VALID_PREQ(pReq)                                         \
  do {                                                                 \
    if (!(pReq)) {                                                     \
      FB_ERRORRETURN(                                                  \
          commInvalidArgument,                                         \
          "Null PersistentRequest passed to {}",                       \
          __func__);                                                   \
    }                                                                  \
    if (pReq->type != CtranPersistentRequest::Type::ALLGATHER_P_WIN) { \
      FB_ERRORRETURN(                                                  \
          commInvalidArgument,                                         \
          "Unexpected PersistentRequest type {} called into {}",       \
          pReq->type,                                                  \
          __func__);                                                   \
    }                                                                  \
  } while (0)

namespace ctran {

namespace {
const std::string algoWinInitName = "CtranAllGatherWinInit";

// GPE callback: populate pArgs remote info from window, then mark initialized.
// Runs on GPE thread to avoid races between init and exec on the mapper epoch
// lock (see D76792218).
commResult_t populateWinPArgs(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->allgatherp_init.pArgs);
  auto* win = op->allgatherp_init.win;
  CtranComm* comm = opGroup.front()->comm_;

  const auto nRanks = comm->statex_->nRanks();

  CtranAlgoLogger logger(algoWinInitName, op->opCount, comm);

  pArgs->remoteRecvBuffs.resize(nRanks);
  pArgs->remoteAccessKeys.resize(nRanks);
  for (int r = 0; r < nRanks; r++) {
    pArgs->remoteRecvBuffs[r] = win->remWinInfo[r].dataAddr;
    pArgs->remoteAccessKeys[r] = win->remWinInfo[r].dataRkey;
  }

  pArgs->initialized.store(true);
  return commSuccess;
}
} // namespace

commResult_t allGatherWinInit(
    CtranWin* win,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request) {
  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();

  if (win->remWinInfo.empty() ||
      static_cast<int>(win->remWinInfo.size()) != nRanks) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Window remWinInfo not populated (size {}). "
        "Was exchange() called?",
        win->remWinInfo.size());
  }

  auto algo = std::make_unique<AlgoImpl>(comm, stream);

  algo->pArgs.recvbuff = win->winDataPtr;
  algo->pArgs.recvHdl = win->dataRegHdl;
  algo->pArgs.initialized.store(false);

  FB_COMMCHECK(algo->initResources());

  // Submit remote info population to GPE thread via submitHost (no kernel).
  // submitHost is not captured by cudagraph, so it works correctly during
  // both graph capture and eager execution. This matches the pattern from
  // allGatherPInit to avoid races between init and exec on the
  // mapper epoch lock.
  auto opCount = comm->ctran_->getOpCount();

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERP_INIT,
      stream,
      algoWinInitName,
      opCount);

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLGATHERP_INIT, stream, comm, opCount);
  op->allgatherp_init.pArgs = &algo->pArgs;
  op->allgatherp_init.win = win;
  opGroup.push_back(std::move(op));

  FB_COMMCHECK(comm->ctran_->gpe->submitHost(
      std::move(opGroup), populateWinPArgs, config, nullptr /* cpuFlag */));

  request = new CtranPersistentRequest(
      CtranPersistentRequest::Type::ALLGATHER_P_WIN, comm, stream);
  request->algo = algo.release();

  return commSuccess;
}

commResult_t allGatherWinExec(
    const void* sendbuff,
    const size_t count,
    commDataType_t datatype,
    CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);

  switch (NCCL_ALLGATHER_P_ALGO) {
    case NCCL_ALLGATHER_P_ALGO::ctdirect:
      return algo->execDirect(sendbuff, count, datatype);
    case NCCL_ALLGATHER_P_ALGO::ctpipeline:
      return algo->execPipeline(sendbuff, count, datatype);
    default:
      return ErrorStackTraceUtil::log(commInternalError);
  }
}

commResult_t allGatherWinDestroy(CtranPersistentRequest* request) {
  CHECK_VALID_PREQ(request);

  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  if (!algo) {
    return commSuccess;
  }
  FB_COMMCHECK(algo->destroy());
  delete algo;
  request->algo = nullptr;

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "allGatherWinDestroy: rank {} destroyed request {}",
      request->comm_->statex_->rank(),
      (void*)request);

  return commSuccess;
}

} // namespace ctran
