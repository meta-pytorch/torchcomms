// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"
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

namespace allgatherp {
std::vector<int> computeRailRanks(const ncclx::CommStateX* statex) {
  const int localRank = statex->localRank();
  const int nNodes = statex->nNodes();

  std::vector<int> ranks;
  ranks.reserve(nNodes);
  for (int node = 0; node < nNodes; ++node) {
    ranks.push_back(statex->localRankToRank(localRank, node));
  }
  return ranks;
}
} // namespace allgatherp

namespace {
const std::string algoWinInitName = "CtranAllGatherWinInit";

bool needsRailExchange(
    const PersistArgs* pArgs,
    const std::vector<int>& railRanks,
    int myRank) {
  for (const int rank : railRanks) {
    if (rank != myRank &&
        pArgs->remoteAccessKeys[rank].backend == CtranMapperBackend::UNSET) {
      return true;
    }
  }
  return false;
}

commResult_t railBarrier(
    CtranMapper* mapper,
    const std::vector<int>& railRanks,
    int myRank) {
  if (railRanks.size() <= 1) {
    return commSuccess;
  }
  std::vector<CtranMapperRequest> reqs((railRanks.size() - 1) * 2);
  int reqIdx = 0;
  for (const int peerRank : railRanks) {
    if (peerRank == myRank) {
      continue;
    }
    FB_COMMCHECK(mapper->irecvCtrl(peerRank, &reqs[reqIdx++]));
    FB_COMMCHECK(mapper->isendCtrl(peerRank, &reqs[reqIdx++]));
  }
  for (auto& req : reqs) {
    FB_COMMCHECK(mapper->waitRequest(&req));
  }
  return commSuccess;
}

commResult_t validateWindowRemoteInfo(const CtranWin* win, CtranComm* comm) {
  const auto nRanks = comm->statex_->nRanks();
  const auto winRanks = static_cast<int>(win->remWinInfo.size());
  if (winRanks == 0) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Window remWinInfo not populated. Was exchange() called?");
  }
  if (winRanks == nRanks) {
    return commSuccess;
  }

  if (win->comm == nullptr || win->comm->statex_ == nullptr) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Window was registered on a communicator without statex");
  }
  const auto& windowRanksToCommRanks = win->comm->parentRanks();
  if (windowRanksToCommRanks.size() != win->remWinInfo.size()) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Window remWinInfo size {} does not match window rank map size {}",
        win->remWinInfo.size(),
        windowRanksToCommRanks.size());
  }

  std::vector<bool> seenRanks(nRanks, false);
  for (const auto commRank : windowRanksToCommRanks) {
    if (commRank < 0 || commRank >= nRanks) {
      FB_ERRORRETURN(
          commInvalidArgument,
          "Window rank map contains parent rank {} outside [0, {})",
          commRank,
          nRanks);
    }
    if (seenRanks[commRank]) {
      FB_ERRORRETURN(
          commInvalidArgument,
          "Window rank map contains duplicate parent rank {}",
          commRank);
    }
    seenRanks[commRank] = true;
  }

  return commSuccess;
}

commResult_t populateRemoteInfoFromWindow(
    const CtranWin* win,
    CtranComm* comm,
    PersistArgs* pArgs) {
  FB_COMMCHECK(validateWindowRemoteInfo(win, comm));

  const auto nRanks = comm->statex_->nRanks();
  pArgs->remoteRecvBuffs.assign(nRanks, nullptr);
  pArgs->remoteAccessKeys.assign(nRanks, CtranMapperRemoteAccessKey{});

  if (static_cast<int>(win->remWinInfo.size()) == nRanks) {
    for (int r = 0; r < nRanks; r++) {
      pArgs->remoteRecvBuffs[r] = win->remWinInfo[r].dataAddr;
      pArgs->remoteAccessKeys[r] = win->remWinInfo[r].dataRkey;
    }
    return commSuccess;
  }

  const auto& windowRanksToCommRanks = win->comm->parentRanks();
  for (int winRank = 0; winRank < static_cast<int>(win->remWinInfo.size());
       ++winRank) {
    const int commRank = windowRanksToCommRanks[winRank];
    pArgs->remoteRecvBuffs[commRank] = win->remWinInfo[winRank].dataAddr;
    pArgs->remoteAccessKeys[commRank] = win->remWinInfo[winRank].dataRkey;
  }

  return commSuccess;
}

// GPE callback: populate pArgs remote info from window, then mark initialized.
// Runs on GPE thread to avoid races between init and exec on the mapper epoch
// lock (see D76792218).
commResult_t populateWinPArgs(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->allgatherp_init.pArgs);
  auto* win = op->allgatherp_init.win;
  CtranComm* comm = opGroup.front()->comm_;

  CtranAlgoLogger logger(algoWinInitName, op->opCount, comm);

  FB_COMMCHECK(populateRemoteInfoFromWindow(win, comm, pArgs));

  const auto statex = comm->statex_.get();
  const auto railRanks = allgatherp::computeRailRanks(statex);

  // Deferred rail wait (production / cudagraph capture path): the local NVL
  // window info is already populated by populateRemoteInfoFromWindow. The
  // cross-node rail entries stay UNSET here and are filled lazily on the first
  // graph replay by ensureRailKeysExchanged() in the exec path. Skip the
  // in-line allGatherCtrl/railBarrier path.
  if (pArgs->railDeferred) {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "allGatherWinInit: rank {} deferred rail rkey wait (filled at first replay)",
        statex->rank());
    pArgs->initialized.store(true);
    return commSuccess;
  }

  // Direct (non-deferred) init: exchange the rail IB rkeys in-line. Only direct
  // callers that do not defer (the windowed-allgather unit tests) reach this;
  // the cudagraph capture path always defers above and fills lazily at replay.
  if (needsRailExchange(pArgs, railRanks, statex->rank())) {
    auto mapper = comm->ctran_->mapper.get();
    FB_COMMCHECK(mapper->allGatherCtrl(
        pArgs->recvbuff,
        pArgs->recvHdl,
        railRanks,
        pArgs->remoteRecvBuffs,
        pArgs->remoteAccessKeys,
        CtranMapperBackend::IB));
    FB_COMMCHECK(railBarrier(mapper, railRanks, statex->rank()));

    CLOGF_SUBSYS(
        INFO,
        INIT,
        "allGatherWinInit: rank {} exchanged rail IB rkeys over {} ranks",
        statex->rank(),
        railRanks.size());
  }

  pArgs->initialized.store(true);
  return commSuccess;
}
} // namespace

// Lazily exchange the cross-node rail IB rkeys on first use. winPersistBuffReg
// leaves the rail entries UNSET (populateWinPArgs only fills the local NVL
// window); the first graph replay's exec gpeFn calls this to fill them in-line
// before reading remote rail keys. At replay all ranks are in lockstep, so the
// allGatherCtrl + barrier is quick and not gated by any straggler's
// capture-time cuMemImport. Subsequent replays find the entries set (the
// needsRailExchange null-check returns false) and this is a no-op.
commResult_t ensureRailKeysExchanged(
    CtranComm* comm,
    allgatherp::PersistArgs* pArgs) {
  const auto statex = comm->statex_.get();
  const auto railRanks = allgatherp::computeRailRanks(statex);
  if (!needsRailExchange(pArgs, railRanks, statex->rank())) {
    return commSuccess;
  }
  auto mapper = comm->ctran_->mapper.get();
  FB_COMMCHECK(mapper->allGatherCtrl(
      pArgs->recvbuff,
      pArgs->recvHdl,
      railRanks,
      pArgs->remoteRecvBuffs,
      pArgs->remoteAccessKeys,
      CtranMapperBackend::IB));
  FB_COMMCHECK(railBarrier(mapper, railRanks, statex->rank()));
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "ensureRailKeysExchanged: rank {} lazily exchanged rail IB rkeys over {} ranks",
      statex->rank(),
      railRanks.size());
  return commSuccess;
}

commResult_t allGatherWinInit(
    CtranWin* win,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request,
    bool deferRail) {
  FB_COMMCHECK(validateWindowRemoteInfo(win, comm));

  auto algo = std::make_unique<AlgoImpl>(comm, stream);

  algo->pArgs.recvbuff = win->winDataPtr;
  algo->pArgs.recvHdl = win->dataRegHdl;
  algo->pArgs.initialized.store(false);

  // Deferred rail wait: the rail IB key exchange was issued in
  // winPersistBuffReg but is NOT waited here. Mark the rail entries deferred so
  // populateWinPArgs skips the in-line allGatherCtrl; the rail entries are
  // filled later by the first-replay drain registered on the comm.
  if (deferRail) {
    algo->pArgs.railDeferred = true;
  }

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
    case NCCL_ALLGATHER_P_ALGO::ctrdpipeline:
      return algo->execRecursiveDoubling(sendbuff, count, datatype);
    case NCCL_ALLGATHER_P_ALGO::ctsrdpipeline:
      return algo->execStreamedRecursiveDoubling(sendbuff, count, datatype);
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
