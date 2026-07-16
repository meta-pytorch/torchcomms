// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/PersistentCleanup.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/regcache/IpcRegCache.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/CtranPerf.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include <folly/ScopeGuard.h>

#include <memory>

using ctran::alltoallp::AlgoImpl;
using ctran::alltoallp::createPersistentRequest;
using ctran::alltoallp::destroyPersistentRequest;
using ctran::alltoallp::PersistArgs;

#define RETURN_ALLTOALLP_IB_IMPL(perfconfig) \
  return ctranAllToAllPIbImpl<perfconfig>(   \
      op->alltoallP.sendbuff,                \
      counts,                                \
      displs,                                \
      pArgs->recvbuff,                       \
      counts,                                \
      displs,                                \
      pArgs->datatype,                       \
      comm,                                  \
      std::move(timestamp),                  \
      pArgs);

#define RETURN_IMPL(config) RETURN_ALLTOALLP_IB_IMPL(config)

namespace {
const auto myAlgo = NCCL_ALLTOALL_ALGO::ctran;

// Inter-node IB control handshake over this rank's IB peer list. Two modes:
//   - exchangeRkey: one-shot rkey exchange — send this rank's recvbuff handle
//     to each IB peer and receive their remote recvbuff pointers / access keys
//     into pArgs->remoteRecvBuffs / remoteAccessKeys (indexed by global rank).
//   - !exchangeRkey: SYNC-only phase-lock — ensure remote buffer is ready to be
//      updated via RDMA (previous use on receiver side is done)
inline commResult_t ibHandshake(
    CtranMapper* mapper,
    const ncclx::CommStateX* statex,
    ctran::alltoallp::PersistArgs* pArgs,
    bool exchangeRkey) {
  const int nRanks = statex->nRanks();
  const int myRank = statex->rank();

  std::vector<int> ibPeers;
  for (int p = 1; p < nRanks; p++) {
    const int peer = (myRank + p) % nRanks;
    if (!statex->isSameNode(myRank, peer)) {
      ibPeers.push_back(peer);
    }
  }
  std::vector<CtranMapperRequest> ibSendCtrlReqs(ibPeers.size());
  std::vector<CtranMapperRequest> ibRecvCtrlReqs(ibPeers.size());
  if (exchangeRkey) {
    std::vector<void*> ibRecvBuffs(nRanks, pArgs->recvbuff);
    FB_COMMCHECK(mapper->isendCtrlBatch(
        ibRecvBuffs,
        pArgs->recvHdl,
        ibPeers,
        ibSendCtrlReqs,
        CtranMapperBackend::IB));
    int id = 0;
    for (const auto peer : ibPeers) {
      FB_COMMCHECK(mapper->irecvCtrl(
          &pArgs->remoteRecvBuffs[peer],
          &pArgs->remoteAccessKeys[peer],
          peer,
          &ibRecvCtrlReqs[id++]));
    }
  } else {
    int id = 0;
    for (const auto peer : ibPeers) {
      FB_COMMCHECK(mapper->irecvCtrl(peer, &ibRecvCtrlReqs[id]));
      FB_COMMCHECK(mapper->isendCtrl(peer, &ibSendCtrlReqs[id]));
      id++;
    }
  }
  FB_COMMCHECK(mapper->waitAllRequests(ibSendCtrlReqs));
  FB_COMMCHECK(mapper->waitAllRequests(ibRecvCtrlReqs));
  return commSuccess;
}

template <typename PerfConfig = DefaultPerfCollConfig>
commResult_t ctranAllToAllPIbImpl(
    const void* sendbuff,
    std::vector<size_t>& sendCounts,
    std::vector<size_t>& sDispls,
    void* recvbuff,
    std::vector<size_t>& recvCounts,
    std::vector<size_t>& rDispls,
    commDataType_t datatype,
    CtranComm* comm,
    std::unique_ptr<CtranMapperTimestamp> timestamp,
    ctran::alltoallp::PersistArgs* pArgs) {
  const auto& statex = comm->statex_;
  const int myRank = statex->rank();
  const int nRanks = statex->nRanks();

  const std::string algoName = AlgoImpl::algoName(myAlgo);
  const bool useProfiler = NCCL_CTRAN_PROFILING != NCCL_CTRAN_PROFILING::none;

  std::vector<const void*> sendBuffs(nRanks);

  void* sendMemHdl = nullptr;
  std::vector<void*> tmpRegHdls;

  std::vector<int> ibRecvPeers, ibSendPeers;
  std::unordered_set<int> ibPeers;

  if (sendCounts.size() > 0) {
    std::vector<size_t> sendSizes(nRanks, 0);
    for (int i = 0; i < nRanks; i++) {
      sendSizes[i] = sendCounts[i] * commTypeSize(datatype);
    }
    std::vector<size_t> recvSizes(nRanks, 0);
    for (int i = 0; i < nRanks; i++) {
      recvSizes[i] = recvCounts[i] * commTypeSize(datatype);
    }
    CtranMapperContext context(algoName, sendSizes, recvSizes);
    comm->ctran_->mapper->setContext(std::move(context));
  }

  // Prepare buffers shifted with displacement, and set ctrl/put/notify
  // schedules. Try to schedule ctrl message and put sequence as rank i start
  // sending to rank i+1 to avoid congestion in potential all-to-one case.
  // Specified in putPeers, sendCtrlPeers.
  size_t contigSendBufSize = 0;
  for (int i = 0; i < nRanks; i++) {
    int peer = (myRank + i) % nRanks;
    if (sendCounts[peer]) {
      sendBuffs[peer] = static_cast<const char*>(sendbuff) +
          sDispls[peer] * commTypeSize(datatype);
      ibSendPeers.push_back(peer);
      ibPeers.insert(peer);
      contigSendBufSize =
          std::max(contigSendBufSize, sDispls[peer] + sendCounts[peer]);
    }
    if (recvCounts[peer]) {
      ibRecvPeers.push_back(peer);
      ibPeers.insert(peer);
    }
  }

  std::vector<CtranMapperNotify> notifyVec(ibRecvPeers.size());
  FB_COMMCHECK(comm->ctran_->mapper->initNotifyBatchIB(ibRecvPeers, notifyVec));

  // Ctrl phase:
  //   - first exec (!ibKeysExchanged): do the one-shot inter-node IB rkey
  //     exchange, populating pArgs->remoteRecvBuffs/remoteAccessKeys, then mark
  //     ibKeysExchanged so later execs skip it.
  //   - later exec without skipCtrlMsg: bare SYNC-only handshake to ensure
  //     remote buffer is ready to be updated.
  //   - later exec with skipCtrlMsg: skip ctrl entirely as user program can
  //     ensure remote buffer is ready to be updated (e.g., double buffering).
  bool doHandshake = !pArgs->skipCtrlMsg || !pArgs->ibKeysExchanged;
  if (doHandshake) {
    FB_COMMCHECK(ibHandshake(
        comm->ctran_->mapper.get(),
        statex.get(),
        pArgs,
        !pArgs->ibKeysExchanged));
  }
  pArgs->ibKeysExchanged = true;

  // Local offset-adjusted remote recvbuf pointers. In symmetric AllToAll my
  // rank's recv-displ is my fixed slot in every peer's recvbuf, so compute the
  // byte offset locally. On first exec the inter-node slots of
  // pArgs->remoteRecvBuffs were just populated above.
  const size_t remoteRecvBuffOffset = rDispls[myRank] * commTypeSize(datatype);
  std::vector<void*> remoteRecvBuffs(nRanks);
  for (int i = 0; i < nRanks; i++) {
    remoteRecvBuffs[i] =
        static_cast<char*>(pArgs->remoteRecvBuffs[i]) + remoteRecvBuffOffset;
  }

  // Search for the handle only when there are SendPeers to avoid attempting to
  // search/register with a buffer size of 0.
  if (!ibSendPeers.empty()) {
    // TODO: move this to main thread before submitting to GPE
    FB_COMMCHECK(searchRegHandle(
        comm,
        sendbuff,
        contigSendBufSize * commTypeSize(datatype),
        sendMemHdl,
        tmpRegHdls));
  }

  std::vector<CtranMapperRequest> ibPutReqs(ibSendPeers.size());
  int idx = 0;
  // Issue network puts using provided remote recvbuff handles
  for (const auto& peer : ibSendPeers) {
    if (useProfiler) {
      timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(peer));
    }
    auto sendSize = sendCounts[peer] * commTypeSize(datatype);
    // FIXME: we should compare sendSize with real maxWqeSize:
    // NCCL_CTRAN_IB_QP_SCALING_THRESHOLD may not be maxWqeSize if user
    // specified NCCL_CTRAN_IB_QP_CONFIG_ALGO to overwrite qp_scaling_threshold
    // for certain algo.
    bool enableFastPath = NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS &&
        (sendSize <= NCCL_CTRAN_IB_QP_SCALING_THRESHOLD);
    FB_COMMCHECK(comm->ctran_->mapper->iput<PerfConfig>(
        sendBuffs[peer],
        remoteRecvBuffs[peer],
        sendSize,
        peer,
        CtranMapperConfig{
            .memHdl_ = sendMemHdl,
            .remoteAccessKey_ = pArgs->remoteAccessKeys[peer],
            .notify_ = true /*notify*/,
            .ibFastPath_ = enableFastPath},
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

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "AllToAllPIbImpl: rank {} completed AllToAllP execution sendbuff {} recvbuff {} count {} size {} ibPeers {}",
      comm->statex_->rank(),
      sendbuff,
      recvbuff,
      sendCounts[ibSendPeers[0]],
      sendCounts[ibSendPeers[0]] * commTypeSize(datatype),
      ibPeers.size());
  return commSuccess;
}

commResult_t gpeFn(const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = op->comm_;
  const auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();

  std::vector<size_t> counts(nRanks, 0);
  std::vector<size_t> displs(nRanks, 0);

  CtranAlgoLogger logger(AlgoImpl::algoName(myAlgo), op->opCount, comm);

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(AlgoImpl::algoName(myAlgo)));

  const int myNode = statex->node();
  for (int i = 0; i < nRanks; i++) {
    int peerNode = statex->node(i);
    displs[i] = op->alltoallP.count * i;
    // GPE thread handles only remote peers
    if (myNode != peerNode) {
      counts[i] = op->alltoallP.count;
    }
  }

  // The per-rank offset-adjusted remote recvbuf pointer vector is built inside
  // ctranAllToAllPIbImpl (it computes my rank's symmetric recv-displ offset
  // locally), after the (possibly lazy) inter-node rkey exchange populates
  // remoteRecvBuffs.
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->alltoallP.pArgs);

  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    RETURN_IMPL(LowLatencyCollConfig)
  } else {
    RETURN_IMPL(DefaultPerfCollConfig)
  }
}

// Exchange intra-node NVL IPC handles into pArgs on the GPE thread (submitted
// via submitHost) and mark initialized. remoteRecvBuffs / remoteAccessKeys are
// indexed by global rank. Inter-node IB rkeys are NOT exchanged here; they are
// exchanged lazily on the first exec via ibHandshake.
commResult_t exchangeMemHdl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = op->comm_;
  auto* pArgs = reinterpret_cast<PersistArgs*>(op->alltoallP.pArgs);
  auto* mapper = comm->ctran_->mapper.get();
  const auto* statex = comm->statex_.get();
  const int nRanks = statex->nRanks();

  pArgs->remoteRecvBuffs.assign(nRanks, nullptr);
  pArgs->remoteAccessKeys.assign(nRanks, CtranMapperRemoteAccessKey());

  // Intra-node NVL IPC handle exchange (blocking). Inter-node IB rkeys are
  // exchanged lazily on the first exec (see ibHandshake), not here.
  FB_COMMCHECK(mapper->intraAllGatherCtrl(
      pArgs->recvbuff,
      pArgs->recvHdl,
      pArgs->remoteRecvBuffs,
      pArgs->remoteAccessKeys,
      pArgs->remoteIpcRegHdls));

  FB_COMMCHECK(mapper->intraBarrier());

  pArgs->initState = ctran::alltoallp::InitState::kInitialized;
  return commSuccess;
}
} // namespace

namespace ctran::alltoallp {

AlgoImpl::~AlgoImpl() = default;

commResult_t AlgoImpl::destroy() {
  // Async init populates pArgs on the GPE thread; wait for it only while that
  // async init is still in flight (kSubmitted). Capture the result rather than
  // early-returning, so the cleanup below always runs.
  auto res = commSuccess;
  if (pArgs.initState.load() == InitState::kSubmitted) {
    res = waitInit();
  }
  // Release the scoped NVL IPC imports and the scoped local recv registration.
  // Both are deferred/SW-only (no CUDA), so safe from the graph-destroy
  // callback.
  pArgs.remoteIpcRegHdls.clear();
  pArgs.recvRegHdl.reset();
  return res;
}

commResult_t createPersistentRequest(
    CtranComm* comm,
    cudaStream_t stream,
    void* recvbuff,
    size_t maxRecvCount,
    commDataType_t datatype,
    CtranPersistentRequest** out,
    bool waitForInit,
    bool skipCtrlMsg) {
  if (out == nullptr) {
    return commInvalidArgument;
  }
  *out = nullptr;

  // Acquire the local recv registration through the scoped regcache API. The
  // recvbuff's segment must already be allocator-cached (CCA hook); otherwise
  // acquireScopedRegister returns commInvalidUsage and we return immediately
  // with nothing allocated (no leak).
  const size_t recvBytes = maxRecvCount * commTypeSize(datatype);
  ctran::ScopedRegHdl localRecvReg;
  auto regCache = ctran::RegCache::getInstance();
  ctran::CHECK_VALID_REGCACHE(regCache);
  FB_COMMCHECK(regCache->acquireScopedRegister(
      recvbuff,
      recvBytes,
      comm->statex_->cudaDev(),
      comm->ctran_->mapper->getBackends(),
      localRecvReg));

  auto algo = std::make_unique<AlgoImpl>(comm, stream);

  // The unique_ptr locals auto-clean the AlgoImpl/request on any failure below.
  // The scoped handle is owned by the persistent request and released at
  // destroy.
  algo->pArgs.recvHdl = localRecvReg.get();
  algo->pArgs.recvRegHdl =
      std::make_unique<ctran::ScopedRegHdl>(std::move(localRecvReg));
  algo->pArgs.recvbuff = recvbuff;
  algo->pArgs.maxRecvCount = maxRecvCount;
  algo->pArgs.datatype = datatype;
  algo->pArgs.skipCtrlMsg = skipCtrlMsg;

  auto request = std::make_unique<CtranPersistentRequest>(
      CtranPersistentRequest::Type::ALLTOALL_P, comm, stream);
  request->algo = algo.release();
  // Once request->algo is set, the AlgoImpl is only released via
  // destroyPersistentRequest, not the unique_ptr destructor. Guard it against
  // early returns below and dismiss right before releasing ownership to *out.
  auto* const rawRequest = request.get();
  auto reqGuard = folly::makeGuard([rawRequest] {
    FB_COMMCHECKIGNORE(destroyPersistentRequest(rawRequest));
  });

  auto* algoPtr = reinterpret_cast<AlgoImpl*>(request->algo);

  // Dummy placeholder for existing submitHost API, no actual kernel launch.
  KernelConfig config(
      KernelConfig::KernelType::ALLTOALL,
      stream,
      "CtranAllToAllPInit",
      comm->ctran_->getOpCount());

  std::vector<std::unique_ptr<OpElem>> opGroup;
  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLTOALLP, stream, comm, config.opCount);
  op->alltoallP.pArgs = &algoPtr->pArgs;
  opGroup.push_back(std::move(op));

  // Publish kSubmitted BEFORE handing work to the GPE thread. exchangeMemHdl
  // sets kInitialized on the GPE thread; setting kSubmitted after submit could
  // clobber that back to kSubmitted and deadlock waitInit()/destroy(). If
  // submit fails, reset to kUninitialized so a later destroy() does not wait on
  // an init that never ran.
  algoPtr->pArgs.initState = InitState::kSubmitted;
  auto submitGuard = folly::makeGuard(
      [algoPtr] { algoPtr->pArgs.initState = InitState::kUninitialized; });
  FB_COMMCHECK(comm->ctran_->gpe->submitHost(
      std::move(opGroup), exchangeMemHdl, config, /* cpuFlag */ nullptr));
  submitGuard.dismiss();

  // Eager init returns async; graph capture waits synchronously so the captured
  // collective ops see fully-populated pArgs.
  if (waitForInit) {
    FB_COMMCHECK(algoPtr->waitInit());
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-A2AP: Rank {} createPersistentRequest ({}): comm {} recvbuff {} recvHdl {} nLocalRanks {} commHash {:x}",
      comm->statex_->rank(),
      waitForInit ? "graph" : "eager",
      (void*)comm,
      algoPtr->pArgs.recvbuff,
      algoPtr->pArgs.recvHdl,
      comm->statex_->nLocalRanks(),
      comm->statex_->commHash());

  reqGuard.dismiss();
  *out = request.release();

  // Route teardown through the one-shot cleanup token: it releases the
  // scoped registration (via destroyPersistentRequest), running at most once
  // regardless of which path (eager free, graph-destroy
  // callback, comm drain before terminate()) fires first.
  auto cleanup = std::make_shared<PersistentCleanup>([request = *out]() {
    FB_COMMCHECKIGNORE(destroyPersistentRequest(request));
  });
  (*out)->cleanup_ = cleanup;
  comm->registerPersistentCleanup(cleanup);
  return commSuccess;
}

commResult_t destroyPersistentRequest(CtranPersistentRequest* const request) {
  if (request == nullptr) {
    return commSuccess;
  }
  auto res = commSuccess;
  auto* algo = reinterpret_cast<AlgoImpl*>(request->algo);
  if (algo != nullptr) {
    // destroy() is best-effort; delete unconditionally so a partial failure
    // does not leak the AlgoImpl or leave request->algo dangling.
    res = algo->destroy();
    delete algo;
    request->algo = nullptr;
  }
  return res;
}

commResult_t AlgoImpl::exec(const void* sendbuff, const size_t count) {
  auto recvbuff = pArgs.recvbuff;
  auto datatype = pArgs.datatype;
  auto opCount = comm_->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      algoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      0UL,
      datatype,
      -1,
      comm_,
      stream_);

  if (count == 0) {
    return commSuccess;
  }
  if (count * comm_->statex_->nRanks() > pArgs.maxRecvCount) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "AllToAllP send/recv count {} times nRanks {} exceeds maximum count {}.",
        count,
        comm_->statex_->nRanks(),
        pArgs.maxRecvCount);
  }

  // Wait till async init is done, so following IPC CE copy can use the imported
  // recvbuf. IB rkey exchange with skipCtrlMsg is guaranteed to finish before
  // exec relying on GPE FIFO cmd queue.
  if (comm_->statex_->nLocalRanks() > 1) {
    FB_COMMCHECK(waitInit());
  }

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALL, stream_, algoName(myAlgo), opCount);
  FB_COMMCHECK(
      ctran::alltoall::setupKernelConfig(
          sendbuff, recvbuff, count, datatype, comm_, stream_, config));

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  // Passing op only when remote peers are present
  if (comm_->statex_->nNodes() > 1) {
    auto op = std::make_unique<OpElem>(
        OpElem::opType::ALLTOALLP, stream_, comm_, opCount);
    op->alltoallP.pArgs = &pArgs;
    op->alltoallP.sendbuff = sendbuff;
    op->alltoallP.count = count;
    opGroup.push_back(std::move(op));
    CLOGF_TRACE(
        COLL,
        "AllToAllPExec: rank {} submit op sendbuff {} count {}",
        comm_->statex_->rank(),
        opGroup.front().get()->alltoallP.sendbuff,
        opGroup.front().get()->alltoallP.count);
  }

  FB_COMMCHECK(comm_->ctran_->gpe->submit(
      std::move(opGroup),
      gpeFn,
      config,
      reinterpret_cast<void*>(ctran::alltoall::alltoallKerns[datatype])));
  return commSuccess;
}
} // namespace ctran::alltoallp
