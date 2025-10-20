// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAllvDedup/AlgoImpl.h"
#include "comms/ctran/algos/AllToAllvDedup/CommonDev.h"
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/gpe/CtranGpe.h"

// Use to comment for metadata used by pytorch version of combine. We host it
// only for short-term till custom combine kernel is completed
#define WITH_PYTORCH_METADATA 1

namespace ctran::alltoallvdedup {
using namespace ncclx;
using namespace ::ctran::utils;

namespace {
// updated when GPE thread starts a new collective, only for logging purpose
thread_local uint64_t thOpCount = -1;

const std::string algoName = AlgoImpl::algoName(AlgoImpl::Phase::kPrepare);
#define TRACE_LOG(fmt, ...) \
  CLOGF_TRACE(COLL, "{} opCount {}: " fmt, algoName, thOpCount, ##__VA_ARGS__);

struct PrepareCtx {
  PersistArgs pArgs;
  ResourceRef* resource;
  CommStateX* commStatex;
  CtranMapper* mapper;
  utils::TraceRecord* ts;
  // metadata for external combine
  int* xnodeInputSplits;
  int* xnodeOutputSplits;
  int* localInputSplits;
  int* localOutputSplits;
  int64_t* xnodeGatherIndices;
  int64_t* localGatherIndices;
  int64_t* eGatherIndices;
};

void setupKernelArgs_(
    const PersistArgs& pArgs,
    const PersistConfig& config,
    const PrepareArgs& args,
    const CommStateX* statex,
    ResourceRef& resRef,
    PrepareKernArgs& kernArgs) {
  // Input arguments
  kernArgs.config = config;
  kernArgs.pArgs = pArgs;
  kernArgs.prepareArgs = args;

  // Resource arguments
  GET_RESOURCE_BUFPTR(&resRef, kBlockRecvBucketsH, kernArgs.blockRecvBucketsH);
  GET_RESOURCE_BUFPTR(&resRef, kNumForwardBlocksH, kernArgs.numForwardBlocksH);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpNumRecvBlocksBuff, kernArgs.tmpNumRecvBlocksBuff);
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpNumRecvBlocksBuffH, kernArgs.tmpNumRecvBlocksBuffH);

  std::vector<size_t*> tmpRemNumRecvBlocksBuffs;
  GET_RESOURCE_REM_BUFPTRS(
      &resRef,
      kTmpNumRecvBlocksBuff,
      statex->nLocalRanks(),
      tmpRemNumRecvBlocksBuffs);
  for (int i = 0; i < statex->nLocalRanks(); i++) {
    kernArgs.tmpRemNumRecvBlocksBuffs[i] = tmpRemNumRecvBlocksBuffs[i];
  }
  GET_RESOURCE_BUFPTR(
      &resRef, kTmpNumSendBlocksBuffH, kernArgs.tmpNumSendBlocksBuffH);
  GET_RESOURCE_BUFPTR(&resRef, kTmpRecvOffsets, kernArgs.tmpRecvOffsets);

  GET_RESOURCE_BUFPTR(
      &resRef, kLocalOutputSplits, kernArgs.tmpLocalOutputSplits);
  std::vector<int*> tmpRemLocalOutputSplits;
  GET_RESOURCE_REM_BUFPTRS(
      &resRef,
      kLocalOutputSplits,
      statex->nLocalRanks(),
      tmpRemLocalOutputSplits);
  for (int i = 0; i < statex->nLocalRanks(); i++) {
    kernArgs.tmpRemLocalOutputSplits[i] = tmpRemLocalOutputSplits[i];
  }
  GET_RESOURCE_BUFPTR(
      &resRef, kLocalOutputSplitsH, kernArgs.tmpLocalOutputSplitsH);

  GET_RESOURCE_BUFPTR(&resRef, kRankBitmaps, kernArgs.tmpRankBitmaps);
  std::vector<int*> tmpRemRankBitmaps;
  GET_RESOURCE_REM_BUFPTRS(
      &resRef, kRankBitmaps, statex->nLocalRanks(), tmpRemRankBitmaps);
  for (int i = 0; i < statex->nLocalRanks(); i++) {
    kernArgs.tmpRemRankBitmaps[i] = tmpRemRankBitmaps[i];
  }
  GET_RESOURCE_BUFPTR(&resRef, kRankBitmapsH, kernArgs.tmpRankBitmapsH);

  kernArgs.kSync = resRef.kSync;
}

void setupKernelConfig_(
    const ICtran* ctran,
    PrepareKernArgs& kernArgs,
    const PersistConfig& pConfig,
    KernelConfig& config) {
  config.numThreads = pConfig.numPrepareThreads;
  config.numBlocks = pConfig.numPrepareThreadBlocks;
  config.args.devState_d = ctran->algo->getDevState();
  config.algoArgs = reinterpret_cast<void*>(&kernArgs);
}

void setupGpeOp_(
    const PersistArgs& pArgs,
    const PrepareArgs& args,
    const uint64_t opCount,
    ResourceRef& resRef,
    PersistConfig& config,
    CtranComm* comm,
    std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    utils::TraceLogger* ctran_trace_logger) {
  auto op = std::unique_ptr<struct OpElem>(
      new OpElem(OpElem::opType::ALLTOALLV_DEDUP_PREPARE, comm, opCount));
  op->alltoallv_dedup_prepare.pArgs =
      const_cast<void*>(reinterpret_cast<const void*>(&pArgs));
  op->alltoallv_dedup_prepare.algoResource = &resRef;
  op->alltoallv_dedup_prepare.algoConfig = &config;
  op->alltoallv_dedup_prepare.xnodeInputSplits = args.xnodeInputSplits;
  op->alltoallv_dedup_prepare.xnodeOutputSplits = args.xnodeOutputSplits;
  op->alltoallv_dedup_prepare.localInputSplits = args.localInputSplits;
  op->alltoallv_dedup_prepare.localOutputSplits = args.localOutputSplits;
  op->alltoallv_dedup_prepare.localGatherIndices = args.localGatherIndices;
  op->alltoallv_dedup_prepare.xnodeGatherIndices = args.xnodeGatherIndices;
  op->alltoallv_dedup_prepare.eGatherIndices = args.eGatherIndices;
  op->alltoallv_dedup_prepare.ctran_trace_logger =
      reinterpret_cast<void*>(ctran_trace_logger);

  opGroup.push_back(std::move(op));
}

inline void computeNumSendBlocks(PrepareCtx& ctx) {
  auto& resRef = ctx.resource;
  auto& pArgs = ctx.pArgs;
  const auto& statex = ctx.commStatex;

  const int nNodes = statex->nNodes();
  const int nLocalRanks = statex->nLocalRanks();
  const int nRanks = statex->nRanks();
  const int myNode = statex->node();

  // Compute how many blocks will be sent to other ranks
  int* blockRecvBucketsH = nullptr;
  GET_RESOURCE_BUFPTR(resRef, kBlockRecvBucketsH, blockRecvBucketsH);
  blockRecvBucketsH +=
      myNode * pArgs.totalNumSendBlocks * pArgs.blockNumRecvBuckets;

  size_t* tmpNumSendBlocksBuffH = nullptr;
  GET_RESOURCE_BUFPTR(resRef, kTmpNumSendBlocksBuffH, tmpNumSendBlocksBuffH);

  TRACE_LOG(
      "Rank {} loaded blockRecvBucketsH {}: {}",
      statex->rank(),
      (void*)blockRecvBucketsH,
      array2DToStr(
          blockRecvBucketsH,
          pArgs.totalNumSendBlocks,
          pArgs.blockNumRecvBuckets));

  const auto nLocalBuckets = pArgs.numRecvBuckets * nLocalRanks;
  const auto tmpNumSendBlocks = ptrElemOffset<size_t>(
      tmpNumSendBlocksBuffH,
      myNode * nNodes * (nLocalBuckets + nLocalRanks + 1));
  memset(
      tmpNumSendBlocks,
      0,
      nNodes * (nLocalBuckets + nLocalRanks + 1) * sizeof(size_t));

  for (int b = 0; b < pArgs.totalNumSendBlocks; b++) {
    std::vector<bool> numSendBlocksToNode(nNodes, 0);
    std::vector<bool> numSendBlocksToRank(nRanks, 0);

    for (int r = 0; r < pArgs.blockNumRecvBuckets; r++) {
      int recvBucket = blockRecvBucketsH[b * pArgs.blockNumRecvBuckets + r];
      int recvRank = bucketToRank(pArgs, recvBucket);
      int recvLocalBucket = recvBucket & (nLocalBuckets - 1);
      int nodeId = statex->node(recvRank);
      const auto offset = nodeId * (nLocalBuckets + nLocalRanks + 1);

      // tmpNumSendBlocks counts the number of blocks sent to each rank in the
      // first nLocalRanks elems of each node and counts the number of unique
      // blocks per node in the last element
      tmpNumSendBlocks[offset + recvLocalBucket]++;

      // Count block once per node
      if (!numSendBlocksToNode[nodeId]) {
        tmpNumSendBlocks[offset + nLocalBuckets + nLocalRanks]++;
        numSendBlocksToNode[nodeId] = true;
      }

      // Count block once per rank
      if (!numSendBlocksToRank[recvRank]) {
        int localRecvRank = statex->localRank(recvRank);
        tmpNumSendBlocks[offset + nLocalBuckets + localRecvRank]++;
        numSendBlocksToRank[recvRank] = true;
      }
    }
  }

  TRACE_LOG(
      "Rank {} computed tmpNumSendBlocks ({}): {}",
      statex->rank(),
      reinterpret_cast<void*>(tmpNumSendBlocks),
      array2DToStr(
          tmpNumSendBlocks, nNodes, (nLocalBuckets + nLocalRanks + 1)));

#ifdef WITH_PYTORCH_METADATA
  // Update xnodeInputSplits based on locally accumulated tmpNumSendBlocks
  const auto myLocalRank = statex->localRank();
  auto& xnodeInputSplits = ctx.xnodeInputSplits;
  memset(xnodeInputSplits, 0, nRanks * sizeof(int));
  for (auto n = 0; n < nNodes; n++) {
    const auto nodeOffset = n * (nLocalBuckets + nLocalRanks + 1);
    auto xnodePeerRank = statex->localRankToRank(myLocalRank, n);
    xnodeInputSplits[xnodePeerRank] =
        tmpNumSendBlocks[nodeOffset + nLocalBuckets + nLocalRanks];
  }
  TRACE_LOG(
      "Rank {} computed xnodeInputSplits {}",
      statex->rank(),
      array2DToStr(xnodeInputSplits, nNodes, nLocalRanks));
#endif
}

inline void computeNumForwardBlocks(PrepareCtx& ctx) {
  size_t* tmpNumSendBlocksBuffH = nullptr;
  GET_RESOURCE_BUFPTR(
      ctx.resource, kTmpNumSendBlocksBuffH, tmpNumSendBlocksBuffH);
  size_t* numForwardBlocksH = nullptr;
  GET_RESOURCE_BUFPTR(ctx.resource, kNumForwardBlocksH, numForwardBlocksH);

  const auto& statex = ctx.commStatex;
  const int nNodes = statex->nNodes();
  const int nLocalRanks = statex->nLocalRanks();
  const int myNode = statex->node();
  const int myLocalRank = statex->localRank();
  const int nRanks = statex->nRanks();

  auto& xnodeOutputSplits = ctx.xnodeOutputSplits;
  auto& localInputSplits = ctx.localInputSplits;
  memset(xnodeOutputSplits, 0, nRanks * sizeof(int));
  memset(localInputSplits, 0, nRanks * sizeof(int));

  const auto nLocalBuckets = ctx.pArgs.numRecvBuckets * nLocalRanks;
  const auto countPerRank = nNodes * (nLocalBuckets + nLocalRanks + 1);
  memset(numForwardBlocksH, 0, nRanks * sizeof(size_t));

  for (int i = 0; i < nNodes; i++) {
    const auto peerTmpNumSendBlocks =
        ptrElemOffset<size_t>(tmpNumSendBlocksBuffH, i * countPerRank);
    const auto myNodeOffset = myNode * (nLocalBuckets + nLocalRanks + 1);

    // number of blocks forwarded from peer node in my rail
    const auto numForwardBlocksFromNode =
        peerTmpNumSendBlocks[myNodeOffset + nLocalBuckets + nLocalRanks];

#ifdef WITH_PYTORCH_METADATA
    // Below updates metadata for Pytorch combine.
    // Update xnodeOutputSplits, localOutputSplits from tmpNumSendBlocksBuffH
    // received from rail peers
    const auto xnodePeerRank = statex->localRankToRank(myLocalRank, i);
    xnodeOutputSplits[xnodePeerRank] = numForwardBlocksFromNode;
    for (auto r = 0; r < nLocalRanks; r++) {
      const auto localPeerRank = statex->localRankToRank(r, myNode);
      localInputSplits[localPeerRank] +=
          peerTmpNumSendBlocks[myNodeOffset + nLocalBuckets + r];
    }
#endif

    // Below updates metadata for custom kernel exec progress.
    // Do not include intra-node forwarding; progress tracked via numSendBlocks
    // in exec.
    if (i == myNode) {
      continue;
    }
    const auto peerNodeFwdOffset = i * nLocalRanks;
    const auto myNodeFwdOffset = myNode * nLocalRanks;
    // accumulate number of blocks forwarded to local ranks from remote nodes
    for (int r = 0; r < nLocalBuckets; r++) {
      numForwardBlocksH[myNodeFwdOffset + bucketToRank(ctx.pArgs, r)] +=
          peerTmpNumSendBlocks[myNodeOffset + r];
    }

    numForwardBlocksH[peerNodeFwdOffset + myLocalRank] =
        numForwardBlocksFromNode;
  }

  TRACE_LOG(
      "Rank {} computed numForwardBlocksH: {}",
      statex->rank(),
      array2DToStr(numForwardBlocksH, nNodes, nLocalRanks));

#ifdef WITH_PYTORCH_METADATA
  TRACE_LOG(
      "Rank {} tmpNumSendBlocksBuffH: {}",
      statex->rank(),
      array2DToStr(
          tmpNumSendBlocksBuffH,
          nNodes * nNodes,
          nLocalBuckets + nLocalRanks + 1,
          nNodes * nNodes,
          nLocalBuckets + nLocalRanks + 1));

  TRACE_LOG(
      "Rank {} computed xnodeOutputSplits: {}",
      statex->rank(),
      array2DToStr(xnodeOutputSplits, nNodes, nLocalRanks));
  TRACE_LOG(
      "Rank {} computed localInputSplits: {}",
      statex->rank(),
      array2DToStr(localInputSplits, nNodes, nLocalRanks));
#endif
}

inline void postKernSync(PrepareCtx& ctx, const PrepareSyncStep step) {
  const auto& statex = ctx.commStatex;
  ctx.resource->prepareGKSync->post((int)step);
  TRACE_LOG(
      "Rank {} posted prepareGKSync {} {}",
      statex->rank(),
      (void*)ctx.resource->prepareGKSync,
      (int)step);
}

inline void waitKernSync(PrepareCtx& ctx, const PrepareSyncStep step) {
  const auto& statex = ctx.commStatex;
  ctx.resource->prepareGKSync->waitComplete((int)step);
  TRACE_LOG(
      "Rank {} waited prepareGKSync {} to post {}",
      statex->rank(),
      (void*)ctx.resource->prepareGKSync,
      (int)step);
}

void updateLocalOutputSplits(PrepareCtx& ctx) {
  const auto& statex = ctx.commStatex;
  auto& resRef = ctx.resource;

  waitKernSync(ctx, PrepareSyncStep::kCopyNumRecvBlocksH);

  size_t* tmpNumRecvBlocksBuffH = nullptr;
  GET_RESOURCE_BUFPTR(resRef, kTmpNumRecvBlocksBuffH, tmpNumRecvBlocksBuffH);

  const auto nNodes = statex->nNodes();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto nRanks = statex->nRanks();
  const auto myNode = statex->node();

  TRACE_LOG(
      "Rank {} received tmpNumRecvBlocksBuffH {}: {}",
      statex->rank(),
      (void*)tmpNumRecvBlocksBuffH,
      array2DToStr(
          tmpNumRecvBlocksBuffH,
          1,
          (ctx.pArgs.numRecvBuckets * nRanks + nRanks),
          1,
          (ctx.pArgs.numRecvBuckets * nRanks + nRanks)));

  auto& localOutputSplits = ctx.localOutputSplits;
  memset(localOutputSplits, 0, nRanks * sizeof(int));
  const auto nLocalBuckets = ctx.pArgs.numRecvBuckets * nLocalRanks;
  const auto offset = nNodes * nLocalBuckets;
  for (auto i = 0; i < nLocalRanks; i++) {
    const auto fwdRank = statex->localRankToRank(i, myNode);
    for (auto n = 0; n < nNodes; n++) {
      const auto sendRank = statex->localRankToRank(i, n);
      localOutputSplits[fwdRank] += tmpNumRecvBlocksBuffH[offset + sendRank];
    }
  }

  TRACE_LOG(
      "Rank {} computed localOutputSplits: {}",
      statex->rank(),
      array2DToStr(localOutputSplits, nNodes, nLocalRanks));
}

commResult_t gpeFn(const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = opGroup.front()->comm_;
  thOpCount = op->opCount;

  const auto& algoConfig =
      reinterpret_cast<PersistConfig*>(op->alltoallv_dedup_prepare.algoConfig);

  auto statex = comm->statex_.get();
  const auto myRank = statex->rank();

  auto ctran_trace_logger = reinterpret_cast<utils::TraceLogger*>(
      op->alltoallv_dedup_prepare.ctran_trace_logger);
  // Always create traceRecord for code simplicity, all recording should be
  // no-op if trace is disabled
  auto ts = std::make_unique<utils::TraceRecord>(
      fmt::format("allToAllvDedupPrepare_{}", thOpCount), myRank);
  ts->addMetadata("opCount", std::to_string(thOpCount));
  ts->addMetadata("rank", std::to_string(myRank));
  ts->addMetadata("localRank", std::to_string(statex->localRank()));
  ts->addMetadata("numRanks", std::to_string(statex->nRanks()));
  ts->addMetadata("numNodes", std::to_string(statex->nNodes()));
  ts->addMetadata("numLocalRanks", std::to_string(statex->nLocalRanks()));
  ts->addMetadata(
      "numPrepareThreadBlocks",
      std::to_string(algoConfig->numPrepareThreadBlocks));
  ts->addMetadata(
      "numPrepareThreads", std::to_string(algoConfig->numPrepareThreads));

  CtranAlgoLogger logger(algoName, op->opCount, comm);

  CtranMapper* mapper = comm->ctran_->mapper.get();

  const int myNode = statex->node();
  const int nNodes = statex->nNodes();
  const int localRank = statex->localRank();
  const int nLocalRanks = statex->nLocalRanks();
  const int myLocalRank = statex->localRank();

  PrepareCtx ctx = {
      .pArgs =
          *reinterpret_cast<PersistArgs*>(op->alltoallv_dedup_prepare.pArgs),
      .resource = reinterpret_cast<ResourceRef*>(
          op->alltoallv_dedup_prepare.algoResource),
      .commStatex = statex,
      .mapper = mapper,
      .ts = ts.get(),
      .xnodeInputSplits =
          reinterpret_cast<int*>(op->alltoallv_dedup_prepare.xnodeInputSplits),
      .xnodeOutputSplits =
          reinterpret_cast<int*>(op->alltoallv_dedup_prepare.xnodeOutputSplits),
      .localInputSplits =
          reinterpret_cast<int*>(op->alltoallv_dedup_prepare.localInputSplits),
      .localOutputSplits =
          reinterpret_cast<int*>(op->alltoallv_dedup_prepare.localOutputSplits),
      .xnodeGatherIndices = reinterpret_cast<int64_t*>(
          op->alltoallv_dedup_prepare.xnodeGatherIndices),
      .localGatherIndices = reinterpret_cast<int64_t*>(
          op->alltoallv_dedup_prepare.localGatherIndices),
      .eGatherIndices = reinterpret_cast<int64_t*>(
          op->alltoallv_dedup_prepare.eGatherIndices),
  };

  ts->startInterval("intranode barrier + copyblockrecvbuckets", op->opCount, 0);
  // Wait for kernel to copy blockRecvBucketsH from device to host
  waitKernSync(ctx, PrepareSyncStep::kCopyBlockRecvBuckets);
  ts->endInterval("intranode barrier + copyblockrecvbuckets", op->opCount);

  ts->startInterval("computeNumSendBlocks", op->opCount, 0);
  computeNumSendBlocks(ctx);
  ts->endInterval("computeNumSendBlocks", op->opCount);

  ts->startInterval("peerExchange", op->opCount, 0);
  auto& tmpNumSendBlocksBuffH =
      ctx.resource->getBuf(ResourceBufName::kTmpNumSendBlocksBuffH);
  auto& tmpRemNumSendBlocksBuffsH =
      ctx.resource->getRemBufs(ResourceBufName::kTmpNumSendBlocksBuffH);

  const auto nLocalBuckets = ctx.pArgs.numRecvBuckets * nLocalRanks;
  const auto countPerRank = nNodes * (nLocalBuckets + nLocalRanks + 1);
  const auto myTmpNumSendBlocks =
      ptrElemOffset<size_t>(tmpNumSendBlocksBuffH.ptr, myNode * countPerRank);

  auto& tmpBlockRecvBucketsH =
      ctx.resource->getBuf(ResourceBufName::kBlockRecvBucketsH);
  auto& tmpRemBlockRecvBucketssH =
      ctx.resource->getRemBufs(ResourceBufName::kBlockRecvBucketsH);
  const auto blockRecvBucketsCountPerRank =
      ctx.pArgs.totalNumSendBlocks * ctx.pArgs.blockNumRecvBuckets;
  const auto myTmpBlockRecvBuckets = ptrElemOffset<int>(
      tmpBlockRecvBucketsH.ptr, myNode * blockRecvBucketsCountPerRank);

  std::vector<std::unique_ptr<CtranMapperRequest>> pReqs, blockRecvBucketsPReqs,
      sReqs, rReqs;
  std::vector<CtranMapperNotify> notifyVec(nNodes, CtranMapperNotify());
  std::vector<CtranMapperNotify> blockRecvBucketsNotifyVec(
      nNodes, CtranMapperNotify());
  pReqs.reserve(nNodes);
  blockRecvBucketsPReqs.reserve(nNodes);
  sReqs.reserve(nNodes);
  rReqs.reserve(nNodes);

  // Ctrl exchange with rail peers -- kernel has already barriered local peers
  // on each node
  for (int n = 1; n < nNodes; n++) {
    int peerNode = (myNode + n) % nNodes;
    int peerRank = statex->localRankToRank(localRank, peerNode);
    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(mapper->isendCtrl(peerRank, &req));
    sReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));

    FB_COMMCHECK(mapper->irecvCtrl(peerRank, &req));
    rReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));

    FB_COMMCHECK(mapper->initNotify(
        peerRank, tmpNumSendBlocksBuffH.regHdl, &notifyVec[peerNode]));

    FB_COMMCHECK(mapper->initNotify(
        peerRank,
        tmpBlockRecvBucketsH.regHdl,
        &blockRecvBucketsNotifyVec[peerNode]));
  }
  CLOGF_TRACE(COLL, "Rank {} exchanged ctrl", myRank);

  // Send numSendBlocks to remote rail peers
  for (int i = 0; i < nNodes - 1; i++) {
    const auto peerNode = (myNode + i + 1) % nNodes;
    const auto peerRank = statex->localRankToRank(myLocalRank, peerNode);

    FB_COMMCHECK(mapper->waitRequest(rReqs[i].get()));
    CtranMapperRequest* req = nullptr;
    auto remBuffH = tmpRemNumSendBlocksBuffsH[peerNode];
    FB_COMMCHECK(mapper->iput(
        myTmpNumSendBlocks,
        reinterpret_cast<size_t*>(remBuffH.ptr) + myNode * countPerRank,
        countPerRank * sizeof(size_t),
        peerRank,
        CtranMapperConfig{
            .memHdl_ = tmpNumSendBlocksBuffH.regHdl,
            .remoteAccessKey_ = remBuffH.rkey,
            .notify_ = true},
        &req));
    pReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
    TRACE_LOG(
        "Rank {} issued put to peerNode {} peerRank {}",
        myRank,
        peerNode,
        peerRank);

#ifdef WITH_PYTORCH_METADATA
    req = nullptr;
    remBuffH = tmpRemBlockRecvBucketssH[peerNode];
    FB_COMMCHECK(mapper->iput(
        myTmpBlockRecvBuckets,
        reinterpret_cast<int*>(remBuffH.ptr) +
            myNode * blockRecvBucketsCountPerRank,
        blockRecvBucketsCountPerRank * sizeof(int),
        peerRank,
        CtranMapperConfig{
            .memHdl_ = tmpBlockRecvBucketsH.regHdl,
            .remoteAccessKey_ = remBuffH.rkey,
            .notify_ = true},
        &req));
    blockRecvBucketsPReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
    TRACE_LOG(
        "Rank {} issued put to peerNode {} peerRank {} myTmpBlockRecvBuckets {}",
        myRank,
        peerNode,
        peerRank,
        array2DToStr(
            myTmpBlockRecvBuckets,
            1,
            ctx.pArgs.totalNumSendBlocks * ctx.pArgs.blockNumRecvBuckets));
#endif
  }

  for (int i = 0; i < nNodes - 1; i++) {
    const auto peerNode = (myNode + i + 1) % nNodes;
    FB_COMMCHECK(mapper->waitNotify(&notifyVec[peerNode]));

    const auto peerRank = statex->localRankToRank(myLocalRank, peerNode);
    const auto peerTmpNumSendBlocks = ptrElemOffset<size_t>(
        tmpNumSendBlocksBuffH.ptr, peerNode * countPerRank);
    TRACE_LOG(
        "Rank {} received from peerNode {} peerRank {} peerTmpNumSendBlocks: {}",
        myRank,
        peerNode,
        peerRank,
        array2DToStr(
            peerTmpNumSendBlocks, nNodes, nLocalBuckets + nLocalRanks + 1));
  }

#ifdef WITH_PYTORCH_METADATA
  for (int i = 0; i < nNodes - 1; i++) {
    const auto peerNode = (myNode + i + 1) % nNodes;
    const auto peerRank = statex->localRankToRank(myLocalRank, peerNode);
    FB_COMMCHECK(mapper->waitNotify(&blockRecvBucketsNotifyVec[peerNode]));

    const auto peerTmpBlockRecvBuckets = ptrElemOffset<int>(
        tmpBlockRecvBucketsH.ptr, peerNode * blockRecvBucketsCountPerRank);
    TRACE_LOG(
        "Rank {} received from peerNode {} peerRank {} peerTmpBlockRecvBuckets: {}",
        myRank,
        peerNode,
        peerRank,
        array2DToStr(
            peerTmpBlockRecvBuckets,
            ctx.pArgs.totalNumSendBlocks,
            ctx.pArgs.blockNumRecvBuckets,
            ctx.pArgs.totalNumSendBlocks,
            ctx.pArgs.blockNumRecvBuckets));
  }
#endif

  ts->endInterval("peerExchange", op->opCount);

  // Notify kernel to forward tmpNumSendBlocksBuffH to device and forward to
  // rest of local ranks for numRecvCounts and recvOffsets calculation
  postKernSync(ctx, PrepareSyncStep::kPostTmpNumSendBlocksBuff);
  ts->startInterval("computeNumForwardBlocks", op->opCount, 0);
  computeNumForwardBlocks(ctx);
  ts->endInterval("computeNumForwardBlocks", op->opCount);

  // Notify kernel to copy numForwardBlocksH to device
  postKernSync(ctx, PrepareSyncStep::kPostNumForwardBlocks);

#ifdef WITH_PYTORCH_METADATA
  // Wait kernel to copy numRecvBlocks to host on receive rank and update
  // localOutputSplits
  ts->startInterval("updateLocalOutputSplits", op->opCount, 0);
  updateLocalOutputSplits(ctx);
  ts->endInterval("updateLocalOutputSplits", op->opCount);
#endif

  ts->startInterval("kernelDone", op->opCount, 0);
  waitKernSync(ctx, PrepareSyncStep::kKernelDone);
  ts->endInterval("kernelDone", op->opCount);

  // Wait for all requests to complete
  for (auto& req : pReqs) {
    if (req) {
      FB_COMMCHECK(mapper->waitRequest(req.get()));
    }
  }
  for (auto& req : blockRecvBucketsPReqs) {
    if (req) {
      FB_COMMCHECK(mapper->waitRequest(req.get()));
    }
  }
  for (auto& req : sReqs) {
    if (req) {
      FB_COMMCHECK(mapper->waitRequest(req.get()));
    }
  }

  ctran_trace_logger->addTraceRecord(std::move(ts));
  return commSuccess;
}

} // namespace

extern __global__ void ncclKernelAllToAllvDedupPrepare(
    int* flag,
    CtranAlgoDeviceState* devState,
    PrepareKernArgs args);

commResult_t AlgoImpl::prepare(
    const PrepareArgs& args,
    const uint64_t opCount) {
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALLV_DEDUP_PREPARE,
      stream_,
      algoName(Phase::kPrepare),
      opCount);

  auto& resRef = resource_->getRef();
  PrepareKernArgs kernArgs;
  setupKernelArgs_(pArgs, config_, args, statex_, resRef, kernArgs);
  // TODO: opCount should be passed to all kernels
  kernArgs.opCount = opCount;
  setupKernelConfig_(ctran_, kernArgs, config_, config);

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  setupGpeOp_(
      pArgs,
      args,
      opCount,
      resRef,
      config_,
      comm_,
      opGroup,
      ctran_trace_logger.get());

  FB_COMMCHECK(ctran_->gpe->submit(
      std::move(opGroup),
      gpeFn,
      config,
      reinterpret_cast<void*>(ncclKernelAllToAllvDedupPrepare)));

  return commSuccess;
}
} // namespace ctran::alltoallvdedup
