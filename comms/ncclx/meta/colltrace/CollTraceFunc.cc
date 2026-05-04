// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/colltrace/CollTraceFunc.h"

#include <folly/logging/xlog.h>

#include "comms/utils/colltrace/DummyCollTraceHandle.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/trainer/TrainerContext.h"
#include "meta/colltrace/CollTraceColl.h"
#include "meta/colltrace/CollTraceWrapper.h"

#include "meta/colltrace/CollTrace.h"
#include "meta/logger/DebugExt.h"

namespace ncclx::colltrace {

namespace {
bool isCapturingStream(cudaStream_t stream) {
  cudaStreamCaptureStatus status;

  auto res = cudaStreamGetCaptureInfo(stream, &status);

  if (res != cudaSuccess) {
    WARN_FIRST_N(
        1, "Internal error: cudaStreamGetCaptureInfo failed by %d", res);
    return false;
  }
  return status != cudaStreamCaptureStatusNone;
}

std::string getAlgoNameFromCollTask(const ncclTaskColl& collTask) {
  return fmt::format(
      "Baseline_{}_{}_{}",
      ncclProtoToString(collTask.protocol),
      ncclAlgoToString(collTask.algorithm),
      collTask.nMaxChannels);
}

std::string
getAlgoNameFromP2PGroup(std::string_view opName, int sendCount, int recvCount) {
  return fmt::format("Baseline_{}_S{}_R{}", opName, sendCount, recvCount);
}

CollTraceColl parseCollInfoFromP2PTasks(
    const ncclTaskP2p& p2pTaskHead,
    int myRank) {
  // Missing: opCount, comm, logMetaData, stream
  // Will add later in this func: opName, algoName, ranksInGroupedP2P
  // Missing inside BaselineAttr: everything
  // Will set in BaselineAttr: coll
  CollTraceColl coll;
  coll.iteration = ncclxGetIteration();
  // Currently do not add the buffer information, as it is not meaningful
  // for grouped send/recv
  coll.sendbuff = std::nullopt;
  coll.recvbuff = std::nullopt;
  coll.count = std::nullopt;
  // Effectively unknown type
  coll.dataType = ncclInt8; // we are counting bytes
  coll.codepath = CollTraceColl::Codepath::BASELINE;
  coll.baselineAttr = CollBaselineAttr{};

  std::set<int> ranksInGroupedP2PSet = {};
  auto sendTaskCount = 0;
  auto recvTaskCount = 0;
  int64_t byteCount = p2pTaskHead.bytes;
  // Root stores the peer rank
  ranksInGroupedP2PSet.insert(myRank);
  ranksInGroupedP2PSet.insert(p2pTaskHead.root);
  if (p2pTaskHead.func == ncclFuncSend) {
    sendTaskCount += 1;
  } else {
    recvTaskCount += 1;
  }

  auto curP2PTask = p2pTaskHead.next;
  while (curP2PTask != nullptr) {
    if (curP2PTask->func == ncclFuncSend) {
      sendTaskCount += 1;
    } else {
      recvTaskCount += 1;
    }
    byteCount += curP2PTask->bytes;
    ranksInGroupedP2PSet.insert(curP2PTask->root);
    curP2PTask = curP2PTask->next;
  }

  if (sendTaskCount > 0 && recvTaskCount > 0) {
    coll.baselineAttr->coll = ncclFuncSendRecv;
  } else if (sendTaskCount > 0) {
    coll.baselineAttr->coll = ncclFuncSend;
  } else {
    coll.baselineAttr->coll = ncclFuncRecv;
  }
  coll.opName = std::string{ncclFuncToString(coll.baselineAttr->coll)};
  coll.algoName =
      getAlgoNameFromP2PGroup(coll.opName, sendTaskCount, recvTaskCount);

  coll.ranksInGroupedP2P = std::vector<int>{
      ranksInGroupedP2PSet.begin(), ranksInGroupedP2PSet.end()};

  coll.count = byteCount;

  return coll;
}

CollTraceColl parseCollInfoFromCollTask(const ncclTaskColl& collTask) {
  // Missing: opCount, comm, logMetaData, stream
  // Missing inside BaselineAttr: pattern, nChannels, channelId
  CollTraceColl collTraceColl;
  collTraceColl.iteration = ncclxGetIteration();
  collTraceColl.opName = std::string{ncclFuncToString(collTask.func)};
  collTraceColl.algoName = getAlgoNameFromCollTask(collTask);
  collTraceColl.sendbuff = collTask.sendbuff;
  collTraceColl.recvbuff = collTask.recvbuff;
  collTraceColl.count = collTask.count;
  collTraceColl.dataType = collTask.datatype;
  collTraceColl.codepath = CollTraceColl::Codepath::BASELINE;
  collTraceColl.baselineAttr = CollBaselineAttr{
      .coll = collTask.func,
      .algorithm = collTask.algorithm,
      .protocol = collTask.protocol,
      .op = collTask.opHost,
      .root = collTask.root,
  };
  return collTraceColl;
}

std::optional<CollTraceColl> parseCollInfoFromNcclKernelPlan(
    ncclKernelPlan& plan,
    cudaStream_t stream) {
  auto collTaskHead = ncclIntruQueueHead(&plan.collTaskQueue);
  auto p2pTaskHead = ncclIntruQueueHead(&plan.p2pTaskQueue);
  // TODO: Limit the frequency of the logging
  if (collTaskHead == nullptr && p2pTaskHead == nullptr) {
    WARN_FIRST_N(
        kDebugRepeatLogCount,
        "CollTrace: no coll or p2p task in this plan, this plan is empty");
    return std::nullopt;
  } else if (collTaskHead != nullptr && collTaskHead->next != nullptr) {
    WARN_FIRST_N(
        kDebugRepeatLogCount,
        "CollTrace: more than one coll task in this plan, this is currently not supported");
    return std::nullopt;
  } else if (collTaskHead != nullptr && p2pTaskHead != nullptr) {
    WARN_FIRST_N(
        kDebugRepeatLogCount,
        "CollTrace: both coll and p2p task in this plan, this is currently not supported");
    return std::nullopt;
  }
  CollTraceColl collInfo = collTaskHead != nullptr
      ? parseCollInfoFromCollTask(*collTaskHead)
      : parseCollInfoFromP2PTasks(*p2pTaskHead, plan.comm->rank);

  // Need to add: opCount, comm, logMetaData, stream
  collInfo.opCount = plan.comm->opCount;
  collInfo.comm = plan.comm->ctranComm_.get();
  collInfo.logMetaData = plan.comm->logMetaData;
  collInfo.stream = stream;

  return collInfo;
}

} // namespace

ncclResult_t collTraceInit(ncclComm* comm) {
  // Do not init if using new colltrace
  if (NCCL_COLLTRACE.empty() || NCCL_COLLTRACE_USE_NEW_COLLTRACE) {
    return ncclSuccess;
  }
  XLOG(
      FATAL,
      "CollTrace is deprecated, please use new colltrace instead by specifying NCCL_COLLTRACE_USE_NEW_COLLTRACE=1");
  return ncclInvalidUsage;
}

ncclResult_t collTraceDestroy(ncclComm* comm) {
  if (comm->collTrace == nullptr) {
    return ncclSuccess;
  }
  comm->collTrace.reset();
  return ncclSuccess;
}

using meta::comms::colltrace::ICollTraceHandle;
std::shared_ptr<ICollTraceHandle> collTraceBaselineGetHandle(
    ncclKernelPlan* plan,
    cudaStream_t stream) {
  // TODO: Remove this guard once v2_27 is deprecated — v2_27's comm.h does not
  // have the algoStats field
#if NCCL_MINOR >= 28
  // Record to standalone AlgoStats (independent of colltrace implementation)
  if (plan->comm->algoStats) {
    auto collOpt = parseCollInfoFromNcclKernelPlan(*plan, stream);
    if (collOpt.has_value()) {
      plan->comm->algoStats->record(collOpt->opName, collOpt->algoName);
    }
  }
#endif

  if (NCCL_COLLTRACE.empty()) {
    return std::make_unique<meta::comms::colltrace::DummyCollTraceHandle>();
  }
  return meta::comms::ncclx::getHandleFromNcclKernelPlan(*plan, stream);
}
} // namespace ncclx::colltrace
