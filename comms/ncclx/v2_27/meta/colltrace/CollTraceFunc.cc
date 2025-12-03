// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/colltrace/CollTraceFunc.h"

#include <chrono>
#include <vector>

#include <folly/logging/xlog.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/trainer/TrainerContext.h"
#include "meta/colltrace/CollTraceColl.h"
#include "meta/colltrace/CollTraceLegacyHandle.h"
#include "meta/colltrace/CollTraceWrapper.h"

#include "meta/logger/DebugExt.h"
#include "meta/wrapper/MetaFactory.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_COLLTRACE_CTRAN_USE_CPU_RECORD
   type        : bool
   default     : true
   description : |-
     For all Ctran collective kernels, use CPU record instead of GPU record.
     This is to avoid the GPU record overhead for kernels.
     One exception is for collectives with checkSum Kernel, because we need
     to wait for the checkSum Kernel to finish in order to get the checksum.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

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

CpuWaitEvent* getCpuWaitEventOrNull(CollWaitEvent* event) {
  if (event == nullptr) {
    return nullptr;
  }
  if (typeid(*event) == typeid(CpuWaitEvent)) {
    return static_cast<CpuWaitEvent*>(event);
  }
  return nullptr;
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

bool shouldUseCPURecord(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    const bool ifchecksum) {
  // For CPU Collectives, we always use CPU record
  if (!kernelConfig.isDevice) {
    return true;
  }
  // For CUDA Graph captured kernels, we always use CPU record
  if (isCapturingStream(kernelConfig.stream)) {
    return true;
  }
  // If NCCL_COLLTRACE_CTRAN_USE_CPU_RECORD is false, we will use GPU record
  // for all the GPU collectives
  if (!NCCL_COLLTRACE_CTRAN_USE_CPU_RECORD) {
    return false;
  }
  // If we have checkSum kernel or opGroup is empty, we need to use GPU record
  // For checkSum kernel, the flag gpe thread is waiting is not the end of the
  // checkSum kernel.
  // If opGroup is empty, there will not be any GPE ops so we need to use GPU
  // record.
  if (ifchecksum || opGroup.empty()) {
    return false;
  }
  // Otherwise, we will use CPU record
  return true;
}

} // namespace

ncclResult_t collTraceInit(ncclComm* comm) {
  // Do not init if using new colltrace
  if (NCCL_COLLTRACE.empty() || NCCL_COLLTRACE_USE_NEW_COLLTRACE) {
    return ncclSuccess;
  }
  comm->collTrace = std::make_shared<CollTrace>(comm);
  return ncclSuccess;
}

ncclResult_t collTraceDestroy(ncclComm* comm) {
  if (comm->collTrace == nullptr) {
    return ncclSuccess;
  }
  comm->collTrace.reset();
  return ncclSuccess;
}

std::unique_ptr<CollTraceEvent> collTraceAquireEventCommon(
    CtranComm* comm,
    CollTraceEvent::EventType type) {
  if (!comm->collTrace_) {
    return nullptr;
  }
  auto event = comm->collTrace_->createEvent(type);
  if (!event) {
    throw CollTraceError("Event init failed");
  }
  return event;
}

std::unique_ptr<CollTraceEvent> collTraceAquireEventBaseline(
    ncclKernelPlan* plan,
    cudaStream_t stream) {
  auto collOpt = parseCollInfoFromNcclKernelPlan(*plan, stream);
  if (!collOpt.has_value()) {
    return nullptr;
  }
  auto comm = plan->comm->ctranComm_.get();
  if (!comm->collTrace_) {
    return nullptr;
  }

  if (isCapturingStream(stream) && !NCCL_COLLTRACE_TRACE_CUDA_GRAPH) {
    XLOG_FIRST_N(INFO, 1) << "Skip tracing CUDA graph capture by env var";
    return nullptr;
  }

  std::unique_ptr<CollTraceEvent> event = nullptr;

  if (isCapturingStream(stream)) {
    event =
        collTraceAquireEventCommon(comm, CollTraceEvent::EventType::COMM_CPU);
    if (event != nullptr) {
      event->isGraphCapture = true;
    }
  } else {
    event = collTraceAquireEventCommon(comm, CollTraceEvent::EventType::COMM);
  }
  if (event == nullptr) {
    return nullptr;
  }
  event->coll = collOpt.value();
  return event;
}

bool collTraceRecordCtranKernelInfo(
    CollTraceColl& coll,
    const KernelConfig& kernelConfig) {
  coll.ctranAttr = CollCtranAttr{
      .nBlocks = kernelConfig.numBlocks,
  };
  coll.nThreads = kernelConfig.numThreads;
  coll.algoName = kernelConfig.algoName;

  switch (kernelConfig.type) {
    // Currently there is no information from send/recv and put/wait kernel
    case KernelConfig::KernelType::SEND:
      coll.opName = "Send";
      break;
    case KernelConfig::KernelType::RECV:
      coll.opName = "Recv";
      break;
    case KernelConfig::KernelType::SENDRECV:
      coll.opName = "SendRecv";
      break;
    case KernelConfig::KernelType::SEND_NOTIFY:
      coll.opName = "SendNotify";
      break;
    case KernelConfig::KernelType::RECV_NOTIFY:
      coll.opName = "RecvNotify";
      break;
    case KernelConfig::KernelType::SENDRECV_NOTIFY:
      coll.opName = "SendRecvNotify";
      break;
    case KernelConfig::KernelType::RECV_UNPACK:
      coll.opName = "RecvUnpack";
      break;
    case KernelConfig::KernelType::SENDRECV_UNPACK:
      coll.opName = "SendRecvUnpack";
      break;
    case KernelConfig::KernelType::ALLGATHERP_INIT:
      coll.opName = "AllGatherPInit";
      break;
    case KernelConfig::KernelType::ALLGATHERP:
      coll.opName = "AllGatherP";
      // FIXME: Temporarily removed the arguments record in
      // CollTrace for AllGatherP, since 1) persistent collective would store
      // argument in separate `struct PersistArg` which may not be passed in
      // kernelConfig, and 2) persistent collective may involve multiple times
      // submit, and transparantly record per submit doesn't fit.
      break;
    case KernelConfig::KernelType::PUTNOTIFY:
      coll.opName = "PutNotify";
      break;
    case KernelConfig::KernelType::WAITNOTIFY:
      coll.opName = "WaitNotify";
      break;
    case KernelConfig::KernelType::PUTSIGNAL:
      coll.opName = "PutSignal";
      break;
    case KernelConfig::KernelType::WAITSIGNAL:
      coll.opName = "WaitSignal";
      break;
    case KernelConfig::KernelType::SIGNAL:
      coll.opName = "Signal";
      break;
    case KernelConfig::KernelType::GET:
      coll.opName = "Get";
      break;
    case KernelConfig::KernelType::ALLGATHER: {
      coll.opName = "AllGather";
      auto allGatherArgs = kernelConfig.args.collective.allgather;
      coll.sendbuff = allGatherArgs.sendbuff;
      coll.recvbuff = allGatherArgs.recvbuff;
      coll.dataType = metaCommToNccl(allGatherArgs.datatype);
      coll.count = allGatherArgs.count;
      break;
    }
    case KernelConfig::KernelType::REDUCESCATTER: {
      coll.opName = "ReduceScatter";
      auto reduceScatterArgs = kernelConfig.args.collective.reducescatter;
      coll.sendbuff = reduceScatterArgs.sendbuff;
      coll.recvbuff = reduceScatterArgs.recvbuff;
      coll.dataType = metaCommToNccl(reduceScatterArgs.datatype);
      coll.count = reduceScatterArgs.recvcount;
      break;
    }
    case KernelConfig::KernelType::ALLREDUCE: {
      coll.opName = "AllReduce";
      auto allReduceArgs = kernelConfig.args.collective.allreduce;
      coll.sendbuff = allReduceArgs.sendbuff;
      coll.recvbuff = allReduceArgs.recvbuff;
      coll.dataType = metaCommToNccl(allReduceArgs.datatype);
      coll.count = allReduceArgs.count;
      break;
    }
    case KernelConfig::KernelType::ALLTOALL: {
      coll.opName = "AllToAll";
      auto allToAllArgs = kernelConfig.args.collective.alltoall;
      coll.sendbuff = allToAllArgs.sendbuff;
      coll.recvbuff = allToAllArgs.recvbuff;
      coll.dataType = metaCommToNccl(allToAllArgs.datatype);
      coll.count = allToAllArgs.count;
      break;
    }
    case KernelConfig::KernelType::ALLTOALLV: {
      coll.opName = "AllToAllV";
      auto allToAllvArgs = kernelConfig.args.collective.alltoallv;
      coll.sendbuff = allToAllvArgs.sendbuff;
      coll.recvbuff = allToAllvArgs.recvbuff;
      coll.dataType = metaCommToNccl(allToAllvArgs.datatype);
      // Explicitly leave count as nullopt because there is no single count for
      // AllToAllV
      break;
    }
    case KernelConfig::KernelType::ALLTOALL_DEDUP: {
      coll.opName = "AllToAllDedup";
      break;
    }
    case KernelConfig::KernelType::ALLTOALLV_DYNAMIC: {
      coll.opName = "AllToAllvDynamic";
      break;
    }
    case KernelConfig::KernelType::ALLTOALLV_DYNAMIC_SPLIT: {
      coll.opName = "AllToAllvDynamicSplit";
      break;
    }
    case KernelConfig::KernelType::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG: {
      coll.opName = "AllToAllvDynamicSplitNonContig";
      break;
    }
    case KernelConfig::KernelType::ALLTOALLV_DEDUP: {
      coll.opName = "AllToAllvDedup";
      // FIXME: need pass in arguments; a placeholder to pass build for now
      break;
    }
    case KernelConfig::KernelType::BROADCAST_UNPACK:
    case KernelConfig::KernelType::BROADCAST: {
      coll.opName = "Broadcast";
      if (kernelConfig.type == KernelConfig::KernelType::BROADCAST_UNPACK) {
        coll.opName = "BroadcastUnpack";
      }
      auto broadcastArgs = kernelConfig.args.collective.broadcast;
      coll.sendbuff = broadcastArgs.sendbuff;
      coll.recvbuff = broadcastArgs.recvbuff;
      coll.dataType = metaCommToNccl(broadcastArgs.datatype);
      coll.count = broadcastArgs.count;
      break;
    }
  }
  return true;
}

bool collTraceRecordCtranCollective(
    CollTraceColl& coll,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  if (opGroup.size() > 1) {
    WARN_FIRST_N(
        kDebugRepeatLogCount,
        "COLLTRACE: do not support grouped collective. Skip this GPE submit.");
    return false;
  }
  if (opGroup.empty()) {
    return true;
  }
  auto& gpeOp = **opGroup.begin();
  coll.comm = gpeOp.comm_;
  switch (gpeOp.type) {
    case OpElem::SEND:
    case OpElem::RECV:
      WARN_FIRST_N(
          kDebugRepeatLogCount,
          "COLLTRACE: Should not encounter send/recv in collTraceRecordCtranCollective. Encountered internal error. Skip this GPE submit");
      return false;
    case OpElem::ALLGATHER:
      coll.opName = "AllGather";
      coll.sendbuff = gpeOp.allgather.sendbuff;
      coll.recvbuff = gpeOp.allgather.recvbuff;
      coll.dataType = metaCommToNccl(gpeOp.allgather.datatype);
      coll.count = gpeOp.allgather.sendcount;
      break;
    case OpElem::ALLGATHERP_INIT: {
      coll.opName = "AllGatherPInit";
      auto pArgs = reinterpret_cast<ctran::allgatherp::PersistArgs*>(
          gpeOp.allgatherP.pArgs);
      coll.sendbuff = nullptr;
      coll.recvbuff = pArgs->recvbuff;
      coll.dataType = metaCommToNccl(pArgs->datatype);
      coll.count = pArgs->maxRecvCount;
      break;
    }
    case OpElem::ALLGATHERP: {
      coll.opName = "AllGatherP";
      auto pArgs = reinterpret_cast<ctran::allgatherp::PersistArgs*>(
          gpeOp.allgatherP.pArgs);
      coll.sendbuff = gpeOp.allgatherP.sendbuff;
      coll.recvbuff = pArgs->recvbuff;
      coll.dataType = metaCommToNccl(gpeOp.allgatherP.datatype);
      coll.count = gpeOp.allgatherP.count;
      break;
    }
    case OpElem::REDUCESCATTER:
      coll.opName = "ReduceScatter";
      coll.sendbuff = gpeOp.reducescatter.sendbuff;
      coll.recvbuff = gpeOp.reducescatter.recvbuff;
      coll.dataType = metaCommToNccl(gpeOp.reducescatter.datatype);
      coll.count = gpeOp.reducescatter.recvcount;
      break;
    case OpElem::ALLREDUCE:
      coll.opName = "AllReduce";
      coll.sendbuff = gpeOp.allreduce.sendbuff;
      coll.recvbuff = gpeOp.allreduce.recvbuff;
      coll.dataType = metaCommToNccl(gpeOp.allreduce.datatype);
      coll.count = gpeOp.allreduce.count;
      break;
    case OpElem::ALLTOALL:
      coll.opName = "AllToAll";
      coll.sendbuff = gpeOp.alltoall.sendbuff;
      coll.recvbuff = gpeOp.alltoall.recvbuff;
      coll.dataType = metaCommToNccl(gpeOp.alltoall.datatype);
      coll.count = gpeOp.alltoall.count;
      break;
    case OpElem::ALLTOALLV:
      coll.opName = "AllToAllV";
      coll.sendbuff = gpeOp.alltoallv.sendbuff;
      coll.recvbuff = gpeOp.alltoallv.recvbuff;
      coll.dataType = metaCommToNccl(gpeOp.alltoallv.datatype);
      // Explicitly leave count as nullopt because there is no single count for
      // AllToAllV
      break;
    case OpElem::ALLTOALLP: {
      coll.opName = "AllToAllP";
      auto pArgs = reinterpret_cast<ctran::alltoallp::PersistArgs*>(
          gpeOp.alltoallP.pArgs);
      coll.sendbuff = gpeOp.alltoallP.sendbuff;
      coll.recvbuff = pArgs->recvbuff;
      coll.dataType = metaCommToNccl(pArgs->datatype);
      coll.count = gpeOp.alltoallP.count;
      break;
    }
    case OpElem::ALLTOALL_DEDUP:
      coll.opName = "AllToAllDedup";
      coll.sendbuff = gpeOp.alltoall_dedup.sendbuff;
      coll.recvbuff = gpeOp.alltoall_dedup.recvbuff;
      coll.dataType = metaCommToNccl(gpeOp.alltoall_dedup.datatype);
      // Explicitly leave count as nullopt because there is no single count for
      // AllToAllDedup
      break;
    case OpElem::ALLTOALLV_DYNAMIC:
      coll.opName = "AllToAllvDynamic";
      break;
    case OpElem::ALLTOALLV_DEDUP: {
      coll.opName = "AllToAllvDedup";
      // FIXME: need pass in arguments; a placeholder to pass build for now
      break;
    }
    case OpElem::ALLTOALLV_DYNAMIC_SPLIT:
      coll.opName = "AllToAllvDynamicSplit";
      break;
    case OpElem::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG:
      coll.opName = "AllToAllvDynamicSplitNonContig";
      break;
    case OpElem::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG_P:
      coll.opName = "AllToAllvDynamicSplitNonContigP";
      break;
    case OpElem::BROADCAST:
      coll.opName = "Broadcast";
      coll.sendbuff = gpeOp.broadcast.sendbuff;
      coll.recvbuff = gpeOp.broadcast.recvbuff;
      coll.dataType = metaCommToNccl(gpeOp.broadcast.datatype);
      coll.count = gpeOp.broadcast.count;
      break;
    case OpElem::PUTNOTIFY:
      coll.opName = "PutNotify";
      coll.dataType = metaCommToNccl(gpeOp.putnotify.datatype);
      break;
    case OpElem::WAITNOTIFY:
      // Due to the one-sided nature of waitnotify, we don't have the datatype
      // and count
      coll.opName = "WaitNotify";
      break;
    case OpElem::PUTSIGNAL:
      coll.opName = "PutSignal";
      coll.dataType = metaCommToNccl(gpeOp.putsignal.datatype);
      break;
    case OpElem::WAITSIGNAL:
      coll.opName = "WaitSignal";
      break;
    case OpElem::SIGNAL:
      coll.opName = "Signal";
      break;
    case OpElem::GET:
      coll.opName = "Get";
      break;
  }
  return true;
}

bool collTraceRecordCtranGroupedP2P(
    CollTraceColl& coll,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  int64_t sendOpCount = 0;
  int64_t recvOpCount = 0;
  bool mixedTypes = false;
  ncclDataType_t dataType = ncclNumTypes;
  std::unordered_set<int> ranksInGroupedP2P{};

  std::string commDesc = "unknown";
  if (opGroup.size() > 0) {
    ranksInGroupedP2P.insert((*opGroup.begin())->comm_->statex_->rank());
    commDesc = (*opGroup.begin())->comm_->config_.commDesc;
  }
  for (const auto& opElemPtr : opGroup) {
    const auto& gpeOp = *opElemPtr;
    switch (gpeOp.type) {
      case OpElem::SEND:
        ranksInGroupedP2P.insert(gpeOp.send.peerRank);
        if (coll.sendbuff == nullptr && sendOpCount == 0) {
          coll.sendbuff = gpeOp.send.sendbuff;
        } else if (coll.sendbuff != gpeOp.send.sendbuff) {
          coll.sendbuff = nullptr;
        }
        if (dataType != metaCommToNccl(gpeOp.send.datatype)) {
          if (dataType == ncclNumTypes) {
            dataType = metaCommToNccl(gpeOp.send.datatype);
          } else {
            mixedTypes = true;
          }
        }
        coll.count = coll.count.value_or(0) + gpeOp.send.count;
        sendOpCount++;
        break;
      case OpElem::RECV:
        ranksInGroupedP2P.insert(gpeOp.recv.peerRank);
        if (coll.recvbuff == nullptr && recvOpCount == 0) {
          coll.recvbuff = gpeOp.recv.recvbuff;
        } else if (coll.recvbuff != gpeOp.recv.recvbuff) {
          coll.recvbuff = nullptr;
        }
        if (dataType != metaCommToNccl(gpeOp.recv.datatype)) {
          if (dataType == ncclNumTypes) {
            dataType = metaCommToNccl(gpeOp.recv.datatype);
          } else {
            mixedTypes = true;
          }
        }
        coll.count = coll.count.value_or(0) + gpeOp.recv.count;
        recvOpCount++;
        break;
      default:
        WARN_FIRST_N(
            kDebugRepeatLogCount,
            "COLLTRACE: Should not encounter anything other than OpElem::SEND/RECV in collTraceRecordCtranCollective. Encountered internal error.");
        return false;
    }
  }
  if (ranksInGroupedP2P.size() > 2) {
    WARN_FIRST_N(
        kDebugRepeatLogCount,
        "COLLTRACE: Got %lu ranks in grouped p2p for comm: %s, analysis might be wrong",
        ranksInGroupedP2P.size(),
        commDesc.c_str());
  }
  std::vector<int> ranksInGroupedP2PVec(
      ranksInGroupedP2P.begin(), ranksInGroupedP2P.end());
  std::ranges::sort(ranksInGroupedP2PVec);
  coll.ranksInGroupedP2P = std::move(ranksInGroupedP2PVec);
  // For any grouped p2p, we don't record count.
  if (sendOpCount + recvOpCount > 1) {
    coll.count = std::nullopt;
  }
  if (mixedTypes) {
    INFO(
        NCCL_COLL,
        "COLLTRACE: mixed data types in grouped p2p, datatype is invalid for colltrace record");
    coll.dataType = ncclNumTypes;
  } else {
    coll.dataType = dataType;
  }
  return true;
}

bool collTraceRecordCtranCollInfo(
    CollTraceColl& coll,
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig) {
  bool isSendRecv{false};
  if (!opGroup.empty()) {
    auto& gpeOp = **opGroup.begin();
    isSendRecv = (gpeOp.type == OpElem::RECV || gpeOp.type == OpElem::SEND);
  }
  collTraceRecordCtranKernelInfo(coll, kernelConfig);

  if (kernelConfig.isDevice) {
    coll.codepath = CollTraceColl::Codepath::CTRAN;
  } else {
    coll.codepath = CollTraceColl::Codepath::CTRAN_CPU;
  }

  coll.iteration = ncclxGetIteration();
  if (isSendRecv) {
    coll.opCount = *comm->opCount_; // opCount is incremented after submit
    return collTraceRecordCtranGroupedP2P(coll, opGroup);
  } else {
    coll.opCount = *comm->opCount_ - 1; // opCount is incremented before submit
    return collTraceRecordCtranCollective(coll, opGroup);
  }
}

std::unique_ptr<CollTraceEvent> collTraceAquireEventCtran(
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    const bool ifchecksum) {
  if (comm == nullptr) {
    WARN_FIRST_N(
        kDebugRepeatLogCount,
        "COLLTRACE: comm is null. Skipping this GPE submit");
    return nullptr;
  }

  auto type = CollTraceEvent::EventType::COMM;
  // Set EventType == COMM_CPU in two cases:
  // 1. CPU Collective, where recording CUDA events is simply impossible
  // 2. NCCL_COLLTRACE_CTRAN_USE_CPU_RECORD && current collective doesn't have
  //    checkSum kernel. With checkSum kernel we will always use CUDA events
  if (shouldUseCPURecord(opGroup, kernelConfig, ifchecksum)) {
    type = CollTraceEvent::EventType::COMM_CPU;
  }
  auto event = collTraceAquireEventCommon(comm, type);
  if (event == nullptr) {
    return nullptr;
  }
  event->coll.comm = comm;
  event->coll.logMetaData = comm->logMetaData_;
  event->coll.stream = kernelConfig.stream;
  bool shouldRecord =
      collTraceRecordCtranCollInfo(event->coll, comm, opGroup, kernelConfig);
  return shouldRecord ? std::move(event) : nullptr;
}

ncclResult_t ncclxCudaGraphAddSideNode(
    cudaStream_t stream,
    cudaHostNodeParams hostParams) {
  cudaStreamCaptureStatus status;
  unsigned long long graphId;
  cudaGraph_t graph;
  cudaGraphNode_t const* nodes;
  size_t count = 0;
#if CUDART_VERSION >= 13000
  cudaError_t res = cudaStreamGetCaptureInfo(
      stream, &status, &graphId, &graph, &nodes, nullptr, &count);
#else
  cudaError_t res = cudaStreamGetCaptureInfo_v2(
      stream, &status, &graphId, &graph, &nodes, &count);
#endif
  if (res != cudaSuccess) {
    WARN("Internal error: cudaStreamGetCaptureInfo failed by %d", res);
    return ncclInternalError;
  }
  if (status != cudaStreamCaptureStatusActive) {
    WARN(
        "Internal error: ncclxCudaGraphAddSideNode called while stream is not being captured");
    return ncclInternalError;
  }
  std::vector<cudaGraphNode_t> nodesVec;
  nodesVec.reserve(count);
  for (size_t i = 0; i < count; i++) {
    nodesVec.push_back(nodes[i]);
  }
  cudaGraphNode_t hostNode;
  CUDACHECK(cudaGraphAddHostNode(&hostNode, graph, nodes, count, &hostParams));
  // Change dependencies back
#if CUDART_VERSION >= 13000
  CUDACHECK(cudaStreamUpdateCaptureDependencies(
      stream, nodesVec.data(), nullptr, count, 0));
#else
  CUDACHECK(
      cudaStreamUpdateCaptureDependencies(stream, nodesVec.data(), count, 0));
#endif
  return ncclSuccess;
}

void CUDART_CB graphStartEvent(void* data) {
  auto event = reinterpret_cast<CollTraceEvent*>(data);
  reinterpret_cast<CpuWaitEvent*>(event->start.get())->setFinished();
  reinterpret_cast<CpuWaitEvent*>(event->stop.get())->setNotFinished();
  auto newEvent = std::make_unique<CollTraceEvent>(*event);
  newEvent->coll.iteration = ncclxGetIteration();
  event->coll.comm->collTrace_->enqueueEvent(std::move(newEvent));
}

void CUDART_CB graphEndEvent(void* data) {
  auto event = reinterpret_cast<CollTraceEvent*>(data);
  reinterpret_cast<CpuWaitEvent*>(event->stop.get())->setFinished();
}

ncclResult_t collTraceRecordStartEvent(
    cudaStream_t launchStream,
    CollTraceEvent* event) {
  if (!event) {
    return ncclSuccess;
  }
  if (typeid(*event->stop) == typeid(CudaWaitEvent)) {
    CUDACHECK(cudaEventRecord(
        static_cast<CudaWaitEvent*>(event->start.get())->getCudaEvent(),
        launchStream));
    static_cast<CudaWaitEvent*>(event->start.get())->setStream(launchStream);
  } else if (
      event->isGraphCapture &&
      event->coll.codepath == CollTraceColl::Codepath::BASELINE) {
    ncclxCudaGraphAddSideNode(
        launchStream,
        {
            .fn = graphStartEvent,
            .userData = reinterpret_cast<void*>(event),
        });
  }
  return ncclSuccess;
}

ncclResult_t collTraceRecordEndEvent(
    CtranComm* comm,
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event) {
  if (comm->collTrace_ && event) {
    if (event->isGraphCapture &&
        event->coll.codepath == CollTraceColl::Codepath::BASELINE) {
      ncclxCudaGraphAddSideNode(
          launchStream,
          {
              .fn = graphEndEvent,
              .userData = reinterpret_cast<void*>(event.get()),
          });
      comm->collTrace_->addGraphEvent(std::move(event));
    } else {
      if (typeid(*event->stop) == typeid(CudaWaitEvent)) {
        CUDACHECK(cudaEventRecord(
            static_cast<CudaWaitEvent*>(event->stop.get())->getCudaEvent(),
            launchStream));
        static_cast<CudaWaitEvent*>(event->stop.get())->setStream(launchStream);
      }
      event->coll.enqueueTs = std::chrono::high_resolution_clock::now();
      comm->collTrace_->enqueueEvent(std::move(event));
    }
  }
  return ncclSuccess;
}

ncclResult_t collTraceCtranRecordEndEvent(
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event) {
  if (!event) {
    return ncclSuccess;
  }

  if (typeid(*event->stop) == typeid(CudaWaitEvent)) {
    CUDACHECK(cudaEventRecord(
        static_cast<CudaWaitEvent*>(event->stop.get())->getCudaEvent(),
        launchStream));
    static_cast<CudaWaitEvent*>(event->stop.get())->setStream(launchStream);
  }
  event->coll.enqueueTs = std::chrono::high_resolution_clock::now();
  auto collTrace = event->coll.comm->collTrace_.get();
  collTrace->enqueueEvent(std::move(event));

  return ncclSuccess;
}

ncclResult_t collTraceCtranSubmitEvent(std::unique_ptr<CollTraceEvent> event) {
  if (!event) {
    return ncclSuccess;
  }

  event->coll.enqueueTs = std::chrono::high_resolution_clock::now();
  auto collTrace = event->coll.comm->collTrace_.get();
  collTrace->enqueueEvent(std::move(event));

  return ncclSuccess;
}

CpuWaitEvent* collTraceGetCpuStartWaitEvent(CollTraceEvent& event) {
  return getCpuWaitEventOrNull(event.start.get());
}

CpuWaitEvent* collTraceGetCpuEndWaitEvent(CollTraceEvent& event) {
  return getCpuWaitEventOrNull(event.stop.get());
}

using meta::comms::colltrace::CollTraceLegacyHandle;
using meta::comms::colltrace::ICollTraceHandle;
std::shared_ptr<ICollTraceHandle> collTraceBaselineGetHandle(
    ncclKernelPlan* plan,
    cudaStream_t stream) {
  if (!NCCL_COLLTRACE.empty() && NCCL_COLLTRACE_USE_NEW_COLLTRACE) {
    return meta::comms::ncclx::getHandleFromNcclKernelPlan(*plan, stream);
  }
  return std::make_unique<CollTraceLegacyHandle>(
      plan->comm->ctranComm_.get(),
      collTraceAquireEventBaseline(plan, stream),
      CollTraceLegacyHandle::HandleType::baseline);
}
} // namespace ncclx::colltrace
