#include "CollTraceFunc.h"

#include <set>

namespace meta::colltrace {

#ifdef BUILD_META_INTERNAL
bool enableGranularScuba() {
  const char* scubaEnable = ncclGetEnv("RCCL_LATENCY_PROFILER_SCUBA");
  if (scubaEnable != NULL) {
    if (strcmp(scubaEnable, "1") == 0) {
      return true;
    }
  }
  return false;
}
#endif

namespace {
bool enableCollTrace() {
  const char* colltraceEnable = ncclGetEnv("RCCL_LATENCY_PROFILER");
  if (colltraceEnable != NULL) {
    INFO(
        NCCL_INIT,
        "RCCL_LATENCY_PROFILER set by environment to %s.",
        colltraceEnable);
    if (strcmp(colltraceEnable, "1") == 0) {
      return true;
    }
  }
  return false;
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

} // namespace

ncclResult_t collTraceInit(ncclComm* comm) {
  if (!enableCollTrace()) {
    return ncclSuccess;
  }
  comm->ctrace = std::make_unique<CollTrace>(comm);
  return ncclSuccess;
}

ncclResult_t collTraceDestroy(ncclComm* comm) {
  if (comm->ctrace == nullptr) {
    return ncclSuccess;
  }
  comm->ctrace.reset();
  return ncclSuccess;
}

ncclResult_t collTraceRecordStartEvent(
    ncclComm* comm,
    cudaStream_t launchStream,
    CollTraceEvent* event) {
  if (comm->ctrace && event) {
    CUDACHECK(
        cudaEventRecord(event->start.get()->getCudaEvent(), launchStream));
  }
  return ncclSuccess;
}

ncclResult_t collTraceRecordEndEvent(
    ncclComm* comm,
    ncclKernelPlan* plan,
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event) {
  if (comm->ctrace && event) {
    CUDACHECK(cudaEventRecord(event->stop.get()->getCudaEvent(), launchStream));
    event->coll.enqueueTs = std::chrono::high_resolution_clock::now();
    comm->ctrace->enqueueEvent(std::move(event));
  }
  return ncclSuccess;
}

CollTraceInfo parseCollInfoFromCollTask(const ncclTaskColl& collTask) {
  return CollTraceInfo{
      .opName = std::string{ncclFuncToString(collTask.func)},
      .dataType = std::string{ncclDatatypeToString(collTask.datatype)},
      .count = (int64_t)collTask.count,
      .algoName = getAlgoNameFromCollTask(collTask),
      .sendbuff = collTask.sendbuff,
      .recvbuff = collTask.recvbuff,
  };
}

CollTraceInfo parseCollInfoFromP2PTasks(
    const ncclTaskP2p& p2pTaskHead,
    int myRank) {
  // Missing: opCount, comm, logMetaData, stream
  // Will add later in this func: opName, algoName, ranksInGroupedP2P
  // Missing inside BaselineAttr: everything
  // Will set in BaselineAttr: coll
  CollTraceInfo coll;
  // Currently do not add the buffer information, as it is not meaningful
  // for grouped send/recv
  coll.sendbuff = std::nullopt;
  coll.recvbuff = std::nullopt;
  coll.count = 0;
  // Effectively unknown type
  coll.dataType = std::string{
      ncclDatatypeToString(ncclDataType_t::ncclInt8)}; // we are counting bytes

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
    coll.opName = std::string{ncclFuncToString(ncclFunc_t::ncclFuncSendRecv)};
  } else if (sendTaskCount > 0) {
    coll.opName = std::string{ncclFuncToString(ncclFunc_t::ncclFuncSend)};
  } else {
    coll.opName = std::string{ncclFuncToString(ncclFunc_t::ncclFuncRecv)};
  }

  coll.algoName =
      getAlgoNameFromP2PGroup(coll.opName, sendTaskCount, recvTaskCount);

  coll.ranksInGroupedP2P = std::vector<int>{
      ranksInGroupedP2PSet.begin(), ranksInGroupedP2PSet.end()};

  coll.count = byteCount;

  return coll;
}

std::optional<CollTraceInfo> parseCollInfoFromNcclKernelPlan(
    ncclKernelPlan& plan,
    cudaStream_t stream) {
  if (plan.comm == nullptr || plan.comm->ctrace == nullptr) {
    return std::nullopt;
  }
  auto collTaskHead = ncclIntruQueueHead(&plan.collTaskQueue);
  auto p2pTaskHead = ncclIntruQueueHead(&plan.p2pTaskQueue);
  if (collTaskHead == nullptr && p2pTaskHead == nullptr) {
    WARN("CollTrace: no coll or p2p task in this plan, this plan is empty");
    return std::nullopt;
  } else if (collTaskHead != nullptr && collTaskHead->next != nullptr) {
    WARN(
        "CollTrace: more than one coll task in this plan, this is currently not supported");
    return std::nullopt;
  } else if (collTaskHead != nullptr && p2pTaskHead != nullptr) {
    WARN(
        "CollTrace: both coll and p2p task in this plan, this is currently not supported");
    return std::nullopt;
  }
  CollTraceInfo collInfo = collTaskHead != nullptr
      ? parseCollInfoFromCollTask(*collTaskHead)
      : parseCollInfoFromP2PTasks(*p2pTaskHead, plan.comm->rank);

  collInfo.opCount = plan.comm->opCount;

  return collInfo;
}

std::unique_ptr<CollTraceEvent> collTraceAquireEventCommon(
    ncclComm* comm,
    CollTraceEvent::EventType type,
    cudaStream_t stream) {
  if (!comm->ctrace) {
    return nullptr;
  }
  struct ncclCudaGraph graph;
  auto res = ncclCudaGetCapturingGraph(&graph, stream);
  if (res != ncclSuccess) {
    WARN("Internal error: ncclCudaGetCapturingGraph failed by %d", res);
    return nullptr;
  }
  if (graph.graph != nullptr) {
    // We are in a cuda graph, this is currently unsupported
    WARN(
        "COLLTRACE: does not support cuda graph. Collectives from comm %lx will be skipped",
        comm->commHash);
    return nullptr;
  }
  auto event = comm->ctrace->createEvent(type);
  if (!event) {
    throw CollTraceError("Event init failed");
    return nullptr; /*Event init failed*/
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
  auto comm = plan->comm;
  if (!comm->ctrace) {
    return nullptr;
  }

  auto event =
      collTraceAquireEventCommon(comm, CollTraceEvent::EventType::COMM, stream);
  if (event == nullptr) {
    WARN("COLLTRACE: failed to aquire event");
    return nullptr;
  }
  event->coll = collOpt.value();
  return event;
}

} // namespace meta::colltrace
