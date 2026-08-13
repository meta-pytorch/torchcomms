// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/colltrace/CollTraceWrapper.h"

#include <algorithm>
#include <atomic>

#include <folly/Conv.h>
#include <folly/Synchronized.h>
#include <folly/Unit.h>
#include "comms/utils/RankUtils.h"
#include "comms/utils/checks.h"
#include "comms/utils/colltrace/CollMetadataImpl.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "meta/logger/DebugExt.h"

#include "comms/utils/colltrace/CudaWaitEvent.h"
#include "comms/utils/colltrace/DummyCollTraceHandle.h"
#include "comms/utils/colltrace/GenericMetadata.h"
#include "comms/utils/colltrace/GraphCudaWaitEvent.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
#include "comms/utils/colltrace/plugins/LifecycleEventFeedPlugin.h"
#include "comms/utils/colltrace/plugins/WatchdogPlugin.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/NcclxLogger.h"
#include "meta/hints/GlobalHints.h"
#include "meta/wrapper/DataTypeConv.h"

#include "debug.h"
#include "nccl.h"

#include "comms/ctran/CtranComm.h"
#include "comms/utils/colltrace/AlgoStats.h"

#include <fmt/core.h>

namespace meta::comms::ncclx {

namespace {
struct LifecycleFeedRegistryEntry {
  ncclComm_t comm{nullptr};
  std::weak_ptr<meta::comms::colltrace::ICollTrace> colltrace;
};

folly::Synchronized<std::vector<LifecycleFeedRegistryEntry>>&
getLifecycleFeedRegistry() {
  static folly::Synchronized<std::vector<LifecycleFeedRegistryEntry>> registry;
  return registry;
}

void registerLifecycleFeed(
    ncclComm_t comm,
    const std::shared_ptr<meta::comms::colltrace::ICollTrace>& colltrace) {
  auto registry = getLifecycleFeedRegistry().wlock();
  const auto existing = std::find_if(
      registry->begin(), registry->end(), [comm](const auto& entry) {
        return entry.comm == comm;
      });
  if (existing != registry->end()) {
    existing->colltrace = colltrace;
    return;
  }
  registry->push_back(
      LifecycleFeedRegistryEntry{.comm = comm, .colltrace = colltrace});
}

void deregisterLifecycleFeed(ncclComm_t comm) {
  auto registry = getLifecycleFeedRegistry().wlock();
  registry->erase(
      std::remove_if(
          registry->begin(),
          registry->end(),
          [comm](const auto& entry) { return entry.comm == comm; }),
      registry->end());
}

uint64_t getNextLifecycleFeedCommId() {
  static std::atomic<uint64_t> nextCommId{1};
  return nextCommId.fetch_add(1, std::memory_order_relaxed);
}

enum class KernelPlanType { none, single, multiple };

template <typename T, T* T::* next>
KernelPlanType getKernelPlanType(ncclIntruQueue<T, next>* taskHead) {
  if (ncclIntruQueueEmpty(taskHead)) {
    return KernelPlanType::none;
  } else if (taskHead->head->*next == nullptr) {
    return KernelPlanType::single;
  } else {
    return KernelPlanType::multiple;
  }
}

struct KernelPlanInfo {
  KernelPlanType collType;
  KernelPlanType p2pType;
};

KernelPlanInfo getKernelPlanInfo(ncclKernelPlan& plan) {
  return KernelPlanInfo{
      .collType = getKernelPlanType(&plan.collTaskQueue),
      .p2pType = getKernelPlanType(&plan.p2pTaskQueue)};
}

bool isCapturingStream(cudaStream_t stream) {
  cudaStreamCaptureStatus status;

  auto res = cudaStreamGetCaptureInfo(stream, &status);

  if (res != cudaSuccess) {
    NCCLX_LOG_FIRST_N(
        WARN,
        1,
        "Internal error: cudaStreamGetCaptureInfo failed by {}",
        static_cast<int>(res));
    return false;
  }
  return status != cudaStreamCaptureStatusNone;
}

bool shouldCheckAsyncError() {
  auto checkAsyncErrorHintStr =
      ::ncclx::getGlobalHint(::ncclx::HintKeys::kCollTraceCrashOnAsyncError);
  if (checkAsyncErrorHintStr.has_value()) {
    auto checkAsyncError = folly::tryTo<bool>(checkAsyncErrorHintStr.value());
    if (checkAsyncError.hasValue()) {
      return checkAsyncError.value();
    } else {
      NCCLX_LOG(
          ERR,
          "CollTrace: Failed to parse {} as valid async error value, skip async error check in colltrace",
          checkAsyncErrorHintStr.value());
    }
  }
  return false;
}

bool shouldCheckTimeout() {
  auto checkTimeoutHintStr =
      ::ncclx::getGlobalHint(::ncclx::HintKeys::kCollTraceCrashOnTimeout);
  if (checkTimeoutHintStr.has_value()) {
    auto checkTimeout = folly::tryTo<bool>(checkTimeoutHintStr.value());
    if (checkTimeout.hasValue()) {
      return checkTimeout.value();
    } else {
      NCCLX_LOG(
          ERR,
          "CollTrace: Failed to parse {} as valid timeout value, skip timeout check in colltrace",
          checkTimeoutHintStr.value());
    }
  }
  return false;
}

std::chrono::milliseconds getCollTraceWatchdogTimeout() {
  auto timeoutSecondsHintStr =
      ::ncclx::getGlobalHint(::ncclx::HintKeys::kCollTraceTimeoutMs);
  if (timeoutSecondsHintStr.has_value()) {
    auto timeoutSeconds = folly::tryTo<int>(timeoutSecondsHintStr.value());
    if (timeoutSeconds.hasValue()) {
      return std::chrono::milliseconds{timeoutSeconds.value()};
    } else {
      NCCLX_LOG(
          ERR,
          "CollTrace: Failed to parse {} as valid timeout value, fallback to default timeout value.",
          timeoutSecondsHintStr.value());
    }
  }
  // 0 will be treated as no timeout
  return std::chrono::seconds{NCCL_COLLTRACE_WATCHDOG_DEFAULT_TIMEOUT_SEC};
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

colltrace::GroupedP2PMetaData getGroupedP2PComponent(
    const ncclTaskP2p* p2pTaskHead,
    int selfRank,
    uint64_t opCount) {
  int sendTaskCount = 0;
  int recvTaskCount = 0;
  std::size_t byteCount = 0;
  std::unordered_set<int> ranksInGroupedP2P{selfRank};

  for (const auto* cur = p2pTaskHead; cur != nullptr; cur = cur->next) {
    if (cur->func == ncclFuncSend) {
      ++sendTaskCount;
    } else {
      ++recvTaskCount;
    }
    byteCount += cur->bytes;
    ranksInGroupedP2P.insert(cur->root);
  }

  ncclFunc_t func;
  if (sendTaskCount > 0 && recvTaskCount > 0) {
    func = ncclFuncSendRecv;
  } else if (sendTaskCount > 0) {
    func = ncclFuncSend;
  } else {
    func = ncclFuncRecv;
  }

  const char* opName = ncclFuncToString(func);
  return colltrace::GroupedP2PMetaData{
      .opName = std::string{opName},
      .algoName = getAlgoNameFromP2PGroup(opName, sendTaskCount, recvTaskCount),
      .opCount = opCount,
      .ranksInGroupedP2P =
          std::vector<int>(ranksInGroupedP2P.begin(), ranksInGroupedP2P.end()),
      .dataType = commInt8, // we are counting bytes
      .count = byteCount};
}

colltrace::CollectiveMetadata getCollectiveComponent(
    const ncclTaskColl& collTask,
    uint64_t opCount) {
  return colltrace::CollectiveMetadata{
      .opName = std::string{ncclFuncToString(collTask.func)},
      .algoName = getAlgoNameFromCollTask(collTask),
      .opCount = opCount,
      .sendbuff = reinterpret_cast<uintptr_t>(collTask.sendbuff),
      .recvbuff = reinterpret_cast<uintptr_t>(collTask.recvbuff),
      .dataType = ncclToCommDataType(collTask.datatype),
      .count = collTask.count};
}

std::unique_ptr<meta::comms::colltrace::ICollMetadata>
getP2PMetadataFromNcclKernelPlan(ncclKernelPlan& plan, cudaStream_t stream) {
  auto comm = plan.comm;
  auto p2pMetadata = getGroupedP2PComponent(
      ncclIntruQueueHead(&plan.p2pTaskQueue), comm->rank, comm->opCount);

  auto baselineMetadata = colltrace::BaselineMetadata{
      .stream = stream,
  };

  return colltrace::makeCollMetadata(
      plan.comm->logMetaData,
      std::move(baselineMetadata),
      std::move(p2pMetadata));
}

std::unique_ptr<meta::comms::colltrace::ICollMetadata>
getCollMetadataFromNcclKernelPlan(
    ncclKernelPlan& plan,
    const KernelPlanInfo& planInfo,
    cudaStream_t stream) {
  auto collTaskHead = ncclIntruQueueHead(&plan.collTaskQueue);
  const auto& collTask = *collTaskHead;
  auto baselineMetadata = colltrace::BaselineMetadata{
      .stream = stream,
      .coll = ncclToCommFunc(collTask.func),
      .algorithm = ncclToCommAlgo(collTask.algorithm),
      .protocol = ncclToCommProtocol(collTask.protocol),
      .redOp = ncclToCommRedOp(collTask.opHost),
      .root = collTask.root,
  };
  auto collMetadata = getCollectiveComponent(collTask, plan.comm->opCount);
  return colltrace::makeCollMetadata(
      plan.comm->logMetaData,
      std::move(baselineMetadata),
      std::move(collMetadata));
}

std::unique_ptr<meta::comms::colltrace::ICollMetadata>
getGroupedCollP2PMetadataFromNcclKernelPlan(
    ncclKernelPlan& plan,
    const KernelPlanInfo& planInfo,
    cudaStream_t stream) {
  auto curCollTask = ncclIntruQueueHead(&plan.collTaskQueue);
  std::vector<colltrace::CollectiveMetadata> collMetadataList;
  while (curCollTask != nullptr) {
    collMetadataList.push_back(
        getCollectiveComponent(*curCollTask, plan.comm->opCount));
    curCollTask = curCollTask->next;
  }

  std::optional<colltrace::GroupedP2PMetaData> p2pMetadata;
  if (planInfo.p2pType != KernelPlanType::none) {
    p2pMetadata = getGroupedP2PComponent(
        ncclIntruQueueHead(&plan.p2pTaskQueue),
        plan.comm->rank,
        plan.comm->opCount);
  }

  auto baselineMetadata = colltrace::BaselineMetadata{
      .stream = stream,
  };

  return colltrace::makeCollMetadata(
      plan.comm->logMetaData,
      std::move(baselineMetadata),
      colltrace::GroupedCollP2PMetaData{
          .colls = std::move(collMetadataList),
          .p2p = std::move(p2pMetadata),
      });
}

std::unique_ptr<meta::comms::colltrace::ICollMetadata>
getEmptyKernelTaskMetadata(
    ncclKernelPlan& plan,
    const KernelPlanInfo& planInfo,
    cudaStream_t stream) {
  auto baselineMetadata = colltrace::BaselineMetadata{
      .stream = stream,
      .coll = CommFunc::NumFuncs,
      .algorithm = CommAlgo::NumAlgorithms,
      .protocol = CommProtocol::NumProtocols,
      .redOp = commRedOp_t::commNumOps,
      .root = 0,
  };
  auto collMetadata = colltrace::CollectiveMetadata{
      .opName = "Unknown",
      .algoName = "EmptyKernelTask",
  };
  return colltrace::makeCollMetadata(
      plan.comm->logMetaData,
      std::move(baselineMetadata),
      std::move(collMetadata));
}
} // namespace

ncclResult_t newCollTraceInit(ncclComm* comm) {
  // Parse NCCL_COLLTRACE configuration flags
  bool algoStatEnabled = false;
  bool lifecycleEnabled = false;
  bool verboseEnabled = false;
  bool traceEnabled = false;
  for (const auto& mode : NCCL_COLLTRACE) {
    if (mode == "ALL" || mode == "all") {
      algoStatEnabled = true;
      lifecycleEnabled = true;
      traceEnabled = true;
    } else if (mode == "algostat") {
      algoStatEnabled = true;
    } else if (mode == "lifecycle") {
      lifecycleEnabled = true;
    } else if (mode == "verbose") {
      verboseEnabled = true;
    } else if (mode == "trace") {
      traceEnabled = true;
    } else {
      NCCLX_LOG(
          ERR,
          "Unknown NCCL_COLLTRACE mode '{}'. Valid modes are ALL, algostat, lifecycle, trace, and verbose.",
          mode);
      return ncclInvalidArgument;
    }
  }

  NCCLX_LOG(
      INFO,
      "CollTrace init - NCCL_COLLTRACE: [algostat: {}, lifecycle: {}, verbose: {}, trace: {}]",
      algoStatEnabled,
      lifecycleEnabled,
      verboseEnabled,
      traceEnabled);

  // Initialize standalone AlgoStats if algostat mode enabled
  // This is independent of which colltrace implementation is used
  if (algoStatEnabled) {
    comm->algoStats = meta::comms::colltrace::AlgoStats::getOrCreate(
        comm->logMetaData.commHash, comm->logMetaData.commDesc);
  }

  // AlgoStats alone does not require the full colltrace infrastructure.
  if (!lifecycleEnabled && !verboseEnabled && !traceEnabled) {
    return ncclSuccess;
  }

  NCCLX_LOG(INFO, "Initializing new CollTrace");

  auto plugins =
      std::vector<std::unique_ptr<meta::comms::colltrace::ICollTracePlugin>>{};

  if (verboseEnabled || traceEnabled) {
    auto commDumpPlugin =
        std::make_unique<meta::comms::colltrace::CommDumpPlugin>(
            meta::comms::colltrace::CommDumpConfig{
                .pastCollSize = NCCL_COLLTRACE_RECORD_MAX,
                .pendingCollSize = NCCL_COLLTRACE_PENDING_QUEUE_SIZE,
            });
    plugins.push_back(std::move(commDumpPlugin));
  }

  if (lifecycleEnabled) {
    plugins.push_back(
        std::make_unique<meta::comms::colltrace::LifecycleEventFeedPlugin>(
            meta::comms::colltrace::LifecycleEventFeedConfig{
                .commId = getNextLifecycleFeedCommId(),
            }));
  }

  auto ifCheckAsync = shouldCheckAsyncError();
  auto ifCheckTimeout = shouldCheckTimeout();
  auto timeout = getCollTraceWatchdogTimeout();
  NCCLX_LOG(
      INFO,
      "CollTrace watchdog config: checkAsyncError: {}, checkTimeout: {}, timeout: {} sec",
      ifCheckAsync,
      ifCheckTimeout,
      timeout.count());
  if (ifCheckAsync || ifCheckTimeout) {
    auto watchdogPlugin =
        std::make_unique<meta::comms::colltrace::WatchdogPlugin>(
            meta::comms::colltrace::WatchdogPluginConfig{
                .checkAsyncError = ifCheckAsync,
                .funcIfError =
                    [comm]() {
                      ncclResult_t asyncError{ncclSuccess};
                      ncclCommGetAsyncError(comm, &asyncError);
                      if (asyncError != ncclSuccess &&
                          asyncError != ncclInProgress) {
                        return true;
                      }
                      return false;
                    },
                .checkTimeout = ifCheckTimeout,
                .timeout = timeout,
            });
    plugins.push_back(std::move(watchdogPlugin));
  }

  auto colltraceNew = std::make_shared<meta::comms::colltrace::CollTrace>(
      meta::comms::colltrace::CollTraceConfig{
          .maxCheckCancelInterval =
              std::chrono::milliseconds{NCCL_COLLTRACE_WAKEUP_INTERVAL_MS},
      },
      comm->logMetaData,
      [metadata = comm->logMetaData,
       cudaDev = comm->cudaDev]() -> CommsMaybeVoid {
        NCCL_NAMED_THREAD_START_EXT(
            "CollTrace", metadata.rank, metadata.commHash, metadata.commDesc);
        CUDA_CHECK_EXPECTED(cudaSetDevice(cudaDev));
        // Ensure we are using the thread local stream capture mode to avoid
        // getting error about stream capture mode.
        auto mode{cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal};
        CUDA_CHECK_EXPECTED(cudaThreadExchangeStreamCaptureMode(&mode));
        return folly::unit;
      },
      std::move(plugins));

  comm->newCollTrace = std::move(colltraceNew);
  if (lifecycleEnabled) {
    registerLifecycleFeed(comm, comm->newCollTrace);
  }

  return ncclSuccess;
}

ncclResult_t newCollTraceDestroy(ncclComm* comm) {
  deregisterLifecycleFeed(comm);
  comm->newCollTrace.reset();
  return ncclSuccess;
}

std::unique_ptr<meta::comms::colltrace::ICollMetadata>
getMetadataFromNcclKernelPlan(ncclKernelPlan& plan, cudaStream_t stream) {
  auto planInfo = getKernelPlanInfo(plan);

  // Handle invalid cases
  if (planInfo.collType == KernelPlanType::none &&
      planInfo.p2pType == KernelPlanType::none) {
    NCCLX_LOG_FIRST_N(
        ERR, 3, "CollTrace: No coll or p2p task in the NCCL Kenrel Plan!");
    if (plan.persistent) {
      // Deadlock safety: a graph-captured plan with no coll/p2p task has no
      // kernel that will publish a start/end timestamp into the graph ring, so
      // registering a graph colltrace record for it would never complete and
      // would hang the poll thread on drain. Skip it --
      // getHandleFromNcclKernelPlan falls back to a DummyCollTraceHandle.
      // (Eager empty-task tracing, which completes normally on the stream, is
      // unaffected.)
      return nullptr;
    }
    return getEmptyKernelTaskMetadata(plan, planInfo, stream);
  }

  // Handle single collective case
  if (planInfo.collType == KernelPlanType::single &&
      planInfo.p2pType == KernelPlanType::none) {
    return getCollMetadataFromNcclKernelPlan(plan, planInfo, stream);
  }
  // Handle grouped p2p case
  if (planInfo.collType == KernelPlanType::none &&
      planInfo.p2pType != KernelPlanType::none) {
    return getP2PMetadataFromNcclKernelPlan(plan, stream);
  }
  return getGroupedCollP2PMetadataFromNcclKernelPlan(plan, planInfo, stream);
}

std::shared_ptr<meta::comms::colltrace::ICollTraceHandle>
getHandleFromNcclKernelPlan(ncclKernelPlan& plan, cudaStream_t stream) {
  auto colltrace = plan.comm->newCollTrace;
  if (colltrace == nullptr) {
    // For all the invalid cases, we ruturn a dummy handle just so that we
    // don't need to add extra checks in the baseline NCCL code
    return std::make_unique<meta::comms::colltrace::DummyCollTraceHandle>();
  }

  auto metadata = getMetadataFromNcclKernelPlan(plan, stream);
  if (metadata == nullptr) {
    return std::make_unique<meta::comms::colltrace::DummyCollTraceHandle>();
  }

  auto makeWaitEvent =
      [&]() -> std::unique_ptr<meta::comms::colltrace::ICollWaitEvent> {
    if (plan.persistent) {
      return std::make_unique<meta::comms::colltrace::GraphCudaWaitEvent>(
          stream);
    }
    return std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream);
  };

  auto res = colltrace->recordCollective(std::move(metadata), makeWaitEvent());

  if (res.hasError()) {
    NCCLX_LOG_FIRST_N(
        ERR,
        1,
        "Failed to get colltrace handle due to: {}",
        res.error().message);
    return std::make_unique<meta::comms::colltrace::DummyCollTraceHandle>();
  }
  return res.value();
}

std::unordered_map<std::string, std::string> collTraceGetInfo() {
  std::unordered_map<std::string, std::string> info;
  info["colltrace_enabled"] = folly::to<std::string>(!NCCL_COLLTRACE.empty());
  info["colltrace_new_colltrace"] = folly::to<std::string>(true);
  info["colltrace_supports_check_async_error"] = folly::to<std::string>(true);
  // Only new colltrace supports checking timeout
  info["colltrace_supports_check_timeout"] =
      folly::to<std::string>(!NCCL_COLLTRACE.empty());

  return info;
}
} // namespace meta::comms::ncclx

namespace ncclx::colltrace {

namespace {

meta::comms::colltrace::LifecycleEventFeedPlugin* getLifecycleEventFeedPlugin(
    const std::shared_ptr<meta::comms::colltrace::ICollTrace>& colltrace) {
  if (colltrace == nullptr) {
    return nullptr;
  }
  return dynamic_cast<meta::comms::colltrace::LifecycleEventFeedPlugin*>(
      colltrace->getPluginByName(
          std::string{meta::comms::colltrace::LifecycleEventFeedPlugin::
                          kLifecycleEventFeedPluginName}));
}

meta::comms::colltrace::LifecycleEventFeedPlugin* getLifecycleEventFeedPlugin(
    ncclComm_t comm) {
  if (comm == nullptr) {
    return nullptr;
  }
  return getLifecycleEventFeedPlugin(comm->newCollTrace);
}

constexpr uint64_t toExternalLifecycleCollId(uint64_t internalCollId) {
  return internalCollId + 1;
}

void appendLifecycleEvents(
    const std::vector<meta::comms::colltrace::LifecycleEventRecord>&
        unreadEvents,
    std::vector<LifecycleEvent>& events) {
  events.reserve(events.size() + unreadEvents.size());
  for (const auto& event : unreadEvents) {
    LifecycleEventType eventType{LifecycleEventType::Enqueue};
    switch (event.eventType) {
      case meta::comms::colltrace::LifecycleEventType::kEnqueue:
        eventType = LifecycleEventType::Enqueue;
        break;
      case meta::comms::colltrace::LifecycleEventType::kStart:
        eventType = LifecycleEventType::Start;
        break;
      case meta::comms::colltrace::LifecycleEventType::kEnd:
        eventType = LifecycleEventType::End;
        break;
    }
    events.push_back(
        LifecycleEvent{
            .replayId = event.replayId.value_or(kInvalidReplayId),
            .commId = event.commId,
            .collId = toExternalLifecycleCollId(
                event.capturedCollId.value_or(event.collId)),
            .executionCollId = toExternalLifecycleCollId(event.collId),
            .eventType = eventType,
            .timestamp = std::chrono::duration<double>(
                             event.timestamp.time_since_epoch())
                             .count(),
        });
  }
}

} // namespace

__attribute__((visibility("default"))) ncclResult_t
getCollTraceCommId(ncclComm_t comm, uint64_t& commId) {
  commId = 0;
  if (comm == nullptr) {
    return ncclInvalidArgument;
  }
  auto* plugin = getLifecycleEventFeedPlugin(comm);
  if (plugin == nullptr) {
    return ncclInvalidUsage;
  }
  commId = plugin->getCommId();
  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclResult_t
getLatestCollTraceCollectiveId(ncclComm_t comm, uint64_t& collId) {
  collId = 0;
  if (comm == nullptr) {
    return ncclInvalidArgument;
  }
  auto* plugin = getLifecycleEventFeedPlugin(comm);
  if (plugin == nullptr) {
    return ncclInvalidUsage;
  }
  collId = plugin->getLatestLifecycleCollectiveId();
  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclResult_t
drainUnreadLifecycleEvents(std::vector<LifecycleEvent>& events) {
  events.clear();
  auto registry = meta::comms::ncclx::getLifecycleFeedRegistry().wlock();
  registry->erase(
      std::remove_if(
          registry->begin(),
          registry->end(),
          [](const auto& entry) { return entry.colltrace.expired(); }),
      registry->end());

  std::vector<
      std::pair<std::shared_ptr<meta::comms::colltrace::ICollTrace>, uint64_t>>
      flushes;
  flushes.reserve(registry->size());
  for (const auto& entry : *registry) {
    if (auto colltrace = entry.colltrace.lock()) {
      flushes.emplace_back(colltrace, colltrace->requestFlush());
    }
  }
  for (const auto& [colltrace, generation] : flushes) {
    colltrace->waitFlush(generation);
  }
  for (const auto& [colltrace, _] : flushes) {
    auto* plugin = getLifecycleEventFeedPlugin(colltrace);
    if (plugin != nullptr) {
      appendLifecycleEvents(plugin->drainUnreadLifecycleEvents(), events);
    }
  }
  std::stable_sort(
      events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.timestamp < rhs.timestamp;
      });
  return ncclSuccess;
}

__attribute__((visibility("default"))) void dumpAlgoStat(
    ncclComm_t comm,
    std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>&
        map) {
  map.clear();
  if (comm == nullptr) {
    return;
  }

  // Baseline and ctran share the same AlgoStats instance via getOrCreate.
  if (comm->algoStats) {
    auto dump = comm->algoStats->dump();
    for (const auto& [opName, algoMap] : dump.entries) {
      for (const auto& [algoName, sizeMap] : algoMap) {
        for (const auto& [sz, count] : sizeMap) {
          map[opName][algoName] += count;
        }
      }
    }
  }
}

namespace {

struct AlgoInfo {
  std::string opName;
  std::string algoName;
  size_t msgSize{0};
};

std::optional<AlgoInfo> parseAlgoInfoFromNcclKernelPlan(ncclKernelPlan& plan) {
  auto collTaskHead = ncclIntruQueueHead(&plan.collTaskQueue);
  auto p2pTaskHead = ncclIntruQueueHead(&plan.p2pTaskQueue);
  if (collTaskHead == nullptr && p2pTaskHead == nullptr) {
    return std::nullopt;
  }
  if (collTaskHead != nullptr && collTaskHead->next != nullptr) {
    return std::nullopt;
  }
  if (collTaskHead != nullptr && p2pTaskHead != nullptr) {
    return std::nullopt;
  }

  if (collTaskHead != nullptr) {
    return AlgoInfo{
        .opName = std::string{ncclFuncToString(collTaskHead->func)},
        .algoName = fmt::format(
            "Baseline_{}_{}_{}",
            ncclProtoToString(collTaskHead->protocol),
            ncclAlgoToString(collTaskHead->algorithm),
            static_cast<int>(collTaskHead->nMaxChannels)),
        .msgSize = collTaskHead->count * ncclTypeSize(collTaskHead->datatype),
    };
  }

  auto sendCount = 0;
  auto recvCount = 0;
  if (p2pTaskHead->func == ncclFuncSend) {
    sendCount++;
  } else {
    recvCount++;
  }
  for (auto cur = p2pTaskHead->next; cur != nullptr; cur = cur->next) {
    if (cur->func == ncclFuncSend) {
      sendCount++;
    } else {
      recvCount++;
    }
  }

  ncclFunc_t func = ncclFuncRecv;
  if (sendCount > 0 && recvCount > 0) {
    func = ncclFuncSendRecv;
  } else if (sendCount > 0) {
    func = ncclFuncSend;
  }
  auto opName = std::string{ncclFuncToString(func)};
  auto algoName =
      fmt::format("Baseline_{}_S{}_R{}", opName, sendCount, recvCount);
  return AlgoInfo{
      .opName = std::move(opName),
      .algoName = std::move(algoName),
  };
}

void armNcclInKernelColltrace(
    [[maybe_unused]] ncclKernelPlan& plan,
    [[maybe_unused]] const std::shared_ptr<
        meta::comms::colltrace::ICollTraceHandle>& handle,
    [[maybe_unused]] int compCap) {
  // `colltraceHdr` only exists when the in-kernel colltrace gate in device.h
  // is on; arming is a no-op otherwise.
#ifdef NCCLX_INKERNEL_COLLTRACE
  // Symmetric-memory kernels use a different arg layout; skip them.
  if (plan.isSymColl || plan.kernelArgs == nullptr || handle == nullptr) {
    return;
  }
  // Default to unarmed so the in-kernel scope is a no-op off the graph path.
  plan.kernelArgs->colltraceHdr = {};
  // The ring's 128b atomic write requires sm_90+; getColltraceDeviceHandle()
  // returns a ring-backed handle only while capturing a CUDA graph.
  if (compCap < 90) {
    return;
  }
  auto devHandle = handle->getColltraceDeviceHandle();
  if (!devHandle.valid()) {
    return;
  }
  // Single-kernel baseline collective: emit both boundaries on this kernel.
  devHandle.emitStart = true;
  devHandle.emitEnd = true;
  plan.kernelArgs->colltraceHdr = devHandle;
#endif // in-kernel colltrace gate
}

} // namespace

std::shared_ptr<meta::comms::colltrace::ICollTraceHandle>
prepareNcclKernelColltrace(
    ncclKernelPlan* plan,
    cudaStream_t stream,
    int compCap) {
  if (plan->comm->algoStats) {
    auto algoInfo = parseAlgoInfoFromNcclKernelPlan(*plan);
    if (algoInfo.has_value()) {
      plan->comm->algoStats->record(
          algoInfo->opName, algoInfo->algoName, algoInfo->msgSize);
    }
  }

  auto handle = NCCL_COLLTRACE.empty()
      ? std::shared_ptr<
            meta::comms::colltrace::ICollTraceHandle>{std::make_unique<
            meta::comms::colltrace::DummyCollTraceHandle>()}
      : meta::comms::ncclx::getHandleFromNcclKernelPlan(*plan, stream);
  armNcclInKernelColltrace(*plan, handle, compCap);
  return handle;
}
} // namespace ncclx::colltrace
