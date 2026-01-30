// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/colltrace/CollTraceWrapper.h"

#include <algorithm>

#include "comms/utils/RankUtils.h"
#include "comms/utils/checks.h"
#include "comms/utils/colltrace/AlgoStats.h"
#include "comms/utils/colltrace/CollMetadataImpl.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "comms/utils/colltrace/CudaWaitEvent.h"
#include "comms/utils/colltrace/DummyCollTraceHandle.h"
#include "comms/utils/colltrace/GenericMetadata.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
#include "comms/utils/colltrace/plugins/WatchdogPlugin.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/hints/GlobalHints.h"
#include "meta/wrapper/DataTypeConv.h"

#include "debug.h"
#include "nccl.h"

#include <fmt/core.h>
#include <folly/logging/xlog.h>

namespace meta::comms::ncclx {

namespace {
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
    XLOG_FIRST_N(
        WARN,
        1,
        fmt::format(
            "Internal error: cudaStreamGetCaptureInfo failed by {}",
            static_cast<int>(res)));
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
      XLOGF(
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
      XLOGF(
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
      XLOGF(
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
  // TODO: this can be removed once new colltrace is fully rolled out
  if (!NCCL_COLLTRACE_USE_NEW_COLLTRACE) {
    XLOGF(
        INFO,
        "Skipping new CollTrace init, NCCL_COLLTRACE_USE_NEW_COLLTRACE is disabled");
    return ncclSuccess;
  }

  // Parse NCCL_COLLTRACE configuration flags
  bool algoStatEnabled = false;
  bool verboseEnabled = false;
  bool traceEnabled = false;
  for (const auto& mode : NCCL_COLLTRACE) {
    if (mode == "algostat") {
      algoStatEnabled = true;
    } else if (mode == "verbose") {
      verboseEnabled = true;
    } else if (mode == "trace") {
      traceEnabled = true;
    }
  }

  XLOGF(
      INFO,
      "CollTrace init - NCCL_COLLTRACE: [algostat: {}, verbose: {}, trace: {}], NCCL_COLLTRACE_USE_NEW_COLLTRACE: {}",
      algoStatEnabled,
      verboseEnabled,
      traceEnabled,
      NCCL_COLLTRACE_USE_NEW_COLLTRACE);

  // Initialize standalone AlgoStats if algostat mode enabled
  // This is independent of which colltrace implementation is used
  if (algoStatEnabled) {
    comm->algoStats = std::make_unique<meta::comms::colltrace::AlgoStats>();
    comm->algoStats->setCommInfo(
        comm->logMetaData.commHash, comm->logMetaData.commDesc);
  }

  // Check if full colltrace is needed (verbose or trace modes)
  // algostat alone does not require full colltrace infrastructure
  if ((!verboseEnabled && !traceEnabled)) {
    return ncclSuccess;
  }

  XLOG(INFO, "Initializing new CollTrace");

  auto plugins =
      std::vector<std::unique_ptr<meta::comms::colltrace::ICollTracePlugin>>{};

  auto commDumpPlugin =
      std::make_unique<meta::comms::colltrace::CommDumpPlugin>(
          meta::comms::colltrace::CommDumpConfig{
              .pastCollSize = NCCL_COLLTRACE_RECORD_MAX,
              .pendingCollSize = NCCL_COLLTRACE_PENDING_QUEUE_SIZE,
          });
  plugins.push_back(std::move(commDumpPlugin));

  auto ifCheckAsync = shouldCheckAsyncError();
  auto ifCheckTimeout = shouldCheckTimeout();
  auto timeout = getCollTraceWatchdogTimeout();
  XLOGF(
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

  return ncclSuccess;
}

ncclResult_t newCollTraceDestroy(ncclComm* comm) {
  comm->newCollTrace.reset();
  return ncclSuccess;
}

std::unique_ptr<meta::comms::colltrace::ICollMetadata>
getMetadataFromNcclKernelPlan(ncclKernelPlan& plan, cudaStream_t stream) {
  auto planInfo = getKernelPlanInfo(plan);

  // Handle invalid cases
  if (planInfo.collType == KernelPlanType::none &&
      planInfo.p2pType == KernelPlanType::none) {
    XLOG_FIRST_N(
        ERR, 3, "CollTrace: No coll or p2p task in the NCCL Kenrel Plan!");
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

  if (isCapturingStream(stream)) {
    if (RankUtils::getGlobalRank().value_or(0) == 0) {
      XLOG_FIRST_N(
          WARN, 1, "CollTrace currently doesn't support capturing streams");
    }
    return std::make_unique<meta::comms::colltrace::DummyCollTraceHandle>();
  }

  auto metadata = getMetadataFromNcclKernelPlan(plan, stream);
  if (metadata == nullptr) {
    return std::make_unique<meta::comms::colltrace::DummyCollTraceHandle>();
  }
  auto res = colltrace->recordCollective(
      std::move(metadata),
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream));

  if (res.hasError()) {
    XLOG_FIRST_N(
        ERR, 5, "Failed to get colltrace handle due to: ", res.error().message);
    return std::make_unique<meta::comms::colltrace::DummyCollTraceHandle>();
  }
  return res.value();
}

std::unordered_map<std::string, std::string> collTraceGetInfo() {
  std::unordered_map<std::string, std::string> info;
  info["colltrace_enabled"] = folly::to<std::string>(!NCCL_COLLTRACE.empty());
  info["colltrace_new_colltrace"] =
      folly::to<std::string>(NCCL_COLLTRACE_USE_NEW_COLLTRACE);
  info["colltrace_supports_check_async_error"] = folly::to<std::string>(true);
  // Only new colltrace supports checking timeout
  info["colltrace_supports_check_timeout"] = folly::to<std::string>(
      !NCCL_COLLTRACE.empty() && NCCL_COLLTRACE_USE_NEW_COLLTRACE);

  return info;
}
} // namespace meta::comms::ncclx
