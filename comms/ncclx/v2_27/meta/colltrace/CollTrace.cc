// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include <folly/String.h>
#include <folly/logging/xlog.h>

#include "CollTrace.h"
#include "CollTraceEvent.h"
#include "CollTraceUtils.h"

#include "checks.h"
#include "comms/utils/StrUtils.h"
#include "nccl.h"

#include "comms/ctran/utils/Checks.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/hints/GlobalHints.h"
#include "meta/logger/DebugExt.h"

using namespace ncclx::colltrace;

std::unordered_map<std::string, std::optional<std::chrono::milliseconds>>
    SlowCollReporter::pgPrefixToSlowThreshold_{};
std::once_flag SlowCollReporter::slowThresholdMapInitFlag_{};

// Initialize the threshold map using the value of
// NCCL_COLLTRACE_SLOW_COLL_THRESHOLD_BY_PG
void SlowCollReporter::initThresholdMap() {
  pgPrefixToSlowThreshold_.clear();

  for (const auto& pgThresholdPairStr :
       NCCL_COLLTRACE_SLOW_COLL_THRESHOLD_BY_PG) {
    std::vector<std::string> pgThresholdPair;
    folly::split(':', pgThresholdPairStr, pgThresholdPair);
    if (pgThresholdPair.size() != 2) {
      WARN(
          "SlowCollReporter Init: Invalid PG threshold pair: %s",
          pgThresholdPairStr.c_str());
      continue;
    }
    std::string pgPrefix = pgThresholdPair[0];
    std::string thresholdStr = pgThresholdPair[1];
    auto thresholdInt = folly::tryTo<int>(thresholdStr);
    if (!thresholdInt.hasValue()) {
      WARN(
          "SlowCollReporter Init: Invalid threshold: %s", thresholdStr.c_str());
      continue;
    }
    if (thresholdInt.value() < 0) {
      pgPrefixToSlowThreshold_[pgPrefix] = std::nullopt;
      INFO(
          NCCL_INIT,
          "SlowCollReporter Init: Not reporting collective for PG Prefix %s",
          pgPrefix.c_str());
    } else {
      pgPrefixToSlowThreshold_[pgPrefix] =
          std::chrono::milliseconds(thresholdInt.value());
      INFO(
          NCCL_INIT,
          "SlowCollReporter Init: Set threshold for PG Prefix %s to be %dms",
          pgPrefix.c_str(),
          thresholdInt.value());
    }
  }
}

SlowCollReporter::DurationOpt SlowCollReporter::getSlowThreshold(
    const std::string& pgName) {
  for (int i = pgName.size(); i > 0; i--) { // find the longest prefix match
    const auto pgPrefix = pgName.substr(0, i);
    if (pgPrefixToSlowThreshold_.contains(pgPrefix)) {
      return pgPrefixToSlowThreshold_[pgPrefix];
    }
  }

  if (pgPrefixToSlowThreshold_.contains("ANY")) {
    return pgPrefixToSlowThreshold_["ANY"];
  }
  return std::nullopt;
}

SlowCollReporter::SlowCollReporter(const CommLogData& logMetaData)
    : logMetaData_{logMetaData} {
  std::call_once(slowThresholdMapInitFlag_, initThresholdMap);
  slowThreshold_ = getSlowThreshold(logMetaData.commDesc);
  reportIntervalSec_ = NCCL_COLLTRACE_REPORT_INTERVAL_SEC;
  if (slowThreshold_.has_value()) {
    INFO(
        NCCL_COLL,
        "SlowCollReporter: Found PG %s, setting its threshold to %ldms",
        logMetaData.commDesc.c_str(),
        slowThreshold_.value().count());
  }
}

void SlowCollReporter::conditionalReportColl(const CollTraceColl& coll) {
  if (shouldReportColl(coll)) {
    WARN_FIRST_N(
        kDebugRepeatLogCount,
        "COLLTRACE: %s taking too long to finish",
        coll.toString().c_str());
    reportCollToScuba("SlowColl", coll, logMetaData_);
    updateLastReportTimeToNow();
  }

  if (coll.collId < NCCL_COLLTRACE_REPORT_FIRST_N_COLL &&
      coll.opName != "PutNotify" && coll.opName != "WaitNotify") {
    reportCollToScuba(
        fmt::format("First{}", NCCL_COLLTRACE_REPORT_FIRST_N_COLL),
        coll,
        logMetaData_);
    updateLastReportTimeToNow();
  }
}

bool SlowCollReporter::shouldReportColl(const CollTraceColl& coll) {
  auto now = std::chrono::steady_clock::now();
  if (reportIntervalSec_ > 0 &&
      std::chrono::duration_cast<std::chrono::seconds>(now - lastReportTime_)
              .count() >= reportIntervalSec_) {
    return true;
  }
  if (slowThreshold_ == std::nullopt) {
    return false;
  }
  if (coll.latency < 0) {
    return false;
  }
  auto latencyMs = std::chrono::duration<float, std::milli>(coll.latency);
  return latencyMs >= slowThreshold_.value();
}

bool SlowCollReporter::shouldReportUnfinishedColl() {
  auto now = std::chrono::steady_clock::now();
  if (reportIntervalSec_ > 0 &&
      std::chrono::duration_cast<std::chrono::seconds>(now - lastReportTime_)
              .count() >= reportIntervalSec_) {
    return true;
  }
  return false;
}

void SlowCollReporter::updateLastReportTimeToNow() {
  lastReportTime_ = std::chrono::steady_clock::now();
}

CudaStreamPtr CollTrace::referenceStream_{nullptr};
CudaEventPtr CollTrace::referenceEvent_{nullptr};
std::chrono::system_clock::time_point CollTrace::referenceTime_{};
std::once_flag CollTrace::referenceInitFlag_{};

CollTrace::CollTrace(ncclComm* comm)
    : comm_(comm),
      logMetaData_(comm->logMetaData),
      slowCollReporter_(SlowCollReporter(comm->logMetaData)),
      collStat_(comm->logMetaData) {
  std::vector<std::string> enabledFeatures;
  if (!NCCL_COLLTRACE.empty()) {
    for (auto& f : NCCL_COLLTRACE) {
      if (f == "verbose") {
        features |= CollTrace::Features::VERBOSE;
        enabledFeatures.push_back(f);
      } else if (f == "trace") {
        features |= CollTrace::Features::TRACE_MODE;
        enabledFeatures.push_back(f);
      }
    }
  }

  auto checkAsyncErrorHintStr =
      ncclx::getGlobalHint(ncclx::HintKeys::kCollTraceCrashOnAsyncError);
  if (checkAsyncErrorHintStr.has_value()) {
    auto checkAsyncError = folly::tryTo<bool>(checkAsyncErrorHintStr.value());
    if (checkAsyncError.hasValue()) {
      checkAsyncError_ = checkAsyncError.value();
    }
  }

  lastStopTime_ = std::chrono::high_resolution_clock::now();

  // Create reference event and stream once for all communicators
  std::call_once(referenceInitFlag_, CollTrace::recordReferenceEvent);

  // create worker thread
  profilingWorkerThread_ =
      std::thread{[this]() { return collTraceThreadFn(comm_->cudaDev); }};

  std::string enabledFeaturesStr = vecToStr(enabledFeatures);
  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lx commDesc %s rank %d enabled features: %s - Init COMPLETE",
      comm_,
      logMetaData_.commHash,
      logMetaData_.commDesc.c_str(),
      logMetaData_.rank,
      enabledFeaturesStr.c_str());
}

CollTrace::~CollTrace() {
  try {
    INFO(
        NCCL_INIT,
        "COLLTRACE: comm %p commHash %lx commDesc %s rank %d - Destroy START",
        comm_,
        logMetaData_.commHash,
        logMetaData_.commDesc.c_str(),
        logMetaData_.rank);

    eventQueue_.push(
        std::unique_ptr<CollTraceEvent>(
            new CollTraceEvent(CollTraceEvent::EventType::TERMINATE)));
    if (profilingWorkerThread_.joinable()) {
      profilingWorkerThread_.join();
    }

    INFO(
        NCCL_INIT,
        "COLLTRACE: comm %p commHash %lx commDesc %s rank %d - Destroy COMPLETE",
        comm_,
        logMetaData_.commHash,
        logMetaData_.commDesc.c_str(),
        logMetaData_.rank);
  } catch (const std::exception& e) {
    WARN(
        "COLLTRACE: comm %p commHash %lx commDesc %s rank %d - Destroy FAILED: %s",
        comm_,
        logMetaData_.commHash,
        logMetaData_.commDesc.c_str(),
        logMetaData_.rank,
        e.what());
  } catch (...) {
    WARN(
        "COLLTRACE: comm %p commHash %lx commDesc %s rank %d - Destroy FAILED: Unknown exception",
        comm_,
        logMetaData_.commHash,
        logMetaData_.commDesc.c_str(),
        logMetaData_.rank);
  }
}

CollTrace::Dump CollTrace::dump() const {
  std::lock_guard<std::mutex> lock(workerMutex_);
  CollTrace::Dump dump{};

  if (curCollState_ == CurrentCollState::IN_PROGRESS ||
      curCollState_ == CurrentCollState::WAIT_START) {
    // copy contents
    dump.currentColl =
        std::unique_ptr<CollTraceColl>(new CollTraceColl(curEvent_->coll));
  }

  dump.pendingColls = eventQueue_.dumpQueue();

  for (auto& result : pastColls_) {
    // copy contents
    dump.pastColls.emplace_back(*result);
  }
  return dump;
}

void CollTrace::resetPastColls() {
  std::lock_guard<std::mutex> lock(workerMutex_);
  pastColls_.clear();
}

void CollTrace::conditionalReportUnFinishedColl() {
  if (!slowCollReporter_.shouldReportUnfinishedColl()) {
    return;
  }
  CollTraceColl coll(curEvent_->coll);
  reportCollToScuba("UnfinishedColl", coll, logMetaData_);
  slowCollReporter_.updateLastReportTimeToNow();
}

void CollTrace::recordCurCollResult(float latency) {
  auto result = std::make_unique<CollTraceColl>(curEvent_->coll);
  result->latency = latency;

  if (curEvent_->ctranChecksumItem != nullptr &&
      result->ctranAttr.has_value()) {
    result->ctranAttr.value().checksum =
        curEvent_->ctranChecksumItem->checksum_;
  }

  if (features & CollTrace::Features::VERBOSE) {
    INFO(NCCL_COLL, "COLLTRACE: %s", result->toString().c_str());
  }

  slowCollReporter_.conditionalReportColl(*result);

  // Record the statistics of collectives
  // This call may take a long time if CollStat is reporting.
  collStat_.recordColl(*result);

  std::lock_guard<std::mutex> lock(workerMutex_);
  pastColls_.push_back(std::move(result));
  while (!pastColls_.empty() &&
         pastColls_.front()->iteration <= pastColls_.back()->iteration -
                 NCCL_COLLTRACE_RECORD_MAX_ITERATIONS) {
    pastColls_.pop_front();
  }
  if (pastColls_.size() > NCCL_COLLTRACE_RECORD_MAX) {
    pastColls_.pop_front();
  }
}

void CollTrace::afterEachEventPoll(CollTraceColl curColl) {
  if (NCCL_COLLTRACE_REPORT_INTERVAL_SEC > 0) {
    conditionalReportUnFinishedColl();
  }
  if (checkAsyncError_) {
    ncclResult_t asyncResult = ncclSuccess;
    ncclCommGetAsyncError(comm_, &asyncResult);
    if (asyncResult != ncclSuccess && asyncResult != ncclInProgress) {
      auto errorString = fmt::format(
          "Collective (OpCount={}, OpType={}, Count={}, DataType={}) for Comm {} raised the following async exception: {}",
          curColl.opCount,
          curColl.opName,
          curColl.count ? folly::to<std::string>(curColl.count.value()) : "N/A",
          ncclDatatypeToString(curColl.dataType),
          this->logMetaData_.commDesc,
          ncclGetErrorString(asyncResult));
      XLOGF(FATAL, errorString);
    }
  }
}

void* CollTrace::collTraceThreadFn(int cudaDev) {
  NCCL_NAMED_THREAD_START_EXT(
      "CollTrace",
      logMetaData_.rank,
      logMetaData_.commHash,
      logMetaData_.commDesc);

  FB_CUDACHECKTHROW(cudaSetDevice(cudaDev));

  // Ensure we are using the thread local stream capture mode to avoid
  // getting error about stream capture mode.
  auto mode{cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal};
  FB_CUDACHECKTHROW(cudaThreadExchangeStreamCaptureMode(&mode));

  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lx commDesc %s rank %d - worker thread STARTED",
      comm_,
      logMetaData_.commHash,
      logMetaData_.commDesc.c_str(),
      logMetaData_.rank);

  while (true) {
    curCollState_ = CurrentCollState::PENDING;

    // For testing purpose only. During testing, we want to ensure the worker
    // thread reached a steady state before dumping so that the trace dump
    // result is predictable. Otherwise the test can be flaky.
    if (waitingForQueueEmpty_ && eventQueue_.isEmpty()) {
      {
        std::unique_lock<std::mutex> lock(waitQueueEmptyMutex_);
        waitingForQueueEmpty_ = false;
      }
      waitQueueEmptyCv_.notify_all();
    }

    // We intentionally didn't hold the event queue lock till curEvent is
    // updated. That will potentially create deadlock.
    // Downside of current approach is we might miss one pending event in the
    // dump in very rare occasion. But since the worker thread haven't started
    // to wait for the event, it should be fine.
    {
      auto tmp_event = eventQueue_.waitPop();
      std::lock_guard<std::mutex> lock(workerMutex_);
      curEvent_ = std::move(tmp_event);
    }

    if (curEvent_->eventType == CollTraceEvent::EventType::TERMINATE) {
      break;
    } else if (curEvent_->eventType == CollTraceEvent::EventType::WAKE_UP) {
      continue;
    }
    curCollState_ = CurrentCollState::WAIT_START;

    ncclResult_t ncclRes;
    auto reportFunc = [this, coll = curEvent_->coll]() {
      return this->afterEachEventPoll(coll);
    };
    ncclRes = curEvent_->start->waitEventFinishAndExecute(reportFunc);
    auto startTs = getEventTime(curEvent_->start.get());
    {
      std::lock_guard<std::mutex> lock(workerMutex_);
      curEvent_->coll.startTs = startTs;
      curEvent_->coll.interCollTime =
          std::chrono::duration_cast<std::chrono::microseconds>(
              curEvent_->coll.startTs - lastStopTime_);
    }
    curCollState_ = CurrentCollState::IN_PROGRESS;
    ncclRes = curEvent_->stop->waitEventFinishAndExecute(reportFunc);
    lastStopTime_ = getEventTime(curEvent_->stop.get());
    curCollState_ = CurrentCollState::DONE;
    float latency = -1;

    if (ncclRes == ncclSuccess) {
      auto latencyMaybe =
          curEvent_->stop->getElapsedTimeSinceEvent(curEvent_->start.get());
      if (latencyMaybe.has_value()) {
        latency = *latencyMaybe;
      }
    }

    recordCurCollResult(latency);

    curEvent_.reset();
  }

  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lx commDesc %s rank %d - worker thread TERMINATE",
      comm_,
      logMetaData_.commHash,
      logMetaData_.commDesc.c_str(),
      logMetaData_.rank);
  return nullptr;
}

std::unique_ptr<CollTraceEvent> CollTrace::createEvent(
    CollTraceEvent::EventType type) {
  auto eventInfo = std::make_unique<CollTraceEvent>(type);
  if (type == CollTraceEvent::EventType::COMM) {
    eventInfo->start = std::make_unique<CudaWaitEvent>(
        cudaEventPool_.takeOne(), cudaEventPool_);
    eventInfo->stop = std::make_unique<CudaWaitEvent>(
        cudaEventPool_.takeOne(), cudaEventPool_);
  } else if (type == CollTraceEvent::EventType::COMM_CPU) {
    eventInfo->start = std::make_unique<CpuWaitEvent>();
    eventInfo->stop = std::make_unique<CpuWaitEvent>();
  }
  if (!eventInfo->start || !eventInfo->stop) {
    std::unique_ptr<CollTraceEvent> nullCollTraceEvent(nullptr);
    return nullCollTraceEvent;
  }
  return eventInfo;
}

// Functions for recording P2P
void CollTrace::addPeerForP2P(int peer) {
  p2pCurRanks.withWLock([&peer](auto& rankSet) { rankSet.insert(peer); });
}

std::vector<int> CollTrace::getRanksForCurGroup() {
  return p2pCurRanks.withWLock([](auto& rankSet) {
    std::vector<int> ranks(rankSet.begin(), rankSet.end());
    rankSet.clear();
    std::ranges::sort(ranks);
    return ranks;
  });
}

void CollTrace::addGraphEvent(std::unique_ptr<CollTraceEvent> event) {
  graphEvents_.emplace_back(std::move(event));
}

void CollTrace::enqueueEvent(std::unique_ptr<CollTraceEvent> event) {
  event->coll.collId = curCollId_.fetch_add(1);
  if (event->coll.collId != event->coll.opCount) {
    XLOG_FIRST_N(
        WARN,
        5,
        fmt::format(
            "CollId and OpCount is different! Collective info: {}",
            event->coll.toString()));
    event->coll.opCount = event->coll.collId;
  }
  eventQueue_.push(std::move(event));
}

void CollTrace::waitForWorkerFinishQueue() {
  std::unique_lock<std::mutex> waitLock(waitQueueEmptyMutex_);
  waitingForQueueEmpty_ = true;
  eventQueue_.push(
      std::unique_ptr<CollTraceEvent>(
          new CollTraceEvent(CollTraceEvent::EventType::WAKE_UP)));
  waitQueueEmptyCv_.wait(waitLock, [this] { return !waitingForQueueEmpty_; });
}

ncclResult_t CollTrace::recordReferenceEvent() {
  cudaStream_t stream{};
  CUDACHECK(cudaStreamCreate(&stream));
  referenceStream_ = CudaStreamPtr(stream);

  cudaEvent_t newEvent{};
  CUDACHECK(cudaEventCreate(&newEvent));
  referenceEvent_ = CudaEventPtr(newEvent);

  CUDACHECK(cudaEventRecord(newEvent, referenceStream_.get()));
  referenceTime_ = std::chrono::system_clock::now();

  return ncclSuccess;
}

std::chrono::time_point<std::chrono::system_clock> CollTrace::getEventTime(
    CollWaitEvent* event) {
  if (typeid(*event) == typeid(CpuWaitEvent)) {
    auto* cpuWaitEvent = dynamic_cast<CpuWaitEvent*>(event);
    return cpuWaitEvent->getFinishTime();
  } else if (typeid(*event) != typeid(CudaWaitEvent)) {
    if (!loggedTimeError_) {
      WARN_FIRST_N(
          kDebugRepeatLogCount,
          "COLLTRACE: Unsupported event type %s",
          typeid(*event).name());
      loggedTimeError_ = true;
    }
    // Return a dummy value so that at least we are not crashing the program
    return std::chrono::high_resolution_clock::now();
  }
  auto* cudaWaitEvent = dynamic_cast<CudaWaitEvent*>(event);
  auto timeMsMaybe = cudaWaitEvent->getElapsedTime(referenceEvent_.get());

  if (timeMsMaybe == std::nullopt) {
    if (!loggedTimeError_) {
      ERR("COLLTRACE: cudaEventElapsedTime failed, all the time measurement will be meaningless.");
      loggedTimeError_ = true;
    }
    // Return a dummy value so that at least we are not crashing the program
    return std::chrono::high_resolution_clock::now();
  }

  return referenceTime_ + std::chrono::microseconds(long(*timeMsMaybe * 1000));
}
