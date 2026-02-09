// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CollTrace.h"
#include "bootstrap.h"
#include "checks.h"
#include "comm.h"
#include "param.h"
#ifdef BUILD_META_INTERNAL
#include "CollTraceFunc.h"
#include "comms/rcclx/develop/meta/lib/RcclxScubaEvent.h"
#endif

NCCL_PARAM(ColltraceRecordMax, "COLLTRACE_RECORD_MAX", 100);
NCCL_PARAM(ColltraceMaxDumpSize, "COLLTRACE_MAX_DUMP_SIZE", 20);
NCCL_PARAM(ColltraceDumpIntervalSec, "COLLTRACE_DUMP_INTERVAL_SEC", 300);

constexpr std::string_view COLL_VARIANCE_TABLE = "rccl_coll_variance_v2";
constexpr int RANKS_PER_HOST = 8;

namespace meta::colltrace {

namespace {
CudaEventPtr getCudaEventPtr() {
  cudaEvent_t newEvent = nullptr;
  CUDACHECKIGNORE(cudaEventCreate(&newEvent));
  CudaEventPtr item(newEvent);
  return item;
}
} // namespace

CollTrace::CollTrace(ncclComm* comm)
    : comm_(comm),
      commHash_(std::to_string(comm->commHash)),
      rank_(comm->rank) {
  lastStopTime_ = std::chrono::high_resolution_clock::now();

  // Create reference event and stream once for all communicators
  std::call_once(referenceInitFlag_, CollTrace::recordReferenceEvent);

  profilingWorkerThread_ =
      std::thread{[this]() { return collTraceThreadFn(comm_->cudaDev); }};
}

CollTrace::~CollTrace() {
  try {
    INFO(
        NCCL_INIT,
        "COLLTRACE: commHash %s rank %d - Destroy START",
        commHash_.c_str(),
        rank_);
    eventQueue_.push(
        std::unique_ptr<CollTraceEvent>(
            new CollTraceEvent(CollTraceEvent::EventType::TERMINATE)));
    if (profilingWorkerThread_.joinable()) {
      profilingWorkerThread_.join();
    }

    if (rank_ == 0) {
      reportToScubaIfNeeded(false);
    }

    INFO(
        NCCL_INIT,
        "COLLTRACE: commHash %s rank %d - Destroy COMPLETE",
        commHash_.c_str(),
        rank_);
  } catch (const std::exception& e) {
    WARN(
        "COLLTRACE: commHash %s rank %d - Destroy FAILED: %s",
        commHash_.c_str(),
        rank_,
        e.what());
  }
}

void* CollTrace::collTraceThreadFn(int cudaDev) {
  INFO(NCCL_INIT, "CollTrace thread started for cudaDev %d", cudaDev);
  auto err = cudaSetDevice(cudaDev);
  if (err != cudaSuccess) {
    WARN("Cuda failure '%s'", cudaGetErrorString(err));
    return nullptr;
  }

  lastReportTime_ = std::chrono::steady_clock::now();

  INFO(
      NCCL_INIT,
      "COLLTRACE: commHash %s rank %d - worker thread STARTED",
      commHash_.c_str(),
      rank_);
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
#ifdef BUILD_META_INTERNAL
    if (enableGranularScuba()) {
      auto event = RcclxScubaEvent{rank_, comm_->commHash, &(curEvent_->coll)};
      event.record("CollTrace WAIT_START");
    }
#endif
    curEvent_->start->waitEventFinish();
    auto startTs = getEventTime(curEvent_->start.get());
    {
      std::lock_guard<std::mutex> lock(workerMutex_);
      curEvent_->coll.startTs = startTs;
      curEvent_->coll.interCollTime =
          std::chrono::duration_cast<std::chrono::microseconds>(
              curEvent_->coll.startTs - lastStopTime_);
    }
    curCollState_ = CurrentCollState::IN_PROGRESS;
#ifdef BUILD_META_INTERNAL
    if (enableGranularScuba()) {
      auto event = RcclxScubaEvent{rank_, comm_->commHash, &(curEvent_->coll)};
      event.record("CollTrace IN_PROGRESS");
    }
#endif
    auto ncclRes = curEvent_->stop->waitEventFinish();
    lastStopTime_ = getEventTime(curEvent_->stop.get());
    curCollState_ = CurrentCollState::DONE;

    float latency = -1;
    if (ncclRes == ncclSuccess) {
      auto latencyMaybe =
          curEvent_->stop->getElapsedTimeSinceEvent(curEvent_->start.get());
      // latencyMaybe could be nullopt when cudaEventElapsedTime failed
      // this could happen when events are not recorded or stream is not valid
      if (!latencyMaybe.has_value()) {
        WARN(
            "CollTrace: getElapsedTimeSinceEvent failed, aborting worker thread");
        return nullptr;
      }
      latency = *latencyMaybe;
    }

    recordCurCollResult(cudaDev, latency);

#ifdef BUILD_META_INTERNAL
    if (enableGranularScuba()) {
      curEvent_->coll.latencyMs = latency;
      auto event = RcclxScubaEvent{rank_, comm_->commHash, &(curEvent_->coll)};
      event.record("CollTrace COMPLETE");
    }
#endif

    curEvent_.reset();
  }

  INFO(
      NCCL_INIT,
      "COLLTRACE: commHash %s rank %d - worker thread TERMINATE",
      commHash_.c_str(),
      rank_);
  return nullptr;
}

void CollTrace::enqueueEvent(std::unique_ptr<CollTraceEvent> event) {
  event->coll.collId = curCollId_.fetch_add(1);
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

std::unique_ptr<CollTraceEvent> CollTrace::createEvent(
    CollTraceEvent::EventType type) {
  auto eventInfo = std::make_unique<CollTraceEvent>(type);
  eventInfo->start = std::make_unique<CudaWaitEvent>(getCudaEventPtr());
  eventInfo->stop = std::make_unique<CudaWaitEvent>(getCudaEventPtr());

  if (!eventInfo->start || !eventInfo->stop) {
    std::unique_ptr<CollTraceEvent> nullCollTraceEvent(nullptr);
    return nullCollTraceEvent;
  }
  return eventInfo;
}

bool shouldAggregateRingBuffer(int collId) {
  const int NCCL_COLLTRACE_RECORD_MAX = ncclParamColltraceRecordMax();
  return ((collId + 1) % NCCL_COLLTRACE_RECORD_MAX == 0);
}

void CollTrace::reportToScubaIfNeeded(bool checkInterval = true) {
  auto now = std::chrono::steady_clock::now();
  auto secs_passed =
      std::chrono::duration_cast<std::chrono::seconds>(now - lastReportTime_)
          .count();
  if (checkInterval) {
    if (secs_passed < ncclParamColltraceDumpIntervalSec() &&
        stats_.size() < ncclParamColltraceMaxDumpSize()) {
      return;
    }
  }

  INFO(
      NCCL_COLL,
      "CollTrace: %d seconds passed since last report, stats size = %d, checkInterval = %d",
      secs_passed,
      stats_.size(),
      checkInterval);
#ifdef BUILD_META_INTERNAL
  reportToScuba(stats_, std::string(COLL_VARIANCE_TABLE), commHash_);
#endif
  lastReportTime_ = std::chrono::steady_clock::now();
  stats_.clear();
}

void CollTrace::recordCurCollResult(int rank, float latency) {
  const int NCCL_COLLTRACE_RECORD_MAX = ncclParamColltraceRecordMax();

  auto result = std::make_unique<CollTraceInfo>(curEvent_->coll);
  auto collId = result->collId;
  result->latencyMs = latency;

  std::lock_guard<std::mutex> lock(workerMutex_);
  pastColls_.push_back(std::move(result));
  if (pastColls_.size() > NCCL_COLLTRACE_RECORD_MAX) {
    pastColls_.pop_front();
  }

  if (shouldAggregateRingBuffer(collId) && pastColls_.size() > 0) {
    std::vector<float> latencyAllGather;
    latencyAllGather.resize(RANKS_PER_HOST * NCCL_COLLTRACE_RECORD_MAX, 0);
    int start = (comm_->localRank) * NCCL_COLLTRACE_RECORD_MAX;
    for (int i = start; i < start + NCCL_COLLTRACE_RECORD_MAX; i++) {
      latencyAllGather[i] = pastColls_[i - start]->latencyMs;
    }
    auto before = std::chrono::high_resolution_clock::now();
    auto ncclResult = bootstrapIntraNodeAllGather(
        comm_->bootstrap,
        comm_->localRankToRank,
        comm_->localRank,
        comm_->localRanks,
        latencyAllGather.data(),
        NCCL_COLLTRACE_RECORD_MAX * sizeof(float));
    auto after = std::chrono::high_resolution_clock::now();
    auto interval_us =
        std::chrono::duration_cast<std::chrono::microseconds>(after - before)
            .count();
    if (ncclResult != ncclSuccess) {
      WARN("CollTrace: All gather exchange latency data failed");
      return;
    }

    if (rank == 0) {
      INFO(NCCL_COLL, "latency metrics all gather takes %d us", interval_us);
      try {
        auto stats = aggregateResults(
            pastColls_,
            latencyAllGather,
            RANKS_PER_HOST,
            NCCL_COLLTRACE_RECORD_MAX);
        stats_.push_back(stats);
        if (stats_.size() > ncclParamColltraceMaxDumpSize()) {
          stats_.pop_front();
        }
        reportToScubaIfNeeded();
      } catch (const std::exception& e) {
        WARN("Aggregating error: %s", e.what());
      }
    }
  }
}

CollTrace::Dump CollTrace::dump() const {
  std::lock_guard<std::mutex> lock(workerMutex_);
  CollTrace::Dump dump{};

  if (curCollState_ == CurrentCollState::IN_PROGRESS ||
      curCollState_ == CurrentCollState::WAIT_START) {
    // copy contents
    dump.currentColl =
        std::unique_ptr<CollTraceInfo>(new CollTraceInfo(curEvent_->coll));
  }

  dump.pendingColls = dumpQueue();

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

std::deque<CollTraceInfo> CollTrace::dumpQueue() const {
  std::deque<CollTraceInfo> tmp{};
  {
    std::unique_lock<std::mutex> lock(eventQueue_.mutex_);
    for (auto& item : eventQueue_.queue_) {
      // copy content of coll within each event
      tmp.emplace_back(item->coll);
    }
  }
  return tmp;
}

CudaStreamPtr CollTrace::referenceStream_{nullptr};
CudaEventPtr CollTrace::referenceEvent_{nullptr};
std::chrono::system_clock::time_point CollTrace::referenceTime_{};
std::once_flag CollTrace::referenceInitFlag_{};

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
    CudaWaitEvent* cudaWaitEvent) {
  auto timeMsMaybe = cudaWaitEvent->getElapsedTime(referenceEvent_.get());

  if (timeMsMaybe == std::nullopt) {
    WARN(
        "COLLTRACE: cudaEventElapsedTime failed, all the time measurement will be meaningless.");

    // Return a dummy value so that at least we are not crashing the program
    return std::chrono::high_resolution_clock::now();
  }

  return referenceTime_ + std::chrono::microseconds(long(*timeMsMaybe * 1000));
}

} // namespace meta::colltrace
