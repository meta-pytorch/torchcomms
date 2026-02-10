// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <mutex>
#include <thread>

#include "CollTraceEvent.h"
#include "comms/rcclx/develop/meta/lib/CollTraceUtils.h"
#include "comms/rcclx/develop/meta/lib/EventQueue.h"

struct ncclComm;

namespace meta::colltrace {

struct CudaStreamDeleter {
  void operator()(cudaStream_t e) {
    // Ignore error at destroy
    cudaStreamDestroy(e);
  }
};
using CudaStreamPtr = std::unique_ptr<
    std::pointer_traits<cudaStream_t>::element_type,
    CudaStreamDeleter>;

struct CollStats;

// CollTrace Workflow
// 1) measure collective latency by adding cuda/hip event before and after
// a kernel launch in RCCL.

// 2) Started a separate worker thread to collect latency data into
// its local ring buffer.

// 3) Perform CPU level all gather to exchange latency data
// between threads (200 to 300us per 100 latency data exchange)

// 4) Report to scuba when results buffer is full or every few minutes

class CollTrace {
 public:
  CollTrace(ncclComm* comm);
  ~CollTrace();

  void enqueueEvent(std::unique_ptr<CollTraceEvent> event);

  // For testing
  void waitForWorkerFinishQueue();

  std::unique_ptr<CollTraceEvent> createEvent(
      CollTraceEvent::EventType type = CollTraceEvent::EventType::COMM);

  void recordCurCollResult(int rank, float latencyMs);

  void reportToScubaIfNeeded(bool checkInterval);

  enum class CurrentCollState {
    PENDING,
    WAIT_START,
    IN_PROGRESS,
    DONE,
  };

  struct Dump {
    std::deque<CollTraceInfo> pastColls;
    std::deque<CollTraceInfo> pendingColls;
    std::unique_ptr<CollTraceInfo> currentColl;
  };

  CollTrace::Dump dump() const;

  // Helper function to reset state for testing.
  void resetPastColls();

  std::deque<CollTraceInfo> dumpQueue() const;

  std::chrono::time_point<std::chrono::system_clock> getEventTime(
      CudaWaitEvent* event);

 private:
  EventQueue<CollTraceEvent> eventQueue_;
  std::thread profilingWorkerThread_;
  void* collTraceThreadFn(int cudaDev);

  std::atomic<uint64_t> curCollId_{0};
  std::unique_ptr<CollTraceEvent> curEvent_;
  std::deque<std::unique_ptr<CollTraceInfo>> pastColls_;
  std::atomic<CurrentCollState> curCollState_{CurrentCollState::PENDING};

  ncclComm* comm_{nullptr};
  std::string commHash_;
  int rank_{-1};
  std::deque<std::vector<CollStats>> stats_;
  std::chrono::time_point<std::chrono::steady_clock> lastReportTime_;

  // Lock changes from worker thread to curEvent_, eventQueue_ and pastColls_
  mutable std::mutex workerMutex_;

  // For testing
  std::atomic<bool> waitingForQueueEmpty_;
  std::mutex waitQueueEmptyMutex_;
  std::condition_variable waitQueueEmptyCv_;

  static ncclResult_t recordReferenceEvent();

  // Used to build a correlation between CPU and GPU timestamps, so we can use
  // it to calculate the latency in the future.
  static CudaStreamPtr referenceStream_;
  static CudaEventPtr referenceEvent_;
  static std::once_flag referenceInitFlag_;
  static std::chrono::system_clock::time_point referenceTime_;
  std::chrono::time_point<std::chrono::high_resolution_clock> lastStopTime_;
};
} // namespace meta::colltrace
