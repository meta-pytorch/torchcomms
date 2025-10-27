// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cuda_runtime.h>
#include <memory>

#include "comms/ctran/gpe/CtranChecksum.h"
#include "meta/colltrace/CollTraceColl.h"

// CUDA event pointer w/ deleter
struct CudaEventDeleter {
  void operator()(cudaEvent_t e) {
    // Ignore error at destroy
    cudaEventDestroy(e);
  }
};
using CudaEventPtr = std::unique_ptr<
    std::pointer_traits<cudaEvent_t>::element_type,
    CudaEventDeleter>;

class SharedPool {
 public:
  ~SharedPool() {};

  void add(CudaEventPtr item) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push_back(std::move(item));
  }

  CudaEventPtr takeOne() {
    std::lock_guard<std::mutex> lock(mutex_);

    // no event available, create new one
    if (pool_.empty()) {
      cudaEvent_t newEvent = nullptr;
      if (NCCL_COLLTRACE_EVENT_BLOCKING_SYNC) {
        CUDACHECKIGNORE(
            cudaEventCreateWithFlags(&newEvent, cudaEventBlockingSync));
      } else {
        CUDACHECKIGNORE(cudaEventCreate(&newEvent));
      }
      CudaEventPtr item(newEvent);
      return item;
    }

    // reuse existing event
    CudaEventPtr tmp = std::move(pool_.front());
    pool_.pop_front();
    return tmp;
  }

 private:
  std::deque<CudaEventPtr> pool_;
  mutable std::mutex mutex_;
};

class CollWaitEvent {
 public:
  virtual ncclResult_t waitEventFinish() = 0;
  // similar to waitEventFinish, but also execute the function f periodically
  // till the event is finished
  virtual ncclResult_t waitEventFinishAndExecute(
      const std::function<void()>& func) = 0;
  virtual std::optional<float> getElapsedTimeSinceEvent(
      CollWaitEvent* start) = 0;
  virtual ~CollWaitEvent() = default;
};

class CpuWaitEvent : public CollWaitEvent {
 public:
  CpuWaitEvent() = default;
  ~CpuWaitEvent() override = default;

  virtual std::optional<float> getElapsedTimeSinceEvent(
      CollWaitEvent* start) override;
  virtual ncclResult_t waitEventFinish() override;
  virtual ncclResult_t waitEventFinishAndExecute(
      const std::function<void()>& func) override;

  void setNotFinished();
  void setFinished();
  std::chrono::time_point<std::chrono::high_resolution_clock> getFinishTime();

 private:
  std::condition_variable cv_;
  folly::Synchronized<bool, std::mutex> finishedSync_;

  // We only expect one thread to call setFinished, so we don't need to protect
  // it under a lock.
  std::chrono::time_point<std::chrono::high_resolution_clock> finishTime_;
};

class CudaWaitEvent : public CollWaitEvent {
 public:
  CudaWaitEvent(CudaEventPtr e, SharedPool& pool)
      : event_(std::move(e)), pool_(pool) {}
  ~CudaWaitEvent() override {
    // Return the event to the pool
    pool_.add(std::move(event_));
  }

  cudaEvent_t getCudaEvent() {
    return event_.get();
  }

  void setStream(cudaStream_t stream);

  std::optional<float> getElapsedTime(cudaEvent_t start);
  virtual std::optional<float> getElapsedTimeSinceEvent(
      CollWaitEvent* start) override;
  virtual ncclResult_t waitEventFinish() override;
  virtual ncclResult_t waitEventFinishAndExecute(
      const std::function<void()>& func) override;

 private:
  cudaStream_t stream_;
  CudaEventPtr event_;
  SharedPool& pool_;
};

// Event data structure
struct CollTraceEvent {
  enum class EventType {
    COMM,
    // CPU only comms, these events do not have a CUDA event associated with it
    COMM_CPU,
    // Wake up the worker thread. Currently used to wake up the worker thread
    // to dump information.
    WAKE_UP,
    TERMINATE
  };

  CollTraceColl coll;
  std::shared_ptr<CollWaitEvent> start{nullptr};
  std::shared_ptr<CollWaitEvent> stop{nullptr};
  EventType eventType = EventType::COMM;

  bool isGraphCapture{false};

  // ChecksumItem is allocated in the pinned memory pool, it should have the
  // same lifetime as the collTraceEvent.
  ChecksumItem* ctranChecksumItem{nullptr};

  CollTraceEvent(EventType type) : eventType(type) {}
  CollTraceEvent() = default;

  ~CollTraceEvent() {
    // Ensure the checksum item is freed before the CollTraceEvent is freed
    if (ctranChecksumItem) {
      ctranChecksumItem->reset();
    }
  }

  // CollTraceEvent is not copyable
  CollTraceEvent(const CollTraceEvent&) = default;
  CollTraceEvent& operator=(const CollTraceEvent&) = default;

  // CollTraceEvent is movable
  CollTraceEvent(CollTraceEvent&&) = default;
  CollTraceEvent& operator=(CollTraceEvent&&) = default;
};
