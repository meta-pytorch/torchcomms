// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/CudaEventPool.h"

#include <folly/concurrency/UnboundedQueue.h>

#include "comms/utils/CudaRAII.h"

namespace meta::comms::colltrace {

// CudaEvent is small enough (8Bytes), that we can initialize the MPMCQueue
// without worrying about the memory overhead. Not using folly::Singleton
// could help reduce some overhead due to atomic operations.
// We could not use folly::MPMCQueue as it doesn't support move semantics.
static folly::UMPMCQueue<CudaEvent, false> cudaEventPoolUMPMCQueue{};

CachedCudaEvent::CachedCudaEvent(CudaEvent event) : event_(std::move(event)) {}

CachedCudaEvent::~CachedCudaEvent() {
  if (!isMoved_) {
    CudaEventPool::returnEvent(std::move(event_));
  }
}

cudaEvent_t CachedCudaEvent::get() const {
  return event_.get();
}

const CudaEvent& CachedCudaEvent::getRef() const {
  return event_;
}

CachedCudaEvent::CachedCudaEvent(CachedCudaEvent&& other) noexcept
    : event_(std::move(other.event_)), isMoved_(other.isMoved_) {
  other.isMoved_ = true;
}

CachedCudaEvent& CachedCudaEvent::operator=(CachedCudaEvent&& other) noexcept {
  if (this != &other) {
    if (!isMoved_) {
      CudaEventPool::returnEvent(std::move(event_));
    }
    event_ = std::move(other.event_);
    isMoved_ = other.isMoved_;
    other.isMoved_ = true;
  }
  return *this;
}

CachedCudaEvent CudaEventPool::getEvent() {
  auto eventMaybe = cudaEventPoolUMPMCQueue.try_dequeue();
  if (FOLLY_LIKELY(eventMaybe.has_value())) {
    return CachedCudaEvent{std::move(eventMaybe.value())};
  }
  // If the queue is empty, create a new event. This would be expensive as it
  // will call cudaEventCreate under the hood. So we should try to reuse the
  // events as much as possible.
  return CachedCudaEvent{CudaEvent{}};
}

void CudaEventPool::returnEvent(CudaEvent event) {
  cudaEventPoolUMPMCQueue.enqueue(std::move(event));
}

} // namespace meta::comms::colltrace
