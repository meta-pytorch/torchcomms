// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/mccl/TorchWorkMCCL.hpp"

namespace torch::comms {

void TorchWorkMCCLQueue::dropAllLocked() {
  // Mark in-flight work timed out so its end hook fires before its resources
  // are released.
  for (auto& [stream, work_queue] : stream_work_queues_) {
    while (!work_queue.empty()) {
      work_queue.front()->closeIncompleteOnTeardown();
      timed_out_work_.push_back(std::move(work_queue.front()));
      work_queue.pop();
    }
  }
  stream_work_queues_.clear();
}

TorchWorkMCCLQueue::~TorchWorkMCCLQueue() {
  // The owner joins the watchdog before destroying the queue, so no lock is
  // needed during teardown.
  dropAllLocked();
}

void TorchWorkMCCLQueue::releaseCompletedTimedOutLocked() {
  std::erase_if(timed_out_work_, [](const auto& work) {
    return work->getResult().has_value();
  });
}

TorchWorkMCCL::WorkStatus TorchWorkMCCLQueue::garbageCollectLocked() {
  releaseCompletedTimedOutLocked();

  // A pending item must not mask a failure or allow a fully drained stream to
  // make the overall result appear completed.
  std::optional<TorchWorkMCCL::WorkStatus> pending;

  // Keep popping terminal elements until we hit an in-progress element
  // or the queue is empty
  // Use an iterator to safely remove empty queues while iterating
  auto it = stream_work_queues_.begin();
  while (it != stream_work_queues_.end()) {
    auto& work_queue = it->second;

    while (!work_queue.empty()) {
      // Use the checkStatus function to determine the work status
      const TorchWorkMCCL::WorkStatus status =
          work_queue.front()->checkStatus();

      if (status == TorchWorkMCCL::WorkStatus::TIMEDOUT ||
          status == TorchWorkMCCL::WorkStatus::ERROR) {
        // Preserve the first failure without hiding later work on this or
        // another stream from the watchdog.
        if (!latched_failure_.has_value()) {
          latched_failure_ = status;
        }
        if (status == TorchWorkMCCL::WorkStatus::TIMEDOUT) {
          timed_out_work_.push_back(work_queue.front());
        }
      } else if (status != TorchWorkMCCL::WorkStatus::COMPLETED) {
        // NOT_STARTED or INPROGRESS. The stream is ordered, so nothing behind
        // this can have finished; move on to the next stream.
        if (!pending.has_value()) {
          pending = status;
        }
        break;
      }

      work_queue.pop();
    }

    // If the queue is now empty, remove it from the map
    if (work_queue.empty()) {
      it = stream_work_queues_.erase(it);
    } else {
      ++it;
    }
  }

  if (latched_failure_.has_value()) {
    return *latched_failure_;
  }
  return pending.value_or(TorchWorkMCCL::WorkStatus::COMPLETED);
}

// Thread-safety: This method is called from the timeout watchdog thread while
// the main thread may be enqueuing work via enqueueWork(). The
// work_queues_mutex_ ensures proper synchronization - both garbageCollect() and
// enqueueWork() acquire the mutex before accessing stream_work_queues_.
TorchWorkMCCL::WorkStatus TorchWorkMCCLQueue::garbageCollect() {
  std::lock_guard<std::mutex> lock(work_queues_mutex_);
  return garbageCollectLocked();
}

TorchWorkMCCL::WorkStatus TorchWorkMCCLQueue::finalize() {
  // Because this function is typically called after the timeout thread has
  // already joined, we might not need to lock here.  But doing the lock anyway,
  // as defensive programming, just in case someone moves the thread join order
  // later.  The cost of the lock itself should be small on modern linux systems
  // (uncontended locks are typically just an atomic operation).
  std::lock_guard<std::mutex> lock(work_queues_mutex_);

  // A latched failure remains reportable after its work entry is retired.
  TorchWorkMCCL::WorkStatus status = garbageCollectLocked();
  while (status == TorchWorkMCCL::WorkStatus::NOT_STARTED ||
         status == TorchWorkMCCL::WorkStatus::INPROGRESS) {
    status = garbageCollectLocked();
  }

  // A latched failure may leave later work in flight.
  dropAllLocked();

  return status;
}

void TorchWorkMCCLQueue::enqueueWork(
    c10::intrusive_ptr<TorchWorkMCCL> work,
    cudaStream_t stream) {
  // Add work to stream's queue after events have been recorded
  std::lock_guard<std::mutex> lock(work_queues_mutex_);
  stream_work_queues_[stream].push(std::move(work));
}

} // namespace torch::comms
