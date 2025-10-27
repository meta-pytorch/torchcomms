// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchWorkNCCLX.hpp"

namespace torch {
namespace comms {

TorchWorkNCCLX::WorkStatus TorchWorkNCCLXQueue::garbageCollect() {
  std::lock_guard<std::recursive_mutex> lock(work_queues_mutex_);

  TorchWorkNCCLX::WorkStatus last_status =
      TorchWorkNCCLX::WorkStatus::COMPLETED;

  // Keep popping completed elements until we hit an in-progress element
  // or the queue is empty
  // Use an iterator to safely remove empty queues while iterating
  auto it = stream_work_queues_.begin();
  while (it != stream_work_queues_.end()) {
    auto& work_queue = it->second;

    while (!work_queue.empty()) {
      // Get the first work object in the queue
      auto work = work_queue.front();

      // Use the checkStatus function to determine the work status
      TorchWorkNCCLX::WorkStatus status = work->checkStatus();
      last_status = status;

      if (status == TorchWorkNCCLX::WorkStatus::COMPLETED) {
        // Work is completed, remove it from the work queue
        work_queue.pop();
        // Continue to the next element in the queue
      } else if (
          status == TorchWorkNCCLX::WorkStatus::TIMEDOUT ||
          status == TorchWorkNCCLX::WorkStatus::ERROR) {
        // Return the error status immediately
        return status;
      } else {
        // NOT_STARTED or INPROGRESS - stop processing this queue
        break;
      }
    }

    // If the queue is now empty, remove it from the map
    if (work_queue.empty()) {
      it = stream_work_queues_.erase(it);
    } else {
      ++it;
    }
  }

  return last_status;
}

TorchWorkNCCLX::WorkStatus TorchWorkNCCLXQueue::finalize() {
  // Because this function is typically called after the timeout thread has
  // already joined, we might not need to lock here.  But doing the lock anyway,
  // as defensive programming, just in case someone moves the thread join order
  // later.  The cost of the lock itself should be small on modern linux systems
  // (uncontended locks are typically just an atomic operation).
  std::lock_guard<std::recursive_mutex> lock(work_queues_mutex_);

  // Initialize the status to COMPLETED to cover the case where the queue is
  // empty
  TorchWorkNCCLX::WorkStatus status = TorchWorkNCCLX::WorkStatus::COMPLETED;
  while (!stream_work_queues_.empty()) {
    status = garbageCollect();
    if (status == TorchWorkNCCLX::WorkStatus::ERROR ||
        status == TorchWorkNCCLX::WorkStatus::TIMEDOUT ||
        status == TorchWorkNCCLX::WorkStatus::COMPLETED) {
      break;
    }
  }

  // Clear all work queues & completed work queue.
  //
  // NOTE: finalize MUST return without holding references to any work object,
  // otherwise it may leak object and cause side effects.
  stream_work_queues_.clear();

  return status;
}

void TorchWorkNCCLXQueue::enqueueWork(
    std::shared_ptr<TorchWorkNCCLX> work,
    cudaStream_t stream) {
  // Add work to stream's queue after events have been recorded
  std::lock_guard<std::recursive_mutex> lock(work_queues_mutex_);
  stream_work_queues_[stream].push(work);
}

} // namespace comms
} // namespace torch
