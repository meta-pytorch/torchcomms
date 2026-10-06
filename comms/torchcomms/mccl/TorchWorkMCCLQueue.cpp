// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/mccl/TorchWorkMCCL.hpp"

namespace torch::comms {

TorchWorkMCCLQueue::~TorchWorkMCCLQueue() {
  // Close out any work still in flight at teardown (e.g. PAFT fault recovery
  // destroys the comm directly with a wedged collective still pending).
  // Otherwise the work is destroyed without ever reaching a terminal status,
  // its end hook never fires, and the clog shows a bare "Q"/"S" with no
  // matching "E". Marking it TIMEDOUT here -- while the queue still holds a
  // strong ref, so the hooks are intact -- fires the end hook before
  // destruction.
  //
  // After a clean finalize() the queues are already cleared, so this is a no-op
  // on the normal shutdown path. The owning TorchCommMCCL joins the timeout
  // watchdog before destroying this member, so no other thread touches the
  // queues here.
  for (auto& [stream, work_queue] : stream_work_queues_) {
    while (!work_queue.empty()) {
      work_queue.front()->closeIncompleteOnTeardown();
      work_queue.pop();
    }
  }
}

TorchWorkMCCL::WorkStatus TorchWorkMCCLQueue::garbageCollectLocked() {
  TorchWorkMCCL::WorkStatus last_status = TorchWorkMCCL::WorkStatus::COMPLETED;

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
      TorchWorkMCCL::WorkStatus status = work->checkStatus();
      last_status = status;

      if (status == TorchWorkMCCL::WorkStatus::COMPLETED) {
        // Work is completed, remove it from the work queue
        work_queue.pop();
      } else if (
          status == TorchWorkMCCL::WorkStatus::TIMEDOUT ||
          status == TorchWorkMCCL::WorkStatus::ERROR) {
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

  // Initialize the status to COMPLETED to cover the case where the queue is
  // empty
  TorchWorkMCCL::WorkStatus status = TorchWorkMCCL::WorkStatus::COMPLETED;
  while (!stream_work_queues_.empty()) {
    status = garbageCollectLocked();
    if (status == TorchWorkMCCL::WorkStatus::ERROR ||
        status == TorchWorkMCCL::WorkStatus::TIMEDOUT ||
        status == TorchWorkMCCL::WorkStatus::COMPLETED) {
      break;
    }
  }

  // Clear all work queues
  stream_work_queues_.clear();

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
