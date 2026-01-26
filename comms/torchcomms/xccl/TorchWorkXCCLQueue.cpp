#include "comms/torchcomms/xccl/TorchWorkXCCL.hpp"

namespace torch::comms {

TorchWorkXCCL::WorkStatus TorchWorkXCCLQueue::garbageCollect(
    bool isMainThread) {
  std::lock_guard<std::recursive_mutex> lock(work_queues_mutex_);

  TorchWorkXCCL::WorkStatus last_status = TorchWorkXCCL::WorkStatus::COMPLETED;

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
      TorchWorkXCCL::WorkStatus status = work->checkStatus();
      last_status = status;

      if (status == TorchWorkXCCL::WorkStatus::COMPLETED) {
        // Work is completed, remove it from the work queue
        work_queue.pop();
        completed_work_queue_.push_back(work);
        // Continue to the next element in the queue
      } else if (
          status == TorchWorkXCCL::WorkStatus::TIMEDOUT ||
          status == TorchWorkXCCL::WorkStatus::ERROR) {
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

  if (isMainThread) {
    // If we are the main thread, clear the completed work queues
    completed_work_queue_.clear();
  }

  return last_status;
}

TorchWorkXCCL::WorkStatus TorchWorkXCCLQueue::finalize() {
  // Because this function is typically called after the timeout thread has
  // already joined, we might not need to lock here.  But doing the lock anyway,
  // as defensive programming, just in case someone moves the thread join order
  // later.  The cost of the lock itself should be small on modern linux systems
  // (uncontended locks are typically just an atomic operation).
  std::lock_guard<std::recursive_mutex> lock(work_queues_mutex_);

  // Initialize the status to COMPLETED to cover the case where the queue is
  // empty
  TorchWorkXCCL::WorkStatus status = TorchWorkXCCL::WorkStatus::COMPLETED;
  while (!stream_work_queues_.empty()) {
    status = garbageCollect(true);
    if (status == TorchWorkXCCL::WorkStatus::ERROR ||
        status == TorchWorkXCCL::WorkStatus::TIMEDOUT ||
        status == TorchWorkXCCL::WorkStatus::COMPLETED) {
      break;
    }
  }

  // Clear all work queues & completed work queue.
  //
  // NOTE: finalize MUST return without holding references to any work object,
  // otherwise it may leak object and cause side effects.
  stream_work_queues_.clear();
  completed_work_queue_.clear();

  return status;
}

void TorchWorkXCCLQueue::enqueueWork(
    c10::intrusive_ptr<TorchWorkXCCL> work,
    xpuStream_t stream) {
  // Add work to stream's queue after events have been recorded
  std::lock_guard<std::recursive_mutex> lock(work_queues_mutex_);
  stream_work_queues_[stream].push(work);
}

} // namespace torch::comms
