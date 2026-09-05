// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <deque>
#include <vector>

#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/utils/Checks.h"

namespace ctran::ib {

/**
 * Tracks completion of flushes that fan out one loopback RDMA READ per IB
 * device.
 *
 * Every device owns its own CQ and its own loopback RC QP, so completion order
 * is guaranteed only within a device, never across devices. A single shared
 * FIFO would therefore allow a later flush's completion on one device to retire
 * an earlier flush's slot, reporting that earlier flush as done while its READ
 * on another device is still in flight. One FIFO per device plus a per-flush
 * reference count makes a flush complete only once every device has reported.
 *
 * This puts a requirement on progress: a flush stays incomplete until every
 * device's CQ has been polled, so a progress loop restricted to a subset of the
 * devices can never retire a flush.
 */
class FlushCompletionTracker {
 public:
  explicit FlushCompletionTracker(const size_t numDevices)
      : perDevice_(numDevices) {
    // Without a per-device FIFO there is no completion to decrement the
    // reference count that track() arms, so the flush would never retire.
    FB_CHECKABORT(
        numDevices > 0,
        "Flush completion tracking requires at least one device, got {}",
        numDevices);
  }

  // Register a flush that has one RDMA READ posted per device. A null request
  // is tracked as a placeholder that drains without completing anything.
  //
  // A request must not be tracked again before it has retired: setRefCount is a
  // plain assignment, so re-arming a partially drained request discards the
  // outstanding count and the earlier flush's remaining CQEs then drive the
  // reference count negative.
  void track(CtranIbRequest* req) {
    if (req != nullptr) {
      req->setRefCount(static_cast<int>(perDevice_.size()));
    }
    for (auto& reqs : perDevice_) {
      reqs.push_back(req);
    }
  }

  // Undo a track() for a flush that never reached a send queue. Posting even
  // one of its READs makes this illegal: the entry is only still at the back of
  // every device's FIFO, and free of any completion that could race the
  // removal, while nothing has been posted.
  //
  // The reference count track() armed is deliberately left as is. Its previous
  // value is not recoverable here, setRefCount is a plain assignment that every
  // re-arm redoes, and the request stays incomplete either way.
  void untrack(const CtranIbRequest* const req) {
    for (size_t device = 0; device < perDevice_.size(); device++) {
      auto& reqs = perDevice_[device];
      FB_CHECKABORT(
          !reqs.empty() && reqs.back() == req,
          "Flush to untrack is not the newest tracked slot of device {}",
          device);
      reqs.pop_back();
    }
  }

  // Retire the oldest flush slot of the given device. An out-of-range device
  // escapes as std::out_of_range rather than as a commResult_t, since it can
  // only mean the caller mismatched the CQ it polled with this tracker.
  commResult_t complete(const int device) {
    FB_CHECKABORT(
        !perDevice_.at(device).empty(),
        "No outstanding flush tracked for device {}",
        device);
    CtranIbRequest* const req = perDevice_.at(device).front();
    perDevice_.at(device).pop_front();
    if (req != nullptr) {
      FB_COMMCHECK(req->complete());
    }
    return commSuccess;
  }

  size_t outstanding(const int device) const {
    return perDevice_.at(device).size();
  }

 private:
  std::vector<std::deque<CtranIbRequest*>> perDevice_;
};

} // namespace ctran::ib
