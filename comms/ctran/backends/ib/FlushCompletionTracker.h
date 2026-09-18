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
  void track(CtranIbRequest* req) {
    if (req != nullptr) {
      req->setRefCount(static_cast<int>(perDevice_.size()));
    }
    for (auto& reqs : perDevice_) {
      reqs.push_back(req);
    }
  }

  // Retire the oldest flush slot of the given device. An out-of-range device
  // escapes as std::out_of_range rather than as a commResult_t, since it can
  // only mean the caller mismatched the CQ it polled with this tracker.
  commResult_t complete(const int device) {
    auto& reqs = perDevice_.at(device);
    FB_CHECKABORT(
        !reqs.empty(), "No outstanding flush tracked for device {}", device);
    CtranIbRequest* const req = reqs.front();
    reqs.pop_front();
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
