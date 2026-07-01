// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/moe_ep/cpp/shared/EventHandle.h"

#include <stdexcept>
#include <string>

namespace comms::prims::moe_ep {

namespace {

void check(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
  }
}

} // namespace

EventHandle::Impl::Impl() {
  cudaStream_t stream = nullptr; // current stream
  check(cudaEventCreate(&raw), "cudaEventCreate failed");
  check(cudaEventRecord(raw, stream), "cudaEventRecord failed");
}

EventHandle::Impl::~Impl() {
  if (raw != nullptr) {
    // Best-effort cleanup; never throw from a destructor.
    (void)cudaEventDestroy(raw);
  }
}

EventHandle::EventHandle() : event_(std::make_shared<Impl>()) {}

EventHandle::EventHandle(cudaEvent_t event)
    : event_(std::make_shared<Impl>(event)) {}

void EventHandle::current_stream_wait() const {
  if (!event_) {
    return;
  }
  cudaStream_t stream = nullptr; // current stream
  check(
      cudaStreamWaitEvent(stream, event_->raw, /*flags=*/0),
      "cudaStreamWaitEvent failed");
}

} // namespace comms::prims::moe_ep
