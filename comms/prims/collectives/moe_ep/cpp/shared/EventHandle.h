// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>

// AMD vs NVIDIA include divergence — same pattern as
// MultipeerIbgdaTransport.cc. HIPify rewrites cuda* symbols to hip* in the
// source after compilation branches; the include path determines which type
// defs are visible.
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace comms::prims::moe_ep {

/**
 * EventHandle — RAII wrapper around a `cudaEvent_t` (HIPified to
 * `hipEvent_t` on AMD), recorded on the current stream at construction.
 *
 * Bound through pybind as `_cpp.EventHandle`. The Python `EventOverlap`
 * wraps it and exposes `current_stream_wait()` so callers can synchronize
 * the default stream against the captured event without touching CUDA APIs.
 *
 * Shared ownership semantics — the Python `Buffer.dispatch / combine` code
 * passes `EventHandle` instances around (they're returned in the result
 * tuple AND captured by `previous_event=` arguments). Implemented as a
 * shared_ptr so multiple wrappers can share a single underlying CUDA event
 * and the destruction is one-shot.
 */
class EventHandle {
 public:
  /** Capture an event on the current stream. */
  EventHandle();

  /** Adopt an existing event (used internally by the runtime when launch
   *  paths construct the event manually). */
  explicit EventHandle(cudaEvent_t event);

  /** Make `torch.cuda.current_stream()` wait on the captured event. */
  void current_stream_wait() const;

  /** Underlying event handle (for runtime passthrough). */
  cudaEvent_t event() const noexcept {
    return event_ ? event_->raw : nullptr;
  }

 private:
  struct Impl {
    cudaEvent_t raw{nullptr};
    Impl();
    explicit Impl(cudaEvent_t event) : raw(event) {}
    ~Impl();
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
  };
  std::shared_ptr<Impl> event_;
};

} // namespace comms::prims::moe_ep
