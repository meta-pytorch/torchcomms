// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <functional>
#include <mutex>
#include <unordered_map>

namespace meta::comms {

// A CUDA "side stream" that lives alongside a main capture stream and
// offers an ``fork_from(stream, fn)`` API to launch work onto the side
// stream during CUDA graph capture. The launched work is captured into
// the graph, but does NOT serialize subsequent operations on the main
// stream.
//
// For each ``fork_from()`` call, the helper internally:
//   1. Forks ``stream → sideStream`` via
//      ``cudaEventRecord(depEvent, stream)`` +
//      ``cudaStreamWaitEvent(sideStream, depEvent)``.
//   2. Captures ``stream``'s current dependency set via
//      ``cudaStreamGetCaptureInfo`` (post-fork, pre-rejoin).
//   3. Invokes ``fn(sideStream)`` so the caller can launch captured ops.
//   4. Rejoins ``sideStream → stream`` via
//      ``cudaEventRecord(depEvent, sideStream)`` +
//      ``cudaStreamWaitEvent(stream, depEvent)``. This rejoin is
//      required — ``cudaStreamEndCapture`` rejects captures with an
//      unjoined fork.
//   5. Restores main's pre-rejoin deps via
//      ``cudaStreamUpdateCaptureDependencies(stream, savedDeps,
//      cudaStreamSetCaptureDependencies)``. The rejoin node still lives
//      in the graph DAG, but ``stream``'s next captured op does
//      not inherit it as a predecessor — so work you launch on the
//      side stream stays off main's critical path at replay.
//
// Cross-call state: the instance owns a single side stream and single
// dependency event shared across ``fork_from()`` calls. That's safe because
// each call consumes the dep event immediately after recording it
// (``record`` + ``waitEvent`` pair), so there is no lingering unconsumed
// recording to "clean up" between calls. The side stream accumulates a
// captured chain of user ops across the same capture, which is fine —
// it's always off main's critical path.
//
// Green contexts: a ``stream`` that belongs to a CUDA green context forks
// onto a side stream created in that same green context (one per green
// context, created on first use and kept until destruction), so the captured
// fork/rejoin stays inside the caller's context instead of hopping to the
// primary context and back. Other streams use the side stream created at
// construction. Within one capture, work forked onto different side streams
// keeps the order it was forked in, as it would on a single side stream: a
// fork onto another side stream first waits for the previous fork's work.
//
// Construction must happen OUTSIDE of an active graph capture (it
// allocates a stream and event). Typical usage: one instance per
// communicator, created at ``init()`` time.
class GraphSideStream {
 public:
  // ``priority`` matches ``cudaStreamCreateWithPriority``; ``0`` is default.
  explicit GraphSideStream(int priority = 0);
  ~GraphSideStream();

  GraphSideStream(const GraphSideStream&) = delete;
  GraphSideStream& operator=(const GraphSideStream&) = delete;
  GraphSideStream(GraphSideStream&&) = delete;
  GraphSideStream& operator=(GraphSideStream&&) = delete;

  // Run ``fn(sideStream)`` on the side stream with fork/save/rejoin/restore
  // scaffolding wrapped around it.
  //
  // If ``stream`` is not currently under CUDA graph capture, ``fn`` is
  // invoked directly with ``stream`` (no fork). Useful for callers that
  // speculatively route through ``fork_from()`` without branching on capture
  // state.
  //
  // Returns the first CUDA error encountered. On error, the user function
  // may or may not have been invoked — callers that need strict ordering
  // should check ``cudaStreamGetCaptureInfo`` up front.
  cudaError_t fork_from(
      cudaStream_t stream,
      std::function<void(cudaStream_t)> fn);

  cudaStream_t get() const {
    return side_stream_;
  }

 private:
  struct SideStream {
    cudaStream_t stream{nullptr};
    cudaEvent_t dep_event{nullptr};
  };

  // The side stream ``stream`` forks onto; falls back to the construction-time
  // side stream for streams outside a green context.
  SideStream side_stream_for(cudaStream_t stream);

  int priority_{0};
  cudaStream_t side_stream_{nullptr};
  cudaEvent_t dep_event_{nullptr};
  std::mutex mutex_;
  // Keyed by the green context's ``cuCtxGetId``, which CUDA never reuses
  // (unlike ``CUgreenCtx`` addresses).
  std::unordered_map<unsigned long long, SideStream> green_side_streams_;
  // The side stream that ran the last fork, and the capture it ran in.
  unsigned long long last_capture_id_{0};
  SideStream last_side_;
};

} // namespace meta::comms
