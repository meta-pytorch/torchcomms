// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "comms/prims/platform/CudaDriverLazy.h"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibrc/MultipeerIbrcTransport.h"
#include "comms/prims/transport/ibrc/P2pIbrcHostWriter.h"

namespace comms::prims {

/**
 * HostCollectiveEngine - thin host-side orchestration for 0-SM collectives.
 *
 * Ties together the SM-free primitives so a host-driven collective needs no
 * kernel: Copy Engine copies, CE<->host synchronization on host-pinned flags,
 * and host-driven RDMA delegated to P2pIbrcHostWriter. Chunking, peer
 * selection and algorithm logic stay in the caller.
 *
 * Flags passed to the stream-memop methods must be GPU-accessible (host-pinned
 * + mapped, or device memory); the host spin methods read the host alias of
 * that same memory.
 */
class HostCollectiveEngine {
 public:
  HostCollectiveEngine(MultipeerIbrcTransport& transport, cudaStream_t stream)
      : transport_(&transport), stream_(stream) {
    initDriver();
  }

  /**
   * Copy-Engine-only engine, for callers that reach the writer themselves
   * through `MultiPeerTransport::getHostWriter` and need only the CE half.
   * `hostWriter()` throws in this mode.
   */
  explicit HostCollectiveEngine(cudaStream_t stream) : stream_(stream) {
    initDriver();
  }

  HostCollectiveEngine(const HostCollectiveEngine&) = delete;
  HostCollectiveEngine& operator=(const HostCollectiveEngine&) = delete;
  HostCollectiveEngine(HostCollectiveEngine&&) = delete;
  HostCollectiveEngine& operator=(HostCollectiveEngine&&) = delete;
  ~HostCollectiveEngine() = default;

  cudaStream_t stream() const {
    return stream_;
  }

  /**
   * Retarget this engine's Copy-Engine work at another stream. Call once per
   * collective invocation: the engine is cached for a communicator's lifetime
   * while the caller's stream varies. Running on the caller's stream is what
   * keeps the collective stream-ordered, with no host drain.
   *
   * nullptr is not a sentinel: it is the legacy default stream, whose semantics
   * propagate this engine's device-side waits to every other blocking stream in
   * the process. Correct, but pass your own stream to stay isolated.
   */
  void setStream(cudaStream_t stream) {
    stream_ = stream;
  }

  // --- Copy Engine ---

  /** Device-to-device Copy-Engine copy on the engine's stream (SM-free). */
  void enqueueCopy(void* dst, const void* src, std::size_t nbytes) {
    checkRt(
        cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream_),
        "cudaMemcpyAsync(D2D)");
  }

  // --- CE -> host notification (stream-memop write to a host-pinned flag) ---

  /**
   * Once preceding CE work on the stream completes, write `value` to
   * `hostFlagDev`; the host observes it by spinning on the host alias.
   */
  void enqueueNotify(uint64_t* hostFlagDev, uint64_t value) {
    checkCu(
        pfn_cuStreamWriteValue64(
            reinterpret_cast<CUstream>(stream_),
            reinterpret_cast<CUdeviceptr>(hostFlagDev),
            value,
            CU_STREAM_WRITE_VALUE_DEFAULT),
        "cuStreamWriteValue64(notify)");
  }

  // --- host -> CE release (gate the stream on a host-written flag) ---

  /** Block the stream until `flagDev` >= `value`, written by the host. */
  void enqueueWaitValue(uint64_t* flagDev, uint64_t value) {
    checkCu(
        pfn_cuStreamWaitValue64(
            reinterpret_cast<CUstream>(stream_),
            reinterpret_cast<CUdeviceptr>(flagDev),
            value,
            CU_STREAM_WAIT_VALUE_GEQ),
        "cuStreamWaitValue64(release)");
  }

  // --- host-driven RDMA (delegated to P2pIbrcHostWriter) ---

  /**
   * Host writer for one peer's ring (post RDMA put/signal from the CPU).
   *
   * Throws if this engine is Copy-Engine-only, and -- via the transport -- if a
   * live P2pIbrcHostLanes already holds `peerRank`. Rings take one logical
   * producer, so a peer is driven by lanes or by bare writers, never both; a
   * collective that stripes to a peer should reach that peer only through its
   * lanes object.
   */
  P2pIbrcHostWriter hostWriter(int peerRank, uint32_t queueIndex = 0) {
    if (transport_ == nullptr) {
      throw std::runtime_error(
          "HostCollectiveEngine: hostWriter() requires the transport-bound "
          "constructor; this engine was built Copy-Engine-only");
    }
    return transport_->getHostWriter(peerRank, queueIndex);
  }

 private:
  /*
   * The stream-memop entry points are resolved lazily (prims never links
   * libcuda, so the wheel still imports on a driverless host). Resolving in
   * the constructor keeps every method below free of a null-pointer check and
   * turns an unavailable driver into a construction-time error rather than a
   * crash on the first enqueue. Idempotent and thread-safe.
   */
  static void initDriver() {
    if (cuda_driver_lazy_init() != 0) {
      throw std::runtime_error(
          "HostCollectiveEngine: CUDA driver entry points unavailable");
    }
  }

  static void checkRt(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
      throw std::runtime_error(
          std::string("HostCollectiveEngine: ") + what + ": " +
          cudaGetErrorString(e));
    }
  }

  static void checkCu(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
      const char* s = nullptr;
      pfn_cuGetErrorString(r, &s);
      throw std::runtime_error(
          std::string("HostCollectiveEngine: ") + what + ": " +
          (s != nullptr ? s : "?"));
    }
  }

  MultipeerIbrcTransport* transport_{nullptr};
  // Always the caller's; never created or destroyed here.
  cudaStream_t stream_{nullptr};
};

} // namespace comms::prims
