// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

// NVIDIA-only: the stream memory ops below are CUDA driver API and have no HIP
// spelling, so the whole header elides on AMD. The guard only fires because the
// BUCK target passes -D__HIP_PLATFORM_AMD__ on the AMD build; without that the
// macro is undefined and hipify, which rewrites text regardless of preprocessor
// guards, would mistranslate the body. See the BUCK comment.
#if !defined(__HIP_PLATFORM_AMD__)

#include <cuda.h>
#include <cuda_runtime.h>

#include "comms/prims/core/Checks.h"
// Declares the pfn_cuStream* entry points the stream memory ops below call; the
// unused-include check does not see through the guarded block.
// NOLINTNEXTLINE(facebook-unused-include-check)
#include "comms/prims/platform/CudaDriverLazy.h"

namespace comms::prims {

/**
 * HostCollectiveEngine - thin host-side orchestration for 0-SM collectives.
 *
 * Ties together the SM-free primitives so a host-driven collective needs no
 * kernel: Copy Engine copies and CE<->host synchronization on host-pinned
 * flags. Chunking, peer selection and algorithm logic stay in the caller, as
 * does the transport: a collective that also needs host-driven RDMA reaches
 * its writer through `MultiPeerTransport::getHostWriter` directly.
 *
 * Assumes the CUDA driver entry points are already resolved, as the rest of
 * PRiMS does: the runtime calls cuda_driver_lazy_init() during setup, through
 * PrimsRuntime::create() -> MultiPeerTransport::prepareExchange(). A standalone
 * user that builds no transport must call it itself.
 *
 * Flags passed to the stream-memop methods must be GPU-accessible (host-pinned
 * + mapped, or device memory); the caller spins on the host alias of that same
 * memory to observe them. They must also be initialized before first use and
 * only ever increase: enqueueWaitValue() compares with
 * CU_STREAM_WAIT_VALUE_GEQ, so an uninitialized flag holding arbitrary memory
 * can satisfy the wait immediately and release the stream before the host has
 * written anything -- a silent race rather than an error.
 *
 * Not thread-safe: one owner thread, or external serialization across
 * setStream() and the enqueue methods.
 */
class HostCollectiveEngine {
 public:
  explicit HostCollectiveEngine(cudaStream_t stream) : stream_{stream} {}

  ~HostCollectiveEngine() = default;

  // Copy stays deleted so a cached engine cannot be duplicated behind the
  // owner's back; move is fine, since the only state is a handle this class
  // does not own.
  HostCollectiveEngine(const HostCollectiveEngine&) = delete;
  HostCollectiveEngine& operator=(const HostCollectiveEngine&) = delete;
  HostCollectiveEngine(HostCollectiveEngine&&) = default;
  HostCollectiveEngine& operator=(HostCollectiveEngine&&) = default;

  cudaStream_t stream() const {
    return stream_;
  }

  /**
   * Retarget every subsequent enqueue -- copies and stream memory ops alike --
   * at another stream. Call once per collective invocation: the engine is
   * cached for a communicator's lifetime while the caller's stream varies.
   * Running on the caller's stream is what keeps the collective
   * stream-ordered, with no host drain.
   *
   * Retargeting orders nothing. Work already enqueued on the previous stream
   * has no dependency on work enqueued after the switch, so a retarget between
   * an enqueueNotify() and the enqueueWaitValue() paired with it silently
   * drops the ordering the caller expects. The caller must ensure no dependent
   * work is still outstanding on the previous stream.
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
    checkCudaError(
        cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream_),
        "HostCollectiveEngine: cudaMemcpyAsync(D2D)");
  }

  // --- CE -> host notification (stream-memop write to a host-pinned flag) ---

  /**
   * Once preceding CE work on the stream completes, write `value` to
   * `hostFlagDev`; the host observes it by spinning on the host alias.
   */
  void enqueueNotify(uint64_t* hostFlagDev, uint64_t value) {
    checkCuError(
        pfn_cuStreamWriteValue64(
            stream_,
            reinterpret_cast<CUdeviceptr>(hostFlagDev),
            value,
            CU_STREAM_WRITE_VALUE_DEFAULT),
        "HostCollectiveEngine: cuStreamWriteValue64(notify)");
  }

  // --- host -> CE release (gate the stream on a host-written flag) ---

  /**
   * Block the stream until `flagDev` >= `value`, written by the host.
   *
   * Deliberately no CU_STREAM_WAIT_VALUE_FLUSH: that flag exists to push
   * outstanding remote writes through before downstream device work reads
   * them. A host-written release flag carries no remote payload, and an IBRC
   * arrival counter is fetch-added after its payload on the same RC QP, so the
   * payload is already visible once the wait succeeds. GEQ alone suffices for
   * both.
   */
  void enqueueWaitValue(uint64_t* flagDev, uint64_t value) {
    checkCuError(
        pfn_cuStreamWaitValue64(
            stream_,
            reinterpret_cast<CUdeviceptr>(flagDev),
            value,
            CU_STREAM_WAIT_VALUE_GEQ),
        "HostCollectiveEngine: cuStreamWaitValue64(release)");
  }

 private:
  // Always the caller's; never created or destroyed here.
  cudaStream_t stream_{nullptr};
};

} // namespace comms::prims

#endif // !defined(__HIP_PLATFORM_AMD__)
