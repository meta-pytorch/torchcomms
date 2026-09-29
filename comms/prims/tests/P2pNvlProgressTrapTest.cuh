// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cuda_runtime.h>

namespace comms::prims::test {

/*
 * Failure modes of the NVL progress API that end in a device trap. Each runs in
 * its own binary: a trap aborts the process, so it cannot share one with cases
 * that must survive.
 */
enum class NvlProgressTrapCase : int {
  // Stalled send with an AbortBehavior::TRAP handle. The poll must trap rather
  // than unwinding, which is what distinguishes TRAP from SKIP.
  AbortTrapBehavior = 0,
  // Second init_send_progress while the channel is still Active.
  ReinitWhileActive = 1,
  // Progress entry point on a transport built without progress storage.
  NullProgressStorage = 2,
};

/*
 * Builds a single-GPU P2pNvlTransportDevice with hand-made state and drives it
 * into `testCase`. Self-loopback: the staging buffers are local, so nothing
 * ever returns credit and a stalled send stays stalled.
 *
 * Synchronizes internally and returns the resulting CUDA status. The wait must
 * happen here rather than in the caller: the `Abort` backing the device handle
 * is owned by this function, and `AbortDevice` is a non-owning view that must
 * not outlive it.
 */
cudaError_t launchNvlProgressTrap(NvlProgressTrapCase testCase);

} // namespace comms::prims::test
