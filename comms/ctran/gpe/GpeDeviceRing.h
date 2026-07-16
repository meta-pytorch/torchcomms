// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_DEVICE_RING_H_
#define CTRAN_GPE_DEVICE_RING_H_

#include <atomic>
#include <cstddef>

#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>

#include "comms/ctran/algos/common/GpeRing.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBufferReader.h"

class CtranGpeCmd;

namespace ctran::gpe {

// Host-only reader over the device dispatch ring. Kept here rather than in
// GpeRing.h because GpeRing.h is device-included and must not pull in the host
// reader.
using GpeRingReader = hrdw_ring_buffer::
    HRDWRingBufferReader<GpeCmdId, kGpeRingScope, kGpeRingPolicy>;

} // namespace ctran::gpe

// Per-comm registry for device-ring dispatch (NCCL_CTRAN_GPE_DEVICE_RING).
// Maps the comm-local id a kernel publishes to the ring back to its
// CtranGpeCmd. Thread-safe: registerCmd on the main thread, erase on the CUDA
// cmdDestroy callback thread, lookup on the GPE worker thread. Kept separate
// from CtranGpe::Impl so id assignment, lookup, and per-fire bookkeeping are
// unit-testable without a GPU.
class GpeDeviceRingCmdRegistry {
 public:
  // Assign the next monotonic id (starting at 1; 0 = unassigned), store the
  // cmd, and mark it registered. Returns the id (also written to cmd->cmdId).
  ctran::gpe::GpeCmdId registerCmd(CtranGpeCmd* cmd);

  // Return the cmd for id, or nullptr if absent (e.g. already erased).
  CtranGpeCmd* lookup(ctran::gpe::GpeCmdId id) const;

  // Look up the cmd for id and, if found, recordRingReplay() it atomically
  // under the registry lock. Doing both under one lock closes the teardown race
  // with erase()/cmdDestroy() (see the impl). Returns the cmd or nullptr.
  CtranGpeCmd* lookupAndFire(ctran::gpe::GpeCmdId id);

  // Remove the entry for id. Idempotent.
  void erase(ctran::gpe::GpeCmdId id);

  // Number of live entries.
  size_t size() const;

  // Apply the per-replay bookkeeping the host-node cmdCb would have done: bump
  // inFlight (so cmdDestroy waits for the GPE to drain this fire) and advance
  // each op's opCount. Only meaningful for persistent (captured) cmds.
  static void recordRingReplay(CtranGpeCmd* cmd);

 private:
  std::atomic<ctran::gpe::GpeCmdId> nextId_{1};
  mutable folly::Synchronized<
      folly::F14FastMap<ctran::gpe::GpeCmdId, CtranGpeCmd*>>
      map_;
};

#endif // CTRAN_GPE_DEVICE_RING_H_
