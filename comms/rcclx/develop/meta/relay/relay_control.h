/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "nccl.h"

/**
 * Host-side control plane for the sharded-relay collectives.
 *
 * WHY THIS EXISTS
 *
 * A relay collective is symmetric over the whole communicator: it happens only
 * when every rank's host thread independently enqueues it. Route selection
 * needs no agreement -- it is a pure function of size, evaluated per rank (see
 * sharded_relay_route.h) -- so nothing here negotiates a route.
 *
 * The problem is that a communicator is a DATA PLANE, NOT A SCHEDULER. In the
 * deployment this targets, the helper ranks are separate processes that do not
 * run the model: no batch, no forward pass, no scheduler. Nothing in ncclComm
 * can make such a process post a call. Per call a helper needs exactly two
 * dynamic things:
 *
 *   - `counts`, derived from the caller's token count, and
 *   - "does this call happen at all?", which depends on contiguity, alignment,
 *     compile state, capture state and phase -- all knowable ONLY on the ranks
 *     running the model.
 *
 * Everything else is static configuration, and a helper's buffers are a
 * one-element placeholder the kernel never reads.
 *
 * The integration that motivated this carried that payload over a TCP key/value
 * store, once per call, at ~0.9 ms -- against a collective that takes ~1 ms and
 * an internal relay/direct crossover at 1 MiB. The transport was ~1000x too
 * slow for the thing it was gating.
 *
 * WHAT MAKES A CHEAP FIX POSSIBLE
 *
 * The active/helper RENDEZVOUS is already handled on-device: the ncclSend and
 * ncclRecv pairs inside a relay schedule block until peers arrive. A helper
 * that enqueues early waits on the GPU; one that enqueues late makes the active
 * rank wait on the GPU. So this control plane needs CORRECT PROGRAM ORDER AND
 * ARGUMENTS, not low-latency wake-up.
 *
 * Hence: publish ONE plan per forward, let the helper run ahead and enqueue
 * every call the plan names, and let RCCL's P2P do the per-call rendezvous. The
 * active rank's host thread then does no control work per call at all.
 *
 * DESIGN
 *
 * One POSIX shm segment per (node, communicator), created at comm init and
 * named from commHash. Inside it, a ring of plan slots each guarded by a
 * seqlock (even = stable, odd = write in progress) plus a small header holding
 * the geometry, an abort flag and per-rank consumption progress.
 *
 * Deliberately absent:
 *
 *   - No route field. Route is a pure function of size, so every rank derives
 *     the same answer from the published counts with zero communication. A
 *     route field would be a second place for that mapping to drift.
 *   - No token count. The DECISION (counts) is published, never the INPUT (T),
 *     so the eligibility predicate has exactly one home: the active side. A
 *     helper that re-derived eligibility would be a second implementation of
 * it.
 *   - No device-visible flags. Producer and consumer are both host threads;
 *     nothing reads a plan from a kernel.
 *
 * COLLECTIVE, AND WHY THAT MATTERS
 *
 * Setup does a bootstrap all-gather, so every rank of the communicator must
 * take part, and the segment is enabled only if EVERY rank reports success --
 * the same unanimity rule, for the same reason, as the one-shot IPC region (see
 * sharded_relay_oneshot.h): a segment only some ranks have is worse than none,
 * because the ranks that have it wait for peers that took the other path, which
 * hangs rather than degrades.
 *
 * THE INVARIANT
 *
 * The control plane's decision point is strictly BEFORE ncclShardedRelay* is
 * called. After that point only RCCL's own abort machinery applies: a rank
 * already blocked inside a collective cannot be rescued by a flag in shared
 * memory. That is what makes the bounded wait below load-bearing rather than
 * decorative, and it is why publish() must be ordered before a forward's first
 * relay enqueue.
 */
namespace rcclx::relay {

// 'RLYC'. Stamped first and checked on attach, so a segment left behind by an
// unrelated program cannot be mistaken for ours.
constexpr uint32_t kRelayControlMagic = 0x524C5943u;

// Bump on ANY change to the header or slot layout. Checked on attach, which is
// what turns a mixed-build deployment into a loud failure at init instead of
// silent corruption at runtime.
constexpr uint32_t kRelayControlVersion = 1u;

// What a plan describes. Shutdown is expressed as an opcode rather than a
// separate entry point so a graceful stop needs no extra API.
enum RelayControlOp : uint32_t {
  kRelayOpShutdown = 0,
  kRelayOpAllReduce = 1,
  kRelayOpReduceScatter = 2,
  kRelayOpAllGather = 3,
  kRelayOpAllToAll = 4,
  kRelayOpCount = 5,
};

enum RelayControlAbort : uint32_t {
  kRelayAbortNone = 0,
  // Set by consume() itself when its budget runs out. Without this, one stuck
  // rank means every other rank independently burns its own full timeout and
  // the operator sees N confusing failures instead of one attributed cause.
  kRelayAbortTimeout = 1,
  // Set explicitly by a caller that is failing for its own reasons.
  kRelayAbortCaller = 2,
};

/**
 * The fixed-size part of a published plan.
 *
 * Mirrors the exported ncclRelayPlanInfo field for field. Counts travel
 * separately rather than inline, which is what lets the record stay a fixed 32
 * bytes -- ABI-stable, with reserved space to grow into -- while the per-plan
 * call capacity remains a RUNTIME parameter. Calls per forward is chunk count
 * in the workload this targets, and roughly two orders of magnitude larger once
 * attention all-to-all is covered, so a compile-time ceiling would have been
 * wrong in the direction of travel.
 */
struct RelayPlanInfo {
  uint32_t nCalls{0}; // 0 with opCode kRelayOpShutdown means shutdown
  uint32_t opCode{0};
  uint32_t dtype{0}; // ncclDataType_t
  uint32_t redOp{0}; // ncclRedOp_t
  uint32_t flags{0}; // reserved, must be 0
  uint32_t reserved[3]{0u, 0u, 0u};
};
static_assert(
    sizeof(RelayPlanInfo) == 32,
    "RelayPlanInfo is a wire format; its size is part of the segment layout");

/**
 * Everything the segment's geometry and identity is derived from.
 *
 * All explicit, with no communicator anywhere, so the protocol can be tested
 * without a comm, a GPU or a bootstrap network -- which is the only way to get
 * a torn-read test that actually runs many writers and readers.
 */
struct RelayControlConfig {
  uint32_t nRanks{0};
  uint32_t nActive{0};
  uint32_t rank{0};
  uint64_t commHash{0};
  uint32_t ringDepth{0};
  uint32_t maxCalls{0};
};

/**
 * A ring of plan slots in POSIX shared memory.
 *
 * Not thread-safe: a comm's entry is serialized by the caller, matching NCCL's
 * own one-thread-per-comm contract.
 */
class RelayControlBlock {
 public:
  RelayControlBlock() = default;
  ~RelayControlBlock();

  // Mapping ownership is not copyable, and there is no reason to move one.
  RelayControlBlock(const RelayControlBlock&) = delete;
  RelayControlBlock& operator=(const RelayControlBlock&) = delete;

  /**
   * Create the segment, zero it, and stamp the header. Exactly one rank per
   * (node, communicator) may call this.
   *
   * Uses O_CREAT|O_EXCL so a collision is detected rather than silently shared.
   * A segment left behind by a crashed run is reclaimed only when its recorded
   * creator pid is gone -- see the implementation, which will not steal one
   * from a live process.
   */
  bool create(const RelayControlConfig& cfg);

  /**
   * Open a segment someone else created and prove it is the one we expect.
   *
   * Validates magic, version, commHash, geometry and creator liveness. The
   * geometry check is why ringDepth and maxCalls are recorded in the header at
   * all: they come from environment parameters, and a rank whose environment
   * disagrees must fail HERE, at init, rather than read a differently-shaped
   * slot at runtime.
   */
  bool attach(const RelayControlConfig& cfg);

  // Unmap, and unlink if we created it. Safe to call on an unopened block.
  void detach();

  bool valid() const {
    return base_ != nullptr;
  }

  /**
   * Publish one forward's plan.
   *
   * EXACTLY ONE RANK per communicator may publish. There is one ring, so two
   * publishers would race the seqlock and drive it backwards; because both know
   * the same token count their plans are byte-identical, so the damage would be
   * invisible in the data and surface only as a spurious desync elsewhere. The
   * first caller claims publishing and any other rank is rejected.
   *
   * Blocks while the slot this epoch lands on still holds a plan that a
   * registered consumer has not taken yet; dropping a plan would desynchronize
   * the communicator, so waiting is the only correct choice. Bounded, and
   * bounded waits report which rank was lagging.
   *
   * Returns ncclInvalidArgument if nCalls exceeds the segment's capacity, with
   * a log line naming the parameter to raise.
   */
  ncclResult_t publish(
      uint64_t epoch,
      const RelayPlanInfo& info,
      const size_t* counts,
      int64_t timeoutNs);

  /**
   * Take one forward's plan.
   *
   * A consumer REGISTERS itself here, on entry, and only from then on will a
   * publisher wait for it. That is a contract, not an oversight: a rank's role
   * is not knowable to the segment -- the active ranks attach too and never
   * consume, so waiting on everyone who attached would deadlock at once.
   *
   * The consequence is that a consumer must reach its first consume() before
   * the publisher completes `ringDepth` forwards. In practice that is free:
   * comm init is a bootstrap barrier both sides leave together, and the
   * publisher would have to finish ringDepth entire forwards, GPU work
   * included, before the helper executes one instruction of its loop. If it is
   * ever violated the late consumer gets the bounded desync error below rather
   * than silent corruption.
   *
   * `countsCapacity` is what stops a publisher from overrunning the caller's
   * buffer; on a too-small buffer `info` is still filled in so the caller can
   * see the size it needed.
   *
   * Returns ncclInternalError on timeout (having set the abort flag) or if a
   * peer aborted, and on a detected desync -- the slot having already advanced
   * a full ring past the requested epoch, which no amount of waiting can undo.
   */
  ncclResult_t consume(
      uint64_t epoch,
      RelayPlanInfo* info,
      size_t* counts,
      uint32_t countsCapacity,
      int64_t timeoutNs);

  // Publish a shutdown plan, so a helper loop exits promptly instead of
  // waiting out its timeout.
  ncclResult_t publishShutdown(uint64_t epoch, int64_t timeoutNs);

  void setAbort(uint32_t reason);
  uint32_t abortReason() const;
  uint32_t abortRank() const;

  // Largest nCalls ever published here. Logged at teardown so the capacity
  // parameter can be set from observation rather than from a guess.
  uint32_t highWaterCalls() const;

  uint32_t ringDepth() const;
  uint32_t maxCalls() const;

  /**
   * Raw consumer-progress word for a rank: 0 if it has never entered consume(),
   * 1 if it is consuming but has completed nothing, else completed-epoch + 2.
   * Exposed so tests can assert that a rejected plan was NOT marked consumed.
   */
  uint64_t consumerProgress(uint32_t rank) const;

  // Byte size of a segment with this geometry. Exposed for tests that fabricate
  // or corrupt a segment on purpose.
  static size_t
  segmentBytes(uint32_t nRanks, uint32_t ringDepth, uint32_t maxCalls);

  // The shm object name for a communicator, e.g. "/rcclx_relay_ctl_<hex>".
  static std::string nameFor(uint64_t commHash);

 private:
  uint8_t* slotAt(uint64_t epoch) const;
  uint64_t* slotSeq(uint64_t epoch) const;
  RelayPlanInfo* slotInfo(uint64_t epoch) const;
  uint64_t* slotCounts(uint64_t epoch) const;
  uint64_t* consumedArray() const;
  bool waitForSlotDrain(uint64_t epoch, int64_t timeoutNs);

  uint8_t* base_{nullptr};
  size_t bytes_{0};
  bool owner_{false};
  std::string name_;
  RelayControlConfig cfg_{};
};

/**
 * Build this communicator's control segment during ncclCommInitRank.
 *
 * Only acts when NCCL_SHARDED_RELAY_MODE_ENABLE=1 -- the same switch that
 * pre-creates the one-shot IPC region, so a relay user sets one variable rather
 * than two. Setup is COLLECTIVE (see the file comment), so init is the one
 * place it is free: every rank is already there, and no later call -- including
 * the first call of a graph capture, which could not run a bootstrap all-gather
 * anyway -- has to pay for it or refuse it.
 *
 * Failure is not an error. Every caller is expected to have a fallback, so a
 * comm that cannot build a segment simply behaves as if the variable were
 * unset.
 */
void relayControlInit(ncclComm_t comm);

/**
 * Drop this communicator's segment. Called from RCCL's commFree(), the only
 * point at which the relay learns a comm is going away. Purely local -- an
 * munmap and, for the creator, an unlink -- so it honours commFree()'s
 * no-sync-among-ranks contract. A no-op for a comm that never had one.
 */
void relayControlRelease(ncclComm_t comm);

// True if this comm has a usable segment, without creating one.
bool relayControlReady(ncclComm_t comm);

ncclResult_t relayControlPublish(
    ncclComm_t comm,
    uint64_t epoch,
    const RelayPlanInfo& info,
    const size_t* counts,
    int64_t timeoutNs);

ncclResult_t relayControlConsume(
    ncclComm_t comm,
    uint64_t epoch,
    RelayPlanInfo* info,
    size_t* counts,
    uint32_t countsCapacity,
    int64_t timeoutNs);

// Configured geometry, after environment overrides. Callers size their consume
// buffer from the capacity; tests use both to build a matching segment.
uint32_t relayControlConfiguredMaxCalls();
uint32_t relayControlConfiguredRingDepth();

} // namespace rcclx::relay
