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

#include "nccl.h"

/**
 * Per-communicator scratch arena for the low-precision relay wire format.
 *
 * WHY NOT ScratchBufferCache
 *
 * The four collectives each keep a grow-only ScratchBufferCache backed by
 * cudaMallocAsync on the caller's stream. That is the right tool for staging
 * that only the owning translation unit sees, and it is deliberately left
 * alone. It is the wrong tool for the LP buffers, for three reasons:
 *
 *   1. It allocates on the CALL PATH. sharded_relay_graph_scratch.h documents
 *      three separate ways that breaks under HIP graph capture: an allocation
 *      inside a capture records a graph allocation node whose address is only
 *      valid while that graph runs; a growth inside a capture records a free
 *      node, so the second replay double-frees; and a growth after a capture
 *      returns a pointer the graph already baked in. A region allocated ONCE,
 *      outside any capture, and never resized has none of those failure modes.
 *   2. One arena is shared by all four collectives. LP buffers are the same
 *      shape whichever collective asks, so per-TU pools would just be four
 *      copies of the same high-water mark.
 *   3. Its addresses move. A deterministic partition from a fixed base gives a
 *      captured graph the same addresses on every replay for free.
 *
 * Lifecycle mirrors the one-shot region exactly: created either at
 * ncclCommInitRank (NCCL_SHARDED_RELAY_LP_PREALLOC=1) or lazily on the first
 * call that asks for low precision, kept for the communicator's life, and freed
 * from RCCL's commFree().
 *
 * COLLECTIVE, AND WHY THAT MATTERS
 *
 * lpArenaAcquire() runs a bootstrap all-gather on first use, so EVERY rank of
 * the communicator must call it on the same call -- including ranks that are
 * pure passthrough helpers for this collective and need no bytes from the arena
 * at all. Callers must therefore gate it on a predicate derived only from
 * sizes, never on whether the calling rank happens to be active. That is the
 * same contract sharded_relay_oneshot.h states, for the same reason.
 *
 * The all-gather is not decoration. Availability HAS to be agreed: a rank with
 * an arena quantizes and sends wire bytes while a rank without one sends its
 * dtype, the two disagree on how many bytes cross the link, and the call hangs
 * or corrupts rather than degrading. A plain local hipMalloc failure on one
 * rank would produce exactly that, which is why the outcome is voted on rather
 * than decided locally.
 *
 * CAPACITY IS CHECKED BY THE CALLER, AGAINST A RANK-INDEPENDENT NUMBER
 *
 * The arena never grows, so a call whose footprint exceeds it declines. That
 * check belongs in the caller's size-only gate and must use a WORST-CASE
 * requirement across roles, compared against lpArenaCapacityBytes() -- both
 * rank-independent, so every rank declines together. A rank comparing its own
 * role's requirement would be back to per-rank divergence.
 */
namespace rcclx::relay {

// Bytes provisioned per communicator, a pure function of
// NCCL_SHARDED_RELAY_LP_MAX_MSG_MB and therefore identical on every rank.
//
// The budget is stated in FULL-PRECISION per-rank message bytes, because that
// is the number a caller knows. bf16 sets the worst case at a given byte
// budget: it is the smallest supported element, so a byte budget buys twice as
// many elements as fp32 does, and the LP footprint scales with ELEMENTS.
//
// The widest shape is the A=4 all-gather, which needs one send shadow of the
// rank's own shard plus A-1 foreign arrival slots of the same size, so the
// budget is multiplied by kLpArenaShadowsPerMessage. At the 1024 MB default
// that is 2112 MiB per communicator -- a real number, and the reason
// NCCL_SHARDED_RELAY_LP_PREALLOC defaults off: nothing is provisioned until a
// caller actually asks for low precision. A job whose messages are smaller
// should say so; NCCL_SHARDED_RELAY_LP_MAX_MSG_MB=256 costs 528 MiB.
size_t lpArenaCapacityBytes();

// The multiplier above, named so the collectives in later commits can raise it
// in one place if a shape turns out to need more than four shadows.
inline constexpr size_t kLpArenaShadowsPerMessage = 4;

/**
 * The whole arena, handed to one call.
 *
 * Not a sub-allocation: the arena has one owner per call because the relay
 * collectives on a given stream are serialized by that stream. Partition it
 * with LpArenaCarver.
 */
struct LpArenaLease {
  char* base{nullptr};
  size_t bytes{0};

  bool valid() const {
    return base != nullptr && bytes > 0;
  }
};

/**
 * Deterministic bump partitioner over a lease.
 *
 * Every call partitions from the same base in the same order, so a captured
 * graph sees the same addresses on every replay -- which is most of the reason
 * the arena exists. Regions are 256-byte aligned, which keeps every LP buffer
 * start comfortably above the 4-byte alignment the inline scales need.
 *
 * take() returns nullptr once the arena is exhausted and latches ok() false, so
 * a caller can carve everything it needs and check once at the end instead of
 * after every region.
 */
class LpArenaCarver {
 public:
  static constexpr size_t kAlign = 256;

  explicit LpArenaCarver(const LpArenaLease& lease)
      : base_(lease.base), capacity_(lease.bytes) {}

  char* take(size_t bytes) {
    if (base_ == nullptr) {
      ok_ = false;
      return nullptr;
    }
    const size_t aligned = ((bytes + kAlign - 1) / kAlign) * kAlign;
    if (aligned > capacity_ - used_) {
      ok_ = false;
      return nullptr;
    }
    char* p = base_ + used_;
    used_ += aligned;
    return p;
  }

  bool ok() const {
    return ok_;
  }

  size_t used() const {
    return used_;
  }

 private:
  char* base_{nullptr};
  size_t capacity_{0};
  size_t used_{0};
  bool ok_{true};
};

/**
 * Hand out this communicator's arena, creating it on first use.
 *
 * Returns false if the arena is unavailable -- allocation failed on any rank,
 * or the communicator has none. A false return is identical on every rank, so
 * callers can fall back to full precision without risking a mixed-format hang.
 *
 * COLLECTIVE on first call for a given communicator. See the file comment.
 */
bool lpArenaAcquire(ncclComm_t comm, LpArenaLease* out);

/**
 * Build this communicator's arena now, during ncclCommInitRank.
 *
 * Only acts when NCCL_SHARDED_RELAY_LP_PREALLOC=1. Creation is collective, so
 * init is the one place it is free: every rank is already there, and doing it
 * here means no later call -- including the first call of a graph capture --
 * has to pay for it or refuse it.
 *
 * Like every NCCL parameter this assumes the variable is set identically on
 * every rank. Setting it on some ranks only would leave those ranks in a
 * bootstrap all-gather the others never join.
 *
 * Failure is not an error: low precision is optional and every caller has a
 * fallback, so a comm that cannot build an arena behaves as if the variable
 * were unset.
 */
void lpArenaInit(ncclComm_t comm);

/**
 * True if this communicator's arena already exists, without creating one.
 *
 * For callers under graph capture. Creation is not capturable -- it runs a
 * bootstrap all-gather -- but an arena that already exists is perfectly usable
 * from a captured kernel, so a caller consults this and declines to full
 * precision rather than refusing every captured call outright. Whether an arena
 * exists is agreed across the communicator, because it is only ever established
 * by an all-gather every rank takes part in, so every rank gets the same
 * answer.
 */
bool lpArenaReady(ncclComm_t comm);

/**
 * Free a communicator's arena. Called from RCCL's commFree(), the only point at
 * which the relay learns a comm is going away. Purely local -- one hipFree --
 * so it honours commFree()'s no-sync-among-ranks contract. A no-op for a comm
 * that never asked for low precision.
 */
void lpArenaRelease(ncclComm_t comm);

} // namespace rcclx::relay
