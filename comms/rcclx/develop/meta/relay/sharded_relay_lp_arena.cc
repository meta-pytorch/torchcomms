/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meta/relay/sharded_relay_lp_arena.h"

#include <hip/hip_runtime.h>
#include <atomic>
#include <map>
#include <mutex>
#include <vector>

#include "bootstrap.h"
#include "comm.h"
#include "debug.h"
#include "meta/relay/sharded_relay_lp.h"

namespace rcclx::relay {

namespace {

struct Arena {
  // Guards this arena's construction.
  //
  // Per-comm rather than process-global because createArena() runs a bootstrap
  // all-gather. Holding one global lock across a collective deadlocks as soon
  // as two communicators are set up concurrently and their peer processes reach
  // the collectives in opposite order: each process would hold its own lock
  // inside one comm's all-gather while the peer it is waiting for is blocked on
  // that same lock. arenaMutex() therefore guards only the map.
  std::mutex mu;
  bool tried{false};
  // ATOMIC, unlike the rest, because these two are the only fields read under a
  // DIFFERENT lock than they are written under: createArena() writes them
  // holding mu, and lpArenaReady() reads them holding only arenaMutex(). Two
  // locks covering one location is a data race however careful each side is,
  // and this one decides whether low precision is on -- a rank that answers
  // differently from its peers is the mixed-format hang the whole design is
  // built to avoid, so it cannot be left to chance.
  //
  // commHash is written BEFORE valid flips and the reader tests valid FIRST, so
  // a reader that sees a valid arena can never pair it with a commHash that has
  // not been published yet.
  std::atomic<bool> valid{false};
  void* base{nullptr};
  size_t bytes{0};
  // Identity of the communicator this arena belongs to. The map is keyed on the
  // comm POINTER, which the allocator recycles; commHash distinguishes a
  // recycled address from the comm that used to live there, so a missed release
  // cannot be mistaken for a live arena.
  std::atomic<uint64_t> commHash{0};
};

std::mutex& arenaMutex() {
  static std::mutex m;
  return m;
}

// One arena per COMMUNICATOR, for the same reasons the one-shot region is
// per-comm: the create decision is then made from this comm's state alone,
// which starts uniformly absent on every rank of it, so all of a comm's ranks
// always agree on whether to enter createArena's collective bootstrap.
std::map<const void*, Arena>& arenas() {
  static std::map<const void*, Arena> a;
  return a;
}

void destroyArena(Arena& arena) {
  if (arena.base != nullptr) {
    hipFree(arena.base);
    arena.base = nullptr;
  }
  arena.bytes = 0;
  arena.valid = false;
}

/**
 * Build the arena. Returns false on any failure, but ALWAYS runs the bootstrap
 * all-gather so every rank stays in lockstep -- an early return on one rank
 * would desynchronize the vote for the others.
 */
bool createArena(ncclComm_t comm, Arena& arena) {
  arena.commHash = comm->commHash;

  const size_t wantBytes = lpArenaCapacityBytes();
  bool ok = true;

  // Plain device memory. Unlike the one-shot region this is never IPC-exported
  // and never polled across processes, so it needs neither
  // hipExtMallocWithFlags nor uncached pages -- it is written by this rank's
  // quantize kernels and by RCCL's own transports servicing ncclRecv, both of
  // which are fine with cached device memory.
  //
  // Not zeroed: every wire byte a call reads was written by that same call's
  // quantize pass or by the ncclRecv that filled it, so a memset would only add
  // a 2 GiB write to comm setup.
  if (hipMalloc(&arena.base, wantBytes) != hipSuccess) {
    arena.base = nullptr;
    ok = false;
    WARN(
        "Sharded relay: could not allocate the %zu MiB low-precision arena; low precision will be unavailable on this communicator",
        wantBytes >> 20);
  } else {
    arena.bytes = wantBytes;
  }

  // Agree on the outcome. An arena that only some ranks have is worse than no
  // arena: the ranks that have it would quantize and send wire bytes while the
  // others send their dtype, which hangs or corrupts instead of degrading.
  std::vector<uint8_t> votes(comm->nRanks, 0);
  votes[comm->rank] = ok ? 1 : 0;
  if (bootstrapAllGather(comm->bootstrap, votes.data(), sizeof(uint8_t)) !=
      ncclSuccess) {
    ok = false;
  } else {
    for (int r = 0; r < comm->nRanks; r++) {
      if (votes[r] == 0) {
        ok = false;
        break;
      }
    }
  }

  if (!ok) {
    destroyArena(arena);
    return false;
  }
  // LAST, and after commHash: this is what publishes the arena to
  // lpArenaReady(), which reads valid before commHash.
  arena.valid = true;
  INFO(
      NCCL_INIT,
      "Sharded relay: low-precision arena ready, %zu MiB",
      arena.bytes >> 20);
  return true;
}

} // namespace

size_t lpArenaCapacityBytes() {
  // Elements, not bytes, is what the LP footprint scales with, and bf16 is the
  // smallest supported element -- so a byte budget read as bf16 gives the
  // worst case for that budget.
  const size_t maxElems = lpMaxMsgBytes() / sizeof(uint16_t);
  return kLpArenaShadowsPerMessage * lpWireBytesRoundUp(maxElems);
}

void lpArenaInit(ncclComm_t comm) {
  if (comm == nullptr || !lpPrealloc()) {
    return;
  }
  // Discard the lease: this is only here to force creation. The arena is kept
  // on the comm, so the next real caller finds it ready.
  LpArenaLease unused{};
  (void)lpArenaAcquire(comm, &unused);
}

bool lpArenaReady(ncclComm_t comm) {
  if (comm == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(arenaMutex());
  // find rather than operator[]: this must not create an entry, or a capturing
  // caller would leave behind a tried=false shell keyed to this comm.
  //
  // arenaMutex() covers the MAP lookup only. valid and commHash are written by
  // createArena() under Arena::mu instead, which is why they are atomic -- see
  // the Arena declaration. Reading valid first is what makes the pair
  // consistent.
  auto it = arenas().find(static_cast<const void*>(comm));
  return it != arenas().end() && it->second.valid &&
      it->second.commHash == comm->commHash;
}

bool lpArenaAcquire(ncclComm_t comm, LpArenaLease* out) {
  if (comm == nullptr || out == nullptr || comm->nRanks < 2) {
    return false;
  }

  // arenaMutex() covers only the map -- the staleness sweep and the lookup. It
  // is deliberately NOT held across createArena(), which runs a bootstrap
  // all-gather: one global lock spanning a collective deadlocks as soon as two
  // communicators are set up concurrently and their peer processes reach the
  // collectives in opposite order.
  Arena* arenap = nullptr;
  {
    std::lock_guard<std::mutex> mapLock(arenaMutex());

    // A stale entry can only exist if this comm's release was missed and the
    // allocator handed its address to a new comm. Drop the whole node rather
    // than hand back a pointer into memory that went away with the old comm;
    // every rank of the new comm sees the same mismatch, because commHash is
    // agreed across it.
    auto it = arenas().find(static_cast<const void*>(comm));
    if (it != arenas().end() && it->second.tried &&
        it->second.commHash != comm->commHash) {
      destroyArena(it->second);
      arenas().erase(it);
    }

    arenap = &arenas()[static_cast<const void*>(comm)];
  }
  // std::map is node based, so the reference stays valid once the map lock is
  // dropped; only erasing this key invalidates it, and that happens either in
  // the sweep above or in lpArenaRelease(), both only for a comm that is done.
  Arena& arena = *arenap;

  std::lock_guard<std::mutex> lock(arena.mu);

  // Attempted once per comm. `tried` is sticky: a retry would have to be
  // collectively agreed, and a rank retrying alone would enter a bootstrap
  // all-gather the others are not in.
  if (!arena.tried) {
    arena.tried = true;
    createArena(comm, arena);
  }
  if (!arena.valid) {
    return false;
  }

  out->base = static_cast<char*>(arena.base);
  out->bytes = arena.bytes;
  return true;
}

void lpArenaRelease(ncclComm_t comm) {
  if (comm == nullptr) {
    return;
  }
  // Erasing the node destroys Arena::mu, so this must not run concurrently with
  // an lpArenaAcquire() for the same comm -- it is called from comm teardown,
  // after the last collective on that comm has been issued.
  std::lock_guard<std::mutex> lock(arenaMutex());
  auto it = arenas().find(static_cast<const void*>(comm));
  if (it == arenas().end()) {
    return;
  }
  destroyArena(it->second);
  arenas().erase(it);
}

} // namespace rcclx::relay
