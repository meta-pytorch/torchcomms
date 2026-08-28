/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meta/relay/sharded_relay_oneshot.h"

#include <hip/hip_runtime.h>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "bootstrap.h"
#include "comm.h"
#include "debug.h"
#include "meta/relay/sharded_relay_route.h"

namespace rcclx::relay {

namespace {

// Per-peer slot capacity. The largest per-peer contribution any relay
// collective stages is bounded by the per-active-rank input size, so the gate
// doubles as the slot size: nRanks * kRelayOneShotMaxBytes in total, which at 8
// ranks and a 1 MiB gate is 8 MiB.
constexpr size_t kSlotBytes = kRelayOneShotMaxBytes;

// Uncached (HSA fine-grained) device memory, matching how RCCL allocates its
// own IPC-exported buffers in transport/p2p.cc. Not a tuning choice: the flags
// in this region are polled across processes from inside a kernel, and cached
// device memory does not reliably expose a peer's writes to that poll on
// gfx94x/gfx95x -- gfx9_threadfence.h's cheap fence is only correct for
// uncached buffers. Our build defines -DHIP_UNCACHED_MEMORY for exactly this
// reason ("needed for MI300 which will hang w/o", see rccl_build_config.bzl).
#if defined(HIP_UNCACHED_MEMORY)
constexpr unsigned int kRegionAllocFlags = hipDeviceMallocUncached;
#else
constexpr unsigned int kRegionAllocFlags = hipDeviceMallocFinegrained;
#endif

// Identity tag stamped at the end of every region. It answers the one question
// an IPC mapping cannot otherwise answer: is the memory this handle opened
// really the peer's CURRENT region, or a previous incarnation at an address the
// allocator recycled? Any rank can compute any other rank's expected value from
// the agreed commHash, so checking costs no extra communication.
constexpr size_t kSentinelBytes = sizeof(uint64_t);

uint64_t sentinelFor(uint64_t commHash, int rank) {
  return (commHash ^ 0x9E3779B97F4A7C15ull) * 0x100000001B3ull +
      static_cast<uint64_t>(rank) + 1ull;
}

struct Region {
  // Guards this region's construction and its epoch counter.
  //
  // Per-comm rather than process-global because createRegion() runs two
  // bootstrap all-gathers. Holding one global lock across a collective
  // deadlocks as soon as two communicators are set up concurrently and their
  // peer processes reach the collectives in opposite order: each process would
  // hold its own lock inside one comm's all-gather while the peer it is waiting
  // for is blocked on that same lock, unable to enter it. regionMutex()
  // therefore guards only the map.
  std::mutex mu;
  bool tried{false};
  bool valid{false};
  void* base{nullptr}; // local allocation, exported
  OneShotPeerTable table{};
  size_t slotBytes{0};
  int nRanks{0};
  uint32_t* seq{nullptr}; // device, kOneShotMaxBlocks entries
  void* openedPeers[kOneShotMaxRanks]{};
  // Identity of the communicator this region belongs to. The map is keyed on
  // the comm POINTER, which the allocator recycles; commHash distinguishes a
  // recycled address from the comm that used to live there, so a missed release
  // cannot be mistaken for a live region.
  uint64_t commHash{0};
};

std::mutex& regionMutex() {
  static std::mutex m;
  return m;
}

/**
 * One region per COMMUNICATOR, lazily created on that comm's first eligible
 * call and released from its teardown (see oneShotRelease, wired into RCCL's
 * commFree).
 *
 * Per-comm rather than per-process, and independent: each region owns its own
 * staging, its own flags and its own counters, so nothing about one comm's
 * one-shot traffic can reach another's. Two consequences that matter:
 *
 *  - The create decision is made from THIS comm's state alone, which starts
 *    uniformly absent on every rank of it. So all of a comm's ranks always
 * agree on whether to enter createRegion's collective bootstrap. An earlier
 *    process-global design decided it from whichever comm arrived first, which
 * let some ranks decline while others entered the bootstrap -- a hang.
 *  - Flags are per region, so concurrent collectives on DIFFERENT comms cannot
 *    alias each other's handshake.
 *
 * Lifetime is exactly the comm's, which is what removes the two failure modes
 * of the earlier attempts: nothing is leaked (release frees the allocation) and
 * nothing is evicted while live (so the epoch never rewinds under a peer that
 * is still waiting on it).
 */
std::map<const void*, Region>& regions() {
  static std::map<const void*, Region> r;
  return r;
}

void destroyRegion(Region& reg) {
  for (int i = 0; i < kOneShotMaxRanks; i++) {
    if (reg.openedPeers[i] != nullptr) {
      hipIpcCloseMemHandle(reg.openedPeers[i]);
      reg.openedPeers[i] = nullptr;
    }
  }
  if (reg.base != nullptr) {
    // Safe to free: NCCL requires comms to be used collectively, so every rank
    // has finished its collectives on this comm before any rank destroys it,
    // and no peer can still be reading this region. A peer that nonetheless
    // opened a mapping against a recycled address is caught by the sentinel
    // check in createRegion.
    hipFree(reg.base);
    reg.base = nullptr;
  }
  reg.valid = false;
}

/**
 * Build the region. Returns false on any local failure, but ALWAYS runs the two
 * bootstrap all-gathers so every rank stays in lockstep -- an early return on
 * one rank would desynchronize the bootstrap for the others.
 */
bool createRegion(ncclComm_t comm, Region& reg) {
  const int nRanks = comm->nRanks;
  const int rank = comm->rank;
  reg.nRanks = nRanks;
  reg.slotBytes = kSlotBytes;
  reg.commHash = comm->commHash;

  const size_t stagingBytes = static_cast<size_t>(nRanks) * kSlotBytes;
  const size_t flagBytes =
      static_cast<size_t>(nRanks) * kOneShotMaxBlocks * sizeof(uint32_t);
  // Trails the flags in the same allocation so the memset below zeroes it too,
  // which is what makes the first epoch 1 and every never-written flag compare
  // as not-yet-arrived.
  const size_t seqBytes =
      static_cast<size_t>(kOneShotMaxBlocks) * sizeof(uint32_t);

  const size_t sentinelOffset = stagingBytes + flagBytes + seqBytes;
  const size_t totalBytes = sentinelOffset + kSentinelBytes;

  bool ok = true;

  // IPC requires a non-mempool allocation: mempool-backed (hipMallocAsync)
  // memory cannot be exported, which is why this does not use
  // ScratchBufferCache.
  if (hipExtMallocWithFlags(&reg.base, totalBytes, kRegionAllocFlags) !=
      hipSuccess) {
    reg.base = nullptr;
    ok = false;
  }
  if (ok && hipMemset(reg.base, 0, totalBytes) != hipSuccess) {
    ok = false;
  }

  // Stamp our identity before the handle is published, so any peer that opens
  // this handle can prove the memory it got is this region.
  const uint64_t mySentinel = sentinelFor(comm->commHash, rank);
  if (ok &&
      hipMemcpy(
          static_cast<char*>(reg.base) + sentinelOffset,
          &mySentinel,
          kSentinelBytes,
          hipMemcpyHostToDevice) != hipSuccess) {
    ok = false;
  }

  hipIpcMemHandle_t myHandle{};
  std::memset(&myHandle, 0, sizeof(myHandle));
  if (ok && hipIpcGetMemHandle(&myHandle, reg.base) != hipSuccess) {
    ok = false;
  }

  // Exchange handles even if we failed locally: the all-gather is collective.
  std::vector<hipIpcMemHandle_t> handles(nRanks);
  std::memset(handles.data(), 0, handles.size() * sizeof(hipIpcMemHandle_t));
  handles[rank] = myHandle;
  if (bootstrapAllGather(
          comm->bootstrap, handles.data(), sizeof(hipIpcMemHandle_t)) !=
      ncclSuccess) {
    ok = false;
  }

  if (ok) {
    // Local only: a rank's own counter, never read through a peer mapping.
    reg.seq = reinterpret_cast<uint32_t*>(
        static_cast<char*>(reg.base) + stagingBytes + flagBytes);
    for (int r = 0; r < nRanks; r++) {
      if (r == rank) {
        reg.table.staging[r] = static_cast<char*>(reg.base);
        reg.table.flags[r] = reinterpret_cast<uint32_t*>(
            static_cast<char*>(reg.base) + stagingBytes);
        continue;
      }
      void* peerBase = nullptr;
      if (hipIpcOpenMemHandle(
              &peerBase, handles[r], hipIpcMemLazyEnablePeerAccess) !=
          hipSuccess) {
        ok = false;
        break;
      }
      reg.openedPeers[r] = peerBase;
      // Prove this mapping is peer r's current region. Without this a wrong
      // mapping is silent: our stores land in memory the peer no longer reads,
      // the peer never sees its epoch, and it spins in the kernel until the job
      // is killed. Rejecting here feeds the collective vote below, which turns
      // it into an agreed fallback to ncclSend/ncclRecv.
      uint64_t peerSentinel = 0;
      if (hipMemcpy(
              &peerSentinel,
              static_cast<char*>(peerBase) + sentinelOffset,
              kSentinelBytes,
              hipMemcpyDeviceToHost) != hipSuccess) {
        ok = false;
        break;
      }
      const uint64_t wantSentinel = sentinelFor(comm->commHash, r);
      if (peerSentinel != wantSentinel) {
        WARN(
            "Sharded relay one-shot: mapping for peer %d is not its current region (sentinel %llx, expected %llx); disabling the one-shot region for this communicator",
            r,
            static_cast<unsigned long long>(peerSentinel),
            static_cast<unsigned long long>(wantSentinel));
        ok = false;
        break;
      }
      reg.table.staging[r] = static_cast<char*>(peerBase);
      reg.table.flags[r] = reinterpret_cast<uint32_t*>(
          static_cast<char*>(peerBase) + stagingBytes);
    }
  }

  // Agree on the outcome. A region that only some ranks have is worse than no
  // region: the ranks that have it would spin for peers that took the
  // ncclSend/ncclRecv path, which hangs instead of degrading.
  std::vector<uint8_t> votes(nRanks, 0);
  votes[rank] = ok ? 1 : 0;
  if (bootstrapAllGather(comm->bootstrap, votes.data(), sizeof(uint8_t)) !=
      ncclSuccess) {
    ok = false;
  } else {
    for (int r = 0; r < nRanks; r++) {
      if (votes[r] == 0) {
        ok = false;
        break;
      }
    }
  }

  if (!ok) {
    destroyRegion(reg);
    return false;
  }
  reg.valid = true;
  return true;
}

} // namespace

bool oneShotReady(ncclComm_t comm) {
  if (comm == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(regionMutex());
  auto it = regions().find(static_cast<const void*>(comm));
  // find rather than operator[]: this must not create a Region entry, or a
  // capturing caller would leave behind a tried=false shell keyed to this comm.
  return it != regions().end() && it->second.valid &&
      it->second.commHash == comm->commHash;
}

bool oneShotAcquire(ncclComm_t comm, OneShotLaunch* out) {
  if (comm == nullptr || out == nullptr) {
    return false;
  }
  if (comm->nRanks > kOneShotMaxRanks || comm->nRanks < 2) {
    return false;
  }

  // regionMutex() covers only the map -- the staleness check and the lookup. It
  // is deliberately NOT held across createRegion(), which runs two bootstrap
  // all-gathers: one global lock spanning a collective deadlocks as soon as two
  // communicators are set up concurrently and their peer processes reach the
  // collectives in opposite order -- each process would hold its own lock
  // inside one comm's all-gather while the peer it is waiting for is blocked on
  // it.
  Region* regp = nullptr;
  {
    std::lock_guard<std::mutex> mapLock(regionMutex());

    // A stale entry can only exist if this comm's release was missed and the
    // allocator handed its address to a new comm. Drop the whole node rather
    // than hand back mappings into memory that went away with the old comm;
    // every rank of the new comm sees the same mismatch, because commHash is
    // agreed across it. Erasing here, under the map lock, is also what keeps
    // Region assignable-free: the node's mutex goes away with it, and the comm
    // it belonged to is gone so nobody can be holding that mutex.
    auto it = regions().find(static_cast<const void*>(comm));
    if (it != regions().end() && it->second.tried &&
        it->second.commHash != comm->commHash) {
      destroyRegion(it->second);
      regions().erase(it);
    }

    regp = &regions()[static_cast<const void*>(comm)];
  }
  // std::map is node based, so the reference stays valid once the map lock is
  // dropped; only erasing this key invalidates it, and that happens either in
  // the sweep above or in oneShotRelease(), both only for a comm that is done.
  Region& reg = *regp;

  std::lock_guard<std::mutex> lock(reg.mu);

  // Attempted once per comm. `tried` is sticky: a retry would have to be
  // collectively agreed, and a rank retrying alone would enter a bootstrap
  // all-gather the others are not in.
  if (!reg.tried) {
    reg.tried = true;
    createRegion(comm, reg);
  }
  if (!reg.valid) {
    return false;
  }

  out->table = reg.table;
  out->slotBytes = reg.slotBytes;
  out->nRanks = reg.nRanks;
  out->seq = reg.seq;
  return true;
}

void oneShotRelease(ncclComm_t comm) {
  if (comm == nullptr) {
    return;
  }
  // Erasing the node destroys Region::mu, so this must not run concurrently
  // with an oneShotAcquire() for the same comm -- it is called from comm
  // teardown, after the last collective on that comm has been issued.
  std::lock_guard<std::mutex> lock(regionMutex());
  auto it = regions().find(static_cast<const void*>(comm));
  if (it == regions().end()) {
    return;
  }
  destroyRegion(it->second);
  regions().erase(it);
}

} // namespace rcclx::relay
