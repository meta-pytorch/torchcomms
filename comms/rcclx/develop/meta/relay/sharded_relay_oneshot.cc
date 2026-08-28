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
#include "meta/relay/sharded_relay_route.h"

namespace rcclx::relay {

namespace {

// Per-peer slot capacity. The largest per-peer contribution any relay
// collective stages is bounded by the per-active-rank input size, so the gate
// doubles as the slot size: nRanks * kRelayOneShotMaxBytes per rank in total,
// which at 8 ranks and a 1 MiB gate is 8 MiB.
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

struct Region {
  // Guards this region's construction and its epoch counter.
  //
  // Per-comm rather than process-global because createRegion() runs two
  // bootstrap all-gathers. Holding one global lock across a collective
  // deadlocks as soon as two communicators are set up concurrently and their
  // peer processes reach the collectives in opposite order: each process would
  // hold its own lock inside one comm's all-gather while the peer it is waiting
  // for is blocked on that same lock, unable to enter it. regionsMutex()
  // therefore guards only the map.
  std::mutex mu;
  // The communicator this region belongs to. Kept so a POINTER that has been
  // recycled by the allocator can be told apart from the comm it used to be.
  const void* commPtr{nullptr};
  bool tried{false};
  bool valid{false};
  void* base{nullptr}; // local allocation, exported
  OneShotPeerTable table{};
  size_t slotBytes{0};
  int nRanks{0};
  uint32_t epoch{0};
  // Peer pointers we opened and must close.
  void* openedPeers[kOneShotMaxRanks]{};
};

std::mutex& regionsMutex() {
  static std::mutex m;
  return m;
}

// Keyed on comm->commHash, NOT on the ncclComm_t pointer.
//
// Keying on the pointer deadlocks. A destroyed communicator leaves its entry
// behind (there is no relay-visible hook on comm teardown), and the allocator
// may hand the same address to the NEXT communicator -- on some ranks but not
// others, since each rank is its own process with its own heap. A rank that
// finds the stale entry skips creation, a rank that does not enters
// bootstrapAllGather, and the participation mismatch hangs the bootstrap.
// Observed exactly that way: a suite stuck 20 minutes in bootstrapAllGather
// under oneShotAcquire.
//
// commHash is identical across the ranks of one communicator and different for
// a different communicator, so it is consistent where the pointer is not.
std::map<uint64_t, Region>& regions() {
  static std::map<uint64_t, Region> r;
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

  const size_t stagingBytes = static_cast<size_t>(nRanks) * kSlotBytes;
  const size_t flagBytes =
      static_cast<size_t>(nRanks) * kOneShotMaxBlocks * sizeof(uint32_t);

  bool ok = true;

  // IPC requires a non-mempool allocation: mempool-backed (hipMallocAsync)
  // memory cannot be exported, which is why this does not use
  // ScratchBufferCache.
  if (hipExtMallocWithFlags(
          &reg.base, stagingBytes + flagBytes, kRegionAllocFlags) !=
      hipSuccess) {
    reg.base = nullptr;
    ok = false;
  }
  if (ok && hipMemset(reg.base, 0, stagingBytes + flagBytes) != hipSuccess) {
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

bool oneShotAcquire(ncclComm_t comm, OneShotLaunch* out) {
  if (comm == nullptr || out == nullptr) {
    return false;
  }
  if (comm->nRanks > kOneShotMaxRanks || comm->nRanks < 2) {
    return false;
  }

  // regionsMutex() covers only the map -- the stale-pointer sweep and the
  // lookup. It is deliberately NOT held across createRegion(), which runs two
  // bootstrap all-gathers: one global lock spanning a collective deadlocks as
  // soon as two communicators are set up concurrently and their peer processes
  // reach the collectives in opposite order.
  Region* regp = nullptr;
  {
    std::lock_guard<std::mutex> mapLock(regionsMutex());

    // Drop any region left by a previous communicator that occupied this
    // address. Its IPC mappings refer to memory that went away with it. That
    // comm is gone, so nothing can be holding the region's mu.
    for (auto it = regions().begin(); it != regions().end();) {
      if (it->second.commPtr == static_cast<const void*>(comm) &&
          it->first != comm->commHash) {
        destroyRegion(it->second);
        it = regions().erase(it);
      } else {
        ++it;
      }
    }

    regp = &regions()[comm->commHash];
  }
  // std::map is node based, so the reference stays valid once the map lock is
  // dropped; only erasing this key invalidates it, and that happens at comm
  // destroy or in the sweep above, both after the last acquire for that comm.
  Region& reg = *regp;

  std::lock_guard<std::mutex> lock(reg.mu);
  if (!reg.tried) {
    reg.tried = true;
    reg.commPtr = static_cast<const void*>(comm);
    createRegion(comm, reg);
  }
  if (!reg.valid) {
    return false;
  }

  // The epoch must be unique per launch and must advance identically on every
  // rank, which it does because every rank takes exactly one epoch per
  // collective call. Bumping it under this comm's lock keeps two streams on one
  // communicator from taking the same value.
  reg.epoch += 1;

  out->table = reg.table;
  out->slotBytes = reg.slotBytes;
  out->nRanks = reg.nRanks;
  out->epoch = reg.epoch;
  return true;
}

void oneShotRelease(ncclComm_t comm) {
  if (comm == nullptr) {
    return;
  }
  // Erasing the node destroys Region::mu, so this must not run concurrently
  // with an oneShotAcquire() for the same comm -- it is called from comm
  // teardown, after the last collective on that comm has been issued.
  std::lock_guard<std::mutex> lock(regionsMutex());
  auto it = regions().find(comm->commHash);
  if (it == regions().end()) {
    return;
  }
  destroyRegion(it->second);
  regions().erase(it);
}

} // namespace rcclx::relay
