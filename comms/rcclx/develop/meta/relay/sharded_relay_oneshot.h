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
 * Peer-addressable staging for the relay's one-shot small-message kernels.
 *
 * WHY THIS EXISTS
 *
 * Below ~576 KB a relay call contains no bandwidth term at all -- the measured
 * time is flat across 4 KB..576 KB -- so the only unit of cost is a launch.
 * Every collective got to >1x by removing launches, except reduce-scatter,
 * which cannot: its output is an arithmetic function of what arrives, and
 * RCCL's grouped-P2P kernel copies but does not reduce, so it needs two
 * launches where NCCL needs one. Deleting the trailing reduce outright still
 * measured 0.90-1.04x, because ONE ncclGroup of P2P ops costs ~0.038 ms while
 * NCCL's entire fused reduce_scatter costs ~0.035 ms. The ncclGroup itself is
 * the floor.
 *
 * The way past it is a single kernel that moves the data AND reduces it, with
 * no group machinery. RCCL ships exactly that (ncclSymRun_ReduceScatter_LL) but
 * it is unreachable on this platform: comm->symmetricSupport requires
 * ncclCuMemEnable(), and ncclIsCuMemSupported() returns 0 unconditionally on
 * AMD, so the symmetric window registration that would supply peer pointers can
 * never be enabled. Hence this: the same idea built on plain hipIpc*.
 *
 * DESIGN
 *
 * One region per communicator, created lazily on the first eligible call and
 * kept for the communicator's life. Each rank allocates
 *
 *     [nRanks slots of slotBytes][nRanks * kOneShotMaxBlocks flags]
 *
 * with hipExtMallocWithFlags, uncached (HSA fine-grained) because the flags are
 * polled across processes from inside a kernel -- and NOT the relay's
 * ScratchBufferCache, whose cudaMallocAsync (mempool-backed) allocations cannot
 * be IPC-exported --
 * exports one handle, and opens every peer's. A rank then reaches peer p's slot
 * s as `table.staging[p] + s * slotBytes`.
 *
 * Callers never register their own buffers. Measured on this build,
 * ncclCommRegister returns success with a NULL handle in 0.001 ms, i.e. it is a
 * no-op, so reading a peer's sendbuff directly is not available. Instead each
 * rank PUSHES its contribution into peers' pre-registered staging. That also
 * means the per-call cost is zero: nothing is registered or mapped per call.
 *
 * Setup cost, measured for 8 ranks: hipIpcGetMemHandle 0.012 ms +
 * bootstrapAllGather 1.7 ms + 7x hipIpcOpenMemHandle 0.65 ms, so ~1.7 ms once.
 *
 * COLLECTIVE, AND WHY THAT MATTERS
 *
 * oneShotAcquire() performs a bootstrap all-gather, so EVERY rank of the
 * communicator must call it on the same call -- including ranks that are
 * helpers for this collective and will not launch anything. Callers must
 * therefore gate it on a predicate derived only from sizes (which are
 * collective-consistent), never on whether the calling rank happens to be
 * active.
 *
 * Success is agreed across ranks before the region is used: each rank
 * all-gathers its own ok flag and the region is enabled only if every rank
 * succeeded. Without that, a partial failure would leave some ranks running a
 * one-shot kernel that spins for a peer that took the ncclSend path, which
 * hangs rather than degrades.
 */
namespace rcclx::relay {

// An 8-GPU node. Sized for the topology the relay targets rather than made
// dynamic, because the peer table is passed to a kernel by value.
constexpr int kOneShotMaxRanks = 8;

// Blocks handshake pairwise (block b waits only on the peer's block b), so
// there is no global barrier and no co-residency requirement: every block
// writes and flags BEFORE it waits, so a block that is not yet resident cannot
// hold another back.
constexpr int kOneShotMaxBlocks = 64;
constexpr int kOneShotThreads = 256;

// Where every rank's staging and flags live in THIS rank's address space. The
// self entry points at the local allocation; peers are IPC-mapped.
struct OneShotPeerTable {
  char* staging[kOneShotMaxRanks];
  uint32_t* flags[kOneShotMaxRanks];
};

// The communicator ranks of a group's active set, indexed by active index.
// Passed to the kernel by value so it can resolve "active index j" to a peer
// table slot.
struct OneShotRanks {
  int r[kOneShotMaxRanks];
};

// Everything one launch needs. Returned by value so the epoch is unique to the
// call that took it.
struct OneShotLaunch {
  OneShotPeerTable table;
  size_t slotBytes;
  int nRanks;
  uint32_t epoch;
};

/**
 * Hand out the communicator's one-shot region, creating it on first use.
 *
 * Returns false if the region is unavailable -- allocation failed, hipIpc*
 * failed, the communicator is larger than kOneShotMaxRanks, or any rank failed.
 * A false return is identical on every rank, so callers can fall back to the
 * ncclSend/ncclRecv path without risking a mixed-schedule hang.
 *
 * COLLECTIVE on first call for a given communicator. See the file comment.
 */
bool oneShotAcquire(ncclComm_t comm, OneShotLaunch* out);

/**
 * Drop a communicator's region. Safe to call for a communicator that never had
 * one. Intended for communicator teardown.
 */
void oneShotRelease(ncclComm_t comm);

} // namespace rcclx::relay
