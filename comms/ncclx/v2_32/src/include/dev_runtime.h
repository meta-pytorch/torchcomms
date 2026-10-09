/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_RUNTIME_H_
#define NCCL_DEVICE_RUNTIME_H_
#include "nccl.h"
#include "nccl_device.h"
#include "nccl_common.h"
#include "allocator.h"
#include "bitops.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////////////////
// ncclDevr[_]: runtime implements for symmetric API.

struct ncclDevrMemory;

// No public NCCL_WIN_REGISTER_* flag means all capabilities. Specifying one or more
// registration flags selects only those capabilities. Add future public flags here.
enum ncclDevrRegisterCapability {
  ncclDevrRegisterGin = 1 << 0,
  ncclDevrRegisterLsa = 1 << 1,
  ncclDevrRegisterCft = 1 << 2,
  ncclDevrRegisterRma = 1 << 3,
  ncclDevrRegisterAll = ncclDevrRegisterGin | ncclDevrRegisterLsa | ncclDevrRegisterCft | ncclDevrRegisterRma,
};

bool ncclDevrWinRegEnabled(int winFlags, enum ncclDevrRegisterCapability capability);

struct ncclDevrWindow {
  struct ncclDevrMemory* memory;
  void* userPtr;
  size_t size;
  size_t bigOffset; // Offset in big VA space.
  int winFlags;
  void* localRegHandle;
  struct ncclWindow_vidmem* vidmem; // key for intrusive map
  struct ncclDevrWindow* next; // next for intrusive map
  struct ncclComm* comm; // comm for intrusive map window <> comm look up
};

// [NCCLX] Local-only window for source buffers (non-collective registration).
// Uses the parent comm's PD but skips the rkey allGather, so it can only be the
// source of a device-side GIN put.
//
// ncclWindow_vidmem::winHost may point at either window type, and the type is
// recovered by reading winFlags for NCCL_WIN_LOCAL_ONLY. That only works while the
// leading members below stay at the same offsets as in ncclDevrWindow; the
// static_asserts under this struct enforce it.
struct ncclDevrLocalWindow {
  void* memory;     // always nullptr -- local-only windows have no ncclDevrMemory
  void* userPtr;
  size_t size;
  size_t bigOffset; // always 0 -- no big VA space mapping
  int winFlags;
  void* localRegHandle;
  struct ncclWindow_vidmem* vidmem;
  void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS];
  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS];
  struct ncclDevrLocalWindow* next; // ncclDevrState::localWinHead
};

static_assert(offsetof(struct ncclDevrLocalWindow, winFlags) == offsetof(struct ncclDevrWindow, winFlags),
              "ncclDevrLocalWindow::winFlags must alias ncclDevrWindow::winFlags -- the window-type check reads it "
              "through whichever type ncclWindow_vidmem::winHost happens to point at");
static_assert(offsetof(struct ncclDevrLocalWindow, userPtr) == offsetof(struct ncclDevrWindow, userPtr),
              "ncclDevrLocalWindow prefix must match ncclDevrWindow");
static_assert(offsetof(struct ncclDevrLocalWindow, size) == offsetof(struct ncclDevrWindow, size),
              "ncclDevrLocalWindow prefix must match ncclDevrWindow");
// The members below are not read through the pun today, but they are the ones
// that would silently corrupt if they ever were -- memory in particular is
// always null for a local window, so anything reaching it via a punned pointer
// dereferences null rather than misbehaving quietly.
static_assert(offsetof(struct ncclDevrLocalWindow, memory) == offsetof(struct ncclDevrWindow, memory),
              "ncclDevrLocalWindow prefix must match ncclDevrWindow");
static_assert(offsetof(struct ncclDevrLocalWindow, bigOffset) == offsetof(struct ncclDevrWindow, bigOffset),
              "ncclDevrLocalWindow prefix must match ncclDevrWindow");
static_assert(offsetof(struct ncclDevrLocalWindow, localRegHandle) == offsetof(struct ncclDevrWindow, localRegHandle),
              "ncclDevrLocalWindow prefix must match ncclDevrWindow");
static_assert(offsetof(struct ncclDevrLocalWindow, vidmem) == offsetof(struct ncclDevrWindow, vidmem),
              "ncclDevrLocalWindow prefix must match ncclDevrWindow");

// [NCCLX] Whether ncclWindow_vidmem::winHost points at a local-only window. The
// winFlags read through ncclDevrLocalWindow is what the asserts above protect.
static inline bool ncclDevrWinIsLocalOnly(void const* winHost) {
  return winHost != nullptr && (static_cast<struct ncclDevrLocalWindow const*>(winHost)->winFlags & NCCL_WIN_LOCAL_ONLY);
}
struct ncclDevrWindowSorted;
struct ncclDevrTeam;

struct ncclDevrRegTask {
  struct ncclDevrRegTask* next;
  void* userPtr;
  size_t userSize;
  int winFlags;
  ncclWindow_t* outWinDev;
};

struct ncclDevrCommCreateTask {
  struct ncclDevrCommCreateTask* next;
  struct ncclDevCommRequirements* reqs;
  struct ncclDevComm* outDevComm;
  uint32_t deviceCodeVersion;
};

struct ncclDevrStateCftUc {
  ncclCftLeId baseId;
};

struct ncclDevrState {
  // Like localRank/localRanks except "lsa" ranks must be consecutive in the world
  // and all lsa subsets have the same number of ranks. If any condition is
  // false then the lsa team is just the singleton of self.
  int lsaSelf;
  int lsaSize;
  int* lsaRankList;
  int nLsaTeams;

  int cftSelf;
  int cftSize;
  int cftMcSelf;
  int cftMcSize;
  struct ncclDevrStateCftUc le[2]; // 0: UC LE ID base, 1: Counted UC LE ID base (rank_i le = base + i)

  size_t granularity; // cuMemGetAllocationGranularity
  bool ginEnabled;
  bool rmaProxyEnabled;
  struct ncclDevrMemory* memHead;
  uint64_t nextRegistryId; // next value for ncclDevrMemory::registryId
  struct ncclDevrWindowSorted* winSorted;
  int winSortedCapacity, winSortedCount;
  // [NCCLX] Local-only windows. Kept out of winSorted and the device window table:
  // a put takes its source window directly, so they never need address lookup.
  struct ncclDevrLocalWindow* localWinHead;
  struct ncclDevrTeam* teamHead;
  size_t bigSize; // size of our big logical space (128GB?)
  struct ncclSpace bigSpace; // allocates our big VA space.
  void* lsaFlatBase; // base ptr for all lsa ranks big VA's concatenated together: size = lsaRanks*bigSize
  struct ncclShadowPool shadows;
  struct ncclDevCommWindowTable* windowTable;

  struct ncclIntruQueue<struct ncclDevrRegTask, &ncclDevrRegTask::next> regTaskQueue;
  struct ncclIntruQueue<struct ncclDevrCommCreateTask, &ncclDevrCommCreateTask::next> commCreateTaskQueue;
};

struct ncclDevCommCompat {
  int minVersion, maxVersion;
  ncclResult_t (*commPropertiesFilter)(ncclComm_t comm, struct ncclCommProperties* props);
  ncclResult_t (*devCommRequirementsFilter)(ncclComm_t comm, ncclDevCommRequirements_t* reqs);
  ncclResult_t (*devCommCopyNewToOld)(ncclComm_t comm, void* oldDevComm, struct ncclDevComm const* newDevComm);
  ncclResult_t (*devCommCopyOldToNew)(ncclComm_t comm, struct ncclDevComm* newDevComm, void const* oldDevComm);
};

// Check if GIN resources have been requested as part of `reqs`.
bool ncclGinResourcesRequested(struct ncclDevCommRequirements const* reqs);

// Check if there is only one LSA team. This function uses the cached value of comm or computes the
// value from the comm topology.
bool ncclDevrIsOneLsaTeam(struct ncclComm* comm);

// Returns the CUDA version supported by CFT on this GPU, or 0 when CFT is unsupported.
ncclResult_t ncclGpuCftSupport(struct ncclComm* comm, int* gpuCftSupport, bool* gpuCftMulticastSupport,
                               bool* gpuCftCountedSupport);

// We assume ncclComm has a `ncclDevrState symState` member.
ncclResult_t ncclDevrInitOnce(struct ncclComm* comm);
ncclResult_t ncclDevrFinalize(struct ncclComm* comm);

// If found *outWinHost will be populated and *outWinId >= 0, otherwise *outWinId == -1
ncclResult_t ncclDevrFindWindow(struct ncclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin);

ncclResult_t ncclDevrWindowRegisterInGroup(struct ncclComm* comm, void* ptr, size_t size, int winFlags,
                                           ncclWindow_t* outWinDev);

ncclResult_t ncclDevrCommCreateInternal(struct ncclComm* comm, struct ncclDevCommRequirements* reqs,
                                        struct ncclDevComm* outDevComm, bool isInternal, uint32_t deviceCodeVersion);
void freeDevCommRequirements(struct ncclDevCommRequirements* reqs);

bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow* win);
bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow* win);

// Get the corresponding pointer in another lsa rank's symmetric memory window
ncclResult_t ncclDevrGetLsaRankPtr(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, int lsaRank,
                                   void** outPtr);

// Convert a world rank to an LSA rank.
ncclResult_t ncclDevrWorldToLsaRank(struct ncclComm* comm, int peerWorldRank, int* peerLsaRank);

// Get the RMA window handle for a specific context
void* ncclDevrGetRmaWin(struct ncclDevrWindow* winHost, int ctx);

// Get the byte offset of a window within its backing memory allocation.
size_t ncclDevrGetWinOffset(struct ncclDevrWindow* winHost);

// Get the multicast address for a given team
ncclResult_t ncclDevrGetLsaTeamPtrMC(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset,
                                     struct ncclTeam lsaTeam, void** outPtr);

#endif
