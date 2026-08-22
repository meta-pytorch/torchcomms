/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "ncclx_dev_runtime.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "comm.h"
#include "dev_runtime.h"
#include "device.h"
#include "nccl_device.h"
#include "nccl_device/core.h"
#include "transport.h"
#include "utils.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#else
#include "gin/gin_host.h"
#endif

// symWindowTableInitOnce (declared in ncclx_dev_runtime.h) and the list*
// templates below are de-static'd upstream utilities whose single definitions
// live in the forked dev_runtime.cc; the extern-template declarations bind the
// uses here to dev_runtime.cc's explicit instantiation rather than instantiating
// a second copy. The tiny ncclDevrWindowSorted record is duplicated here rather
// than exported from the forked dev_runtime.cc, so that upstream file needs no
// struct/declaration removal; keep this definition byte-identical to the one in
// dev_runtime.cc.
struct ncclDevrWindowSorted {
  uintptr_t userAddr;
  size_t size;
  struct ncclDevrWindow* win;
};

template <typename Obj, typename Key>
int listFindSortedLub(Key Obj::*key, Obj* sorted, int count, Key arg);
template <typename Obj>
void listInsert(Obj** list, int* capacity, int* count, int index, Obj val);
template <typename Obj>
void listRemove(Obj* list, int* count, int index);

extern template int listFindSortedLub<ncclDevrWindowSorted, uintptr_t>(
    uintptr_t ncclDevrWindowSorted::*, ncclDevrWindowSorted*, int, uintptr_t);
extern template void listInsert<ncclDevrWindowSorted>(
    ncclDevrWindowSorted**, int*, int*, int, ncclDevrWindowSorted);
extern template void listRemove<ncclDevrWindowSorted>(
    ncclDevrWindowSorted*, int*, int);

////////////////////////////////////////////////////////////////////////////////
// Local-only window functions for source buffers (non-collective registration).
// Uses parent's PD but skips rkey allGather. Can only be used as source for put.

ncclResult_t symLocalWindowCreate(
    struct ncclComm* comm, void* userPtr, size_t userSize, int winFlags, void* localReg,
    struct ncclWindow_vidmem** outWinDev, struct ncclDevrLocalWindow** outWin,
    cudaStream_t stream
  ) {
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr);
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclDevrLocalWindow* win;

  win = (struct ncclDevrLocalWindow*)malloc(sizeof(struct ncclDevrLocalWindow));
  memset(win, 0, sizeof(*win));
  win->memory = nullptr;   // No ncclDevrMemory for local-only windows
  win->userPtr = userPtr;
  win->size = userSize;
  win->bigOffset = 0;      // No big VA space mapping for local-only windows
  win->winFlags = winFlags;
  win->localRegHandle = localReg;

  // Register with GIN using local-only registration (no allGather).
  // GIN must already be connected via parent comm.
  NCCLCHECK(ncclGinRegisterLocal(comm, userPtr, userSize, win->ginHostWins, win->ginDevWins));

  struct ncclWindow_vidmem* winDev;
  struct ncclWindow_vidmem* winDevHost;
  NCCLCHECK(ncclShadowPoolAlloc(&devr->shadows, &winDev, &winDevHost, stream));
  win->vidmem = winDev;

  // For local-only windows, we don't have lsaFlatBase mapping (no collective).
  // Set lsaFlatBase to the user's pointer directly (only valid for local access).
  winDevHost->lsaFlatBase = (char*)userPtr;
  winDevHost->mcOffset4K = 0;  // Not applicable for local-only
  winDevHost->stride4G = 0;    // Not applicable for local-only
  winDevHost->lsaRank = devr->lsaSelf;
  winDevHost->worldRank = comm->rank;
  winDevHost->winHost = (void*)win;
  winDevHost->ginOffset4K = 0;  // Offset within local buffer
  for (int i = 0; i < NCCL_GIN_MAX_CONNECTIONS; i++) {
    winDevHost->ginWins[i] = win->ginDevWins[i];
  }
  CUDACHECK(cudaMemcpyAsync(winDev, winDevHost, sizeof(struct ncclWindow_vidmem), cudaMemcpyHostToDevice, stream));

  NCCLCHECK(symWindowTableInitOnce(comm, stream)); // ensure devr->windowTable exists
  struct ncclDevCommWindowTable* tableDev = devr->windowTable;
  while (true) {
    struct ncclDevCommWindowTable* tableHost;
    NCCLCHECK(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost));
    int i = 0;
    while (i < 32 && tableHost->entries[i].window != nullptr) i += 1;
    if (i < 32) {
      tableHost->entries[i].base = userAddr;
      tableHost->entries[i].size = userSize;
      tableHost->entries[i].window = winDev;
      CUDACHECK(cudaMemcpyAsync(&tableDev->entries[i], &tableHost->entries[i], sizeof(tableHost->entries[i]), cudaMemcpyHostToDevice, stream));
      break;
    }
    if (tableHost->next == nullptr) {
      NCCLCHECK(ncclShadowPoolAlloc<ncclDevCommWindowTable>(&devr->shadows, &tableHost->next, nullptr, stream));
      CUDACHECK(cudaMemcpyAsync(&tableDev->next, &tableHost->next, sizeof(tableHost->next), cudaMemcpyHostToDevice, stream));
    }
    tableDev = tableHost->next;
  }

  { // insert into winSorted[]
    int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, userAddr);
    struct ncclDevrWindowSorted winSort;
    winSort.userAddr = userAddr;
    winSort.size = userSize;
    // Note: We store nullptr for local-only windows in winSorted.win since it's a different type.
    // This is safe because winSorted is only used for lookups, not for type-specific operations.
    winSort.win = nullptr;
    listInsert(&devr->winSorted, &devr->winSortedCapacity, &devr->winSortedCount, i, winSort);
  }

  if (outWinDev) *outWinDev = winDev;
  if (outWin) *outWin = win;
  return ncclSuccess;
}

ncclResult_t symLocalWindowDestroy(struct ncclComm* comm, struct ncclWindow_vidmem* winDev, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclWindow_vidmem* winDevHost;
  struct ncclDevrLocalWindow* winHost;

  NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, winDev, &winDevHost), ret, fail);
  winHost = (struct ncclDevrLocalWindow*)winDevHost->winHost;

  // Deregister from GIN using local-only deregistration.
  NCCLCHECKGOTO(ncclGinDeregisterLocal(comm, winHost->ginHostWins), ret, remove_table);

remove_table:
  { struct ncclDevCommWindowTable* tableDev = devr->windowTable;
    while (true) {
      struct ncclDevCommWindowTable* tableHost;
      NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost), ret, remove_winSorted);
      int i = 0;
      while (i < 32 && tableHost->entries[i].window != winDev) i += 1;
      if (i < 32) {
        memset(&tableHost->entries[i], 0, sizeof(tableHost->entries[i]));
        CUDACHECKGOTO(cudaMemsetAsync(&tableDev->entries[i], 0, sizeof(tableDev->entries[i]), stream), ret, remove_winSorted);
        break;
      }
      if (tableHost->next == nullptr) break; // Error didn't find window in table
      tableDev = tableHost->next;
    }
  }
  NCCLCHECKGOTO(ncclShadowPoolFree(&devr->shadows, winDev, stream), ret, remove_winSorted);

  if (winHost->localRegHandle != nullptr) {
    NCCLCHECKGOTO(ncclCommDeregister(comm, winHost->localRegHandle), ret, remove_winSorted);
  }

remove_winSorted:
  { int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, reinterpret_cast<uintptr_t>(winHost->userPtr));
    i -= 1; // least upper bound is just after ours.
    listRemove(devr->winSorted, &devr->winSortedCount, i);
  }
  free(winHost);
fail:
  return ret;
}
