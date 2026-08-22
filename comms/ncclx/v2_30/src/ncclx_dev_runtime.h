/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCLX_DEV_RUNTIME_H_
#define NCCLX_DEV_RUNTIME_H_

#include <cstddef>
#include <cstdint>

#include "nccl.h"
#include "nccl_device.h"

struct ncclComm;
struct ncclWindow_vidmem;
struct ncclDevrWindow;

// Local-only window for source buffers (non-collective registration). Uses the
// parent's PD but skips the rkey allGather; usable only as the source of a put.
// NOTE: the first 5 fields (memory through winFlags) must match ncclDevrWindow
// (in dev_runtime.h) so winFlags can be read at a matching offset to distinguish
// window types. See task T282779046 for enforcing this layout at compile time.
struct ncclDevrLocalWindow {
  void* memory;      // nullptr for local-only windows (no ncclDevrMemory)
  void* userPtr;
  size_t size;
  size_t bigOffset;  // 0 for local-only windows (no big VA space mapping)
  int winFlags;      // Must be at same offset as ncclDevrWindow::winFlags
  void* localRegHandle;
  struct ncclWindow_vidmem* vidmem;
  void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS];
  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS];
};

// De-static'd upstream window-table init utility whose single definition lives
// in the forked dev_runtime.cc; declared here so the extracted NCCLX local-only
// window path (ncclx_dev_runtime.cc) can call it. The list* templates and the
// tiny ncclDevrWindowSorted record it also needs are handled inside
// ncclx_dev_runtime.cc (see the note there), so the forked dev_runtime.cc keeps
// its struct/declarations in place and needs no removals.
ncclResult_t symWindowTableInitOnce(struct ncclComm* comm, cudaStream_t stream);

// Local-only window registration for source buffers (non-collective). Uses the
// parent comm's PD but skips the rkey allGather, so a window can only be used as
// the source of a put. Extracted out of the forked upstream dev_runtime.cc to
// keep the fork's footprint minimal; the register / deregister entry points in
// dev_runtime.cc dispatch here on NCCL_WIN_LOCAL_ONLY.
ncclResult_t symLocalWindowCreate(
    struct ncclComm* comm, void* userPtr, size_t userSize, int winFlags, void* localReg,
    struct ncclWindow_vidmem** outWinDev, struct ncclDevrLocalWindow** outWin,
    cudaStream_t stream);

ncclResult_t symLocalWindowDestroy(
    struct ncclComm* comm, struct ncclWindow_vidmem* winDev, cudaStream_t stream);

#endif
