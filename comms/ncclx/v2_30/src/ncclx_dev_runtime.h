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

struct ncclComm;
struct ncclWindow_vidmem;
struct ncclDevrLocalWindow;
struct ncclDevrWindow;

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
