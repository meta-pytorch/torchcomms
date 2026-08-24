/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "gin.h"
#include "gin/gin_host.h"
#include "transport.h"

// ncclGinRegisterLocal / ncclGinDeregisterLocal are NCCLX additions: local-only,
// non-collective GIN registration for source buffers. They were extracted out of
// the forked upstream gin/gin_host.cc to keep the fork's footprint minimal; the
// declarations remain in include/gin/gin_host.h so callers are unaffected.

// Local-only registration for source buffers (non-collective)
// Uses the shared ginState to get the parent's PD, but skips the rkey allGather.
// GIN must already be connected before calling this function.
ncclResult_t ncclGinRegisterLocal(struct ncclComm* comm, void* address, size_t size,
                                  void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                                  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS]) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;

  // GIN must already be connected
  if (!ginState->connected) {
    ERR(ncclInvalidUsage, "ncclGinRegisterLocal: GIN not connected.");
    return ncclInvalidUsage;
  }

  for (int n = 0; n < ginState->ginCommCount; n++) {
    if (ginState->ginType == NCCL_GIN_TYPE_PROXY) {
      // Proxy path not yet supported for local-only registration
      ERR(ncclInvalidUsage, "ncclGinRegisterLocal: Proxy path not yet supported");
      return ncclInvalidUsage;
    } else {
      NCCLCHECK(ginState->ncclGin->regMrLocal(ginState->ginComms[n], address, size, NCCL_PTR_CUDA, 0,
                                              &ginHostWins[n], &ginDevWins[n]));
    }
    if (ginHostWins[n] == NULL) {
      ERR(ncclSystemError, "rank %d - GIN Local register failed: buff %p, size %ld", comm->rank, address, size);
      return ncclSystemError;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinDeregisterLocal(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  for (int n = 0; n < ginState->ginCommCount; n++) {
    if (ginState->ginType == NCCL_GIN_TYPE_PROXY) {
      ERR(ncclInvalidUsage, "ncclGinDeregisterLocal: Proxy path not yet supported");
      return ncclInvalidUsage;
    } else {
      NCCLCHECK(ginState->ncclGin->deregMrLocal(ginState->ginComms[n], ginHostWins[n]));
    }
  }
  return ncclSuccess;
}
