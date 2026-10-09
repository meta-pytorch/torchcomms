/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_GIN_HOST_H_
#define _NCCL_GIN_HOST_H_

#include "allocator.h"
#include "nccl.h"
#include "nccl_gin.h"
#include "os.h"
#include "nccl_device/gin/gin_device_host_common.h"
#include <atomic>
#include <shared_mutex>
#include <thread>

struct ncclGinStateDevComm {
  int contextCount;
  int backendIndex;  // index into ncclGinState::backends[] for the backend that owns these contexts
  void* ginCtx[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetDeviceHandle_t* devHandles[NCCL_GIN_MAX_CONNECTIONS];
  struct ncclGinStateDevComm* next;
};

struct ncclGinBackendState {
  ncclGinType_t ginType;      // GIN backend type.
  ncclGin_t* ncclGin;
  void* ginInstance;          // Plugin's per-comm opaque context.
  int pluginIndex;            // Index into pluginLibs[].
  // [NCCLX] Loaded from an external plugin library. Its ncclGin_t may be an
  // upstream-sized struct, so the NCCLX members past finalize must not be read.
  bool isExternal;
  int ginCommCount;
  void* ginComms[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetProperties_t ginProps[NCCL_GIN_MAX_CONNECTIONS];
  bool supportsStrongSignals;
  bool supportsVASignals;
};

struct ncclGinState {
  ncclAffinity cpuAffinity;
  bool connected;
  bool supported;              // True if any backend is loaded on this comm.
  int proxyNthreads;           // Number of GIN progress threads.
  bool proxyThreadsCreated;     // Set once the GIN progress thread is spawned.
  std::atomic<bool> proxyThreadStopSignal;  // Signals the GIN progress thread to exit.
  // Use std::shared_timed_mutex (C++14) rather than std::shared_mutex (C++17)
  // because NCCL targets C++14 for CUDA < 13.
  std::shared_timed_mutex devCommRwMutex;  // Readers: proxy threads; Writer: main thread (setup/free).
  // When true, readers skip locking devCommRwMutex. Prevents writer starvation.
  std::atomic<bool> writePending;
  std::thread thread[NCCL_GIN_MAX_CONNECTIONS];
  ncclResult_t asyncResult;

  struct ncclGinStateDevComm* devComms;
  ncclGinConnectionType_t ginConnectionType;

  int numActiveBackends;
  struct ncclGinBackendState backends[NCCL_GIN_MAX_ACTIVE_BACKENDS];
};

extern int64_t ncclParamGinType();
extern int64_t ncclParamGinEnable();

// Get the GIN type from comm. ginType is set to the GIN type that can be used
// by the comm to communicate with other nodes.
ncclResult_t ncclGetGinType(struct ncclComm* comm, ncclGinType_t* ginType);
ncclResult_t ncclGetRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType);

// FIXME change to ncclGinState instead of ncclComm, no need to pass comm
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm);
ncclResult_t ncclGinHostFinalize(struct ncclComm* comm);
ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
                                 struct ncclDevComm* devComm, uint32_t deviceCodeVersion);
ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm);
ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS],
                             int winFlags, bool multiSegment = false, int memType = NCCL_PTR_CUDA);
// [NCCLX] Local-only registration for source buffers (non-collective).
// Uses each backend's existing PD but skips the rkey allGather, so the resulting
// window can only be the source of a device-side GIN put. GIN must already be
// connected. Backends that cannot register locally (the proxy, or a plugin without
// regMrLocal) are skipped and leave their slots null; backend 0 is required.
ncclResult_t ncclGinRegisterLocal(struct ncclComm* comm, void* address, size_t size, int winFlags,
                                  void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS],
                                  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS]);
ncclResult_t ncclGinDeregisterLocal(struct ncclComm* comm,
                                    void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS]);
ncclResult_t ncclGinDeregister(struct ncclComm* comm,
                               void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS]);

ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError);

ncclResult_t ncclGinSetDefaultBackend(struct ncclComm* comm, uint64_t globalBitmask);

#endif
