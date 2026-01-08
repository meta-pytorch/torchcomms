/*************************************************************************
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Stub implementations for MSCCL functions when MSCCL is disabled
 ************************************************************************/

#include "nccl.h"
#include <hip/hip_runtime.h>
#include <cstddef>

// Forward declarations to avoid including MSCCL headers
struct ncclComm;
typedef struct ncclComm* ncclComm_t;

// mscclFunc_t enum stub
typedef enum { mscclFuncAllGather        =  0,
               mscclFuncAllReduce        =  1,
               mscclFuncBroadcast        =  2,
               mscclFuncReduce           =  3,
               mscclFuncReduceScatter    =  4,
               mscclFuncSend             =  5,
               mscclFuncRecv             =  6,
               mscclFuncGather           =  7,
               mscclFuncScatter          =  8,
               mscclFuncAllToAll         =  9,
               mscclFuncAllToAllv        =  10,
               mscclNumFuncs             =  11 } mscclFunc_t;

// mscclStatus struct stub
struct mscclStatus {
  bool needsProxy;
};

// Stub implementations that return false/no-op when MSCCL is disabled

bool mscclEnabled() {
  return false;
}

bool mscclForceEnabled() {
  return false;
}

void mscclSetIsCallerFlag() {
  // No-op
}

void mscclClearIsCallerFlag() {
  // No-op
}

bool mscclIsCaller() {
  return false;
}

bool mscclAvailable(const ncclComm_t comm) {
  (void)comm;  // Suppress unused parameter warning
  return false;
}

ncclResult_t mscclSchedulerInit(ncclComm_t comm, int* numChannelsRequired) {
  (void)comm;
  (void)numChannelsRequired;
  return ncclSuccess;
}

ncclResult_t mscclInit(ncclComm_t comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t mscclTeardown(ncclComm_t comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t mscclGroupStart() {
  return ncclSuccess;
}

ncclResult_t mscclGroupEnd() {
  return ncclSuccess;
}

ncclResult_t mscclEnqueueCheck(
    const void* sendbuff, const size_t sendcounts[], const size_t sdispls[],
    void* recvbuff, const size_t recvcounts[], const size_t rdispls[],
    size_t count, ncclDataType_t datatype, int root, int peer, ncclRedOp_t op,
    mscclFunc_t mscclFunc, ncclComm_t comm, hipStream_t stream) {
  (void)sendbuff; (void)sendcounts; (void)sdispls;
  (void)recvbuff; (void)recvcounts; (void)rdispls;
  (void)count; (void)datatype; (void)root; (void)peer;
  (void)op; (void)mscclFunc; (void)comm; (void)stream;
  return ncclInternalError;  // Should never be called when MSCCL is disabled
}

ncclResult_t mscclInternalSchedulerSelectAlgo(ncclComm_t comm, void* param) {
  (void)comm;
  (void)param;
  return ncclSuccess;
}

mscclStatus& mscclGetStatus(ncclComm_t comm) {
  (void)comm;
  static mscclStatus stubStatus;
  stubStatus.needsProxy = false;
  return stubStatus;
}
