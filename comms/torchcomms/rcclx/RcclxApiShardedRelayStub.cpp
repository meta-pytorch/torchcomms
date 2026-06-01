// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Stub implementation of the RcclxApi methods that depend on rcclx-dev-only
// symbols (`ncclx::Hints`, `ncclx::allGatherInit`, `ncclx::allGatherExec`,
// `ncclx::pFree`, `ncclShardedRelayMultiGroupAllReduce`). These symbols are
// not yet present in the frozen rcclx-stable / rcclx-last-stable snapshots
// under `comms/rcclx/snapshots/`, so this translation unit provides
// compilable no-op overrides that return `ncclInternalError` at runtime.
// This keeps `DefaultRcclxApi` instantiable when building against
// rcclx-stable / rcclx-last-stable while preserving compile-time safety.
//
// The selection between this file and `RcclxApiShardedRelay.cpp` is made
// at the BUCK level via `select()` on the active rccl constraint — see
// `comms/torchcomms/rcclx/BUCK`. Callers that need the sharded-relay path
// must build with `-m rcclx_dev` (or otherwise opt into rcclx-dev) so the
// real implementations are linked instead of these stubs.

#include "comms/torchcomms/rcclx/RcclxApi.hpp"

namespace torch::comms {

ncclResult_t DefaultRcclxApi::allGatherInit(
    void* /*recvbuff*/,
    size_t /*maxRecvCount*/,
    const RcclxHints& /*hints*/,
    ncclDataType_t /*datatype*/,
    ncclComm_t /*comm*/,
    hipStream_t /*stream*/,
    void** /*request*/) {
  return ncclInternalError;
}

ncclResult_t DefaultRcclxApi::allGatherExec(
    const void* /*sendbuff*/,
    size_t /*count*/,
    ncclDataType_t /*datatype*/,
    void* /*request*/) {
  return ncclInternalError;
}

ncclResult_t DefaultRcclxApi::pFree(void* /*request*/) {
  return ncclInternalError;
}

ncclResult_t DefaultRcclxApi::shardedRelayMultiGroupAllReduce(
    const void* const* /*sendBuffs*/,
    void* const* /*recvBuffs*/,
    const size_t* /*counts*/,
    ncclDataType_t /*datatype*/,
    ncclRedOp_t /*op*/,
    ncclComm_t /*comm*/,
    hipStream_t /*stream*/,
    const int* const* /*allActiveRanks*/,
    int /*nActiveRanksPerGroup*/,
    int /*nGroups*/) {
  return ncclInternalError;
}

} // namespace torch::comms
