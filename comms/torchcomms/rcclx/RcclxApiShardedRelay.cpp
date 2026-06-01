// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// rcclx-dev implementation of the RcclxApi methods that depend on symbols
// only present in rcclx-dev (the live trunk of comms/rcclx/develop). These
// symbols (`ncclx::Hints`, `ncclx::allGatherInit`, `ncclx::allGatherExec`,
// `ncclx::pFree`, `ncclShardedRelayMultiGroupAllReduce`) are not yet in the
// frozen rcclx-stable / rcclx-last-stable snapshots under
// `comms/rcclx/snapshots/`, so this translation unit must NOT be linked
// into binaries that build against those snapshots.
//
// The split is enforced at the BUCK level via `select()` on the active
// rccl constraint — see `comms/torchcomms/rcclx/BUCK`. The corresponding
// stub TU is `RcclxApiShardedRelayStub.cpp`, which provides drop-in
// replacements that return `ncclInternalError` so the class remains
// instantiable under rcclx-stable.

#include "comms/torchcomms/rcclx/RcclxApi.hpp"

namespace torch::comms {

ncclResult_t DefaultRcclxApi::allGatherInit(
    void* recvbuff,
    size_t maxRecvCount,
    const RcclxHints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    hipStream_t stream,
    void** request) {
  // Convert RcclxHints to ncclx::Hints
  ncclx::Hints ncclxHints;
  for (const auto& [key, value] : hints) {
    ncclxHints.set(key, value);
  }
  return ncclx::allGatherInit(
      recvbuff, maxRecvCount, ncclxHints, datatype, comm, stream, request);
}

ncclResult_t DefaultRcclxApi::allGatherExec(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    void* request) {
  return ncclx::allGatherExec(sendbuff, count, datatype, request);
}

ncclResult_t DefaultRcclxApi::pFree(void* request) {
  return ncclx::pFree(request);
}

ncclResult_t DefaultRcclxApi::shardedRelayMultiGroupAllReduce(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* counts,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    hipStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups) {
  return ncclShardedRelayMultiGroupAllReduce(
      sendBuffs,
      recvBuffs,
      counts,
      datatype,
      op,
      comm,
      stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
}

} // namespace torch::comms
