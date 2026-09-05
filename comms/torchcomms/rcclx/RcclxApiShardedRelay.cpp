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

#include <cstddef>

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
    int nGroups,
    int lowPrecision) {
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
      nGroups,
      lowPrecision);
}

ncclResult_t DefaultRcclxApi::shardedRelayMultiGroupReduceScatter(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    hipStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups,
    int lowPrecision) {
  return ncclShardedRelayMultiGroupReduceScatter(
      sendBuffs,
      recvBuffs,
      recvCounts,
      datatype,
      op,
      comm,
      stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups,
      lowPrecision);
}

ncclResult_t DefaultRcclxApi::shardedRelayMultiGroupAllToAll(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* segmentCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    hipStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups,
    int lowPrecision) {
  return ncclShardedRelayMultiGroupAllToAll(
      sendBuffs,
      recvBuffs,
      segmentCounts,
      datatype,
      comm,
      stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups,
      lowPrecision);
}

ncclResult_t DefaultRcclxApi::shardedRelayMultiGroupAllGather(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* sendCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    hipStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups,
    int lowPrecision) {
  return ncclShardedRelayMultiGroupAllGather(
      sendBuffs,
      recvBuffs,
      sendCounts,
      datatype,
      comm,
      stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups,
      lowPrecision);
}

// Hop three of the plan's wire path: RcclxRelayPlan -> ncclRelayPlanInfo -> the
// internal RelayPlanInfo. The second hop is guarded where both of its types are
// visible; this is the only translation unit where THESE two are, so the guard
// belongs here. Five adjacent same-typed uint32s copied by hand is precisely
// the shape in which a transposition is invisible -- two swapped zero-valued
// fields change nothing observable until something finally reads them.
//
// This cannot be a memcpy, unlike the second hop: RcclxRelayPlan deliberately
// carries no reserved words, so the two records are different sizes. What must
// hold is that the fields it does carry sit at the same offsets.
static_assert(
    sizeof(ncclRelayPlanInfo) == 32,
    "ncclRelayPlanInfo is a wire format; its size is part of the protocol");
static_assert(
    offsetof(RcclxRelayPlan, nCalls) == offsetof(ncclRelayPlanInfo, nCalls) &&
        offsetof(RcclxRelayPlan, opCode) ==
            offsetof(ncclRelayPlanInfo, opCode) &&
        offsetof(RcclxRelayPlan, dtype) == offsetof(ncclRelayPlanInfo, dtype) &&
        offsetof(RcclxRelayPlan, redOp) == offsetof(ncclRelayPlanInfo, redOp) &&
        offsetof(RcclxRelayPlan, flags) == offsetof(ncclRelayPlanInfo, flags),
    "RcclxRelayPlan and ncclRelayPlanInfo must agree field for field");

ncclResult_t DefaultRcclxApi::relayControlPublish(
    ncclComm_t comm,
    uint64_t epoch,
    const RcclxRelayPlan& plan,
    const size_t* counts,
    int64_t timeoutNs) {
  ncclRelayPlanInfo info{};
  info.nCalls = plan.nCalls;
  info.opCode = plan.opCode;
  info.dtype = plan.dtype;
  info.redOp = plan.redOp;
  info.flags = plan.flags;
  return ncclRelayControlPublish(comm, epoch, &info, counts, timeoutNs);
}

ncclResult_t DefaultRcclxApi::relayControlConsume(
    ncclComm_t comm,
    uint64_t epoch,
    RcclxRelayPlan* plan,
    size_t* counts,
    uint32_t countsCapacity,
    int64_t timeoutNs) {
  ncclRelayPlanInfo info{};
  const ncclResult_t res = ncclRelayControlConsume(
      comm, epoch, &info, counts, countsCapacity, timeoutNs);
  // Copied out on success, and on the one failure that carries information: an
  // over-capacity plan reports the number of calls the caller needed room for,
  // which is the only way to recover from it. Mirrors what
  // ncclRelayControlConsume itself does, and for the same reason -- on any
  // other failure `info` is untouched, and a zeroed record is nCalls 0 with
  // opCode 0, which is a valid ncclRelayOpShutdown rather than a marker for "no
  // plan".
  if (res == ncclSuccess ||
      (res == ncclInvalidArgument && info.nCalls > countsCapacity)) {
    plan->nCalls = info.nCalls;
    plan->opCode = info.opCode;
    plan->dtype = info.dtype;
    plan->redOp = info.redOp;
    plan->flags = info.flags;
  }
  return res;
}

} // namespace torch::comms
