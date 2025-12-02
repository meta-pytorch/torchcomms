// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/NcclxApi.hpp"

// Check NCCL version at compile time
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 25, 0)
#error \
    "NCCL version less than 2.25 is not supported. Please upgrade your NCCL installation."
#endif

namespace torch {
namespace comms {

// DefaultNcclxApi implementation

const char* DefaultNcclxApi::getErrorString(ncclResult_t result) {
  return ncclGetErrorString(result);
}

ncclResult_t DefaultNcclxApi::getUniqueId(ncclUniqueId* uniqueId) {
  return ncclGetUniqueId(uniqueId);
}

ncclResult_t DefaultNcclxApi::commInitRankConfig(
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId commId,
    int rank,
    ncclConfig_t* config) {
  return ncclCommInitRankConfig(comm, nranks, commId, rank, config);
}

ncclResult_t DefaultNcclxApi::commDestroy(ncclComm_t comm) {
  return ncclCommDestroy(comm);
}

ncclResult_t DefaultNcclxApi::commAbort(ncclComm_t comm) {
  return ncclCommAbort(comm);
}

ncclResult_t DefaultNcclxApi::commGetAsyncError(
    ncclComm_t comm,
    ncclResult_t* asyncError) {
  return ncclCommGetAsyncError(comm, asyncError);
}

ncclResult_t DefaultNcclxApi::commSplit(
    ncclComm_t comm,
    int color,
    int key,
    ncclComm_t* newcomm,
    ncclConfig_t* config) {
  return ncclCommSplit(comm, color, key, newcomm, config);
}

ncclResult_t DefaultNcclxApi::commRegister(
    ncclComm_t comm,
    void* buffer,
    size_t size,
    void** handle) {
  return ncclCommRegister(comm, buffer, size, handle);
}

ncclResult_t DefaultNcclxApi::commDeregister(ncclComm_t comm, void* handle) {
  return ncclCommDeregister(comm, handle);
}

ncclResult_t DefaultNcclxApi::send(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclSend(sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t DefaultNcclxApi::recv(
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}

ncclResult_t DefaultNcclxApi::broadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t DefaultNcclxApi::bcast(
    void* buff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclBcast(buff, count, datatype, root, comm, stream);
}

ncclResult_t DefaultNcclxApi::allReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t DefaultNcclxApi::reduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    int root,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclReduce(
      sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t DefaultNcclxApi::allGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

ncclResult_t DefaultNcclxApi::reduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclReduceScatter(
      sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

ncclResult_t DefaultNcclxApi::allToAll(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

ncclResult_t DefaultNcclxApi::allToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclAllToAllv(
      sendbuff,
      sendcounts,
      sdispls,
      recvbuff,
      recvcounts,
      rdispls,
      datatype,
      comm,
      stream);
}

ncclResult_t DefaultNcclxApi::alltoallvDynamicDispatch(
    const void* sendbuff,
    const size_t* sendSplitLengths,
    size_t numSendSplitLengths,
    const size_t* sendIndices,
    const size_t* sendIndicesBlockLengths,
    void* const* recvbuffs,
    size_t* recvAllSplitLengths,
    size_t maxSendcount,
    size_t maxRecvcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
#ifdef NCCL_ALLTOALLV_DYNAMIC_SUPPORTED
  ncclx::Hints hints;
  hints.set("ncclx_alltoallv_dynamic_sendbuffs_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_recvbuffs_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_sendcounts_location", "gpu");
  hints.set("ncclx_alltoallv_dynamic_max_sendcounts_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_max_recvcounts_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_actual_recvcounts_location", "gpu");
  return ncclx::alltoallvDynamicDispatch(
      sendbuff,
      sendSplitLengths,
      numSendSplitLengths,
      sendIndices,
      sendIndicesBlockLengths,
      recvbuffs,
      recvAllSplitLengths,
      maxSendcount,
      maxRecvcount,
      hints,
      datatype,
      comm,
      stream);
#else
  throw std::logic_error(
      "NCCL alltoallvDynamicDispatch is not supported in this build");
#endif
}

ncclResult_t DefaultNcclxApi::alltoallvDynamicCombine(
    const void* sendbuff,
    const size_t* sendSplitLengths,
    size_t numSendSplitLengths,
    const size_t* sendIndices,
    const size_t* sendIndicesBlockLengths,
    void* recvbuff,
    size_t maxSendcount,
    size_t maxRecvcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
#ifdef NCCL_ALLTOALLV_DYNAMIC_SUPPORTED
  ncclx::Hints hints;
  hints.set("ncclx_alltoallv_dynamic_sendbuffs_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_recvbuffs_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_sendcounts_location", "gpu");
  hints.set("ncclx_alltoallv_dynamic_max_sendcounts_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_max_recvcounts_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_actual_recvcounts_location", "gpu");
  return ncclx::alltoallvDynamicCombine(
      sendbuff,
      sendSplitLengths,
      numSendSplitLengths,
      sendIndices,
      sendIndicesBlockLengths,
      recvbuff,
      maxSendcount,
      maxRecvcount,
      hints,
      datatype,
      comm,
      stream);
#else
  throw std::logic_error(
      "NCCL alltoallvDynamicCombine is not supported in this build");
#endif
}

ncclResult_t DefaultNcclxApi::winAllocate(
    size_t size,
    ncclComm_t comm,
    void** baseptr,
    NcclxWindow* winPtr,
    bool cpuBuf,
    const size_t signal_size) {
#ifdef NCCL_RMA_SUPPORTED
  ncclx::Hints hints;
  hints.set("window_buffer_location", cpuBuf ? "cpu" : "gpu");
  hints.set("window_signal_size", std::to_string(signal_size));
  return ncclWinAllocate(size, comm, baseptr, winPtr, hints);
#else
  throw std::logic_error("NCCL RMA is not supported in this build");
#endif
}

ncclResult_t DefaultNcclxApi::winFree(ncclComm_t comm, NcclxWindow win) {
#ifdef NCCL_RMA_SUPPORTED
  return ncclWinFree(comm, win);
#else
  throw std::logic_error("NCCL RMA is not supported in this build");
#endif
}

ncclResult_t DefaultNcclxApi::winPut(
    const void* originBuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t targetDisp,
    NcclxWindow win,
    cudaStream_t stream) {
#ifdef NCCL_RMA_SUPPORTED
  return ncclPut(originBuff, count, datatype, peer, targetDisp, win, stream);
#else
  throw std::logic_error(
      "NCCL does not support window, NCCL_RMA_SUPPORTED is not set");
#endif
};

ncclResult_t DefaultNcclxApi::winSharedQuery(
    int rank,
    ncclComm_t comm,
    NcclxWindow win,
    void** addr) {
#ifdef NCCL_RMA_SUPPORTED
  return ncclWinSharedQuery(rank, comm, win, addr);
#else
  throw std::logic_error(
      "NCCL does not support window, NCCL_RMA_SUPPORTED is not set");
#endif
}

ncclResult_t DefaultNcclxApi::winSignal(
    size_t signalDisp,
    uint64_t signalVal,
    int peer,
    NcclxWindow win,
    cudaStream_t stream) {
#ifdef NCCL_RMA_SUPPORTED
  return ncclSignal(signalDisp, signalVal, peer, win, stream);
#else
  throw std::logic_error(
      "NCCL does not support window, NCCL_RMA_SUPPORTED is not set");
#endif
}

ncclResult_t DefaultNcclxApi::winWaitSignal(
    size_t signal_disp,
    uint64_t cmp_val,
    NcclxWindowCmpOp cmp_op,
    NcclxWindow win,
    cudaStream_t stream) {
#ifdef NCCL_RMA_SUPPORTED
  return ncclWaitSignal_v2(signal_disp, cmp_val, cmp_op, win, stream);
#else
  throw std::logic_error(
      "NCCL does not support window, NCCL_RMA_SUPPORTED is not set");
#endif
}

ncclResult_t DefaultNcclxApi::memAlloc(void** buff, size_t size) {
  return ncclMemAlloc(buff, size);
}

ncclResult_t DefaultNcclxApi::memFree(void* buff) {
  return ncclMemFree(buff);
}

ncclResult_t DefaultNcclxApi::groupStart() {
  return ncclGroupStart();
}

ncclResult_t DefaultNcclxApi::groupEnd() {
  return ncclGroupEnd();
}

ncclResult_t DefaultNcclxApi::commUserRank(const ncclComm_t comm, int* myRank) {
  return ncclCommUserRank(comm, myRank);
}

ncclResult_t DefaultNcclxApi::commCount(const ncclComm_t comm, int* count) {
  return ncclCommCount(comm, count);
}

ncclResult_t DefaultNcclxApi::redOpCreatePreMulSum(
    ncclRedOp_t* op,
    void* scalar,
    ncclDataType_t datatype,
    ncclScalarResidence_t residence,
    ncclComm_t comm) {
  return ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
}

ncclResult_t DefaultNcclxApi::redOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  return ncclRedOpDestroy(op, comm);
}

} // namespace comms
} // namespace torch
