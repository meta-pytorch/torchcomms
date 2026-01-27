// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/NcclxApi.hpp"

// Check NCCL version at compile time
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 25, 0)
#error \
    "NCCL version less than 2.25 is not supported. Please upgrade your NCCL installation."
#endif

namespace torch::comms {

// NCCLXException implementation

NCCLXException::NCCLXException(
    NcclxApi& nccl_api,
    const std::string& message,
    ncclResult_t result,
    ncclComm_t comm)
    : message_(
          message + ": " + nccl_api.getErrorString(result) +
          " \nNCCL Last Error: " + nccl_api.getLastError(comm)),
      result_(result) {}

const char* NCCLXException::what() const noexcept {
  return message_.c_str();
}

ncclResult_t NCCLXException::getResult() const noexcept {
  return result_;
}

// DefaultNcclxApi implementation

const char* DefaultNcclxApi::getErrorString(ncclResult_t result) {
  return ncclGetErrorString(result);
}

std::string DefaultNcclxApi::getLastError(ncclComm_t comm) {
  const char* error = ncclGetLastError(comm);
  return error != nullptr ? std::string(error) : std::string();
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
      "NCCLX alltoallvDynamicDispatch is not supported in this build");
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
      "NCCLX alltoallvDynamicCombine is not supported in this build");
#endif
}

ncclResult_t DefaultNcclxApi::alltoallvDedupInit(
    const size_t totalNumSendBlocks,
    const size_t blockCount,
    const size_t blockNumRecvBuckets,
    const int numRecvBuckets,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    void** request) {
#ifdef NCCL_ALLTOALLV_DEDUP_SUPPORTED
  ncclx::Hints hints;
  return ncclx::allToAllvDedupInit(
      totalNumSendBlocks,
      blockCount,
      blockNumRecvBuckets,
      numRecvBuckets,
      hints,
      datatype,
      comm,
      stream,
      request);
#else
  throw std::logic_error(
      "NCCLX alltoallvDedupInit is not supported in this build");
#endif
}

ncclResult_t DefaultNcclxApi::alltoallvDedupExec(
    const void* sendBuff,
    const int* sendIdx,
    const int* fwdIdx,
    const int* recvIdx,
    void* recvBuff,
    int recvBlockIds[],
    void* request) {
#ifdef NCCL_ALLTOALLV_DEDUP_SUPPORTED
  return ncclx::allToAllvDedupExec(
      sendBuff, sendIdx, fwdIdx, recvIdx, recvBuff, recvBlockIds, request);
#else
  throw std::logic_error(
      "NCCLX allToAllvDedupExec is not supported in this build");
#endif
}

ncclResult_t DefaultNcclxApi::alltoallvDedupCombine(
    const void* /* sendBuff */,
    const int* /* sendIdx */,
    const int* /* fwdIdx */,
    const int* /* recvIdx */,
    void* /* recvBuff */,
    void* /* request */) {
  // placeholder for now; will add support after landed NCCLX side
  throw std::logic_error(
      "NCCLX allToAllvDedupCombine is not supported in this build");
}

ncclResult_t DefaultNcclxApi::pFree(void* request) {
  return ncclx::pFree(request);
}

ncclResult_t DefaultNcclxApi::commWindowRegister(
    void* baseptr,
    const size_t size,
    ncclComm_t comm,
    NcclxWindow* winPtr) {
  return ncclCommWindowRegister(comm, baseptr, size, winPtr, NCCL_WIN_DEFAULT);
}

ncclResult_t DefaultNcclxApi::commWindowDeregister(
    ncclComm_t comm,
    NcclxWindow win) {
  return ncclCommWindowDeregister(comm, win);
}

ncclResult_t DefaultNcclxApi::winPut(
    const void* originBuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t targetOffsetNelems,
    NcclxWindow win,
    cudaStream_t stream) {
  return ncclPut(
      originBuff, count, datatype, peer, targetOffsetNelems, win, stream);
};

ncclResult_t DefaultNcclxApi::winSharedQuery(
    int rank,
    ncclComm_t comm,
    NcclxWindow win,
    void** addr) {
  return ncclWinSharedQuery(rank, comm, win, addr);
}

ncclResult_t
DefaultNcclxApi::winSignal(int peer, NcclxWindow win, cudaStream_t stream) {
  return ncclSignal(peer, 0, peer, win, stream);
}

ncclResult_t
DefaultNcclxApi::winWaitSignal(int peer, NcclxWindow win, cudaStream_t stream) {
  return ncclWaitSignal(peer, win, stream);
}

ncclResult_t DefaultNcclxApi::winGetAttributes(
    int peer,
    NcclxWindow win,
    NcclxWindowAttr* attrPtr) {
  return ncclWinGetAttributes(peer, win, attrPtr);
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

} // namespace torch::comms
