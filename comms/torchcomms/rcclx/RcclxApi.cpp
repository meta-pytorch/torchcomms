// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rcclx/RcclxApi.hpp"

namespace torch {
namespace comms {

// DefaultRcclxApi implementation

const char* DefaultRcclxApi::getErrorString(ncclResult_t result) {
  return ncclGetErrorString(result);
}

ncclResult_t DefaultRcclxApi::getUniqueId(ncclUniqueId* uniqueId) {
  return ncclGetUniqueId(uniqueId);
}

ncclResult_t DefaultRcclxApi::commInitRankConfig(
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId commId,
    int rank,
    ncclConfig_t* config) {
  return ncclCommInitRankConfig(comm, nranks, commId, rank, config);
}

ncclResult_t DefaultRcclxApi::commDestroy(ncclComm_t comm) {
  return ncclCommDestroy(comm);
}

ncclResult_t DefaultRcclxApi::commAbort(ncclComm_t comm) {
  return ncclCommAbort(comm);
}

ncclResult_t DefaultRcclxApi::commGetAsyncError(
    ncclComm_t comm,
    ncclResult_t* asyncError) {
  return ncclCommGetAsyncError(comm, asyncError);
}

ncclResult_t DefaultRcclxApi::commSplit(
    ncclComm_t comm,
    int color,
    int key,
    ncclComm_t* newcomm,
    ncclConfig_t* config) {
  return ncclCommSplit(comm, color, key, newcomm, config);
}

ncclResult_t DefaultRcclxApi::commRegister(
    ncclComm_t comm,
    void* buffer,
    size_t size,
    void** handle) {
  return ncclCommRegister(comm, buffer, size, handle);
}

ncclResult_t DefaultRcclxApi::commDeregister(ncclComm_t comm, void* handle) {
  return ncclCommDeregister(comm, handle);
}

ncclResult_t DefaultRcclxApi::send(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    hipStream_t stream) {
  return ncclSend(sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t DefaultRcclxApi::recv(
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    hipStream_t stream) {
  return ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}

ncclResult_t DefaultRcclxApi::broadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    hipStream_t stream) {
  return ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t DefaultRcclxApi::bcast(
    void* buff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    hipStream_t stream) {
  return ncclBcast(buff, count, datatype, root, comm, stream);
}

ncclResult_t DefaultRcclxApi::allReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    hipStream_t stream) {
  return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t DefaultRcclxApi::reduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    int root,
    ncclComm_t comm,
    hipStream_t stream) {
  return ncclReduce(
      sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t DefaultRcclxApi::allGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    hipStream_t stream) {
  return ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

ncclResult_t DefaultRcclxApi::reduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    hipStream_t stream) {
  return ncclReduceScatter(
      sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

ncclResult_t DefaultRcclxApi::allToAll(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    hipStream_t stream) {
  return ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

ncclResult_t DefaultRcclxApi::allToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    hipStream_t stream) {
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

ncclResult_t DefaultRcclxApi::winAllocate(
    size_t size,
    ncclComm_t comm,
    void** baseptr,
    RcclxWindow* winPtr,
    bool cpuBuf,
    const size_t signal_size) {
  (void)size;
  (void)comm;
  (void)baseptr;
  (void)winPtr;
  (void)cpuBuf;
  (void)signal_size;
  throw std::runtime_error("winAllocate not supported in RCCLX backend");
}

ncclResult_t DefaultRcclxApi::winFree(ncclComm_t comm, RcclxWindow win) {
  (void)comm;
  (void)win;
  throw std::runtime_error("winFree not supported in RCCLX backend");
}

ncclResult_t DefaultRcclxApi::winPut(
    const void* originBuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t targetDisp,
    RcclxWindow win,
    hipStream_t stream) {
  (void)originBuff;
  (void)count;
  (void)datatype;
  (void)peer;
  (void)targetDisp;
  (void)win;
  (void)stream;
  throw std::runtime_error("winPut not supported in RCCLX backend");
};

ncclResult_t DefaultRcclxApi::winSharedQuery(
    int rank,
    ncclComm_t comm,
    RcclxWindow win,
    void** addr) {
  (void)rank;
  (void)comm;
  (void)win;
  (void)addr;
  throw std::runtime_error("winSharedQuery not supported in RCCLX backend");
}

ncclResult_t DefaultRcclxApi::winSignal(
    size_t signalDisp,
    uint64_t signalVal,
    int peer,
    RcclxWindow win,
    hipStream_t stream) {
  (void)signalDisp;
  (void)signalVal;
  (void)peer;
  (void)win;
  (void)stream;
  throw std::runtime_error("winSignal not supported in RCCLX backend");
}

ncclResult_t DefaultRcclxApi::winWaitSignal(
    size_t signal_disp,
    uint64_t cmp_val,
    RcclxWindowCmpOp cmp_op,
    RcclxWindow win,
    hipStream_t stream) {
  (void)signal_disp;
  (void)cmp_val;
  (void)cmp_op;
  (void)win;
  (void)stream;
  throw std::runtime_error("winWaitSignal not supported in RCCLX backend");
}

ncclResult_t DefaultRcclxApi::groupStart() {
  return ncclGroupStart();
}

ncclResult_t DefaultRcclxApi::groupEnd() {
  return ncclGroupEnd();
}

ncclResult_t DefaultRcclxApi::commUserRank(const ncclComm_t comm, int* myRank) {
  return ncclCommUserRank(comm, myRank);
}

ncclResult_t DefaultRcclxApi::commCount(const ncclComm_t comm, int* count) {
  return ncclCommCount(comm, count);
}

ncclResult_t DefaultRcclxApi::redOpCreatePreMulSum(
    ncclRedOp_t* op,
    void* scalar,
    ncclDataType_t datatype,
    ncclScalarResidence_t residence,
    ncclComm_t comm) {
  return ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
}

ncclResult_t DefaultRcclxApi::redOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  return ncclRedOpDestroy(op, comm);
}

} // namespace comms
} // namespace torch
