// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rccl/RcclApi.hpp"

namespace torch::comms {

// DefaultRcclApi implementation

const char* DefaultRcclApi::getErrorString(ncclResult_t result) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGetErrorString(result);
}

std::string DefaultRcclApi::getLastError(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0)
  const char* lastError = ncclGetLastError(comm);
  return lastError ? std::string(lastError) : std::string();
#else
  (void)comm; // Suppress unused parameter warning
  return std::string();
#endif
}

ncclResult_t DefaultRcclApi::getUniqueId(ncclUniqueId* uniqueId) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGetUniqueId(uniqueId);
}

ncclResult_t DefaultRcclApi::commInitRankConfig(
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId commId,
    int rank,
    ncclConfig_t* config) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommInitRankConfig(comm, nranks, commId, rank, config);
}

ncclResult_t DefaultRcclApi::commDestroy(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommDestroy(comm);
}

ncclResult_t DefaultRcclApi::commAbort(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommAbort(comm);
}

ncclResult_t DefaultRcclApi::commGetAsyncError(
    ncclComm_t comm,
    ncclResult_t* asyncError) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommGetAsyncError(comm, asyncError);
}

ncclResult_t DefaultRcclApi::commSplit(
    ncclComm_t comm,
    int color,
    int key,
    ncclComm_t* newcomm,
    ncclConfig_t* config) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommSplit(comm, color, key, newcomm, config);
}

ncclResult_t DefaultRcclApi::commRegister(
    ncclComm_t comm,
    void* buffer,
    size_t size,
    void** handle) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommRegister(comm, buffer, size, handle);
}

ncclResult_t DefaultRcclApi::commDeregister(ncclComm_t comm, void* handle) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommDeregister(comm, handle);
}

ncclResult_t DefaultRcclApi::send(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclSend(sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t DefaultRcclApi::recv(
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}

ncclResult_t DefaultRcclApi::broadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t DefaultRcclApi::bcast(
    void* buff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclBcast(buff, count, datatype, root, comm, stream);
}

ncclResult_t DefaultRcclApi::allReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t DefaultRcclApi::reduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    int root,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclReduce(
      sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t DefaultRcclApi::allGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

ncclResult_t DefaultRcclApi::reduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclReduceScatter(
      sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

ncclResult_t DefaultRcclApi::allToAll(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

ncclResult_t DefaultRcclApi::allToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
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

ncclResult_t DefaultRcclApi::groupStart() {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGroupStart();
}

ncclResult_t DefaultRcclApi::groupEnd() {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGroupEnd();
}

ncclResult_t DefaultRcclApi::commUserRank(const ncclComm_t comm, int* myRank) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommUserRank(comm, myRank);
}

ncclResult_t DefaultRcclApi::commCount(const ncclComm_t comm, int* count) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommCount(comm, count);
}

ncclResult_t DefaultRcclApi::redOpCreatePreMulSum(
    ncclRedOp_t* op,
    void* scalar,
    ncclDataType_t datatype,
    ncclScalarResidence_t residence,
    ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
}

ncclResult_t DefaultRcclApi::redOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclRedOpDestroy(op, comm);
}

} // namespace torch::comms
