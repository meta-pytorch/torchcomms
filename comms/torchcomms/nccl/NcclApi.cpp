// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/nccl/NcclApi.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"

namespace torch {
namespace comms {

#if defined(IS_NCCLX)
#error "NCCLX should not be used"
#endif

// DefaultNcclApi implementation
const char* DefaultNcclApi::getErrorString(ncclResult_t result) {
  return ncclGetErrorString(result);
}

std::string DefaultNcclApi::getLastError(ncclComm_t comm) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0)
  const char* lastError = ncclGetLastError(comm);
  return lastError ? std::string(lastError) : std::string();
#else
  (void)comm; // Suppress unused parameter warning
  return std::string();
#endif
}

ncclResult_t DefaultNcclApi::getUniqueId(ncclUniqueId* uniqueId) {
  return ncclGetUniqueId(uniqueId);
}

ncclResult_t DefaultNcclApi::commInitRankConfig(
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId commId,
    int rank,
    ncclConfig_t* config) {
  return ncclCommInitRankConfig(comm, nranks, commId, rank, config);
}

ncclResult_t DefaultNcclApi::commDestroy(ncclComm_t comm) {
  return ncclCommDestroy(comm);
}

ncclResult_t DefaultNcclApi::commAbort(ncclComm_t comm) {
  return ncclCommAbort(comm);
}

ncclResult_t DefaultNcclApi::commGetAsyncError(
    ncclComm_t comm,
    ncclResult_t* asyncError) {
  return ncclCommGetAsyncError(comm, asyncError);
}

ncclResult_t DefaultNcclApi::commSplit(
    ncclComm_t comm,
    int color,
    int key,
    ncclComm_t* newcomm,
    ncclConfig_t* config) {
  return ncclCommSplit(comm, color, key, newcomm, config);
}

ncclResult_t DefaultNcclApi::commRegister(
    ncclComm_t comm,
    void* buffer,
    size_t size,
    void** handle) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
  return ncclCommRegister(comm, buffer, size, handle);
#else
  throw std::runtime_error(
      "NCCL version " + std::to_string(NCCL_VERSION_CODE) +
      " does not support ncclCommRegister API");
#endif
}

ncclResult_t DefaultNcclApi::commDeregister(ncclComm_t comm, void* handle) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
  return ncclCommDeregister(comm, handle);
#else
  throw std::runtime_error(
      "NCCL version " + std::to_string(NCCL_VERSION_CODE) +
      " does not support ncclCommDeregister API");
#endif
}

ncclResult_t DefaultNcclApi::send(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclSend(sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t DefaultNcclApi::recv(
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}

ncclResult_t DefaultNcclApi::broadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t DefaultNcclApi::bcast(
    void* buff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclBcast(buff, count, datatype, root, comm, stream);
}

ncclResult_t DefaultNcclApi::allReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t DefaultNcclApi::reduce(
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

ncclResult_t DefaultNcclApi::allGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

ncclResult_t DefaultNcclApi::reduceScatter(
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

ncclResult_t DefaultNcclApi::allToAll(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  return ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
#else
  (void)sendbuff;
  (void)recvbuff;
  (void)count;
  (void)datatype;
  (void)comm;
  (void)stream;
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclAlltoAll API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::groupStart() {
  return ncclGroupStart();
}

ncclResult_t DefaultNcclApi::groupEnd() {
  return ncclGroupEnd();
}

ncclResult_t DefaultNcclApi::commUserRank(const ncclComm_t comm, int* myRank) {
  return ncclCommUserRank(comm, myRank);
}

ncclResult_t DefaultNcclApi::commCount(const ncclComm_t comm, int* count) {
  return ncclCommCount(comm, count);
}

ncclResult_t DefaultNcclApi::redOpCreatePreMulSum(
    ncclRedOp_t* op,
    void* scalar,
    ncclDataType_t datatype,
    ncclScalarResidence_t residence,
    ncclComm_t comm) {
  return ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
}

ncclResult_t DefaultNcclApi::redOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  return ncclRedOpDestroy(op, comm);
}

ncclResult_t DefaultNcclApi::memAlloc(void** buff, size_t size) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
  return ncclMemAlloc(buff, size);
#else
  throw std::runtime_error(
      "NCCL version " + std::to_string(NCCL_VERSION_CODE) +
      " does not support ncclMemAlloc API");
#endif
}

ncclResult_t DefaultNcclApi::memFree(void* buff) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
  return ncclMemFree(buff);
#else
  throw std::runtime_error(
      "NCCL version " + std::to_string(NCCL_VERSION_CODE) +
      " does not support ncclMemFree API");
#endif
}

} // namespace comms
} // namespace torch
