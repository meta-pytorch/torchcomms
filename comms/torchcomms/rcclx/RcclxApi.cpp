// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rcclx/RcclxApi.hpp"

namespace torch::comms {

// DefaultRcclxApi implementation

const char* DefaultRcclxApi::getErrorString(ncclResult_t result) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGetErrorString(result);
}

std::string DefaultRcclxApi::getLastError(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0)
  const char* lastError = ncclGetLastError(comm);
  return lastError ? std::string(lastError) : std::string();
#else
  (void)comm; // Suppress unused parameter warning
  return std::string();
#endif
}

ncclResult_t DefaultRcclxApi::getUniqueId(ncclUniqueId* uniqueId) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGetUniqueId(uniqueId);
}

ncclResult_t DefaultRcclxApi::commInitRankConfig(
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId commId,
    int rank,
    ncclConfig_t* config) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommInitRankConfig(comm, nranks, commId, rank, config);
}

ncclResult_t DefaultRcclxApi::commDestroy(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommDestroy(comm);
}

ncclResult_t DefaultRcclxApi::commAbort(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommAbort(comm);
}

ncclResult_t DefaultRcclxApi::commGetAsyncError(
    ncclComm_t comm,
    ncclResult_t* asyncError) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommGetAsyncError(comm, asyncError);
}

ncclResult_t DefaultRcclxApi::commSplit(
    ncclComm_t comm,
    int color,
    int key,
    ncclComm_t* newcomm,
    ncclConfig_t* config) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommSplit(comm, color, key, newcomm, config);
}

ncclResult_t DefaultRcclxApi::commRegister(
    ncclComm_t comm,
    void* buffer,
    size_t size,
    void** handle) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommRegister(comm, buffer, size, handle);
}

ncclResult_t DefaultRcclxApi::commDeregister(ncclComm_t comm, void* handle) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommDeregister(comm, handle);
}

ncclResult_t DefaultRcclxApi::send(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclSend(sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t DefaultRcclxApi::recv(
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t DefaultRcclxApi::bcast(
    void* buff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    hipStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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

ncclResult_t DefaultRcclxApi::winAllocate(
    size_t size,
    ncclComm_t comm,
    void** baseptr,
    RcclxWindow* winPtr,
    bool cpuBuf,
    const size_t signal_size) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  (void)size;
  (void)comm;
  (void)baseptr;
  (void)winPtr;
  (void)cpuBuf;
  (void)signal_size;
  throw std::runtime_error("winAllocate not supported in RCCLX backend");
}

ncclResult_t DefaultRcclxApi::winFree(ncclComm_t comm, RcclxWindow win) {
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
  (void)signal_disp;
  (void)cmp_val;
  (void)cmp_op;
  (void)win;
  (void)stream;
  throw std::runtime_error("winWaitSignal not supported in RCCLX backend");
}

ncclResult_t DefaultRcclxApi::groupStart() {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGroupStart();
}

ncclResult_t DefaultRcclxApi::groupEnd() {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGroupEnd();
}

ncclResult_t DefaultRcclxApi::commUserRank(const ncclComm_t comm, int* myRank) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommUserRank(comm, myRank);
}

ncclResult_t DefaultRcclxApi::commCount(const ncclComm_t comm, int* count) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommCount(comm, count);
}

ncclResult_t DefaultRcclxApi::redOpCreatePreMulSum(
    ncclRedOp_t* op,
    void* scalar,
    ncclDataType_t datatype,
    ncclScalarResidence_t residence,
    ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
}

ncclResult_t DefaultRcclxApi::redOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclRedOpDestroy(op, comm);
}

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

ncclResult_t DefaultRcclxApi::memAlloc(void** ptr, size_t size) {
  return ncclMemAlloc(ptr, size);
}

ncclResult_t DefaultRcclxApi::memFree(void* ptr) {
  return ncclMemFree(ptr);
}

} // namespace torch::comms
