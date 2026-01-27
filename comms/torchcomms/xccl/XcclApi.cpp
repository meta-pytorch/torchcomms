#include "comms/torchcomms/xccl/XcclApi.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"

namespace torch::comms {

const char* DefaultXcclApi::getErrorString(onecclResult_t result) {
  return onecclGetErrorString(result);
}

onecclResult_t DefaultXcclApi::setDevice(int device) {
  return onecclSetDevice(device);
}

onecclResult_t DefaultXcclApi::getUniqueId(onecclUniqueId* uniqueId) {
  return onecclGetUniqueId(uniqueId);
}

onecclResult_t DefaultXcclApi::commInitRankConfig(
    onecclComm_t* comm,
    int nranks,
    onecclUniqueId commId,
    int rank,
    onecclConfig_t* config) {
  return onecclCommInitRankConfig(comm, nranks, commId, rank, config);
}

onecclResult_t DefaultXcclApi::commDestroy(onecclComm_t comm) {
  return onecclCommDestroy(comm);
}

onecclResult_t DefaultXcclApi::commAbort(onecclComm_t comm) {
  // return onecclCommAbort(comm);
  return onecclNotImplemented;
}

onecclResult_t DefaultXcclApi::commGetAsyncError(
    onecclComm_t comm,
    onecclResult_t* asyncError) {
  // return onecclCommGetAsyncError(comm);
  return onecclNotImplemented;
}

onecclResult_t DefaultXcclApi::commSplit(
    onecclComm_t comm,
    int color,
    int key,
    onecclComm_t* newcomm,
    onecclConfig_t* config) {
  return onecclCommSplit(comm, color, key, newcomm, config);
}

onecclResult_t DefaultXcclApi::commRegister(
    onecclComm_t comm,
    void* buffer,
    size_t size,
    void** handle) {
  // return onecclCommRegister(comm, buffer, size, handle);
  return onecclNotImplemented;
}

onecclResult_t DefaultXcclApi::commDeregister(onecclComm_t comm, void* handle) {
  // return onecclCommDeregister(comm, handle);
  return onecclNotImplemented;
}

onecclResult_t DefaultXcclApi::send(
    const void* sendbuff,
    size_t count,
    onecclDataType_t datatype,
    int peer,
    onecclComm_t comm,
    xpuStream_t stream) {
  return onecclSend(
      const_cast<void*>(sendbuff), count, datatype, peer, comm, stream);
}

onecclResult_t DefaultXcclApi::recv(
    void* recvbuff,
    size_t count,
    onecclDataType_t datatype,
    int peer,
    onecclComm_t comm,
    xpuStream_t stream) {
  return onecclRecv(recvbuff, count, datatype, peer, comm, stream);
}

onecclResult_t DefaultXcclApi::broadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    onecclDataType_t datatype,
    int root,
    onecclComm_t comm,
    xpuStream_t stream) {
  return onecclBroadcast(
      const_cast<void*>(sendbuff),
      recvbuff,
      count,
      datatype,
      root,
      comm,
      stream);
}

onecclResult_t DefaultXcclApi::bcast(
    void* buff,
    size_t count,
    onecclDataType_t datatype,
    int root,
    onecclComm_t comm,
    xpuStream_t stream) {
  return onecclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

onecclResult_t DefaultXcclApi::allReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    onecclDataType_t datatype,
    onecclRedOp_t op,
    onecclComm_t comm,
    xpuStream_t stream) {
  return onecclAllReduce(
      const_cast<void*>(sendbuff), recvbuff, count, datatype, op, comm, stream);
}

onecclResult_t DefaultXcclApi::reduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    onecclDataType_t datatype,
    onecclRedOp_t op,
    int root,
    onecclComm_t comm,
    xpuStream_t stream) {
  return onecclReduce(
      const_cast<void*>(sendbuff),
      recvbuff,
      count,
      datatype,
      op,
      root,
      comm,
      stream);
}

onecclResult_t DefaultXcclApi::allGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    onecclDataType_t datatype,
    onecclComm_t comm,
    xpuStream_t stream) {
  return onecclAllGather(
      const_cast<void*>(sendbuff), recvbuff, sendcount, datatype, comm, stream);
}

onecclResult_t DefaultXcclApi::reduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    onecclDataType_t datatype,
    onecclRedOp_t op,
    onecclComm_t comm,
    xpuStream_t stream) {
  return onecclReduceScatter(
      const_cast<void*>(sendbuff),
      recvbuff,
      recvcount,
      datatype,
      op,
      comm,
      stream);
}

onecclResult_t DefaultXcclApi::allToAll(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    onecclDataType_t datatype,
    onecclComm_t comm,
    xpuStream_t stream) {
  return onecclAllToAll(
      const_cast<void*>(sendbuff), recvbuff, count, datatype, comm, stream);
}

onecclResult_t DefaultXcclApi::groupStart() {
  return onecclGroupStart();
}

onecclResult_t DefaultXcclApi::groupEnd() {
  return onecclGroupEnd();
}

onecclResult_t DefaultXcclApi::commUserRank(
    const onecclComm_t comm,
    int* myRank) {
  return onecclCommUserRank(comm, myRank);
}

onecclResult_t DefaultXcclApi::commCount(const onecclComm_t comm, int* count) {
  return onecclCommCount(comm, count);
}

onecclResult_t DefaultXcclApi::redOpCreatePreMulSum(
    onecclRedOp_t* op,
    void* scalar,
    onecclDataType_t datatype,
    onecclScalarResidence_t residence,
    onecclComm_t comm) {
  return onecclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
}

onecclResult_t DefaultXcclApi::redOpDestroy(
    onecclRedOp_t op,
    onecclComm_t comm) {
  return onecclRedOpDestroy(op, comm);
}

} // namespace torch::comms
