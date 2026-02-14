#include "comms/torchcomms/xccl/XcclApi.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"

namespace torch::comms {

const char* DefaultXcclApi::getErrorString(onecclResult_t result) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclGetErrorString(result);
}

onecclResult_t DefaultXcclApi::setDevice(int device) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclSetDevice(device);
}

onecclResult_t DefaultXcclApi::getUniqueId(onecclUniqueId* uniqueId) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclGetUniqueId(uniqueId);
}

onecclResult_t DefaultXcclApi::commInitRankConfig(
    onecclComm_t* comm,
    int nranks,
    onecclUniqueId commId,
    int rank,
    onecclConfig_t* config) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclCommInitRankConfig(comm, nranks, commId, rank, config);
}

onecclResult_t DefaultXcclApi::commDestroy(onecclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
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
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclAllToAll(
      const_cast<void*>(sendbuff), recvbuff, count, datatype, comm, stream);
}

onecclResult_t DefaultXcclApi::groupStart() {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclGroupStart();
}

onecclResult_t DefaultXcclApi::groupEnd() {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclGroupEnd();
}

onecclResult_t DefaultXcclApi::commUserRank(
    const onecclComm_t comm,
    int* myRank) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclCommUserRank(comm, myRank);
}

onecclResult_t DefaultXcclApi::commCount(const onecclComm_t comm, int* count) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclCommCount(comm, count);
}

onecclResult_t DefaultXcclApi::redOpCreatePreMulSum(
    onecclRedOp_t* op,
    void* scalar,
    onecclDataType_t datatype,
    onecclScalarResidence_t residence,
    onecclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
}

onecclResult_t DefaultXcclApi::redOpDestroy(
    onecclRedOp_t op,
    onecclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclRedOpDestroy(op, comm);
}

void DefaultXcclApi::setVersionInfo() {
  int version = -1;
  onecclResult_t res = onecclGetVersion(&version);

  if (res != onecclSuccess) {
    TC_LOG(ERROR) << "XCCL getVersion failed with error: "
                  << getErrorString(res);
    return;
  }

  version_info_.version = version;

  int major = -1;
  int minor = -1;
  int patch = -1;
  res = onecclExtractVersionComponents(version, &major, &minor, &patch);

  if (res != onecclSuccess) {
    TC_LOG(ERROR) << "XCCL extractVersionComponents failed with error: "
                  << getErrorString(res);
    TC_LOG(INFO) << "XCCL Version: " << version_info_.version;
    TC_LOG(WARNING) << "XCCL Major/Minor/Patch info not available";
    return;
  }

  version_info_.major = major;
  version_info_.minor = minor;
  version_info_.patch = patch;

  TC_LOG(INFO) << "XCCL Version: " << version_info_.version
               << " (Major: " << version_info_.major
               << " Minor: " << version_info_.minor
               << " Patch: " << version_info_.patch << ")";
}

} // namespace torch::comms
