#include "comms/torchcomms/hccl/HcclApi.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"

namespace torch::comms {

const char* DefaultHcclApi::getErrorString(HcclResult result) {
  // HCCL error codes are typically defined in hccl_types.h
  switch (result) {
    case HCCL_SUCCESS:
      return "success";
    case HCCL_E_PARA:
      return "invalid parameter";
    case HCCL_E_PTR:
      return "invalid pointer";
    case HCCL_E_MEMORY:
      return "memory error";
    case HCCL_E_NOT_SUPPORT:
      return "not supported";
    case HCCL_E_NOT_FOUND:
      return "not found";
    case HCCL_E_UNAVAIL:
      return "unavailable";
    case HCCL_E_SYSCALL:
      return "system call error";
    case HCCL_E_TIMEOUT:
      return "timeout";
    case HCCL_E_OPEN_FILE_FAILURE:
      return "open file failure";
    case HCCL_E_TCP_CONNECT:
      return "TCP connect error";
    case HCCL_E_ROCE_CONNECT:
      return "ROCE connect error";
    case HCCL_E_TCP_TRANSFER:
      return "TCP transfer error";
    case HCCL_E_ROCE_TRANSFER:
      return "ROCE transfer error";
    case HCCL_E_RUNTIME:
      return "runtime error";
    case HCCL_E_DRV:
      return "driver error";
    case HCCL_E_PROFILING:
      return "profiling error";
    case HCCL_E_CCE:
      return "CCE error";
    case HCCL_E_NETWORK:
      return "network error";
    case HCCL_E_RESERVED:
      return "reserved error";
    case HCCL_E_INTERNAL:
      return "internal error";
    case HCCL_E_AGAIN:
      return "try again";
    default:
      return "unknown error";
  }
}

HcclResult DefaultHcclApi::getUniqueId(HcclRootInfo* uniqueId) {
  return HcclGetRootInfo(uniqueId);
}

HcclResult DefaultHcclApi::commInitRankConfig(
    HcclComm* comm,
    int nranks,
    HcclRootInfo rootInfo,
    int rank,
    const HcclCommConfig* config) {
  if (config) {
    return HcclCommInitRootInfoConfig(
        nranks, &rootInfo, rank, config, comm);
  } else {
    return HcclCommInitRootInfo(nranks, &rootInfo, rank, comm);
  }
}

HcclResult DefaultHcclApi::commDestroy(HcclComm comm) {
  return HcclCommDestroy(comm);
}

HcclResult DefaultHcclApi::commAbort(HcclComm comm) {
  // HCCL may not have an explicit abort function
  // Destroy is the closest equivalent
  return HcclCommDestroy(comm);
}

HcclResult DefaultHcclApi::commGetAsyncError(
    HcclComm comm,
    HcclResult* asyncError) {
  // HCCL may not support async error checking
  // Return not supported
  if (asyncError) {
    *asyncError = HCCL_SUCCESS;
  }
  return HCCL_E_NOT_SUPPORT;
}

HcclResult DefaultHcclApi::send(
    const void* sendbuff,
    uint64_t count,
    HcclDataType datatype,
    uint32_t peer,
    HcclComm comm,
    npuStream_t stream) {
  // HCCL C API doesn't use const for sendbuff, but it shouldn't modify it
  return HcclSend(const_cast<void*>(sendbuff), count, datatype, peer, comm, stream);
}

HcclResult DefaultHcclApi::recv(
    void* recvbuff,
    uint64_t count,
    HcclDataType datatype,
    uint32_t peer,
    HcclComm comm,
    npuStream_t stream) {
  return HcclRecv(recvbuff, count, datatype, peer, comm, stream);
}

HcclResult DefaultHcclApi::broadcast(
    void* buff,
    uint64_t count,
    HcclDataType datatype,
    uint32_t root,
    HcclComm comm,
    npuStream_t stream) {
  return HcclBroadcast(buff, count, datatype, root, comm, stream);
}

HcclResult DefaultHcclApi::allReduce(
    const void* sendbuff,
    void* recvbuff,
    uint64_t count,
    HcclDataType datatype,
    HcclReduceOp op,
    HcclComm comm,
    npuStream_t stream) {
  // HCCL C API doesn't use const for sendbuff, but it shouldn't modify it
  return HcclAllReduce(const_cast<void*>(sendbuff), recvbuff, count, datatype, op, comm, stream);
}

HcclResult DefaultHcclApi::reduce(
    const void* sendbuff,
    void* recvbuff,
    uint64_t count,
    HcclDataType datatype,
    HcclReduceOp op,
    uint32_t root,
    HcclComm comm,
    npuStream_t stream) {
  // HCCL C API doesn't use const for sendbuff, but it shouldn't modify it
  return HcclReduce(const_cast<void*>(sendbuff), recvbuff, count, datatype, op, root, comm, stream);
}

HcclResult DefaultHcclApi::allGather(
    const void* sendbuff,
    void* recvbuff,
    uint64_t sendcount,
    HcclDataType datatype,
    HcclComm comm,
    npuStream_t stream) {
  // HCCL C API doesn't use const for sendbuff, but it shouldn't modify it
  return HcclAllGather(const_cast<void*>(sendbuff), recvbuff, sendcount, datatype, comm, stream);
}

HcclResult DefaultHcclApi::reduceScatter(
    const void* sendbuff,
    void* recvbuff,
    uint64_t recvcount,
    HcclDataType datatype,
    HcclReduceOp op,
    HcclComm comm,
    npuStream_t stream) {
  // HCCL C API doesn't use const for sendbuff, but it shouldn't modify it
  return HcclReduceScatter(const_cast<void*>(sendbuff), recvbuff, recvcount, datatype, op, comm, stream);
}

HcclResult DefaultHcclApi::allToAll(
    const void* sendbuff,
    void* recvbuff,
    uint64_t count,
    HcclDataType datatype,
    HcclComm comm,
    npuStream_t stream) {
  // HCCL may not have a direct alltoall, might need to use alltoallv
  // For now, return not supported
  return HCCL_E_NOT_SUPPORT;
}

HcclResult DefaultHcclApi::allToAllv(
    const void* sendbuff,
    const uint64_t* sendcounts,
    const uint64_t* sdispls,
    HcclDataType sendtype,
    void* recvbuff,
    const uint64_t* recvcounts,
    const uint64_t* rdispls,
    HcclDataType recvtype,
    HcclComm comm,
    npuStream_t stream) {
  // HCCL alltoallv may have a different signature
  // This is a placeholder implementation
  return HCCL_E_NOT_SUPPORT;
}

HcclResult DefaultHcclApi::groupStart() {
  // HCCL may not support explicit group operations
  // Return success as a no-op
  return HCCL_SUCCESS;
}

HcclResult DefaultHcclApi::groupEnd() {
  // HCCL may not support explicit group operations
  // Return success as a no-op
  return HCCL_SUCCESS;
}

HcclResult DefaultHcclApi::getRankId(HcclComm comm, uint32_t* userRank) {
  return HcclGetRankId(comm, userRank);
}

HcclResult DefaultHcclApi::getRankSize(HcclComm comm, uint32_t* count) {
  return HcclGetRankSize(comm, count);
}

} // namespace torch::comms
