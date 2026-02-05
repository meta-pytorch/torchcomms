#pragma once

#include <hccl/hccl.h>
#include <hccl/hccl_types.h>

#include "comms/torchcomms/device/npu/NpuApi.hpp"

namespace torch::comms {

class HcclApi {
 public:
  virtual ~HcclApi() = default;

  virtual const char* getErrorString(HcclResult result) = 0;

  virtual HcclResult getUniqueId(HcclRootInfo* uniqueId) = 0;

  virtual HcclResult commInitRankConfig(
      HcclComm* comm,
      int nranks,
      HcclRootInfo commId,
      int rank,
      const HcclCommConfig* config) = 0;

  virtual HcclResult commDestroy(HcclComm comm) = 0;

  virtual HcclResult commAbort(HcclComm comm) = 0;

  virtual HcclResult commGetAsyncError(
      HcclComm comm,
      HcclResult* asyncError) = 0;

  // Point-to-point operations
  virtual HcclResult send(
      const void* sendbuff,
      uint64_t count,
      HcclDataType datatype,
      uint32_t peer,
      HcclComm comm,
      npuStream_t stream) = 0;

  virtual HcclResult recv(
      void* recvbuff,
      uint64_t count,
      HcclDataType datatype,
      uint32_t peer,
      HcclComm comm,
      npuStream_t stream) = 0;

  // Collective operations
  virtual HcclResult broadcast(
      void* buff,
      uint64_t count,
      HcclDataType datatype,
      uint32_t root,
      HcclComm comm,
      npuStream_t stream) = 0;

  virtual HcclResult allReduce(
      const void* sendbuff,
      void* recvbuff,
      uint64_t count,
      HcclDataType datatype,
      HcclReduceOp op,
      HcclComm comm,
      npuStream_t stream) = 0;

  virtual HcclResult reduce(
      const void* sendbuff,
      void* recvbuff,
      uint64_t count,
      HcclDataType datatype,
      HcclReduceOp op,
      uint32_t root,
      HcclComm comm,
      npuStream_t stream) = 0;

  virtual HcclResult allGather(
      const void* sendbuff,
      void* recvbuff,
      uint64_t sendcount,
      HcclDataType datatype,
      HcclComm comm,
      npuStream_t stream) = 0;

  virtual HcclResult reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      uint64_t recvcount,
      HcclDataType datatype,
      HcclReduceOp op,
      HcclComm comm,
      npuStream_t stream) = 0;

  virtual HcclResult allToAll(
      const void* sendbuff,
      void* recvbuff,
      uint64_t count,
      HcclDataType datatype,
      HcclComm comm,
      npuStream_t stream) = 0;

  virtual HcclResult allToAllv(
      const void* sendbuff,
      const uint64_t* sendcounts,
      const uint64_t* sdispls,
      HcclDataType sendtype,
      void* recvbuff,
      const uint64_t* recvcounts,
      const uint64_t* rdispls,
      HcclDataType recvtype,
      HcclComm comm,
      npuStream_t stream) = 0;

  // Group operations
  virtual HcclResult groupStart() = 0;
  virtual HcclResult groupEnd() = 0;

  virtual HcclResult getRankId(HcclComm comm, uint32_t* userRank) = 0;
  virtual HcclResult getRankSize(HcclComm comm, uint32_t* count) = 0;
};

/**
 * Default implementation that calls the underlying HCCL APIs directly.
 */
class DefaultHcclApi : public HcclApi {
 public:
  ~DefaultHcclApi() override = default;

  // Error handling
  const char* getErrorString(HcclResult result) override;

  // Unique ID generation
  HcclResult getUniqueId(HcclRootInfo* uniqueId) override;

  // Communicator management
  HcclResult commInitRankConfig(
      HcclComm* comm,
      int nranks,
      HcclRootInfo commId,
      int rank,
      const HcclCommConfig* config) override;

  HcclResult commDestroy(HcclComm comm) override;

  HcclResult commAbort(HcclComm comm) override;

  HcclResult commGetAsyncError(
      HcclComm comm,
      HcclResult* asyncError) override;

  // Point-to-point operations
  HcclResult send(
      const void* sendbuff,
      uint64_t count,
      HcclDataType datatype,
      uint32_t peer,
      HcclComm comm,
      npuStream_t stream) override;

  HcclResult recv(
      void* recvbuff,
      uint64_t count,
      HcclDataType datatype,
      uint32_t peer,
      HcclComm comm,
      npuStream_t stream) override;

  // Collective operations
  HcclResult broadcast(
      void* buff,
      uint64_t count,
      HcclDataType datatype,
      uint32_t root,
      HcclComm comm,
      npuStream_t stream) override;

  HcclResult allReduce(
      const void* sendbuff,
      void* recvbuff,
      uint64_t count,
      HcclDataType datatype,
      HcclReduceOp op,
      HcclComm comm,
      npuStream_t stream) override;

  HcclResult reduce(
      const void* sendbuff,
      void* recvbuff,
      uint64_t count,
      HcclDataType datatype,
      HcclReduceOp op,
      uint32_t root,
      HcclComm comm,
      npuStream_t stream) override;

  HcclResult allGather(
      const void* sendbuff,
      void* recvbuff,
      uint64_t sendcount,
      HcclDataType datatype,
      HcclComm comm,
      npuStream_t stream) override;

  HcclResult reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      uint64_t recvcount,
      HcclDataType datatype,
      HcclReduceOp op,
      HcclComm comm,
      npuStream_t stream) override;

  HcclResult allToAll(
      const void* sendbuff,
      void* recvbuff,
      uint64_t count,
      HcclDataType datatype,
      HcclComm comm,
      npuStream_t stream) override;

  HcclResult allToAllv(
      const void* sendbuff,
      const uint64_t* sendcounts,
      const uint64_t* sdispls,
      HcclDataType sendtype,
      void* recvbuff,
      const uint64_t* recvcounts,
      const uint64_t* rdispls,
      HcclDataType recvtype,
      HcclComm comm,
      npuStream_t stream) override;

  // Group operations
  HcclResult groupStart() override;
  HcclResult groupEnd() override;

  HcclResult getRankId(HcclComm comm, uint32_t* userRank) override;
  HcclResult getRankSize(HcclComm comm, uint32_t* count) override;
};

} // namespace torch::comms
