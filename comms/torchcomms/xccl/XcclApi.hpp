#pragma once

#include <oneapi/ccl.h>
#include <oneapi/ccl.hpp>

#include "comms/torchcomms/device/XpuApi.hpp"

namespace torch {
namespace comms {

class XcclApi {
 public:
  virtual ~XcclApi() = default;

  virtual const char* getErrorString(onecclResult_t result) = 0;

  virtual onecclResult_t setDevice(int device) = 0;

  virtual onecclResult_t getUniqueId(onecclUniqueId* uniqueId) = 0;

  virtual onecclResult_t commInitRankConfig(
      onecclComm_t* comm,
      int nranks,
      onecclUniqueId commId,
      int rank,
      onecclConfig_t* config) = 0;

  virtual onecclResult_t commDestroy(onecclComm_t comm) = 0;

  virtual onecclResult_t commAbort(onecclComm_t comm) = 0;

  virtual onecclResult_t commGetAsyncError(
      onecclComm_t comm,
      onecclResult_t* asyncError) = 0;

  virtual onecclResult_t commSplit(
      onecclComm_t comm,
      int color,
      int key,
      onecclComm_t* newcomm,
      onecclConfig_t* config) = 0;

  virtual onecclResult_t
  commRegister(onecclComm_t comm, void* buffer, size_t size, void** handle) = 0;

  virtual onecclResult_t commDeregister(onecclComm_t comm, void* handle) = 0;

  // Point-to-point operations
  virtual onecclResult_t send(
      const void* sendbuff,
      size_t count,
      onecclDataType_t datatype,
      int peer,
      onecclComm_t comm,
      xpuStream_t stream) = 0;

  virtual onecclResult_t recv(
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      int peer,
      onecclComm_t comm,
      xpuStream_t stream) = 0;

  // Collective operations
  virtual onecclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      int root,
      onecclComm_t comm,
      xpuStream_t stream) = 0;

  virtual onecclResult_t bcast(
      void* buff,
      size_t count,
      onecclDataType_t datatype,
      int root,
      onecclComm_t comm,
      xpuStream_t stream) = 0;

  virtual onecclResult_t allReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      onecclRedOp_t op,
      onecclComm_t comm,
      xpuStream_t stream) = 0;

  virtual onecclResult_t reduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      onecclRedOp_t op,
      int root,
      onecclComm_t comm,
      xpuStream_t stream) = 0;

  virtual onecclResult_t allGather(
      const void* sendbuff,
      void* recvbuff,
      size_t sendcount,
      onecclDataType_t datatype,
      onecclComm_t comm,
      xpuStream_t stream) = 0;

  virtual onecclResult_t reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      size_t recvcount,
      onecclDataType_t datatype,
      onecclRedOp_t op,
      onecclComm_t comm,
      xpuStream_t stream) = 0;

  virtual onecclResult_t allToAll(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      onecclComm_t comm,
      xpuStream_t stream) = 0;

  // Group operations
  virtual onecclResult_t groupStart() = 0;
  virtual onecclResult_t groupEnd() = 0;

  virtual onecclResult_t commUserRank(
      const onecclComm_t comm,
      int* userRank) = 0;
  virtual onecclResult_t commCount(const onecclComm_t comm, int* count) = 0;

  virtual onecclResult_t redOpCreatePreMulSum(
      onecclRedOp_t* op,
      void* scalar,
      onecclDataType_t datatype,
      onecclScalarResidence_t residence,
      onecclComm_t comm) = 0;
  virtual onecclResult_t redOpDestroy(onecclRedOp_t op, onecclComm_t comm) = 0;
};

/**
 * Default implementation that calls the underlying XCCL APIs directly.
 */
class DefaultXcclApi : public XcclApi {
 public:
  ~DefaultXcclApi() override = default;

  // Error handling
  const char* getErrorString(onecclResult_t result) override;

  // Device management
  onecclResult_t setDevice(int device) override;

  // Unique ID generation
  onecclResult_t getUniqueId(onecclUniqueId* uniqueId) override;

  // Communicator management
  onecclResult_t commInitRankConfig(
      onecclComm_t* comm,
      int nranks,
      onecclUniqueId commId,
      int rank,
      onecclConfig_t* config) override;

  onecclResult_t commDestroy(onecclComm_t comm) override;

  onecclResult_t commAbort(onecclComm_t comm) override;

  onecclResult_t commGetAsyncError(
      onecclComm_t comm,
      onecclResult_t* asyncError) override;

  onecclResult_t commSplit(
      onecclComm_t comm,
      int color,
      int key,
      onecclComm_t* newcomm,
      onecclConfig_t* config) override;

  onecclResult_t commRegister(
      onecclComm_t comm,
      void* buffer,
      size_t size,
      void** handle) override;

  onecclResult_t commDeregister(onecclComm_t comm, void* handle) override;

  // Point-to-point operations
  onecclResult_t send(
      const void* sendbuff,
      size_t count,
      onecclDataType_t datatype,
      int peer,
      onecclComm_t comm,
      xpuStream_t stream) override;

  onecclResult_t recv(
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      int peer,
      onecclComm_t comm,
      xpuStream_t stream) override;

  // Collective operations
  onecclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      int root,
      onecclComm_t comm,
      xpuStream_t stream) override;

  onecclResult_t bcast(
      void* buff,
      size_t count,
      onecclDataType_t datatype,
      int root,
      onecclComm_t comm,
      xpuStream_t stream) override;

  onecclResult_t allReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      onecclRedOp_t op,
      onecclComm_t comm,
      xpuStream_t stream) override;

  onecclResult_t reduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      onecclRedOp_t op,
      int root,
      onecclComm_t comm,
      xpuStream_t stream) override;

  onecclResult_t allGather(
      const void* sendbuff,
      void* recvbuff,
      size_t sendcount,
      onecclDataType_t datatype,
      onecclComm_t comm,
      xpuStream_t stream) override;

  onecclResult_t reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      size_t recvcount,
      onecclDataType_t datatype,
      onecclRedOp_t op,
      onecclComm_t comm,
      xpuStream_t stream) override;

  onecclResult_t allToAll(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      onecclComm_t comm,
      xpuStream_t stream) override;

  // Group operations
  onecclResult_t groupStart() override;
  onecclResult_t groupEnd() override;

  onecclResult_t commUserRank(const onecclComm_t comm, int* userRank) override;
  onecclResult_t commCount(const onecclComm_t comm, int* count) override;

  onecclResult_t redOpCreatePreMulSum(
      onecclRedOp_t* op,
      void* scalar,
      onecclDataType_t datatype,
      onecclScalarResidence_t residence,
      onecclComm_t comm) override;
  onecclResult_t redOpDestroy(onecclRedOp_t op, onecclComm_t comm) override;
};

} // namespace comms
} // namespace torch
