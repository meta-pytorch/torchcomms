// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <rccl.h> // @manual=//comms/rccl:rccl-dev

namespace torch {
namespace comms {

/**
 * Abstract interface for RCCL API operations.
 * This allows for dependency injection and testing by providing
 * a way to override NCCL API calls.
 */
class RcclApi {
 public:
  virtual ~RcclApi() = default;

  // Error handling
  virtual const char* getErrorString(ncclResult_t result) = 0;

  // Unique ID generation
  virtual ncclResult_t getUniqueId(ncclUniqueId* uniqueId) = 0;

  // Communicator management
  virtual ncclResult_t commInitRankConfig(
      ncclComm_t* comm,
      int nranks,
      ncclUniqueId commId,
      int rank,
      ncclConfig_t* config) = 0;

  virtual ncclResult_t commDestroy(ncclComm_t comm) = 0;

  virtual ncclResult_t commAbort(ncclComm_t comm) = 0;

  virtual ncclResult_t commGetAsyncError(
      ncclComm_t comm,
      ncclResult_t* asyncError) = 0;

  virtual ncclResult_t commSplit(
      ncclComm_t comm,
      int color,
      int key,
      ncclComm_t* newcomm,
      ncclConfig_t* config) = 0;

  // Memory registration
  virtual ncclResult_t
  commRegister(ncclComm_t comm, void* buffer, size_t size, void** handle) = 0;

  virtual ncclResult_t commDeregister(ncclComm_t comm, void* handle) = 0;

  // Point-to-point operations
  virtual ncclResult_t send(
      const void* sendbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  virtual ncclResult_t recv(
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  // Collective operations
  virtual ncclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  virtual ncclResult_t bcast(
      void* buff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  virtual ncclResult_t allReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  virtual ncclResult_t reduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      int root,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  virtual ncclResult_t allGather(
      const void* sendbuff,
      void* recvbuff,
      size_t sendcount,
      ncclDataType_t datatype,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  virtual ncclResult_t reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      size_t recvcount,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  virtual ncclResult_t allToAll(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  virtual ncclResult_t allToAllv(
      const void* sendbuff,
      const size_t sendcounts[],
      const size_t sdispls[],
      void* recvbuff,
      const size_t recvcounts[],
      const size_t rdispls[],
      ncclDataType_t datatype,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  // Group operations
  virtual ncclResult_t groupStart() = 0;
  virtual ncclResult_t groupEnd() = 0;

  virtual ncclResult_t commUserRank(const ncclComm_t comm, int* userRank) = 0;
  virtual ncclResult_t commCount(const ncclComm_t comm, int* count) = 0;

  // Custom reduction operations
  virtual ncclResult_t redOpCreatePreMulSum(
      ncclRedOp_t* op,
      void* scalar,
      ncclDataType_t datatype,
      ncclScalarResidence_t residence,
      ncclComm_t comm) = 0;

  virtual ncclResult_t redOpDestroy(ncclRedOp_t op, ncclComm_t comm) = 0;
};

/**
 * Default implementation that calls the underlying NCCL APIs directly.
 */
class DefaultRcclApi : public RcclApi {
 public:
  ~DefaultRcclApi() override = default;

  // Error handling
  const char* getErrorString(ncclResult_t result) override;

  // Unique ID generation
  ncclResult_t getUniqueId(ncclUniqueId* uniqueId) override;

  // Communicator management
  ncclResult_t commInitRankConfig(
      ncclComm_t* comm,
      int nranks,
      ncclUniqueId commId,
      int rank,
      ncclConfig_t* config) override;

  ncclResult_t commDestroy(ncclComm_t comm) override;

  ncclResult_t commAbort(ncclComm_t comm) override;

  ncclResult_t commGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError)
      override;

  ncclResult_t commSplit(
      ncclComm_t comm,
      int color,
      int key,
      ncclComm_t* newcomm,
      ncclConfig_t* config) override;

  ncclResult_t commRegister(
      ncclComm_t comm,
      void* buffer,
      size_t size,
      void** handle) override;

  ncclResult_t commDeregister(ncclComm_t comm, void* handle) override;

  // Point-to-point operations
  ncclResult_t send(
      const void* sendbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      hipStream_t stream) override;

  ncclResult_t recv(
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      hipStream_t stream) override;

  // Collective operations
  ncclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      hipStream_t stream) override;

  ncclResult_t bcast(
      void* buff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      hipStream_t stream) override;

  ncclResult_t allReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      hipStream_t stream) override;

  ncclResult_t reduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      int root,
      ncclComm_t comm,
      hipStream_t stream) override;

  ncclResult_t allGather(
      const void* sendbuff,
      void* recvbuff,
      size_t sendcount,
      ncclDataType_t datatype,
      ncclComm_t comm,
      hipStream_t stream) override;

  ncclResult_t reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      size_t recvcount,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      hipStream_t stream) override;

  ncclResult_t allToAll(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclComm_t comm,
      hipStream_t stream) override;

  ncclResult_t allToAllv(
      const void* sendbuff,
      const size_t sendcounts[],
      const size_t senddispls[],
      void* recvbuff,
      const size_t recvcounts[],
      const size_t recvdispls[],
      ncclDataType_t datatype,
      ncclComm_t comm,
      hipStream_t stream) override;

  // Group operations
  ncclResult_t groupStart() override;
  ncclResult_t groupEnd() override;

  ncclResult_t commUserRank(const ncclComm_t comm, int* userRank) override;
  ncclResult_t commCount(const ncclComm_t comm, int* count) override;

  // Custom reduction operations
  ncclResult_t redOpCreatePreMulSum(
      ncclRedOp_t* op,
      void* scalar,
      ncclDataType_t datatype,
      ncclScalarResidence_t residence,
      ncclComm_t comm) override;

  ncclResult_t redOpDestroy(ncclRedOp_t op, ncclComm_t comm) override;
};

} // namespace comms
} // namespace torch
