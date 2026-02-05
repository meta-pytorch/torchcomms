// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <mutex>
#include <string>

#include <rccl.h> // @manual=//comms/rcclx:rcclx-dev

namespace torch::comms {

#ifdef NCCL_RMA_SUPPORTED
using RcclxWindow = ncclWindow_t;
using RcclxWindowCmpOp = ncclCmpOp_t;
#else
using RcclxWindow = void*;
using RcclxWindowCmpOp = int;
#endif

/**
 * Abstract interface for RCCLX API operations.
 * This allows for dependency injection and testing by providing
 * a way to override NCCL API calls.
 */
class RcclxApi {
 public:
  virtual ~RcclxApi() = default;

  // Error handling
  virtual const char* getErrorString(ncclResult_t result) = 0;
  virtual std::string getLastError(ncclComm_t comm) = 0;

  // Unique ID generation
  [[nodiscard]] virtual ncclResult_t getUniqueId(ncclUniqueId* uniqueId) = 0;

  // Communicator management
  [[nodiscard]] virtual ncclResult_t commInitRankConfig(
      ncclComm_t* comm,
      int nranks,
      ncclUniqueId commId,
      int rank,
      ncclConfig_t* config) = 0;

  [[nodiscard]] virtual ncclResult_t commDestroy(ncclComm_t comm) = 0;

  [[nodiscard]] virtual ncclResult_t commAbort(ncclComm_t comm) = 0;

  [[nodiscard]] virtual ncclResult_t commGetAsyncError(
      ncclComm_t comm,
      ncclResult_t* asyncError) = 0;

  [[nodiscard]] virtual ncclResult_t commSplit(
      ncclComm_t comm,
      int color,
      int key,
      ncclComm_t* newcomm,
      ncclConfig_t* config) = 0;

  // Memory registration
  [[nodiscard]] virtual ncclResult_t
  commRegister(ncclComm_t comm, void* buffer, size_t size, void** handle) = 0;

  [[nodiscard]] virtual ncclResult_t commDeregister(
      ncclComm_t comm,
      void* handle) = 0;

  // Point-to-point operations
  [[nodiscard]] virtual ncclResult_t send(
      const void* sendbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t recv(
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  // Collective operations
  [[nodiscard]] virtual ncclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t bcast(
      void* buff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t allReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t reduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      int root,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t allGather(
      const void* sendbuff,
      void* recvbuff,
      size_t sendcount,
      ncclDataType_t datatype,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      size_t recvcount,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t allToAll(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t allToAllv(
      const void* sendbuff,
      const size_t sendcounts[],
      const size_t sdispls[],
      void* recvbuff,
      const size_t recvcounts[],
      const size_t rdispls[],
      ncclDataType_t datatype,
      ncclComm_t comm,
      hipStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t winAllocate(
      size_t size,
      ncclComm_t comm,
      void** baseptr,
      RcclxWindow* winPtr,
      bool cpuBuf,
      const size_t signal_size = 256) = 0;

  [[nodiscard]] virtual ncclResult_t winFree(
      ncclComm_t comm,
      RcclxWindow win) = 0;

  [[nodiscard]] virtual ncclResult_t winPut(
      const void* originBuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      size_t targetDisp,
      RcclxWindow win,
      hipStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t
  winSharedQuery(int rank, ncclComm_t comm, RcclxWindow win, void** addr) = 0;

  [[nodiscard]] virtual ncclResult_t winSignal(
      size_t signalDisp,
      uint64_t signalVal,
      int peer,
      RcclxWindow win,
      hipStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t winWaitSignal(
      size_t signal_disp,
      uint64_t cmp_val,
      RcclxWindowCmpOp cmp_op,
      RcclxWindow win,
      hipStream_t stream) = 0;

  // Group operations
  [[nodiscard]] virtual ncclResult_t groupStart() = 0;
  [[nodiscard]] virtual ncclResult_t groupEnd() = 0;

  [[nodiscard]] virtual ncclResult_t commUserRank(
      const ncclComm_t comm,
      int* userRank) = 0;
  [[nodiscard]] virtual ncclResult_t commCount(
      const ncclComm_t comm,
      int* count) = 0;

  // Custom reduction operations
  [[nodiscard]] virtual ncclResult_t redOpCreatePreMulSum(
      ncclRedOp_t* op,
      void* scalar,
      ncclDataType_t datatype,
      ncclScalarResidence_t residence,
      ncclComm_t comm) = 0;

  [[nodiscard]] virtual ncclResult_t redOpDestroy(
      ncclRedOp_t op,
      ncclComm_t comm) = 0;
};

/**
 * Default implementation that calls the underlying NCCL APIs directly.
 */
class DefaultRcclxApi : public RcclxApi {
 public:
  ~DefaultRcclxApi() override = default;

 private:
  mutable std::mutex api_mutex_;

  // Error handling
  const char* getErrorString(ncclResult_t result) override;
  std::string getLastError(ncclComm_t comm) override;

  // Unique ID generation
  [[nodiscard]] ncclResult_t getUniqueId(ncclUniqueId* uniqueId) override;

  // Communicator management
  [[nodiscard]] ncclResult_t commInitRankConfig(
      ncclComm_t* comm,
      int nranks,
      ncclUniqueId commId,
      int rank,
      ncclConfig_t* config) override;

  [[nodiscard]] ncclResult_t commDestroy(ncclComm_t comm) override;

  [[nodiscard]] ncclResult_t commAbort(ncclComm_t comm) override;

  [[nodiscard]] ncclResult_t commGetAsyncError(
      ncclComm_t comm,
      ncclResult_t* asyncError) override;

  [[nodiscard]] ncclResult_t commSplit(
      ncclComm_t comm,
      int color,
      int key,
      ncclComm_t* newcomm,
      ncclConfig_t* config) override;

  [[nodiscard]] ncclResult_t commRegister(
      ncclComm_t comm,
      void* buffer,
      size_t size,
      void** handle) override;

  [[nodiscard]] ncclResult_t commDeregister(ncclComm_t comm, void* handle)
      override;

  // Point-to-point operations
  [[nodiscard]] ncclResult_t send(
      const void* sendbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      hipStream_t stream) override;

  [[nodiscard]] ncclResult_t recv(
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      hipStream_t stream) override;

  // Collective operations
  [[nodiscard]] ncclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      hipStream_t stream) override;

  [[nodiscard]] ncclResult_t bcast(
      void* buff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      hipStream_t stream) override;

  [[nodiscard]] ncclResult_t allReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      hipStream_t stream) override;

  [[nodiscard]] ncclResult_t reduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      int root,
      ncclComm_t comm,
      hipStream_t stream) override;

  [[nodiscard]] ncclResult_t allGather(
      const void* sendbuff,
      void* recvbuff,
      size_t sendcount,
      ncclDataType_t datatype,
      ncclComm_t comm,
      hipStream_t stream) override;

  [[nodiscard]] ncclResult_t reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      size_t recvcount,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      hipStream_t stream) override;

  [[nodiscard]] ncclResult_t allToAll(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclComm_t comm,
      hipStream_t stream) override;

  [[nodiscard]] ncclResult_t allToAllv(
      const void* sendbuff,
      const size_t sendcounts[],
      const size_t senddispls[],
      void* recvbuff,
      const size_t recvcounts[],
      const size_t recvdispls[],
      ncclDataType_t datatype,
      ncclComm_t comm,
      hipStream_t stream) override;

  // Window RMA operations
  [[nodiscard]] ncclResult_t winAllocate(
      size_t size,
      ncclComm_t comm,
      void** baseptr,
      RcclxWindow* winPtr,
      bool cpuBuf,
      const size_t signal_size = 256) override;
  [[nodiscard]] ncclResult_t winFree(ncclComm_t comm, RcclxWindow win) override;
  [[nodiscard]] ncclResult_t winPut(
      const void* originBuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      size_t targetDisp,
      RcclxWindow win,
      hipStream_t stream) override;
  [[nodiscard]] ncclResult_t winSharedQuery(
      int rank,
      ncclComm_t comm,
      RcclxWindow win,
      void** addr) override;
  [[nodiscard]] ncclResult_t winSignal(
      size_t signalDisp,
      uint64_t signalVal,
      int peer,
      RcclxWindow win,
      hipStream_t stream) override;
  [[nodiscard]] ncclResult_t winWaitSignal(
      size_t signal_disp,
      uint64_t cmp_val,
      RcclxWindowCmpOp cmp_op,
      RcclxWindow win,
      hipStream_t stream) override;

  // Group operations
  [[nodiscard]] ncclResult_t groupStart() override;
  [[nodiscard]] ncclResult_t groupEnd() override;

  [[nodiscard]] ncclResult_t commUserRank(const ncclComm_t comm, int* userRank)
      override;
  [[nodiscard]] ncclResult_t commCount(const ncclComm_t comm, int* count)
      override;

  // Custom reduction operations
  [[nodiscard]] ncclResult_t redOpCreatePreMulSum(
      ncclRedOp_t* op,
      void* scalar,
      ncclDataType_t datatype,
      ncclScalarResidence_t residence,
      ncclComm_t comm) override;

  [[nodiscard]] ncclResult_t redOpDestroy(ncclRedOp_t op, ncclComm_t comm)
      override;
};

} // namespace torch::comms
