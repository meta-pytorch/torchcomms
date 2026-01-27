// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <exception>
#include <string>

#include <glog/logging.h>
#include <nccl.h> // @manual=//comms/ncclx:nccl

namespace torch::comms {

// Forward declaration for NCCLXException
class NcclxApi;

// Custom exception class for better error handling
class NCCLXException : public std::exception {
 public:
  NCCLXException(
      NcclxApi& api,
      const std::string& message,
      ncclResult_t result,
      ncclComm_t comm);

  const char* what() const noexcept override;
  [[nodiscard]] ncclResult_t getResult() const noexcept;

 private:
  std::string message_;
  ncclResult_t result_;
};

#define NCCLX_CHECK(nccl_api, nccl_comm, call, err_str)            \
  do {                                                             \
    ncclResult_t status = call;                                    \
    if (status != ncclSuccess) {                                   \
      throw NCCLXException(*nccl_api, err_str, status, nccl_comm); \
    }                                                              \
  } while (0)

// Ignore variant for use in destructors - logs errors instead of throwing
#define NCCLX_CHECK_IGNORE(nccl_api, call, err_str)                        \
  do {                                                                     \
    ncclResult_t status = call;                                            \
    if (status != ncclSuccess) {                                           \
      LOG(ERROR) << "[TC] " << err_str << ": "                             \
                 << nccl_api->getErrorString(status) << " at " << __FILE__ \
                 << ":" << __LINE__;                                       \
    }                                                                      \
  } while (0)

using NcclxWindow = ncclWindow_t;
using NcclxWindowAccessType = ncclWinAccessType;
using NcclxWindowAttr = ncclWinAttr_t;

/**
 * Abstract interface for NCCL API operations.
 * This allows for dependency injection and testing by providing
 * a way to override NCCL API calls.
 */
class NcclxApi {
 public:
  virtual ~NcclxApi() = default;

  // Error handling
  virtual const char* getErrorString(ncclResult_t result) = 0;
  virtual std::string getLastError(ncclComm_t comm) = 0;

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
      cudaStream_t stream) = 0;

  virtual ncclResult_t recv(
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  // Collective operations
  virtual ncclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  virtual ncclResult_t bcast(
      void* buff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  virtual ncclResult_t allReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  virtual ncclResult_t reduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      int root,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  virtual ncclResult_t allGather(
      const void* sendbuff,
      void* recvbuff,
      size_t sendcount,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  virtual ncclResult_t reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      size_t recvcount,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  virtual ncclResult_t allToAll(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  virtual ncclResult_t allToAllv(
      const void* sendbuff,
      const size_t sendcounts[],
      const size_t sdispls[],
      void* recvbuff,
      const size_t recvcounts[],
      const size_t rdispls[],
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  virtual ncclResult_t alltoallvDynamicDispatch(
      const void* sendbuff,
      const size_t* sendSplitLengths,
      size_t numSendSplitLengths,
      const size_t* sendIndices,
      const size_t* sendIndicesBlockLengths,
      void* const* recvbuffs,
      size_t* recvAllSplitLengths,
      size_t maxSendcount,
      size_t maxRecvcount,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  virtual ncclResult_t alltoallvDynamicCombine(
      const void* sendbuff,
      const size_t* sendSplitLengths,
      size_t numSendSplitLengths,
      const size_t* sendIndices,
      const size_t* sendIndicesBlockLengths,
      void* recvbuff,
      size_t maxSendcount,
      size_t maxRecvcount,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  virtual ncclResult_t alltoallvDedupInit(
      const size_t totalNumSendBlocks, // number of blocks (tokens) per batch
      const size_t blockCount, // number of elements per block (token)
      const size_t blockNumRecvBuckets, // number of receiving buckets for each
                                        // block (experts per token, topK)
      const int numRecvBuckets, // number of receiving buckets per rank (expert
                                // per rank)
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream,
      void** request) = 0;

  virtual ncclResult_t alltoallvDedupExec(
      const void* sendBuff,
      const int* sendIdx,
      const int* fwdIdx,
      const int* recvIdx,
      void* recvBuff,
      int recvBlockIds[],
      void* request) = 0;

  virtual ncclResult_t alltoallvDedupCombine(
      const void* sendBuff,
      const int* sendIdx,
      const int* fwdIdx,
      const int* recvIdx,
      void* recvBuff,
      void* request) = 0;

  virtual ncclResult_t pFree(void* request) = 0;

  virtual ncclResult_t commWindowRegister(
      void* baseptr,
      const size_t size,
      ncclComm_t comm,
      NcclxWindow* winPtr) = 0;
  virtual ncclResult_t commWindowDeregister(
      ncclComm_t comm,
      NcclxWindow win) = 0;
  virtual ncclResult_t winPut(
      const void* originBuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      size_t targetOffsetNelems,
      NcclxWindow win,
      cudaStream_t stream) = 0;
  virtual ncclResult_t
  winSharedQuery(int rank, ncclComm_t comm, NcclxWindow win, void** addr) = 0;
  virtual ncclResult_t
  winSignal(int peer, NcclxWindow win, cudaStream_t stream) = 0;
  virtual ncclResult_t
  winWaitSignal(int peer, NcclxWindow win, cudaStream_t stream) = 0;
  virtual ncclResult_t
  winGetAttributes(int peer, NcclxWindow win, NcclxWindowAttr* attrPtr) = 0;

  virtual ncclResult_t memAlloc(void** buff, size_t size) = 0;
  virtual ncclResult_t memFree(void* buff) = 0;

  // Group operations
  virtual ncclResult_t groupStart() = 0;
  virtual ncclResult_t groupEnd() = 0;

  virtual ncclResult_t commUserRank(const ncclComm_t comm, int* userRank) = 0;
  virtual ncclResult_t commCount(const ncclComm_t comm, int* count) = 0;

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
class DefaultNcclxApi : public NcclxApi {
 public:
  ~DefaultNcclxApi() override = default;

  // Error handling
  const char* getErrorString(ncclResult_t result) override;
  std::string getLastError(ncclComm_t comm) override;

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
      cudaStream_t stream) override;

  ncclResult_t recv(
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      cudaStream_t stream) override;

  // Collective operations
  ncclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      cudaStream_t stream) override;

  ncclResult_t bcast(
      void* buff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      cudaStream_t stream) override;

  ncclResult_t allReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      cudaStream_t stream) override;

  ncclResult_t reduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      int root,
      ncclComm_t comm,
      cudaStream_t stream) override;

  ncclResult_t allGather(
      const void* sendbuff,
      void* recvbuff,
      size_t sendcount,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) override;

  ncclResult_t reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      size_t recvcount,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      cudaStream_t stream) override;

  ncclResult_t allToAll(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) override;

  ncclResult_t allToAllv(
      const void* sendbuff,
      const size_t sendcounts[],
      const size_t senddispls[],
      void* recvbuff,
      const size_t recvcounts[],
      const size_t recvdispls[],
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) override;

  ncclResult_t alltoallvDynamicDispatch(
      const void* sendbuff,
      const size_t* sendSplitLengths,
      size_t numSendSplitLengths,
      const size_t* sendIndices,
      const size_t* sendIndicesBlockLengths,
      void* const* recvbuffs,
      size_t* recvAllSplitLengths,
      size_t maxSendcount,
      size_t maxRecvcount,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) override;

  ncclResult_t alltoallvDynamicCombine(
      const void* sendbuff,
      const size_t* sendSplitLengths,
      size_t numSendSplitLengths,
      const size_t* sendIndices,
      const size_t* sendIndicesBlockLengths,
      void* recvbuff,
      size_t maxSendcount,
      size_t maxRecvcount,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) override;

  ncclResult_t alltoallvDedupInit(
      const size_t totalNumSendBlocks,
      const size_t blockCount,
      const size_t blockNumRecvBuckets,
      const int numRecvBuckets,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream,
      void** request) override;

  ncclResult_t alltoallvDedupExec(
      const void* sendBuff,
      const int* sendIdx,
      const int* fwdIdx,
      const int* recvIdx,
      void* recvBuff,
      int recvBlockIds[],
      void* request) override;

  ncclResult_t alltoallvDedupCombine(
      const void* sendBuff,
      const int* sendIdx,
      const int* fwdIdx,
      const int* recvIdx,
      void* recvBuff,
      void* request) override;

  ncclResult_t pFree(void* request) override;

  // Window RMA operations
  ncclResult_t commWindowRegister(
      void* baseptr,
      const size_t size,
      ncclComm_t comm,
      NcclxWindow* winPtr) override;
  ncclResult_t commWindowDeregister(ncclComm_t comm, NcclxWindow win) override;
  ncclResult_t winPut(
      const void* originBuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      size_t targetOffsetNelems,
      NcclxWindow win,
      cudaStream_t stream) override;
  ncclResult_t winSharedQuery(
      int rank,
      ncclComm_t comm,
      NcclxWindow win,
      void** addr) override;
  ncclResult_t winSignal(int peer, NcclxWindow win, cudaStream_t stream)
      override;
  ncclResult_t winWaitSignal(int peer, NcclxWindow win, cudaStream_t stream)
      override;
  ncclResult_t winGetAttributes(
      int peer,
      NcclxWindow win,
      NcclxWindowAttr* attrPtr) override;

  ncclResult_t memAlloc(void** buff, size_t size) override;
  ncclResult_t memFree(void* buff) override;

  // Group operations
  ncclResult_t groupStart() override;
  ncclResult_t groupEnd() override;

  ncclResult_t commUserRank(const ncclComm_t comm, int* userRank) override;
  ncclResult_t commCount(const ncclComm_t comm, int* count) override;

  ncclResult_t redOpCreatePreMulSum(
      ncclRedOp_t* op,
      void* scalar,
      ncclDataType_t datatype,
      ncclScalarResidence_t residence,
      ncclComm_t comm) override;
  ncclResult_t redOpDestroy(ncclRedOp_t op, ncclComm_t comm) override;
};

} // namespace torch::comms
