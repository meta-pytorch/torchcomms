// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>
#include <nccl.h> // @manual
#include "comms/torchcomms/ncclx/NcclxApi.hpp"

namespace torch {
namespace comms {
namespace test {

/**
 * Mock implementation of NcclxApi using Google Mock.
 * This class provides mock implementations of all NCCL API operations
 * for testing purposes without requiring actual NCCL hardware/setup.
 */
class NcclxMock : public NcclxApi {
 public:
  ~NcclxMock() override = default;

  // Error handling
  MOCK_METHOD(const char*, getErrorString, (ncclResult_t result), (override));

  // Unique ID generation
  MOCK_METHOD(ncclResult_t, getUniqueId, (ncclUniqueId * uniqueId), (override));

  // Communicator management
  MOCK_METHOD(
      ncclResult_t,
      commInitRankConfig,
      (ncclComm_t * comm,
       int nranks,
       ncclUniqueId commId,
       int rank,
       ncclConfig_t* config),
      (override));

  MOCK_METHOD(ncclResult_t, commDestroy, (ncclComm_t comm), (override));

  MOCK_METHOD(ncclResult_t, commAbort, (ncclComm_t comm), (override));

  MOCK_METHOD(
      ncclResult_t,
      commGetAsyncError,
      (ncclComm_t comm, ncclResult_t* asyncError),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      commSplit,
      (ncclComm_t comm,
       int color,
       int key,
       ncclComm_t* newcomm,
       ncclConfig_t* config),
      (override));

  // Memory registration
  MOCK_METHOD(
      ncclResult_t,
      commRegister,
      (ncclComm_t comm, void* buffer, size_t size, void** handle),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      commDeregister,
      (ncclComm_t comm, void* handle),
      (override));

  // Point-to-point operations
  MOCK_METHOD(
      ncclResult_t,
      send,
      (const void* sendbuff,
       size_t count,
       ncclDataType_t datatype,
       int peer,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      recv,
      (void* recvbuff,
       size_t count,
       ncclDataType_t datatype,
       int peer,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  // Collective operations
  MOCK_METHOD(
      ncclResult_t,
      broadcast,
      (const void* sendbuff,
       void* recvbuff,
       size_t count,
       ncclDataType_t datatype,
       int root,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      bcast,
      (void* buff,
       size_t count,
       ncclDataType_t datatype,
       int root,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      allReduce,
      (const void* sendbuff,
       void* recvbuff,
       size_t count,
       ncclDataType_t datatype,
       ncclRedOp_t op,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      reduce,
      (const void* sendbuff,
       void* recvbuff,
       size_t count,
       ncclDataType_t datatype,
       ncclRedOp_t op,
       int root,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      allGather,
      (const void* sendbuff,
       void* recvbuff,
       size_t sendcount,
       ncclDataType_t datatype,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      reduceScatter,
      (const void* sendbuff,
       void* recvbuff,
       size_t recvcount,
       ncclDataType_t datatype,
       ncclRedOp_t op,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      allToAll,
      (const void* sendbuff,
       void* recvbuff,
       size_t count,
       ncclDataType_t datatype,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      allToAllv,
      (const void* sendbuff,
       const size_t* sendcounts,
       const size_t* sdispls,
       void* recvbuff,
       const size_t* recvcounts,
       const size_t* rdispls,
       ncclDataType_t datatype,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      alltoallvDynamicDispatch,
      (const void* sendbuff,
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
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      alltoallvDynamicCombine,
      (const void* sendbuff,
       const size_t* sendSplitLengths,
       size_t numSendSplitLengths,
       const size_t* sendIndices,
       const size_t* sendIndicesBlockLengths,
       void* recvbuff,
       size_t maxSendcount,
       size_t maxRecvcount,
       ncclDataType_t datatype,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      alltoallvDedupInit,
      (const size_t totalNumSendBlocks,
       const size_t blockCount,
       const size_t blockNumRecvBuckets,
       const int numRecvBuckets,
       ncclDataType_t datatype,
       ncclComm_t comm,
       cudaStream_t stream,
       void** reques),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      alltoallvDedupExec,
      (const void* sendBuff,
       const int* sendIdx,
       const int* fwdIdx,
       const int* recvIdx,
       void* recvBuff,
       int recvBlockIds[],
       void* request),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      alltoallvDedupCombine,
      (const void* sendBuff,
       const int* sendIdx,
       const int* fwdIdx,
       const int* recvIdx,
       void* recvBuff,
       void* request),
      (override));

  MOCK_METHOD(ncclResult_t, pFree, (void* request), (override));

  MOCK_METHOD(
      ncclResult_t,
      commWindowRegister,
      (void* baseptr, size_t size, ncclComm_t comm, NcclxWindow* win),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      commWindowDeregister,
      (ncclComm_t comm, NcclxWindow win),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      winPut,
      (const void* originBuff,
       size_t count,
       ncclDataType_t datatype,
       int peer,
       size_t targetDisp,
       NcclxWindow win,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      winSharedQuery,
      (int rank, ncclComm_t comm, NcclxWindow win, void** addr),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      winSignal,
      (int peer, NcclxWindow win, cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      winWaitSignal,
      (int peer, NcclxWindow win, cudaStream_t stream),
      (override));

  MOCK_METHOD(ncclResult_t, memAlloc, (void** buff, size_t size), (override));
  MOCK_METHOD(ncclResult_t, memFree, (void* buff), (override));

  // Group operations
  MOCK_METHOD(ncclResult_t, groupStart, (), (override));
  MOCK_METHOD(ncclResult_t, groupEnd, (), (override));

  MOCK_METHOD(
      ncclResult_t,
      commUserRank,
      (const ncclComm_t comm, int* userRank),
      (override));
  MOCK_METHOD(
      ncclResult_t,
      commCount,
      (const ncclComm_t comm, int* count),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      redOpCreatePreMulSum,
      (ncclRedOp_t * op,
       void* scalar,
       ncclDataType_t datatype,
       ncclScalarResidence_t residence,
       ncclComm_t comm),
      (override));
  MOCK_METHOD(
      ncclResult_t,
      redOpDestroy,
      (ncclRedOp_t op, ncclComm_t comm),
      (override));

  /**
   * Set up default behaviors for common NCCL operations.
   * This method configures the mock to return success for most operations
   * and provides reasonable default values for queries.
   */
  void setupDefaultBehaviors();

  /**
   * Reset all mock expectations and call counts.
   */
  void reset();
};

} // namespace test
} // namespace comms
} // namespace torch
