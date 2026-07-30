// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <gmock/gmock.h>

#include "comms/torchcomms/nccl/NcclApi.hpp"

namespace torch::comms::test {

/**
 * Mock implementation of NcclApi using Google Mock.
 * This class provides mock implementations of all NCCL API operations
 * for testing purposes without requiring actual NCCL hardware/setup.
 */
class NcclMock : public NcclApi {
 public:
  ~NcclMock() override = default;

  // Error handling
  MOCK_METHOD(const char*, getErrorString, (ncclResult_t result), (override));
  MOCK_METHOD(std::string, getLastError, (ncclComm_t comm), (override));

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

  MOCK_METHOD(ncclResult_t, commRevoke, (ncclComm_t comm), (override));

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

  MOCK_METHOD(
      ncclResult_t,
      commShrink,
      (ncclComm_t comm,
       int* excludeRanksList,
       int excludeRanksCount,
       ncclComm_t* newcomm,
       ncclConfig_t* config,
       int shrinkFlags),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      commGetUniqueId,
      (ncclComm_t comm, ncclUniqueId* uniqueId),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      commGrow,
      (ncclComm_t comm,
       int nRanks,
       const ncclUniqueId* uniqueId,
       int rank,
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

  MOCK_METHOD(ncclResult_t, memAlloc, (void** buff, size_t size), (override));
  MOCK_METHOD(ncclResult_t, memFree, (void* buff), (override));

  // Window / one-sided RMA operations
  MOCK_METHOD(
      ncclResult_t,
      commWindowRegister,
      (ncclComm_t comm,
       void* buffer,
       size_t size,
       ncclWindow_t* win,
       int winFlags),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      commWindowDeregister,
      (ncclComm_t comm, ncclWindow_t win),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      winGetUserPtr,
      (ncclComm_t comm, ncclWindow_t win, void** outUserPtr),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      putSignal,
      (const void* localbuff,
       size_t count,
       ncclDataType_t datatype,
       int peer,
       ncclWindow_t peerWin,
       size_t peerWinOffset,
       int sigIdx,
       int ctx,
       unsigned int flags,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      signal,
      (int peer,
       int sigIdx,
       int ctx,
       unsigned int flags,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));

  MOCK_METHOD(
      ncclResult_t,
      waitSignal,
      (int peer,
       int sigIdx,
       int ctx,
       int opCnt,
       ncclComm_t comm,
       cudaStream_t stream),
      (override));
};

} // namespace torch::comms::test
