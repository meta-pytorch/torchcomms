#pragma once

#include <gmock/gmock.h>
#include "comms/torchcomms/xccl/XcclApi.hpp"

namespace torch::comms::test {

class XcclMock : public XcclApi {
 public:
  ~XcclMock() override = default;

  MOCK_METHOD(const char*, getErrorString, (onecclResult_t result), (override));
  MOCK_METHOD(onecclResult_t, setDevice, (int device), (override));
  MOCK_METHOD(onecclResult_t, getUniqueId, (onecclUniqueId* uniqueId), (override));
  MOCK_METHOD(onecclResult_t, commInitRankConfig, (onecclComm_t* comm, int nranks, onecclUniqueId commId, int rank, onecclConfig_t* config), (override));
  MOCK_METHOD(onecclResult_t, commDestroy, (onecclComm_t comm), (override));
  MOCK_METHOD(onecclResult_t, commAbort, (onecclComm_t comm), (override));
  MOCK_METHOD(onecclResult_t, commGetAsyncError, (onecclComm_t comm, onecclResult_t* asyncError), (override));
  MOCK_METHOD(onecclResult_t, commSplit, (onecclComm_t comm, int color, int key, onecclComm_t* newcomm, onecclConfig_t* config), (override));
  MOCK_METHOD(onecclResult_t, commRegister, (onecclComm_t comm, void* buffer, size_t size, void** handle), (override));
  MOCK_METHOD(onecclResult_t, commDeregister, (onecclComm_t comm, void* handle), (override));

  // Point-to-point operations
  MOCK_METHOD(onecclResult_t, send, (const void* sendbuff, size_t count, onecclDataType_t datatype, int peer, onecclComm_t comm, xpuStream_t stream), (override));
  MOCK_METHOD(onecclResult_t, recv, (void* recvbuff, size_t count, onecclDataType_t datatype, int peer, onecclComm_t comm, xpuStream_t stream), (override));

  // Collective operations
  MOCK_METHOD(onecclResult_t, broadcast, (const void* sendbuff, void* recvbuff, size_t count, onecclDataType_t datatype, int root, onecclComm_t comm, xpuStream_t stream), (override));
  MOCK_METHOD(onecclResult_t, bcast, (void* buff, size_t count, onecclDataType_t datatype, int root, onecclComm_t comm, xpuStream_t stream), (override));
  MOCK_METHOD(onecclResult_t, allReduce, (const void* sendbuff, void* recvbuff, size_t count, onecclDataType_t datatype, onecclRedOp_t op, onecclComm_t comm, xpuStream_t stream), (override));
  MOCK_METHOD(onecclResult_t, reduce, (const void* sendbuff, void* recvbuff, size_t count, onecclDataType_t datatype, onecclRedOp_t op, int root, onecclComm_t comm, xpuStream_t stream), (override));
  MOCK_METHOD(onecclResult_t, allGather, (const void* sendbuff, void* recvbuff, size_t sendcount, onecclDataType_t datatype, onecclComm_t comm, xpuStream_t stream), (override));
  MOCK_METHOD(onecclResult_t, reduceScatter, (const void* sendbuff, void* recvbuff, size_t recvcount, onecclDataType_t datatype, onecclRedOp_t op, onecclComm_t comm, xpuStream_t stream), (override));
  MOCK_METHOD(onecclResult_t, allToAll, (const void* sendbuff, void* recvbuff, size_t count, onecclDataType_t datatype, onecclComm_t comm, xpuStream_t stream), (override));

  // Group operations
  MOCK_METHOD(onecclResult_t, groupStart, (), (override));
  MOCK_METHOD(onecclResult_t, groupEnd, (), (override));

  MOCK_METHOD(onecclResult_t, commUserRank, (const onecclComm_t comm, int* userRank), (override));
  MOCK_METHOD(onecclResult_t, commCount, (const onecclComm_t comm, int* count), (override));

  MOCK_METHOD(onecclResult_t, redOpCreatePreMulSum, (onecclRedOp_t* op, void* scalar, onecclDataType_t datatype, onecclScalarResidence_t residence, onecclComm_t comm), (override));
  MOCK_METHOD(onecclResult_t, redOpDestroy, (onecclRedOp_t op, onecclComm_t comm), (override));

  void setupDefaultBehaviors() {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::DoAll;
    using ::testing::SetArgPointee;
    
    ON_CALL(*this, setDevice(_)).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, getUniqueId(_)).WillByDefault(Return(onecclSuccess));
    
    ON_CALL(*this, commInitRankConfig(_, _, _, _, _))
        .WillByDefault(DoAll(SetArgPointee<0>(reinterpret_cast<onecclComm_t>(0x1234)), Return(onecclSuccess)));
        
    ON_CALL(*this, commDestroy(_)).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, commAbort(_)).WillByDefault(Return(onecclNotImplemented));
    ON_CALL(*this, commGetAsyncError(_, _)).WillByDefault(Return(onecclNotImplemented));
    ON_CALL(*this, commSplit(_, _, _, _, _)).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, commRegister(_, _, _, _)).WillByDefault(Return(onecclNotImplemented));
    ON_CALL(*this, commDeregister(_, _)).WillByDefault(Return(onecclNotImplemented));
    
    ON_CALL(*this, send(_, _, _, _, _, _)).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, recv(_, _, _, _, _, _)).WillByDefault(Return(onecclSuccess));
    
    ON_CALL(*this, broadcast(_, _, _, _, _, _, _)).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, bcast(_, _, _, _, _, _)).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, allReduce(_, _, _, _, _, _, _)).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, reduce(_, _, _, _, _, _, _, _)).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, allGather(_, _, _, _, _, _)).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, reduceScatter(_, _, _, _, _, _, _)).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, allToAll(_, _, _, _, _, _)).WillByDefault(Return(onecclSuccess));
    
    ON_CALL(*this, groupStart()).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, groupEnd()).WillByDefault(Return(onecclSuccess));
    
    ON_CALL(*this, commUserRank(_, _)).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, commCount(_, _)).WillByDefault(Return(onecclSuccess));
    
    ON_CALL(*this, redOpCreatePreMulSum(_, _, _, _, _)).WillByDefault(Return(onecclSuccess));
    ON_CALL(*this, redOpDestroy(_, _)).WillByDefault(Return(onecclSuccess));
    
    ON_CALL(*this, getErrorString(_)).WillByDefault(Return("Mock XCCL Error"));
  }
};

} // namespace torch::comms::test