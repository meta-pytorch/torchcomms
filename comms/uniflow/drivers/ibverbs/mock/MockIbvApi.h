// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>

#include "comms/uniflow/drivers/ibverbs/IbvApi.h"

namespace uniflow {

/// gmock-based mock for IbvApi.
/// All virtual methods are mocked. Use ON_CALL / EXPECT_CALL to configure
/// behavior. Wrap with testing::NiceMock to suppress warnings for unconfigured
/// methods.
class MockIbvApi : public IbvApi {
 public:
  MOCK_METHOD(Status, init, (), (override));

  // Device management
  MOCK_METHOD(
      Result<ibv_device**>,
      getDeviceList,
      (int* numDevices),
      (override));
  MOCK_METHOD(Status, freeDeviceList, (ibv_device * *list), (override));
  MOCK_METHOD(
      Result<const char*>,
      getDeviceName,
      (ibv_device * device),
      (override));
  MOCK_METHOD(
      Result<ibv_context*>,
      openDevice,
      (ibv_device * device),
      (override));
  MOCK_METHOD(Status, closeDevice, (ibv_context * context), (override));

  // Protection domain
  MOCK_METHOD(Result<ibv_pd*>, allocPd, (ibv_context * context), (override));
  MOCK_METHOD(Status, deallocPd, (ibv_pd * pd), (override));

  // Memory registration
  MOCK_METHOD(
      Result<ibv_mr*>,
      regMr,
      (ibv_pd * pd, void* addr, size_t length, int access),
      (override));
  MOCK_METHOD(Status, deregMr, (ibv_mr * mr), (override));
  MOCK_METHOD(Result<bool>, isDmaBufSupported, (ibv_pd * pd), (override));
  MOCK_METHOD(
      Result<ibv_mr*>,
      regDmabufMr,
      (ibv_pd * pd,
       uint64_t offset,
       size_t length,
       uint64_t iova,
       int fd,
       int access),
      (override));

  // Completion queue
  MOCK_METHOD(
      Result<ibv_cq*>,
      createCq,
      (ibv_context * context,
       int cqe,
       void* cqContext,
       ibv_comp_channel* channel,
       int compVector),
      (override));
  MOCK_METHOD(Status, destroyCq, (ibv_cq * cq), (override));
  MOCK_METHOD(
      Result<int>,
      pollCq,
      (ibv_cq * cq, int numEntries, ibv_wc* wc),
      (override));

  // Queue pair
  MOCK_METHOD(
      Result<ibv_qp*>,
      createQp,
      (ibv_pd * pd, ibv_qp_init_attr* attr),
      (override));
  MOCK_METHOD(Status, destroyQp, (ibv_qp * qp), (override));
  MOCK_METHOD(
      Status,
      modifyQp,
      (ibv_qp * qp, ibv_qp_attr* attr, int attrMask),
      (override));

  // Data path
  MOCK_METHOD(
      Status,
      postSend,
      (ibv_qp * qp, ibv_send_wr* wr, ibv_send_wr** badWr),
      (override));
  MOCK_METHOD(
      Status,
      postRecv,
      (ibv_qp * qp, ibv_recv_wr* wr, ibv_recv_wr** badWr),
      (override));

  // Query
  MOCK_METHOD(
      Status,
      queryDevice,
      (ibv_context * context, ibv_device_attr* attr),
      (override));
  MOCK_METHOD(
      Status,
      queryPort,
      (ibv_context * context, uint8_t portNum, ibv_port_attr* attr),
      (override));
  MOCK_METHOD(
      Status,
      queryGid,
      (ibv_context * context, uint8_t portNum, int index, ibv_gid* gid),
      (override));

  // MLX5 direct verbs
  MOCK_METHOD(
      Result<bool>,
      mlx5dvIsSupported,
      (ibv_device * device),
      (override));
  MOCK_METHOD(
      Status,
      mlx5dvGetDataDirectSysfsPath,
      (ibv_context * context, char* buf, size_t bufLen),
      (override));
  MOCK_METHOD(
      Result<ibv_mr*>,
      mlx5dvRegDmabufMr,
      (ibv_pd * pd,
       uint64_t offset,
       size_t length,
       uint64_t iova,
       int fd,
       int access,
       int mlx5Access),
      (override));
};

} // namespace uniflow
