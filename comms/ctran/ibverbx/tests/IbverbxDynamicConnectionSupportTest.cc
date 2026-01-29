// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/ibverbx/Ibverbx.h"

namespace ibverbx {

// DC transport only supports mlx5 NICs
const std::string kNicPrefix("mlx5_");

constexpr uint8_t kPortNum = 1;

#if defined(USE_FE_NIC)
constexpr int kGidIndex = 1;
#else
constexpr int kGidIndex = 3;
#endif

// DC Authentication Key
constexpr uint64_t DC_KEY = 0x1234;

class IbverbxDcTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(ibvInit());
  }
};

// Test SRQ creation and basic operations
TEST_F(IbverbxDcTestFixture, IbvSrqCreation) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  auto pd = device.allocPd();
  ASSERT_TRUE(pd);
  ASSERT_NE(pd->pd(), nullptr);

  // Create SRQ init attributes
  ibv_srq_init_attr srqInitAttr{};
  memset(&srqInitAttr, 0, sizeof(ibv_srq_init_attr));
  srqInitAttr.attr.max_wr = 256;
  srqInitAttr.attr.max_sge = 1;

  auto srq = pd->createSrq(&srqInitAttr);
  // SRQ creation may fail if ibv_create_srq is not available on this platform
  if (srq.hasError() && srq.error().errNum == ENOSYS) {
    GTEST_SKIP() << "ibv_create_srq not available on this platform";
  }
  ASSERT_TRUE(srq) << "createSrq failed: " << srq.error().errStr;
  ASSERT_NE(srq->srq(), nullptr);

  // Test querySrq
  auto queryResult = srq->querySrq();
  ASSERT_TRUE(queryResult);
  EXPECT_GT(queryResult->max_wr, 0);

  // Test move constructor
  auto srqRawPtr = srq->srq();
  auto srq1 = std::move(*srq);
  ASSERT_EQ(srq->srq(), nullptr);
  ASSERT_EQ(srq1.srq(), srqRawPtr);

  IbvSrq srq2(std::move(srq1));
  ASSERT_EQ(srq1.srq(), nullptr);
  ASSERT_EQ(srq2.srq(), srqRawPtr);
}

// Test posting receive work requests to SRQ
TEST_F(IbverbxDcTestFixture, IbvSrqPostRecv) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  // Create SRQ
  ibv_srq_init_attr srqInitAttr{};
  memset(&srqInitAttr, 0, sizeof(ibv_srq_init_attr));
  srqInitAttr.attr.max_wr = 256;
  srqInitAttr.attr.max_sge = 1;

  auto srq = pd->createSrq(&srqInitAttr);
  if (srq.hasError() && srq.error().errNum == ENOSYS) {
    GTEST_SKIP() << "ibv_create_srq not available on this platform";
  }
  ASSERT_TRUE(srq) << "createSrq failed: " << srq.error().errStr;

  // Post a receive work request to the SRQ
  // For DC transport, we typically post receives with no scatter-gather entries
  // since RDMA_WRITE_WITH_IMM only delivers immediate data, not payload
  ibv_sge recvSgList{};
  ibv_recv_wr recvWr{};
  recvWr.wr_id = 12345;
  recvWr.next = nullptr;
  recvWr.sg_list = &recvSgList;
  recvWr.num_sge = 0;

  ibv_recv_wr* badRecvWr = nullptr;
  auto postResult = srq->postRecv(&recvWr, &badRecvWr);
  ASSERT_TRUE(postResult) << "postRecv failed";
}

// Test DCI (DC Initiator) creation
TEST_F(IbverbxDcTestFixture, IbvDciCreation) {
#if defined(__HIP_PLATFORM_AMD__)
  GTEST_SKIP() << "Skipping DC QP test on AMD platform: mlx5dv not supported";
#else
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int cqe = 1024;
  auto cq = device.createCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);

  // Create DCI (DC Initiator)
  ibv_qp_init_attr_ex initAttrEx{};
  memset(&initAttrEx, 0, sizeof(ibv_qp_init_attr_ex));
  initAttrEx.qp_type = IBV_QPT_DRIVER;
  initAttrEx.send_cq = cq->cq();
  initAttrEx.recv_cq = cq->cq();
  initAttrEx.comp_mask = IBV_QP_INIT_ATTR_PD;
  initAttrEx.cap.max_send_wr = 1024;
  initAttrEx.cap.max_send_sge = 1;

  mlx5dv_qp_init_attr mlx5InitAttr{};
  memset(&mlx5InitAttr, 0, sizeof(mlx5dv_qp_init_attr));
  mlx5InitAttr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
  mlx5InitAttr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;

  auto dci = pd->createDcQp(&initAttrEx, &mlx5InitAttr);
  if (dci.hasError() && dci.error().errNum == ENOTSUP) {
    GTEST_SKIP() << "mlx5dv_create_qp not available on this platform";
  }
  ASSERT_TRUE(dci) << "createDcQp (DCI) failed: " << dci.error().errStr;
  ASSERT_NE(dci->qp(), nullptr);

  // Test move constructor
  auto dciRawPtr = dci->qp();
  auto dci1 = std::move(*dci);
  ASSERT_EQ(dci->qp(), nullptr);
  ASSERT_EQ(dci1.qp(), dciRawPtr);

  IbvQp dci2(std::move(dci1));
  ASSERT_EQ(dci1.qp(), nullptr);
  ASSERT_EQ(dci2.qp(), dciRawPtr);
#endif
}

// Test DCT (DC Target) creation with SRQ
TEST_F(IbverbxDcTestFixture, IbvDctCreation) {
#if defined(__HIP_PLATFORM_AMD__)
  GTEST_SKIP() << "Skipping DC QP test on AMD platform: mlx5dv not supported";
#else
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int cqe = 1024;
  auto cq = device.createCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);

  // Create SRQ (required for DCT)
  ibv_srq_init_attr srqInitAttr{};
  memset(&srqInitAttr, 0, sizeof(ibv_srq_init_attr));
  srqInitAttr.attr.max_wr = 256;
  srqInitAttr.attr.max_sge = 1;

  auto srq = pd->createSrq(&srqInitAttr);
  if (srq.hasError() && srq.error().errNum == ENOSYS) {
    GTEST_SKIP() << "ibv_create_srq not available on this platform";
  }
  ASSERT_TRUE(srq) << "createSrq failed: " << srq.error().errStr;

  // Create DCT (DC Target)
  ibv_qp_init_attr_ex initAttrEx{};
  memset(&initAttrEx, 0, sizeof(ibv_qp_init_attr_ex));
  initAttrEx.qp_type = IBV_QPT_DRIVER;
  initAttrEx.send_cq = cq->cq();
  initAttrEx.recv_cq = cq->cq();
  initAttrEx.srq = srq->srq(); // DCT requires SRQ
  initAttrEx.comp_mask = IBV_QP_INIT_ATTR_PD;

  mlx5dv_qp_init_attr mlx5InitAttr{};
  memset(&mlx5InitAttr, 0, sizeof(mlx5dv_qp_init_attr));
  mlx5InitAttr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
  mlx5InitAttr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
  mlx5InitAttr.dc_init_attr.dct_access_key = DC_KEY;

  auto dct = pd->createDcQp(&initAttrEx, &mlx5InitAttr);
  if (dct.hasError() && dct.error().errNum == ENOTSUP) {
    GTEST_SKIP() << "mlx5dv_create_qp not available on this platform";
  }
  ASSERT_TRUE(dct) << "createDcQp (DCT) failed: " << dct.error().errStr;
  ASSERT_NE(dct->qp(), nullptr);

  // Test move constructor
  auto dctRawPtr = dct->qp();
  auto dct1 = std::move(*dct);
  ASSERT_EQ(dct->qp(), nullptr);
  ASSERT_EQ(dct1.qp(), dctRawPtr);

  IbvQp dct2(std::move(dct1));
  ASSERT_EQ(dct1.qp(), nullptr);
  ASSERT_EQ(dct2.qp(), dctRawPtr);
#endif
}

// Test full DC setup: DCI + DCT + SRQ together
TEST_F(IbverbxDcTestFixture, IbvDcFullSetup) {
#if defined(__HIP_PLATFORM_AMD__)
  GTEST_SKIP() << "Skipping DC QP test on AMD platform: mlx5dv not supported";
#else
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int cqe = 1024;
  auto cq = device.createCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);

  // Create SRQ
  ibv_srq_init_attr srqInitAttr{};
  memset(&srqInitAttr, 0, sizeof(ibv_srq_init_attr));
  srqInitAttr.attr.max_wr = 256;
  srqInitAttr.attr.max_sge = 1;

  auto srq = pd->createSrq(&srqInitAttr);
  if (srq.hasError() && srq.error().errNum == ENOSYS) {
    GTEST_SKIP() << "ibv_create_srq not available on this platform";
  }
  ASSERT_TRUE(srq) << "createSrq failed: " << srq.error().errStr;

  // Create DCI
  ibv_qp_init_attr_ex initAttrExDci{};
  memset(&initAttrExDci, 0, sizeof(ibv_qp_init_attr_ex));
  initAttrExDci.qp_type = IBV_QPT_DRIVER;
  initAttrExDci.send_cq = cq->cq();
  initAttrExDci.recv_cq = cq->cq();
  initAttrExDci.comp_mask = IBV_QP_INIT_ATTR_PD;
  initAttrExDci.cap.max_send_wr = 1024;
  initAttrExDci.cap.max_send_sge = 1;

  mlx5dv_qp_init_attr mlx5InitAttrDci{};
  memset(&mlx5InitAttrDci, 0, sizeof(mlx5dv_qp_init_attr));
  mlx5InitAttrDci.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
  mlx5InitAttrDci.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;

  auto dci = pd->createDcQp(&initAttrExDci, &mlx5InitAttrDci);
  if (dci.hasError() && dci.error().errNum == ENOTSUP) {
    GTEST_SKIP() << "mlx5dv_create_qp not available on this platform";
  }
  ASSERT_TRUE(dci) << "createDcQp (DCI) failed: " << dci.error().errStr;
  ASSERT_NE(dci->qp(), nullptr);

  // Create DCT
  ibv_qp_init_attr_ex initAttrExDct{};
  memset(&initAttrExDct, 0, sizeof(ibv_qp_init_attr_ex));
  initAttrExDct.qp_type = IBV_QPT_DRIVER;
  initAttrExDct.send_cq = cq->cq();
  initAttrExDct.recv_cq = cq->cq();
  initAttrExDct.srq = srq->srq();
  initAttrExDct.comp_mask = IBV_QP_INIT_ATTR_PD;

  mlx5dv_qp_init_attr mlx5InitAttrDct{};
  memset(&mlx5InitAttrDct, 0, sizeof(mlx5dv_qp_init_attr));
  mlx5InitAttrDct.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
  mlx5InitAttrDct.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
  mlx5InitAttrDct.dc_init_attr.dct_access_key = DC_KEY;

  auto dct = pd->createDcQp(&initAttrExDct, &mlx5InitAttrDct);
  ASSERT_TRUE(dct) << "createDcQp (DCT) failed: " << dct.error().errStr;
  ASSERT_NE(dct->qp(), nullptr);

  // Post receives to SRQ
  for (int i = 0; i < 10; i++) {
    ibv_sge recvSgList{};
    ibv_recv_wr recvWr{};
    recvWr.wr_id = static_cast<uint64_t>(i);
    recvWr.next = nullptr;
    recvWr.sg_list = &recvSgList;
    recvWr.num_sge = 0;

    ibv_recv_wr* badRecvWr = nullptr;
    auto postResult = srq->postRecv(&recvWr, &badRecvWr);
    ASSERT_TRUE(postResult) << "postRecv failed for wr_id " << i;
  }

  // Verify we can query the GID (needed for DC address handles)
  auto gid = device.queryGid(kPortNum, kGidIndex);
  ASSERT_TRUE(gid) << "queryGid failed";
  EXPECT_NE(gid->global.interface_id, 0);
#endif
}

// Test Address Handle creation (needed for DC routing)
TEST_F(IbverbxDcTestFixture, IbvAddressHandleCreation) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  // Query GID for address handle
  auto gid = device.queryGid(kPortNum, kGidIndex);
  ASSERT_TRUE(gid) << "queryGid failed";

  // Create address handle
  ibv_ah_attr ahAttr{};
  memset(&ahAttr, 0, sizeof(ibv_ah_attr));
  ahAttr.is_global = 1;
  ahAttr.grh.dgid = *gid; // Use local GID as destination for testing
  ahAttr.grh.flow_label = 0;
  ahAttr.grh.sgid_index = kGidIndex;
  ahAttr.grh.hop_limit = 255;
  ahAttr.grh.traffic_class = 0;
  ahAttr.sl = 0;
  ahAttr.src_path_bits = 0;
  ahAttr.port_num = kPortNum;

  auto ah = pd->createAh(&ahAttr);
  if (ah.hasError() && ah.error().errNum == ENOSYS) {
    GTEST_SKIP() << "ibv_create_ah not available on this platform";
  }
  ASSERT_TRUE(ah) << "createAh failed: " << ah.error().errStr;
  ASSERT_NE(ah->ah(), nullptr);

  // Test move constructor
  auto ahRawPtr = ah->ah();
  auto ah1 = std::move(*ah);
  ASSERT_EQ(ah->ah(), nullptr);
  ASSERT_EQ(ah1.ah(), ahRawPtr);

  IbvAh ah2(std::move(ah1));
  ASSERT_EQ(ah1.ah(), nullptr);
  ASSERT_EQ(ah2.ah(), ahRawPtr);
}

} // namespace ibverbx
