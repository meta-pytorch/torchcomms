// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/tests/IbverbxTestFixture.h"

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

namespace ibverbx {

TEST_F(IbverbxTestFixture, IbvVirtualCq) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto cq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);
  ASSERT_NE(cq->getPhysicalCqsRef().at(0).cq(), nullptr);
  auto cqRawPtr = cq->getPhysicalCqsRef().at(0).cq();

  // move constructor
  auto cq1 = std::move(*cq);
  ASSERT_TRUE(cq->getPhysicalCqsRef().empty());
  ASSERT_EQ(cq1.getPhysicalCqsRef().size(), 1);
  ASSERT_EQ(cq1.getPhysicalCqsRef().at(0).cq(), cqRawPtr);

  IbvVirtualCq cq2(std::move(cq1));
  ASSERT_TRUE(cq1.getPhysicalCqsRef().empty());
  ASSERT_EQ(cq2.getPhysicalCqsRef().size(), 1);
  ASSERT_EQ(cq2.getPhysicalCqsRef().at(0).cq(), cqRawPtr);
}

TEST_F(IbverbxTestFixture, IbvVirtualCqRegisterPhysicalQp) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  auto virtualCq = std::move(*maybeVirtualCq);

  // Test 1: Register and lookup
  {
    IbvVirtualQp* fakeVqp = reinterpret_cast<IbvVirtualQp*>(0xBEEF);
    virtualCq.registerPhysicalQp(
        /*physicalQpNum=*/42,
        /*deviceId=*/0,
        fakeVqp,
        /*isMultiQp=*/true,
        /*virtualQpNum=*/100);

    auto* info = virtualCq.findRegisteredQpInfo(42, 0);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->vqp, fakeVqp);
    ASSERT_TRUE(info->isMultiQp);
    ASSERT_EQ(info->virtualQpNum, 100);
  }

  // Test 2: Unregister and verify lookup returns nullptr
  {
    virtualCq.unregisterPhysicalQp(42, 0);
    auto* info = virtualCq.findRegisteredQpInfo(42, 0);
    ASSERT_EQ(info, nullptr);
  }

  // Test 3: Same qpNum, different deviceIds are independent entries
  {
    IbvVirtualQp* fakeVqp1 = reinterpret_cast<IbvVirtualQp*>(0x1111);
    IbvVirtualQp* fakeVqp2 = reinterpret_cast<IbvVirtualQp*>(0x2222);

    virtualCq.registerPhysicalQp(99, /*deviceId=*/0, fakeVqp1, false, 200);
    virtualCq.registerPhysicalQp(99, /*deviceId=*/1, fakeVqp2, true, 201);

    auto* info0 = virtualCq.findRegisteredQpInfo(99, 0);
    auto* info1 = virtualCq.findRegisteredQpInfo(99, 1);
    ASSERT_NE(info0, nullptr);
    ASSERT_NE(info1, nullptr);
    ASSERT_EQ(info0->vqp, fakeVqp1);
    ASSERT_FALSE(info0->isMultiQp);
    ASSERT_EQ(info0->virtualQpNum, 200);
    ASSERT_EQ(info1->vqp, fakeVqp2);
    ASSERT_TRUE(info1->isMultiQp);
    ASSERT_EQ(info1->virtualQpNum, 201);

    // Unregister one, verify the other still exists
    virtualCq.unregisterPhysicalQp(99, 0);
    ASSERT_EQ(virtualCq.findRegisteredQpInfo(99, 0), nullptr);
    ASSERT_NE(virtualCq.findRegisteredQpInfo(99, 1), nullptr);

    virtualCq.unregisterPhysicalQp(99, 1);
  }

  // Test 4: Overwrite registration with same key
  {
    IbvVirtualQp* fakeVqp1 = reinterpret_cast<IbvVirtualQp*>(0x3333);
    IbvVirtualQp* fakeVqp2 = reinterpret_cast<IbvVirtualQp*>(0x4444);

    virtualCq.registerPhysicalQp(50, 0, fakeVqp1, false, 300);
    auto* info = virtualCq.findRegisteredQpInfo(50, 0);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->vqp, fakeVqp1);
    ASSERT_FALSE(info->isMultiQp);

    // Overwrite with different values
    virtualCq.registerPhysicalQp(50, 0, fakeVqp2, true, 301);
    info = virtualCq.findRegisteredQpInfo(50, 0);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->vqp, fakeVqp2);
    ASSERT_TRUE(info->isMultiQp);
    ASSERT_EQ(info->virtualQpNum, 301);

    virtualCq.unregisterPhysicalQp(50, 0);
  }

  // Test 5: Lookup unregistered QP returns nullptr
  {
    auto* info = virtualCq.findRegisteredQpInfo(12345, 99);
    ASSERT_EQ(info, nullptr);
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualCqRegisterPhysicalQpMoveSemantics) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;

  // Test 1: Move constructor preserves registrations and updates
  // VirtualQp back-pointer
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq1 = std::move(*maybeVirtualCq);

    auto initAttr =
        makeIbvQpInitAttr(virtualCq1.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    int totalQps = 4;
    auto maybeVirtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq1);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    // Register a physical QP with back-pointer to the VirtualQp
    uint32_t testQpNum = 77;
    virtualCq1.registerPhysicalQp(
        testQpNum, 0, &virtualQp, true, virtualQp.getVirtualQpNum());
    virtualQp.virtualCq_ = &virtualCq1;

    // Move-construct a new VirtualCq
    IbvVirtualCq virtualCq2(std::move(virtualCq1));

    // Verify registration was moved
    auto* info = virtualCq2.findRegisteredQpInfo(testQpNum, 0);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->vqp, &virtualQp);
    ASSERT_TRUE(info->isMultiQp);
    ASSERT_EQ(info->virtualQpNum, virtualQp.getVirtualQpNum());

    // Verify VirtualQp back-pointer was updated to point to new VirtualCq
    ASSERT_EQ(virtualQp.virtualCq_, &virtualCq2);

    virtualCq2.unregisterPhysicalQp(testQpNum, 0);
  }

  // Test 2: Move assignment preserves registrations and updates
  // VirtualQp back-pointer
  {
    auto maybeVirtualCq1 = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq1);
    auto virtualCq1 = std::move(*maybeVirtualCq1);

    auto maybeVirtualCq2 = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq2);
    auto virtualCq2 = std::move(*maybeVirtualCq2);

    auto initAttr =
        makeIbvQpInitAttr(virtualCq1.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    int totalQps = 4;
    auto maybeVirtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq1);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    // Register a physical QP with back-pointer to the VirtualQp
    uint32_t testQpNum = 88;
    virtualCq1.registerPhysicalQp(
        testQpNum, 0, &virtualQp, false, virtualQp.getVirtualQpNum());
    virtualQp.virtualCq_ = &virtualCq1;

    // Move-assign
    virtualCq2 = std::move(virtualCq1);

    // Verify registration was moved
    auto* info = virtualCq2.findRegisteredQpInfo(testQpNum, 0);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->vqp, &virtualQp);
    ASSERT_FALSE(info->isMultiQp);
    ASSERT_EQ(info->virtualQpNum, virtualQp.getVirtualQpNum());

    // Verify VirtualQp back-pointer was updated to point to new VirtualCq
    ASSERT_EQ(virtualQp.virtualCq_, &virtualCq2);

    virtualCq2.unregisterPhysicalQp(testQpNum, 0);
  }
}

} // namespace ibverbx

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
