// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <random>

#define IBVERBX_TEST_FRIENDS       \
  friend class IbverbxTestFixture; \
  FRIEND_TEST(                     \
      IbverbxTestFixture, IbvVirtualQpUpdatePhysicalSendWrFromVirtualSendWr);

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/utils/checks.h"

namespace ibverbx {
// use broadcom nic for AMD platform, use mellanox nic for NV platform
#if defined(__HIP_PLATFORM_AMD__) && !defined(USE_FE_NIC)
const std::string kNicPrefix("bnxt_re");
#else
const std::string kNicPrefix("mlx5_");
#endif

constexpr uint8_t kPortNum = 1;

#if defined(USE_FE_NIC)
constexpr int kGidIndex = 1;
#else
constexpr int kGidIndex = 3;
#endif

// helper functions
ibv_qp_init_attr makeIbvQpInitAttr(ibv_cq* cq) {
  ibv_qp_init_attr initAttr{};
  memset(&initAttr, 0, sizeof(ibv_qp_init_attr));
  initAttr.send_cq = cq;
  initAttr.recv_cq = cq;
  initAttr.qp_type = IBV_QPT_RC; // Reliable Connection
  initAttr.sq_sig_all = 0;
  initAttr.cap.max_send_wr = 256; // maximum outstanding send WRs
  initAttr.cap.max_recv_wr = 256; // maximum outstanding recv WRs
  initAttr.cap.max_send_sge = 1;
  initAttr.cap.max_recv_sge = 1;
  initAttr.cap.max_inline_data = 0;
  return initAttr;
}

class IbverbxTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(ibvInit());
  }
};

TEST_F(IbverbxTestFixture, MultiThreadInit) {
  std::thread t1([]() { ASSERT_TRUE(ibvInit()); });
  std::thread t2([]() { ASSERT_TRUE(ibvInit()); });
  t1.join();
  t2.join();
}

TEST_F(IbverbxTestFixture, IbvGetDeviceList) {
#if defined(USE_FE_NIC)
  GTEST_SKIP() << "Skipping IbvGetDeviceList test when using Frontend NIC";
#endif

  // Setup random generator with fixed seed for reproducible tests
  std::random_device rd;
  std::mt19937 gen(rd());

  // First, get all available devices with prefix match to use for dynamic
  // selection
  auto allPrefixDevices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(allPrefixDevices);
  ASSERT_GT(allPrefixDevices->size(), 0);

  // Extract available device names
  std::vector<std::string> availableDeviceNames;
  std::vector<std::string> availableDeviceNamesWithPort;
  for (const auto& device : *allPrefixDevices) {
    std::string deviceName = device.device()->name;
    availableDeviceNames.push_back(deviceName);
    availableDeviceNamesWithPort.push_back(deviceName + ":1");
  }

  {
    // get all ib-devices
    auto ibvDevices = IbvDevice::ibvGetDeviceList();
    ASSERT_TRUE(ibvDevices);
    ASSERT_GT(ibvDevices->size(), 0);

    // Print all found ibv device names
    XLOGF(INFO, "Found {} InfiniBand devices:", ibvDevices->size());
    for (size_t i = 0; i < ibvDevices->size(); ++i) {
      const auto& device = ibvDevices->at(i);
      XLOGF(INFO, "  Device[{}]: {}", i, device.device()->name);
    }
  }
  {
    // get ib devices with prefix match
    auto ibvDevices = IbvDevice::ibvGetDeviceList({kNicPrefix});
    ASSERT_TRUE(ibvDevices);
    ASSERT_GT(ibvDevices->size(), 0);
  }
  {
    // Get ib devices with exact match (in order) - dynamically select subset
    if (availableDeviceNamesWithPort.size() >= 3) {
      // Randomly select 3-5 devices (or max available if less than 5)
      size_t numDevicesToSelect = std::min(
          static_cast<size_t>(3 + (gen() % 3)),
          availableDeviceNamesWithPort.size());

      std::vector<std::string> selectedDevices;
      std::sample(
          availableDeviceNamesWithPort.begin(),
          availableDeviceNamesWithPort.end(),
          std::back_inserter(selectedDevices),
          numDevicesToSelect,
          gen);

      XLOGF(
          INFO,
          "Testing exact match (in order) with {} randomly selected devices",
          numDevicesToSelect);

      auto ibvDevices = IbvDevice::ibvGetDeviceList(selectedDevices, "=", -1);
      ASSERT_TRUE(ibvDevices);
      ASSERT_EQ(ibvDevices->size(), selectedDevices.size());
      for (size_t i = 0; i < ibvDevices->size(); ++i) {
        const auto& device = ibvDevices->at(i);
        ASSERT_EQ(device.device()->name, selectedDevices[i]);
      }
    }
  }
  {
    // Get ib devices with exact match (out of order) - dynamically select and
    // shuffle
    if (availableDeviceNamesWithPort.size() >= 2) {
      // Randomly select devices (2 to all available)
      size_t numDevicesToSelect = std::min(
          static_cast<size_t>(
              2 + (gen() % (availableDeviceNamesWithPort.size() - 1))),
          availableDeviceNamesWithPort.size());

      std::vector<std::string> selectedDevices;
      std::sample(
          availableDeviceNamesWithPort.begin(),
          availableDeviceNamesWithPort.end(),
          std::back_inserter(selectedDevices),
          numDevicesToSelect,
          gen);

      // Shuffle to create out-of-order
      std::shuffle(selectedDevices.begin(), selectedDevices.end(), gen);

      XLOGF(
          INFO,
          "Testing exact match (out of order) with {} randomly selected and shuffled devices",
          numDevicesToSelect);

      auto ibvDevices = IbvDevice::ibvGetDeviceList(selectedDevices, "=", -1);
      ASSERT_TRUE(ibvDevices);
      ASSERT_EQ(ibvDevices->size(), selectedDevices.size());
      for (size_t i = 0; i < ibvDevices->size(); ++i) {
        const auto& device = ibvDevices->at(i);
        ASSERT_EQ(device.device()->name, selectedDevices[i]);
      }
    }
  }
  {
    // Get ib devices with exclude (in order) - dynamically select devices to
    // exclude
    if (availableDeviceNames.size() >= 2) {
      // Randomly select 1-3 devices to exclude (but not all)
      size_t numDevicesToExclude = std::min(
          static_cast<size_t>(1 + (gen() % 3)),
          availableDeviceNames.size() - 1 // Leave at least 1 device
      );

      std::vector<std::string> devicesToExclude;
      std::sample(
          availableDeviceNames.begin(),
          availableDeviceNames.end(),
          std::back_inserter(devicesToExclude),
          numDevicesToExclude,
          gen);

      XLOGF(
          INFO,
          "Testing exclude (in order) with {} randomly selected devices to exclude",
          numDevicesToExclude);

      auto ibvDevices = IbvDevice::ibvGetDeviceList(devicesToExclude, "^", -1);
      ASSERT_TRUE(ibvDevices);

      // Verify excluded devices are not present
      for (auto it = ibvDevices->begin(); it != ibvDevices->end(); ++it) {
        std::string deviceName = it->device()->name;
        for (const auto& excludedDevice : devicesToExclude) {
          ASSERT_NE(deviceName, excludedDevice);
        }
      }

      // Verify we got the expected number of devices (total - excluded)
      size_t expectedDeviceCount =
          allPrefixDevices->size() - numDevicesToExclude;
      ASSERT_EQ(ibvDevices->size(), expectedDeviceCount);
    }
  }
  {
    // Get ib devices with exclude (out of order) - dynamically select and
    // shuffle excluded devices
    if (availableDeviceNames.size() >= 2) {
      // Randomly select devices to exclude (but not all)
      size_t numDevicesToExclude = std::min(
          static_cast<size_t>(1 + (gen() % 2)),
          availableDeviceNames.size() - 1 // Leave at least 1 device
      );

      std::vector<std::string> devicesToExclude;
      std::sample(
          availableDeviceNames.begin(),
          availableDeviceNames.end(),
          std::back_inserter(devicesToExclude),
          numDevicesToExclude,
          gen);

      // Shuffle the exclusion list
      std::shuffle(devicesToExclude.begin(), devicesToExclude.end(), gen);

      XLOGF(
          INFO,
          "Testing exclude (out of order) with {} randomly selected and shuffled devices to exclude",
          numDevicesToExclude);

      auto ibvDevices = IbvDevice::ibvGetDeviceList(devicesToExclude, "^", -1);
      ASSERT_TRUE(ibvDevices);

      // Verify excluded devices are not present
      for (auto it = ibvDevices->begin(); it != ibvDevices->end(); ++it) {
        std::string deviceName = it->device()->name;
        for (const auto& excludedDevice : devicesToExclude) {
          ASSERT_NE(deviceName, excludedDevice);
        }
      }

      // Verify we got the expected number of devices (total - excluded)
      size_t expectedDeviceCount =
          allPrefixDevices->size() - numDevicesToExclude;
      ASSERT_EQ(ibvDevices->size(), expectedDeviceCount);
    }
  }
}

TEST_F(IbverbxTestFixture, IbvDevice) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);
  ASSERT_NE(device.device(), nullptr);
  ASSERT_NE(device.context(), nullptr);
  ASSERT_NE(device.device()->name, nullptr);
  ASSERT_NE(device.context()->device, nullptr);

  auto devRawPtr = device.device();
  auto contextRawPtr = device.context();

  // move constructor
  auto device1 = std::move(device);
  ASSERT_EQ(device.device(), nullptr);
  ASSERT_EQ(device.context(), nullptr);
  ASSERT_EQ(device1.device(), devRawPtr);
  ASSERT_EQ(device1.context(), contextRawPtr);

  IbvDevice device2(std::move(device1));
  ASSERT_EQ(device1.device(), nullptr);
  ASSERT_EQ(device1.context(), nullptr);
  ASSERT_EQ(device2.device(), devRawPtr);
  ASSERT_EQ(device2.context(), contextRawPtr);
}

TEST_F(IbverbxTestFixture, IbvPd) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  // constructor
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);
  ASSERT_NE(pd->pd(), nullptr);

  auto pdRawPtr = pd->pd();

  // move constructor
  auto pd1 = std::move(*pd);
  ASSERT_EQ(pd->pd(), nullptr);
  ASSERT_EQ(pd1.pd(), pdRawPtr);

  IbvPd pd2(std::move(pd1));
  ASSERT_EQ(pd1.pd(), nullptr);
  ASSERT_EQ(pd2.pd(), pdRawPtr);
}

TEST_F(IbverbxTestFixture, IbvMr) {
  auto devices = IbvDevice::ibvGetDeviceList();
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  // alloc Pd
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);
  ASSERT_NE(pd->pd(), nullptr);

  // register Mr
  void* devBuf{nullptr};
  size_t devBufSize = 8192;
  CUDA_CHECK(cudaMalloc(&devBuf, devBufSize));

  ibv_access_flags access = static_cast<ibv_access_flags>(
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ);
  auto mr = pd->regMr(devBuf, devBufSize, access);
  EXPECT_TRUE(mr);

  auto mrRawPtr = mr->mr();

  // move constructor
  auto mr1 = std::move(*mr);
  ASSERT_EQ(mr->mr(), nullptr);
  ASSERT_EQ(mr1.mr(), mrRawPtr);

  IbvMr mr2(std::move(mr1));
  ASSERT_EQ(mr1.mr(), nullptr);
  ASSERT_EQ(mr2.mr(), mrRawPtr);
}

// Skip this test on AMD platform as there is no current support for
// cuMemGetHandleForAddressRange function on AMD GPUs according to docs-6.4.1
TEST_F(IbverbxTestFixture, regDmabufMr) {
#if defined(__HIP_PLATFORM_AMD__)
  GTEST_SKIP()
      << "Skipping regDmabufMr test on AMD platform: cuMemGetHandleForAddressRange not supported";
#else
  auto devices = IbvDevice::ibvGetDeviceList();
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  // alloc Pd
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);
  ASSERT_NE(pd->pd(), nullptr);

  // register Mr
  void* devBuf{nullptr};
  size_t devBufSize = 8192;
  CUDA_CHECK(cudaMalloc(&devBuf, devBufSize));

  // get fd
  int fd;
  cuMemGetHandleForAddressRange(
      (void*)(&fd),
      reinterpret_cast<CUdeviceptr>(devBuf),
      devBufSize,
      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
      0);

  ibv_access_flags access = static_cast<ibv_access_flags>(
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ);

  auto dmabufMr = pd->regDmabufMr(
      0, devBufSize, reinterpret_cast<CUdeviceptr>(devBuf), fd, access);
  EXPECT_TRUE(dmabufMr);

  auto mrRawPtr = dmabufMr->mr();

  // move constructor
  auto dmabufMr1 = std::move(*dmabufMr);
  ASSERT_EQ(dmabufMr->mr(), nullptr);
  ASSERT_EQ(dmabufMr1.mr(), mrRawPtr);

  IbvMr dmabufMr2(std::move(dmabufMr1));
  ASSERT_EQ(dmabufMr1.mr(), nullptr);
  ASSERT_EQ(dmabufMr2.mr(), mrRawPtr);
#endif
}

TEST_F(IbverbxTestFixture, regMr) {
  auto devices = IbvDevice::ibvGetDeviceList();
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  // alloc Pd
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);
  ASSERT_NE(pd->pd(), nullptr);

  ibv_access_flags access = static_cast<ibv_access_flags>(
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ);

  // Setup random generator for dynamic buffer size
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(1, 4096);

  // Test various buffer sizes for regMr call
  std::vector<size_t> sizes = {1, 8, 1024, 2048, 4096, dis(gen)};

  for (size_t size : sizes) {
    void* devBuf{nullptr};
    XLOGF(INFO, "regMr testing with buffer size: {} bytes", size);
    CUDA_CHECK(cudaMalloc(&devBuf, size));
    auto mr = pd->regMr(devBuf, size, access);
    EXPECT_TRUE(mr);
    CUDA_CHECK(cudaFree(devBuf));
  }
}

TEST_F(IbverbxTestFixture, IbvDeviceQueries) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  // query device
  auto devAttr = device.queryDevice();
  ASSERT_TRUE(devAttr);
  ASSERT_GT(devAttr->phys_port_cnt, 0);

  // query port
  auto portAttr = device.queryPort(kPortNum);
  ASSERT_TRUE(portAttr);
  ASSERT_GT(portAttr->gid_tbl_len, 0);

  // query gid
  auto gid = device.queryGid(kPortNum, kGidIndex);
  ASSERT_TRUE(gid);
  ASSERT_NE(gid->raw, nullptr);
}

TEST_F(IbverbxTestFixture, IbvCq) {
  auto devices = IbvDevice::ibvGetDeviceList();
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto cq = device.createCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);
  ASSERT_NE(cq->cq(), nullptr);
  auto cqRawPtr = cq->cq();

  // move constructor
  auto cq1 = std::move(*cq);
  ASSERT_EQ(cq->cq(), nullptr);
  ASSERT_EQ(cq1.cq(), cqRawPtr);

  IbvCq cq2(std::move(cq1));
  ASSERT_EQ(cq1.cq(), nullptr);
  ASSERT_EQ(cq2.cq(), cqRawPtr);
}

TEST_F(IbverbxTestFixture, IbvVirtualCq) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto cq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);
  ASSERT_NE(cq->getPhysicalCqRef().cq(), nullptr);
  auto cqRawPtr = cq->getPhysicalCqRef().cq();

  // move constructor
  auto cq1 = std::move(*cq);
  ASSERT_EQ(cq->getPhysicalCqRef().cq(), nullptr);
  ASSERT_EQ(cq1.getPhysicalCqRef().cq(), cqRawPtr);

  IbvVirtualCq cq2(std::move(cq1));
  ASSERT_EQ(cq1.getPhysicalCqRef().cq(), nullptr);
  ASSERT_EQ(cq2.getPhysicalCqRef().cq(), cqRawPtr);
}

TEST_F(IbverbxTestFixture, IbvQp) {
  auto devices = IbvDevice::ibvGetDeviceList();
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto cq = device.createCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);
  ASSERT_NE(cq->cq(), nullptr);

  auto initAttr = makeIbvQpInitAttr(cq->cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);
  auto qp = pd->createQp(&initAttr);

  ASSERT_TRUE(qp);
  ASSERT_NE(qp->qp(), nullptr);
  auto pqRawPtr = qp->qp();

  // Test queryQp with multiple attributes before any moves
  {
    int attrMask = IBV_QP_STATE | IBV_QP_CAP | IBV_QP_PKEY_INDEX | IBV_QP_PORT;
    auto queryResult = qp->queryQp(attrMask);
    ASSERT_TRUE(queryResult);

    auto [qpAttr, qpInitAttr] = queryResult.value();

    // Verify the QP is initially in RESET state
    ASSERT_EQ(qpAttr.qp_state, IBV_QPS_RESET);

    // Verify QP type matches
    ASSERT_EQ(qpInitAttr.qp_type, IBV_QPT_RC);

    // Verify capabilities match what we set
    ASSERT_EQ(qpInitAttr.cap.max_send_wr, initAttr.cap.max_send_wr);
    ASSERT_EQ(qpInitAttr.cap.max_recv_wr, initAttr.cap.max_recv_wr);
    ASSERT_EQ(qpInitAttr.cap.max_send_sge, initAttr.cap.max_send_sge);
    ASSERT_EQ(qpInitAttr.cap.max_recv_sge, initAttr.cap.max_recv_sge);
  }

  // Test move constructor
  auto qp1 = std::move(*qp);
  ASSERT_EQ(qp->qp(), nullptr);
  ASSERT_EQ(qp1.qp(), pqRawPtr);

  IbvQp qp2(std::move(qp1));
  ASSERT_EQ(qp1.qp(), nullptr);
  ASSERT_EQ(qp2.qp(), pqRawPtr);

  // Test queryQp after move
  {
    auto queryResult = qp2.queryQp(IBV_QP_STATE);
    ASSERT_TRUE(queryResult);

    auto [qpAttr, qpInitAttr] = queryResult.value();
    ASSERT_EQ(qpAttr.qp_state, IBV_QPS_RESET);
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualQp) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqRef().cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqRef().cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  auto virtualQp =
      pd->createVirtualQp(totalQps, &initAttr, &virtualCq, &virtualCq);
  ASSERT_TRUE(virtualQp);
  ASSERT_EQ(virtualQp->getQpsRef().size(), totalQps);
  ASSERT_EQ(virtualQp->getTotalQps(), totalQps);

  // Store the first QP's raw pointer, and notifyQp's raw pointer for comparison
  // after move
  const auto firstQpPtr = virtualQp->getQpsRef()[0].qp();
  const auto notifyQpPtr = virtualQp->getNotifyQpRef().qp();

  // move constructor
  auto qpg1 = std::move(*virtualQp);
  ASSERT_TRUE(
      virtualQp->getQpsRef().empty()); // After move, vector should be empty
  ASSERT_EQ(qpg1.getQpsRef().size(), totalQps); // Size should match original
  ASSERT_EQ(qpg1.getQpsRef()[0].qp(), firstQpPtr); // First element should match
  ASSERT_EQ(qpg1.getNotifyQpRef().qp(), notifyQpPtr); // Notify QP should match

  IbvVirtualQp qpg2(std::move(qpg1));
  ASSERT_TRUE(qpg1.getQpsRef().empty()); // After move, vector should be empty
  ASSERT_EQ(qpg2.getQpsRef().size(), totalQps); // Size should match original
  ASSERT_EQ(qpg2.getQpsRef()[0].qp(), firstQpPtr); // First element should match
  ASSERT_EQ(qpg2.getNotifyQpRef().qp(), notifyQpPtr); // Notify QP should match
}

TEST_F(IbverbxTestFixture, IbvVirtualQpFindAvailableSendQp) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqRef().cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqRef().cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 4;
  int maxMsgCntPerQp = 100;

  // Test setting 1: default maxMsgCntPerQp (100)
  {
    auto virtualQp = pd->createVirtualQp(
        totalQps, &initAttr, &virtualCq, &virtualCq, maxMsgCntPerQp);
    ASSERT_TRUE(virtualQp);
    ASSERT_EQ(virtualQp->getQpsRef().size(), totalQps);
    ASSERT_EQ(virtualQp->getTotalQps(), totalQps);

    // Mock send maxMsgCntPerQp-1 messages on all QPs
    for (int i = 0; i < totalQps; i++) {
      for (int j = 0; j < maxMsgCntPerQp - 1; j++) {
        virtualQp->getQpsRef().at(i).enquePhysicalSendWrStatus(0, 0);
      }
    }

    // Find available QP should return 0. Then, after mock send 1 more
    // message on QP 0, this QP should no longer be available
    int curQp = 0;
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // Find available QP should return 1. Then, after mock send 1 more
    // message on QP 1, this QP should no longer be available
    curQp = 1;
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // Find available QP should return 2. Then, after mock send 1 more
    // message on QP 2, this QP should no longer be available
    curQp = 2;
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // Find available QP should return 3. Then, after mock send 1 more
    // message on QP 3, this QP should no longer be available
    curQp = 3;
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // Now, all QPs are full, find available QP should return -1
    ASSERT_EQ(virtualQp->findAvailableSendQp(), -1);

    // After clear up on QP 2, QP 2 should be available again. Then, mock send 1
    // more message on this QP again and it should no longer be available
    curQp = 2;
    virtualQp->getQpsRef().at(curQp).dequePhysicalSendWrStatus();
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // After clear up on QP 3, QP 3 should be available again. Then, mock send 1
    // more message on this QP again and it should no longer be available
    curQp = 3;
    virtualQp->getQpsRef().at(curQp).dequePhysicalSendWrStatus();
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // After clear up on QP 0, QP 0 should be available again. Then, mock send 1
    // more message on this QP again and it should no longer be available
    curQp = 0;
    virtualQp->getQpsRef().at(curQp).dequePhysicalSendWrStatus();
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // After clear up on QP 1, QP 1 should be available again. Then, mock send 1
    // more message on this QP again and it should no longer be available
    curQp = 1;
    virtualQp->getQpsRef().at(curQp).dequePhysicalSendWrStatus();
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // Now, all QPs are full, find available QP should return -1
    ASSERT_EQ(virtualQp->findAvailableSendQp(), -1);
  }

  // Test setting 2: No limit on maxMsgCntPerQp and maxMsgSize
  {
    auto virtualQp =
        pd->createVirtualQp(totalQps, &initAttr, &virtualCq, &virtualCq, -1);
    ASSERT_TRUE(virtualQp);
    ASSERT_EQ(virtualQp->getQpsRef().size(), totalQps);
    ASSERT_EQ(virtualQp->getTotalQps(), totalQps);

    // Mock send default maxMsgCntPerQp messages on all QPs
    for (int i = 0; i < totalQps; i++) {
      for (int j = 0; j < maxMsgCntPerQp; j++) {
        virtualQp->getQpsRef().at(i).enquePhysicalSendWrStatus(0, 0);
      }
    }

    // findAvailableSendQp should return 0, 1, 2, 3, in order
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 0);
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 1);
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 2);
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 3);

    // After mock send one more message on all QPs, findAvailableSendQp should
    // return 0, 1, 2, 3, in order again
    for (int i = 0; i < totalQps; i++) {
      virtualQp->getQpsRef().at(i).enquePhysicalSendWrStatus(0, 0);
    }
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 0);
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 1);
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 2);
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 3);
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualQpUpdatePhysicalSendWrFromVirtualSendWr) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqRef().cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqRef().cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  int maxMsgCntPerQp = 100;
  int maxMsgSizeByte = 100;
  auto virtualQp = pd->createVirtualQp(
      totalQps,
      &initAttr,
      &virtualCq,
      &virtualCq,
      maxMsgCntPerQp,
      maxMsgSizeByte);
  ASSERT_TRUE(virtualQp);
  ASSERT_EQ(virtualQp->getQpsRef().size(), totalQps);
  ASSERT_EQ(virtualQp->getTotalQps(), totalQps);

  // init device buffer
  void* devBuf{nullptr};
  size_t devBufSize = 8192;
  CUDA_CHECK(cudaMalloc(&devBuf, devBufSize));

  // register mr
  ibv_access_flags access = static_cast<ibv_access_flags>(
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ);
  auto mr = pd->regMr(devBuf, devBufSize, access);
  ASSERT_TRUE(mr);

  ibv_sge userProvidedSge = {
      .addr = (uint64_t)devBuf,
      .length = static_cast<uint32_t>(devBufSize),
      .lkey = mr->mr()->lkey};
  ibv_send_wr userProvidedSendWr = {
      .wr_id = 0,
      .next = nullptr,
      .sg_list = &userProvidedSge,
      .num_sge = 1,
      .opcode = IBV_WR_SEND,
      .send_flags = IBV_SEND_SIGNALED};
  int expectedMsgCnt =
      (userProvidedSendWr.sg_list->length + maxMsgSizeByte - 1) /
      maxMsgSizeByte;

  // Test setting 1: OP code is IBV_WR_SEND
  {
    VirtualSendWr virtualSendWr(
        userProvidedSendWr, expectedMsgCnt, expectedMsgCnt);

    ibv_sge sendSge{};
    ibv_send_wr sendWr{};
    virtualQp->updatePhysicalSendWrFromVirtualSendWr(
        virtualSendWr, &sendWr, &sendSge);

    // With initial offset 0, the first message addr should be the same as user
    // provided
    ASSERT_EQ(sendWr.sg_list->addr, userProvidedSendWr.sg_list->addr);
    ASSERT_EQ(sendWr.sg_list->length, maxMsgSizeByte);
    ASSERT_EQ(sendWr.sg_list->lkey, userProvidedSendWr.sg_list->lkey);
    ASSERT_EQ(sendWr.opcode, userProvidedSendWr.opcode);
    ASSERT_EQ(sendWr.send_flags, userProvidedSendWr.send_flags);

    // After sending the first message and changing the offset, the second
    // message should be user provided address + offset, with the rest the same
    virtualSendWr.offset += maxMsgSizeByte;
    virtualQp->updatePhysicalSendWrFromVirtualSendWr(
        virtualSendWr, &sendWr, &sendSge);
    ASSERT_EQ(
        sendWr.sg_list->addr,
        userProvidedSendWr.sg_list->addr + maxMsgSizeByte);
    ASSERT_EQ(sendWr.sg_list->length, maxMsgSizeByte);
    ASSERT_EQ(sendWr.sg_list->lkey, userProvidedSendWr.sg_list->lkey);
    ASSERT_EQ(sendWr.opcode, userProvidedSendWr.opcode);
    ASSERT_EQ(sendWr.send_flags, userProvidedSendWr.send_flags);
  }

  // Test setting 2: OP code is IBV_WR_RDMA_WRITE
  {
    userProvidedSendWr.opcode = IBV_WR_RDMA_WRITE;
    userProvidedSendWr.wr.rdma.remote_addr =
        0x12345678; // Use a random address to mock the remote address
    userProvidedSendWr.wr.rdma.rkey =
        0x87654321; // Use a random rkey to mock the remote rkey
    VirtualSendWr virtualSendWr(
        userProvidedSendWr, expectedMsgCnt, expectedMsgCnt);

    ibv_sge sendSge{};
    ibv_send_wr sendWr{};
    virtualQp->updatePhysicalSendWrFromVirtualSendWr(
        virtualSendWr, &sendWr, &sendSge);

    // With initial offset 0, the first message addr should be the same as user
    // provided
    ASSERT_EQ(sendWr.sg_list->addr, userProvidedSendWr.sg_list->addr);
    ASSERT_EQ(sendWr.sg_list->length, maxMsgSizeByte);
    ASSERT_EQ(sendWr.sg_list->lkey, userProvidedSendWr.sg_list->lkey);
    ASSERT_EQ(sendWr.opcode, userProvidedSendWr.opcode);
    ASSERT_EQ(sendWr.send_flags, userProvidedSendWr.send_flags);
    ASSERT_EQ(
        sendWr.wr.rdma.remote_addr, userProvidedSendWr.wr.rdma.remote_addr);
    ASSERT_EQ(sendWr.wr.rdma.rkey, userProvidedSendWr.wr.rdma.rkey);

    // After sending the first message and changing the offset, the second
    // message should be user provided address + offset, with the rest the same
    virtualSendWr.offset += maxMsgSizeByte;
    virtualQp->updatePhysicalSendWrFromVirtualSendWr(
        virtualSendWr, &sendWr, &sendSge);
    ASSERT_EQ(
        sendWr.sg_list->addr,
        userProvidedSendWr.sg_list->addr + maxMsgSizeByte);
    ASSERT_EQ(sendWr.sg_list->length, maxMsgSizeByte);
    ASSERT_EQ(sendWr.sg_list->lkey, userProvidedSendWr.sg_list->lkey);
    ASSERT_EQ(sendWr.opcode, userProvidedSendWr.opcode);
    ASSERT_EQ(sendWr.send_flags, userProvidedSendWr.send_flags);
    ASSERT_EQ(
        sendWr.wr.rdma.remote_addr,
        userProvidedSendWr.wr.rdma.remote_addr + maxMsgSizeByte);
    ASSERT_EQ(sendWr.wr.rdma.rkey, userProvidedSendWr.wr.rdma.rkey);
  }

  // Test setting 3: OP code is IBV_WR_RDMA_WRITE_WITH_IMM
  {
    userProvidedSendWr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    userProvidedSendWr.wr.rdma.remote_addr =
        0x12345678; // Use a random address to mock the remote address
    userProvidedSendWr.wr.rdma.rkey =
        0x87654321; // Use a random rkey to mock the remote rkey
    VirtualSendWr virtualSendWr(
        userProvidedSendWr, expectedMsgCnt, expectedMsgCnt);

    ibv_sge sendSge{};
    ibv_send_wr sendWr{};
    virtualQp->updatePhysicalSendWrFromVirtualSendWr(
        virtualSendWr, &sendWr, &sendSge);

    // With initial offset 0, the first message addr should be the same as user
    // provided
    ASSERT_EQ(sendWr.sg_list->addr, userProvidedSendWr.sg_list->addr);
    ASSERT_EQ(sendWr.sg_list->length, maxMsgSizeByte);
    ASSERT_EQ(sendWr.sg_list->lkey, userProvidedSendWr.sg_list->lkey);
    ASSERT_EQ(
        sendWr.opcode,
        IBV_WR_RDMA_WRITE); // For IBV_WR_RDMA_WRITE_WITH_IMM, IMM will
                            // only follow after completing send all
                            // other messages
    ASSERT_EQ(
        sendWr.send_flags,
        IBV_SEND_SIGNALED); // For IBV_WR_RDMA_WRITE_WITH_IMM, all messages
                            // before IMM must be signaled so Ibverbx can track
                            // the completion and send the final IMM message
                            // after
    ASSERT_EQ(
        sendWr.wr.rdma.remote_addr, userProvidedSendWr.wr.rdma.remote_addr);
    ASSERT_EQ(sendWr.wr.rdma.rkey, userProvidedSendWr.wr.rdma.rkey);

    // After sending the first message and changing the offset, the second
    // message should be user provided address + offset, with the rest the same
    virtualSendWr.offset += maxMsgSizeByte;
    virtualQp->updatePhysicalSendWrFromVirtualSendWr(
        virtualSendWr, &sendWr, &sendSge);
    ASSERT_EQ(
        sendWr.sg_list->addr,
        userProvidedSendWr.sg_list->addr + maxMsgSizeByte);
    ASSERT_EQ(sendWr.sg_list->length, maxMsgSizeByte);
    ASSERT_EQ(sendWr.sg_list->lkey, userProvidedSendWr.sg_list->lkey);
    ASSERT_EQ(sendWr.opcode, IBV_WR_RDMA_WRITE);
    ASSERT_EQ(sendWr.send_flags, IBV_SEND_SIGNALED);
    ASSERT_EQ(
        sendWr.wr.rdma.remote_addr,
        userProvidedSendWr.wr.rdma.remote_addr + maxMsgSizeByte);
    ASSERT_EQ(sendWr.wr.rdma.rkey, userProvidedSendWr.wr.rdma.rkey);
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualQpBusinessCard) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqRef().cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqRef().cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  auto virtualQp =
      pd->createVirtualQp(totalQps, &initAttr, &virtualCq, &virtualCq);
  ASSERT_TRUE(virtualQp);
  ASSERT_EQ(virtualQp->getQpsRef().size(), totalQps);
  ASSERT_EQ(virtualQp->getTotalQps(), totalQps);

  auto virtualQpBusinessCard = virtualQp->getVirtualQpBusinessCard();
  ASSERT_EQ(virtualQpBusinessCard.qpNums_.size(), totalQps);
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        virtualQpBusinessCard.qpNums_.at(i),
        virtualQp->getQpsRef().at(i).qp()->qp_num);
  }
  ASSERT_EQ(
      virtualQpBusinessCard.notifyQpNum_,
      virtualQp->getNotifyQpRef().qp()->qp_num);

  // move constructor
  auto virtualQpBusinessCard1 = std::move(virtualQpBusinessCard);
  ASSERT_TRUE(virtualQpBusinessCard.qpNums_.empty());
  ASSERT_EQ(virtualQpBusinessCard1.qpNums_.size(), totalQps);
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        virtualQpBusinessCard1.qpNums_.at(i),
        virtualQp->getQpsRef().at(i).qp()->qp_num);
  }
  ASSERT_EQ(
      virtualQpBusinessCard1.notifyQpNum_,
      virtualQp->getNotifyQpRef().qp()->qp_num);

  auto virtualQpBusinessCard2(std::move(virtualQpBusinessCard1));
  ASSERT_TRUE(virtualQpBusinessCard1.qpNums_.empty());
  ASSERT_EQ(virtualQpBusinessCard2.qpNums_.size(), totalQps);
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        virtualQpBusinessCard2.qpNums_.at(i),
        virtualQp->getQpsRef().at(i).qp()->qp_num);
  }
  ASSERT_EQ(
      virtualQpBusinessCard2.notifyQpNum_,
      virtualQp->getNotifyQpRef().qp()->qp_num);

  // Copy constructor
  IbvVirtualQpBusinessCard virtualQpBusinessCardCopy1(virtualQpBusinessCard2);
  ASSERT_EQ(
      virtualQpBusinessCardCopy1.qpNums_.size(),
      virtualQpBusinessCard2.qpNums_.size());
  ASSERT_EQ(
      virtualQpBusinessCardCopy1.qpNums_.size(),
      virtualQpBusinessCard2.qpNums_.size());
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        virtualQpBusinessCardCopy1.qpNums_.at(i),
        virtualQpBusinessCard2.qpNums_.at(i));
  }
  ASSERT_EQ(
      virtualQpBusinessCardCopy1.notifyQpNum_,
      virtualQpBusinessCard2.notifyQpNum_);

  IbvVirtualQpBusinessCard virtualQpBusinessCardCopy2;
  virtualQpBusinessCardCopy2 = virtualQpBusinessCard2;
  ASSERT_EQ(
      virtualQpBusinessCardCopy2.qpNums_.size(),
      virtualQpBusinessCard2.qpNums_.size());
  ASSERT_EQ(
      virtualQpBusinessCardCopy2.qpNums_.size(),
      virtualQpBusinessCard2.qpNums_.size());
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        virtualQpBusinessCardCopy2.qpNums_.at(i),
        virtualQpBusinessCard2.qpNums_.at(i));
  }
  ASSERT_EQ(
      virtualQpBusinessCardCopy2.notifyQpNum_,
      virtualQpBusinessCard2.notifyQpNum_);
}

TEST_F(IbverbxTestFixture, IbvVirtualQpBusinessCardSerializeAndDeserialize) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqRef().cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqRef().cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  auto virtualQp =
      pd->createVirtualQp(totalQps, &initAttr, &virtualCq, &virtualCq);
  ASSERT_TRUE(virtualQp);
  ASSERT_EQ(virtualQp->getQpsRef().size(), totalQps);
  ASSERT_EQ(virtualQp->getTotalQps(), totalQps);

  auto virtualQpBusinessCard = virtualQp->getVirtualQpBusinessCard();
  ASSERT_EQ(virtualQpBusinessCard.qpNums_.size(), totalQps);
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        virtualQpBusinessCard.qpNums_.at(i),
        virtualQp->getQpsRef().at(i).qp()->qp_num);
  }
  ASSERT_EQ(
      virtualQpBusinessCard.notifyQpNum_,
      virtualQp->getNotifyQpRef().qp()->qp_num);

  // Serialize and deserialize
  auto serializedVirtualQpBusinessCard = virtualQpBusinessCard.serialize();
  auto maybeDeserializedVirtualQpBusinessCard =
      IbvVirtualQpBusinessCard::deserialize(serializedVirtualQpBusinessCard);
  ASSERT_TRUE(maybeDeserializedVirtualQpBusinessCard);
  auto& deserializedVirtualQpBusinessCard =
      *maybeDeserializedVirtualQpBusinessCard;
  ASSERT_EQ(
      deserializedVirtualQpBusinessCard.qpNums_.size(),
      virtualQpBusinessCard.qpNums_.size());
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        deserializedVirtualQpBusinessCard.qpNums_.at(i),
        virtualQp->getQpsRef().at(i).qp()->qp_num);
  }
  ASSERT_EQ(
      deserializedVirtualQpBusinessCard.notifyQpNum_,
      virtualQp->getNotifyQpRef().qp()->qp_num);

  // Deserialize fail case
  std::string emptyStr;
  std::string& jsonStr = emptyStr;
  auto maybeDeserializedVirtualQpBusinessCardError =
      IbvVirtualQpBusinessCard::deserialize(jsonStr);
  ASSERT_FALSE(maybeDeserializedVirtualQpBusinessCardError);
}

TEST_F(IbverbxTestFixture, Coordinator) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqRef().cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqRef().cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  auto maybeVirtualQp =
      pd->createVirtualQp(totalQps, &initAttr, &virtualCq, &virtualCq);
  ASSERT_TRUE(maybeVirtualQp);
  auto virtualQp = std::move(*maybeVirtualQp);
  ASSERT_EQ(virtualQp.getQpsRef().size(), totalQps);
  ASSERT_EQ(virtualQp.getTotalQps(), totalQps);

  // Test coordinator mappings are correctly established
  const auto& coordinator = device.coordinator_;
  uint32_t virtualQpNum = virtualQp.getVirtualQpNum();

  // 1. Test virtualQpNumToVirtualSendCq_ mapping
  const auto& sendCqMap = coordinator.getVirtualSendCqMap();
  ASSERT_EQ(sendCqMap.size(), 1);
  auto sendCqIt = sendCqMap.find(virtualQpNum);
  ASSERT_NE(sendCqIt, sendCqMap.end());
  ASSERT_EQ(sendCqIt->second, &virtualCq);

  // 2. Test virtualQpNumToVirtualRecvCq_ mapping
  const auto& recvCqMap = coordinator.getVirtualRecvCqMap();
  ASSERT_EQ(recvCqMap.size(), 1);
  auto recvCqIt = recvCqMap.find(virtualQpNum);
  ASSERT_NE(recvCqIt, recvCqMap.end());
  ASSERT_EQ(recvCqIt->second, &virtualCq);

  // 3. Test physicalQpNumToVirtualQp_ mapping
  const auto& physicalQpMap = coordinator.getPhysicalQpMap();
  ASSERT_EQ(
      physicalQpMap.size(), totalQps + 1); // totalQos + 1 to consider notifyQp
  for (const auto& physicalQp : virtualQp.getQpsRef()) {
    uint32_t physicalQpNum = physicalQp.qp()->qp_num;
    auto virtualQpIt = physicalQpMap.find(physicalQpNum);
    ASSERT_NE(virtualQpIt, physicalQpMap.end());
    ASSERT_EQ(virtualQpIt->second, &virtualQp);
  }
  uint32_t notifyQpNum = virtualQp.getNotifyQpRef().qp()->qp_num;
  auto notifyQpIt = physicalQpMap.find(notifyQpNum);
  ASSERT_NE(notifyQpIt, physicalQpMap.end());
  ASSERT_EQ(notifyQpIt->second, &virtualQp);

  // 4. Test that all physical QP numbers are unique and properly mapped
  std::set<uint32_t> physicalQpNums;
  for (const auto& physicalQp : virtualQp.getQpsRef()) {
    uint32_t physicalQpNum = physicalQp.qp()->qp_num;
    ASSERT_TRUE(
        physicalQpNums.insert(physicalQpNum).second); // Should be unique
    ASSERT_EQ(coordinator.getVirtualQp(physicalQpNum), &virtualQp);
  }
  ASSERT_TRUE(physicalQpNums.insert(notifyQpNum).second); // Should be unique
  ASSERT_EQ(coordinator.getVirtualQp(notifyQpNum), &virtualQp);
}
} // namespace ibverbx

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
