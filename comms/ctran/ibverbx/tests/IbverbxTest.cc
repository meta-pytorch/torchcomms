// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <random>

#define IBVERBX_TEST_FRIENDS                                                  \
  friend class IbverbxTestFixture;                                            \
  FRIEND_TEST(                                                                \
      IbverbxTestFixture, IbvVirtualQpUpdatePhysicalSendWrFromVirtualSendWr); \
  FRIEND_TEST(IbverbxTestFixture, IbvVirtualCqRegisterPhysicalQp);            \
  FRIEND_TEST(                                                                \
      IbverbxTestFixture, IbvVirtualCqRegisterPhysicalQpMoveSemantics);       \
  FRIEND_TEST(                                                                \
      IbverbxTestFixture, IbvVirtualQpRegisterUnregisterWithVirtualCq);       \
  FRIEND_TEST(IbverbxTestFixture, IbvVirtualQpUpdateWrState);                 \
  FRIEND_TEST(IbverbxTestFixture, IbvVirtualQpBuildVirtualWc);                \
  FRIEND_TEST(IbverbxTestFixture, IbvVirtualQpIsSendOpcode);                  \
  FRIEND_TEST(IbverbxTestFixture, IbvVirtualQpHasQpCapacity);                 \
  FRIEND_TEST(IbverbxTestFixture, IbvVirtualQpBuildPhysicalSendWr);
#include "comms/ctran/ibverbx/IbvVirtualWr.h"
#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/utils/checks.h"

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

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

  // find active port
  auto activePort = device.findActivePort(
      {IBV_LINK_LAYER_INFINIBAND, IBV_LINK_LAYER_ETHERNET});
  ASSERT_TRUE(activePort);
  EXPECT_GT(activePort.value(), 0);
}

TEST_F(IbverbxTestFixture, IbvDeviceMultiThreadUniqueDeviceId) {
  constexpr int kNumThreads = 4;
  constexpr int kDevicesPerThread = 10;
  constexpr int kTotalDevices = kNumThreads * kDevicesPerThread;

  std::set<int32_t> deviceIds;
  std::mutex numsMutex;

  auto createDevices = [&]() {
    std::vector<int32_t> localDeviceIds;
    localDeviceIds.reserve(kDevicesPerThread);

    for (int i = 0; i < kDevicesPerThread; i++) {
      auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
      ASSERT_TRUE(devices);
      for (auto& device : *devices) {
        localDeviceIds.push_back(device.getDeviceId());
      }
    }

    std::lock_guard<std::mutex> lock(numsMutex);
    deviceIds.insert(localDeviceIds.begin(), localDeviceIds.end());
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; i++) {
    threads.emplace_back(createDevices);
  }

  for (auto& t : threads) {
    t.join();
  }

  ASSERT_EQ(deviceIds.size(), kTotalDevices)
      << "All device IDs should be distinct";
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
  ASSERT_NE(maybeVirtualCq->getPhysicalCqsRef().at(0).cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  auto virtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
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

TEST_F(IbverbxTestFixture, IbvVirtualQpMultiThreadUniqueQpNum) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  constexpr int kNumThreads = 4;
  constexpr int kVirtualQpsPerThread = 10;
  constexpr int kTotalVirtualQps = kNumThreads * kVirtualQpsPerThread;

  std::set<uint32_t> virtualQpNums;
  std::set<uint32_t> virtualCqNums;
  std::mutex numsMutex;

  auto createVirtualQps = [&]() {
    std::vector<uint32_t> localQpNums;
    std::vector<uint32_t> localCqNums;
    localQpNums.reserve(kVirtualQpsPerThread);
    localCqNums.reserve(kVirtualQpsPerThread);

    for (int i = 0; i < kVirtualQpsPerThread; i++) {
      int cqe = 100;
      auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
      ASSERT_TRUE(maybeVirtualCq);
      auto virtualCq = std::move(*maybeVirtualCq);

      auto initAttr =
          makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
      auto pd = device.allocPd();
      ASSERT_TRUE(pd);

      int totalQps = 4;
      auto virtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
      ASSERT_TRUE(virtualQp);

      localQpNums.push_back(virtualQp->getVirtualQpNum());
      localCqNums.push_back(virtualCq.getVirtualCqNum());
    }

    std::lock_guard<std::mutex> lock(numsMutex);
    virtualQpNums.insert(localQpNums.begin(), localQpNums.end());
    virtualCqNums.insert(localCqNums.begin(), localCqNums.end());
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; i++) {
    threads.emplace_back(createVirtualQps);
  }

  for (auto& t : threads) {
    t.join();
  }

  ASSERT_EQ(virtualQpNums.size(), kTotalVirtualQps)
      << "All virtual QP numbers should be distinct";
  ASSERT_EQ(virtualCqNums.size(), kTotalVirtualQps)
      << "All virtual CQ numbers should be distinct";
}

TEST_F(IbverbxTestFixture, IbvVirtualQpFindAvailableSendQp) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqsRef().at(0).cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 4;
  int maxMsgCntPerQp = 100;

  // Test setting 1: default maxMsgCntPerQp (100)
  {
    auto virtualQp =
        pd->createVirtualQp(totalQps, &initAttr, &virtualCq, maxMsgCntPerQp);
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
    auto virtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq, -1);
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
  ASSERT_NE(maybeVirtualCq->getPhysicalCqsRef().at(0).cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  int maxMsgCntPerQp = 100;
  int maxMsgSizeByte = 100;
  auto virtualQp = pd->createVirtualQp(
      totalQps, &initAttr, &virtualCq, maxMsgCntPerQp, maxMsgSizeByte);
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
  bool sendExtraNotifyImm = false;

  // Test setting 1: OP code is IBV_WR_SEND
  {
    VirtualSendWr virtualSendWr(
        userProvidedSendWr, expectedMsgCnt, expectedMsgCnt, sendExtraNotifyImm);

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
        userProvidedSendWr, expectedMsgCnt, expectedMsgCnt, sendExtraNotifyImm);

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
        userProvidedSendWr, expectedMsgCnt, expectedMsgCnt, sendExtraNotifyImm);

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
  ASSERT_NE(maybeVirtualCq->getPhysicalCqsRef().at(0).cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  auto virtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
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
  ASSERT_NE(maybeVirtualCq->getPhysicalCqsRef().at(0).cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  auto virtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
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
  ASSERT_NE(maybeVirtualCq->getPhysicalCqsRef().at(0).cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  auto maybeVirtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
  ASSERT_TRUE(maybeVirtualQp);
  auto virtualQp = std::move(*maybeVirtualQp);
  ASSERT_EQ(virtualQp.getQpsRef().size(), totalQps);
  ASSERT_EQ(virtualQp.getTotalQps(), totalQps);

  // Test coordinator mappings are correctly established
  auto coordinator = Coordinator::getCoordinator();
  ASSERT_NE(coordinator, nullptr);
  uint32_t virtualQpNum = virtualQp.getVirtualQpNum();
  uint32_t virtualCqNum = virtualCq.getVirtualCqNum();

  // 1. Test virtualQpNumToVirtualSendCqNum_ mapping
  const auto& virtualQpToSendCqMap =
      coordinator->getVirtualQpToVirtualSendCqMap();
  ASSERT_EQ(virtualQpToSendCqMap.size(), 1);
  auto sendCqIt = virtualQpToSendCqMap.find(virtualQpNum);
  ASSERT_NE(sendCqIt, virtualQpToSendCqMap.end());
  ASSERT_EQ(sendCqIt->second, virtualCqNum);

  // 2. Test virtualQpNumToVirtualRecvCqNum_ mapping
  const auto& virtualQpToRecvCqMap =
      coordinator->getVirtualQpToVirtualRecvCqMap();
  ASSERT_EQ(virtualQpToRecvCqMap.size(), 1);
  auto recvCqIt = virtualQpToRecvCqMap.find(virtualQpNum);
  ASSERT_NE(recvCqIt, virtualQpToRecvCqMap.end());
  ASSERT_EQ(recvCqIt->second, virtualCqNum);

  // 3. Test qpIdToVirtualQpNum_ mapping
  const auto& qpIdToVirtualQpMap = coordinator->getQpIdToVirtualQpMap();
  ASSERT_EQ(
      qpIdToVirtualQpMap.size(),
      totalQps + 1); // totalQps + 1 to consider notifyQp
  for (const auto& physicalQp : virtualQp.getQpsRef()) {
    uint32_t physicalQpNum = physicalQp.qp()->qp_num;
    int32_t deviceId = physicalQp.getDeviceId();
    QpId key{.deviceId = deviceId, .qpNum = physicalQpNum};
    auto virtualQpNumIt = qpIdToVirtualQpMap.find(key);
    ASSERT_NE(virtualQpNumIt, qpIdToVirtualQpMap.end());
    ASSERT_EQ(virtualQpNumIt->second, virtualQpNum);
  }
  uint32_t notifyQpNum = virtualQp.getNotifyQpRef().qp()->qp_num;
  int32_t notifyDeviceId = virtualQp.getNotifyQpRef().getDeviceId();
  QpId notifyKey{.deviceId = notifyDeviceId, .qpNum = notifyQpNum};
  auto notifyQpIt = qpIdToVirtualQpMap.find(notifyKey);
  ASSERT_NE(notifyQpIt, qpIdToVirtualQpMap.end());
  ASSERT_EQ(notifyQpIt->second, virtualQpNum);

  // 4. Test that all physical QP numbers are unique and properly mapped
  std::set<uint32_t> physicalQpNums;
  for (const auto& physicalQp : virtualQp.getQpsRef()) {
    uint32_t physicalQpNum = physicalQp.qp()->qp_num;
    uint32_t deviceId = physicalQp.getDeviceId();
    ASSERT_TRUE(
        physicalQpNums.insert(physicalQpNum).second); // Should be unique
    ASSERT_EQ(
        coordinator->getVirtualQpByPhysicalQpNumAndDeviceId(
            physicalQpNum, deviceId),
        &virtualQp);
  }
  ASSERT_TRUE(physicalQpNums.insert(notifyQpNum).second); // Should be unique
  ASSERT_EQ(
      coordinator->getVirtualQpByPhysicalQpNumAndDeviceId(
          notifyQpNum, notifyDeviceId),
      &virtualQp);
}

TEST_F(IbverbxTestFixture, CoordinatorRegisterUnregisterUpdateApis) {
  // Common setup for all test scenarios
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);
  int cqe = 100;
  auto coordinator = Coordinator::getCoordinator();
  ASSERT_NE(coordinator, nullptr);

  // Test 1: Test that destructors properly unregister objects
  {
    uint32_t virtualQpNum1;
    uint32_t virtualQpNum2;
    uint32_t virtualCqNum1;
    uint32_t virtualCqNum2;

    // Create objects within a nested scope.
    // Each object is automatically constructed and registered upon creation.
    // When the scope ends, the object's destructor is called, which
    // automatically unregisters and destroys the object.
    {
      auto maybeVirtualCq1 = device.createVirtualCq(cqe, nullptr, nullptr, 0);
      ASSERT_TRUE(maybeVirtualCq1);
      auto virtualCq1 = std::move(*maybeVirtualCq1);
      virtualCqNum1 = virtualCq1.getVirtualCqNum();

      auto maybeVirtualCq2 = device.createVirtualCq(cqe, nullptr, nullptr, 0);
      ASSERT_TRUE(maybeVirtualCq2);
      auto virtualCq2 = std::move(*maybeVirtualCq2);
      virtualCqNum2 = virtualCq2.getVirtualCqNum();

      auto initAttr =
          makeIbvQpInitAttr(virtualCq1.getPhysicalCqsRef().at(0).cq());
      auto pd = device.allocPd();
      ASSERT_TRUE(pd);

      int totalQps = 4;
      auto maybeVirtualQp1 =
          pd->createVirtualQp(totalQps, &initAttr, &virtualCq1);
      ASSERT_TRUE(maybeVirtualQp1);
      auto virtualQp1 = std::move(*maybeVirtualQp1);
      virtualQpNum1 = virtualQp1.getVirtualQpNum();

      auto maybeVirtualQp2 =
          pd->createVirtualQp(totalQps, &initAttr, &virtualCq2);
      ASSERT_TRUE(maybeVirtualQp2);
      auto virtualQp2 = std::move(*maybeVirtualQp2);
      virtualQpNum2 = virtualQp2.getVirtualQpNum();

      // Verify registration while objects are alive
      const auto& sendCqMap = coordinator->getVirtualQpToVirtualSendCqMap();
      const auto& recvCqMap = coordinator->getVirtualQpToVirtualRecvCqMap();
      const auto& virtualQpMap = coordinator->getVirtualQpMap();
      const auto& virtualCqMap = coordinator->getVirtualCqMap();

      ASSERT_GE(sendCqMap.size(), 2);
      ASSERT_GE(recvCqMap.size(), 2);
      ASSERT_EQ(sendCqMap.at(virtualQpNum1), virtualCqNum1);
      ASSERT_EQ(recvCqMap.at(virtualQpNum1), virtualCqNum1);
      ASSERT_EQ(sendCqMap.at(virtualQpNum2), virtualCqNum2);
      ASSERT_EQ(recvCqMap.at(virtualQpNum2), virtualCqNum2);
      ASSERT_EQ(virtualQpMap.at(virtualQpNum1), &virtualQp1);
      ASSERT_EQ(virtualQpMap.at(virtualQpNum2), &virtualQp2);
      ASSERT_EQ(virtualCqMap.at(virtualCqNum1), &virtualCq1);
      ASSERT_EQ(virtualCqMap.at(virtualCqNum2), &virtualCq2);
    } // Objects destroyed here - destructors should call unregister

    // Verify that all objects were unregistered by their destructors
    const auto& sendCqMapAfter = coordinator->getVirtualQpToVirtualSendCqMap();
    const auto& recvCqMapAfter = coordinator->getVirtualQpToVirtualRecvCqMap();
    const auto& virtualQpMapAfter = coordinator->getVirtualQpMap();
    const auto& virtualCqMapAfter = coordinator->getVirtualCqMap();

    // All entries should be removed
    ASSERT_EQ(sendCqMapAfter.count(virtualQpNum1), 0);
    ASSERT_EQ(recvCqMapAfter.count(virtualQpNum1), 0);
    ASSERT_EQ(sendCqMapAfter.count(virtualQpNum2), 0);
    ASSERT_EQ(recvCqMapAfter.count(virtualQpNum2), 0);
    ASSERT_EQ(virtualQpMapAfter.count(virtualQpNum1), 0);
    ASSERT_EQ(virtualQpMapAfter.count(virtualQpNum2), 0);
    ASSERT_EQ(virtualCqMapAfter.count(virtualCqNum1), 0);
    ASSERT_EQ(virtualCqMapAfter.count(virtualCqNum2), 0);
  }

  // Test 2: Pointer-based Unregister - Test that unregister with wrong pointer
  // does nothing
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    int totalQps = 4;
    auto maybeVirtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    uint32_t virtualQpNum = virtualQp.getVirtualQpNum();
    uint32_t virtualCqNum = virtualCq.getVirtualCqNum();

    // Verify registration
    const auto& virtualQpMap = coordinator->getVirtualQpMap();
    const auto& virtualCqMap = coordinator->getVirtualCqMap();
    ASSERT_EQ(virtualQpMap.at(virtualQpNum), &virtualQp);
    ASSERT_EQ(virtualCqMap.at(virtualCqNum), &virtualCq);

    // Try to unregister with a different pointer - should do nothing
    IbvVirtualQp* wrongQpPtr = reinterpret_cast<IbvVirtualQp*>(0x1234);
    IbvVirtualCq* wrongCqPtr = reinterpret_cast<IbvVirtualCq*>(0x5678);

    coordinator->unregisterVirtualQp(virtualQpNum, wrongQpPtr);
    coordinator->unregisterVirtualCq(virtualCqNum, wrongCqPtr);

    // Verify nothing was unregistered (pointers didn't match)
    const auto& virtualQpMapAfter = coordinator->getVirtualQpMap();
    const auto& virtualCqMapAfter = coordinator->getVirtualCqMap();
    ASSERT_EQ(virtualQpMapAfter.at(virtualQpNum), &virtualQp);
    ASSERT_EQ(virtualCqMapAfter.at(virtualCqNum), &virtualCq);

    // When objects go out of scope, destructors will properly unregister
    // with the correct pointers and clean up resources
  }

  // Test 3: Test move constructors for both IbvVirtualCq and IbvVirtualQp
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
    auto virtualQp1 = std::move(*maybeVirtualQp);

    uint32_t virtualQpNum = virtualQp1.getVirtualQpNum();
    uint32_t virtualCqNum = virtualCq1.getVirtualCqNum();

    // Verify initial pointers are correctly registered
    ASSERT_EQ(coordinator->getVirtualQpById(virtualQpNum), &virtualQp1);
    ASSERT_EQ(coordinator->getVirtualCqById(virtualCqNum), &virtualCq1);

    // The move constructor automatically calls updateVirtualCqPointer,
    // and we verify CQ pointer was updated in coordinator
    IbvVirtualCq virtualCq2(std::move(virtualCq1));
    ASSERT_EQ(coordinator->getVirtualCqById(virtualCqNum), &virtualCq2);

    // Try to unregister with old CQ pointer (should fail - pointer doesn't
    // match)
    coordinator->unregisterVirtualCq(virtualCqNum, &virtualCq1);

    // Verify CQ wasn't unregistered (pointer didn't match)
    const auto& virtualCqMapAfter = coordinator->getVirtualCqMap();
    ASSERT_EQ(virtualCqMapAfter.count(virtualCqNum), 1);
    ASSERT_EQ(virtualCqMapAfter.at(virtualCqNum), &virtualCq2);

    // The move constructor automatically calls updateVirtualQpPointer,
    // and we verify QP pointer was updated in coordinator
    IbvVirtualQp virtualQp2(std::move(virtualQp1));
    ASSERT_EQ(coordinator->getVirtualQpById(virtualQpNum), &virtualQp2);

    // Try to unregister with old QP pointer (should fail - pointer doesn't
    // match)
    coordinator->unregisterVirtualQp(virtualQpNum, &virtualQp1);

    // Verify QP wasn't unregistered (pointer didn't match)
    const auto& virtualQpMapAfter = coordinator->getVirtualQpMap();
    ASSERT_EQ(virtualQpMapAfter.count(virtualQpNum), 1);
    ASSERT_EQ(virtualQpMapAfter.at(virtualQpNum), &virtualQp2);
  }
}

TEST_F(IbverbxTestFixture, DqplbSeqTrackerGetSendImm) {
  DqplbSeqTracker tracker;

  // Test 1: Basic sequence number generation
  {
    uint32_t imm1 = tracker.getSendImm(2); // remainingMsgCnt = 2, not last msg
    ASSERT_EQ(imm1 & kSeqNumMask, 0); // First sequence number should be 0
    ASSERT_EQ(imm1 & (1U << kNotifyBit), 0); // Notify bit should not be set

    uint32_t imm2 = tracker.getSendImm(2); // remainingMsgCnt = 2, not last msg
    ASSERT_EQ(imm2 & kSeqNumMask, 1); // Second sequence number should be 1
    ASSERT_EQ(imm2 & (1U << kNotifyBit), 0); // Notify bit should not be set

    uint32_t imm3 = tracker.getSendImm(2); // remainingMsgCnt = 2, not last msg
    ASSERT_EQ(imm3 & kSeqNumMask, 2); // Third sequence number should be 2
    ASSERT_EQ(imm3 & (1U << kNotifyBit), 0); // Notify bit should not be set
  }

  // Test 2: Notify bit is set when remainingMsgCnt == 1
  {
    uint32_t imm = tracker.getSendImm(1); // remainingMsgCnt = 1, last msg
    ASSERT_EQ(imm & kSeqNumMask, 3); // Fourth sequence number should be 3
    ASSERT_EQ(
        imm & (1U << kNotifyBit),
        (1U << kNotifyBit)); // Notify bit should be set
  }

  // Test 3: Sequence number wraps around after reaching kSeqNumMask
  {
    // Continue incrementing to test wrap-around
    // Current sequence is 4 (from previous tests: 0, 1, 2, 3)
    // kSeqNumMask = 0xFFFFFF (24 bits), so we need to reach 0xFFFFFF and wrap
    // to 0

    // Fast forward to near the end of sequence space
    for (uint32_t i = 4; i < kSeqNumMask - 1; i++) {
      uint32_t imm = tracker.getSendImm(2);
      ASSERT_EQ(imm & kSeqNumMask, i);
      ASSERT_EQ(imm & (1U << kNotifyBit), 0);
    }

    // Test wrap-around: sequence should go from kSeqNumMask - 1 to 0
    uint32_t immBeforeWrap = tracker.getSendImm(2);
    ASSERT_EQ(immBeforeWrap & kSeqNumMask, kSeqNumMask - 1);
    ASSERT_EQ(immBeforeWrap & (1U << kNotifyBit), 0);

    uint32_t immAfterWrap = tracker.getSendImm(2);
    ASSERT_EQ(immAfterWrap & kSeqNumMask, 0); // Should wrap to 0
    ASSERT_EQ(immAfterWrap & (1U << kNotifyBit), 0);

    // Verify it continues from 0
    uint32_t immAfterWrap2 = tracker.getSendImm(1);
    ASSERT_EQ(immAfterWrap2 & kSeqNumMask, 1);
    ASSERT_EQ(
        immAfterWrap2 & (1U << kNotifyBit),
        (1U << kNotifyBit)); // Last message has notify bit
  }
}

TEST_F(IbverbxTestFixture, DqplbSeqTrackerProcessReceivedImm) {
  // Test 1: Process messages in order without notify bit
  {
    DqplbSeqTracker tracker;

    uint32_t imm0 = 0 % kSeqNumMask; // Seq 0, no notify
    uint32_t imm1 = 1 % kSeqNumMask; // Seq 1, no notify
    uint32_t imm2 = 2 % kSeqNumMask; // Seq 2, no notify

    int notify0 = tracker.processReceivedImm(imm0);
    ASSERT_EQ(notify0, 0); // No notify bit, so notify count is 0

    int notify1 = tracker.processReceivedImm(imm1);
    ASSERT_EQ(notify1, 0); // No notify bit, so notify count is 0

    int notify2 = tracker.processReceivedImm(imm2);
    ASSERT_EQ(notify2, 0); // No notify bit, so notify count is 0
  }

  // Test 2: Process messages in order with notify bit
  {
    DqplbSeqTracker tracker2;

    uint32_t imm0 = (0 % kSeqNumMask) | (1U << kNotifyBit); // Seq 0 with notify
    uint32_t imm1 = (1 % kSeqNumMask); // Seq 1, no notify
    uint32_t imm2 = (2 % kSeqNumMask) | (1U << kNotifyBit); // Seq 2 with notify

    int notify0 = tracker2.processReceivedImm(imm0);
    ASSERT_EQ(notify0, 1); // Notify bit set, expect notify count 1

    int notify1 = tracker2.processReceivedImm(imm1);
    ASSERT_EQ(notify1, 0); // No notify bit, so notify count is 0

    int notify2 = tracker2.processReceivedImm(imm2);
    ASSERT_EQ(notify2, 1); // Notify bit set, expect notify count 1
  }

  // Test 3: Process messages out of order - later messages arrive first
  {
    DqplbSeqTracker tracker3;

    // Receive messages out of order: 2, 1, 0
    uint32_t imm2 = (2 % kSeqNumMask) | (1U << kNotifyBit); // Seq 2 with notify
    uint32_t imm1 = (1 % kSeqNumMask); // Seq 1, no notify
    uint32_t imm0 = (0 % kSeqNumMask); // Seq 0, no notify

    // Receive seq 2 first (out of order)
    int notify2 = tracker3.processReceivedImm(imm2);
    ASSERT_EQ(
        notify2,
        0); // Even though notify bit is set, can't process until seq 0, 1
            // arrive

    // Receive seq 1 (still out of order)
    int notify1 = tracker3.processReceivedImm(imm1);
    ASSERT_EQ(notify1, 0); // Still waiting for seq 0

    // Receive seq 0 (now in order)
    int notify0 = tracker3.processReceivedImm(imm0);
    ASSERT_EQ(
        notify0,
        1); // Now all messages are in order, process all: 0 (no notify), 1 (no
            // notify), 2 (notify) = 1 notify total
  }

  // Test 4: Process messages out of order with multiple notify bits
  {
    DqplbSeqTracker tracker4;

    // Receive messages: 3, 1, 0, 2
    uint32_t imm0 = (0 % kSeqNumMask) | (1U << kNotifyBit); // Seq 0 with notify
    uint32_t imm1 = (1 % kSeqNumMask); // Seq 1, no notify
    uint32_t imm2 = (2 % kSeqNumMask) | (1U << kNotifyBit); // Seq 2 with notify
    uint32_t imm3 = (3 % kSeqNumMask) | (1U << kNotifyBit); // Seq 3 with notify

    // Receive seq 3 first (far out of order)
    int notify3 = tracker4.processReceivedImm(imm3);
    ASSERT_EQ(notify3, 0); // Waiting for seq 0

    // Receive seq 1 (out of order)
    int notify1 = tracker4.processReceivedImm(imm1);
    ASSERT_EQ(notify1, 0); // Still waiting for seq 0

    // Receive seq 0 (can process 0 and 1 now)
    int notify0 = tracker4.processReceivedImm(imm0);
    ASSERT_EQ(
        notify0, 1); // Process seq 0 (notify) and seq 1 (no notify) = 1 notify

    // Receive seq 2 (can process 2 and 3 now)
    int notify2 = tracker4.processReceivedImm(imm2);
    ASSERT_EQ(
        notify2, 2); // Process seq 2 (notify) and seq 3 (notify) = 2 notifies
  }

  // Test 5: Process with sequence wrap-around
  {
    DqplbSeqTracker tracker5;

    // Fast forward receiveNext_ to near the end of sequence space
    for (uint32_t i = 0; i < kSeqNumMask - 2; i++) {
      uint32_t imm = (i % kSeqNumMask);
      tracker5.processReceivedImm(imm);
    }

    // Now receiveNext_ should be at kSeqNumMask - 2
    // Test wrap-around: receive messages kSeqNumMask - 2, kSeqNumMask - 1, 0, 1
    uint32_t immBeforeWrap1 =
        ((kSeqNumMask - 2) % kSeqNumMask) | (1U << kNotifyBit);
    uint32_t immBeforeWrap2 = ((kSeqNumMask - 1) % kSeqNumMask);
    uint32_t immAfterWrap1 = (kSeqNumMask % kSeqNumMask);
    uint32_t immAfterWrap2 =
        ((kSeqNumMask + 1) % kSeqNumMask) | (1U << kNotifyBit);

    int notify1 = tracker5.processReceivedImm(immBeforeWrap1);
    ASSERT_EQ(notify1, 1); // Seq kSeqNumMask - 2 with notify = 1

    int notify2 = tracker5.processReceivedImm(immBeforeWrap2);
    ASSERT_EQ(notify2, 0); // Seq kSeqNumMask - 1, no notify = 0

    int notify3 = tracker5.processReceivedImm(immAfterWrap1);
    ASSERT_EQ(notify3, 0); // Seq 0, no notify = 0 (wrap around works)

    int notify4 = tracker5.processReceivedImm(immAfterWrap2);
    ASSERT_EQ(notify4, 1); // Seq 1 with notify = 1
  }

  // Test 6: Process with sequence wrap-around and out-of-order
  {
    DqplbSeqTracker tracker6;

    // Fast forward receiveNext_ to near the end of sequence space
    for (uint32_t i = 0; i < kSeqNumMask - 2; i++) {
      uint32_t imm = (i & kSeqNumMask);
      tracker6.processReceivedImm(imm);
    }

    // Now receiveNext_ should be at kSeqNumMask - 2
    // Test wrap-around: receive messages kSeqNumMask - 2, kSeqNumMask - 1, 0, 1
    uint32_t immBeforeWrap1 =
        ((kSeqNumMask - 2) % kSeqNumMask) | (1U << kNotifyBit);
    uint32_t immBeforeWrap2 = ((kSeqNumMask - 1) % kSeqNumMask);
    uint32_t immAfterWrap1 = (kSeqNumMask % kSeqNumMask);
    uint32_t immAfterWrap2 =
        ((kSeqNumMask + 1) % kSeqNumMask) | (1U << kNotifyBit);

    int notify1 = tracker6.processReceivedImm(immBeforeWrap2);
    ASSERT_EQ(
        notify1,
        0); // Receiver receives immBeforeWrap2, no notification so notify1 is 0

    int notify2 = tracker6.processReceivedImm(immAfterWrap2);
    ASSERT_EQ(
        notify2, 0); // Receiver receives immAfterWrap2. Though notification bit
                     // is set, but received out of order so notify2 is 0

    int notify3 = tracker6.processReceivedImm(immBeforeWrap1);
    ASSERT_EQ(notify3, 1); // Receiver receives immBeforeWrap1, notification bit
                           // is set, and received in order, so notify3 is 1

    int notify4 = tracker6.processReceivedImm(immAfterWrap1);
    ASSERT_EQ(notify4, 1); // Receiver receives immAfterWrap1, notification bit
                           // is not set. We received immAfterWrap2 before with
                           // notification bit set. So notify4 is 1
  }
}

TEST_F(IbverbxTestFixture, IbvCqGetDeviceCq) {
#if defined(__HIP_PLATFORM_AMD__)
  GTEST_SKIP() << "Skipping getDeviceCq test on AMD platform: not supported";
#else
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 256;
  auto cq = device.createCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);
  ASSERT_NE(cq->cq(), nullptr);

  // Test getDeviceCq
  auto maybeDeviceCq = cq->getDeviceCq();
  ASSERT_TRUE(maybeDeviceCq);
  auto& deviceCq = *maybeDeviceCq;

  // Verify device CQ fields are populated
  ASSERT_NE(deviceCq.cq_buf, nullptr);
  ASSERT_GT(deviceCq.ncqes, 0);
  // MLX5 may allocate more CQEs than requested (driver may round up for
  // alignment/performance)
  // TODO: we see driver allocates 512 CQE even when we only ask for 256
  ASSERT_GE(deviceCq.ncqes, cqe);
  ASSERT_NE(deviceCq.cq_dbrec, nullptr);

  XLOGF(
      INFO,
      "Device CQ: cq_buf={}, ncqes={}, cq_dbrec={}",
      deviceCq.cq_buf,
      deviceCq.ncqes,
      (void*)deviceCq.cq_dbrec);
#endif
}

TEST_F(IbverbxTestFixture, IbvQpGetDeviceQp) {
#if defined(__HIP_PLATFORM_AMD__)
  GTEST_SKIP() << "Skipping getDeviceQp test on AMD platform: not supported";
#else
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 256;
  auto cq = device.createCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);
  ASSERT_NE(cq->cq(), nullptr);

  // Get device CQ first
  auto maybeDeviceCq = cq->getDeviceCq();
  ASSERT_TRUE(maybeDeviceCq);
  auto deviceCq = *maybeDeviceCq;

  // Allocate device memory for device_cq
  device_cq* d_cq;
  CUDA_CHECK(cudaMalloc(&d_cq, sizeof(device_cq)));
  CUDA_CHECK(
      cudaMemcpy(d_cq, &deviceCq, sizeof(device_cq), cudaMemcpyHostToDevice));

  // Create QP
  auto initAttr = makeIbvQpInitAttr(cq->cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);
  auto qp = pd->createQp(&initAttr);
  ASSERT_TRUE(qp);
  ASSERT_NE(qp->qp(), nullptr);

  // Test getDeviceQp
  auto maybeDeviceQp = qp->getDeviceQp(d_cq);
  ASSERT_TRUE(maybeDeviceQp);
  auto& deviceQp = *maybeDeviceQp;

  // Verify device QP fields are populated
  ASSERT_EQ(deviceQp.qp_num, qp->qp()->qp_num);
  ASSERT_NE(deviceQp.wq_buf, nullptr);
  ASSERT_GT(deviceQp.nwqes, 0);
  ASSERT_NE(deviceQp.sq_dbrec, nullptr);
  ASSERT_NE(deviceQp.bf_reg, nullptr);
  ASSERT_EQ(deviceQp.cq, d_cq);
  ASSERT_EQ(deviceQp.producer_idx, 0);
  ASSERT_EQ(deviceQp.consumer_idx, 0);

  XLOGF(
      INFO,
      "Device QP: qp_num={}, wq_buf={}, nwqes={}, sq_dbrec={}, bf_reg={}, cq={}",
      deviceQp.qp_num,
      deviceQp.wq_buf,
      deviceQp.nwqes,
      (void*)deviceQp.sq_dbrec,
      (void*)deviceQp.bf_reg,
      (void*)deviceQp.cq);

  // Cleanup
  CUDA_CHECK(cudaFree(d_cq));
#endif
}

TEST_F(IbverbxTestFixture, ActiveVirtualWr) {
  // isComplete
  {
    ActiveVirtualWr wr;
    wr.remainingMsgCnt = 2;
    ASSERT_FALSE(wr.isComplete());
    wr.remainingMsgCnt = 1;
    ASSERT_FALSE(wr.isComplete());
    wr.remainingMsgCnt = 0;
    ASSERT_TRUE(wr.isComplete());
  }

  // isSendOp
  {
    ActiveVirtualWr wr;
    for (auto opcode :
         {IBV_WR_SEND,
          IBV_WR_RDMA_WRITE,
          IBV_WR_RDMA_WRITE_WITH_IMM,
          IBV_WR_RDMA_READ,
          IBV_WR_ATOMIC_FETCH_AND_ADD,
          IBV_WR_ATOMIC_CMP_AND_SWP}) {
      wr.opcode = opcode;
      ASSERT_TRUE(wr.isSendOp()) << "opcode " << opcode << " should be send op";
    }
    wr.opcode = IBV_WR_SEND_WITH_IMM;
    ASSERT_FALSE(wr.isSendOp());
  }
}

TEST_F(IbverbxTestFixture, WrTracker) {
  // add, find, remove
  {
    WrTracker<ActiveVirtualWr> tracker;

    ActiveVirtualWr wr0;
    wr0.userWrId = 100;
    ActiveVirtualWr wr1;
    wr1.userWrId = 200;

    ASSERT_EQ(tracker.add(std::move(wr0)), 0);
    ASSERT_EQ(tracker.add(std::move(wr1)), 1);
    ASSERT_EQ(tracker.activeCount(), 2);

    ASSERT_NE(tracker.find(0), nullptr);
    ASSERT_EQ(tracker.find(0)->userWrId, 100);
    ASSERT_NE(tracker.find(1), nullptr);
    ASSERT_EQ(tracker.find(1)->userWrId, 200);

    // const find
    const auto& constTracker = tracker;
    ASSERT_NE(constTracker.find(0), nullptr);

    // find non-existent
    ASSERT_EQ(tracker.find(999), nullptr);

    // remove
    tracker.remove(0);
    ASSERT_EQ(tracker.activeCount(), 1);
    ASSERT_EQ(tracker.find(0), nullptr);
    ASSERT_NE(tracker.find(1), nullptr);

    // remove non-existent is a no-op
    tracker.remove(999);
    ASSERT_EQ(tracker.activeCount(), 1);
  }

  // queues and counts
  {
    WrTracker<ActiveVirtualWr> tracker;
    ASSERT_FALSE(tracker.hasPendingPost());
    ASSERT_FALSE(tracker.hasPendingCompletion());

    for (int i = 0; i < 3; i++) {
      ActiveVirtualWr wr;
      wr.userWrId = i;
      tracker.add(std::move(wr));
    }
    ASSERT_EQ(tracker.activeCount(), 3);
    ASSERT_EQ(tracker.pendingPostCount(), 3);
    ASSERT_EQ(tracker.pendingCompletionCount(), 3);

    // FIFO order for pending queue
    ASSERT_EQ(tracker.frontPendingPost(), 0);
    tracker.popPendingPost();
    ASSERT_EQ(tracker.frontPendingPost(), 1);
    tracker.popPendingPost();
    ASSERT_EQ(tracker.frontPendingPost(), 2);
    tracker.popPendingPost();
    ASSERT_FALSE(tracker.hasPendingPost());

    // Outstanding queue is independent
    ASSERT_EQ(tracker.pendingCompletionCount(), 3);
    ASSERT_EQ(tracker.frontPendingCompletion(), 0);
    tracker.popPendingCompletion();
    ASSERT_EQ(tracker.frontPendingCompletion(), 1);
    tracker.popPendingCompletion();
    tracker.popPendingCompletion();
    ASSERT_FALSE(tracker.hasPendingCompletion());

    // Popping queues does not affect active map
    ASSERT_EQ(tracker.activeCount(), 3);
  }

  // move semantics
  {
    WrTracker<ActiveVirtualWr> tracker;

    ActiveVirtualWr wr;
    wr.userWrId = 42;
    wr.deviceKeys[0] = MemoryRegionKeys{111, 222};
    wr.deviceKeys[1] = MemoryRegionKeys{333, 444};

    uint64_t id = tracker.add(std::move(wr));
    ASSERT_TRUE(wr.deviceKeys.empty()); // NOLINT(bugprone-use-after-move)

    auto* found = tracker.find(id);
    ASSERT_NE(found, nullptr);
    ASSERT_EQ(found->deviceKeys.size(), 2);
    ASSERT_EQ(found->deviceKeys[0].lkey, 111);
    ASSERT_EQ(found->deviceKeys[1].rkey, 444);
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualWcDefaults) {
  IbvVirtualWc wc;
  ASSERT_EQ(wc.wrId, 0);
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
  ASSERT_EQ(wc.opcode, IBV_WC_SEND);
  ASSERT_EQ(wc.qpNum, 0);
  ASSERT_EQ(wc.immData, 0);
  ASSERT_EQ(wc.byteLen, 0);
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

TEST_F(IbverbxTestFixture, IbvVirtualQpIsMultiQp) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;

  // totalQps > 1 → isMultiQp() returns true
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    auto virtualQp = pd->createVirtualQp(4, &initAttr, &virtualCq);
    ASSERT_TRUE(virtualQp);
    ASSERT_TRUE(virtualQp->isMultiQp());
  }

  // totalQps == 1 → isMultiQp() returns false
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    auto virtualQp = pd->createVirtualQp(1, &initAttr, &virtualCq);
    ASSERT_TRUE(virtualQp);
    ASSERT_FALSE(virtualQp->isMultiQp());
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualQpRegisterUnregisterWithVirtualCq) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;

  // Test 1: Constructor auto-registers all physical QPs + notifyQp
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    int totalQps = 4;
    auto maybeVirtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    // Should have totalQps + 1 (notify QP) registrations
    ASSERT_EQ(virtualCq.registeredQps_.size(), totalQps + 1);

    // Verify virtualCq_ back-pointer
    ASSERT_EQ(virtualQp.virtualCq_, &virtualCq);

    // Verify each physical QP is registered correctly
    for (size_t i = 0; i < virtualQp.getQpsRef().size(); i++) {
      auto& pqp = virtualQp.getQpsRef().at(i);
      auto* info =
          virtualCq.findRegisteredQpInfo(pqp.qp()->qp_num, pqp.getDeviceId());
      ASSERT_NE(info, nullptr);
      ASSERT_EQ(info->vqp, &virtualQp);
      ASSERT_TRUE(info->isMultiQp);
      ASSERT_EQ(info->virtualQpNum, virtualQp.getVirtualQpNum());
    }

    // Verify notify QP is registered
    auto& notifyQp = virtualQp.getNotifyQpRef();
    auto* notifyInfo = virtualCq.findRegisteredQpInfo(
        notifyQp.qp()->qp_num, notifyQp.getDeviceId());
    ASSERT_NE(notifyInfo, nullptr);
    ASSERT_EQ(notifyInfo->vqp, &virtualQp);
  }

  // Test 2: Destructor auto-unregisters all physical QPs
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    int totalQps = 4;
    std::vector<std::pair<uint32_t, int32_t>> qpIds;

    {
      auto maybeVirtualQp =
          pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
      ASSERT_TRUE(maybeVirtualQp);
      auto virtualQp = std::move(*maybeVirtualQp);

      // Record QP identifiers before destruction
      for (size_t i = 0; i < virtualQp.getQpsRef().size(); i++) {
        auto& pqp = virtualQp.getQpsRef().at(i);
        qpIds.emplace_back(pqp.qp()->qp_num, pqp.getDeviceId());
      }
      auto& notifyQp = virtualQp.getNotifyQpRef();
      qpIds.emplace_back(notifyQp.qp()->qp_num, notifyQp.getDeviceId());

      ASSERT_EQ(virtualCq.registeredQps_.size(), totalQps + 1);
    } // virtualQp destroyed here

    // All registrations should be gone
    ASSERT_TRUE(virtualCq.registeredQps_.empty());
    for (const auto& [qpNum, deviceId] : qpIds) {
      ASSERT_EQ(virtualCq.findRegisteredQpInfo(qpNum, deviceId), nullptr);
    }
  }

  // Test 3: Move constructor re-registers with new pointer
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    int totalQps = 4;
    auto maybeVirtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp1 = std::move(*maybeVirtualQp);

    // Move-construct
    IbvVirtualQp virtualQp2(std::move(virtualQp1));

    // Old virtualCq_ should be nullptr
    ASSERT_EQ(virtualQp1.virtualCq_, nullptr);

    // Registration count unchanged
    ASSERT_EQ(virtualCq.registeredQps_.size(), totalQps + 1);

    // All entries point to new VirtualQp
    for (size_t i = 0; i < virtualQp2.getQpsRef().size(); i++) {
      auto& pqp = virtualQp2.getQpsRef().at(i);
      auto* info =
          virtualCq.findRegisteredQpInfo(pqp.qp()->qp_num, pqp.getDeviceId());
      ASSERT_NE(info, nullptr);
      ASSERT_EQ(info->vqp, &virtualQp2);
    }
  }

  // Test 4: Move assignment re-registers with new pointer
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    int totalQps = 4;
    auto maybeVirtualQp1 = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
    ASSERT_TRUE(maybeVirtualQp1);
    auto virtualQp1 = std::move(*maybeVirtualQp1);

    auto maybeVirtualQp2 = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
    ASSERT_TRUE(maybeVirtualQp2);
    auto virtualQp2 = std::move(*maybeVirtualQp2);

    // Record QP1's physical QP ids before move
    std::vector<std::pair<uint32_t, int32_t>> qp1Ids;
    for (size_t i = 0; i < virtualQp1.getQpsRef().size(); i++) {
      auto& pqp = virtualQp1.getQpsRef().at(i);
      qp1Ids.emplace_back(pqp.qp()->qp_num, pqp.getDeviceId());
    }

    // Move-assign: virtualQp2's old QPs should be unregistered,
    // virtualQp1's QPs re-registered under virtualQp2's pointer
    virtualQp2 = std::move(virtualQp1);

    // Source's virtualCq_ should be nullptr
    ASSERT_EQ(virtualQp1.virtualCq_, nullptr);

    // virtualQp1's physical QPs should now be registered under virtualQp2
    for (const auto& [qpNum, deviceId] : qp1Ids) {
      auto* info = virtualCq.findRegisteredQpInfo(qpNum, deviceId);
      ASSERT_NE(info, nullptr);
      ASSERT_EQ(info->vqp, &virtualQp2);
    }
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualQpUpdateWrState) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  auto maybeVirtualQp = pd->createVirtualQp(4, &initAttr, &virtualCq);
  ASSERT_TRUE(maybeVirtualQp);
  auto virtualQp = std::move(*maybeVirtualQp);

  // Test 1: Basic decrement and opcode update
  {
    ActiveVirtualWr wr;
    wr.userWrId = 100;
    wr.remainingMsgCnt = 3;
    uint64_t id = virtualQp.sendTracker_.add(std::move(wr));

    auto result = virtualQp.updateWrState(
        virtualQp.sendTracker_, id, IBV_WC_SUCCESS, IBV_WC_RDMA_WRITE);
    ASSERT_TRUE(result.hasValue());

    auto* found = virtualQp.sendTracker_.find(id);
    ASSERT_NE(found, nullptr);
    ASSERT_EQ(found->remainingMsgCnt, 2);
    ASSERT_EQ(found->wcOpcode, IBV_WC_RDMA_WRITE);
    ASSERT_EQ(found->aggregatedStatus, IBV_WC_SUCCESS);

    virtualQp.sendTracker_.remove(id);
  }

  // Test 2: First-error-wins aggregation
  {
    ActiveVirtualWr wr;
    wr.userWrId = 200;
    wr.remainingMsgCnt = 3;
    uint64_t id = virtualQp.sendTracker_.add(std::move(wr));

    // First call: success
    auto r1 = virtualQp.updateWrState(
        virtualQp.sendTracker_, id, IBV_WC_SUCCESS, IBV_WC_RDMA_WRITE);
    ASSERT_TRUE(r1.hasValue());
    ASSERT_EQ(
        virtualQp.sendTracker_.find(id)->aggregatedStatus, IBV_WC_SUCCESS);

    // Second call: error
    auto r2 = virtualQp.updateWrState(
        virtualQp.sendTracker_, id, IBV_WC_REM_ACCESS_ERR, IBV_WC_RDMA_WRITE);
    ASSERT_TRUE(r2.hasValue());
    ASSERT_EQ(
        virtualQp.sendTracker_.find(id)->aggregatedStatus,
        IBV_WC_REM_ACCESS_ERR);

    // Third call: different error — first error should stick
    auto r3 = virtualQp.updateWrState(
        virtualQp.sendTracker_, id, IBV_WC_RETRY_EXC_ERR, IBV_WC_RDMA_WRITE);
    ASSERT_TRUE(r3.hasValue());
    ASSERT_EQ(
        virtualQp.sendTracker_.find(id)->aggregatedStatus,
        IBV_WC_REM_ACCESS_ERR);

    ASSERT_EQ(virtualQp.sendTracker_.find(id)->remainingMsgCnt, 0);

    virtualQp.sendTracker_.remove(id);
  }

  // Test 3: Not-found WR triggers CHECK failure
  {
    ASSERT_DEATH(
        virtualQp.updateWrState(
            virtualQp.sendTracker_, 99999, IBV_WC_SUCCESS, IBV_WC_SEND),
        "not found in tracker");
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualQpBuildVirtualWc) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  auto maybeVirtualQp = pd->createVirtualQp(4, &initAttr, &virtualCq);
  ASSERT_TRUE(maybeVirtualQp);
  auto virtualQp = std::move(*maybeVirtualQp);

  ActiveVirtualWr wr;
  wr.userWrId = 42;
  wr.aggregatedStatus = IBV_WC_REM_ACCESS_ERR;
  wr.length = 8192;
  wr.immData = 0xDEAD;
  wr.wcOpcode = IBV_WC_RDMA_WRITE;

  IbvVirtualWc wc = virtualQp.buildVirtualWc(wr);
  ASSERT_EQ(wc.wrId, 42);
  ASSERT_EQ(wc.status, IBV_WC_REM_ACCESS_ERR);
  ASSERT_EQ(wc.byteLen, 8192);
  ASSERT_EQ(wc.immData, 0xDEAD);
  ASSERT_EQ(wc.opcode, IBV_WC_RDMA_WRITE);
  ASSERT_EQ(wc.qpNum, virtualQp.getVirtualQpNum());
}

TEST_F(IbverbxTestFixture, IbvVirtualQpIsSendOpcode) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  auto maybeVirtualQp = pd->createVirtualQp(4, &initAttr, &virtualCq);
  ASSERT_TRUE(maybeVirtualQp);
  auto virtualQp = std::move(*maybeVirtualQp);

  // Send opcodes → true
  ASSERT_TRUE(virtualQp.isSendOpcode(IBV_WC_SEND));
  ASSERT_TRUE(virtualQp.isSendOpcode(IBV_WC_RDMA_WRITE));
  ASSERT_TRUE(virtualQp.isSendOpcode(IBV_WC_RDMA_READ));
  ASSERT_TRUE(virtualQp.isSendOpcode(IBV_WC_FETCH_ADD));
  ASSERT_TRUE(virtualQp.isSendOpcode(IBV_WC_COMP_SWAP));

  // Recv opcodes → false
  ASSERT_FALSE(virtualQp.isSendOpcode(IBV_WC_RECV));
  ASSERT_FALSE(virtualQp.isSendOpcode(IBV_WC_RECV_RDMA_WITH_IMM));
}

TEST_F(IbverbxTestFixture, IbvVirtualQpHasQpCapacity) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 4;
  int maxMsgCntPerQp = 2;

  // With limit: capacity depends on outstanding count
  {
    auto maybeVirtualQp =
        pd->createVirtualQp(totalQps, &initAttr, &virtualCq, maxMsgCntPerQp);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    ASSERT_TRUE(virtualQp.hasQpCapacity(0));

    // Fill to limit
    virtualQp.getQpsRef().at(0).enquePhysicalSendWrStatus(0, 0);
    ASSERT_TRUE(virtualQp.hasQpCapacity(0));
    virtualQp.getQpsRef().at(0).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(virtualQp.hasQpCapacity(0));

    // Free one, capacity returns
    virtualQp.getQpsRef().at(0).dequePhysicalSendWrStatus();
    ASSERT_TRUE(virtualQp.hasQpCapacity(0));
  }

  // No limit (maxMsgCntPerQp == -1): always has capacity
  {
    auto maybeVirtualQp =
        pd->createVirtualQp(totalQps, &initAttr, &virtualCq, -1);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    virtualQp.getQpsRef().at(0).enquePhysicalSendWrStatus(0, 0);
    virtualQp.getQpsRef().at(0).enquePhysicalSendWrStatus(0, 0);
    virtualQp.getQpsRef().at(0).enquePhysicalSendWrStatus(0, 0);
    ASSERT_TRUE(virtualQp.hasQpCapacity(0));
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualQpBuildPhysicalSendWr) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int32_t deviceId = 0;

  // Test 1: IBV_WR_RDMA_WRITE (default SPRAY mode)
  {
    auto maybeVirtualQp =
        pd->createVirtualQp(4, &initAttr, &virtualCq, 100, 1024);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    ActiveVirtualWr pending;
    pending.localAddr = reinterpret_cast<void*>(0x1000);
    pending.remoteAddr = 0x2000;
    pending.length = 2048;
    pending.offset = 512;
    pending.opcode = IBV_WR_RDMA_WRITE;
    pending.deviceKeys[deviceId] = MemoryRegionKeys{111, 222};

    uint32_t fragLen = 1024;
    auto [sendWr, sendSge] =
        virtualQp.buildPhysicalSendWr(pending, deviceId, fragLen);
    sendWr.sg_list = &sendSge;

    ASSERT_EQ(sendSge.addr, 0x1000 + 512);
    ASSERT_EQ(sendSge.length, fragLen);
    ASSERT_EQ(sendSge.lkey, 111);
    ASSERT_EQ(sendWr.sg_list, &sendSge);
    ASSERT_EQ(sendWr.num_sge, 1);
    ASSERT_EQ(sendWr.send_flags, IBV_SEND_SIGNALED);
    ASSERT_EQ(sendWr.wr.rdma.remote_addr, 0x2000 + 512);
    ASSERT_EQ(sendWr.wr.rdma.rkey, 222);
    ASSERT_EQ(sendWr.opcode, IBV_WR_RDMA_WRITE);
  }

  // Test 2: IBV_WR_RDMA_WRITE_WITH_IMM in SPRAY mode → opcode downgraded
  //         to IBV_WR_RDMA_WRITE (notify sent separately)
  {
    auto maybeVirtualQp =
        pd->createVirtualQp(4, &initAttr, &virtualCq, 100, 1024);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    ActiveVirtualWr pending;
    pending.localAddr = reinterpret_cast<void*>(0x1000);
    pending.remoteAddr = 0x2000;
    pending.length = 2048;
    pending.offset = 0;
    pending.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    pending.deviceKeys[deviceId] = MemoryRegionKeys{111, 222};

    auto [sendWr, sendSge] =
        virtualQp.buildPhysicalSendWr(pending, deviceId, 1024);
    sendWr.sg_list = &sendSge;

    ASSERT_EQ(sendWr.opcode, IBV_WR_RDMA_WRITE);
  }

  // Test 3: IBV_WR_RDMA_WRITE_WITH_IMM in DQPLB mode → opcode preserved,
  //         imm_data populated
  {
    auto maybeVirtualQp = pd->createVirtualQp(
        4, &initAttr, &virtualCq, 100, 1024, LoadBalancingScheme::DQPLB);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    ActiveVirtualWr pending;
    pending.localAddr = reinterpret_cast<void*>(0x1000);
    pending.remoteAddr = 0x2000;
    pending.length = 1024;
    pending.offset = 0;
    pending.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    pending.deviceKeys[deviceId] = MemoryRegionKeys{111, 222};

    // Last fragment (offset + fragLen >= length) → notify bit set
    auto [sendWr, sendSge] =
        virtualQp.buildPhysicalSendWr(pending, deviceId, 1024);
    sendWr.sg_list = &sendSge;

    ASSERT_EQ(sendWr.opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
    // imm_data should have notify bit set (last fragment)
    ASSERT_NE(sendWr.imm_data & (1U << kNotifyBit), 0);
  }
}

} // namespace ibverbx

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
