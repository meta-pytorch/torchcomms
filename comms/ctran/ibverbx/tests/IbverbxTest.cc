// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/tests/IbverbxTestFixture.h"

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

namespace ibverbx {

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

} // namespace ibverbx

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
