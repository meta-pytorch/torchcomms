// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/// Integration test for RDMA transport.
/// Requires at least 2 RDMA NICs. GPU tests require CUDA devices.
/// Uses two transport instances on different NICs within the same process.

#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/rdma/RdmaRegistrationHandle.h"
#include "comms/uniflow/transport/rdma/RdmaSlabPool.h"
#include "comms/uniflow/transport/rdma/RdmaTransport.h"

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy
#include <cstring>
#include <random>

#include <gtest/gtest.h>

namespace uniflow {

/// Friend-class wrapper to construct RegisteredSegment /
/// RemoteRegisteredSegment with handles for testing. The name must be exactly
/// "SegmentTest" to match the friend declaration in Segment.h.
class SegmentTest {
 public:
  static RegisteredSegment makeRegistered(
      Segment& segment,
      std::unique_ptr<RegistrationHandle> handle) {
    RegisteredSegment reg(segment);
    reg.handles_.push_back(std::move(handle));
    return reg;
  }

  static RemoteRegisteredSegment makeRemote(
      void* buf,
      size_t len,
      std::unique_ptr<RemoteRegistrationHandle> handle) {
    RemoteRegisteredSegment remote(buf, len);
    remote.handles_.push_back(std::move(handle));
    return remote;
  }
};

class SingleHostTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ibvApi_ = std::make_shared<IbvApi>();
    cudaDriverApi_ = std::make_shared<CudaDriverApi>();
    auto initStatus = ibvApi_->init();
    ASSERT_FALSE(initStatus.hasError())
        << "Failed to init IbvApi: " << initStatus.error().message();

    auto devResult = ibvApi_->getDeviceList(&numDevices_);
    ASSERT_TRUE(devResult.hasValue())
        << "Failed to get device list: " << devResult.error().message();
    deviceList_ = devResult.value();
    ASSERT_GE(numDevices_, 2)
        << "Need at least 2 RDMA devices, found " << numDevices_;

    for (int i = 0; i < numDevices_; ++i) {
      auto nameResult = ibvApi_->getDeviceName(deviceList_[i]);
      ASSERT_TRUE(nameResult.hasValue());
      deviceNames_.emplace_back(nameResult.value());
    }

    evbThread_ = std::make_unique<ScopedEventBaseThread>();
  }

  void TearDown() override {
    evbThread_.reset();
    if (deviceList_) {
      ibvApi_->freeDeviceList(deviceList_);
    }
  }

  /// Create two factories on different NICs, create transports, connect them.
  /// Returns the two connected transports and their factories.
  ///
  /// When a NIC list is empty (the default), the first two enumerated devices
  /// are used. This keeps the data-path tests vendor-agnostic so they run on
  /// any RDMA host (mlx5, bnxt_re/Thor2, etc.) without hardcoded device names.
  struct ConnectedPair {
    std::unique_ptr<RdmaTransportFactory> factory0;
    std::unique_ptr<RdmaTransportFactory> factory1;
    std::unique_ptr<Transport> transport0;
    std::unique_ptr<Transport> transport1;
  };

  ConnectedPair connectPair(
      std::vector<std::string> nicVec0 = {},
      std::vector<std::string> nicVec1 = {}) {
    if (nicVec0.empty()) {
      nicVec0 = {deviceNames_[0]};
    }
    if (nicVec1.empty()) {
      nicVec1 = {deviceNames_[1]};
    }
    ConnectedPair pair;
    auto* evb = evbThread_->getEventBase();
    pair.factory0 = std::make_unique<RdmaTransportFactory>(
        nicVec0, evb, RdmaTransportConfig{}, ibvApi_);
    pair.factory1 = std::make_unique<RdmaTransportFactory>(
        nicVec1, evb, RdmaTransportConfig{}, ibvApi_);

    auto topo0 = pair.factory0->getTopology();
    auto topo1 = pair.factory1->getTopology();

    auto r0 = pair.factory0->createTransport(topo1);
    auto r1 = pair.factory1->createTransport(topo0);
    EXPECT_TRUE(r0.hasValue()) << r0.error().message();
    EXPECT_TRUE(r1.hasValue()) << r1.error().message();

    pair.transport0 = std::move(r0.value());
    pair.transport1 = std::move(r1.value());

    auto info0 = pair.transport0->bind();
    auto info1 = pair.transport1->bind();
    auto status0 = pair.transport0->connect(info1);
    auto status1 = pair.transport1->connect(info0);
    EXPECT_TRUE(status0) << "transport0 connect failed: "
                         << status0.error().message();
    EXPECT_TRUE(status1) << "transport1 connect failed: "
                         << status1.error().message();

    return pair;
  }

  std::shared_ptr<IbvApi> ibvApi_;
  std::shared_ptr<CudaDriverApi> cudaDriverApi_;
  ibv_device** deviceList_{nullptr};
  int numDevices_{0};
  std::vector<std::string> deviceNames_;
  std::unique_ptr<ScopedEventBaseThread> evbThread_;
};

// --- Registration integration tests ---

TEST_F(SingleHostTest, DramRegisterSerializeImportRoundTrip) {
  RdmaTransportFactory factory(
      {deviceNames_[0]}, evbThread_->getEventBase(), {}, ibvApi_);

  // Register a DRAM segment.
  char buf[4096]{};
  Segment segment(buf, sizeof(buf), MemoryType::DRAM);
  auto regResult = factory.registerSegment(segment);
  ASSERT_TRUE(regResult.hasValue()) << regResult.error().message();

  // Serialize and import on the same factory (validates wire format).
  auto serialized = regResult.value()->serialize();
  auto importResult = factory.importSegment(sizeof(buf), serialized);
  ASSERT_TRUE(importResult.hasValue()) << importResult.error().message();

  auto* local = dynamic_cast<RdmaRegistrationHandle*>(regResult.value().get());
  auto* remote =
      dynamic_cast<RdmaRemoteRegistrationHandle*>(importResult.value().get());
  ASSERT_NE(local, nullptr);
  ASSERT_NE(remote, nullptr);
  EXPECT_EQ(remote->numMrs(), local->numMrs());
  EXPECT_EQ(remote->rkey(0), local->rkey(0));
  EXPECT_EQ(remote->domainId(), local->domainId());
}

TEST_F(SingleHostTest, MultiNicDramRegistration) {
  RdmaTransportFactory factory(
      {deviceNames_[0], deviceNames_[1]},
      evbThread_->getEventBase(),
      {},
      ibvApi_);

  char buf[8192]{};
  Segment segment(buf, sizeof(buf), MemoryType::DRAM);
  auto regResult = factory.registerSegment(segment);
  ASSERT_TRUE(regResult.hasValue()) << regResult.error().message();

  auto* handle = dynamic_cast<RdmaRegistrationHandle*>(regResult.value().get());
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->numMrs(), 2u);
}

TEST_F(SingleHostTest, RepeatedRegisterDeregister) {
  RdmaTransportFactory factory(
      {deviceNames_[0]}, evbThread_->getEventBase(), {}, ibvApi_);
  char buf[4096]{};

  // Run 100 register/deregister cycles to catch MR or fd leaks.
  for (int i = 0; i < 100; ++i) {
    Segment segment(buf, sizeof(buf), MemoryType::DRAM);
    auto result = factory.registerSegment(segment);
    ASSERT_TRUE(result.hasValue())
        << "Iteration " << i << ": " << result.error().message();
    // Handle goes out of scope → deregMr called automatically.
  }
}

// --- VRAM registration integration tests ---

TEST_F(SingleHostTest, VramRegistrationWithCudaMalloc) {
  RdmaTransportFactory factory(
      {deviceNames_[0]},
      evbThread_->getEventBase(),
      {},
      ibvApi_,
      cudaDriverApi_);

  // Allocate GPU memory via cudaMalloc (page-aligned by default).
  constexpr size_t kSize = 1 << 20; // 1 MiB
  void* devPtr = nullptr;
  ASSERT_EQ(cudaMalloc(&devPtr, kSize), cudaSuccess) << "cudaMalloc failed";
  ASSERT_NE(devPtr, nullptr);

  Segment segment(devPtr, kSize, MemoryType::VRAM, 0);
  auto regResult = factory.registerSegment(segment);
  ASSERT_TRUE(regResult.hasValue()) << regResult.error().message();

  auto* handle = dynamic_cast<RdmaRegistrationHandle*>(regResult.value().get());
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->numMrs(), 1u);

  // Serialize → import round-trip.
  auto serialized = regResult.value()->serialize();
  auto importResult = factory.importSegment(kSize, serialized);
  ASSERT_TRUE(importResult.hasValue()) << importResult.error().message();

  auto* remote =
      dynamic_cast<RdmaRemoteRegistrationHandle*>(importResult.value().get());
  ASSERT_NE(remote, nullptr);
  EXPECT_EQ(remote->rkey(0), handle->rkey(0));
  EXPECT_EQ(remote->domainId(), handle->domainId());

  // Clean up: release handle before freeing GPU memory.
  regResult.value().reset();
  (void)cudaFree(devPtr);
}

TEST_F(SingleHostTest, VramRegistrationWithMultiNics) {
  RdmaTransportFactory factory(
      {deviceNames_[0], deviceNames_[1]},
      evbThread_->getEventBase(),
      {},
      ibvApi_,
      cudaDriverApi_);

  // Allocate GPU memory via cudaMalloc (page-aligned by default).
  constexpr size_t kSize = 1 << 20; // 1 MiB
  void* devPtr = nullptr;
  ASSERT_EQ(cudaMalloc(&devPtr, kSize), cudaSuccess) << "cudaMalloc failed";
  ASSERT_NE(devPtr, nullptr);

  Segment segment(devPtr, kSize, MemoryType::VRAM, 0);
  auto regResult = factory.registerSegment(segment);
  ASSERT_TRUE(regResult.hasValue()) << regResult.error().message();

  auto* handle = dynamic_cast<RdmaRegistrationHandle*>(regResult.value().get());
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->numMrs(), 2u);

  // Serialize → import round-trip.
  auto serialized = regResult.value()->serialize();
  auto importResult = factory.importSegment(kSize, serialized);
  ASSERT_TRUE(importResult.hasValue()) << importResult.error().message();

  auto* remote =
      dynamic_cast<RdmaRemoteRegistrationHandle*>(importResult.value().get());
  ASSERT_NE(remote, nullptr);
  EXPECT_EQ(remote->rkey(0), handle->rkey(0));
  EXPECT_EQ(remote->rkey(1), handle->rkey(1));
  EXPECT_EQ(remote->domainId(), handle->domainId());

  // Clean up: release handle before freeing GPU memory.
  regResult.value().reset();
  (void)cudaFree(devPtr);
}

TEST_F(SingleHostTest, VramRegistrationWithUnalignedAddress) {
  RdmaTransportFactory factory(
      {deviceNames_[0]},
      evbThread_->getEventBase(),
      {},
      ibvApi_,
      cudaDriverApi_);

  // Allocate a large GPU buffer and offset into it to create a
  // non-page-aligned address. cudaMalloc returns page-aligned memory,
  // so any non-zero sub-page offset produces a misaligned pointer.
  constexpr size_t kAllocSize = 1 << 20; // 1 MiB
  constexpr size_t kOffset = 127; // deliberately misaligned
  constexpr size_t kRegSize = kAllocSize - kOffset;
  void* devPtr = nullptr;
  ASSERT_EQ(cudaMalloc(&devPtr, kAllocSize), cudaSuccess)
      << "cudaMalloc failed";
  ASSERT_NE(devPtr, nullptr);

  void* unaligned = static_cast<char*>(devPtr) + kOffset;
  Segment segment(unaligned, kRegSize, MemoryType::VRAM, 0);
  auto regResult = factory.registerSegment(segment);
  ASSERT_TRUE(regResult.hasValue()) << regResult.error().message();

  auto* handle = dynamic_cast<RdmaRegistrationHandle*>(regResult.value().get());
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->numMrs(), 1u);

  // Clean up.
  regResult.value().reset();
  (void)cudaFree(devPtr);
}

// --- Connection tests ---

TEST_F(SingleHostTest, TwoTransportsConnectOnDifferentNICs) {
  auto pair = connectPair({deviceNames_[0]}, {deviceNames_[1]});
  EXPECT_EQ(pair.transport0->state(), TransportState::Connected);
  EXPECT_EQ(pair.transport1->state(), TransportState::Connected);

  pair.transport0->shutdown();
  pair.transport1->shutdown();
  EXPECT_EQ(pair.transport0->state(), TransportState::Disconnected);
  EXPECT_EQ(pair.transport1->state(), TransportState::Disconnected);
}

TEST_F(SingleHostTest, MultiNicTransportConnects) {
  auto pair = connectPair(
      {deviceNames_[0], deviceNames_[1]}, {deviceNames_[0], deviceNames_[1]});
  EXPECT_EQ(pair.transport0->state(), TransportState::Connected);
  EXPECT_EQ(pair.transport1->state(), TransportState::Connected);

  pair.transport0->shutdown();
  pair.transport1->shutdown();
  EXPECT_EQ(pair.transport0->state(), TransportState::Disconnected);
  EXPECT_EQ(pair.transport1->state(), TransportState::Disconnected);
}

// --- Parameterized DRAM put/get tests ---

struct TransferParam {
  size_t bufSize;
  size_t numRequests;
  std::string name;
};

std::string transferParamName(
    const ::testing::TestParamInfo<TransferParam>& info) {
  return info.param.name;
}

class DramTransferTest : public SingleHostTest,
                         public ::testing::WithParamInterface<TransferParam> {};

TEST_P(DramTransferTest, Put) {
  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  auto pair = connectPair();

  std::vector<char> sendBuf(totalSize);
  std::vector<char> recvBuf(totalSize, 0);

  for (size_t r = 0; r < numRequests; ++r) {
    std::memset(
        sendBuf.data() + r * bufSize, static_cast<int>(0xA0 + r), bufSize);
  }

  Segment sendSeg(sendBuf.data(), totalSize, MemoryType::DRAM);
  Segment recvSeg(recvBuf.data(), totalSize, MemoryType::DRAM);

  auto sendReg = pair.factory0->registerSegment(sendSeg);
  ASSERT_TRUE(sendReg.hasValue()) << sendReg.error().message();
  auto recvReg = pair.factory1->registerSegment(recvSeg);
  ASSERT_TRUE(recvReg.hasValue()) << recvReg.error().message();

  auto recvPayload = recvReg.value()->serialize();
  auto remoteHandle = pair.factory0->importSegment(totalSize, recvPayload);
  ASSERT_TRUE(remoteHandle.hasValue()) << remoteHandle.error().message();

  auto localReg =
      SegmentTest::makeRegistered(sendSeg, std::move(sendReg.value()));
  auto remoteReg = SegmentTest::makeRemote(
      recvBuf.data(), totalSize, std::move(remoteHandle.value()));

  std::vector<TransferRequest> reqs;
  reqs.reserve(numRequests);
  for (size_t r = 0; r < numRequests; ++r) {
    reqs.push_back(
        TransferRequest{
            .local = localReg.span(r * bufSize, bufSize),
            .remote = remoteReg.span(r * bufSize, bufSize),
        });
  }

  auto putStatus = pair.transport0->put(reqs, {}).get();
  ASSERT_FALSE(putStatus.hasError())
      << "put failed: " << putStatus.error().message();

  for (size_t r = 0; r < numRequests; ++r) {
    uint8_t expected = static_cast<uint8_t>(0xA0 + r);
    for (size_t i = 0; i < bufSize; ++i) {
      ASSERT_EQ(static_cast<uint8_t>(recvBuf[r * bufSize + i]), expected)
          << "Data mismatch at request " << r << " byte " << i;
    }
  }

  pair.transport0->shutdown();
  pair.transport1->shutdown();
}

TEST_P(DramTransferTest, Get) {
  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  auto pair = connectPair();

  std::vector<char> localBuf(totalSize, 0);
  std::vector<char> remoteBuf(totalSize);

  for (size_t r = 0; r < numRequests; ++r) {
    std::memset(
        remoteBuf.data() + r * bufSize, static_cast<int>(0xB0 + r), bufSize);
  }

  Segment localSeg(localBuf.data(), totalSize, MemoryType::DRAM);
  Segment remoteSeg(remoteBuf.data(), totalSize, MemoryType::DRAM);

  auto localRegResult = pair.factory0->registerSegment(localSeg);
  ASSERT_TRUE(localRegResult.hasValue()) << localRegResult.error().message();
  auto remoteRegResult = pair.factory1->registerSegment(remoteSeg);
  ASSERT_TRUE(remoteRegResult.hasValue()) << remoteRegResult.error().message();

  auto remotePayload = remoteRegResult.value()->serialize();
  auto remoteHandle = pair.factory0->importSegment(totalSize, remotePayload);
  ASSERT_TRUE(remoteHandle.hasValue()) << remoteHandle.error().message();

  auto localReg =
      SegmentTest::makeRegistered(localSeg, std::move(localRegResult.value()));
  auto remoteReg = SegmentTest::makeRemote(
      remoteBuf.data(), totalSize, std::move(remoteHandle.value()));

  std::vector<TransferRequest> reqs;
  reqs.reserve(numRequests);
  for (size_t r = 0; r < numRequests; ++r) {
    reqs.push_back(
        TransferRequest{
            .local = localReg.span(r * bufSize, bufSize),
            .remote = remoteReg.span(r * bufSize, bufSize),
        });
  }

  auto getStatus = pair.transport0->get(reqs, {}).get();
  ASSERT_FALSE(getStatus.hasError())
      << "get failed: " << getStatus.error().message();

  for (size_t r = 0; r < numRequests; ++r) {
    uint8_t expected = static_cast<uint8_t>(0xB0 + r);
    for (size_t i = 0; i < bufSize; ++i) {
      ASSERT_EQ(static_cast<uint8_t>(localBuf[r * bufSize + i]), expected)
          << "Data mismatch at request " << r << " byte " << i;
    }
  }

  pair.transport0->shutdown();
  pair.transport1->shutdown();
}

INSTANTIATE_TEST_SUITE_P(
    DramTransfer,
    DramTransferTest,
    ::testing::Values(
        TransferParam{4096, 1, "4KB_x1"},
        TransferParam{12 * 1024 * 1024 + 12 * 1024, 1, "12MB12KB_x1"},
        TransferParam{12 * 1024 * 1024 + 12 * 1024, 4, "12MB12KB_x4"}),
    transferParamName);

// --- Parameterized GPU put/get tests ---

struct CudaBuffer {
  void* ptr{nullptr};
  size_t size{0};

  explicit CudaBuffer(size_t n, int device = 0) : size(n) {
    (void)cudaSetDevice(device);
    if (cudaMalloc(&ptr, n) != cudaSuccess) {
      ptr = nullptr;
    }
  }

  ~CudaBuffer() {
    if (ptr) {
      (void)cudaFree(ptr);
    }
  }

  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
};

class GpuTransferTest : public SingleHostTest,
                        public ::testing::WithParamInterface<TransferParam> {};

TEST_P(GpuTransferTest, Put) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 2) {
    GTEST_SKIP() << "Need at least 2 CUDA devices, found " << deviceCount;
  }

  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  auto pair = connectPair();

  CudaBuffer sendGpu(totalSize, 0);
  CudaBuffer recvGpu(totalSize, 1);
  ASSERT_NE(sendGpu.ptr, nullptr) << "cudaMalloc failed on device 0";
  ASSERT_NE(recvGpu.ptr, nullptr) << "cudaMalloc failed on device 1";

  std::vector<char> staging(totalSize);
  for (size_t r = 0; r < numRequests; ++r) {
    std::memset(
        staging.data() + r * bufSize, static_cast<int>(0xC0 + r), bufSize);
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess) << "cudaSetDevice failed";
  ASSERT_EQ(
      cudaMemcpy(
          sendGpu.ptr, staging.data(), totalSize, cudaMemcpyHostToDevice),
      cudaSuccess)
      << "cudaMemcpy failed";
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess) << "cudaSetDevice failed";
  ASSERT_EQ(cudaMemset(recvGpu.ptr, 0, totalSize), cudaSuccess)
      << "cudaMemset failed";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
      << "cudaDeviceSynchronize failed";

  Segment sendSeg(sendGpu.ptr, totalSize, MemoryType::VRAM, 0);
  Segment recvSeg(recvGpu.ptr, totalSize, MemoryType::VRAM, 1);

  auto sendReg = pair.factory0->registerSegment(sendSeg);
  ASSERT_TRUE(sendReg.hasValue()) << sendReg.error().message();
  auto recvReg = pair.factory1->registerSegment(recvSeg);
  ASSERT_TRUE(recvReg.hasValue()) << recvReg.error().message();

  auto recvPayload = recvReg.value()->serialize();
  auto remoteHandle = pair.factory0->importSegment(totalSize, recvPayload);
  ASSERT_TRUE(remoteHandle.hasValue()) << remoteHandle.error().message();

  auto localReg =
      SegmentTest::makeRegistered(sendSeg, std::move(sendReg.value()));
  auto remoteReg = SegmentTest::makeRemote(
      recvGpu.ptr, totalSize, std::move(remoteHandle.value()));

  std::vector<TransferRequest> reqs;
  reqs.reserve(numRequests);
  for (size_t r = 0; r < numRequests; ++r) {
    reqs.push_back(
        TransferRequest{
            .local = localReg.span(r * bufSize, bufSize),
            .remote = remoteReg.span(r * bufSize, bufSize),
        });
  }

  auto putStatus = pair.transport0->put(reqs, {}).get();
  ASSERT_FALSE(putStatus.hasError())
      << "GPU put failed: " << putStatus.error().message();

  std::vector<char> verify(totalSize, 0);
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess) << "cudaSetDevice failed";
  ASSERT_EQ(
      cudaMemcpy(verify.data(), recvGpu.ptr, totalSize, cudaMemcpyDeviceToHost),
      cudaSuccess)
      << "cudaMemcpy failed";
  for (size_t r = 0; r < numRequests; ++r) {
    uint8_t expected = static_cast<uint8_t>(0xC0 + r);
    for (size_t i = 0; i < bufSize; ++i) {
      ASSERT_EQ(static_cast<uint8_t>(verify[r * bufSize + i]), expected)
          << "GPU data mismatch at request " << r << " byte " << i;
    }
  }

  pair.transport0->shutdown();
  pair.transport1->shutdown();
}

TEST_P(GpuTransferTest, Get) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 2) {
    GTEST_SKIP() << "Need at least 2 CUDA devices, found " << deviceCount;
  }

  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  auto pair = connectPair();

  CudaBuffer localGpu(totalSize, 0);
  CudaBuffer remoteGpu(totalSize, 1);
  ASSERT_NE(localGpu.ptr, nullptr) << "cudaMalloc failed on device 0";
  ASSERT_NE(remoteGpu.ptr, nullptr) << "cudaMalloc failed on device 1";

  ASSERT_EQ(cudaSetDevice(0), cudaSuccess) << "cudaSetDevice failed";
  ASSERT_EQ(cudaMemset(localGpu.ptr, 0, totalSize), cudaSuccess)
      << "cudaMemset failed";
  std::vector<char> staging(totalSize);
  for (size_t r = 0; r < numRequests; ++r) {
    std::memset(
        staging.data() + r * bufSize, static_cast<int>(0xD0 + r), bufSize);
  }
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess) << "cudaSetDevice failed";
  ASSERT_EQ(
      cudaMemcpy(
          remoteGpu.ptr, staging.data(), totalSize, cudaMemcpyHostToDevice),
      cudaSuccess)
      << "cudaMemcpy failed";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
      << "cudaDeviceSynchronize failed";

  Segment localSeg(localGpu.ptr, totalSize, MemoryType::VRAM, 0);
  Segment remoteSeg(remoteGpu.ptr, totalSize, MemoryType::VRAM, 1);

  auto localRegResult = pair.factory0->registerSegment(localSeg);
  ASSERT_TRUE(localRegResult.hasValue()) << localRegResult.error().message();
  auto remoteRegResult = pair.factory1->registerSegment(remoteSeg);
  ASSERT_TRUE(remoteRegResult.hasValue()) << remoteRegResult.error().message();

  auto remotePayload = remoteRegResult.value()->serialize();
  auto remoteHandle = pair.factory0->importSegment(totalSize, remotePayload);
  ASSERT_TRUE(remoteHandle.hasValue()) << remoteHandle.error().message();

  auto localReg =
      SegmentTest::makeRegistered(localSeg, std::move(localRegResult.value()));
  auto remoteReg = SegmentTest::makeRemote(
      remoteGpu.ptr, totalSize, std::move(remoteHandle.value()));

  std::vector<TransferRequest> reqs;
  reqs.reserve(numRequests);
  for (size_t r = 0; r < numRequests; ++r) {
    reqs.push_back(
        TransferRequest{
            .local = localReg.span(r * bufSize, bufSize),
            .remote = remoteReg.span(r * bufSize, bufSize),
        });
  }

  auto getStatus = pair.transport0->get(reqs, {}).get();
  ASSERT_FALSE(getStatus.hasError())
      << "GPU get failed: " << getStatus.error().message();

  std::vector<char> verify(totalSize, 0);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess) << "cudaSetDevice failed";
  ASSERT_EQ(
      cudaMemcpy(
          verify.data(), localGpu.ptr, totalSize, cudaMemcpyDeviceToHost),
      cudaSuccess)
      << "cudaMemcpy failed";
  for (size_t r = 0; r < numRequests; ++r) {
    uint8_t expected = static_cast<uint8_t>(0xD0 + r);
    for (size_t i = 0; i < bufSize; ++i) {
      ASSERT_EQ(static_cast<uint8_t>(verify[r * bufSize + i]), expected)
          << "GPU data mismatch at request " << r << " byte " << i;
    }
  }

  pair.transport0->shutdown();
  pair.transport1->shutdown();
}

INSTANTIATE_TEST_SUITE_P(
    GpuTransfer,
    GpuTransferTest,
    ::testing::Values(
        TransferParam{4096, 1, "4KB_x1"},
        TransferParam{12 * 1024 * 1024 + 12 * 1024, 1, "12MB12KB_x1"},
        TransferParam{12 * 1024 * 1024 + 12 * 1024, 4, "12MB12KB_x4"}),
    transferParamName);

// --- DRAM send/recv tests ---

struct SendRecvParam {
  size_t bufSize;
  std::string name;
};

std::string sendRecvParamName(
    const ::testing::TestParamInfo<SendRecvParam>& info) {
  return info.param.name;
}

class DramSendRecvTest : public SingleHostTest,
                         public ::testing::WithParamInterface<SendRecvParam> {
 protected:
  struct SendRecvPair {
    std::unique_ptr<ScopedEventBaseThread> evbThread0;
    std::unique_ptr<ScopedEventBaseThread> evbThread1;
    std::shared_ptr<RdmaSlabPool> slabPool0;
    std::shared_ptr<RdmaSlabPool> slabPool1;
    std::unique_ptr<RdmaTransport> transport0;
    std::unique_ptr<RdmaTransport> transport1;
  };

  SendRecvPair connectSendRecvPair() {
    SendRecvPair pair;
    pair.evbThread0 = std::make_unique<ScopedEventBaseThread>();
    pair.evbThread1 = std::make_unique<ScopedEventBaseThread>();
    auto ibv = ibvApi_;
    auto cudaApi = std::make_shared<CudaApi>();
    auto cudaDriverApi = std::make_shared<CudaDriverApi>();

    int nic1Idx = numDevices_ > 3 ? 3 : numDevices_ - 1;
    auto nics0 = std::make_shared<std::vector<NicResources>>();
    nics0->reserve(1);
    nics0->emplace_back(deviceList_[0], ibv);

    auto nics1 = std::make_shared<std::vector<NicResources>>();
    nics1->reserve(1);
    nics1->emplace_back(deviceList_[nic1Idx], ibv);

    RdmaSlabPoolConfig poolConfig{.slabNum = 8, .slabSize = 512 * 1024};
    pair.slabPool0 =
        std::make_shared<RdmaSlabPool>(poolConfig, cudaApi, ibv, nics0);
    pair.slabPool1 =
        std::make_shared<RdmaSlabPool>(poolConfig, cudaApi, ibv, nics1);

    std::mt19937_64 rng{std::random_device{}()};
    uint64_t domain0 = rng();
    uint64_t domain1 = rng();

    RdmaTransportConfig config{.pipelineDepth = 8};
    pair.transport0 = std::make_unique<RdmaTransport>(
        ibv,
        cudaApi,
        cudaDriverApi,
        pair.evbThread0->getEventBase(),
        nics0,
        domain0,
        config,
        pair.slabPool0);
    pair.transport1 = std::make_unique<RdmaTransport>(
        ibv,
        cudaApi,
        cudaDriverApi,
        pair.evbThread1->getEventBase(),
        nics1,
        domain1,
        config,
        pair.slabPool1);

    auto info0 = pair.transport0->bind();
    auto info1 = pair.transport1->bind();
    EXPECT_FALSE(info0.empty());
    EXPECT_FALSE(info1.empty());

    auto st0 = pair.transport0->connect(info1);
    auto st1 = pair.transport1->connect(info0);
    EXPECT_TRUE(st0) << st0.error().message();
    EXPECT_TRUE(st1) << st1.error().message();

    return pair;
  }
};

TEST_P(DramSendRecvTest, SendRecv) {
  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;

  auto pair = connectSendRecvPair();

  std::vector<char> sendBuf(bufSize);
  std::vector<char> recvBuf(bufSize, 0);
  for (size_t i = 0; i < bufSize; ++i) {
    sendBuf[i] = static_cast<char>(i & 0xFF);
  }

  Segment sendSeg(sendBuf.data(), bufSize, MemoryType::DRAM);
  Segment recvSeg(recvBuf.data(), bufSize, MemoryType::DRAM);

  auto recvFuture = pair.transport1->recv(recvSeg.span());
  auto sendFuture = pair.transport0->send(sendSeg.span());

  auto sendStatus = sendFuture.get();
  ASSERT_FALSE(sendStatus.hasError())
      << "send failed: " << sendStatus.error().message();

  auto recvResult = recvFuture.get();
  ASSERT_TRUE(recvResult.hasValue())
      << "recv failed: " << recvResult.error().message();

  EXPECT_EQ(std::memcmp(sendBuf.data(), recvBuf.data(), bufSize), 0)
      << "Data mismatch";

  pair.transport0->shutdown();
  pair.transport1->shutdown();
}

INSTANTIATE_TEST_SUITE_P(
    DramSendRecv,
    DramSendRecvTest,
    ::testing::Values(
        SendRecvParam{4096, "4KB"},
        SendRecvParam{512 * 1024, "512KB"},
        SendRecvParam{2 * 1024 * 1024 + 1234, "2MB_unaligned"},
        SendRecvParam{100 * 1024 * 1024, "100MB"}),
    sendRecvParamName);

// --- VRAM send/recv tests ---

class VramSendRecvTest : public DramSendRecvTest {};

TEST_P(VramSendRecvTest, SendRecv) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 2) {
    GTEST_SKIP() << "Need at least 2 CUDA device, found " << deviceCount;
  }

  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;

  auto pair = connectSendRecvPair();

  CudaBuffer sendGpu(bufSize, 0);
  CudaBuffer recvGpu(bufSize, 1);
  ASSERT_NE(sendGpu.ptr, nullptr) << "cudaMalloc failed for send buffer";
  ASSERT_NE(recvGpu.ptr, nullptr) << "cudaMalloc failed for recv buffer";

  std::vector<char> staging(bufSize);
  for (size_t i = 0; i < bufSize; ++i) {
    staging[i] = static_cast<char>(i & 0xFF);
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess) << "cudaSetDevice failed";
  ASSERT_EQ(
      cudaMemcpy(sendGpu.ptr, staging.data(), bufSize, cudaMemcpyHostToDevice),
      cudaSuccess)
      << "cudaMemcpy failed";
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess) << "cudaSetDevice failed";
  ASSERT_EQ(cudaMemset(recvGpu.ptr, 0, bufSize), cudaSuccess)
      << "cudaMemset failed";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
      << "cudaDeviceSynchronize failed";

  Segment sendSeg(sendGpu.ptr, bufSize, MemoryType::VRAM, 0);
  Segment recvSeg(recvGpu.ptr, bufSize, MemoryType::VRAM, 1);

  auto recvFuture = pair.transport1->recv(recvSeg.span(), {.stream = nullptr});
  auto sendFuture = pair.transport0->send(sendSeg.span(), {.stream = nullptr});

  auto sendStatus = sendFuture.get();
  ASSERT_FALSE(sendStatus.hasError())
      << "send failed: " << sendStatus.error().message();

  auto recvResult = recvFuture.get();
  ASSERT_TRUE(recvResult.hasValue())
      << "recv failed: " << recvResult.error().message();

  std::vector<char> verify(bufSize, 0);
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess) << "cudaSetDevice failed";
  ASSERT_EQ(
      cudaMemcpy(verify.data(), recvGpu.ptr, bufSize, cudaMemcpyDeviceToHost),
      cudaSuccess)
      << "cudaMemcpy failed";
  EXPECT_EQ(std::memcmp(staging.data(), verify.data(), bufSize), 0)
      << "Data mismatch";

  pair.transport0->shutdown();
  pair.transport1->shutdown();
}

INSTANTIATE_TEST_SUITE_P(
    VramSendRecv,
    VramSendRecvTest,
    ::testing::Values(
        SendRecvParam{4096, "4KB"},
        SendRecvParam{512 * 1024, "512KB"},
        SendRecvParam{2 * 1024 * 1024 + 1234, "2MB_unaligned"},
        SendRecvParam{100 * 1024 * 1024, "100MB"}),
    sendRecvParamName);

} // namespace uniflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
