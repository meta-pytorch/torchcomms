// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/nvlink/NVLinkRegistrationHandle.h"
#include "comms/uniflow/transport/nvlink/NVLinkTransport.h"

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace uniflow {

/// Helper that leverages the `friend class SegmentTest` declarations
/// in RegisteredSegment and RemoteRegisteredSegment to construct them
/// from tests.
///
/// TODO: Remove this once production APIs exist for constructing
/// RegisteredSegment (via RegisteredSegment::create) and
/// RemoteRegisteredSegment (via Connection::recvSegment with handle
/// import — Phase 4). Currently these constructors are private, so
/// test code must go through this friend class.
class SegmentTest {
 public:
  static RegisteredSegment makeRegisteredSegment(
      void* buf,
      size_t len,
      MemoryType memType = MemoryType::DRAM,
      int deviceId = -1) {
    return RegisteredSegment(buf, len, memType, deviceId);
  }

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

namespace {

// RAII wrapper for cuMem VMM GPU memory allocation.
// Follows the VMM path required by registerSegment:
//   cuMemCreate -> cuMemAddressReserve -> cuMemMap -> cuMemSetAccess
class VmmAllocation {
 public:
  // Use init() after construction; ASSERT_* macros cannot be used in ctors.
  VmmAllocation() = default;

  Status init(
      CudaDriverApi& driverApi,
      int deviceId,
      size_t requestedSize,
      CUmemAllocationHandleType handleType =
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    driverApi_ = &driverApi;

    CUdevice device;
    CHECK_RETURN(driverApi_->cuDeviceGet(&device, deviceId));

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = handleType;

    CHECK_RETURN(driverApi_->cuMemGetAllocationGranularity(
        &granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    size_ = ((requestedSize + granularity_ - 1) / granularity_) * granularity_;

    CHECK_RETURN(driverApi_->cuMemCreate(&allocHandle_, size_, &prop, 0));
    created_ = true;

    CHECK_RETURN(
        driverApi_->cuMemAddressReserve(&ptr_, size_, granularity_, 0, 0));
    reserved_ = true;

    CHECK_RETURN(driverApi_->cuMemMap(ptr_, size_, 0, allocHandle_, 0));
    mapped_ = true;

    CUmemAccessDesc accessDesc{};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_RETURN(driverApi_->cuMemSetAccess(ptr_, size_, &accessDesc, 1));

    return Ok();
  }

  ~VmmAllocation() {
    if (!driverApi_) {
      return;
    }
    if (mapped_) {
      driverApi_->cuMemUnmap(ptr_, size_);
    }
    if (reserved_) {
      driverApi_->cuMemAddressFree(ptr_, size_);
    }
    if (created_) {
      driverApi_->cuMemRelease(allocHandle_);
    }
  }

  VmmAllocation(const VmmAllocation&) = delete;
  VmmAllocation& operator=(const VmmAllocation&) = delete;

  void* ptr() const {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(ptr_);
  }

  size_t size() const {
    return size_;
  }

 private:
  CudaDriverApi* driverApi_{nullptr};
  CUmemGenericAllocationHandle allocHandle_{};
  CUdeviceptr ptr_{0};
  size_t size_{0};
  size_t granularity_{0};
  bool created_{false};
  bool reserved_{false};
  bool mapped_{false};
};

// Fill a host buffer with a deterministic byte pattern.
void fillPattern(std::vector<uint8_t>& buf) {
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<uint8_t>(i ^ 0xA5);
  }
}

constexpr size_t kTransferSize = 4096;
constexpr int kDeviceA = 0;
constexpr int kDeviceB = 1;

class NVLinkTransportIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check that at least 2 GPUs are available.
    int deviceCount = 0;
    auto err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount < 2) {
      GTEST_SKIP() << "Need at least 2 GPUs, found " << deviceCount;
    }

#ifdef EXPECTED_GPU_NAME
    // Verify the GPU matches the expected type for this test target.
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, kDeviceA);
    if (std::string(prop.name).find(EXPECTED_GPU_NAME) == std::string::npos) {
      GTEST_SKIP() << "Expected " << EXPECTED_GPU_NAME << " GPU, found "
                   << prop.name;
    }
#endif

    // Check P2P support between device 0 and device 1.
    auto canAccess = cudaApi_.deviceCanAccessPeer(kDeviceA, kDeviceB);
    if (canAccess.hasError() || !canAccess.value()) {
      GTEST_SKIP() << "P2P not supported between device " << kDeviceA
                   << " and device " << kDeviceB;
    }
  }

  // Common setup: create factories, allocations, register/import segments,
  // create transports, and connect. Returns false on failure (ASSERT fires).
  struct TestEnv {
    NVLinkTransportFactory factoryA;
    NVLinkTransportFactory factoryB;
    VmmAllocation allocA;
    VmmAllocation allocB;
    std::unique_ptr<RegistrationHandle> handleA;
    std::unique_ptr<RegistrationHandle> handleB;
    std::unique_ptr<RemoteRegistrationHandle> remoteBOwner;
    NVLinkRemoteRegistrationHandle* remoteBHandle{nullptr};
    std::unique_ptr<Transport> transportA;
    std::unique_ptr<Transport> transportB;
    std::optional<RegisteredSegment> regA;
    std::optional<RemoteRegisteredSegment> remoteSegB;

    explicit TestEnv(EventBase* evb)
        : factoryA(kDeviceA, evb), factoryB(kDeviceB, evb) {}
  };

  // Initialize the full test environment for a given transfer size.
  // On failure, ASSERT macros fire and the test aborts.
  std::unique_ptr<TestEnv> makeEnv(size_t transferSize) {
    auto env = std::make_unique<TestEnv>(evbThread_.getEventBase());

    auto initA = env->allocA.init(
        driverApi_, kDeviceA, transferSize, env->factoryA.handleType());
    EXPECT_TRUE(initA.hasValue()) << initA.error().message();
    if (initA.hasError()) {
      return nullptr;
    }

    auto initB = env->allocB.init(
        driverApi_, kDeviceB, transferSize, env->factoryB.handleType());
    EXPECT_TRUE(initB.hasValue()) << initB.error().message();
    if (initB.hasError()) {
      return nullptr;
    }

    Segment segA(
        env->allocA.ptr(), env->allocA.size(), MemoryType::VRAM, kDeviceA);
    Segment segB(
        env->allocB.ptr(), env->allocB.size(), MemoryType::VRAM, kDeviceB);

    auto handleA = env->factoryA.registerSegment(segA);
    EXPECT_TRUE(handleA.hasValue()) << handleA.error().message();
    if (handleA.hasError()) {
      return nullptr;
    }
    env->handleA = std::move(handleA.value());

    auto handleB = env->factoryB.registerSegment(segB);
    EXPECT_TRUE(handleB.hasValue()) << handleB.error().message();
    if (handleB.hasError()) {
      return nullptr;
    }
    env->handleB = std::move(handleB.value());

    auto payloadB = env->handleB->serialize();
    auto remoteBResult =
        env->factoryA.importSegment(env->allocB.size(), payloadB);
    EXPECT_TRUE(remoteBResult.hasValue()) << remoteBResult.error().message();
    if (remoteBResult.hasError()) {
      return nullptr;
    }
    env->remoteBOwner = std::move(remoteBResult.value());
    env->remoteBHandle =
        dynamic_cast<NVLinkRemoteRegistrationHandle*>(env->remoteBOwner.get());
    EXPECT_NE(env->remoteBHandle, nullptr);
    if (!env->remoteBHandle) {
      return nullptr;
    }

    auto topoA = env->factoryA.getTopology();
    auto topoB = env->factoryB.getTopology();

    auto transportAResult = env->factoryA.createTransport(topoB);
    EXPECT_TRUE(transportAResult.hasValue())
        << transportAResult.error().message();
    if (transportAResult.hasError()) {
      return nullptr;
    }
    env->transportA = std::move(transportAResult.value());

    auto transportBResult = env->factoryB.createTransport(topoA);
    EXPECT_TRUE(transportBResult.hasValue())
        << transportBResult.error().message();
    if (transportBResult.hasError()) {
      return nullptr;
    }
    env->transportB = std::move(transportBResult.value());

    auto infoA = env->transportA->bind();
    auto infoB = env->transportB->bind();
    auto connA = env->transportA->connect(infoB);
    EXPECT_TRUE(connA.hasValue()) << connA.error().message();
    if (connA.hasError()) {
      return nullptr;
    }
    auto connB = env->transportB->connect(infoA);
    EXPECT_TRUE(connB.hasValue()) << connB.error().message();
    if (connB.hasError()) {
      return nullptr;
    }

    // TODO: Replace with RegisteredSegment::create() once available.
    env->regA = SegmentTest::makeRegisteredSegment(
        env->allocA.ptr(), env->allocA.size(), MemoryType::VRAM, kDeviceA);

    // TODO: Replace with Connection::recvSegment() once Phase 4 lands.
    env->remoteSegB = SegmentTest::makeRemote(
        env->remoteBHandle->mappedPtr(),
        env->remoteBHandle->mappedSize(),
        std::move(env->remoteBOwner));

    return env;
  }

  CudaApi cudaApi_;
  CudaDriverApi driverApi_;
  ScopedEventBaseThread evbThread_{"NVLinkIntegrationTest"};
};

TEST_F(NVLinkTransportIntegrationTest, PutTransfersData) {
  auto env = makeEnv(kTransferSize);
  ASSERT_NE(env, nullptr);

  // Fill device A with a known pattern.
  std::vector<uint8_t> srcHost(kTransferSize);
  fillPattern(srcHost);
  cudaApi_.setDevice(kDeviceA);
  cudaApi_.memcpyAsync(
      env->allocA.ptr(),
      srcHost.data(),
      kTransferSize,
      cudaMemcpyHostToDevice,
      nullptr);
  cudaApi_.streamSynchronize(nullptr);

  // Put local A -> remote B.
  TransferRequest req{env->regA->span(), env->remoteSegB->span()};
  auto status = env->transportA->put({&req, 1}).get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();

  // Copy device B back to host and verify.
  std::vector<uint8_t> dstHost(kTransferSize, 0);
  cudaApi_.setDevice(kDeviceB);
  cudaApi_.memcpyAsync(
      dstHost.data(),
      env->allocB.ptr(),
      kTransferSize,
      cudaMemcpyDeviceToHost,
      nullptr);
  cudaApi_.streamSynchronize(nullptr);
  EXPECT_EQ(srcHost, dstHost);
}

TEST_F(NVLinkTransportIntegrationTest, GetTransfersData) {
  auto env = makeEnv(kTransferSize);
  ASSERT_NE(env, nullptr);

  // Fill device B with a known pattern.
  std::vector<uint8_t> srcHost(kTransferSize);
  fillPattern(srcHost);
  cudaApi_.setDevice(kDeviceB);
  cudaApi_.memcpyAsync(
      env->allocB.ptr(),
      srcHost.data(),
      kTransferSize,
      cudaMemcpyHostToDevice,
      nullptr);
  cudaApi_.streamSynchronize(nullptr);

  // Get remote B -> local A.
  TransferRequest req{env->regA->span(), env->remoteSegB->span()};
  auto status = env->transportA->get({&req, 1}).get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();

  // Copy device A back to host and verify.
  std::vector<uint8_t> dstHost(kTransferSize, 0);
  cudaApi_.setDevice(kDeviceA);
  cudaApi_.memcpyAsync(
      dstHost.data(),
      env->allocA.ptr(),
      kTransferSize,
      cudaMemcpyDeviceToHost,
      nullptr);
  cudaApi_.streamSynchronize(nullptr);
  EXPECT_EQ(srcHost, dstHost);
}

TEST_F(NVLinkTransportIntegrationTest, PutMultipleRequests) {
  constexpr size_t kChunkSize = 1024;
  constexpr size_t kNumChunks = 4;
  constexpr size_t kTotalSize = kChunkSize * kNumChunks;

  auto env = makeEnv(kTotalSize);
  ASSERT_NE(env, nullptr);

  // Fill source with pattern.
  std::vector<uint8_t> srcHost(kTotalSize);
  fillPattern(srcHost);
  cudaApi_.setDevice(kDeviceA);
  cudaApi_.memcpyAsync(
      env->allocA.ptr(),
      srcHost.data(),
      kTotalSize,
      cudaMemcpyHostToDevice,
      nullptr);
  cudaApi_.streamSynchronize(nullptr);

  // Build multiple transfer requests, one per chunk.
  std::vector<TransferRequest> requests;
  requests.reserve(kNumChunks);
  for (size_t i = 0; i < kNumChunks; ++i) {
    size_t offset = i * kChunkSize;
    requests.push_back(
        TransferRequest{
            env->regA->span(offset, kChunkSize),
            env->remoteSegB->span(offset, kChunkSize)});
  }

  auto status = env->transportA->put(requests).get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();

  // Verify destination.
  std::vector<uint8_t> dstHost(kTotalSize, 0);
  cudaApi_.setDevice(kDeviceB);
  cudaApi_.memcpyAsync(
      dstHost.data(),
      env->allocB.ptr(),
      kTotalSize,
      cudaMemcpyDeviceToHost,
      nullptr);
  cudaApi_.streamSynchronize(nullptr);
  EXPECT_EQ(srcHost, dstHost);
}

TEST_F(NVLinkTransportIntegrationTest, RoundTripPutThenGet) {
  auto env = makeEnv(kTransferSize);
  ASSERT_NE(env, nullptr);

  // Fill device A with a known pattern.
  std::vector<uint8_t> srcHost(kTransferSize);
  fillPattern(srcHost);
  cudaApi_.setDevice(kDeviceA);
  cudaApi_.memcpyAsync(
      env->allocA.ptr(),
      srcHost.data(),
      kTransferSize,
      cudaMemcpyHostToDevice,
      nullptr);
  cudaApi_.streamSynchronize(nullptr);

  // Step 1: Put device A -> device B.
  {
    TransferRequest req{env->regA->span(), env->remoteSegB->span()};
    auto status = env->transportA->put({&req, 1}).get();
    ASSERT_TRUE(status.hasValue()) << status.error().message();
  }

  // Verify device B has the data.
  {
    std::vector<uint8_t> checkHost(kTransferSize, 0);
    cudaApi_.setDevice(kDeviceB);
    cudaApi_.memcpyAsync(
        checkHost.data(),
        env->allocB.ptr(),
        kTransferSize,
        cudaMemcpyDeviceToHost,
        nullptr);
    cudaApi_.streamSynchronize(nullptr);
    ASSERT_EQ(srcHost, checkHost);
  }

  // Clear device A memory so we can verify the get fills it back.
  std::vector<uint8_t> zeros(kTransferSize, 0);
  cudaApi_.setDevice(kDeviceA);
  cudaApi_.memcpyAsync(
      env->allocA.ptr(),
      zeros.data(),
      kTransferSize,
      cudaMemcpyHostToDevice,
      nullptr);
  cudaApi_.streamSynchronize(nullptr);

  // Step 2: Get device B -> device A.
  {
    TransferRequest req{env->regA->span(), env->remoteSegB->span()};
    auto status = env->transportA->get({&req, 1}).get();
    ASSERT_TRUE(status.hasValue()) << status.error().message();
  }

  // Verify device A has the original pattern back.
  std::vector<uint8_t> dstHost(kTransferSize, 0);
  cudaApi_.setDevice(kDeviceA);
  cudaApi_.memcpyAsync(
      dstHost.data(),
      env->allocA.ptr(),
      kTransferSize,
      cudaMemcpyDeviceToHost,
      nullptr);
  cudaApi_.streamSynchronize(nullptr);
  EXPECT_EQ(srcHost, dstHost);
}

} // namespace
} // namespace uniflow
