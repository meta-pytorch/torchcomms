// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/// Cross-host integration test for NVLink transport (MNNVL).
/// Requires MPI with 2 ranks on 2 different GB200 hosts in the same
/// NVLink fabric domain (nnodes=2, ppn=1).
/// Each rank creates an NVLinkTransportFactory on its local GPU (device 0)
/// and connects to the peer via MPI-exchanged topology and connection info.

#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/drivers/nvml/NvmlApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/nvlink/NVLinkRegistrationHandle.h"
#include "comms/uniflow/transport/nvlink/NVLinkTransport.h"

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace uniflow {

/// Friend-class wrapper to construct RegisteredSegment /
/// RemoteRegisteredSegment with handles for testing. The name must be exactly
/// "SegmentTest" to match the friend declaration in Segment.h.
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

/// Exchange a variable-length byte vector between rank 0 and rank 1 via MPI.
/// Each rank sends its own data and receives the peer's data.
std::vector<uint8_t> mpiExchange(
    const std::vector<uint8_t>& localData,
    int rank) {
  int peerRank = 1 - rank;

  // Exchange sizes first.
  int localSize = static_cast<int>(localData.size());
  int remoteSize = 0;
  MPI_Sendrecv(
      &localSize,
      1,
      MPI_INT,
      peerRank,
      0,
      &remoteSize,
      1,
      MPI_INT,
      peerRank,
      0,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  // Exchange payload.
  std::vector<uint8_t> remoteData(remoteSize);
  MPI_Sendrecv(
      localData.data(),
      localSize,
      MPI_BYTE,
      peerRank,
      1,
      remoteData.data(),
      remoteSize,
      MPI_BYTE,
      peerRank,
      1,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  return remoteData;
}

constexpr size_t kTransferSize = 4096;
constexpr size_t kLargeBufferSize = 12 * 1024 * 1024 + 12 * 1024; // 12MB+12KB
constexpr int kDevice = 0; // Each rank uses its local GPU 0.

class CrossHostNVLinkTest : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ASSERT_EQ(numRanks, 2)
        << "CrossHostNVLinkTest requires exactly 2 MPI ranks";

    // Verify at least 1 GPU is available on this host.
    int deviceCount = 0;
    auto err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount < 1) {
      GTEST_SKIP() << "Need at least 1 GPU, found " << deviceCount;
    }

#ifdef EXPECTED_GPU_NAME
    // Verify the GPU matches the expected type for this test target.
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, kDevice);
    if (std::string(prop.name).find(EXPECTED_GPU_NAME) == std::string::npos) {
      GTEST_SKIP() << "Expected " << EXPECTED_GPU_NAME << " GPU, found "
                   << prop.name;
    }
#endif

    cudaApi_ = std::make_shared<CudaApi>();
    driverApi_ = std::make_shared<CudaDriverApi>();
    nvmlApi_ = std::make_shared<NvmlApi>();
    evbThread_ = std::make_unique<ScopedEventBaseThread>("CrossHostNVLinkTest");
  }

  void TearDown() override {
    evbThread_.reset();
    MpiBaseTestFixture::TearDown();
  }

  struct ConnectedEnv {
    std::unique_ptr<NVLinkTransportFactory> factory;
    std::unique_ptr<Transport> transport;
    VmmAllocation alloc;
    std::unique_ptr<RegistrationHandle> localHandle;
    std::unique_ptr<RemoteRegistrationHandle> remoteHandle;
    std::optional<RegisteredSegment> localReg;
    std::optional<RemoteRegisteredSegment> remoteReg;
  };

  /// Create factories, exchange topology, connect transports, allocate
  /// VMM memory, register/import segments, and exchange handle info
  /// across hosts via MPI.
  std::unique_ptr<ConnectedEnv> connectCrossHost(size_t transferSize) {
    auto env = std::make_unique<ConnectedEnv>();
    auto* evb = evbThread_->getEventBase();

    // Create factory on this rank's local GPU.
    env->factory = std::make_unique<NVLinkTransportFactory>(
        kDevice, evb, nvmlApi_, cudaApi_, driverApi_);

    // Allocate VMM GPU memory with the required handle type.
    auto initStatus = env->alloc.init(
        *driverApi_, kDevice, transferSize, env->factory->handleType());
    EXPECT_TRUE(initStatus.hasValue()) << initStatus.error().message();
    if (initStatus.hasError()) {
      return nullptr;
    }

    // Register local segment.
    Segment localSeg(
        env->alloc.ptr(), env->alloc.size(), MemoryType::VRAM, kDevice);
    auto regResult = env->factory->registerSegment(localSeg);
    EXPECT_TRUE(regResult.hasValue()) << regResult.error().message();
    if (regResult.hasError()) {
      return nullptr;
    }
    env->localHandle = std::move(regResult.value());

    // Exchange serialized handles via MPI to import peer's segment.
    auto localPayload = env->localHandle->serialize();
    auto remotePayload = mpiExchange(localPayload, globalRank);

    auto importResult =
        env->factory->importSegment(env->alloc.size(), remotePayload);
    EXPECT_TRUE(importResult.hasValue()) << importResult.error().message();
    if (importResult.hasError()) {
      return nullptr;
    }
    env->remoteHandle = std::move(importResult.value());

    // Exchange topology and create transports.
    auto localTopo = env->factory->getTopology();
    auto remoteTopo = mpiExchange(localTopo, globalRank);

    auto transportResult = env->factory->createTransport(remoteTopo);
    EXPECT_TRUE(transportResult.hasValue())
        << transportResult.error().message();
    if (transportResult.hasError()) {
      return nullptr;
    }
    env->transport = std::move(transportResult.value());

    // Exchange TransportInfo via MPI and connect.
    auto localInfo = env->transport->bind();
    auto remoteInfo = mpiExchange(localInfo, globalRank);
    auto connectStatus = env->transport->connect(remoteInfo);
    EXPECT_TRUE(connectStatus.hasValue()) << connectStatus.error().message();
    if (connectStatus.hasError()) {
      return nullptr;
    }

    // Build local RegisteredSegment.
    env->localReg = SegmentTest::makeRegisteredSegment(
        env->alloc.ptr(), env->alloc.size(), MemoryType::VRAM, kDevice);

    // Build remote segment using a fake VA — the transport must use
    // mappedPtr from the imported handle, not this buf_ address.
    auto* nvlinkRemote =
        dynamic_cast<NVLinkRemoteRegistrationHandle*>(env->remoteHandle.get());
    EXPECT_NE(nvlinkRemote, nullptr);
    if (!nvlinkRemote) {
      return nullptr;
    }

    env->remoteReg = SegmentTest::makeRemote(
        nvlinkRemote->mappedPtr(),
        nvlinkRemote->mappedSize(),
        std::move(env->remoteHandle));

    return env;
  }

  std::shared_ptr<CudaApi> cudaApi_;
  std::shared_ptr<CudaDriverApi> driverApi_;
  std::shared_ptr<NvmlApi> nvmlApi_;
  std::unique_ptr<ScopedEventBaseThread> evbThread_;
};

// ---------------------------------------------------------------------------
// Connection test
// ---------------------------------------------------------------------------

TEST_F(CrossHostNVLinkTest, TransportsConnectAcrossHosts) {
  auto env = connectCrossHost(kTransferSize);
  ASSERT_NE(env, nullptr);
  // If we got here, both ranks successfully created factories in the same
  // MNNVL domain and connected transports across hosts.
}

// ---------------------------------------------------------------------------
// Data transfer tests
// ---------------------------------------------------------------------------

TEST_F(CrossHostNVLinkTest, PutTransfersData) {
  auto env = connectCrossHost(kTransferSize);
  ASSERT_NE(env, nullptr);

  // Rank 0 fills its GPU with a known pattern; rank 1 zeroes its GPU.
  std::vector<uint8_t> srcHost(kTransferSize);
  if (globalRank == 0) {
    fillPattern(srcHost);
  } else {
    std::memset(srcHost.data(), 0, kTransferSize);
  }
  cudaApi_->setDevice(kDevice);
  cudaApi_->memcpyAsync(
      env->alloc.ptr(),
      srcHost.data(),
      kTransferSize,
      cudaMemcpyHostToDevice,
      nullptr);
  cudaApi_->streamSynchronize(nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 0 puts its data to rank 1's remote segment.
  if (globalRank == 0) {
    TransferRequest req{env->localReg->span(), env->remoteReg->span()};
    auto status = env->transport->put({&req, 1}).get();
    ASSERT_TRUE(status.hasValue()) << status.error().message();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies it received the correct data.
  if (globalRank == 1) {
    std::vector<uint8_t> dstHost(kTransferSize, 0);
    cudaApi_->setDevice(kDevice);
    cudaApi_->memcpyAsync(
        dstHost.data(),
        env->alloc.ptr(),
        kTransferSize,
        cudaMemcpyDeviceToHost,
        nullptr);
    cudaApi_->streamSynchronize(nullptr);

    std::vector<uint8_t> expected(kTransferSize);
    fillPattern(expected);
    EXPECT_EQ(expected, dstHost);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(CrossHostNVLinkTest, GetTransfersData) {
  auto env = connectCrossHost(kTransferSize);
  ASSERT_NE(env, nullptr);

  // Rank 1 fills its GPU with a known pattern; rank 0 zeroes its GPU.
  std::vector<uint8_t> srcHost(kTransferSize);
  if (globalRank == 1) {
    fillPattern(srcHost);
  } else {
    std::memset(srcHost.data(), 0, kTransferSize);
  }
  cudaApi_->setDevice(kDevice);
  cudaApi_->memcpyAsync(
      env->alloc.ptr(),
      srcHost.data(),
      kTransferSize,
      cudaMemcpyHostToDevice,
      nullptr);
  cudaApi_->streamSynchronize(nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 0 gets data from rank 1's remote segment.
  if (globalRank == 0) {
    TransferRequest req{env->localReg->span(), env->remoteReg->span()};
    auto status = env->transport->get({&req, 1}).get();
    ASSERT_TRUE(status.hasValue()) << status.error().message();

    // Verify rank 0's GPU now has the pattern.
    std::vector<uint8_t> dstHost(kTransferSize, 0);
    cudaApi_->setDevice(kDevice);
    cudaApi_->memcpyAsync(
        dstHost.data(),
        env->alloc.ptr(),
        kTransferSize,
        cudaMemcpyDeviceToHost,
        nullptr);
    cudaApi_->streamSynchronize(nullptr);

    std::vector<uint8_t> expected(kTransferSize);
    fillPattern(expected);
    EXPECT_EQ(expected, dstHost);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(CrossHostNVLinkTest, PutLargeBuffer) {
  auto env = connectCrossHost(kLargeBufferSize);
  ASSERT_NE(env, nullptr);

  std::vector<uint8_t> srcHost(kLargeBufferSize);
  if (globalRank == 0) {
    fillPattern(srcHost);
  } else {
    std::memset(srcHost.data(), 0, kLargeBufferSize);
  }
  cudaApi_->setDevice(kDevice);
  cudaApi_->memcpyAsync(
      env->alloc.ptr(),
      srcHost.data(),
      kLargeBufferSize,
      cudaMemcpyHostToDevice,
      nullptr);
  cudaApi_->streamSynchronize(nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == 0) {
    TransferRequest req{env->localReg->span(), env->remoteReg->span()};
    auto status = env->transport->put({&req, 1}).get();
    ASSERT_TRUE(status.hasValue()) << status.error().message();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == 1) {
    std::vector<uint8_t> dstHost(kLargeBufferSize, 0);
    cudaApi_->setDevice(kDevice);
    cudaApi_->memcpyAsync(
        dstHost.data(),
        env->alloc.ptr(),
        kLargeBufferSize,
        cudaMemcpyDeviceToHost,
        nullptr);
    cudaApi_->streamSynchronize(nullptr);

    std::vector<uint8_t> expected(kLargeBufferSize);
    fillPattern(expected);
    EXPECT_EQ(expected, dstHost);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(CrossHostNVLinkTest, RoundTripPutThenGet) {
  auto env = connectCrossHost(kTransferSize);
  ASSERT_NE(env, nullptr);

  // Rank 0 fills its GPU with a known pattern.
  std::vector<uint8_t> srcHost(kTransferSize);
  fillPattern(srcHost);

  if (globalRank == 0) {
    cudaApi_->setDevice(kDevice);
    cudaApi_->memcpyAsync(
        env->alloc.ptr(),
        srcHost.data(),
        kTransferSize,
        cudaMemcpyHostToDevice,
        nullptr);
    cudaApi_->streamSynchronize(nullptr);
  } else {
    // Rank 1 zeroes its GPU.
    std::vector<uint8_t> zeros(kTransferSize, 0);
    cudaApi_->setDevice(kDevice);
    cudaApi_->memcpyAsync(
        env->alloc.ptr(),
        zeros.data(),
        kTransferSize,
        cudaMemcpyHostToDevice,
        nullptr);
    cudaApi_->streamSynchronize(nullptr);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Step 1: Rank 0 puts A -> B.
  if (globalRank == 0) {
    TransferRequest req{env->localReg->span(), env->remoteReg->span()};
    auto status = env->transport->put({&req, 1}).get();
    ASSERT_TRUE(status.hasValue()) << status.error().message();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify rank 1 has the data.
  if (globalRank == 1) {
    std::vector<uint8_t> checkHost(kTransferSize, 0);
    cudaApi_->setDevice(kDevice);
    cudaApi_->memcpyAsync(
        checkHost.data(),
        env->alloc.ptr(),
        kTransferSize,
        cudaMemcpyDeviceToHost,
        nullptr);
    cudaApi_->streamSynchronize(nullptr);
    EXPECT_EQ(srcHost, checkHost);
  }

  // Clear rank 0's GPU memory so we can verify the get fills it back.
  if (globalRank == 0) {
    std::vector<uint8_t> zeros(kTransferSize, 0);
    cudaApi_->setDevice(kDevice);
    cudaApi_->memcpyAsync(
        env->alloc.ptr(),
        zeros.data(),
        kTransferSize,
        cudaMemcpyHostToDevice,
        nullptr);
    cudaApi_->streamSynchronize(nullptr);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Step 2: Rank 0 gets B -> A.
  if (globalRank == 0) {
    TransferRequest req{env->localReg->span(), env->remoteReg->span()};
    auto status = env->transport->get({&req, 1}).get();
    ASSERT_TRUE(status.hasValue()) << status.error().message();

    // Verify rank 0 has the original pattern back.
    std::vector<uint8_t> dstHost(kTransferSize, 0);
    cudaApi_->setDevice(kDevice);
    cudaApi_->memcpyAsync(
        dstHost.data(),
        env->alloc.ptr(),
        kTransferSize,
        cudaMemcpyDeviceToHost,
        nullptr);
    cudaApi_->streamSynchronize(nullptr);
    EXPECT_EQ(srcHost, dstHost);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace
} // namespace uniflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase());
  return RUN_ALL_TESTS();
}
