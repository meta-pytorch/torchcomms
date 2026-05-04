// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/// Cross-host integration test for MultiTransport put/get.
/// Requires MPI with 2 ranks on 2 different hosts (nnodes=2, ppn=1).
/// Each rank creates a MultiTransportFactory, exchanges topology and
/// connection info via MPI, then tests DRAM and GPU put/get transfers.

#include <mpi.h>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/transport/Topology.h"

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace uniflow {

/// Friend-class wrapper to construct RegisteredSegment /
/// RemoteRegisteredSegment with handles for testing.
class SegmentTest {
 public:
  static RegisteredSegment makeRegistered(
      Segment& segment,
      std::vector<std::unique_ptr<RegistrationHandle>> handles) {
    RegisteredSegment reg(segment);
    reg.handles_ = std::move(handles);
    return reg;
  }

  static RemoteRegisteredSegment makeRemote(
      void* buf,
      size_t len,
      MemoryType memType,
      int deviceId,
      std::vector<std::unique_ptr<RemoteRegistrationHandle>> handles) {
    RemoteRegisteredSegment remote(buf, len, memType, deviceId);
    remote.handles_ = std::move(handles);
    return remote;
  }
};

/// Exchange a variable-length byte vector between rank 0 and rank 1 via MPI.
static std::vector<uint8_t> mpiExchange(
    const std::vector<uint8_t>& localData,
    int rank) {
  int peerRank = 1 - rank;

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

/// Returns true if any rank wants to skip (synchronized across all ranks).
static bool anyRankWantsToSkip(bool localSkip) {
  int localVal = localSkip ? 1 : 0;
  int globalVal = 0;
  MPI_Allreduce(&localVal, &globalVal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return globalVal != 0;
}

class MultiTransportCrossHostTest : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ASSERT_EQ(numRanks, 2)
        << "MultiTransportCrossHostTest requires exactly 2 MPI ranks";

    auto& topo = Topology::get();
    ASSERT_TRUE(topo.available());
    ASSERT_GT(topo.nicCount(), 0u) << "Need at least 1 NIC";
  }

  struct ConnectedPair {
    std::unique_ptr<MultiTransportFactory> factory;
    std::unique_ptr<MultiTransport> transport;
  };

  struct SegmentRegistration {
    RegisteredSegment local;
    RemoteRegisteredSegment remote;
  };

  /// Create a connected MultiTransport pair across hosts using MPI to
  /// exchange topology and connection info.
  ConnectedPair connectCrossHost(int deviceId) {
    ConnectedPair pair;
    pair.factory = std::make_unique<MultiTransportFactory>(deviceId);

    // Exchange topology via MPI.
    auto localTopo = pair.factory->getTopology();
    auto remoteTopo = mpiExchange(localTopo, globalRank);

    auto transportResult = pair.factory->createTransport(remoteTopo);
    EXPECT_TRUE(transportResult.hasValue())
        << "createTransport failed: " << transportResult.error().message();
    pair.transport = std::move(transportResult.value());

    // Exchange TransportInfo via MPI.
    auto bindResult = pair.transport->bind();
    EXPECT_TRUE(bindResult.hasValue())
        << "bind failed: " << bindResult.error().message();
    auto remoteInfo = mpiExchange(bindResult.value(), globalRank);
    auto connectStatus = pair.transport->connect(remoteInfo);
    EXPECT_FALSE(connectStatus.hasError())
        << "connect failed: " << connectStatus.error().message();

    return pair;
  }

  /// Register a local segment, exchange registration payloads via MPI,
  /// import the remote segment, and return both registered segments.
  std::optional<SegmentRegistration> registerAndExchangeSegments(
      MultiTransportFactory& factory,
      void* buf,
      size_t totalSize,
      MemoryType memType,
      int deviceId = -1) {
    Segment seg(buf, totalSize, memType, deviceId);
    auto regResult = factory.registerSegment(seg);
    EXPECT_TRUE(regResult.hasValue()) << regResult.error().message();
    if (regResult.hasError()) {
      return std::nullopt;
    }

    auto localPayload = regResult.value().exportId().value();
    auto remotePayload = mpiExchange(localPayload, globalRank);

    auto importResult = factory.importSegment(remotePayload);
    EXPECT_TRUE(importResult.hasValue()) << importResult.error().message();
    if (importResult.hasError()) {
      return std::nullopt;
    }

    return SegmentRegistration{
        std::move(regResult.value()),
        std::move(importResult.value()),
    };
  }

  /// Build a vector of TransferRequests, one per chunk of bufSize.
  static std::vector<TransferRequest> buildTransferRequests(
      RegisteredSegment& local,
      RemoteRegisteredSegment& remote,
      size_t bufSize,
      size_t numRequests) {
    std::vector<TransferRequest> reqs;
    reqs.reserve(numRequests);
    for (size_t r = 0; r < numRequests; ++r) {
      reqs.push_back(
          TransferRequest{
              .local = local.span(r * bufSize, bufSize),
              .remote = remote.span(r * bufSize, bufSize),
          });
    }
    return reqs;
  }
};

// --- Transfer test parameterization ---

enum class TransferOp { Put, Get };

// Memory type for test parameterization.
// Combines memory location (CPU vs GPU) with GPU allocation method.
enum class MemType {
  Dram, // CPU memory (std::vector)
  CudaMalloc, // GPU memory via cudaMalloc
  Fabric, // GPU memory via cuMem VMM with CU_MEM_HANDLE_TYPE_FABRIC
};

struct CrossHostTransferParam {
  size_t bufSize;
  size_t numRequests;
  TransferOp op;
  MemType localMemType; // rank 0's memory type
  MemType remoteMemType; // rank 1's memory type
  std::string name;
};

std::string crossHostParamName(
    const ::testing::TestParamInfo<CrossHostTransferParam>& info) {
  return info.param.name;
}

bool isGpu(MemType t) {
  return t == MemType::CudaMalloc || t == MemType::Fabric;
}

bool isFabric(MemType t) {
  return t == MemType::Fabric;
}

const size_t kLargeBufferSize = 12 * 1024 * 1024 + 12 * 1024; // 12MB + 12KB

// --- Helper types ---

struct CudaBuffer {
  void* ptr{nullptr};
  size_t size{0};

  explicit CudaBuffer(size_t n, int device = 0) : size(n) {
    cudaSetDevice(device);
    if (cudaMalloc(&ptr, n) != cudaSuccess) {
      ptr = nullptr;
    }
  }

  ~CudaBuffer() {
    if (ptr) {
      cudaFree(ptr);
    }
  }

  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
};

// RAII wrapper for cuMem VMM GPU memory allocation.
// Uses cuMemCreate with CU_MEM_HANDLE_TYPE_FABRIC for MNNVL.
class VmmAllocation {
 public:
  VmmAllocation() = default;

  Status alloc(CudaDriverApi& driverApi, int deviceId, size_t requestedSize) {
    driverApi_ = &driverApi;

    CUdevice device;
    CHECK_RETURN(driverApi_->cuDeviceGet(&device, deviceId));

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

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

// --- Same-device transfer tests (parameterized by mem type + alloc type) ---

class SameDeviceTransferTest
    : public MultiTransportCrossHostTest,
      public ::testing::WithParamInterface<CrossHostTransferParam> {
 protected:
  void SetUp() override {
    MultiTransportCrossHostTest::SetUp();
    driverApi_ = std::make_shared<CudaDriverApi>();
    auto initStatus = driverApi_->init();
    driverApiAvailable_ = initStatus.hasValue();
  }

  // Allocate GPU memory for the given MemType. For CudaMalloc, allocates into
  // cudaBuf. For Fabric, allocates into vmmBuf. Returns the data pointer.
  // For Dram, returns nullptr (caller should use cpuBuf directly).
  void* allocGpuBuffer(
      MemType memType,
      int cudaDev,
      size_t totalSize,
      CudaBuffer& cudaBuf,
      VmmAllocation& vmmBuf) {
    if (memType == MemType::Fabric) {
      auto st = vmmBuf.alloc(*driverApi_, cudaDev, totalSize);
      return st.hasValue() ? vmmBuf.ptr() : nullptr;
    }
    // CudaMalloc — cudaBuf already constructed by caller.
    return cudaBuf.ptr;
  }

  // Fill buffer with pattern (GPU buffers go through a staging copy).
  void fillBuffer(
      void* ptr,
      MemType memType,
      int cudaDev,
      size_t bufSize,
      size_t numRequests,
      uint8_t patternBase) {
    size_t totalSize = bufSize * numRequests;
    if (isGpu(memType)) {
      std::vector<char> staging(totalSize);
      for (size_t r = 0; r < numRequests; ++r) {
        std::memset(
            staging.data() + r * bufSize,
            static_cast<int>(patternBase + r),
            bufSize);
      }
      cudaSetDevice(cudaDev);
      cudaMemcpy(ptr, staging.data(), totalSize, cudaMemcpyHostToDevice);
    } else {
      for (size_t r = 0; r < numRequests; ++r) {
        std::memset(
            static_cast<char*>(ptr) + r * bufSize,
            static_cast<int>(patternBase + r),
            bufSize);
      }
    }
  }

  // Zero the buffer.
  void zeroBuffer(void* ptr, MemType memType, int cudaDev, size_t totalSize) {
    if (isGpu(memType)) {
      cudaSetDevice(cudaDev);
      cudaMemset(ptr, 0, totalSize);
    } else {
      std::memset(ptr, 0, totalSize);
    }
  }

  // Read buffer contents into a host vector for verification.
  std::vector<char>
  readBuffer(void* ptr, MemType memType, int cudaDev, size_t totalSize) {
    std::vector<char> out(totalSize);
    if (isGpu(memType)) {
      cudaSetDevice(cudaDev);
      cudaMemcpy(out.data(), ptr, totalSize, cudaMemcpyDeviceToHost);
    } else {
      std::memcpy(out.data(), ptr, totalSize);
    }
    return out;
  }

  uniflow::MemoryType toSegmentMemType(MemType t) {
    return isGpu(t) ? MemoryType::VRAM : MemoryType::DRAM;
  }

  std::shared_ptr<CudaDriverApi> driverApi_;
  bool driverApiAvailable_{false};
};

TEST_P(SameDeviceTransferTest, Transfer) {
  const auto& param = GetParam();
  const bool needsCuda =
      isGpu(param.localMemType) || isGpu(param.remoteMemType);
  const bool needsFabric =
      isFabric(param.localMemType) || isFabric(param.remoteMemType);

  if (needsCuda) {
    int deviceCount = 0;
    bool noCuda =
        cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 1;
    if (anyRankWantsToSkip(noCuda)) {
      GTEST_SKIP() << "Some rank lacks CUDA devices (local: " << deviceCount
                   << ")";
    }
  }

  if (needsFabric) {
    bool canFabric = driverApiAvailable_;
    if (canFabric) {
      VmmAllocation probe;
      canFabric = probe.alloc(*driverApi_, 0, 4096).hasValue();
    }
    if (anyRankWantsToSkip(!canFabric)) {
      GTEST_SKIP() << "Fabric (MNNVL) not supported on some rank";
    }
  }

  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  const bool isPut = param.op == TransferOp::Put;

  // Use GPU factory if either side needs VRAM, CPU factory otherwise.
  const int cudaDev = 0;
  const int factoryDevice = needsCuda ? cudaDev : -1;
  auto pair = connectCrossHost(factoryDevice);

  // Determine which MemType this rank uses.
  MemType myMemType =
      (globalRank == 0) ? param.localMemType : param.remoteMemType;

  // Allocate buffer for this rank.
  std::vector<char> cpuBuf;
  CudaBuffer cudaBuf(myMemType == MemType::CudaMalloc ? totalSize : 0, cudaDev);
  VmmAllocation vmmBuf;
  void* myPtr = nullptr;
  if (myMemType == MemType::Dram) {
    cpuBuf.resize(totalSize);
    myPtr = cpuBuf.data();
  } else {
    myPtr = allocGpuBuffer(myMemType, cudaDev, totalSize, cudaBuf, vmmBuf);
  }
  ASSERT_NE(myPtr, nullptr)
      << "Buffer allocation failed on rank " << globalRank;

  const int fillRank = isPut ? 0 : 1;
  const int verifyRank = isPut ? 1 : 0;
  const uint8_t patternBase = 0xA0;

  if (globalRank == fillRank) {
    fillBuffer(myPtr, myMemType, cudaDev, bufSize, numRequests, patternBase);
  } else {
    zeroBuffer(myPtr, myMemType, cudaDev, totalSize);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  int segDeviceId = isGpu(myMemType) ? cudaDev : -1;
  auto segments = registerAndExchangeSegments(
      *pair.factory,
      myPtr,
      totalSize,
      toSegmentMemType(myMemType),
      segDeviceId);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    auto reqs = buildTransferRequests(
        segments->local, segments->remote, bufSize, numRequests);
    auto status = isPut ? pair.transport->put(reqs).get()
                        : pair.transport->get(reqs).get();
    ASSERT_FALSE(status.hasError())
        << (isPut ? "put" : "get") << " failed: " << status.error().message();

    // VRAM→VRAM with Fabric → NVLink; everything else → RDMA.
    bool expectNvlink = param.localMemType == MemType::Fabric &&
        param.remoteMemType == MemType::Fabric;
    auto expectedTransport =
        expectNvlink ? TransportType::NVLink : TransportType::RDMA;
    EXPECT_EQ(pair.transport->transferCount(expectedTransport), 1u);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == verifyRank) {
    auto verify = readBuffer(myPtr, myMemType, cudaDev, totalSize);
    for (size_t r = 0; r < numRequests; ++r) {
      uint8_t expected = static_cast<uint8_t>(patternBase + r);
      for (size_t i = 0; i < bufSize; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(verify[r * bufSize + i]), expected)
            << "Data mismatch at request " << r << " byte " << i;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    CrossHostTransfer,
    SameDeviceTransferTest,
    ::testing::Values(
        // DRAM → DRAM
        CrossHostTransferParam{4096, 1, TransferOp::Put, MemType::Dram, MemType::Dram, "DRAM_DRAM_Put_4KB"},
        CrossHostTransferParam{4096, 1, TransferOp::Get, MemType::Dram, MemType::Dram, "DRAM_DRAM_Get_4KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Put, MemType::Dram, MemType::Dram, "DRAM_DRAM_Put_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Get, MemType::Dram, MemType::Dram, "DRAM_DRAM_Get_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 4, TransferOp::Put, MemType::Dram, MemType::Dram, "DRAM_DRAM_Put_12MB12KB_batch"},
        CrossHostTransferParam{kLargeBufferSize, 4, TransferOp::Get, MemType::Dram, MemType::Dram, "DRAM_DRAM_Get_12MB12KB_batch"},
        // VRAM → VRAM (cudaMalloc)
        CrossHostTransferParam{4096, 1, TransferOp::Put, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Put_4KB"},
        CrossHostTransferParam{4096, 1, TransferOp::Get, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Get_4KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Put, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Put_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Get, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Get_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 4, TransferOp::Put, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Put_12MB12KB_batch"},
        CrossHostTransferParam{kLargeBufferSize, 4, TransferOp::Get, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Get_12MB12KB_batch"},
        // VRAM → VRAM (Fabric / MNNVL)
        CrossHostTransferParam{4096, 1, TransferOp::Put, MemType::Fabric, MemType::Fabric, "Fabric_Put_4KB"},
        CrossHostTransferParam{4096, 1, TransferOp::Get, MemType::Fabric, MemType::Fabric, "Fabric_Get_4KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Put, MemType::Fabric, MemType::Fabric, "Fabric_Put_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Get, MemType::Fabric, MemType::Fabric, "Fabric_Get_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 4, TransferOp::Put, MemType::Fabric, MemType::Fabric, "Fabric_Put_12MB12KB_batch"},
        CrossHostTransferParam{kLargeBufferSize, 4, TransferOp::Get, MemType::Fabric, MemType::Fabric, "Fabric_Get_12MB12KB_batch"},
        // VRAM → DRAM (mixed)
        CrossHostTransferParam{4096, 1, TransferOp::Put, MemType::CudaMalloc, MemType::Dram, "VRAM_DRAM_Put_4KB"},
        CrossHostTransferParam{4096, 1, TransferOp::Get, MemType::CudaMalloc, MemType::Dram, "VRAM_DRAM_Get_4KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Put, MemType::CudaMalloc, MemType::Dram, "VRAM_DRAM_Put_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Get, MemType::CudaMalloc, MemType::Dram, "VRAM_DRAM_Get_12MB12KB"},
        // DRAM → VRAM (mixed)
        CrossHostTransferParam{4096, 1, TransferOp::Put, MemType::Dram, MemType::CudaMalloc, "DRAM_VRAM_Put_4KB"},
        CrossHostTransferParam{4096, 1, TransferOp::Get, MemType::Dram, MemType::CudaMalloc, "DRAM_VRAM_Get_4KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Put, MemType::Dram, MemType::CudaMalloc, "DRAM_VRAM_Put_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Get, MemType::Dram, MemType::CudaMalloc, "DRAM_VRAM_Get_12MB12KB"}),
    crossHostParamName);
// clang-format on

// --- VRAM→VRAM cross-device tests (parameterized by alloc type) ---
// Rank 0 uses GPU 0, rank 1 uses GPU 1. cudaMalloc → RDMA, Fabric → NVLink.

class CrossDeviceGpuTransferTest
    : public MultiTransportCrossHostTest,
      public ::testing::WithParamInterface<CrossHostTransferParam> {
 protected:
  void SetUp() override {
    MultiTransportCrossHostTest::SetUp();
    driverApi_ = std::make_shared<CudaDriverApi>();
    auto initStatus = driverApi_->init();
    driverApiAvailable_ = initStatus.hasValue();
  }

  std::shared_ptr<CudaDriverApi> driverApi_;
  bool driverApiAvailable_{false};
};

TEST_P(CrossDeviceGpuTransferTest, Transfer) {
  int deviceCount = 0;
  bool noCuda =
      cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 2;
  if (anyRankWantsToSkip(noCuda)) {
    GTEST_SKIP() << "Need at least 2 CUDA devices per rank (local: "
                 << deviceCount << ")";
  }

  const auto& param = GetParam();
  // Cross-device uses localMemType for the GPU alloc method on both ranks.
  const bool useFabric = isFabric(param.localMemType);

  if (useFabric) {
    bool canFabric = driverApiAvailable_;
    if (canFabric) {
      VmmAllocation probe;
      canFabric = probe.alloc(*driverApi_, globalRank, 4096).hasValue();
    }
    if (anyRankWantsToSkip(!canFabric)) {
      GTEST_SKIP() << "Fabric (MNNVL) not supported on some rank";
    }
  }

  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  const bool isPut = param.op == TransferOp::Put;

  const int cudaDev = globalRank;
  auto pair = connectCrossHost(cudaDev);

  CudaBuffer cudaBuf(useFabric ? 0 : totalSize, cudaDev);
  VmmAllocation vmmBuf;
  void* gpuPtr = nullptr;
  if (useFabric) {
    auto allocStatus = vmmBuf.alloc(*driverApi_, cudaDev, totalSize);
    ASSERT_TRUE(allocStatus.hasValue())
        << "Fabric VMM alloc failed: " << allocStatus.error().message();
    gpuPtr = vmmBuf.ptr();
  } else {
    ASSERT_NE(cudaBuf.ptr, nullptr)
        << "cudaMalloc failed on device " << cudaDev;
    gpuPtr = cudaBuf.ptr;
  }

  const int fillRank = isPut ? 0 : 1;
  const int verifyRank = isPut ? 1 : 0;
  const uint8_t patternBase = 0xD0;

  cudaSetDevice(cudaDev);
  if (globalRank == fillRank) {
    std::vector<char> staging(totalSize);
    for (size_t r = 0; r < numRequests; ++r) {
      std::memset(
          staging.data() + r * bufSize,
          static_cast<int>(patternBase + r),
          bufSize);
    }
    cudaMemcpy(gpuPtr, staging.data(), totalSize, cudaMemcpyHostToDevice);
  } else {
    cudaMemset(gpuPtr, 0, totalSize);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  auto segments = registerAndExchangeSegments(
      *pair.factory, gpuPtr, totalSize, MemoryType::VRAM, cudaDev);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    auto reqs = buildTransferRequests(
        segments->local, segments->remote, bufSize, numRequests);
    auto status = isPut ? pair.transport->put(reqs).get()
                        : pair.transport->get(reqs).get();
    ASSERT_FALSE(status.hasError())
        << "Cross-device " << (isPut ? "put" : "get")
        << " failed: " << status.error().message();

    auto expectedTransport =
        useFabric ? TransportType::NVLink : TransportType::RDMA;
    EXPECT_EQ(pair.transport->transferCount(expectedTransport), 1u);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == verifyRank) {
    std::vector<char> verify(totalSize, 0);
    cudaSetDevice(cudaDev);
    cudaMemcpy(verify.data(), gpuPtr, totalSize, cudaMemcpyDeviceToHost);
    for (size_t r = 0; r < numRequests; ++r) {
      uint8_t expected = static_cast<uint8_t>(patternBase + r);
      for (size_t i = 0; i < bufSize; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(verify[r * bufSize + i]), expected)
            << "Cross-device data mismatch at request " << r << " byte " << i;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    CrossDeviceGpuTransfer,
    CrossDeviceGpuTransferTest,
    ::testing::Values(
        // cudaMalloc
        CrossHostTransferParam{4096, 1, TransferOp::Put, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Put_4KB"},
        CrossHostTransferParam{4096, 1, TransferOp::Get, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Get_4KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Put, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Put_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Get, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Get_12MB12KB"},
        // Fabric (MNNVL)
        CrossHostTransferParam{4096, 1, TransferOp::Put, MemType::Fabric, MemType::Fabric, "Fabric_Put_4KB"},
        CrossHostTransferParam{4096, 1, TransferOp::Get, MemType::Fabric, MemType::Fabric, "Fabric_Get_4KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Put, MemType::Fabric, MemType::Fabric, "Fabric_Put_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Get, MemType::Fabric, MemType::Fabric, "Fabric_Get_12MB12KB"}),
    crossHostParamName);
// clang-format on

} // namespace uniflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase());
  return RUN_ALL_TESTS();
}
