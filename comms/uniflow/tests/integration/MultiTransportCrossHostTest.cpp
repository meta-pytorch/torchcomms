// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/// Cross-host integration test for MultiTransport put/get.
/// Requires 2 ranks on 2 different hosts (nnodes=2, ppn=1).
/// Each rank creates a MultiTransportFactory, exchanges topology and
/// connection info via TCP, then tests DRAM and GPU put/get transfers.
/// Uses MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE env vars set by re_launcher.

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/transport/Topology.h"

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

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

// --- TCP-based inter-rank communication (replaces MPI) ---

class TestEnv : public ::testing::Environment {
 public:
  void SetUp() override {
    const char* rankStr = std::getenv("RANK");
    if (!rankStr)
      rankStr = std::getenv("OMPI_COMM_WORLD_RANK");
    const char* sizeStr = std::getenv("WORLD_SIZE");
    if (!sizeStr)
      sizeStr = std::getenv("OMPI_COMM_WORLD_SIZE");
    const char* addr = std::getenv("MASTER_ADDR");
    const char* portStr = std::getenv("MASTER_PORT");

    if (!rankStr || !sizeStr || !addr || !portStr) {
      throw std::runtime_error(
          "Missing required env vars: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT");
    }

    rank_ = std::atoi(rankStr);
    worldSize_ = std::atoi(sizeStr);
    if (worldSize_ != 2) {
      throw std::runtime_error(
          "MultiTransportCrossHostTest requires exactly 2 ranks");
    }

    int port = std::atoi(portStr);

    if (rank_ == 0) {
      int listenFd = ::socket(AF_INET6, SOCK_STREAM, 0);
      if (listenFd < 0)
        throw std::runtime_error("socket() failed");
      int opt = 1;
      ::setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
      int off = 0;
      ::setsockopt(listenFd, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off));

      struct sockaddr_in6 sa{};
      sa.sin6_family = AF_INET6;
      sa.sin6_addr = in6addr_any;
      sa.sin6_port = htons(static_cast<uint16_t>(port));

      if (::bind(listenFd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) < 0) {
        ::close(listenFd);
        throw std::runtime_error(
            "bind() failed on port " + std::to_string(port));
      }
      if (::listen(listenFd, 1) < 0) {
        ::close(listenFd);
        throw std::runtime_error("listen() failed");
      }
      peerFd_ = ::accept(listenFd, nullptr, nullptr);
      ::close(listenFd);
      if (peerFd_ < 0)
        throw std::runtime_error("accept() failed");
    } else {
      struct addrinfo hints{};
      hints.ai_family = AF_UNSPEC;
      hints.ai_socktype = SOCK_STREAM;
      struct addrinfo* res = nullptr;
      std::string portS = std::to_string(port);
      if (int r = ::getaddrinfo(addr, portS.c_str(), &hints, &res); r != 0) {
        throw std::runtime_error(
            std::string("getaddrinfo: ") + gai_strerror(r));
      }

      peerFd_ = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
      if (peerFd_ < 0) {
        ::freeaddrinfo(res);
        throw std::runtime_error("socket() failed");
      }

      for (int i = 0; i < 50; ++i) {
        if (::connect(peerFd_, res->ai_addr, res->ai_addrlen) == 0)
          break;
        if (i == 49) {
          ::freeaddrinfo(res);
          ::close(peerFd_);
          throw std::runtime_error(
              std::string("connect to ") + addr + ":" + portS + " failed");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      ::freeaddrinfo(res);
    }

    int nodelay = 1;
    ::setsockopt(peerFd_, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
  }

  void TearDown() override {
    if (peerFd_ >= 0) {
      ::close(peerFd_);
      peerFd_ = -1;
    }
  }

  static int rank() {
    return rank_;
  }
  static int worldSize() {
    return worldSize_;
  }
  static int peerFd() {
    return peerFd_;
  }

 private:
  static int rank_;
  static int worldSize_;
  static int peerFd_;
};

int TestEnv::rank_ = -1;
int TestEnv::worldSize_ = -1;
int TestEnv::peerFd_ = -1;

static void sendAll(int fd, const void* buf, size_t len) {
  auto* p = static_cast<const char*>(buf);
  while (len > 0) {
    ssize_t n = ::send(fd, p, len, 0);
    if (n <= 0)
      throw std::runtime_error("TCP send failed");
    p += n;
    len -= static_cast<size_t>(n);
  }
}

static void recvAll(int fd, void* buf, size_t len) {
  auto* p = static_cast<char*>(buf);
  while (len > 0) {
    ssize_t n = ::recv(fd, p, len, 0);
    if (n <= 0)
      throw std::runtime_error("TCP recv failed");
    p += n;
    len -= static_cast<size_t>(n);
  }
}

static std::vector<uint8_t> tcpExchange(
    const std::vector<uint8_t>& localData,
    int /*rank*/) {
  int fd = TestEnv::peerFd();
  uint32_t localSize = static_cast<uint32_t>(localData.size());
  uint32_t remoteSize = 0;
  sendAll(fd, &localSize, sizeof(localSize));
  recvAll(fd, &remoteSize, sizeof(remoteSize));
  std::vector<uint8_t> remoteData(remoteSize);
  sendAll(fd, localData.data(), localSize);
  recvAll(fd, remoteData.data(), remoteSize);
  return remoteData;
}

static bool anyRankWantsToSkip(bool localSkip) {
  int fd = TestEnv::peerFd();
  uint8_t local = localSkip ? 1 : 0;
  uint8_t remote = 0;
  sendAll(fd, &local, 1);
  recvAll(fd, &remote, 1);
  return local != 0 || remote != 0;
}

static void tcpBarrier() {
  int fd = TestEnv::peerFd();
  uint8_t token = 1;
  sendAll(fd, &token, 1);
  recvAll(fd, &token, 1);
}

class MultiTransportCrossHostTest : public ::testing::Test {
 protected:
  void SetUp() override {
    globalRank = TestEnv::rank();
    numRanks = TestEnv::worldSize();
    ASSERT_EQ(numRanks, 2)
        << "MultiTransportCrossHostTest requires exactly 2 ranks";

    auto& topo = Topology::get();
    if (!topo.available()) {
      GTEST_SKIP() << "Topology not available";
    }
    if (topo.nicCount() == 0u) {
      GTEST_SKIP() << "Need at least 1 NIC";
    }
  }

  int globalRank{0};
  int numRanks{0};

  struct ConnectedPair {
    std::unique_ptr<MultiTransportFactory> factory;
    std::unique_ptr<MultiTransport> transport;
  };

  struct SegmentRegistration {
    RegisteredSegment local;
    RemoteRegisteredSegment remote;
  };

  /// Create a connected MultiTransport pair across hosts using TCP to
  /// exchange topology and connection info.
  ConnectedPair connectCrossHost(int deviceId) {
    ConnectedPair pair;
    pair.factory = std::make_unique<MultiTransportFactory>(deviceId);

    auto localTopo = pair.factory->getTopology();
    auto remoteTopo = tcpExchange(localTopo, globalRank);

    auto transportResult = pair.factory->createTransport(remoteTopo);
    EXPECT_TRUE(transportResult.hasValue())
        << "createTransport failed: " << transportResult.error().message();
    pair.transport = std::move(transportResult.value());

    auto bindResult = pair.transport->bind();
    EXPECT_TRUE(bindResult.hasValue())
        << "bind failed: " << bindResult.error().message();
    auto remoteInfo = tcpExchange(bindResult.value(), globalRank);
    auto connectStatus = pair.transport->connect(remoteInfo);
    EXPECT_FALSE(connectStatus.hasError())
        << "connect failed: " << connectStatus.error().message();

    return pair;
  }

  /// Register a local segment, exchange registration payloads via TCP,
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
    auto remotePayload = tcpExchange(localPayload, globalRank);

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

  tcpBarrier();

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

  tcpBarrier();

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

  tcpBarrier();
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

  tcpBarrier();

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

  tcpBarrier();

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

  tcpBarrier();
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
  ::testing::AddGlobalTestEnvironment(new uniflow::TestEnv());
  return RUN_ALL_TESTS();
}
