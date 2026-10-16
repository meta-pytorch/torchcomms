// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/// Cross-host integration test for MultiTransport put/get.
/// Requires MPI with 2 ranks on 2 different hosts (nnodes=2, ppn=1).
/// Each rank creates a MultiTransportFactory, exchanges topology and
/// connection info via MPI, then tests DRAM and GPU put/get transfers.

#include <mpi.h>
#include <chrono>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/drivers/TopologyDiscovery.h"
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

/// Returns the (request, byte) of the first byte that differs from the
/// per-request pattern patternBase + r, or nullopt if none does.
static std::optional<std::pair<size_t, size_t>> firstMismatch(
    const std::vector<char>& data,
    size_t bufSize,
    size_t numRequests,
    uint8_t patternBase) {
  for (size_t r = 0; r < numRequests; ++r) {
    const auto expected = static_cast<uint8_t>(patternBase + r);
    for (size_t i = 0; i < bufSize; ++i) {
      if (static_cast<uint8_t>(data[r * bufSize + i]) != expected) {
        return std::make_pair(r, i);
      }
    }
  }
  return std::nullopt;
}

/// Returns true if `local` is true on any rank. Every rank must call it, so
/// ranks that use the result to return early stay in step.
static bool anyRank(bool local) {
  int localVal = local ? 1 : 0;
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

    auto& topo = sharedTopology();
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
  /// exchange topology and connection info. A step that fails on any rank
  /// leaves every rank's pair without a transport, so callers on both ranks
  /// fail together instead of one rank waiting in MPI for the other.
  ConnectedPair connectCrossHost(
      int deviceId,
      MultiTransportFactoryOptions options = {}) {
    ConnectedPair pair;
    pair.factory =
        std::make_unique<MultiTransportFactory>(deviceId, std::move(options));

    // Exchange topology via MPI.
    auto localTopo = pair.factory->getTopology();
    auto remoteTopo = mpiExchange(localTopo, globalRank);

    auto transportResult = pair.factory->createTransport(remoteTopo);
    EXPECT_TRUE(transportResult.hasValue())
        << "createTransport failed: " << transportResult.error().message();
    if (anyRank(transportResult.hasError())) {
      return pair;
    }
    auto transport = std::move(transportResult.value());

    // Exchange TransportInfo via MPI. Both ranks join the exchange even if
    // bind() failed locally.
    auto bindResult = transport->bind();
    EXPECT_TRUE(bindResult.hasValue())
        << "bind failed: " << bindResult.error().message();
    TransportInfo localInfo;
    if (bindResult.hasValue()) {
      localInfo = std::move(bindResult.value());
    }
    auto remoteInfo = mpiExchange(localInfo, globalRank);
    if (anyRank(bindResult.hasError())) {
      return pair;
    }
    auto connectStatus = transport->connect(remoteInfo);
    EXPECT_FALSE(connectStatus.hasError())
        << "connect failed: " << connectStatus.error().message();
    if (anyRank(connectStatus.hasError())) {
      return pair;
    }

    pair.transport = std::move(transport);
    return pair;
  }

  /// Register a local segment, exchange registration payloads via MPI,
  /// import the remote segment, and return both registered segments. Returns
  /// nullopt on every rank if registration or import failed on any rank.
  std::optional<SegmentRegistration> registerAndExchangeSegments(
      MultiTransportFactory& factory,
      void* buf,
      size_t totalSize,
      MemoryType memType,
      int deviceId = -1) {
    Segment seg(buf, totalSize, memType, deviceId);
    auto regResult = factory.registerSegment(seg);
    EXPECT_TRUE(regResult.hasValue()) << regResult.error().message();

    // Both ranks join the exchange even if registration failed locally.
    std::vector<uint8_t> localPayload;
    if (regResult.hasValue()) {
      localPayload = regResult.value().exportId().value();
    }
    auto remotePayload = mpiExchange(localPayload, globalRank);
    if (anyRank(regResult.hasError())) {
      return std::nullopt;
    }

    auto importResult = factory.importSegment(remotePayload);
    EXPECT_TRUE(importResult.hasValue()) << importResult.error().message();
    if (anyRank(importResult.hasError())) {
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

// Verifies that MultiTransportFactoryOptions::rdmaRequestTimeout reaches the
// RDMA transport unchanged: a request timeout that differs from it is rejected
// with InvalidArgument before dispatch, while a request that inherits it or
// names exactly it succeeds.
TEST_F(MultiTransportCrossHostTest, RdmaRequestTimeoutReachesRdmaTransport) {
  constexpr size_t kTransferSize = 4096;
  constexpr std::chrono::milliseconds kRequestTimeout{1000};
  auto pair = connectCrossHost(
      -1,
      MultiTransportFactoryOptions{
          .preferredTransport = TransportType::RDMA,
          .rdmaRequestTimeout = kRequestTimeout,
      });
  ASSERT_NE(pair.transport, nullptr);
  std::vector<char> buffer(kTransferSize, 0);
  auto segments = registerAndExchangeSegments(
      *pair.factory, buffer.data(), buffer.size(), MemoryType::DRAM);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    const std::vector<TransferRequest> requests = {{
        .local = segments->local.span(size_t{0}, kTransferSize),
        .remote = segments->remote.span(size_t{0}, kTransferSize),
    }};
    const auto mismatched =
        pair.transport
            ->put(requests, RequestOptions{.timeout = 2 * kRequestTimeout})
            .get();
    // EXPECT, not ASSERT: an early return would leave rank 1 at the barrier.
    EXPECT_TRUE(mismatched.hasError());
    if (mismatched.hasError()) {
      EXPECT_EQ(mismatched.error().code(), ErrCode::InvalidArgument);
    }

    // Bounded by kRequestTimeout, which this request inherits.
    const auto inherited = pair.transport->put(requests).get();
    EXPECT_FALSE(inherited.hasError()) << inherited.error().message();

    // Accepted only if the RDMA transport got exactly kRequestTimeout.
    const auto exact =
        pair.transport
            ->put(requests, RequestOptions{.timeout = kRequestTimeout})
            .get();
    EXPECT_FALSE(exact.hasError()) << exact.error().message();
    EXPECT_EQ(pair.transport->transferCount(TransportType::RDMA), 3u);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verifies on real NICs, with no fault injection, that a PUT too large to
// finish within requestTimeout times out through MultiTransport while the NIC
// is still transferring: the PUT gets Timeout, a later PUT gets Aborted, and
// both ranks still shut down.
TEST_F(MultiTransportCrossHostTest, PutLongerThanRequestTimeoutTimesOut) {
  // Over 5 ms even across four 400 Gb/s NICs, far longer than the 1 ms timeout.
  constexpr size_t kTransferSize = size_t{1} << 30;
  constexpr std::chrono::milliseconds kRequestTimeout{1};
  auto pair = connectCrossHost(
      -1,
      MultiTransportFactoryOptions{
          .preferredTransport = TransportType::RDMA,
          .rdmaRequestTimeout = kRequestTimeout,
      });
  ASSERT_NE(pair.transport, nullptr);
  std::vector<char> buffer(
      kTransferSize, globalRank == 0 ? static_cast<char>(0x5A) : 0);
  auto segments = registerAndExchangeSegments(
      *pair.factory, buffer.data(), buffer.size(), MemoryType::DRAM);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    const std::vector<TransferRequest> requests = {{
        .local = segments->local.span(size_t{0}, kTransferSize),
        .remote = segments->remote.span(size_t{0}, kTransferSize),
    }};
    const auto start = std::chrono::steady_clock::now();
    const auto timedOut = pair.transport->put(requests).get();
    const auto elapsedMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    // EXPECT, not ASSERT: an early return would leave rank 1 at the barrier.
    EXPECT_TRUE(timedOut.hasError())
        << "1 GB PUT finished in " << elapsedMs << " ms without timing out";
    if (timedOut.hasError()) {
      EXPECT_EQ(timedOut.error().code(), ErrCode::Timeout)
          << timedOut.error().message();
    }

    const auto later = pair.transport->put(requests).get();
    EXPECT_TRUE(later.hasError());
    if (later.hasError()) {
      EXPECT_EQ(later.error().code(), ErrCode::Aborted);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  pair.transport->shutdown();
  MPI_Barrier(MPI_COMM_WORLD);
}

// --- Transfer test parameterization ---

enum class TransferOp { Put, Get };

// Memory type for test parameterization.
// Combines memory location (CPU vs GPU) with GPU allocation method.
enum class MemType {
  Dram, // CPU memory (std::vector)
  CudaMalloc, // GPU memory via cudaMalloc
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
  return t == MemType::CudaMalloc;
}

const size_t kLargeBufferSize = 12 * 1024 * 1024 + 12 * 1024; // 12MB + 12KB

// --- Helper types ---

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

// --- Same-device transfer tests (parameterized by mem type + alloc type) ---

class SameDeviceTransferTest
    : public MultiTransportCrossHostTest,
      public ::testing::WithParamInterface<CrossHostTransferParam> {
 protected:
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
      (void)cudaSetDevice(cudaDev);
      (void)cudaMemcpy(ptr, staging.data(), totalSize, cudaMemcpyHostToDevice);
      // A copy from pageable memory can return before the data lands; the RDMA
      // transfer must not start until it has.
      (void)cudaDeviceSynchronize();
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
      (void)cudaSetDevice(cudaDev);
      (void)cudaMemset(ptr, 0, totalSize);
      // cudaMemset is asynchronous; a late clear would overwrite data the NIC
      // already wrote.
      (void)cudaDeviceSynchronize();
    } else {
      std::memset(ptr, 0, totalSize);
    }
  }

  // Read buffer contents into a host vector for verification.
  std::vector<char>
  readBuffer(void* ptr, MemType memType, int cudaDev, size_t totalSize) {
    std::vector<char> out(totalSize);
    if (isGpu(memType)) {
      (void)cudaSetDevice(cudaDev);
      (void)cudaMemcpy(out.data(), ptr, totalSize, cudaMemcpyDeviceToHost);
    } else {
      std::memcpy(out.data(), ptr, totalSize);
    }
    return out;
  }

  uniflow::MemoryType toSegmentMemType(MemType t) {
    return isGpu(t) ? MemoryType::VRAM : MemoryType::DRAM;
  }
};

TEST_P(SameDeviceTransferTest, Transfer) {
  const auto& param = GetParam();
  const bool needsCuda =
      isGpu(param.localMemType) || isGpu(param.remoteMemType);

  if (needsCuda) {
    int deviceCount = 0;
    bool noCuda =
        cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 1;
    if (anyRank(noCuda)) {
      GTEST_SKIP() << "Some rank lacks CUDA devices (local: " << deviceCount
                   << ")";
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
  ASSERT_NE(pair.transport, nullptr);

  // Determine which MemType this rank uses.
  MemType myMemType =
      (globalRank == 0) ? param.localMemType : param.remoteMemType;

  // Allocate buffer for this rank.
  std::vector<char> cpuBuf;
  CudaBuffer cudaBuf(myMemType == MemType::CudaMalloc ? totalSize : 0, cudaDev);
  void* myPtr = nullptr;
  if (myMemType == MemType::Dram) {
    cpuBuf.resize(totalSize);
    myPtr = cpuBuf.data();
  } else {
    myPtr = cudaBuf.ptr;
  }
  EXPECT_NE(myPtr, nullptr)
      << "Buffer allocation failed on rank " << globalRank;
  ASSERT_FALSE(anyRank(myPtr == nullptr));

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
    // EXPECT, not ASSERT, on rank-specific checks: an early return would skip
    // the barrier below and deadlock the other rank.
    EXPECT_FALSE(status.hasError())
        << (isPut ? "put" : "get") << " failed: " << status.error().message();

    EXPECT_EQ(pair.transport->transferCount(TransportType::RDMA), 1u);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == verifyRank) {
    auto verify = readBuffer(myPtr, myMemType, cudaDev, totalSize);
    const auto mismatch =
        firstMismatch(verify, bufSize, numRequests, patternBase);
    EXPECT_FALSE(mismatch.has_value())
        << "Data mismatch at request " << mismatch->first << " byte "
        << mismatch->second;
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
        CrossHostTransferParam{kLargeBufferSize, 4, TransferOp::Get, MemType::Dram, MemType::Dram, "DRAM_DRAM_Get_12MB12KB_batch"}
#ifndef UNIFLOW_TEST_MTIA
        ,
        // VRAM → VRAM (cudaMalloc)
        CrossHostTransferParam{4096, 1, TransferOp::Put, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Put_4KB"},
        CrossHostTransferParam{4096, 1, TransferOp::Get, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Get_4KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Put, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Put_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Get, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Get_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 4, TransferOp::Put, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Put_12MB12KB_batch"},
        CrossHostTransferParam{kLargeBufferSize, 4, TransferOp::Get, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Get_12MB12KB_batch"},
        // VRAM → DRAM (mixed)
        CrossHostTransferParam{4096, 1, TransferOp::Put, MemType::CudaMalloc, MemType::Dram, "VRAM_DRAM_Put_4KB"},
        CrossHostTransferParam{4096, 1, TransferOp::Get, MemType::CudaMalloc, MemType::Dram, "VRAM_DRAM_Get_4KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Put, MemType::CudaMalloc, MemType::Dram, "VRAM_DRAM_Put_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Get, MemType::CudaMalloc, MemType::Dram, "VRAM_DRAM_Get_12MB12KB"},
        // DRAM → VRAM (mixed)
        CrossHostTransferParam{4096, 1, TransferOp::Put, MemType::Dram, MemType::CudaMalloc, "DRAM_VRAM_Put_4KB"},
        CrossHostTransferParam{4096, 1, TransferOp::Get, MemType::Dram, MemType::CudaMalloc, "DRAM_VRAM_Get_4KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Put, MemType::Dram, MemType::CudaMalloc, "DRAM_VRAM_Put_12MB12KB"},
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Get, MemType::Dram, MemType::CudaMalloc, "DRAM_VRAM_Get_12MB12KB"}
#endif
        ),
    crossHostParamName);
// clang-format on

#ifndef UNIFLOW_TEST_MTIA
// --- VRAM→VRAM cross-device tests (parameterized by alloc type) ---
// Rank 0 uses GPU 0, rank 1 uses GPU 1.

class CrossDeviceGpuTransferTest
    : public MultiTransportCrossHostTest,
      public ::testing::WithParamInterface<CrossHostTransferParam> {};

TEST_P(CrossDeviceGpuTransferTest, Transfer) {
  int deviceCount = 0;
  bool noCuda =
      cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 2;
  if (anyRank(noCuda)) {
    GTEST_SKIP() << "Need at least 2 CUDA devices per rank (local: "
                 << deviceCount << ")";
  }

  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  const bool isPut = param.op == TransferOp::Put;

  const int cudaDev = globalRank;
  auto pair = connectCrossHost(cudaDev);
  ASSERT_NE(pair.transport, nullptr);

  CudaBuffer cudaBuf(totalSize, cudaDev);
  EXPECT_NE(cudaBuf.ptr, nullptr) << "cudaMalloc failed on device " << cudaDev;
  void* gpuPtr = cudaBuf.ptr;
  ASSERT_FALSE(anyRank(gpuPtr == nullptr));

  const int fillRank = isPut ? 0 : 1;
  const int verifyRank = isPut ? 1 : 0;
  const uint8_t patternBase = 0xD0;

  (void)cudaSetDevice(cudaDev);
  if (globalRank == fillRank) {
    std::vector<char> staging(totalSize);
    for (size_t r = 0; r < numRequests; ++r) {
      std::memset(
          staging.data() + r * bufSize,
          static_cast<int>(patternBase + r),
          bufSize);
    }
    (void)cudaMemcpy(gpuPtr, staging.data(), totalSize, cudaMemcpyHostToDevice);
  } else {
    (void)cudaMemset(gpuPtr, 0, totalSize);
  }
  // As in fillBuffer() and zeroBuffer(): finish the setup before any transfer.
  (void)cudaDeviceSynchronize();

  MPI_Barrier(MPI_COMM_WORLD);

  auto segments = registerAndExchangeSegments(
      *pair.factory, gpuPtr, totalSize, MemoryType::VRAM, cudaDev);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    auto reqs = buildTransferRequests(
        segments->local, segments->remote, bufSize, numRequests);
    auto status = isPut ? pair.transport->put(reqs).get()
                        : pair.transport->get(reqs).get();
    EXPECT_FALSE(status.hasError())
        << "Cross-device " << (isPut ? "put" : "get")
        << " failed: " << status.error().message();

    EXPECT_EQ(pair.transport->transferCount(TransportType::RDMA), 1u);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == verifyRank) {
    std::vector<char> verify(totalSize, 0);
    (void)cudaSetDevice(cudaDev);
    (void)cudaMemcpy(verify.data(), gpuPtr, totalSize, cudaMemcpyDeviceToHost);
    const auto mismatch =
        firstMismatch(verify, bufSize, numRequests, patternBase);
    EXPECT_FALSE(mismatch.has_value())
        << "Cross-device data mismatch at request " << mismatch->first
        << " byte " << mismatch->second;
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
        CrossHostTransferParam{kLargeBufferSize, 1, TransferOp::Get, MemType::CudaMalloc, MemType::CudaMalloc, "CudaMalloc_Get_12MB12KB"}),
    crossHostParamName);
// clang-format on

#endif

} // namespace uniflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase());
  return RUN_ALL_TESTS();
}
