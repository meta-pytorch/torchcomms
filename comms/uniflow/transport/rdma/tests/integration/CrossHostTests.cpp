// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/// Cross-host integration test for RDMA transport.
/// Requires MPI with 2 ranks on 2 different hosts (nnodes=2, ppn=1).
/// Each rank creates an RdmaTransport on its local NIC(s) and connects
/// to the peer via MPI-exchanged connection info.
/// GPU tests require at least 1 CUDA device per host.

#include <dirent.h>
#include <mpi.h>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/rdma/RdmaTransport.h"

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

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

/// Exchange a variable-length byte vector between rank 0 and rank 1 via MPI.
/// Each rank sends its own data and receives the peer's data.
static std::vector<uint8_t> mpiExchange(
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

/// Exchange a uint64_t value between rank 0 and rank 1 via MPI.
static uint64_t mpiExchangeAddr(uint64_t localVal, int rank) {
  int peerRank = 1 - rank;
  uint64_t remoteVal = 0;
  MPI_Sendrecv(
      &localVal,
      1,
      MPI_UINT64_T,
      peerRank,
      2,
      &remoteVal,
      1,
      MPI_UINT64_T,
      peerRank,
      2,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);
  return remoteVal;
}

class CrossHostTest : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ASSERT_EQ(numRanks, 2) << "CrossHostTest requires exactly 2 MPI ranks";

    ibvApi_ = std::make_shared<IbvApi>();
    auto initStatus = ibvApi_->init();
    ASSERT_FALSE(initStatus.hasError())
        << "Failed to init IbvApi: " << initStatus.error().message();

    int nDev = 0;
    auto devResult = ibvApi_->getDeviceList(&nDev);
    ASSERT_TRUE(devResult.hasValue())
        << "Failed to get device list: " << devResult.error().message();
    deviceList_ = devResult.value();
    ASSERT_GE(nDev, 1) << "Need at least 1 RDMA device, found " << nDev;

    // Only use beth (backend ethernet) NICs — they are on the same backend
    // fabric and can reach each other across hosts.
    for (int i = 0; i < nDev; ++i) {
      auto nameResult = ibvApi_->getDeviceName(deviceList_[i]);
      ASSERT_TRUE(nameResult.hasValue());
      if (isBethNic(nameResult.value())) {
        deviceNames_.emplace_back(nameResult.value());
      }
    }
    numDevices_ = deviceNames_.size();
    ASSERT_GE(numDevices_, 1) << "Need at least 1 beth NIC, found 0";

    evbThread_ = std::make_unique<ScopedEventBaseThread>();
  }

  void TearDown() override {
    evbThread_.reset();
    if (deviceList_) {
      ibvApi_->freeDeviceList(deviceList_);
    }
    MpiBaseTestFixture::TearDown();
  }

  /// Check if an RDMA device's netdev name starts with "beth" (backend
  /// ethernet) by reading /sys/class/infiniband/<dev>/device/net/.
  static bool isBethNic(const std::string& devName) {
    std::string netdevDir = "/sys/class/infiniband/" + devName + "/device/net/";
    DIR* dir = opendir(netdevDir.c_str());
    if (!dir) {
      return false;
    }
    bool found = false;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
      std::string name = entry->d_name;
      if (name.rfind("beth", 0) == 0) {
        found = true;
        break;
      }
    }
    closedir(dir);
    return found;
  }

  struct ConnectedPair {
    std::unique_ptr<RdmaTransportFactory> factory;
    std::unique_ptr<Transport> transport;
  };

  struct SegmentRegistration {
    RegisteredSegment local;
    RemoteRegisteredSegment remote;
  };

  /// Register a local segment, exchange registration payloads via MPI,
  /// import the remote segment, and return both registered segments.
  std::optional<SegmentRegistration> registerAndExchangeSegments(
      RdmaTransportFactory& factory,
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

    auto localPayload = regResult.value()->serialize();
    auto remotePayload = mpiExchange(localPayload, globalRank);

    auto remoteHandle = factory.importSegment(totalSize, remotePayload);
    EXPECT_TRUE(remoteHandle.hasValue()) << remoteHandle.error().message();
    if (remoteHandle.hasError()) {
      return std::nullopt;
    }

    auto localReg =
        SegmentTest::makeRegistered(seg, std::move(regResult.value()));

    uint64_t localAddr = reinterpret_cast<uint64_t>(buf);
    uint64_t remoteAddr = mpiExchangeAddr(localAddr, globalRank);

    auto remoteReg = SegmentTest::makeRemote(
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<void*>(remoteAddr),
        totalSize,
        std::move(remoteHandle.value()));

    return SegmentRegistration{std::move(localReg), std::move(remoteReg)};
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

  /// Create a connected transport pair across hosts using MPI to exchange
  /// topology and connection info.
  /// @param numNics  Number of beth NICs to use for this rank's factory.
  /// @param config   RdmaTransportConfig (numQps, etc.).
  ConnectedPair connectCrossHost(
      size_t numNics = 1,
      RdmaTransportConfig config = {}) {
    EXPECT_GE(deviceNames_.size(), numNics)
        << "Need at least " << numNics << " beth NICs, found "
        << deviceNames_.size();

    std::vector<std::string> nicVec(
        deviceNames_.begin(), deviceNames_.begin() + numNics);

    ConnectedPair pair;
    auto* evb = evbThread_->getEventBase();

    pair.factory =
        std::make_unique<RdmaTransportFactory>(nicVec, evb, config, ibvApi_);

    // Exchange topology via MPI.
    auto localTopo = pair.factory->getTopology();
    auto remoteTopo = mpiExchange(localTopo, globalRank);

    auto transportResult = pair.factory->createTransport(remoteTopo);
    EXPECT_TRUE(transportResult.hasValue())
        << "createTransport failed: " << transportResult.error().message();
    pair.transport = std::move(transportResult.value());

    // Exchange TransportInfo via MPI.
    auto localInfo = pair.transport->bind();
    auto remoteInfo = mpiExchange(localInfo, globalRank);
    auto connectStatus = pair.transport->connect(remoteInfo);
    EXPECT_FALSE(connectStatus.hasError())
        << "connect failed: " << connectStatus.error().message();

    return pair;
  }

  std::shared_ptr<IbvApi> ibvApi_;
  ibv_device** deviceList_{nullptr};
  size_t numDevices_{0};
  std::vector<std::string> deviceNames_;
  std::unique_ptr<ScopedEventBaseThread> evbThread_;
};

// --- Connection test ---

TEST_F(CrossHostTest, TransportsConnectAcrossHosts) {
  auto pair = connectCrossHost();
  EXPECT_EQ(pair.transport->state(), TransportState::Connected);

  pair.transport->shutdown();
  EXPECT_EQ(pair.transport->state(), TransportState::Disconnected);
}

// --- Parameterized transfer tests ---

struct CrossHostTransferParam {
  size_t bufSize;
  size_t numRequests;
  size_t numNicsRank0;
  size_t numNicsRank1;
  size_t numQps;
  std::string name;
};

std::string crossHostParamName(
    const ::testing::TestParamInfo<CrossHostTransferParam>& info) {
  return info.param.name;
}

// Returns true if any rank wants to skip (synchronized across all ranks).
bool anyRankWantsToSkip(bool localSkip) {
  int localVal = localSkip ? 1 : 0;
  int globalVal = 0;
  MPI_Allreduce(&localVal, &globalVal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return globalVal != 0;
}

// --- Parameterized DRAM put/get tests ---

class DramCrossHostTransferTest
    : public CrossHostTest,
      public ::testing::WithParamInterface<CrossHostTransferParam> {};

TEST_P(DramCrossHostTransferTest, Put) {
  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  const size_t numNics =
      (globalRank == 0) ? param.numNicsRank0 : param.numNicsRank1;

  if (anyRankWantsToSkip(numDevices_ < numNics)) {
    GTEST_SKIP() << "Some rank lacks sufficient NICs (local: " << numDevices_
                 << ", needed: " << numNics << ")";
  }

  RdmaTransportConfig config;
  config.numQps = static_cast<uint32_t>(param.numQps);
  auto pair = connectCrossHost(numNics, config);

  // Rank 0 is sender, rank 1 is receiver.
  std::vector<char> localBuf(totalSize);
  if (globalRank == 0) {
    for (size_t r = 0; r < numRequests; ++r) {
      std::memset(
          localBuf.data() + r * bufSize, static_cast<int>(0xA0 + r), bufSize);
    }
  } else {
    std::memset(localBuf.data(), 0, totalSize);
  }

  auto segments = registerAndExchangeSegments(
      *pair.factory, localBuf.data(), totalSize, MemoryType::DRAM);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    auto reqs = buildTransferRequests(
        segments->local, segments->remote, bufSize, numRequests);
    auto putStatus = pair.transport->put(reqs, {}).get();
    ASSERT_FALSE(putStatus.hasError())
        << "put failed: " << putStatus.error().message();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == 1) {
    for (size_t r = 0; r < numRequests; ++r) {
      uint8_t expected = static_cast<uint8_t>(0xA0 + r);
      for (size_t i = 0; i < bufSize; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(localBuf[r * bufSize + i]), expected)
            << "Data mismatch at request " << r << " byte " << i;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_P(DramCrossHostTransferTest, Get) {
  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  const size_t numNics =
      (globalRank == 0) ? param.numNicsRank0 : param.numNicsRank1;

  if (anyRankWantsToSkip(numDevices_ < numNics)) {
    GTEST_SKIP() << "Some rank lacks sufficient NICs (local: " << numDevices_
                 << ", needed: " << numNics << ")";
  }

  RdmaTransportConfig config;
  config.numQps = static_cast<uint32_t>(param.numQps);
  auto pair = connectCrossHost(numNics, config);

  // Rank 0 is reader (zeroed), rank 1 is source (filled).
  std::vector<char> localBuf(totalSize);
  if (globalRank == 0) {
    std::memset(localBuf.data(), 0, totalSize);
  } else {
    for (size_t r = 0; r < numRequests; ++r) {
      std::memset(
          localBuf.data() + r * bufSize, static_cast<int>(0xB0 + r), bufSize);
    }
  }

  auto segments = registerAndExchangeSegments(
      *pair.factory, localBuf.data(), totalSize, MemoryType::DRAM);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    auto reqs = buildTransferRequests(
        segments->local, segments->remote, bufSize, numRequests);
    auto getStatus = pair.transport->get(reqs, {}).get();
    ASSERT_FALSE(getStatus.hasError())
        << "get failed: " << getStatus.error().message();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == 0) {
    for (size_t r = 0; r < numRequests; ++r) {
      uint8_t expected = static_cast<uint8_t>(0xB0 + r);
      for (size_t i = 0; i < bufSize; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(localBuf[r * bufSize + i]), expected)
            << "Data mismatch at request " << r << " byte " << i;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

const size_t kLargeBufferSize = 12 * 1024 * 1024 + 12 * 1024; // 12MB + 12KB

INSTANTIATE_TEST_SUITE_P(
    DramCrossHostTransfer,
    DramCrossHostTransferTest,
    ::testing::Values(
        // Single NIC, single QP, varying buffer sizes and request counts.
        CrossHostTransferParam{4096, 1, 1, 1, 1, "4KB_single_req_1nic_1qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            1,
            1,
            1,
            "12MB12KB_single_req_1nic_1qp"},
        CrossHostTransferParam{
            1024 * 1024 * 1024,
            1,
            1,
            1,
            1,
            "1G_single_req_1nic_1qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            1,
            1,
            1,
            "12MB12KB_batch_req_1nic_1qp"},
        // Multi-NIC symmetric: both ranks use 2 NICs, 2 QPs.
        CrossHostTransferParam{4096, 1, 2, 2, 2, "4KB_single_req_2nic_2qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            2,
            2,
            2,
            "12MB12KB_single_req_2nic_2qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            2,
            2,
            2,
            "12MB12KB_batch_req_2nic_2qp"},
        // Asymmetric NICs: rank 0 uses 1 NIC, rank 1 uses 2 NICs, same QPs.
        CrossHostTransferParam{4096, 1, 1, 2, 2, "4KB_single_req_1v2nic_2qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            3,
            4,
            20,
            "12MB12KB_single_req_3v4nic_20qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            3,
            4,
            20,
            "12MB12KB_batch_req_3v4nic_20qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            3,
            7,
            20,
            "12MB12KB_single_req_3v7nic_20qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            3,
            7,
            20,
            "12MB12KB_batch_req_3v7nic_20qp"}),
    crossHostParamName);

// --- Parameterized GPU put/get tests ---

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

class GpuCrossHostTransferTest
    : public CrossHostTest,
      public ::testing::WithParamInterface<CrossHostTransferParam> {};

TEST_P(GpuCrossHostTransferTest, Put) {
  int deviceCount = 0;
  bool noCuda =
      cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 1;
  if (anyRankWantsToSkip(noCuda)) {
    GTEST_SKIP() << "Some rank lacks CUDA devices (local: " << deviceCount
                 << ")";
  }

  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  const size_t numNics =
      (globalRank == 0) ? param.numNicsRank0 : param.numNicsRank1;

  if (anyRankWantsToSkip(numDevices_ < numNics)) {
    GTEST_SKIP() << "Some rank lacks sufficient NICs (local: " << numDevices_
                 << ", needed: " << numNics << ")";
  }

  // Rank 0 uses cuda:0, rank 1 uses cuda:1.
  const int cudaDev = globalRank;
  if (anyRankWantsToSkip(cudaDev >= deviceCount)) {
    GTEST_SKIP() << "Some rank lacks required CUDA device (need device "
                 << cudaDev << ", have " << deviceCount << ")";
  }

  RdmaTransportConfig config;
  config.numQps = static_cast<uint32_t>(param.numQps);
  auto pair = connectCrossHost(numNics, config);

  CudaBuffer gpuBuf(totalSize, cudaDev);
  ASSERT_NE(gpuBuf.ptr, nullptr) << "cudaMalloc failed on device " << cudaDev;

  cudaSetDevice(cudaDev);
  if (globalRank == 0) {
    std::vector<char> staging(totalSize);
    for (size_t r = 0; r < numRequests; ++r) {
      std::memset(
          staging.data() + r * bufSize, static_cast<int>(0xC0 + r), bufSize);
    }
    cudaMemcpy(gpuBuf.ptr, staging.data(), totalSize, cudaMemcpyHostToDevice);
  } else {
    cudaMemset(gpuBuf.ptr, 0, totalSize);
  }
  cudaDeviceSynchronize();

  auto segments = registerAndExchangeSegments(
      *pair.factory, gpuBuf.ptr, totalSize, MemoryType::VRAM, cudaDev);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    auto reqs = buildTransferRequests(
        segments->local, segments->remote, bufSize, numRequests);
    auto putStatus = pair.transport->put(reqs, {}).get();
    ASSERT_FALSE(putStatus.hasError())
        << "GPU put failed: " << putStatus.error().message();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == 1) {
    std::vector<char> verify(totalSize, 0);
    cudaSetDevice(cudaDev);
    cudaMemcpy(verify.data(), gpuBuf.ptr, totalSize, cudaMemcpyDeviceToHost);
    for (size_t r = 0; r < numRequests; ++r) {
      uint8_t expected = static_cast<uint8_t>(0xC0 + r);
      for (size_t i = 0; i < bufSize; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(verify[r * bufSize + i]), expected)
            << "GPU data mismatch at request " << r << " byte " << i;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_P(GpuCrossHostTransferTest, Get) {
  int deviceCount = 0;
  bool noCuda =
      cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 1;
  if (anyRankWantsToSkip(noCuda)) {
    GTEST_SKIP() << "Some rank lacks CUDA devices (local: " << deviceCount
                 << ")";
  }

  const auto& param = GetParam();
  const size_t bufSize = param.bufSize;
  const size_t numRequests = param.numRequests;
  const size_t totalSize = bufSize * numRequests;
  const size_t numNics =
      (globalRank == 0) ? param.numNicsRank0 : param.numNicsRank1;

  if (anyRankWantsToSkip(numDevices_ < numNics)) {
    GTEST_SKIP() << "Some rank lacks sufficient NICs (local: " << numDevices_
                 << ", needed: " << numNics << ")";
  }

  // Rank 0 uses cuda:0, rank 1 uses cuda:1.
  const int cudaDev = globalRank;
  if (anyRankWantsToSkip(cudaDev >= deviceCount)) {
    GTEST_SKIP() << "Some rank lacks required CUDA device (need device "
                 << cudaDev << ", have " << deviceCount << ")";
  }

  RdmaTransportConfig config;
  config.numQps = static_cast<uint32_t>(param.numQps);
  auto pair = connectCrossHost(numNics, config);

  CudaBuffer gpuBuf(totalSize, cudaDev);
  ASSERT_NE(gpuBuf.ptr, nullptr) << "cudaMalloc failed on device " << cudaDev;

  cudaSetDevice(cudaDev);
  if (globalRank == 0) {
    cudaMemset(gpuBuf.ptr, 0, totalSize);
  } else {
    std::vector<char> staging(totalSize);
    for (size_t r = 0; r < numRequests; ++r) {
      std::memset(
          staging.data() + r * bufSize, static_cast<int>(0xD0 + r), bufSize);
    }
    cudaMemcpy(gpuBuf.ptr, staging.data(), totalSize, cudaMemcpyHostToDevice);
  }
  cudaDeviceSynchronize();

  auto segments = registerAndExchangeSegments(
      *pair.factory, gpuBuf.ptr, totalSize, MemoryType::VRAM, cudaDev);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    auto reqs = buildTransferRequests(
        segments->local, segments->remote, bufSize, numRequests);
    auto getStatus = pair.transport->get(reqs, {}).get();
    ASSERT_FALSE(getStatus.hasError())
        << "GPU get failed: " << getStatus.error().message();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == 0) {
    std::vector<char> verify(totalSize, 0);
    cudaSetDevice(cudaDev);
    cudaMemcpy(verify.data(), gpuBuf.ptr, totalSize, cudaMemcpyDeviceToHost);
    for (size_t r = 0; r < numRequests; ++r) {
      uint8_t expected = static_cast<uint8_t>(0xD0 + r);
      for (size_t i = 0; i < bufSize; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(verify[r * bufSize + i]), expected)
            << "GPU data mismatch at request " << r << " byte " << i;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

INSTANTIATE_TEST_SUITE_P(
    GpuCrossHostTransfer,
    GpuCrossHostTransferTest,
    ::testing::Values(
        // Single NIC, single QP.
        CrossHostTransferParam{4096, 1, 1, 1, 1, "4KB_single_req_1nic_1qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            1,
            1,
            1,
            "12MB12KB_single_req_1nic_1qp"},
        CrossHostTransferParam{
            1024 * 1024 * 1024,
            1,
            1,
            1,
            1,
            "1G_single_req_1nic_1qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            1,
            1,
            1,
            "12MB12KB_batch_req_1nic_1qp"},
        // Multi-NIC symmetric: both ranks use 2 NICs, 2 QPs.
        CrossHostTransferParam{4096, 1, 2, 2, 2, "4KB_single_req_2nic_2qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            2,
            2,
            2,
            "12MB12KB_single_req_2nic_2qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            2,
            2,
            2,
            "12MB12KB_batch_req_2nic_2qp"},
        // Asymmetric NICs: rank 0 uses 1 NIC, rank 1 uses 2 NICs, same QPs.
        CrossHostTransferParam{4096, 1, 1, 2, 2, "4KB_single_req_1v2nic_2qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            3,
            4,
            20,
            "12MB12KB_single_req_3v4nic_20qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            3,
            4,
            20,
            "12MB12KB_batch_req_3v4nic_20qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            1,
            3,
            7,
            20,
            "12MB12KB_single_req_3v7nic_20qp"},
        CrossHostTransferParam{
            kLargeBufferSize,
            4,
            3,
            7,
            20,
            "12MB12KB_batch_req_3v7nic_20qp"}),
    crossHostParamName);

} // namespace uniflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase());
  return RUN_ALL_TESTS();
}
