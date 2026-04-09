// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/nvlink/NVLinkTransport.h"
#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/drivers/cuda/mock/MockCudaDriverApi.h"
#include "comms/uniflow/drivers/nvml/mock/MockNvmlApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/nvlink/NVLinkRegistrationHandle.h"

#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace uniflow {

/// Helper that leverages the `friend class SegmentTest` declarations
/// in RegisteredSegment and RemoteRegisteredSegment to construct them
/// from tests.
class SegmentTest {
 public:
  static RegisteredSegment makeRegisteredSegment(
      void* buf,
      size_t len,
      MemoryType memType = MemoryType::DRAM,
      int deviceId = -1) {
    return RegisteredSegment(buf, len, memType, deviceId);
  }

  static RemoteRegisteredSegment makeRemoteRegisteredSegment(
      void* buf,
      size_t len,
      MemoryType memType = MemoryType::DRAM,
      int deviceId = -1) {
    return RemoteRegisteredSegment(buf, len, memType, deviceId);
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

// ---------------------------------------------------------------------------
// Mock device layout — 5 groups across 3 nodes
//
//   Node A (P2P available within):
//     Group 1 (H100 intra-node): devices 0, 1
//         clusterId = all-zero, cliqueId = 0
//         Simulates GPUs without MNNVL fabric manager.
//
//   Node B (P2P available within):
//     Group 2 (MNNVL clique A): devices 2, 3
//         clusterId = 0xAB…, cliqueId = 42
//         Simulates GB200 GPUs in same MNNVL domain on same node.
//
//   Node C (no P2P — each device on a separate remote node):
//     Group 3 (MNNVL clique B): devices 4, 5
//         clusterId = 0xCD…, cliqueId = 24
//     Group 4 (same cluster as A, different clique): device 6
//         clusterId = 0xAB…, cliqueId = 99
//     Group 5 (different cluster, same clique as A): device 7
//         clusterId = 0xEF…, cliqueId = 42
// ---------------------------------------------------------------------------

constexpr int kNumMockDevices = 8;
// 2(version) + 16(clusterId) + 4(cliqueId) + 4(deviceId) + 8(hostHash)
constexpr size_t kTopoSize = 34;

// Group boundaries (half-open ranges).
constexpr int kIntraNodeBegin = 0;
constexpr int kIntraNodeEnd = 2;
constexpr int kCliqueABegin = 2;
constexpr int kCliqueAEnd = 4;
constexpr int kCliqueBBegin = 4;
constexpr int kCliqueBEnd = 6;
// Single-device groups for sameDomain edge cases.
constexpr int kSameClusterDiffCliqueDevice = 6;
constexpr int kDiffClusterSameCliqueDevice = 7;

// Fabric info per group.
constexpr uint8_t kClusterIdA = 0xAB;
constexpr uint32_t kCliqueIdA = 42;
constexpr uint8_t kClusterIdB = 0xCD;
constexpr uint32_t kCliqueIdB = 24;
constexpr uint8_t kClusterIdC = 0xEF;
constexpr uint32_t kCliqueIdC = 99;

/// Configure @p cudaMock with P2P access for same-node device pairs.
/// Group 1 (devices 0, 1) and Group 2 (devices 2, 3) are same-node.
/// All other pairs are cross-node (P2P not available).
void configureP2PMock(MockCudaApi& cudaMock) {
  using ::testing::_;
  using ::testing::Return;

  // Device context management (needed by CudaDeviceGuard in registerSegment).
  ON_CALL(cudaMock, getDevice()).WillByDefault(Return(Result<int>(0)));
  ON_CALL(cudaMock, setDevice(_)).WillByDefault(Return(Ok()));

  // Default: P2P not available.
  ON_CALL(cudaMock, deviceCanAccessPeer(_, _))
      .WillByDefault(Return(Result<bool>(false)));

  // Group 1 (H100 intra-node): devices 0 and 1 can P2P.
  ON_CALL(cudaMock, deviceCanAccessPeer(0, 0))
      .WillByDefault(Return(Result<bool>(true)));
  ON_CALL(cudaMock, deviceCanAccessPeer(0, 1))
      .WillByDefault(Return(Result<bool>(true)));
  ON_CALL(cudaMock, deviceCanAccessPeer(1, 0))
      .WillByDefault(Return(Result<bool>(true)));
  ON_CALL(cudaMock, deviceCanAccessPeer(1, 1))
      .WillByDefault(Return(Result<bool>(true)));

  // Group 2 (MNNVL clique A, same node): devices 2 and 3 can P2P.
  ON_CALL(cudaMock, deviceCanAccessPeer(2, 2))
      .WillByDefault(Return(Result<bool>(true)));
  ON_CALL(cudaMock, deviceCanAccessPeer(2, 3))
      .WillByDefault(Return(Result<bool>(true)));
  ON_CALL(cudaMock, deviceCanAccessPeer(3, 2))
      .WillByDefault(Return(Result<bool>(true)));
  ON_CALL(cudaMock, deviceCanAccessPeer(3, 3))
      .WillByDefault(Return(Result<bool>(true)));
}

/// Configure @p mock with the five-group layout described above.
void configureMultiGroupMock(MockNvmlApi& mock) {
  using ::testing::_;
  using ::testing::Invoke;
  using ::testing::Return;

  ON_CALL(mock, deviceCount())
      .WillByDefault(Return(Result<int>(kNumMockDevices)));

  ON_CALL(mock, deviceInfo(_))
      .WillByDefault(Invoke([](int dev) -> Result<NvmlApi::DeviceInfo> {
        if (dev < 0 || dev >= kNumMockDevices) {
          return Err(
              ErrCode::InvalidArgument, "MockNvmlApi: device out of range");
        }
        NvmlApi::DeviceInfo info;
        info.handle =
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(dev));
        info.computeCapabilityMajor = 9;
        info.computeCapabilityMinor = 0;
        return info;
      }));

  ON_CALL(mock, nvmlDeviceGetGpuFabricInfoV(_, _))
      .WillByDefault(
          Invoke([](nvmlDevice_t device, nvmlGpuFabricInfoV_t* info) -> Status {
            auto dev = static_cast<int>(reinterpret_cast<uintptr_t>(device));
            std::memset(info, 0, sizeof(*info));
            info->version = nvmlPlatformInfo_v2;
            info->state = NVML_GPU_FABRIC_STATE_COMPLETED;
            info->status = NVML_SUCCESS;

            if (dev >= kCliqueABegin && dev < kCliqueAEnd) {
              std::memset(
                  info->clusterUuid, kClusterIdA, NVML_GPU_FABRIC_UUID_LEN);
              info->cliqueId = kCliqueIdA;
            } else if (dev >= kCliqueBBegin && dev < kCliqueBEnd) {
              std::memset(
                  info->clusterUuid, kClusterIdB, NVML_GPU_FABRIC_UUID_LEN);
              info->cliqueId = kCliqueIdB;
            } else if (dev == kSameClusterDiffCliqueDevice) {
              // Group 4: same clusterId as A, different cliqueId.
              std::memset(
                  info->clusterUuid, kClusterIdA, NVML_GPU_FABRIC_UUID_LEN);
              info->cliqueId = kCliqueIdC;
            } else if (dev == kDiffClusterSameCliqueDevice) {
              // Group 5: different clusterId, same cliqueId as A.
              std::memset(
                  info->clusterUuid, kClusterIdC, NVML_GPU_FABRIC_UUID_LEN);
              info->cliqueId = kCliqueIdA;
            }
            // Group 1 (intra-node): zero clusterId and cliqueId (default).
            return Ok();
          }));
}

} // namespace

// ---------------------------------------------------------------------------
// NVLinkTransportTest — no NVML dependency, pure transport logic
// ---------------------------------------------------------------------------

class NVLinkTransportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    transport_ =
        std::make_unique<NVLinkTransport>(0, evbThread_.getEventBase());
  }

  ScopedEventBaseThread evbThread_{"NVLinkTransportTest"};
  std::unique_ptr<NVLinkTransport> transport_;
};

TEST_F(NVLinkTransportTest, ConstructorSetsName) {
  EXPECT_EQ(transport_->name(), "cuda:0");
}

TEST_F(NVLinkTransportTest, InitialState) {
  EXPECT_EQ(transport_->state(), TransportState::Uninitialized);
}

TEST_F(NVLinkTransportTest, NameReflectsDeviceId) {
  NVLinkTransport t3(3, evbThread_.getEventBase());
  EXPECT_EQ(t3.name(), "cuda:3");

  NVLinkTransport t7(7, evbThread_.getEventBase());
  EXPECT_EQ(t7.name(), "cuda:7");
}

TEST_F(NVLinkTransportTest, ShutdownKeepsDisconnected) {
  transport_->shutdown();
  EXPECT_EQ(transport_->state(), TransportState::Disconnected);
}

TEST_F(NVLinkTransportTest, ConstructorRejectsNullEventBase) {
  EXPECT_THROW({ NVLinkTransport t(0, nullptr); }, std::invalid_argument);
}

// --- bind / connect ---

TEST_F(NVLinkTransportTest, BindReturnsDeviceIdSerialized) {
  auto info = transport_->bind();
  ASSERT_EQ(info.size(), sizeof(int32_t));

  int32_t devId = -1;
  std::memcpy(&devId, info.data(), sizeof(devId));
  EXPECT_EQ(devId, 0);
}

TEST_F(NVLinkTransportTest, ConnectWithBindOutput) {
  NVLinkTransport local(0, evbThread_.getEventBase());
  NVLinkTransport remote(1, evbThread_.getEventBase());

  auto remoteInfo = remote.bind();
  auto status = local.connect(remoteInfo);

  EXPECT_TRUE(status.hasValue());
  EXPECT_EQ(local.state(), TransportState::Connected);
}

TEST_F(NVLinkTransportTest, ConnectRejectsEmptyInfo) {
  TransportInfo empty;
  auto status = transport_->connect(empty);
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
  EXPECT_EQ(transport_->state(), TransportState::Uninitialized);
}

TEST_F(NVLinkTransportTest, ConnectRejectsTruncatedInfo) {
  TransportInfo truncated(sizeof(int32_t) - 1, 0x00);
  auto status = transport_->connect(truncated);
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_F(NVLinkTransportTest, ConnectRejectsOversizedInfo) {
  TransportInfo oversized(sizeof(int32_t) + 1, 0x00);
  auto status = transport_->connect(oversized);
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_F(NVLinkTransportTest, ShutdownAfterConnectResetsState) {
  auto info = transport_->bind();
  transport_->connect(info);
  EXPECT_EQ(transport_->state(), TransportState::Connected);

  transport_->shutdown();
  EXPECT_EQ(transport_->state(), TransportState::Disconnected);
}

// ---------------------------------------------------------------------------
// NVLinkTransportFactoryTest — uses MockNvmlApi + MockCudaApi
//
// Node A (P2P within): Group 1 (devices 0-1), H100 intra-node.
//     canConnect() succeeds via sameHost + deviceCanAccessPeer.
//
// Node B (P2P within): Group 2 (devices 2-3), MNNVL clique A.
//     canConnect() succeeds via sameDomain (same clique).
//
// Remote nodes (no P2P): Groups 3-5 (devices 4-7).
//     canConnect() fails — different domain and no P2P access.
// ---------------------------------------------------------------------------

class NVLinkTransportFactoryTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    nvmlMock_ = std::make_shared<::testing::NiceMock<MockNvmlApi>>();
    configureMultiGroupMock(*nvmlMock_);
    cudaApiMock_ = std::make_shared<::testing::NiceMock<MockCudaApi>>();
    configureP2PMock(*cudaApiMock_);
    cudaDriverMock_ =
        std::make_shared<::testing::NiceMock<MockCudaDriverApi>>();
    ON_CALL(*cudaDriverMock_, getCuMemHandleType())
        .WillByDefault(
            ::testing::Return(
                Result<CUmemAllocationHandleType>(
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)));
  }

  static void TearDownTestSuite() {
    nvmlMock_.reset();
    cudaApiMock_.reset();
    cudaDriverMock_.reset();
  }

  NVLinkTransportFactory makeFactory(int device) {
    return NVLinkTransportFactory(
        device,
        evbThread_.getEventBase(),
        nvmlMock_,
        cudaApiMock_,
        cudaDriverMock_);
  }

  ScopedEventBaseThread evbThread_{"NVLinkTransportFactoryTest"};

  // NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
  static std::shared_ptr<::testing::NiceMock<MockNvmlApi>> nvmlMock_;
  // NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
  static std::shared_ptr<::testing::NiceMock<MockCudaApi>> cudaApiMock_;
  // NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
  static std::shared_ptr<::testing::NiceMock<MockCudaDriverApi>>
      cudaDriverMock_;
};

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
std::shared_ptr<::testing::NiceMock<MockNvmlApi>>
    NVLinkTransportFactoryTest::nvmlMock_;
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
std::shared_ptr<::testing::NiceMock<MockCudaApi>>
    NVLinkTransportFactoryTest::cudaApiMock_;
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
std::shared_ptr<::testing::NiceMock<MockCudaDriverApi>>
    NVLinkTransportFactoryTest::cudaDriverMock_;

// --- bounds checking ---

TEST_F(NVLinkTransportFactoryTest, ConstructorRejectsNegativeDevice) {
  EXPECT_THROW(
      {
        NVLinkTransportFactory factory(
            -1, evbThread_.getEventBase(), nvmlMock_);
      },
      std::out_of_range);
}

TEST_F(NVLinkTransportFactoryTest, ConstructorRejectsExcessiveDevice) {
  EXPECT_THROW(
      {
        NVLinkTransportFactory factory(
            999, evbThread_.getEventBase(), nvmlMock_);
      },
      std::out_of_range);
}

TEST_F(NVLinkTransportFactoryTest, ConstructorRejectsDeviceCountBoundary) {
  EXPECT_THROW(
      {
        NVLinkTransportFactory factory(
            kNumMockDevices, evbThread_.getEventBase(), nvmlMock_);
      },
      std::out_of_range);
}

TEST_F(NVLinkTransportFactoryTest, ConstructorAcceptsAllValidDevices) {
  for (int i = 0; i < kNumMockDevices; ++i) {
    EXPECT_NO_THROW({ auto factory = makeFactory(i); }) << "device " << i;
  }
}

TEST_F(NVLinkTransportFactoryTest, ConstructorRejectsNullEventBase) {
  EXPECT_THROW(
      { NVLinkTransportFactory factory(0, nullptr, nvmlMock_); },
      std::invalid_argument);
}

// --- topology serialization ---

TEST_F(NVLinkTransportFactoryTest, TopologySize) {
  for (int i = 0; i < kNumMockDevices; ++i) {
    auto factory = makeFactory(i);
    EXPECT_EQ(factory.getTopology().size(), kTopoSize) << "device " << i;
  }
}

TEST_F(NVLinkTransportFactoryTest, SameGroupSharesTopologyPrefix) {
  // Devices in the same group share clusterId + cliqueId (first 20 bytes).
  auto checkGroup = [&](int begin, int end) {
    auto ref = makeFactory(begin).getTopology();
    for (int dev = begin + 1; dev < end; ++dev) {
      auto topo = makeFactory(dev).getTopology();
      EXPECT_EQ(
          std::vector<uint8_t>(ref.begin(), ref.begin() + 20),
          std::vector<uint8_t>(topo.begin(), topo.begin() + 20))
          << "devices " << begin << " and " << dev;
    }
  };
  checkGroup(kIntraNodeBegin, kIntraNodeEnd);
  checkGroup(kCliqueABegin, kCliqueAEnd);
  checkGroup(kCliqueBBegin, kCliqueBEnd);
}

TEST_F(NVLinkTransportFactoryTest, TopologyConsistentAcrossFactories) {
  auto f1 = makeFactory(0);
  auto f2 = makeFactory(0);
  EXPECT_EQ(f1.getTopology(), f2.getTopology());
}

// --- intra-node transport creation succeeds (Group 1) ---
// Devices with zero clusterUuid on the same host with P2P access.

TEST_F(NVLinkTransportFactoryTest, IntraNodeCreateTransportSucceeds) {
  auto factory = makeFactory(0);
  auto topo = factory.getTopology();
  auto result = factory.createTransport(topo);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
  EXPECT_NE(result.value(), nullptr);
}

TEST_F(NVLinkTransportFactoryTest, IntraNodeCreateTransportCrossDevice) {
  auto f0 = makeFactory(0);
  auto topo1 = makeFactory(1).getTopology();
  auto result = f0.createTransport(topo1);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
}

TEST_F(NVLinkTransportFactoryTest, IntraNodeRejectsCrossGroupP2P) {
  // Zero-UUID device (0) connecting to MNNVL device (2) — different node, no
  // P2P.
  auto factory = makeFactory(0);
  auto peerTopo = makeFactory(kCliqueABegin).getTopology();
  auto result = factory.createTransport(peerTopo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

TEST_F(NVLinkTransportFactoryTest, IntraNodeRejectsWhenP2PNotAvailable) {
  // Override: temporarily disable P2P between device 0 and device 1.
  ON_CALL(
      *cudaApiMock_, deviceCanAccessPeer(kIntraNodeBegin, kIntraNodeEnd - 1))
      .WillByDefault(::testing::Return(Result<bool>(false)));

  auto factory = makeFactory(kIntraNodeBegin);
  auto peerTopo = makeFactory(kIntraNodeEnd - 1).getTopology();
  auto result = factory.createTransport(peerTopo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);

  // Restore P2P for subsequent tests.
  ON_CALL(
      *cudaApiMock_, deviceCanAccessPeer(kIntraNodeBegin, kIntraNodeEnd - 1))
      .WillByDefault(::testing::Return(Result<bool>(true)));
}

TEST_F(NVLinkTransportFactoryTest, IntraNodeRejectsDifferentHost) {
  // Construct a peer topology with a different hostHash.
  auto peerTopo = makeFactory(kIntraNodeEnd - 1).getTopology();
  // Tamper with the hostHash field (last 8 bytes of the serialized topology).
  uint64_t differentHash = 0xDEADBEEFCAFEBABEULL;
  std::memcpy(
      peerTopo.data() + peerTopo.size() - sizeof(differentHash),
      &differentHash,
      sizeof(differentHash));

  auto factory = makeFactory(kIntraNodeBegin);
  auto result = factory.createTransport(peerTopo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

// --- multi-node (MNNVL) transport creation succeeds (Groups 2, 3) ---

TEST_F(NVLinkTransportFactoryTest, MnnvlCreateTransportSameDevice) {
  auto factory = makeFactory(kCliqueABegin);
  auto topo = factory.getTopology();
  auto result = factory.createTransport(topo);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
  EXPECT_NE(result.value(), nullptr);
  EXPECT_EQ(result.value()->name(), "cuda:" + std::to_string(kCliqueABegin));
  EXPECT_EQ(result.value()->state(), TransportState::Uninitialized);
}

TEST_F(NVLinkTransportFactoryTest, MnnvlCreateTransportSameClique) {
  auto factory = makeFactory(kCliqueABegin);
  auto peerTopo = makeFactory(kCliqueAEnd - 1).getTopology();
  auto result = factory.createTransport(peerTopo);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
}

TEST_F(NVLinkTransportFactoryTest, MnnvlCreateTransportSameCliqueB) {
  auto factory = makeFactory(kCliqueBBegin);
  auto peerTopo = makeFactory(kCliqueBEnd - 1).getTopology();
  auto result = factory.createTransport(peerTopo);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
}

// --- cross-group transport creation fails ---

TEST_F(NVLinkTransportFactoryTest, MnnvlRejectsDifferentCliqueNoP2P) {
  // Different MNNVL cliques, different node (no P2P) → disconnects.
  auto factory = makeFactory(kCliqueABegin);
  auto peerTopo = makeFactory(kCliqueBBegin).getTopology();
  auto result = factory.createTransport(peerTopo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

TEST_F(NVLinkTransportFactoryTest, MnnvlRejectsIntraNodePeerNoP2P) {
  // MNNVL device (2) connecting to intra-node device (0) — no P2P.
  auto factory = makeFactory(kCliqueABegin);
  auto peerTopo = makeFactory(kIntraNodeBegin).getTopology();
  auto result = factory.createTransport(peerTopo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

// --- malformed topology rejection ---

TEST_F(NVLinkTransportFactoryTest, RejectsEmptyTopology) {
  auto factory = makeFactory(kCliqueABegin);
  std::vector<uint8_t> empty;
  auto result = factory.createTransport(empty);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(NVLinkTransportFactoryTest, RejectsTruncatedTopology) {
  auto factory = makeFactory(kCliqueABegin);
  std::vector<uint8_t> truncated(kTopoSize - 1, 0x00);
  auto result = factory.createTransport(truncated);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(NVLinkTransportFactoryTest, RejectsOversizedTopology) {
  auto factory = makeFactory(kCliqueABegin);
  std::vector<uint8_t> oversized(kTopoSize + 1, 0x00);
  auto result = factory.createTransport(oversized);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

// --- sameDomain edge cases ---
// Uses groups 4 and 5 to test domain matching without assuming wire format.

// Same clusterId but different cliqueId, no P2P → disconnects.
TEST_F(NVLinkTransportFactoryTest, SameClusterIdDifferentCliqueIdDisconnects) {
  auto factory = makeFactory(kCliqueABegin);
  auto peerTopo = makeFactory(kSameClusterDiffCliqueDevice).getTopology();
  auto result = factory.createTransport(peerTopo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

// Different clusterId but same cliqueId, no P2P → disconnects.
TEST_F(NVLinkTransportFactoryTest, DifferentClusterIdSameCliqueIdDisconnects) {
  auto factory = makeFactory(kCliqueABegin);
  auto peerTopo = makeFactory(kDiffClusterSameCliqueDevice).getTopology();
  auto result = factory.createTransport(peerTopo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

// Same clusterId and same cliqueId, different device → same domain (succeeds).
// Clique A device 2 vs clique A device 3: both clusterId=0xAB, cliqueId=42.
TEST_F(NVLinkTransportFactoryTest, SameDomainDifferentDeviceIdConnects) {
  auto factory = makeFactory(kCliqueABegin);
  auto peerTopo = makeFactory(kCliqueAEnd - 1).getTopology();
  auto result = factory.createTransport(peerTopo);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
}

// ---------------------------------------------------------------------------
// NVLinkRegistrationTest — tests for registerSegment / registration handles
// using MockCudaDriverApi
// ---------------------------------------------------------------------------

class NVLinkRegistrationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    nvmlMock_ = std::make_shared<::testing::NiceMock<MockNvmlApi>>();
    configureMultiGroupMock(*nvmlMock_);
  }

  static void TearDownTestSuite() {
    nvmlMock_.reset();
  }

  void SetUp() override {
    using ::testing::_;
    using ::testing::Return;

    cudaDriverMock_ =
        std::make_shared<::testing::NiceMock<MockCudaDriverApi>>();
    cudaApiMock_ = std::make_shared<::testing::NiceMock<MockCudaApi>>();

    // Device context management (needed by CudaDeviceGuard in registerSegment).
    ON_CALL(*cudaApiMock_, getDevice()).WillByDefault(Return(Result<int>(0)));
    ON_CALL(*cudaApiMock_, setDevice(::testing::_)).WillByDefault(Return(Ok()));

    // Default handle type for factory constructor.
    ON_CALL(*cudaDriverMock_, getCuMemHandleType())
        .WillByDefault(Return(
            Result<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_FABRIC)));
  }

  void TearDown() override {
    cudaDriverMock_.reset();
    cudaApiMock_.reset();
  }

  NVLinkTransportFactory makeFactory(int device) {
    return NVLinkTransportFactory(
        device,
        evbThread_.getEventBase(),
        nvmlMock_,
        cudaApiMock_,
        cudaDriverMock_);
  }

  NVLinkTransportFactory makeFactory(
      int device,
      CUmemAllocationHandleType handleType) {
    ON_CALL(*cudaDriverMock_, getCuMemHandleType())
        .WillByDefault(
            ::testing::Return(Result<CUmemAllocationHandleType>(handleType)));
    return NVLinkTransportFactory(
        device,
        evbThread_.getEventBase(),
        nvmlMock_,
        cudaApiMock_,
        cudaDriverMock_);
  }

  // NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
  static std::shared_ptr<::testing::NiceMock<MockNvmlApi>> nvmlMock_;
  ScopedEventBaseThread evbThread_{"NVLinkRegistrationTest"};
  std::shared_ptr<::testing::NiceMock<MockCudaDriverApi>> cudaDriverMock_;
  std::shared_ptr<::testing::NiceMock<MockCudaApi>> cudaApiMock_;
};

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
std::shared_ptr<::testing::NiceMock<MockNvmlApi>>
    NVLinkRegistrationTest::nvmlMock_;

// --- registerSegment rejects non-VRAM ---

TEST_F(NVLinkRegistrationTest, RejectsNonVramSegment) {
  auto factory = makeFactory(kCliqueABegin);
  uint8_t buf[64];
  Segment seg(buf, sizeof(buf), MemoryType::DRAM);
  auto result = factory.registerSegment(seg);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

// --- registerSegment succeeds ---

TEST_F(NVLinkRegistrationTest, RegisterSegmentSuccess) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  CUmemGenericAllocationHandle fakeAllocHandle = 0xDEAD;
  CUmemFabricHandle fakeFabricHandle{};
  fakeFabricHandle.data[0] = 0x42;

  ON_CALL(*cudaDriverMock_, cuMemRetainAllocationHandle(_, _))
      .WillByDefault(DoAll(SetArgPointee<0>(fakeAllocHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemExportToShareableHandle(_, _, _, _))
      .WillByDefault(DoAll(
          [&](void* shareableHandle,
              CUmemGenericAllocationHandle,
              CUmemAllocationHandleType,
              unsigned long long) {
            std::memcpy(
                shareableHandle, &fakeFabricHandle, sizeof(fakeFabricHandle));
          },
          Return(Ok())));

  auto factory = makeFactory(kCliqueABegin);
  uint8_t buf[64];
  Segment seg(buf, sizeof(buf), MemoryType::VRAM, kCliqueABegin);
  auto result = factory.registerSegment(seg);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
  ASSERT_NE(result.value(), nullptr);

  auto* handle =
      dynamic_cast<NVLinkFabricRegistrationHandle*>(result.value().get());
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->allocHandle(), fakeAllocHandle);
  EXPECT_EQ(
      std::memcmp(
          &handle->fabricHandle(), &fakeFabricHandle, sizeof(fakeFabricHandle)),
      0);
}

// --- registerSegment fails when cuMemRetainAllocationHandle fails ---

TEST_F(NVLinkRegistrationTest, FailsWhenRetainHandleFails) {
  using ::testing::_;
  using ::testing::Return;

  EXPECT_CALL(*cudaDriverMock_, cuMemRetainAllocationHandle(_, _))
      .Times(1)
      .WillOnce(Return(Err(ErrCode::DriverError, "retain failed")));

  auto factory = makeFactory(kCliqueABegin);
  uint8_t buf[64];
  Segment seg(buf, sizeof(buf), MemoryType::VRAM, kCliqueABegin);
  auto result = factory.registerSegment(seg);
  ASSERT_TRUE(result.hasError());
}

// --- registerSegment fails when cuMemExportToShareableHandle fails ---
// Verifies that cuMemRelease is called to clean up the retained handle.

TEST_F(NVLinkRegistrationTest, FailsWhenExportFails_ReleasesHandle) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  CUmemGenericAllocationHandle fakeAllocHandle = 0xBEEF;

  ON_CALL(*cudaDriverMock_, cuMemRetainAllocationHandle(_, _))
      .WillByDefault(DoAll(SetArgPointee<0>(fakeAllocHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemExportToShareableHandle(_, _, _, _))
      .WillByDefault(Return(Err(ErrCode::DriverError, "export failed")));

  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(fakeAllocHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  auto factory = makeFactory(kCliqueABegin);
  uint8_t buf[64];
  Segment seg(buf, sizeof(buf), MemoryType::VRAM, kCliqueABegin);
  auto result = factory.registerSegment(seg);
  ASSERT_TRUE(result.hasError());
}

TEST_F(NVLinkRegistrationTest, FabricRegisteredSegmentCleanUp) {
  using ::testing::Return;

  size_t fakeAllocSize = 4;
  CUmemGenericAllocationHandle fakeAllocHandle = 0x1234;
  CUmemFabricHandle fakeFabricHandle{};

  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(fakeAllocHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  {
    auto handle = NVLinkFabricRegistrationHandle(
        fakeAllocSize, fakeAllocHandle, fakeFabricHandle, cudaDriverMock_);
  }
}

TEST_F(NVLinkRegistrationTest, RemoteRegisteredSegmentCleanUpAll) {
  using ::testing::Return;

  constexpr CUmemGenericAllocationHandle kHandle = 0x5678;
  constexpr CUdeviceptr kPtr = 0x7F000000;
  constexpr size_t kSize = 2 * 1024 * 1024;

  {
    ::testing::InSequence seq;
    EXPECT_CALL(*cudaDriverMock_, cuMemUnmap(kPtr, kSize))
        .Times(1)
        .WillOnce(Return(Ok()));
    EXPECT_CALL(*cudaDriverMock_, cuMemAddressFree(kPtr, kSize))
        .Times(1)
        .WillOnce(Return(Ok()));
    EXPECT_CALL(*cudaDriverMock_, cuMemRelease(kHandle))
        .Times(1)
        .WillOnce(Return(Ok()));
  }

  {
    auto handle =
        NVLinkRemoteRegistrationHandle(kHandle, kPtr, kSize, cudaDriverMock_);
  }
}

// --- transportType ---

TEST_F(NVLinkRegistrationTest, FabricTransportTypeReturnsNVLink) {
  CUmemFabricHandle fakeFabricHandle{};
  auto handle =
      NVLinkFabricRegistrationHandle(0, 0, fakeFabricHandle, cudaDriverMock_);
  EXPECT_EQ(handle.transportType(), TransportType::NVLink);
}

// --- serialize ---

TEST_F(NVLinkRegistrationTest, FabricSerializeProducesCorrectSize) {
  CUmemFabricHandle fakeFabricHandle{};
  fakeFabricHandle.data[0] = 0xAA;
  auto handle =
      NVLinkFabricRegistrationHandle(0, 0, fakeFabricHandle, cudaDriverMock_);
  auto serialized = handle.serialize();
  EXPECT_EQ(serialized.size(), NVLinkFabricRegistrationHandle::kSerializedSize);
}

TEST_F(NVLinkRegistrationTest, FdTransportTypeReturnsNVLink) {
  auto handle = NVLinkFdRegistrationHandle(
      0,
      /*exportedFd=*/7,
      /*ownerPid=*/1234,
      /*allocationSize=*/4096,
      cudaDriverMock_);
  EXPECT_EQ(handle.transportType(), TransportType::NVLink);
}

TEST_F(NVLinkRegistrationTest, FdSerializeProducesCorrectSize) {
  auto handle = NVLinkFdRegistrationHandle(
      0,
      /*exportedFd=*/7,
      /*ownerPid=*/1234,
      /*allocationSize=*/4096,
      cudaDriverMock_);
  auto serialized = handle.serialize();
  EXPECT_EQ(serialized.size(), NVLinkFdRegistrationHandle::kSerializedSize);
}

TEST_F(NVLinkRegistrationTest, FdRegisteredSegmentCleanUp) {
  using ::testing::Return;

  CUmemGenericAllocationHandle fakeAllocHandle = 0x5555;

  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(fakeAllocHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  {
    auto handle = NVLinkFdRegistrationHandle(
        fakeAllocHandle,
        /*exportedFd=*/99,
        /*ownerPid=*/1234,
        /*allocationSize=*/4096,
        cudaDriverMock_);
  }
}

// ---------------------------------------------------------------------------
// NVLinkImportTest — tests for importSegment / NVLinkRemoteRegistrationHandle
// using MockCudaDriverApi
// ---------------------------------------------------------------------------

class NVLinkImportTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    nvmlMock_ = std::make_shared<::testing::NiceMock<MockNvmlApi>>();
    configureMultiGroupMock(*nvmlMock_);
  }

  static void TearDownTestSuite() {
    nvmlMock_.reset();
  }

  void SetUp() override {
    using ::testing::_;
    using ::testing::Return;

    cudaDriverMock_ =
        std::make_shared<::testing::NiceMock<MockCudaDriverApi>>();
    cudaApiMock_ = std::make_shared<::testing::NiceMock<MockCudaApi>>();

    // Device context management (needed by CudaDeviceGuard in registerSegment).
    ON_CALL(*cudaApiMock_, getDevice()).WillByDefault(Return(Result<int>(0)));
    ON_CALL(*cudaApiMock_, setDevice(::testing::_)).WillByDefault(Return(Ok()));

    // Default handle type for factory constructor.
    ON_CALL(*cudaDriverMock_, getCuMemHandleType())
        .WillByDefault(Return(
            Result<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_FABRIC)));
  }

  void TearDown() override {
    cudaDriverMock_.reset();
    cudaApiMock_.reset();
  }

  NVLinkTransportFactory makeFactory(int device) {
    return NVLinkTransportFactory(
        device,
        evbThread_.getEventBase(),
        nvmlMock_,
        cudaApiMock_,
        cudaDriverMock_);
  }

  /// Build a valid Fabric-mode serialized payload.
  static std::vector<uint8_t> makePayload(
      const CUmemFabricHandle& fabricHandle,
      uint64_t allocationSize = 4096) {
    NVLinkFabricRegistrationHandle::Payload p{
        MemSharingMode::Fabric, allocationSize, fabricHandle};
    std::vector<uint8_t> buf(sizeof(p));
    std::memcpy(buf.data(), &p, sizeof(p));
    return buf;
  }

  /// Configure mocks for a successful full VMM import flow.
  /// Verifies that cuMemGetAllocationGranularity and cuMemSetAccess
  /// target the correct device via their struct parameters.
  void configureMocksForImportSuccess(
      int deviceId,
      CUmemGenericAllocationHandle importedHandle,
      size_t granularity,
      CUdeviceptr mappedPtr) {
    using ::testing::_;
    using ::testing::DoAll;
    using ::testing::Invoke;
    using ::testing::Return;
    using ::testing::SetArgPointee;

    ON_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
        .WillByDefault(DoAll(SetArgPointee<0>(importedHandle), Return(Ok())));

    ON_CALL(*cudaDriverMock_, cuMemGetAllocationGranularity(_, _, _))
        .WillByDefault(Invoke(
            [deviceId, granularity](
                size_t* out,
                const CUmemAllocationProp* prop,
                CUmemAllocationGranularity_flags) -> Status {
              EXPECT_EQ(prop->location.id, deviceId);
              EXPECT_EQ(prop->location.type, CU_MEM_LOCATION_TYPE_DEVICE);
              *out = granularity;
              return Ok();
            }));

    ON_CALL(*cudaDriverMock_, cuMemAddressReserve(_, _, _, _, _))
        .WillByDefault(DoAll(SetArgPointee<0>(mappedPtr), Return(Ok())));
    ON_CALL(*cudaDriverMock_, cuMemMap(_, _, _, _, _))
        .WillByDefault(Return(Ok()));

    ON_CALL(*cudaDriverMock_, cuMemSetAccess(_, _, _, _))
        .WillByDefault(Invoke(
            [deviceId](
                CUdeviceptr, size_t, const CUmemAccessDesc* desc, size_t count)
                -> Status {
              EXPECT_EQ(count, 1u);
              EXPECT_EQ(desc->location.id, deviceId);
              EXPECT_EQ(desc->location.type, CU_MEM_LOCATION_TYPE_DEVICE);
              EXPECT_EQ(desc->flags, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
              return Ok();
            }));
  }

  // NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
  static std::shared_ptr<::testing::NiceMock<MockNvmlApi>> nvmlMock_;
  ScopedEventBaseThread evbThread_{"NVLinkImportTest"};
  std::shared_ptr<::testing::NiceMock<MockCudaDriverApi>> cudaDriverMock_;
  std::shared_ptr<::testing::NiceMock<MockCudaApi>> cudaApiMock_;
};

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
std::shared_ptr<::testing::NiceMock<MockNvmlApi>> NVLinkImportTest::nvmlMock_;

// --- importSegment rejects wrong payload size ---

TEST_F(NVLinkImportTest, RejectsEmptyPayload) {
  auto factory = makeFactory(kCliqueABegin);
  std::vector<uint8_t> empty;
  auto result = factory.importSegment(4096, empty);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(NVLinkImportTest, RejectsTruncatedPayload) {
  auto factory = makeFactory(kCliqueABegin);
  std::vector<uint8_t> truncated(
      NVLinkFabricRegistrationHandle::kSerializedSize - 1, 0x00);
  auto result = factory.importSegment(4096, truncated);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(NVLinkImportTest, RejectsOversizedPayload) {
  auto factory = makeFactory(kCliqueABegin);
  std::vector<uint8_t> oversized(
      NVLinkFabricRegistrationHandle::kSerializedSize + 1, 0x00);
  auto result = factory.importSegment(4096, oversized);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

// --- importSegment succeeds with full VMM flow ---

TEST_F(NVLinkImportTest, ImportSegmentSuccess) {
  constexpr CUmemGenericAllocationHandle kImportedHandle = 0xCAFE;
  constexpr size_t kGranularity = 2 * 1024 * 1024; // 2 MiB
  constexpr CUdeviceptr kMappedPtr = 0x7F000000;
  constexpr size_t kSegLen = 4096;

  configureMocksForImportSuccess(
      kCliqueABegin, kImportedHandle, kGranularity, kMappedPtr);

  CUmemFabricHandle fakeFabricHandle{};
  fakeFabricHandle.data[0] = 0x42;
  auto payload = makePayload(fakeFabricHandle);

  auto factory = makeFactory(kCliqueABegin);
  auto result = factory.importSegment(kSegLen, payload);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
  ASSERT_NE(result.value(), nullptr);
  EXPECT_NE(
      dynamic_cast<NVLinkRemoteRegistrationHandle*>(result.value().get()),
      nullptr);
}

// --- importSegment rounds up to granularity correctly ---

TEST_F(NVLinkImportTest, ImportSegmentAlignsSizeToGranularity) {
  using ::testing::_;

  constexpr CUmemGenericAllocationHandle kImportedHandle = 0xABCD;
  constexpr size_t kGranularity = 2 * 1024 * 1024; // 2 MiB
  constexpr CUdeviceptr kMappedPtr = 0x7F000000;
  // 3 MiB segment should round up to 4 MiB (2 * granularity).
  constexpr size_t kSegLen = 3 * 1024 * 1024;
  constexpr size_t kExpectedAlignedSize = 4 * 1024 * 1024;

  configureMocksForImportSuccess(
      kCliqueABegin, kImportedHandle, kGranularity, kMappedPtr);

  // Verify cuMemAddressReserve and cuMemMap receive the aligned size.
  EXPECT_CALL(
      *cudaDriverMock_, cuMemAddressReserve(_, kExpectedAlignedSize, _, _, _))
      .WillOnce(
          ::testing::DoAll(
              ::testing::SetArgPointee<0>(kMappedPtr),
              ::testing::Return(Ok())));
  EXPECT_CALL(
      *cudaDriverMock_, cuMemMap(kMappedPtr, kExpectedAlignedSize, _, _, _))
      .WillOnce(::testing::Return(Ok()));

  CUmemFabricHandle fakeFabricHandle{};
  auto payload = makePayload(fakeFabricHandle, kSegLen);
  auto factory = makeFactory(kCliqueABegin);
  auto result = factory.importSegment(kSegLen, payload);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
}

// --- importSegment with different factory device ---

TEST_F(NVLinkImportTest, ImportSegmentUsesCorrectDeviceCliqueB) {
  constexpr CUmemGenericAllocationHandle kImportedHandle = 0x9999;
  constexpr size_t kGranularity = 2 * 1024 * 1024;
  constexpr CUdeviceptr kMappedPtr = 0x8F000000;
  constexpr size_t kSegLen = 4096;

  // Use kCliqueBBegin — the device ID checks inside
  // configureMocksForImportSuccess will verify prop.location.id and
  // accessDesc.location.id match kCliqueBBegin.
  configureMocksForImportSuccess(
      kCliqueBBegin, kImportedHandle, kGranularity, kMappedPtr);

  CUmemFabricHandle fakeFabricHandle{};
  auto payload = makePayload(fakeFabricHandle);
  auto factory = makeFactory(kCliqueBBegin);
  auto result = factory.importSegment(kSegLen, payload);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
}

// --- registerSegment + serialize → importSegment round-trip across devices ---

TEST_F(NVLinkImportTest, RegisterSerializeImportRoundTrip) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  constexpr CUmemGenericAllocationHandle kAllocHandle = 0xDEAD;
  constexpr CUmemGenericAllocationHandle kImportedHandle = 0xBEEF;
  constexpr size_t kGranularity = 2 * 1024 * 1024;
  constexpr CUdeviceptr kMappedPtr = 0x7F000000;
  constexpr size_t kSegLen = 64;

  CUmemFabricHandle fakeFabricHandle{};
  for (size_t i = 0; i < sizeof(fakeFabricHandle.data); ++i) {
    fakeFabricHandle.data[i] = static_cast<uint8_t>(i ^ 0x55);
  }

  // Mock registerSegment's CUDA calls.
  ON_CALL(*cudaDriverMock_, cuMemRetainAllocationHandle(_, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kAllocHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemExportToShareableHandle(_, _, _, _))
      .WillByDefault(DoAll(
          [&](void* shareableHandle,
              CUmemGenericAllocationHandle,
              CUmemAllocationHandleType,
              unsigned long long) {
            std::memcpy(
                shareableHandle, &fakeFabricHandle, sizeof(fakeFabricHandle));
          },
          Return(Ok())));

  // Register on device in clique A and serialize.
  auto exportFactory = makeFactory(kCliqueABegin);
  uint8_t buf[kSegLen];
  Segment seg(buf, sizeof(buf), MemoryType::VRAM, kCliqueABegin);
  auto regResult = exportFactory.registerSegment(seg);
  ASSERT_TRUE(regResult.hasValue()) << regResult.error().message();
  auto serialized = regResult.value()->serialize();

  // Mock importSegment's VMM CUDA calls — import on device in clique B.
  configureMocksForImportSuccess(
      kCliqueBBegin, kImportedHandle, kGranularity, kMappedPtr);

  // Import on a different device using the serialized payload.
  auto importFactory = makeFactory(kCliqueBBegin);
  auto importResult = importFactory.importSegment(kSegLen, serialized);
  ASSERT_TRUE(importResult.hasValue()) << importResult.error().message();
  EXPECT_NE(
      dynamic_cast<NVLinkRemoteRegistrationHandle*>(importResult.value().get()),
      nullptr);
}

// --- importSegment fails when cuMemImportFromShareableHandle fails ---

TEST_F(NVLinkImportTest, FailsWhenImportHandleFails) {
  using ::testing::_;
  using ::testing::Return;

  EXPECT_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
      .Times(1)
      .WillOnce(Return(Err(ErrCode::DriverError, "import failed")));

  CUmemFabricHandle fakeFabricHandle{};
  auto payload = makePayload(fakeFabricHandle);
  auto factory = makeFactory(kCliqueABegin);
  auto result = factory.importSegment(4096, payload);
  ASSERT_TRUE(result.hasError());
}

// --- importSegment cleans up on cuMemGetAllocationGranularity failure ---

TEST_F(NVLinkImportTest, FailsWhenGranularityFails_ReleasesHandle) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  CUmemGenericAllocationHandle fakeHandle = 0xAAAA;
  ON_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(fakeHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemGetAllocationGranularity(_, _, _))
      .WillByDefault(Return(Err(ErrCode::DriverError, "granularity failed")));

  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(fakeHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  CUmemFabricHandle fakeFabricHandle{};
  auto payload = makePayload(fakeFabricHandle);
  auto factory = makeFactory(kCliqueABegin);
  auto result = factory.importSegment(4096, payload);
  ASSERT_TRUE(result.hasError());
}

// --- importSegment cleans up on cuMemAddressReserve failure ---

TEST_F(NVLinkImportTest, FailsWhenReserveFails_ReleasesHandle) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  CUmemGenericAllocationHandle fakeHandle = 0xBBBB;
  ON_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(fakeHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemGetAllocationGranularity(_, _, _))
      .WillByDefault(
          DoAll(SetArgPointee<0>(size_t{2 * 1024 * 1024}), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemAddressReserve(_, _, _, _, _))
      .WillByDefault(Return(Err(ErrCode::DriverError, "reserve failed")));

  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(fakeHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  CUmemFabricHandle fakeFabricHandle{};
  auto payload = makePayload(fakeFabricHandle);
  auto factory = makeFactory(kCliqueABegin);
  auto result = factory.importSegment(4096, payload);
  ASSERT_TRUE(result.hasError());
}

// --- importSegment cleans up on cuMemMap failure ---

TEST_F(NVLinkImportTest, FailsWhenMapFails_CleansUpReserveAndRelease) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  constexpr CUmemGenericAllocationHandle kHandle = 0xCCCC;
  constexpr CUdeviceptr kPtr = 0x7F000000;
  constexpr size_t kGranularity = 2 * 1024 * 1024;

  ON_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemGetAllocationGranularity(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kGranularity), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemAddressReserve(_, _, _, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kPtr), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemMap(_, _, _, _, _))
      .WillByDefault(Return(Err(ErrCode::DriverError, "map failed")));

  EXPECT_CALL(*cudaDriverMock_, cuMemAddressFree(kPtr, kGranularity))
      .Times(1)
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(kHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  CUmemFabricHandle fakeFabricHandle{};
  auto payload = makePayload(fakeFabricHandle);
  auto factory = makeFactory(kCliqueABegin);
  auto result = factory.importSegment(4096, payload);
  ASSERT_TRUE(result.hasError());
}

// --- importSegment cleans up on cuMemSetAccess failure ---

TEST_F(NVLinkImportTest, FailsWhenSetAccessFails_CleansUpAll) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  constexpr CUmemGenericAllocationHandle kHandle = 0xDDDD;
  constexpr CUdeviceptr kPtr = 0x7F000000;
  constexpr size_t kGranularity = 2 * 1024 * 1024;

  ON_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemGetAllocationGranularity(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kGranularity), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemAddressReserve(_, _, _, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kPtr), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemMap(_, _, _, _, _))
      .WillByDefault(Return(Ok()));
  ON_CALL(*cudaDriverMock_, cuMemSetAccess(_, _, _, _))
      .WillByDefault(Return(Err(ErrCode::DriverError, "set access failed")));

  EXPECT_CALL(*cudaDriverMock_, cuMemUnmap(kPtr, kGranularity))
      .Times(1)
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaDriverMock_, cuMemAddressFree(kPtr, kGranularity))
      .Times(1)
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(kHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  CUmemFabricHandle fakeFabricHandle{};
  auto payload = makePayload(fakeFabricHandle);
  auto factory = makeFactory(kCliqueABegin);
  auto result = factory.importSegment(4096, payload);
  ASSERT_TRUE(result.hasError());
}

// ---------------------------------------------------------------------------
// NVLinkTransportPutGetTest — tests for put/get using MockCudaApi +
// ScopedEventBaseThread
// ---------------------------------------------------------------------------

class NVLinkTransportPutGetTest : public ::testing::Test {
 protected:
  static constexpr int dev0_ = kCliqueABegin;
  static constexpr int dev1_ = kCliqueABegin + 1;

  void SetUp() override {
    cudaApiMock_ = std::make_shared<::testing::NiceMock<MockCudaApi>>();
    cudaDriverMock_ =
        std::make_shared<::testing::NiceMock<MockCudaDriverApi>>();
    transport_ = std::make_unique<NVLinkTransport>(
        /*deviceId=*/dev0_, evbThread_.getEventBase(), cudaApiMock_);

    // Default: setDevice/getDevice succeed.
    ON_CALL(*cudaApiMock_, setDevice(::testing::_))
        .WillByDefault(::testing::Return(Ok()));
    ON_CALL(*cudaApiMock_, getDevice())
        .WillByDefault(::testing::Return(Result<int>(0)));

    // Use a fake "remote VA" (0xDEAD0000) as buf_ to simulate a real remote
    // peer's address. The actual local address (remoteBuf_) is provided via
    // the handle's mappedPtr. This ensures tests fail if put/get use buf_
    // instead of mappedPtr.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    remoteSeg_ = SegmentTest::makeRemote(
        reinterpret_cast<void*>(0xDEAD0000),
        sizeof(remoteBuf_),
        std::make_unique<NVLinkRemoteRegistrationHandle>(
            /*allocHandle=*/0,
            reinterpret_cast<CUdeviceptr>(remoteBuf_),
            sizeof(remoteBuf_),
            cudaDriverMock_));
  }

  void TearDown() override {
    remoteSeg_.reset();
    transport_.reset();
    cudaApiMock_.reset();
    cudaDriverMock_.reset();
  }

  /// Connect transport with peer device.
  void connectTransport() {
    TransportInfo peerInfo(sizeof(int32_t));
    int32_t peerId = dev1_;
    std::memcpy(peerInfo.data(), &peerId, sizeof(peerId));
    auto status = transport_->connect(peerInfo);
    ASSERT_TRUE(status.hasValue());
  }

  ScopedEventBaseThread evbThread_{"NVLinkPutGetTest"};
  std::shared_ptr<::testing::NiceMock<MockCudaApi>> cudaApiMock_;
  std::shared_ptr<::testing::NiceMock<MockCudaDriverApi>> cudaDriverMock_;
  std::unique_ptr<NVLinkTransport> transport_;

  // Fake buffers and segments for transfer requests.
  uint8_t localBuf_[128]{};
  uint8_t remoteBuf_[128]{};
  RegisteredSegment localSeg_{SegmentTest::makeRegisteredSegment(
      localBuf_,
      sizeof(localBuf_),
      MemoryType::VRAM,
      dev0_)};
  std::optional<RemoteRegisteredSegment> remoteSeg_;
};

TEST_F(NVLinkTransportPutGetTest, PutRejectsNotConnected) {
  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::NotConnected);
}

TEST_F(NVLinkTransportPutGetTest, GetRejectsNotConnected) {
  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->get(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::NotConnected);
}

TEST_F(NVLinkTransportPutGetTest, PutDispatchesMemcpyAndEvent) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;

  connectTransport();

  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x42);

  // put: dst = remote, src = local.
  EXPECT_CALL(
      *cudaApiMock_,
      memcpyAsync(
          remoteBuf_,
          localBuf_,
          sizeof(localBuf_),
          cudaMemcpyDeviceToDevice,
          nullptr))
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(DoAll(::testing::SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*cudaApiMock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventQuery(fakeEvent))
      .WillOnce(Return(Result<bool>(true)));
  EXPECT_CALL(*cudaApiMock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();
}

TEST_F(NVLinkTransportPutGetTest, GetDispatchesMemcpyAndEvent) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;

  connectTransport();

  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x43);

  // get: dst = local, src = remote.
  EXPECT_CALL(
      *cudaApiMock_,
      memcpyAsync(
          localBuf_,
          remoteBuf_,
          sizeof(localBuf_),
          cudaMemcpyDeviceToDevice,
          nullptr))
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(DoAll(::testing::SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*cudaApiMock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventQuery(fakeEvent))
      .WillOnce(Return(Result<bool>(true)));
  EXPECT_CALL(*cudaApiMock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->get(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();
}

TEST_F(NVLinkTransportPutGetTest, PutHandlesMemcpyError) {
  using ::testing::_;
  using ::testing::Return;

  connectTransport();

  EXPECT_CALL(
      *cudaApiMock_,
      memcpyAsync(
          remoteBuf_,
          localBuf_,
          sizeof(localBuf_),
          cudaMemcpyDeviceToDevice,
          nullptr))
      .WillOnce(Return(Err(ErrCode::DriverError, "memcpy failed")));
  // eventCreate should NOT be called when memcpy fails.
  EXPECT_CALL(*cudaApiMock_, eventCreate(_)).Times(0);

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::DriverError);
}

TEST_F(NVLinkTransportPutGetTest, PutPollsUntilEventComplete) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;

  connectTransport();

  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x45);

  EXPECT_CALL(
      *cudaApiMock_,
      memcpyAsync(
          remoteBuf_,
          localBuf_,
          sizeof(localBuf_),
          cudaMemcpyDeviceToDevice,
          nullptr))
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(DoAll(::testing::SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*cudaApiMock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  // First two queries return not-ready, third returns complete.
  EXPECT_CALL(*cudaApiMock_, eventQuery(fakeEvent))
      .WillOnce(Return(Result<bool>(false)))
      .WillOnce(Return(Result<bool>(false)))
      .WillOnce(Return(Result<bool>(true)));
  EXPECT_CALL(*cudaApiMock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();
}

// --- buffer size mismatch ---

TEST_F(NVLinkTransportPutGetTest, PutRejectsMismatchedBufferSizes) {
  connectTransport();

  auto localSeg = SegmentTest::makeRegisteredSegment(
      localBuf_, sizeof(localBuf_), MemoryType::VRAM, 0);
  // Remote buffer is smaller than local.
  auto remoteSeg = SegmentTest::makeRemoteRegisteredSegment(
      remoteBuf_, sizeof(remoteBuf_) / 2, MemoryType::VRAM, 1);

  TransferRequest req{localSeg.span(), remoteSeg.span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_F(NVLinkTransportPutGetTest, GetRejectsMismatchedBufferSizes) {
  connectTransport();

  // Local buffer is smaller than remote.
  auto localSeg = SegmentTest::makeRegisteredSegment(
      localBuf_, sizeof(localBuf_) / 2, MemoryType::VRAM, 0);
  auto remoteSeg = SegmentTest::makeRemoteRegisteredSegment(
      remoteBuf_, sizeof(remoteBuf_), MemoryType::VRAM, 1);

  TransferRequest req{localSeg.span(), remoteSeg.span()};
  auto future = transport_->get(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

// --- empty requests ---

TEST_F(NVLinkTransportPutGetTest, PutEmptyRequestsSucceeds) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;

  connectTransport();

  EXPECT_CALL(*cudaApiMock_, memcpyAsync(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*cudaApiMock_, eventCreate(_)).Times(0);
  EXPECT_CALL(*cudaApiMock_, eventRecord(_, _)).Times(0);
  EXPECT_CALL(*cudaApiMock_, eventQuery(_)).Times(0);
  EXPECT_CALL(*cudaApiMock_, eventDestroy(_)).Times(0);

  std::span<TransferRequest> empty;
  auto future = transport_->put(empty);
  auto status = future.get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();
}

TEST_F(NVLinkTransportPutGetTest, GetEmptyRequestsSucceeds) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;

  connectTransport();

  EXPECT_CALL(*cudaApiMock_, memcpyAsync(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*cudaApiMock_, eventCreate(_)).Times(0);
  EXPECT_CALL(*cudaApiMock_, eventRecord(_, _)).Times(0);
  EXPECT_CALL(*cudaApiMock_, eventQuery(_)).Times(0);
  EXPECT_CALL(*cudaApiMock_, eventDestroy(_)).Times(0);

  std::span<TransferRequest> empty;
  auto future = transport_->get(empty);
  auto status = future.get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();
}

// --- data correctness ---

TEST_F(NVLinkTransportPutGetTest, PutCopiesDataFromLocalToRemote) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Invoke;
  using ::testing::Return;

  connectTransport();

  // Fill local buffer with a known pattern; remote starts zeroed.
  for (size_t i = 0; i < sizeof(localBuf_); ++i) {
    localBuf_[i] = static_cast<uint8_t>(i ^ 0xA5);
  }
  std::memset(remoteBuf_, 0, sizeof(remoteBuf_));

  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x50);

  // Mock memcpyAsync to perform a real memcpy, verifying dst/src pointers.
  EXPECT_CALL(
      *cudaApiMock_,
      memcpyAsync(
          remoteBuf_,
          localBuf_,
          sizeof(localBuf_),
          cudaMemcpyDeviceToDevice,
          nullptr))
      .WillOnce(Invoke(
          [](void* dst,
             const void* src,
             size_t count,
             cudaMemcpyKind,
             cudaStream_t) -> Status {
            std::memcpy(dst, src, count);
            return Ok();
          }));
  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(DoAll(::testing::SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*cudaApiMock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventQuery(fakeEvent))
      .WillOnce(Return(Result<bool>(false)))
      .WillOnce(Return(Result<bool>(true)));
  EXPECT_CALL(*cudaApiMock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();

  // Verify remote buffer now matches local buffer.
  EXPECT_EQ(std::memcmp(localBuf_, remoteBuf_, sizeof(localBuf_)), 0);
}

TEST_F(NVLinkTransportPutGetTest, GetCopiesDataFromRemoteToLocal) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Invoke;
  using ::testing::Return;

  connectTransport();

  // Fill remote buffer with a known pattern; local starts zeroed.
  for (size_t i = 0; i < sizeof(remoteBuf_); ++i) {
    remoteBuf_[i] = static_cast<uint8_t>(i ^ 0x5A);
  }
  std::memset(localBuf_, 0, sizeof(localBuf_));

  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x51);

  // Mock memcpyAsync to perform a real memcpy, verifying dst/src pointers.
  EXPECT_CALL(
      *cudaApiMock_,
      memcpyAsync(
          localBuf_,
          remoteBuf_,
          sizeof(remoteBuf_),
          cudaMemcpyDeviceToDevice,
          nullptr))
      .WillOnce(Invoke(
          [](void* dst,
             const void* src,
             size_t count,
             cudaMemcpyKind,
             cudaStream_t) -> Status {
            std::memcpy(dst, src, count);
            return Ok();
          }));
  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(DoAll(::testing::SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*cudaApiMock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventQuery(fakeEvent))
      .WillOnce(Return(Result<bool>(false)))
      .WillOnce(Return(Result<bool>(true)));
  EXPECT_CALL(*cudaApiMock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->get(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();

  // Verify local buffer now matches remote buffer.
  EXPECT_EQ(std::memcmp(localBuf_, remoteBuf_, sizeof(remoteBuf_)), 0);
}

// --- transfer error paths: eventCreate / eventRecord / eventQuery failures ---

TEST_F(NVLinkTransportPutGetTest, PutHandlesEventCreateError) {
  using ::testing::_;
  using ::testing::Return;

  connectTransport();

  EXPECT_CALL(*cudaApiMock_, memcpyAsync(_, _, _, cudaMemcpyDeviceToDevice, _))
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(Return(Err(ErrCode::DriverError, "eventCreate failed")));
  EXPECT_CALL(*cudaApiMock_, eventRecord(_, _)).Times(0);
  EXPECT_CALL(*cudaApiMock_, eventQuery(_)).Times(0);
  EXPECT_CALL(*cudaApiMock_, eventDestroy(_)).Times(0);

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::DriverError);
}

TEST_F(NVLinkTransportPutGetTest, PutHandlesEventRecordError) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;

  connectTransport();

  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x60);

  EXPECT_CALL(*cudaApiMock_, memcpyAsync(_, _, _, cudaMemcpyDeviceToDevice, _))
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(DoAll(::testing::SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*cudaApiMock_, eventRecord(fakeEvent, _))
      .WillOnce(Return(Err(ErrCode::DriverError, "eventRecord failed")));
  // The event should be destroyed as cleanup.
  EXPECT_CALL(*cudaApiMock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventQuery(_)).Times(0);

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::DriverError);
}

TEST_F(NVLinkTransportPutGetTest, PutHandlesEventQueryError) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;

  connectTransport();

  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x61);

  EXPECT_CALL(*cudaApiMock_, memcpyAsync(_, _, _, cudaMemcpyDeviceToDevice, _))
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(DoAll(::testing::SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*cudaApiMock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  // eventQuery returns an error status (not false — a hard failure).
  EXPECT_CALL(*cudaApiMock_, eventQuery(fakeEvent))
      .WillOnce(Return(Err(ErrCode::DriverError, "eventQuery failed")));
  // The event should be destroyed as cleanup.
  EXPECT_CALL(*cudaApiMock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::DriverError);
}

TEST_F(NVLinkTransportPutGetTest, PutHandlesEventQueryErrorAfterPolling) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;

  connectTransport();

  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x62);

  EXPECT_CALL(*cudaApiMock_, memcpyAsync(_, _, _, cudaMemcpyDeviceToDevice, _))
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(DoAll(::testing::SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*cudaApiMock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  // First poll returns not-ready, second returns error.
  EXPECT_CALL(*cudaApiMock_, eventQuery(fakeEvent))
      .WillOnce(Return(Result<bool>(false)))
      .WillOnce(Return(Err(ErrCode::DriverError, "eventQuery failed")));
  EXPECT_CALL(*cudaApiMock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::DriverError);
}

TEST_F(NVLinkTransportPutGetTest, PutHandlesEventDestroyError) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;

  connectTransport();

  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x63);

  EXPECT_CALL(*cudaApiMock_, memcpyAsync(_, _, _, cudaMemcpyDeviceToDevice, _))
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(DoAll(::testing::SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*cudaApiMock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventQuery(fakeEvent))
      .WillOnce(Return(Result<bool>(true)));
  // Event completed successfully, but eventDestroy fails on cleanup.
  EXPECT_CALL(*cudaApiMock_, eventDestroy(fakeEvent))
      .WillOnce(Return(Err(ErrCode::DriverError, "eventDestroy failed")));

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::DriverError);
}

// --- multiple transfer requests with data correctness ---

TEST_F(NVLinkTransportPutGetTest, PutMultipleRequestsCopiesAllData) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Invoke;
  using ::testing::Return;

  connectTransport();

  // Split the 128-byte buffers into 3 sub-regions of different sizes.
  constexpr size_t kLen0 = 32;
  constexpr size_t kLen1 = 48;
  constexpr size_t kLen2 = 48;
  static_assert(kLen0 + kLen1 + kLen2 == 128);

  // Fill local buffer with distinct patterns per region.
  for (size_t i = 0; i < kLen0; ++i) {
    localBuf_[i] = static_cast<uint8_t>(i ^ 0xA1);
  }
  for (size_t i = 0; i < kLen1; ++i) {
    localBuf_[kLen0 + i] = static_cast<uint8_t>(i ^ 0xB2);
  }
  for (size_t i = 0; i < kLen2; ++i) {
    localBuf_[kLen0 + kLen1 + i] = static_cast<uint8_t>(i ^ 0xC3);
  }
  std::memset(remoteBuf_, 0, sizeof(remoteBuf_));

  // Build 3 registered/remote segment sub-spans.
  auto localSeg0 = SegmentTest::makeRegisteredSegment(
      localBuf_, kLen0, MemoryType::VRAM, dev0_);
  auto localSeg1 = SegmentTest::makeRegisteredSegment(
      localBuf_ + kLen0, kLen1, MemoryType::VRAM, dev0_);
  auto localSeg2 = SegmentTest::makeRegisteredSegment(
      localBuf_ + kLen0 + kLen1, kLen2, MemoryType::VRAM, dev0_);
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto remoteSeg0 = SegmentTest::makeRemote(
      reinterpret_cast<void*>(0xAA000000),
      kLen0,
      std::make_unique<NVLinkRemoteRegistrationHandle>(
          0,
          reinterpret_cast<CUdeviceptr>(remoteBuf_),
          kLen0,
          cudaDriverMock_));
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto remoteSeg1 = SegmentTest::makeRemote(
      reinterpret_cast<void*>(0xBB000000),
      kLen1,
      std::make_unique<NVLinkRemoteRegistrationHandle>(
          0,
          reinterpret_cast<CUdeviceptr>(remoteBuf_ + kLen0),
          kLen1,
          cudaDriverMock_));
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto remoteSeg2 = SegmentTest::makeRemote(
      reinterpret_cast<void*>(0xCC000000),
      kLen2,
      std::make_unique<NVLinkRemoteRegistrationHandle>(
          0,
          reinterpret_cast<CUdeviceptr>(remoteBuf_ + kLen0 + kLen1),
          kLen2,
          cudaDriverMock_));

  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x70);

  // Expect 3 memcpyAsync calls in order, put: dst=remote, src=local.
  {
    ::testing::InSequence seq;
    EXPECT_CALL(
        *cudaApiMock_,
        memcpyAsync(
            remoteBuf_, localBuf_, kLen0, cudaMemcpyDeviceToDevice, nullptr))
        .WillOnce(Invoke(
            [](void* dst,
               const void* src,
               size_t count,
               cudaMemcpyKind,
               cudaStream_t) -> Status {
              std::memcpy(dst, src, count);
              return Ok();
            }));
    EXPECT_CALL(
        *cudaApiMock_,
        memcpyAsync(
            remoteBuf_ + kLen0,
            localBuf_ + kLen0,
            kLen1,
            cudaMemcpyDeviceToDevice,
            nullptr))
        .WillOnce(Invoke(
            [](void* dst,
               const void* src,
               size_t count,
               cudaMemcpyKind,
               cudaStream_t) -> Status {
              std::memcpy(dst, src, count);
              return Ok();
            }));
    EXPECT_CALL(
        *cudaApiMock_,
        memcpyAsync(
            remoteBuf_ + kLen0 + kLen1,
            localBuf_ + kLen0 + kLen1,
            kLen2,
            cudaMemcpyDeviceToDevice,
            nullptr))
        .WillOnce(Invoke(
            [](void* dst,
               const void* src,
               size_t count,
               cudaMemcpyKind,
               cudaStream_t) -> Status {
              std::memcpy(dst, src, count);
              return Ok();
            }));
  }

  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(DoAll(::testing::SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*cudaApiMock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  // Poll: not-ready twice, then complete.
  EXPECT_CALL(*cudaApiMock_, eventQuery(fakeEvent))
      .WillOnce(Return(Result<bool>(false)))
      .WillOnce(Return(Result<bool>(false)))
      .WillOnce(Return(Result<bool>(true)));
  EXPECT_CALL(*cudaApiMock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  std::array<TransferRequest, 3> reqs{
      TransferRequest{localSeg0.span(), remoteSeg0.span()},
      TransferRequest{localSeg1.span(), remoteSeg1.span()},
      TransferRequest{localSeg2.span(), remoteSeg2.span()},
  };
  auto future = transport_->put(reqs);
  auto status = future.get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();

  // Verify all remote regions match their local counterparts.
  EXPECT_EQ(std::memcmp(localBuf_, remoteBuf_, sizeof(localBuf_)), 0);
}

TEST_F(NVLinkTransportPutGetTest, PutSubSpanTransfersPartialData) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Invoke;
  using ::testing::Return;

  connectTransport();

  constexpr size_t kLen0 = 32;
  constexpr size_t kLen1 = 48;
  constexpr size_t kLen2 = 48;
  static_assert(kLen0 + kLen1 + kLen2 == 128);

  for (size_t i = 0; i < sizeof(localBuf_); ++i) {
    localBuf_[i] = static_cast<uint8_t>(i ^ 0xA1);
  }
  std::memset(remoteBuf_, 0, sizeof(remoteBuf_));

  auto localSeg0 = SegmentTest::makeRegisteredSegment(
      localBuf_, kLen0, MemoryType::VRAM, dev0_);
  auto localSeg1 = SegmentTest::makeRegisteredSegment(
      localBuf_ + kLen0, kLen1, MemoryType::VRAM, dev0_);
  auto localSeg2 = SegmentTest::makeRegisteredSegment(
      localBuf_ + kLen0 + kLen1, kLen2, MemoryType::VRAM, dev0_);
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto remoteSeg0 = SegmentTest::makeRemote(
      reinterpret_cast<void*>(0xAA000000),
      kLen0,
      std::make_unique<NVLinkRemoteRegistrationHandle>(
          0,
          reinterpret_cast<CUdeviceptr>(remoteBuf_),
          kLen0,
          cudaDriverMock_));
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto remoteSeg1 = SegmentTest::makeRemote(
      reinterpret_cast<void*>(0xBB000000),
      kLen1,
      std::make_unique<NVLinkRemoteRegistrationHandle>(
          0,
          reinterpret_cast<CUdeviceptr>(remoteBuf_ + kLen0),
          kLen1,
          cudaDriverMock_));
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto remoteSeg2 = SegmentTest::makeRemote(
      reinterpret_cast<void*>(0xCC000000),
      kLen2,
      std::make_unique<NVLinkRemoteRegistrationHandle>(
          0,
          reinterpret_cast<CUdeviceptr>(remoteBuf_ + kLen0 + kLen1),
          kLen2,
          cudaDriverMock_));

  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x72);

  // Sub-spans: only transfer the first half of each region.
  {
    ::testing::InSequence seq;
    EXPECT_CALL(
        *cudaApiMock_,
        memcpyAsync(
            remoteBuf_,
            localBuf_,
            kLen0 / 2,
            cudaMemcpyDeviceToDevice,
            nullptr))
        .WillOnce(Invoke(
            [](void* dst,
               const void* src,
               size_t count,
               cudaMemcpyKind,
               cudaStream_t) -> Status {
              std::memcpy(dst, src, count);
              return Ok();
            }));
    EXPECT_CALL(
        *cudaApiMock_,
        memcpyAsync(
            remoteBuf_ + kLen0,
            localBuf_ + kLen0,
            kLen1 / 2,
            cudaMemcpyDeviceToDevice,
            nullptr))
        .WillOnce(Invoke(
            [](void* dst,
               const void* src,
               size_t count,
               cudaMemcpyKind,
               cudaStream_t) -> Status {
              std::memcpy(dst, src, count);
              return Ok();
            }));
    EXPECT_CALL(
        *cudaApiMock_,
        memcpyAsync(
            remoteBuf_ + kLen0 + kLen1,
            localBuf_ + kLen0 + kLen1,
            kLen2 / 2,
            cudaMemcpyDeviceToDevice,
            nullptr))
        .WillOnce(Invoke(
            [](void* dst,
               const void* src,
               size_t count,
               cudaMemcpyKind,
               cudaStream_t) -> Status {
              std::memcpy(dst, src, count);
              return Ok();
            }));
  }

  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(DoAll(::testing::SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*cudaApiMock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaApiMock_, eventQuery(fakeEvent))
      .WillOnce(Return(Result<bool>(true)));
  EXPECT_CALL(*cudaApiMock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  std::array<TransferRequest, 3> reqs{
      TransferRequest{
          localSeg0.span(size_t{0}, kLen0 / 2),
          remoteSeg0.span(size_t{0}, kLen0 / 2)},
      TransferRequest{
          localSeg1.span(size_t{0}, kLen1 / 2),
          remoteSeg1.span(size_t{0}, kLen1 / 2)},
      TransferRequest{
          localSeg2.span(size_t{0}, kLen2 / 2),
          remoteSeg2.span(size_t{0}, kLen2 / 2)},
  };
  auto future = transport_->put(reqs);
  auto status = future.get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();
}

TEST_F(NVLinkTransportPutGetTest, GetMultipleRequestsCopiesAllData) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Invoke;
  using ::testing::Return;

  connectTransport();

  // Split the 128-byte buffers into 3 sub-regions of different sizes.
  constexpr size_t kLen0 = 32;
  constexpr size_t kLen1 = 48;
  constexpr size_t kLen2 = 48;
  static_assert(kLen0 + kLen1 + kLen2 == 128);

  // Fill remote buffer with distinct patterns per region.
  for (size_t i = 0; i < kLen0; ++i) {
    remoteBuf_[i] = static_cast<uint8_t>(i ^ 0xD4);
  }
  for (size_t i = 0; i < kLen1; ++i) {
    remoteBuf_[kLen0 + i] = static_cast<uint8_t>(i ^ 0xE5);
  }
  for (size_t i = 0; i < kLen2; ++i) {
    remoteBuf_[kLen0 + kLen1 + i] = static_cast<uint8_t>(i ^ 0xF6);
  }
  std::memset(localBuf_, 0, sizeof(localBuf_));

  // Build 3 registered/remote segment sub-spans.
  auto localSeg0 = SegmentTest::makeRegisteredSegment(
      localBuf_, kLen0, MemoryType::VRAM, dev0_);
  auto localSeg1 = SegmentTest::makeRegisteredSegment(
      localBuf_ + kLen0, kLen1, MemoryType::VRAM, dev0_);
  auto localSeg2 = SegmentTest::makeRegisteredSegment(
      localBuf_ + kLen0 + kLen1, kLen2, MemoryType::VRAM, dev0_);
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto remoteSeg0 = SegmentTest::makeRemote(
      reinterpret_cast<void*>(0xAA000000),
      kLen0,
      std::make_unique<NVLinkRemoteRegistrationHandle>(
          0,
          reinterpret_cast<CUdeviceptr>(remoteBuf_),
          kLen0,
          cudaDriverMock_));
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto remoteSeg1 = SegmentTest::makeRemote(
      reinterpret_cast<void*>(0xBB000000),
      kLen1,
      std::make_unique<NVLinkRemoteRegistrationHandle>(
          0,
          reinterpret_cast<CUdeviceptr>(remoteBuf_ + kLen0),
          kLen1,
          cudaDriverMock_));
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto remoteSeg2 = SegmentTest::makeRemote(
      reinterpret_cast<void*>(0xCC000000),
      kLen2,
      std::make_unique<NVLinkRemoteRegistrationHandle>(
          0,
          reinterpret_cast<CUdeviceptr>(remoteBuf_ + kLen0 + kLen1),
          kLen2,
          cudaDriverMock_));

  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x71);

  // Expect 3 memcpyAsync calls in order, get: dst=local, src=remote.
  {
    ::testing::InSequence seq;
    EXPECT_CALL(
        *cudaApiMock_,
        memcpyAsync(
            localBuf_, remoteBuf_, kLen0, cudaMemcpyDeviceToDevice, nullptr))
        .WillOnce(Invoke(
            [](void* dst,
               const void* src,
               size_t count,
               cudaMemcpyKind,
               cudaStream_t) -> Status {
              std::memcpy(dst, src, count);
              return Ok();
            }));
    EXPECT_CALL(
        *cudaApiMock_,
        memcpyAsync(
            localBuf_ + kLen0,
            remoteBuf_ + kLen0,
            kLen1,
            cudaMemcpyDeviceToDevice,
            nullptr))
        .WillOnce(Invoke(
            [](void* dst,
               const void* src,
               size_t count,
               cudaMemcpyKind,
               cudaStream_t) -> Status {
              std::memcpy(dst, src, count);
              return Ok();
            }));
    EXPECT_CALL(
        *cudaApiMock_,
        memcpyAsync(
            localBuf_ + kLen0 + kLen1,
            remoteBuf_ + kLen0 + kLen1,
            kLen2,
            cudaMemcpyDeviceToDevice,
            nullptr))
        .WillOnce(Invoke(
            [](void* dst,
               const void* src,
               size_t count,
               cudaMemcpyKind,
               cudaStream_t) -> Status {
              std::memcpy(dst, src, count);
              return Ok();
            }));
  }

  EXPECT_CALL(*cudaApiMock_, eventCreate(_))
      .WillOnce(DoAll(::testing::SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*cudaApiMock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  // Poll: not-ready three times, then complete.
  EXPECT_CALL(*cudaApiMock_, eventQuery(fakeEvent))
      .WillOnce(Return(Result<bool>(false)))
      .WillOnce(Return(Result<bool>(false)))
      .WillOnce(Return(Result<bool>(false)))
      .WillOnce(Return(Result<bool>(true)));
  EXPECT_CALL(*cudaApiMock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  std::array<TransferRequest, 3> reqs{
      TransferRequest{localSeg0.span(), remoteSeg0.span()},
      TransferRequest{localSeg1.span(), remoteSeg1.span()},
      TransferRequest{localSeg2.span(), remoteSeg2.span()},
  };
  auto future = transport_->get(std::span(reqs));
  auto status = future.get();
  ASSERT_TRUE(status.hasValue()) << status.error().message();

  // Verify all local regions match their remote counterparts.
  EXPECT_EQ(std::memcmp(localBuf_, remoteBuf_, sizeof(remoteBuf_)), 0);
}

// ---------------------------------------------------------------------------
// NVLinkFdTest — tests for FD (POSIX file descriptor) mode using
// MockCudaDriverApi. The factory is configured to report fabricSupported=0
// so it selects MemSharingMode::PosixFd. FD import tests use real eventfd
// and pidfd syscalls (same process) instead of mocking.
// ---------------------------------------------------------------------------

class NVLinkFdTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    nvmlMock_ = std::make_shared<::testing::NiceMock<MockNvmlApi>>();
    configureMultiGroupMock(*nvmlMock_);
  }

  static void TearDownTestSuite() {
    nvmlMock_.reset();
  }

  void SetUp() override {
    cudaDriverMock_ =
        std::make_shared<::testing::NiceMock<MockCudaDriverApi>>();
    cudaApiMock_ = std::make_shared<::testing::NiceMock<MockCudaApi>>();

    // Device context management (needed by CudaDeviceGuard in registerSegment).
    ON_CALL(*cudaApiMock_, getDevice())
        .WillByDefault(::testing::Return(Result<int>(0)));
    ON_CALL(*cudaApiMock_, setDevice(::testing::_))
        .WillByDefault(::testing::Return(Ok()));

    configureFdMode();
  }

  void TearDown() override {
    cudaDriverMock_.reset();
    cudaApiMock_.reset();
  }

  void configureFdMode() {
    using ::testing::_;
    using ::testing::Return;

    ON_CALL(*cudaDriverMock_, getCuMemHandleType())
        .WillByDefault(Return(
            Result<CUmemAllocationHandleType>(
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)));
    ON_CALL(*cudaApiMock_, deviceCanAccessPeer(_, _))
        .WillByDefault(Return(Result<bool>(true)));
  }

  NVLinkTransportFactory makeFactory(int device) {
    return NVLinkTransportFactory(
        device,
        evbThread_.getEventBase(),
        nvmlMock_,
        cudaApiMock_,
        cudaDriverMock_);
  }

  static std::vector<uint8_t>
  makeFdPayload(int32_t fd, int32_t pid, uint64_t size) {
    NVLinkFdRegistrationHandle::Payload p{
        MemSharingMode::PosixFd, fd, pid, size};
    std::vector<uint8_t> buf(sizeof(p));
    std::memcpy(buf.data(), &p, sizeof(p));
    return buf;
  }

  /// Configure mocks so that the full fabric probe pipeline succeeds in the
  /// NVLinkTransportFactory constructor (cuMemCreate → export → import →
  /// reserve → map → setAccess → cleanup).
  void configureMocksForFabricProbe() {
    using ::testing::_;
    using ::testing::DoAll;
    using ::testing::Return;
    using ::testing::SetArgPointee;

    constexpr CUmemGenericAllocationHandle kProbeHandle = 0xFAB0;
    constexpr CUmemGenericAllocationHandle kImportedProbeHandle = 0xFAB1;
    constexpr size_t kProbeGranularity = 2 * 1024 * 1024;
    constexpr CUdeviceptr kProbePtr = 0x7E000000;

    ON_CALL(*cudaDriverMock_, cuMemGetAllocationGranularity(_, _, _))
        .WillByDefault(
            DoAll(SetArgPointee<0>(kProbeGranularity), Return(Ok())));
    ON_CALL(*cudaDriverMock_, cuMemCreate(_, _, _, _))
        .WillByDefault(DoAll(SetArgPointee<0>(kProbeHandle), Return(Ok())));
    ON_CALL(*cudaDriverMock_, cuMemExportToShareableHandle(_, _, _, _))
        .WillByDefault(Return(Ok()));
    ON_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
        .WillByDefault(
            DoAll(SetArgPointee<0>(kImportedProbeHandle), Return(Ok())));
    ON_CALL(*cudaDriverMock_, cuMemAddressReserve(_, _, _, _, _))
        .WillByDefault(DoAll(SetArgPointee<0>(kProbePtr), Return(Ok())));
    ON_CALL(*cudaDriverMock_, cuMemMap(_, _, _, _, _))
        .WillByDefault(Return(Ok()));
    ON_CALL(*cudaDriverMock_, cuMemSetAccess(_, _, _, _))
        .WillByDefault(Return(Ok()));
    ON_CALL(*cudaDriverMock_, cuMemUnmap(_, _)).WillByDefault(Return(Ok()));
    ON_CALL(*cudaDriverMock_, cuMemAddressFree(_, _))
        .WillByDefault(Return(Ok()));
    ON_CALL(*cudaDriverMock_, cuMemRelease(_)).WillByDefault(Return(Ok()));
  }

  void configureMocksForFdImportVmm(
      int deviceId,
      CUmemGenericAllocationHandle importedHandle,
      size_t granularity,
      CUdeviceptr mappedPtr) {
    using ::testing::_;
    using ::testing::DoAll;
    using ::testing::Invoke;
    using ::testing::Return;
    using ::testing::SetArgPointee;

    ON_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
        .WillByDefault(DoAll(SetArgPointee<0>(importedHandle), Return(Ok())));

    ON_CALL(*cudaDriverMock_, cuMemGetAllocationGranularity(_, _, _))
        .WillByDefault(Invoke(
            [deviceId, granularity](
                size_t* out,
                const CUmemAllocationProp* prop,
                CUmemAllocationGranularity_flags) -> Status {
              EXPECT_EQ(prop->location.id, deviceId);
              EXPECT_EQ(prop->location.type, CU_MEM_LOCATION_TYPE_DEVICE);
              *out = granularity;
              return Ok();
            }));

    ON_CALL(*cudaDriverMock_, cuMemAddressReserve(_, _, _, _, _))
        .WillByDefault(DoAll(SetArgPointee<0>(mappedPtr), Return(Ok())));
    ON_CALL(*cudaDriverMock_, cuMemMap(_, _, _, _, _))
        .WillByDefault(Return(Ok()));

    ON_CALL(*cudaDriverMock_, cuMemSetAccess(_, _, _, _))
        .WillByDefault(Invoke(
            [deviceId](
                CUdeviceptr, size_t, const CUmemAccessDesc* desc, size_t count)
                -> Status {
              EXPECT_EQ(count, 1u);
              EXPECT_EQ(desc->location.id, deviceId);
              EXPECT_EQ(desc->location.type, CU_MEM_LOCATION_TYPE_DEVICE);
              EXPECT_EQ(desc->flags, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
              return Ok();
            }));
  }

  // NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
  static std::shared_ptr<::testing::NiceMock<MockNvmlApi>> nvmlMock_;
  ScopedEventBaseThread evbThread_{"NVLinkFdTest"};
  std::shared_ptr<::testing::NiceMock<MockCudaDriverApi>> cudaDriverMock_;
  std::shared_ptr<::testing::NiceMock<MockCudaApi>> cudaApiMock_;
};

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
std::shared_ptr<::testing::NiceMock<MockNvmlApi>> NVLinkFdTest::nvmlMock_;

TEST_F(NVLinkFdTest, FdModeDetected) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  CUmemGenericAllocationHandle fakeAllocHandle = 0xDEAD;
  int efd = eventfd(0, 0);
  ASSERT_GE(efd, 0);

  ON_CALL(*cudaDriverMock_, cuMemRetainAllocationHandle(_, _))
      .WillByDefault(DoAll(SetArgPointee<0>(fakeAllocHandle), Return(Ok())));

  // cuMemGetAddressRange_v2: return segment base and size.
  ON_CALL(*cudaDriverMock_, cuMemGetAddressRange_v2(_, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(CUdeviceptr{0}),
          SetArgPointee<1>(size_t{64}),
          Return(Ok())));

  EXPECT_CALL(
      *cudaDriverMock_,
      cuMemExportToShareableHandle(
          _, _, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, _))
      .WillOnce(DoAll(
          [efd](
              void* shareableHandle,
              CUmemGenericAllocationHandle,
              CUmemAllocationHandleType,
              unsigned long long) {
            std::memcpy(shareableHandle, &efd, sizeof(efd));
          },
          Return(Ok())));

  auto factory = makeFactory(kIntraNodeBegin);
  uint8_t buf[64];
  Segment seg(buf, sizeof(buf), MemoryType::VRAM, kIntraNodeBegin);
  auto result = factory.registerSegment(seg);
  ASSERT_TRUE(result.hasValue()) << result.error().message();

  auto* handle =
      dynamic_cast<NVLinkFdRegistrationHandle*>(result.value().get());
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->exportedFd(), efd);
  EXPECT_EQ(handle->ownerPid(), getpid());
  EXPECT_EQ(handle->allocationSize(), sizeof(buf));
}

TEST_F(NVLinkFdTest, FdRegisterSegmentExportFails_ReleasesHandle) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  CUmemGenericAllocationHandle fakeAllocHandle = 0xBEEF;

  ON_CALL(*cudaDriverMock_, cuMemRetainAllocationHandle(_, _))
      .WillByDefault(DoAll(SetArgPointee<0>(fakeAllocHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemExportToShareableHandle(_, _, _, _))
      .WillByDefault(Return(Err(ErrCode::DriverError, "export FD failed")));

  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(fakeAllocHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  auto factory = makeFactory(kIntraNodeBegin);
  uint8_t buf[64];
  Segment seg(buf, sizeof(buf), MemoryType::VRAM, kIntraNodeBegin);
  auto result = factory.registerSegment(seg);
  ASSERT_TRUE(result.hasError());
}

TEST_F(NVLinkFdTest, FdSerializeProducesCorrectFormat) {
  int fakeFd = 7;
  pid_t fakePid = 5555;
  size_t fakeSize = 0x100000;

  auto handle = NVLinkFdRegistrationHandle(
      /*allocHandle=*/0x1234, fakeFd, fakePid, fakeSize, cudaDriverMock_);
  auto serialized = handle.serialize();
  ASSERT_EQ(serialized.size(), NVLinkFdRegistrationHandle::kSerializedSize);

  EXPECT_EQ(serialized[0], static_cast<uint8_t>(MemSharingMode::PosixFd));

  int32_t fd;
  std::memcpy(&fd, serialized.data() + 1, sizeof(fd));
  EXPECT_EQ(fd, fakeFd);

  int32_t pid;
  std::memcpy(&pid, serialized.data() + 1 + sizeof(fd), sizeof(pid));
  EXPECT_EQ(pid, static_cast<int32_t>(fakePid));

  uint64_t size;
  std::memcpy(
      &size, serialized.data() + 1 + sizeof(fd) + sizeof(pid), sizeof(size));
  EXPECT_EQ(size, static_cast<uint64_t>(fakeSize));
}

TEST_F(NVLinkFdTest, FdHandleDestructorReleasesResources) {
  using ::testing::Return;

  CUmemGenericAllocationHandle fakeAllocHandle = 0x1234;

  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(fakeAllocHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  {
    auto handle = NVLinkFdRegistrationHandle(
        fakeAllocHandle,
        /*exportedFd=*/99,
        /*ownerPid=*/1234,
        /*allocationSize=*/4096,
        cudaDriverMock_);
  }
}

TEST_F(NVLinkFdTest, FdImportSegmentSuccess) {
  constexpr CUmemGenericAllocationHandle kImportedHandle = 0xCAFE;
  constexpr size_t kGranularity = 2 * 1024 * 1024;
  constexpr CUdeviceptr kMappedPtr = 0x7F000000;
  constexpr uint64_t kAllocSize = 4096;

  int efd = eventfd(0, 0);
  ASSERT_GE(efd, 0);

  configureMocksForFdImportVmm(
      kIntraNodeBegin, kImportedHandle, kGranularity, kMappedPtr);

  auto payload = makeFdPayload(efd, getpid(), kAllocSize);
  auto factory = makeFactory(kIntraNodeBegin);
  auto result = factory.importSegment(kAllocSize, payload);
  ::close(efd);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
  EXPECT_NE(
      dynamic_cast<NVLinkRemoteRegistrationHandle*>(result.value().get()),
      nullptr);
}

TEST_F(NVLinkFdTest, FdImportFailsWhenPidfdOpenFails) {
  constexpr int kRemoteFd = 10;
  constexpr int32_t kBogusPid = 2000000000;
  constexpr uint64_t kAllocSize = 4096;

  auto payload = makeFdPayload(kRemoteFd, kBogusPid, kAllocSize);
  auto factory = makeFactory(kIntraNodeBegin);
  auto result = factory.importSegment(kAllocSize, payload);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::DriverError);
}

TEST_F(NVLinkFdTest, FdImportFailsWhenPidfdGetFdFails) {
  constexpr int kBogusRemoteFd = 999999;
  constexpr uint64_t kAllocSize = 4096;

  auto payload = makeFdPayload(kBogusRemoteFd, getpid(), kAllocSize);
  auto factory = makeFactory(kIntraNodeBegin);
  auto result = factory.importSegment(kAllocSize, payload);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::DriverError);
}

TEST_F(NVLinkFdTest, FdImportFailsWhenCuMemImportFails) {
  using ::testing::_;
  using ::testing::Return;

  constexpr uint64_t kAllocSize = 4096;

  int efd = eventfd(0, 0);
  ASSERT_GE(efd, 0);

  ON_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
      .WillByDefault(Return(Err(ErrCode::DriverError, "import failed")));

  auto payload = makeFdPayload(efd, getpid(), kAllocSize);
  auto factory = makeFactory(kIntraNodeBegin);
  auto result = factory.importSegment(kAllocSize, payload);
  ::close(efd);
  ASSERT_TRUE(result.hasError());
}

TEST_F(NVLinkFdTest, FdImportFailsWhenGranularityFails_ReleasesHandle) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  constexpr CUmemGenericAllocationHandle kImportedHandle = 0xAAAA;
  constexpr uint64_t kAllocSize = 4096;

  int efd = eventfd(0, 0);
  ASSERT_GE(efd, 0);

  ON_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kImportedHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemGetAllocationGranularity(_, _, _))
      .WillByDefault(Return(Err(ErrCode::DriverError, "granularity failed")));

  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(kImportedHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  auto payload = makeFdPayload(efd, getpid(), kAllocSize);
  auto factory = makeFactory(kIntraNodeBegin);
  auto result = factory.importSegment(kAllocSize, payload);
  ::close(efd);
  ASSERT_TRUE(result.hasError());
}

TEST_F(NVLinkFdTest, FdImportFailsWhenReserveFails_ReleasesHandle) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  constexpr CUmemGenericAllocationHandle kImportedHandle = 0xBBBB;
  constexpr uint64_t kAllocSize = 4096;

  int efd = eventfd(0, 0);
  ASSERT_GE(efd, 0);

  ON_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kImportedHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemGetAllocationGranularity(_, _, _))
      .WillByDefault(
          DoAll(SetArgPointee<0>(size_t{2 * 1024 * 1024}), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemAddressReserve(_, _, _, _, _))
      .WillByDefault(Return(Err(ErrCode::DriverError, "reserve failed")));

  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(kImportedHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  auto payload = makeFdPayload(efd, getpid(), kAllocSize);
  auto factory = makeFactory(kIntraNodeBegin);
  auto result = factory.importSegment(kAllocSize, payload);
  ::close(efd);
  ASSERT_TRUE(result.hasError());
}

TEST_F(NVLinkFdTest, FdImportFailsWhenMapFails_CleansUpReserveAndRelease) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  constexpr CUmemGenericAllocationHandle kHandle = 0xCCCC;
  constexpr CUdeviceptr kPtr = 0x7F000000;
  constexpr size_t kGranularity = 2 * 1024 * 1024;
  constexpr uint64_t kAllocSize = 4096;

  int efd = eventfd(0, 0);
  ASSERT_GE(efd, 0);

  ON_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemGetAllocationGranularity(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kGranularity), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemAddressReserve(_, _, _, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kPtr), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemMap(_, _, _, _, _))
      .WillByDefault(Return(Err(ErrCode::DriverError, "map failed")));

  EXPECT_CALL(*cudaDriverMock_, cuMemAddressFree(kPtr, kGranularity))
      .Times(1)
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(kHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  auto payload = makeFdPayload(efd, getpid(), kAllocSize);
  auto factory = makeFactory(kIntraNodeBegin);
  auto result = factory.importSegment(kAllocSize, payload);
  ::close(efd);
  ASSERT_TRUE(result.hasError());
}

TEST_F(NVLinkFdTest, FdImportFailsWhenSetAccessFails_CleansUpAll) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  constexpr CUmemGenericAllocationHandle kHandle = 0xDDDD;
  constexpr CUdeviceptr kPtr = 0x7F000000;
  constexpr size_t kGranularity = 2 * 1024 * 1024;
  constexpr uint64_t kAllocSize = 4096;

  int efd = eventfd(0, 0);
  ASSERT_GE(efd, 0);

  ON_CALL(*cudaDriverMock_, cuMemImportFromShareableHandle(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemGetAllocationGranularity(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kGranularity), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemAddressReserve(_, _, _, _, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kPtr), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemMap(_, _, _, _, _))
      .WillByDefault(Return(Ok()));
  ON_CALL(*cudaDriverMock_, cuMemSetAccess(_, _, _, _))
      .WillByDefault(Return(Err(ErrCode::DriverError, "set access failed")));

  EXPECT_CALL(*cudaDriverMock_, cuMemUnmap(kPtr, kGranularity))
      .Times(1)
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaDriverMock_, cuMemAddressFree(kPtr, kGranularity))
      .Times(1)
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*cudaDriverMock_, cuMemRelease(kHandle))
      .Times(1)
      .WillOnce(Return(Ok()));

  auto payload = makeFdPayload(efd, getpid(), kAllocSize);
  auto factory = makeFactory(kIntraNodeBegin);
  auto result = factory.importSegment(kAllocSize, payload);
  ::close(efd);
  ASSERT_TRUE(result.hasError());
}

TEST_F(NVLinkFdTest, FdImportRejectsWrongPayloadSize) {
  std::vector<uint8_t> bad(10);
  bad[0] = static_cast<uint8_t>(MemSharingMode::PosixFd);

  auto factory = makeFactory(kIntraNodeBegin);
  auto result = factory.importSegment(4096, bad);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(NVLinkFdTest, ImportRejectsUnknownModeByte) {
  std::vector<uint8_t> bad(65);
  bad[0] = 0xFF;

  auto factory = makeFactory(kIntraNodeBegin);
  auto result = factory.importSegment(4096, bad);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(NVLinkFdTest, FabricModeStillWorksWhenSupported) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  ON_CALL(*cudaDriverMock_, getCuMemHandleType())
      .WillByDefault(
          Return(Result<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_FABRIC)));

  CUmemGenericAllocationHandle fakeAllocHandle = 0xDEAD;
  CUmemFabricHandle fakeFabricHandle{};
  fakeFabricHandle.data[0] = 0x42;

  ON_CALL(*cudaDriverMock_, cuMemRetainAllocationHandle(_, _))
      .WillByDefault(DoAll(SetArgPointee<0>(fakeAllocHandle), Return(Ok())));
  ON_CALL(*cudaDriverMock_, cuMemExportToShareableHandle(_, _, _, _))
      .WillByDefault(DoAll(
          [&](void* shareableHandle,
              CUmemGenericAllocationHandle,
              CUmemAllocationHandleType,
              unsigned long long) {
            std::memcpy(
                shareableHandle, &fakeFabricHandle, sizeof(fakeFabricHandle));
          },
          Return(Ok())));

  auto factory = NVLinkTransportFactory(
      kCliqueABegin,
      evbThread_.getEventBase(),
      nvmlMock_,
      cudaApiMock_,
      cudaDriverMock_);

  uint8_t buf[64];
  Segment seg(buf, sizeof(buf), MemoryType::VRAM, kCliqueABegin);
  auto result = factory.registerSegment(seg);
  ASSERT_TRUE(result.hasValue()) << result.error().message();

  auto* handle =
      dynamic_cast<NVLinkFabricRegistrationHandle*>(result.value().get());
  ASSERT_NE(handle, nullptr);
}

TEST_F(NVLinkFdTest, FdRegisterSerializeImportRoundTrip) {
  using ::testing::_;
  using ::testing::DoAll;
  using ::testing::Return;
  using ::testing::SetArgPointee;

  constexpr CUmemGenericAllocationHandle kAllocHandle = 0xDEAD;
  constexpr CUmemGenericAllocationHandle kImportedHandle = 0xBEEF;
  constexpr size_t kGranularity = 2 * 1024 * 1024;
  constexpr CUdeviceptr kMappedPtr = 0x7F000000;
  constexpr size_t kSegLen = 64;

  int efd = eventfd(0, 0);
  ASSERT_GE(efd, 0);

  ON_CALL(*cudaDriverMock_, cuMemRetainAllocationHandle(_, _))
      .WillByDefault(DoAll(SetArgPointee<0>(kAllocHandle), Return(Ok())));
  ON_CALL(
      *cudaDriverMock_,
      cuMemExportToShareableHandle(
          _, _, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, _))
      .WillByDefault(DoAll(
          [efd](
              void* shareableHandle,
              CUmemGenericAllocationHandle,
              CUmemAllocationHandleType,
              unsigned long long) {
            int fd = efd;
            std::memcpy(shareableHandle, &fd, sizeof(fd));
          },
          Return(Ok())));

  auto exportFactory = makeFactory(kIntraNodeBegin);
  uint8_t buf[kSegLen];
  Segment seg(buf, sizeof(buf), MemoryType::VRAM, kIntraNodeBegin);
  auto regResult = exportFactory.registerSegment(seg);
  ASSERT_TRUE(regResult.hasValue()) << regResult.error().message();
  auto serialized = regResult.value()->serialize();
  EXPECT_EQ(serialized.size(), NVLinkFdRegistrationHandle::kSerializedSize);

  configureMocksForFdImportVmm(
      kIntraNodeEnd - 1, kImportedHandle, kGranularity, kMappedPtr);

  auto importFactory = makeFactory(kIntraNodeEnd - 1);
  auto importResult = importFactory.importSegment(kSegLen, serialized);
  ::close(efd);
  ASSERT_TRUE(importResult.hasValue()) << importResult.error().message();
  EXPECT_NE(
      dynamic_cast<NVLinkRemoteRegistrationHandle*>(importResult.value().get()),
      nullptr);
}

// --- requiredHandleType returns correct type for each sharing mode ---

TEST_F(NVLinkFdTest, RequiredHandleTypeReturnsPosixFdForFdMode) {
  auto factory = makeFactory(kIntraNodeBegin);
  EXPECT_EQ(factory.handleType(), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

TEST_F(NVLinkFdTest, HandleTypeReturnsFabricWhenConfigured) {
  using ::testing::Return;

  ON_CALL(*cudaDriverMock_, getCuMemHandleType())
      .WillByDefault(
          Return(Result<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_FABRIC)));

  auto factory = NVLinkTransportFactory(
      kCliqueABegin,
      evbThread_.getEventBase(),
      nvmlMock_,
      cudaApiMock_,
      cudaDriverMock_);
  EXPECT_EQ(factory.handleType(), CU_MEM_HANDLE_TYPE_FABRIC);
}

// --- NVLinkRemoteRegistrationHandle accessors ---

TEST(NVLinkRemoteRegistrationHandleTest, MappedPtrAndSizeAccessors) {
  constexpr CUdeviceptr kPtr = 0x7F000000;
  constexpr size_t kSize = 2 * 1024 * 1024;
  constexpr CUmemGenericAllocationHandle kHandle = 0xDEAD;

  auto driverApi = std::make_shared<::testing::NiceMock<MockCudaDriverApi>>();
  NVLinkRemoteRegistrationHandle handle(kHandle, kPtr, kSize, driverApi);

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  EXPECT_EQ(handle.mappedPtr(), reinterpret_cast<void*>(kPtr));
  EXPECT_EQ(handle.mappedSize(), kSize);
}

} // namespace uniflow
