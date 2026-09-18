// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/// Cross-host GPU (VRAM) integration test for the TCP transport.
/// Requires MPI with 2 ranks on 2 different hosts (nnodes=2, ppn=1), each with
/// at least one GPU. VRAM segments are transferred via host-staging inside the
/// transport (D2H on the source, H2D on the destination); TCP never touches
/// device memory directly. Front-end NIC (eth2) is auto-detected per host.
///
/// Running it for real (2 ranks): standard single-process CI can't form a
/// 2-rank MPI world, so those lanes skip (see SetUp). Exercise it on two GPU
/// hosts with the mpirun harness, pointing it at the GPU binary (set env vars):
///
///   BUILD_MODE=@mode/opt-amd-gpu
///   ROCM_MODIFIER=rocm70
///   BUILD_TARGET=fbcode//comms/uniflow/transport/tcp/tests/integration:cross_host_gpu_test_binary
///   fbcode/scripts/cppc/run_uniflow_cross_host_tcp.sh rtptest521.maz5
///   rtptest522.maz5

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/uniflow/Segment.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/tcp/TcpTransport.h"

using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace uniflow {

/// Friend-class wrapper (name must be exactly "SegmentTest").
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

namespace {

std::string getInterfaceIpv6(const std::string& iface) {
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) != 0) {
    return "";
  }
  std::string result;
  for (auto* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET6 ||
        iface != ifa->ifa_name) {
      continue;
    }
    auto* sa = reinterpret_cast<sockaddr_in6*>(ifa->ifa_addr);
    if (sa->sin6_addr.s6_addr[0] == 0xfe &&
        (sa->sin6_addr.s6_addr[1] & 0xc0) == 0x80) {
      continue; // link-local
    }
    char buf[INET6_ADDRSTRLEN] = {};
    if (inet_ntop(AF_INET6, &sa->sin6_addr, buf, sizeof(buf)) != nullptr) {
      result = buf;
      break;
    }
  }
  freeifaddrs(ifaddr);
  return result;
}

std::vector<uint8_t> mpiExchange(
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

uint64_t mpiExchangeAddr(uint64_t localVal, int rank) {
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

bool anyRankWantsToSkip(bool localSkip) {
  int localVal = localSkip ? 1 : 0;
  int globalVal = 0;
  MPI_Allreduce(&localVal, &globalVal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return globalVal != 0;
}

/// RAII device buffer.
struct CudaBuffer {
  void* ptr{nullptr};
  size_t size{0};
  int device{0};

  CudaBuffer(size_t n, int dev) : size(n), device(dev) {
    (void)cudaSetDevice(dev);
    if (cudaMalloc(&ptr, n) != cudaSuccess) {
      ptr = nullptr;
    }
  }
  ~CudaBuffer() {
    if (ptr) {
      (void)cudaSetDevice(device);
      (void)cudaFree(ptr);
    }
  }
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
};

} // namespace

class TcpGpuCrossHostTest : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    // See CrossHostTests.cpp: skip on single-process CI lanes; real 2-rank
    // coverage is the mpirun 2-host harness.
    if (numRanks != 2) {
      GTEST_SKIP() << "requires exactly 2 MPI ranks (run under mpirun -n 2)";
    }
    eth2Host_ = getInterfaceIpv6("eth2");
    ASSERT_FALSE(eth2Host_.empty()) << "no global IPv6 on eth2";
    evbThread_ = std::make_unique<ScopedEventBaseThread>();
  }

  void TearDown() override {
    evbThread_.reset();
    MpiBaseTestFixture::TearDown();
  }

  struct ConnectedPair {
    std::unique_ptr<TcpTransportFactory> factory;
    std::unique_ptr<Transport> transport;
  };

  struct SegmentRegistration {
    RegisteredSegment local;
    RemoteRegisteredSegment remote;
  };

  ConnectedPair connectCrossHost() {
    ConnectedPair pair;
    auto* evb = evbThread_->getEventBase();
    pair.factory = std::make_unique<TcpTransportFactory>(
        /*deviceId=*/-1, evb, controller::TcpSocketConfig{}, eth2Host_);

    auto localTopo = pair.factory->getTopology();
    auto remoteTopo = mpiExchange(localTopo, globalRank);
    auto transportResult = pair.factory->createTransport(remoteTopo);
    EXPECT_TRUE(transportResult.hasValue())
        << transportResult.error().message();
    pair.transport = std::move(transportResult.value());

    auto localInfo = pair.transport->bind();
    auto remoteInfo = mpiExchange(localInfo, globalRank);
    auto connectStatus = pair.transport->connect(remoteInfo);
    EXPECT_FALSE(connectStatus.hasError()) << connectStatus.error().message();
    return pair;
  }

  std::optional<SegmentRegistration> registerAndExchangeSegments(
      TcpTransportFactory& factory,
      void* buf,
      size_t totalSize,
      MemoryType memType,
      int deviceId) {
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
    uint64_t remoteAddr =
        mpiExchangeAddr(reinterpret_cast<uint64_t>(buf), globalRank);
    auto remoteReg = SegmentTest::makeRemote(
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<void*>(remoteAddr),
        totalSize,
        std::move(remoteHandle.value()));
    return SegmentRegistration{std::move(localReg), std::move(remoteReg)};
  }

  std::string eth2Host_;
  std::unique_ptr<ScopedEventBaseThread> evbThread_;
};

// rank 0 puts a VRAM buffer into rank 1's VRAM segment (host-staged over TCP).
TEST_F(TcpGpuCrossHostTest, VramPut) {
  int deviceCount = 0;
  bool noCuda =
      cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 1;
  if (anyRankWantsToSkip(noCuda)) {
    GTEST_SKIP() << "some rank lacks a GPU";
  }

  constexpr int kDev = 0;
  constexpr size_t kLen = 4 * 1024 * 1024; // 4 MiB
  auto pair = connectCrossHost();

  CudaBuffer gpuBuf(kLen, kDev);
  ASSERT_NE(gpuBuf.ptr, nullptr) << "cudaMalloc failed";
  ASSERT_EQ(cudaSetDevice(kDev), cudaSuccess);

  std::vector<uint8_t> host(kLen);
  if (globalRank == 0) {
    for (size_t i = 0; i < kLen; ++i) {
      host[i] = static_cast<uint8_t>((i * 131 + 7) & 0xFF);
    }
    ASSERT_EQ(
        cudaMemcpy(gpuBuf.ptr, host.data(), kLen, cudaMemcpyHostToDevice),
        cudaSuccess);
  } else {
    ASSERT_EQ(cudaMemset(gpuBuf.ptr, 0, kLen), cudaSuccess);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto segments = registerAndExchangeSegments(
      *pair.factory, gpuBuf.ptr, kLen, MemoryType::VRAM, kDev);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    std::vector<TransferRequest> reqs;
    reqs.push_back(
        TransferRequest{
            .local = segments->local.span(size_t{0}, kLen),
            .remote = segments->remote.span(size_t{0}, kLen)});
    auto st = pair.transport->put(reqs, {}).get();
    ASSERT_FALSE(st.hasError()) << "put failed: " << st.error().message();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (globalRank == 1) {
    std::vector<uint8_t> verify(kLen, 0);
    ASSERT_EQ(cudaSetDevice(kDev), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(verify.data(), gpuBuf.ptr, kLen, cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (size_t i = 0; i < kLen; ++i) {
      ASSERT_EQ(verify[i], static_cast<uint8_t>((i * 131 + 7) & 0xFF))
          << "VRAM put mismatch at byte " << i;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

// rank 0 pulls (get) rank 1's VRAM segment into its own VRAM buffer.
TEST_F(TcpGpuCrossHostTest, VramGet) {
  int deviceCount = 0;
  bool noCuda =
      cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 1;
  if (anyRankWantsToSkip(noCuda)) {
    GTEST_SKIP() << "some rank lacks a GPU";
  }

  constexpr int kDev = 0;
  constexpr size_t kLen = 4 * 1024 * 1024; // 4 MiB
  auto pair = connectCrossHost();

  CudaBuffer gpuBuf(kLen, kDev);
  ASSERT_NE(gpuBuf.ptr, nullptr) << "cudaMalloc failed";
  ASSERT_EQ(cudaSetDevice(kDev), cudaSuccess);

  if (globalRank == 0) {
    ASSERT_EQ(cudaMemset(gpuBuf.ptr, 0, kLen), cudaSuccess);
  } else {
    std::vector<uint8_t> host(kLen);
    for (size_t i = 0; i < kLen; ++i) {
      host[i] = static_cast<uint8_t>((i * 17 + 3) & 0xFF);
    }
    ASSERT_EQ(
        cudaMemcpy(gpuBuf.ptr, host.data(), kLen, cudaMemcpyHostToDevice),
        cudaSuccess);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto segments = registerAndExchangeSegments(
      *pair.factory, gpuBuf.ptr, kLen, MemoryType::VRAM, kDev);
  ASSERT_TRUE(segments.has_value());

  if (globalRank == 0) {
    std::vector<TransferRequest> reqs;
    reqs.push_back(
        TransferRequest{
            .local = segments->local.span(size_t{0}, kLen),
            .remote = segments->remote.span(size_t{0}, kLen)});
    auto st = pair.transport->get(reqs, {}).get();
    ASSERT_FALSE(st.hasError()) << "get failed: " << st.error().message();

    std::vector<uint8_t> verify(kLen, 0);
    ASSERT_EQ(cudaSetDevice(kDev), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(verify.data(), gpuBuf.ptr, kLen, cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (size_t i = 0; i < kLen; ++i) {
      ASSERT_EQ(verify[i], static_cast<uint8_t>((i * 17 + 3) & 0xFF))
          << "VRAM get mismatch at byte " << i;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace uniflow

int main(int argc, char** argv) {
  // Force MPI to bootstrap over the TCP BTL (see CrossHostTests.cpp): the CI
  // RDMA NIC can fail openib init, preventing the 2-rank world from forming.
  setenv("OMPI_MCA_btl", "tcp,self", 0);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase());
  return RUN_ALL_TESTS();
}
