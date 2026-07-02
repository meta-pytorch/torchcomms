// Copyright (c) Meta Platforms, Inc. and affiliates.

/// Single-host integration test for the intra-node P2P (XGMI) transport.
/// Requires >= 2 GPUs; uses two P2pTransportFactory instances on devices 0 and
/// 1 within one process (the same-process import path), exercising the IPC
/// export + device-to-device data plane on real hardware. Runs on AMD GPU CI.

#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/p2p/P2pTransport.h"

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace uniflow {

/// Friend wrapper to assemble RegisteredSegment / RemoteRegisteredSegment with
/// handles. The name must be exactly "SegmentTest" to match Segment.h.
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

// ASSERT-based check so a failed CUDA setup call produces a clean test failure
// rather than a silently-ignored error code.
#define ASSERT_CUDA(expr) ASSERT_EQ((expr), cudaSuccess) << #expr

int gpuCount() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    return 0;
  }
  return count;
}

struct CudaBuffer {
  void* ptr{nullptr};
  size_t size{0};

  CudaBuffer(size_t n, int device) : size(n) {
    // On a setDevice failure, skip the malloc so ptr stays null and the
    // caller's ASSERT_NE(ptr, nullptr) fails cleanly at the allocation site,
    // rather than silently allocating on the wrong device.
    if (cudaSetDevice(device) != cudaSuccess) {
      return;
    }
    if (cudaMalloc(&ptr, n) != cudaSuccess) {
      ptr = nullptr;
    }
  }
  ~CudaBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
};

class P2pSingleHostTest : public ::testing::Test {
 protected:
  void SetUp() override {
    evbThread_ = std::make_unique<ScopedEventBaseThread>();
  }
  void TearDown() override {
    evbThread_.reset();
  }

  struct ConnectedPair {
    std::unique_ptr<P2pTransportFactory> factory0;
    std::unique_ptr<P2pTransportFactory> factory1;
    std::unique_ptr<Transport> transport0;
    std::unique_ptr<Transport> transport1;
  };

  void connectPair(ConnectedPair& pair) {
    auto* evb = evbThread_->getEventBase();
    pair.factory0 = std::make_unique<P2pTransportFactory>(0, evb);
    pair.factory1 = std::make_unique<P2pTransportFactory>(1, evb);

    auto topo0 = pair.factory0->getTopology();
    auto topo1 = pair.factory1->getTopology();

    auto r0 = pair.factory0->createTransport(topo1);
    auto r1 = pair.factory1->createTransport(topo0);
    ASSERT_TRUE(r0.hasValue()) << r0.error().message();
    ASSERT_TRUE(r1.hasValue()) << r1.error().message();
    pair.transport0 = std::move(r0.value());
    pair.transport1 = std::move(r1.value());

    auto info0 = pair.transport0->bind();
    auto info1 = pair.transport1->bind();
    ASSERT_FALSE(pair.transport0->connect(info1).hasError());
    ASSERT_FALSE(pair.transport1->connect(info0).hasError());
  }

  std::unique_ptr<ScopedEventBaseThread> evbThread_;
};

TEST_F(P2pSingleHostTest, TwoTransportsConnect) {
  if (gpuCount() < 2) {
    GTEST_SKIP() << "need >= 2 GPUs, found " << gpuCount();
  }
  auto pair = ConnectedPair{};
  ASSERT_NO_FATAL_FAILURE(connectPair(pair));
  EXPECT_EQ(pair.transport0->state(), TransportState::Connected);
  EXPECT_EQ(pair.transport1->state(), TransportState::Connected);
  pair.transport0->shutdown();
  pair.transport1->shutdown();
}

TEST_F(P2pSingleHostTest, GpuPut) {
  if (gpuCount() < 2) {
    GTEST_SKIP() << "need >= 2 GPUs, found " << gpuCount();
  }
  constexpr size_t kSize = 1 << 20; // 1 MiB
  ConnectedPair pair;
  ASSERT_NO_FATAL_FAILURE(connectPair(pair));

  CudaBuffer sendGpu(kSize, 0);
  CudaBuffer recvGpu(kSize, 1);
  ASSERT_NE(sendGpu.ptr, nullptr);
  ASSERT_NE(recvGpu.ptr, nullptr);

  std::vector<char> staging(kSize, static_cast<char>(0xC5));
  ASSERT_CUDA(cudaSetDevice(0));
  ASSERT_CUDA(
      cudaMemcpy(sendGpu.ptr, staging.data(), kSize, cudaMemcpyHostToDevice));
  ASSERT_CUDA(cudaSetDevice(1));
  ASSERT_CUDA(cudaMemset(recvGpu.ptr, 0, kSize));
  ASSERT_CUDA(cudaDeviceSynchronize());

  Segment sendSeg(sendGpu.ptr, kSize, MemoryType::VRAM, 0);
  Segment recvSeg(recvGpu.ptr, kSize, MemoryType::VRAM, 1);

  auto sendReg = pair.factory0->registerSegment(sendSeg);
  ASSERT_TRUE(sendReg.hasValue()) << sendReg.error().message();
  auto recvReg = pair.factory1->registerSegment(recvSeg);
  ASSERT_TRUE(recvReg.hasValue()) << recvReg.error().message();

  auto recvPayload = recvReg.value()->serialize();
  auto remoteHandle = pair.factory0->importSegment(kSize, recvPayload);
  ASSERT_TRUE(remoteHandle.hasValue()) << remoteHandle.error().message();

  auto localReg =
      SegmentTest::makeRegistered(sendSeg, std::move(sendReg.value()));
  auto remoteReg = SegmentTest::makeRemote(
      recvGpu.ptr, kSize, std::move(remoteHandle.value()));

  std::vector<TransferRequest> reqs;
  reqs.push_back(
      TransferRequest{
          .local = localReg.span(size_t{0}, kSize),
          .remote = remoteReg.span(size_t{0}, kSize),
      });

  auto putStatus = pair.transport0->put(reqs, {}).get();
  ASSERT_FALSE(putStatus.hasError()) << putStatus.error().message();

  std::vector<char> verify(kSize, 0);
  ASSERT_CUDA(cudaSetDevice(1));
  ASSERT_CUDA(
      cudaMemcpy(verify.data(), recvGpu.ptr, kSize, cudaMemcpyDeviceToHost));
  const std::vector<char> expected(kSize, static_cast<char>(0xC5));
  EXPECT_EQ(verify, expected);

  pair.transport0->shutdown();
  pair.transport1->shutdown();
}

TEST_F(P2pSingleHostTest, GpuGet) {
  if (gpuCount() < 2) {
    GTEST_SKIP() << "need >= 2 GPUs, found " << gpuCount();
  }
  constexpr size_t kSize = 1 << 20; // 1 MiB
  ConnectedPair pair;
  ASSERT_NO_FATAL_FAILURE(connectPair(pair));

  CudaBuffer localGpu(kSize, 0);
  CudaBuffer remoteGpu(kSize, 1);
  ASSERT_NE(localGpu.ptr, nullptr);
  ASSERT_NE(remoteGpu.ptr, nullptr);

  ASSERT_CUDA(cudaSetDevice(0));
  ASSERT_CUDA(cudaMemset(localGpu.ptr, 0, kSize));
  std::vector<char> staging(kSize, static_cast<char>(0xD7));
  ASSERT_CUDA(cudaSetDevice(1));
  ASSERT_CUDA(
      cudaMemcpy(remoteGpu.ptr, staging.data(), kSize, cudaMemcpyHostToDevice));
  ASSERT_CUDA(cudaDeviceSynchronize());

  Segment localSeg(localGpu.ptr, kSize, MemoryType::VRAM, 0);
  Segment remoteSeg(remoteGpu.ptr, kSize, MemoryType::VRAM, 1);

  auto localReg = pair.factory0->registerSegment(localSeg);
  ASSERT_TRUE(localReg.hasValue()) << localReg.error().message();
  auto remoteRegResult = pair.factory1->registerSegment(remoteSeg);
  ASSERT_TRUE(remoteRegResult.hasValue()) << remoteRegResult.error().message();

  auto remotePayload = remoteRegResult.value()->serialize();
  auto remoteHandle = pair.factory0->importSegment(kSize, remotePayload);
  ASSERT_TRUE(remoteHandle.hasValue()) << remoteHandle.error().message();

  auto localRegSeg =
      SegmentTest::makeRegistered(localSeg, std::move(localReg.value()));
  auto remoteReg = SegmentTest::makeRemote(
      remoteGpu.ptr, kSize, std::move(remoteHandle.value()));

  std::vector<TransferRequest> reqs;
  reqs.push_back(
      TransferRequest{
          .local = localRegSeg.span(size_t{0}, kSize),
          .remote = remoteReg.span(size_t{0}, kSize),
      });

  auto getStatus = pair.transport0->get(reqs, {}).get();
  ASSERT_FALSE(getStatus.hasError()) << getStatus.error().message();

  std::vector<char> verify(kSize, 0);
  ASSERT_CUDA(cudaSetDevice(0));
  ASSERT_CUDA(
      cudaMemcpy(verify.data(), localGpu.ptr, kSize, cudaMemcpyDeviceToHost));
  const std::vector<char> expected(kSize, static_cast<char>(0xD7));
  EXPECT_EQ(verify, expected);

  pair.transport0->shutdown();
  pair.transport1->shutdown();
}

// Registers a segment that is a SUB-RANGE of a larger allocation and verifies a
// put lands at base+offset (AMD: getMemAddressRange reports the real allocation
// base, and the recorded offset is applied on import). Proves the pre-offset
// region is untouched, i.e. data did not land at the allocation base.
TEST_F(P2pSingleHostTest, GpuPutSubAllocation) {
  if (gpuCount() < 2) {
    GTEST_SKIP() << "need >= 2 GPUs, found " << gpuCount();
  }
  constexpr size_t kSize = 1 << 20; // 1 MiB segment
  constexpr size_t kOffset = 1
      << 20; // segment starts 1 MiB into the allocation
  constexpr size_t kAlloc = kSize + kOffset; // 2 MiB allocation
  ConnectedPair pair;
  ASSERT_NO_FATAL_FAILURE(connectPair(pair));

  CudaBuffer sendGpu(kSize, 0);
  CudaBuffer recvGpu(
      kAlloc, 1); // larger allocation; the segment is a sub-range
  ASSERT_NE(sendGpu.ptr, nullptr);
  ASSERT_NE(recvGpu.ptr, nullptr);

  std::vector<char> staging(kSize, static_cast<char>(0xA9));
  ASSERT_CUDA(cudaSetDevice(0));
  ASSERT_CUDA(
      cudaMemcpy(sendGpu.ptr, staging.data(), kSize, cudaMemcpyHostToDevice));
  ASSERT_CUDA(cudaSetDevice(1));
  ASSERT_CUDA(cudaMemset(recvGpu.ptr, 0, kAlloc)); // zero the whole allocation
  ASSERT_CUDA(cudaDeviceSynchronize());

  void* recvSubPtr = static_cast<uint8_t*>(recvGpu.ptr) + kOffset;
  Segment sendSeg(sendGpu.ptr, kSize, MemoryType::VRAM, 0);
  Segment recvSeg(recvSubPtr, kSize, MemoryType::VRAM, 1);

  auto sendReg = pair.factory0->registerSegment(sendSeg);
  ASSERT_TRUE(sendReg.hasValue()) << sendReg.error().message();
  auto recvReg = pair.factory1->registerSegment(recvSeg);
  ASSERT_TRUE(recvReg.hasValue()) << recvReg.error().message();

  auto recvPayload = recvReg.value()->serialize();
  auto remoteHandle = pair.factory0->importSegment(kSize, recvPayload);
  ASSERT_TRUE(remoteHandle.hasValue()) << remoteHandle.error().message();

  auto localReg =
      SegmentTest::makeRegistered(sendSeg, std::move(sendReg.value()));
  auto remoteReg = SegmentTest::makeRemote(
      recvSubPtr, kSize, std::move(remoteHandle.value()));

  std::vector<TransferRequest> reqs;
  reqs.push_back(
      TransferRequest{
          .local = localReg.span(size_t{0}, kSize),
          .remote = remoteReg.span(size_t{0}, kSize),
      });

  auto putStatus = pair.transport0->put(reqs, {}).get();
  ASSERT_FALSE(putStatus.hasError()) << putStatus.error().message();

  // Read back the whole allocation: [0, kOffset) must remain zero (data did NOT
  // land at the allocation base) and [kOffset, kOffset+kSize) holds the
  // pattern.
  std::vector<char> verify(kAlloc, 0);
  ASSERT_CUDA(cudaSetDevice(1));
  ASSERT_CUDA(
      cudaMemcpy(verify.data(), recvGpu.ptr, kAlloc, cudaMemcpyDeviceToHost));
  const std::vector<char> zeros(kOffset, 0);
  const std::vector<char> pattern(kSize, static_cast<char>(0xA9));
  EXPECT_EQ(std::vector<char>(verify.begin(), verify.begin() + kOffset), zeros);
  EXPECT_EQ(std::vector<char>(verify.begin() + kOffset, verify.end()), pattern);

  pair.transport0->shutdown();
  pair.transport1->shutdown();
}

} // namespace
} // namespace uniflow
