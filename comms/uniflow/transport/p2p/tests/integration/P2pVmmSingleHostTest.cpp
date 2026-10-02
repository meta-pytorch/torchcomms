// Copyright (c) Meta Platforms, Inc. and affiliates.

/// Single-host integration test for VMM segments over the P2P (XGMI)
/// transport. Requires >= 2 GPUs with VMM support; shares VMM segments on
/// devices 0 and 1 as POSIX fds within one process, through P2pTransport
/// directly and through MultiTransport. AMD only: on NVIDIA the NVLink
/// transport, not P2P, carries intra-node VMM segments.

#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/p2p/tests/integration/P2pVmmTestUtils.h"

#include <algorithm>
#include <array>
#include <optional>
#include <thread>
#include <vector>

namespace uniflow {
namespace {

void* offsetPtr(void* ptr, size_t bytes) {
  return static_cast<uint8_t*>(ptr) + bytes;
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
      // Best-effort cleanup; explicitly discard the nodiscard error code.
      (void)cudaFree(ptr);
    }
  }
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
};

// VMM segments with every IPC call failing, so each case either shares its
// segments as POSIX fds or fails.
class P2pVmmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    evbThread_ = std::make_unique<ScopedEventBaseThread>();
    if (gpuCount() < 2) {
      GTEST_SKIP() << "need >= 2 GPUs, found " << gpuCount();
    }
    auto supported = driver_->isCuMemSupported();
    if (!supported.hasValue() || !supported.value()) {
      GTEST_SKIP() << "VMM is not supported";
    }
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
    pair.factory0 = std::make_unique<P2pTransportFactory>(0, evb, ipcDisabled_);
    pair.factory1 = std::make_unique<P2pTransportFactory>(1, evb, ipcDisabled_);

    auto r0 = pair.factory0->createTransport(pair.factory1->getTopology());
    auto r1 = pair.factory1->createTransport(pair.factory0->getTopology());
    ASSERT_TRUE(r0.hasValue()) << r0.error().message();
    ASSERT_TRUE(r1.hasValue()) << r1.error().message();
    pair.transport0 = std::move(r0.value());
    pair.transport1 = std::move(r1.value());

    auto info0 = pair.transport0->bind();
    auto info1 = pair.transport1->bind();
    ASSERT_FALSE(pair.transport0->connect(info1).hasError());
    ASSERT_FALSE(pair.transport1->connect(info0).hasError());
  }

  const std::shared_ptr<CudaDriverApi> driver_{
      std::make_shared<CudaDriverApi>()};
  const std::shared_ptr<CudaApi> ipcDisabled_{
      std::make_shared<IpcDisabledCudaApi>()};
  std::unique_ptr<ScopedEventBaseThread> evbThread_;
};

// A put into a segment that starts mid-chunk and spans three chunks lands at
// the segment offset and nowhere else.
TEST_F(P2pVmmTest, PutIntoSegmentSpanningChunks) {
  VmmBuffer remote{driver_};
  ASSERT_NO_FATAL_FAILURE(allocVmm(remote, 1, 3, {1}));
  const size_t offset = remote.chunkSize() / 2;
  const size_t len = 2 * remote.chunkSize();
  CudaBuffer local(len, 0);
  ASSERT_NE(local.ptr, nullptr);
  const auto pattern = iotaWords(len);
  ASSERT_NO_FATAL_FAILURE(writeWords(0, local.ptr, pattern));
  ASSERT_NO_FATAL_FAILURE(zeroDevice(1, remote.ptr(), remote.size()));

  ConnectedPair pair;
  ASSERT_NO_FATAL_FAILURE(connectPair(pair));
  void* remotePtr = offsetPtr(remote.ptr(), offset);
  Segment localSeg(local.ptr, len, MemoryType::VRAM, 0);
  Segment remoteSeg(remotePtr, len, MemoryType::VRAM, 1);
  std::unique_ptr<RegistrationHandle> exported;
  ASSERT_NO_FATAL_FAILURE(registerVmm(*pair.factory1, remoteSeg, exported));
  auto imported = pair.factory0->importSegment(len, exported->serialize());
  ASSERT_TRUE(imported.hasValue()) << imported.error().message();
  auto localReg = SegmentTest::makeUnregistered(localSeg);
  auto remoteReg =
      SegmentTest::makeRemote(remotePtr, len, std::move(imported.value()));

  auto status =
      pair.transport0->put(wholeSegment(localReg, remoteReg, len), {}).get();
  ASSERT_FALSE(status.hasError()) << status.error().message();

  std::vector<uint32_t> expected(remote.size() / sizeof(uint32_t), 0);
  std::copy(
      pattern.begin(),
      pattern.end(),
      expected.begin() + offset / sizeof(uint32_t));
  EXPECT_EQ(readWords(1, remote.ptr(), remote.size()), expected);
  pair.transport0->shutdown();
  pair.transport1->shutdown();
}

// A get from a segment spanning three chunks into a VMM destination.
TEST_F(P2pVmmTest, GetIntoVmmDestination) {
  VmmBuffer remote{driver_};
  VmmBuffer local{driver_};
  ASSERT_NO_FATAL_FAILURE(allocVmm(remote, 1, 3, {1}));
  ASSERT_NO_FATAL_FAILURE(allocVmm(local, 0, 2, {0}));
  const size_t offset = remote.chunkSize() / 2;
  const size_t len = local.size();
  void* remotePtr = offsetPtr(remote.ptr(), offset);
  const auto pattern = iotaWords(len);
  ASSERT_NO_FATAL_FAILURE(writeWords(1, remotePtr, pattern));
  ASSERT_NO_FATAL_FAILURE(zeroDevice(0, local.ptr(), len));

  ConnectedPair pair;
  ASSERT_NO_FATAL_FAILURE(connectPair(pair));
  Segment localSeg(local.ptr(), len, MemoryType::VRAM, 0);
  Segment remoteSeg(remotePtr, len, MemoryType::VRAM, 1);
  std::unique_ptr<RegistrationHandle> localHandle;
  std::unique_ptr<RegistrationHandle> exported;
  ASSERT_NO_FATAL_FAILURE(registerVmm(*pair.factory0, localSeg, localHandle));
  ASSERT_NO_FATAL_FAILURE(registerVmm(*pair.factory1, remoteSeg, exported));
  auto imported = pair.factory0->importSegment(len, exported->serialize());
  ASSERT_TRUE(imported.hasValue()) << imported.error().message();
  auto localReg = SegmentTest::makeRegistered(localSeg, std::move(localHandle));
  auto remoteReg =
      SegmentTest::makeRemote(remotePtr, len, std::move(imported.value()));

  auto status =
      pair.transport0->get(wholeSegment(localReg, remoteReg, len), {}).get();
  ASSERT_FALSE(status.hasError()) << status.error().message();

  EXPECT_EQ(readWords(0, local.ptr(), len), pattern);
  pair.transport0->shutdown();
  pair.transport1->shutdown();
}

constexpr size_t kSides = 2;
using TransportPair = std::array<std::unique_ptr<MultiTransport>, kSides>;

void connectMultiTransports(
    MultiTransportFactory& factory0,
    MultiTransportFactory& factory1,
    TransportPair& transports) {
  auto r0 = factory0.createTransport(factory1.getTopology());
  auto r1 = factory1.createTransport(factory0.getTopology());
  ASSERT_TRUE(r0.hasValue()) << r0.error().message();
  ASSERT_TRUE(r1.hasValue()) << r1.error().message();
  transports[0] = std::move(r0.value());
  transports[1] = std::move(r1.value());
  auto info0 = transports[0]->bind();
  auto info1 = transports[1]->bind();
  ASSERT_TRUE(info0.hasValue()) << info0.error().message();
  ASSERT_TRUE(info1.hasValue()) << info1.error().message();
  // connect() blocks on the listener side, so run both peers concurrently.
  Status c0 = Ok();
  Status c1 = Ok();
  std::thread t0([&]() { c0 = transports[0]->connect(info1.value()); });
  std::thread t1([&]() { c1 = transports[1]->connect(info0.value()); });
  t0.join();
  t1.join();
  ASSERT_FALSE(c0.hasError()) << c0.error().message();
  ASSERT_FALSE(c1.hasError()) << c1.error().message();
}

// A VMM segment on each of GPUs 0 and 1 registered with MultiTransport
// factories. The initiator puts its segment into the target's, clears its own,
// and reads it back with a get.
class P2pVmmMultiTransportTest : public P2pVmmTest {
 protected:
  void TearDown() override {
    for (const auto& transport : transports_) {
      if (transport) {
        transport->shutdown();
      }
    }
    P2pVmmTest::TearDown();
  }

  static int deviceOf(size_t side) {
    return static_cast<int>(side);
  }

  // Owner-only segments, as an allocator hands them out: the importer's own
  // access grant is all the P2P tier has.
  void allocSegments() {
    for (size_t side = 0; side < kSides; ++side) {
      const int device = deviceOf(side);
      ASSERT_NO_FATAL_FAILURE(allocVmm(buffers_[side], device, 2, {device}));
      segments_[side].emplace(
          buffers_[side].ptr(),
          buffers_[side].size(),
          MemoryType::VRAM,
          device);
    }
  }

  // Stops on a VMM export failure before MultiTransport sees the segments, then
  // registers them and checks their P2P handles are POSIX fds, all before any
  // import.
  void registerSegments() {
    for (auto& segment : segments_) {
      ASSERT_NO_FATAL_FAILURE(
          assertVmmExportable(*segment, evbThread_->getEventBase()));
    }
    for (size_t side = 0; side < kSides; ++side) {
      factories_[side] = std::make_unique<MultiTransportFactory>(
          deviceOf(side), multiTransportOptions());
      auto reg = factories_[side]->registerSegment(*segments_[side]);
      ASSERT_TRUE(reg.hasValue()) << reg.error().message();
      regs_[side].emplace(std::move(reg.value()));
      ASSERT_NO_FATAL_FAILURE(assertPosixFd(
          SegmentTest::findHandle(*regs_[side], TransportType::NVLink)));
    }
  }

  void importAndConnect(size_t initiator) {
    auto targetId = regs_[1 - initiator]->exportId();
    ASSERT_TRUE(targetId.hasValue()) << targetId.error().message();
    auto target = factories_[initiator]->importSegment(targetId.value());
    ASSERT_TRUE(target.hasValue()) << target.error().message();
    target_.emplace(std::move(target.value()));
    ASSERT_NO_FATAL_FAILURE(
        connectMultiTransports(*factories_[0], *factories_[1], transports_));
  }

  void putThenGet(size_t initiator) {
    const size_t target = 1 - initiator;
    const size_t len = buffers_[initiator].size();
    const auto pattern = iotaWords(len);
    void* const src = buffers_[initiator].ptr();
    void* const dst = buffers_[target].ptr();
    ASSERT_NO_FATAL_FAILURE(writeWords(deviceOf(initiator), src, pattern));
    ASSERT_NO_FATAL_FAILURE(zeroDevice(deviceOf(target), dst, len));
    const auto reqs = wholeSegment(*regs_[initiator], *target_, len);

    auto put = transports_[initiator]->put(reqs).get();
    ASSERT_FALSE(put.hasError()) << put.error().message();
    EXPECT_EQ(readWords(deviceOf(target), dst, len), pattern);
    ASSERT_NO_FATAL_FAILURE(zeroDevice(deviceOf(initiator), src, len));
    auto get = transports_[initiator]->get(reqs).get();
    ASSERT_FALSE(get.hasError()) << get.error().message();
    EXPECT_EQ(readWords(deviceOf(initiator), src, len), pattern);
  }

  void expectCarriedOverP2p(size_t initiator) {
    ASSERT_NO_FATAL_FAILURE(allocSegments());
    ASSERT_NO_FATAL_FAILURE(registerSegments());
    ASSERT_NO_FATAL_FAILURE(importAndConnect(initiator));
    ASSERT_NO_FATAL_FAILURE(putThenGet(initiator));
    EXPECT_EQ(
        transferCounts(*transports_[initiator]),
        (TransferCounts{.p2p = 2, .rdma = 0, .tcp = 0}));
  }

  // Declared first so the buffers outlive every registration and mapping.
  std::array<VmmBuffer, kSides> buffers_{
      VmmBuffer{driver_},
      VmmBuffer{driver_}};
  std::array<std::optional<Segment>, kSides> segments_;
  std::array<std::unique_ptr<MultiTransportFactory>, kSides> factories_;
  std::array<std::optional<RegisteredSegment>, kSides> regs_;
  std::optional<RemoteRegisteredSegment> target_;
  TransportPair transports_;
};

// VMM segments registered through MultiTransport are shared as POSIX fds, and
// the P2P tier carries both transfers.
TEST_F(P2pVmmMultiTransportTest, CarriesVmmSegmentsOverP2p) {
  expectCarriedOverP2p(/*initiator=*/0);
}

// The same with device 1 initiating, so the importer is not device 0.
TEST_F(P2pVmmMultiTransportTest, CarriesVmmSegmentsOverP2pFromDevice1) {
  expectCarriedOverP2p(/*initiator=*/1);
}

} // namespace
} // namespace uniflow
