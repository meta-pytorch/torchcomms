// Copyright (c) Meta Platforms, Inc. and affiliates.

/// Cross-process integration test for VMM segments over the P2P (XGMI)
/// transport. Two MPI ranks on one host each own a GPU and a VMM segment, and
/// each imports the other's segment through pidfd_getfd: the import path whose
/// POSIX-fd handle convention differs across ROCm runtimes. Each rank also gets
/// from the peer's hipMalloc segment, shared over HIP IPC, into its VMM
/// segment. Runs with 2 ranks on one host (nnodes=1, ppn=2).

#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/p2p/tests/integration/P2pVmmTestUtils.h"

#include <mpi.h>
#include <sys/prctl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace uniflow {
namespace {

constexpr int kRanks = 2;
constexpr size_t kChunks = 2;

int mpiExchangeInt(int local, int rank) {
  const int peer = 1 - rank;
  int peerValue = 0;
  MPI_Sendrecv(
      &local,
      1,
      MPI_INT,
      peer,
      0,
      &peerValue,
      1,
      MPI_INT,
      peer,
      0,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);
  return peerValue;
}

std::vector<uint8_t> mpiExchange(const std::vector<uint8_t>& local, int rank) {
  const int peer = 1 - rank;
  const int localSize = static_cast<int>(local.size());
  const int peerSize = mpiExchangeInt(localSize, rank);
  std::vector<uint8_t> peerData(peerSize);
  MPI_Sendrecv(
      local.data(),
      localSize,
      MPI_BYTE,
      peer,
      1,
      peerData.data(),
      peerSize,
      MPI_BYTE,
      peer,
      1,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);
  return peerData;
}

bool anyRank(bool local) {
  int localValue = local ? 1 : 0;
  int globalValue = 0;
  MPI_Allreduce(&localValue, &globalValue, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return globalValue != 0;
}

// pidfd_getfd on a sibling rank needs ptrace-attach rights. Yama scope 1 grants
// them to a non-ancestor only when the target names it with PR_SET_PTRACER;
// scopes 2 and 3 withhold them from an unprivileged peer. Empty without Yama.
std::optional<int> yamaPtraceScope() {
  std::ifstream file("/proc/sys/kernel/yama/ptrace_scope");
  int scope = 0;
  if (!(file >> scope)) {
    return std::nullopt;
  }
  return scope;
}

struct CudaFree {
  void operator()(void* ptr) const {
    (void)cudaFree(ptr);
  }
};

class P2pVmmCrossProcessTest : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ASSERT_EQ(numRanks, kRanks) << "needs exactly " << kRanks << " MPI ranks";
    const auto supported = driver_->isCuMemSupported();
    const bool canRun = gpuCount() >= kRanks && supported.hasValue() &&
        supported.value() && yamaScope_.value_or(0) <= 1;
    if (anyRank(!canRun)) {
      GTEST_SKIP() << "needs " << kRanks
                   << " GPUs, VMM support and Yama ptrace_scope <= 1";
    }
    const int peerPid =
        mpiExchangeInt(static_cast<int>(::getpid()), globalRank);
    ASSERT_NO_FATAL_FAILURE(step([&] { allowPtraceFrom(peerPid); }));
    ASSERT_NO_FATAL_FAILURE(step([this] { allocSegment(); }));
  }

  void TearDown() override {
    if (transport_) {
      transport_->shutdown();
    }
    // Close the peer's IPC mapping before the peer frees the memory behind it.
    peerDevice_.reset();
    MPI_Barrier(MPI_COMM_WORLD);
    // Withdraw the ptrace grant once the peer can no longer pull fds, so it
    // does not outlive the case.
    if (yamaScope_ == 1) {
      EXPECT_EQ(::prctl(PR_SET_PTRACER, 0UL, 0UL, 0UL, 0UL), 0)
          << std::strerror(errno);
    }
    MpiBaseTestFixture::TearDown();
  }

  // Runs a rank-local step, then fails every rank if any rank failed or threw
  // in it, so no rank waits in a later MPI call for a rank that has returned.
  template <typename Step>
  void step(Step&& localStep) {
    try {
      localStep();
    } catch (const std::exception& e) {
      ADD_FAILURE() << "step threw: " << e.what();
    }
    ASSERT_FALSE(anyRank(HasFailure())) << "a rank failed this step";
  }

  // The peer pulls this rank's fds with pidfd_getfd, which Yama scope 1 allows
  // a sibling only after this rank names it as its tracer.
  void allowPtraceFrom(int peerPid) {
    if (yamaScope_ != 1) {
      return;
    }
    const auto tracer = static_cast<unsigned long>(peerPid);
    ASSERT_EQ(::prctl(PR_SET_PTRACER, tracer, 0UL, 0UL, 0UL), 0)
        << std::strerror(errno);
  }

  // Owner-only access, as an allocator hands segments out: the importer's own
  // access grant is all the P2P tier has.
  void allocSegment() {
    ASSERT_CUDA(cudaSetDevice(localRank));
    ASSERT_NO_FATAL_FAILURE(allocVmm(buffer_, localRank, kChunks, {localRank}));
    segment_.emplace(
        buffer_.ptr(), buffer_.size(), MemoryType::VRAM, localRank);
  }

  // A VMM export failure fails the test before MultiTransport registers the
  // segment; its P2P tier would fall back to IPC, unsafe on VMM memory.
  void registerSegment() {
    ASSERT_NO_FATAL_FAILURE(
        assertVmmExportable(*segment_, evbThread_.getEventBase()));
    factory_ = std::make_unique<MultiTransportFactory>(
        localRank, multiTransportOptions());
    auto reg = factory_->registerSegment(*segment_);
    ASSERT_TRUE(reg.hasValue()) << reg.error().message();
    reg_.emplace(std::move(reg.value()));
    ASSERT_NO_FATAL_FAILURE(
        assertPosixFd(SegmentTest::findHandle(*reg_, TransportType::NVLink)));
    auto id = reg_->exportId();
    ASSERT_TRUE(id.hasValue()) << id.error().message();
    exportId_ = std::move(id.value());
  }

  // A hipMalloc segment beside the VMM one, which the P2P tier shares over HIP
  // IPC.
  void registerDeviceSegment() {
    ASSERT_CUDA(cudaSetDevice(localRank));
    void* ptr = nullptr;
    ASSERT_CUDA(cudaMalloc(&ptr, len()));
    deviceBuffer_.reset(ptr);
    deviceSegment_.emplace(ptr, len(), MemoryType::VRAM, localRank);
    auto reg = factory_->registerSegment(*deviceSegment_);
    ASSERT_TRUE(reg.hasValue()) << reg.error().message();
    deviceReg_.emplace(std::move(reg.value()));
    const auto* p2p = dynamic_cast<const P2pRegistrationHandle*>(
        SegmentTest::findHandle(*deviceReg_, TransportType::NVLink));
    ASSERT_NE(p2p, nullptr) << "no P2P handle";
    ASSERT_EQ(p2p->sharingMode(), P2pSharingMode::Ipc);
    auto id = deviceReg_->exportId();
    ASSERT_TRUE(id.hasValue()) << id.error().message();
    deviceExportId_ = std::move(id.value());
  }

  void bindTransport(const std::vector<uint8_t>& peerTopology) {
    auto transport = factory_->createTransport(peerTopology);
    ASSERT_TRUE(transport.hasValue()) << transport.error().message();
    transport_ = std::move(transport.value());
    auto info = transport_->bind();
    ASSERT_TRUE(info.hasValue()) << info.error().message();
    bindInfo_ = std::move(info.value());
  }

  void connectTransport(const std::vector<uint8_t>& peerInfo) {
    const auto status = transport_->connect(peerInfo);
    ASSERT_FALSE(status.hasError()) << status.error().message();
  }

  void importPeerSegment(const std::vector<uint8_t>& peerId) {
    auto peer = factory_->importSegment(peerId);
    ASSERT_TRUE(peer.hasValue()) << peer.error().message();
    peer_.emplace(std::move(peer.value()));
  }

  // Distinct per initiator, so a round cannot pass on an earlier round's data.
  std::vector<uint32_t> roundPattern(int initiator) const {
    return iotaWords(
        buffer_.size(), 1 + static_cast<uint32_t>(initiator) * (1u << 28));
  }

  void prepareRound(int initiator, const std::vector<uint32_t>& pattern) {
    if (globalRank == initiator) {
      ASSERT_NO_FATAL_FAILURE(writeWords(localRank, buffer_.ptr(), pattern));
    } else {
      ASSERT_NO_FATAL_FAILURE(
          zeroDevice(localRank, buffer_.ptr(), buffer_.size()));
    }
  }

  void putFrom(int initiator) {
    if (globalRank != initiator) {
      return;
    }
    auto status = transport_->put(wholeSegment(*reg_, *peer_, len())).get();
    ASSERT_FALSE(status.hasError()) << status.error().message();
  }

  // The target checks the put landed while the initiator clears its own
  // segment and reads the target's back with a get.
  void checkThenGet(int initiator, const std::vector<uint32_t>& pattern) {
    if (globalRank != initiator) {
      EXPECT_EQ(readWords(localRank, buffer_.ptr(), len()), pattern);
      return;
    }
    ASSERT_NO_FATAL_FAILURE(zeroDevice(localRank, buffer_.ptr(), len()));
    auto status = transport_->get(wholeSegment(*reg_, *peer_, len())).get();
    ASSERT_FALSE(status.hasError()) << status.error().message();
    EXPECT_EQ(readWords(localRank, buffer_.ptr(), len()), pattern);
  }

  void importPeerDeviceSegment(const std::vector<uint8_t>& peerId) {
    auto peer = factory_->importSegment(peerId);
    ASSERT_TRUE(peer.hasValue()) << peer.error().message();
    peerDevice_.emplace(std::move(peer.value()));
  }

  // The initiator clears its VMM segment and gets bytes from the peer's
  // hipMalloc segment into it. Only the first bytes land, so a get that
  // writes past its length or misses a chunk fails the check.
  void getFromPeerDevice(
      int initiator,
      size_t bytes,
      const std::vector<uint32_t>& pattern) {
    if (globalRank != initiator) {
      return;
    }
    ASSERT_NO_FATAL_FAILURE(zeroDevice(localRank, buffer_.ptr(), len()));
    auto status =
        transport_->get(wholeSegment(*reg_, *peerDevice_, bytes)).get();
    ASSERT_FALSE(status.hasError()) << status.error().message();
    std::vector<uint32_t> expected(len() / sizeof(uint32_t), 0);
    std::copy(
        pattern.begin(),
        pattern.begin() + bytes / sizeof(uint32_t),
        expected.begin());
    EXPECT_EQ(readWords(localRank, buffer_.ptr(), len()), expected);
  }

  size_t len() const {
    return buffer_.size();
  }

  const std::shared_ptr<CudaDriverApi> driver_{
      std::make_shared<CudaDriverApi>()};
  const std::optional<int> yamaScope_{yamaPtraceScope()};
  ScopedEventBaseThread evbThread_;
  // Declared before the factory so the buffer outlives every registration.
  VmmBuffer buffer_{driver_};
  std::unique_ptr<void, CudaFree> deviceBuffer_;
  std::optional<Segment> segment_;
  std::optional<Segment> deviceSegment_;
  std::unique_ptr<MultiTransportFactory> factory_;
  std::optional<RegisteredSegment> reg_;
  std::optional<RegisteredSegment> deviceReg_;
  std::vector<uint8_t> exportId_;
  std::vector<uint8_t> deviceExportId_;
  std::unique_ptr<MultiTransport> transport_;
  std::vector<uint8_t> bindInfo_;
  std::optional<RemoteRegisteredSegment> peer_;
  std::optional<RemoteRegisteredSegment> peerDevice_;
};

// Each rank imports the other's VMM segment across processes, then each in
// turn puts into the other's segment and gets it back, all over the P2P tier.
TEST_F(P2pVmmCrossProcessTest, PutAndGetBetweenProcessesOverP2p) {
  ASSERT_NO_FATAL_FAILURE(step([this] { registerSegment(); }));
  const auto peerTopology = mpiExchange(factory_->getTopology(), globalRank);
  ASSERT_NO_FATAL_FAILURE(step([&] { bindTransport(peerTopology); }));
  const auto peerInfo = mpiExchange(bindInfo_, globalRank);
  ASSERT_NO_FATAL_FAILURE(step([&] { connectTransport(peerInfo); }));
  const auto peerId = mpiExchange(exportId_, globalRank);
  ASSERT_NO_FATAL_FAILURE(step([&] { importPeerSegment(peerId); }));

  for (int initiator = 0; initiator < kRanks; ++initiator) {
    const auto pattern = roundPattern(initiator);
    ASSERT_NO_FATAL_FAILURE(step([&] { prepareRound(initiator, pattern); }));
    ASSERT_NO_FATAL_FAILURE(step([&] { putFrom(initiator); }));
    ASSERT_NO_FATAL_FAILURE(step([&] { checkThenGet(initiator, pattern); }));
  }

  // Each rank initiated one put and one get.
  EXPECT_EQ(
      transferCounts(*transport_),
      (TransferCounts{.p2p = 2, .rdma = 0, .tcp = 0}));
}

// Each rank in turn gets from the peer's hipMalloc segment, imported over HIP
// IPC, into its own VMM segment. ROCm runs such a copy on the peer's device,
// which can write the VMM segment only through the access its registration
// granted; without it the copy faults the GPU.
TEST_F(P2pVmmCrossProcessTest, GetFromPeerIpcSegmentIntoVmmSegment) {
  ASSERT_NO_FATAL_FAILURE(step([this] { registerSegment(); }));
  ASSERT_NO_FATAL_FAILURE(step([this] { registerDeviceSegment(); }));
  const auto peerTopology = mpiExchange(factory_->getTopology(), globalRank);
  ASSERT_NO_FATAL_FAILURE(step([&] { bindTransport(peerTopology); }));
  const auto peerInfo = mpiExchange(bindInfo_, globalRank);
  ASSERT_NO_FATAL_FAILURE(step([&] { connectTransport(peerInfo); }));
  const auto peerId = mpiExchange(deviceExportId_, globalRank);
  ASSERT_NO_FATAL_FAILURE(step([&] { importPeerDeviceSegment(peerId); }));

  // 32 KiB is the smallest get that faulted without the grant; the whole
  // segment spans both chunks.
  const std::vector<size_t> sizes{size_t{32} << 10, len()};
  for (int initiator = 0; initiator < kRanks; ++initiator) {
    const auto pattern = roundPattern(initiator);
    ASSERT_NO_FATAL_FAILURE(step([&] {
      if (globalRank != initiator) {
        writeWords(localRank, deviceBuffer_.get(), pattern);
      }
    }));
    for (const size_t bytes : sizes) {
      ASSERT_NO_FATAL_FAILURE(
          step([&] { getFromPeerDevice(initiator, bytes, pattern); }));
    }
  }

  // Each rank initiated one get per size.
  EXPECT_EQ(
      transferCounts(*transport_),
      (TransferCounts{.p2p = sizes.size(), .rdma = 0, .tcp = 0}));
}

} // namespace
} // namespace uniflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase());
  return RUN_ALL_TESTS();
}
