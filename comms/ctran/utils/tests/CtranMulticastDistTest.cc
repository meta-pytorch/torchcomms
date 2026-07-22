// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "nccl.h"

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/CtranMulticast.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/commSpecs.h"

// Distributed test for the NVL CE-multicast registration mechanism, driven the
// way production drives it: one rank per GPU. The root createRoot()s the
// multicast object and broadcasts its fabric handle over allGatherNvlDomain;
// every peer importShareableHandle()s + adoptImported()s it; each rank
// import()s its own buffer -- self-detecting + retaining its own segment
// handles -- then addDeviceAndBind()s and mapVA()s. A single multicast write
// from the root must then fan out (via NVSwitch) into every rank's own buffer
// -- the nvlCeBcast primitive. Requires >= 2 GPUs in one NVL domain with
// fabric; skipped otherwise.

using ctran::utils::CtranIpcHandle;
using ctran::utils::CtranIpcMem;
using ctran::utils::CtranMulticast;

namespace {

size_t roundUp(size_t size, size_t align) {
  return ((size + align - 1) / align) * align;
}

} // namespace

class MulticastDistTest : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    ctran::CtranDistTestFixture::SetUp();
    COMMCHECK_TEST(ctran::utils::commCudaLibraryInit());
    CUDACHECK_TEST(cudaSetDevice(localRank));
    comm_ = makeCtranComm();
  }

  void TearDown() override {
    comm_.reset();
    ctran::CtranDistTestFixture::TearDown();
  }

 protected:
  std::unique_ptr<CtranComm> comm_;
  const char* desc_{"MulticastDistTest"};

  // Drive the full create -> export -> allGatherNvlDomain -> import ->
  // addDeviceAndBind -> mapVA -> broadcast-fanout path over a recvbuff backed
  // by `numSegments` disjoint physical segments. numSegments>1 exercises
  // addDeviceAndBind's per-segment running-offset bind and a multicast write
  // that must fan out across every segment.
  void runCreateBindBroadcast(int numSegments) {
    const int rank = comm_->statex_->rank();
    const int nRanks = comm_->statex_->nRanks();
    const int lRank = comm_->statex_->localRank();
    const int nLocalRanks = comm_->statex_->nLocalRanks();

    if (nRanks < 2 || comm_->statex_->nNodes() > 1) {
      GTEST_SKIP() << "requires >= 2 ranks in a single NVL domain, got nRanks="
                   << nRanks << " nNodes=" << comm_->statex_->nNodes();
    }
    if (!ctran::utils::getCuMemSysSupported() ||
        !CtranMulticast::isSupported(lRank)) {
      GTEST_SKIP() << "cuMem + multicast not supported on this device";
    }
    if ((ctran::utils::getCuMemAllocHandleType() & CU_MEM_HANDLE_TYPE_FABRIC) ==
        0) {
      GTEST_SKIP()
          << "FABRIC handle type required to share the multicast object";
    }

    size_t gran = 0;
    COMMCHECK_TEST(CtranMulticast::granularity(lRank, nLocalRanks, gran));
    ASSERT_GT(gran, 0u);
    // Each disjoint segment must be aligned to both cuMem (2MB) and the
    // multicast granularity.
    const size_t segSize = roundUp(2 * 1024 * 1024, gran);
    std::vector<size_t> segSizes(numSegments, segSize);
    const size_t total = segSize * numSegments;

    // Each rank: a `numSegments`-segment cuMem recvbuff on its own GPU.
    void* buf = nullptr;
    std::vector<TestMemSegment> allocSegs;
    COMMCHECK_TEST(
        ctran::commMemAllocDisjoint(
            &buf,
            segSizes,
            allocSegs,
            true,
            ctran::utils::getCuMemAllocHandleType()));
    auto ipcMem = std::make_unique<CtranIpcMem>(lRank, desc_);
    bool supported = false;
    COMMCHECK_TEST(ipcMem->tryLoad(buf, total, supported, false));
    ASSERT_TRUE(supported);

    // Root (rank 0) creates the object sized to the summed segments; its fabric
    // handle is all-gathered so peers import a distinct reference.
    constexpr int kRoot = 0;
    auto mc = std::make_shared<CtranMulticast>(lRank, nLocalRanks, lRank);
    std::vector<CtranIpcHandle> handles(nLocalRanks);
    std::memset(handles.data(), 0, handles.size() * sizeof(CtranIpcHandle));
    if (rank == kRoot) {
      CUmemGenericAllocationHandle rootHandle{};
      COMMCHECK_TEST(
          mc->createRoot(total, CU_MEM_HANDLE_TYPE_FABRIC, rootHandle));
      COMMCHECK_TEST(
          ctran::utils::exportShareableHandle(
              rootHandle, handles[lRank], /*isFabric=*/true));
    }
    allGatherNvlDomain(comm_.get(), handles);
    if (rank != kRoot) {
      CUmemGenericAllocationHandle imported{};
      COMMCHECK_TEST(
          ctran::utils::importShareableHandle(
              handles[kRoot], imported, /*isFabric=*/true));
      mc->adoptImported(imported);
    }

    // Each rank imports its own buffer's segments (self-detect + retain its own
    // handles, independent of ipcMem's registration above), then binds them and
    // maps the multicast VA. Both are local ops needing no cross-rank barrier.
    COMMCHECK_TEST(mc->import(buf, total));
    EXPECT_EQ(mc->importedSize(), total) << "import should enumerate the full "
                                         << numSegments << "-segment buffer";
    COMMCHECK_TEST(mc->addDeviceAndBind());
    COMMCHECK_TEST(mc->mapVA(total, gran));
    ASSERT_NE(mc->getMulticastPtr(), nullptr);
    // Every rank must have bound its device before the root broadcasts below:
    // cuMulticast only fans a write out to devices bound at write time, so a
    // peer still in addDeviceAndBind would silently miss the pattern.
    barrierNvlDomain(comm_.get());

    // Root multicasts a pattern across the whole (multi-segment) VA; every rank
    // must observe it across its entire buffer -- i.e. every bound segment.
    constexpr uint8_t kPattern = 0x5A;
    if (rank == kRoot) {
      CUDACHECK_TEST(cudaMemset(mc->getMulticastPtr(), kPattern, total));
      CUDACHECK_TEST(cudaDeviceSynchronize());
    }
    barrierNvlDomain(comm_.get());
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<uint8_t> observed(total, 0x00);
    CUDACHECK_TEST(cudaMemcpy(
        observed.data(), ipcMem->getBase(), total, cudaMemcpyDeviceToHost));
    const std::vector<uint8_t> expected(total, kPattern);
    EXPECT_EQ(observed, expected)
        << "rank " << rank << " missed the multicast broadcast across "
        << numSegments << " segment(s)";

    // Tear the overlay down before freeing its backing.
    barrierNvlDomain(comm_.get());
    mc.reset();
    ipcMem.reset();
    ctran::commMemFreeDisjoint(buf, segSizes);
  }
};

TEST_F(MulticastDistTest, CreateBindBroadcast) {
  runCreateBindBroadcast(/*numSegments=*/1);
}

// Multi-segment: a recvbuff backed by 2 disjoint physical segments exercises
// addDeviceAndBind's per-segment running-offset bind, and the broadcast must
// fan out across both segments.
TEST_F(MulticastDistTest, CreateBindBroadcastMultiSegment) {
  runCreateBindBroadcast(/*numSegments=*/2);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
