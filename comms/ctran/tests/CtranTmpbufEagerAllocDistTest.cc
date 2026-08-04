// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <folly/init/Init.h>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/memory/memCacheAllocator.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

// Verifies that ctran collectives still initialize and run correctly when
// CtranComm::tmpbufEagerAlloc_ is FALSE. With the flag false, the eager tmpbuf
// slab is skipped during ctranInit: devState_d_ is still allocated
// (getDevState() stays non-null), nvlTransports_ is skipped
// (getNvlTransportsBase() returns null), and sharedRes_ stays null. At ppn1
// (nLocalRanks==1, via nolocal) the covered collectives never touch sharedRes_,
// so they must still succeed. At ppn>1 (plain 1x8 and vnode) the NVL-staging
// collectives (AllGatherP/AllToAll/AllToAllP) are skipped because their sync/
// staging maps are null when sharedRes_ is absent; only the init/memory
// contract runs there. Each test logs ctran pool usage with the [CTRAN_MEM]
// tag.
class CtranTmpbufEagerAllocDistTest : public ctran::CtranDistTestFixture,
                                      public CtranBaseTest {
 public:
  CtranTmpbufEagerAllocDistTest() = default;

  void SetUp() override {
    // Always run ctran alltoall regardless of message size.
    setenv("NCCL_CTRAN_ALLTOALL_THRESHOLD", "0", 0);
    ctran::CtranDistTestFixture::SetUp();
  }

  void TearDown() override {
    ctran::CtranDistTestFixture::TearDown();
  }

  // Create a comm with eager tmpbuf allocation disabled. Topology (nLocalRanks)
  // is driven by the per-config NCCL_COMM_STATE_DEBUG_TOPO env, not forced
  // here.
  std::unique_ptr<CtranComm> makeEagerAllocFalseComm() {
    return makeCtranComm(
        /*noLocal=*/false, /*ibLazyConnect=*/false, /*tmpbufEagerAlloc=*/false);
  }

  // Current ctran memory pool in-use bytes; null-safe.
  static size_t ctranUsedMemBytes() {
    const auto alloc = ncclx::memory::memCacheAllocator::getInstance();
    return alloc ? alloc->getUsedMem() : 0;
  }

  // Free device memory (bytes) from cudaMemGetInfo; used-memory deltas are
  // computed as (baseFree - currentFree). Captures allocations the memCache
  // pool metric misses (e.g. the NVL SharedResource).
  static size_t cudaFreeBytes() {
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    CUDACHECK_TEST(cudaMemGetInfo(&freeBytes, &totalBytes));
    return freeBytes;
  }

  void* createDataBuf(size_t nbytes, bool doRegister) {
    void* buf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&buf, nbytes));
    if (buf && doRegister) {
      COMMCHECK_TEST(ctran::globalRegisterWithPtr(buf, nbytes));
    }
    return buf;
  }

  void releaseDataBuf(void* buf, size_t nbytes, bool doDeregister) {
    if (doDeregister) {
      COMMCHECK_TEST(ctran::globalDeregisterWithPtr(buf, nbytes));
    }
    CUDACHECK_TEST(cudaFree(buf));
  }

  // Persistent-collective recv buffers must be CCA-cached (via the regcache
  // globalRegister path) so the eager scoped registration can acquire them.
  void* createDataBufCached(size_t nbytes) {
    void* buf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&buf, nbytes));
    if (buf) {
      COMMCHECK_TEST(
          ctran::RegCache::getInstance()->globalRegister(buf, nbytes));
    }
    return buf;
  }

  void releaseDataBufCached(void* buf, size_t nbytes) {
    COMMCHECK_TEST(
        ctran::RegCache::getInstance()->globalDeregister(buf, nbytes));
    CUDACHECK_TEST(cudaFree(buf));
  }

  void runAllGather(CtranComm* comm, enum NCCL_ALLGATHER_ALGO algo) {
    const size_t numElements = 8192;
    const size_t sendBytes = numElements * sizeof(int32_t);
    const size_t recvBytes = sendBytes * numRanks;

    void* sendbuf = createDataBuf(sendBytes, /*doRegister=*/true);
    void* recvbuf = createDataBuf(recvBytes, /*doRegister=*/true);

    std::vector<int32_t> input_h(numElements, globalRank);
    CUDACHECK_TEST(
        cudaMemcpy(sendbuf, input_h.data(), sendBytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recvbuf, 0, recvBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    const auto res = ctranAllGather(
        sendbuf, recvbuf, numElements, commInt32, comm, testStream, algo);
    ASSERT_EQ(res, commSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(testStream));

    std::vector<int32_t> output_h(numElements * numRanks);
    CUDACHECK_TEST(cudaMemcpy(
        output_h.data(), recvbuf, recvBytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < numElements * numRanks; i++) {
      const int expected = i / numElements;
      EXPECT_EQ(output_h[i], expected)
          << "rank " << globalRank << " mismatch at index " << i;
    }

    verifyGpeLeak(comm->ctran_.get());

    releaseDataBuf(sendbuf, sendBytes, /*doDeregister=*/true);
    releaseDataBuf(recvbuf, recvBytes, /*doDeregister=*/true);
  }

  void runAllGatherP(CtranComm* comm) {
    const size_t count = 8192;
    const size_t maxRecvCount = count * numRanks;
    // commInt8 keeps sendBytes/recvBytes == count/maxRecvCount (no conversion).
    const size_t sendBytes = count;
    const size_t recvBytes = maxRecvCount;

    char* sendbuf = reinterpret_cast<char*>(createDataBufCached(sendBytes));
    char* recvbuf = reinterpret_cast<char*>(createDataBufCached(recvBytes));

    CUDACHECK_TEST(
        cudaMemset(sendbuf, static_cast<char>(globalRank), sendBytes));
    CUDACHECK_TEST(cudaMemset(recvbuf, 0, recvBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    meta::comms::Hints hints;
    CtranPersistentRequest* request = nullptr;
    COMMCHECK_TEST(
        ctran::allGatherPInit(
            recvbuf, maxRecvCount, hints, commInt8, comm, testStream, request));
    ASSERT_EQ(cudaStreamSynchronize(testStream), cudaSuccess);

    ASSERT_EQ(
        ctran::allGatherPExec(sendbuf, count, commInt8, request), commSuccess);
    ASSERT_EQ(cudaStreamSynchronize(testStream), cudaSuccess);

    std::vector<char> observed(recvBytes);
    CUDACHECK_TEST(cudaMemcpy(
        observed.data(), recvbuf, recvBytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < numRanks; i++) {
      const std::vector<char> chunk(
          observed.begin() + i * count, observed.begin() + (i + 1) * count);
      EXPECT_THAT(chunk, testing::Each(static_cast<char>(i)))
          << "rank " << globalRank << " chunk from peer " << i;
    }

    verifyGpeLeak(comm->ctran_.get());

    ASSERT_EQ(ctran::allGatherPDestroy(request), commSuccess);
    delete request;

    releaseDataBufCached(sendbuf, sendBytes);
    releaseDataBufCached(recvbuf, recvBytes);
  }

  void runAllToAll(CtranComm* comm) {
    const size_t count = 8192;
    const size_t bufCount = count * numRanks;
    const size_t bufNbytes = bufCount * sizeof(int32_t);
    // Fixed (rank-independent) base value so every rank derives the same
    // expected chunk values without an out-of-band exchange.
    const int expectedVal = 0;

    int32_t* sendBuf = reinterpret_cast<int32_t*>(
        createDataBuf(bufNbytes, /*doRegister=*/true));
    int32_t* recvBuf = reinterpret_cast<int32_t*>(
        createDataBuf(bufNbytes, /*doRegister=*/true));

    for (int i = 0; i < numRanks; i++) {
      assignChunkValue<int32_t>(
          sendBuf + i * count, count, expectedVal + globalRank * 10 + i + 1);
    }

    const auto res = ctranAllToAll(
        sendBuf,
        recvBuf,
        count,
        commInt32,
        comm,
        testStream,
        NCCL_ALLTOALL_ALGO::ctran);
    ASSERT_EQ(res, commSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(testStream));

    for (int i = 0; i < numRanks; i++) {
      const int errs = checkChunkValue<int32_t>(
          recvBuf + i * count, count, expectedVal + i * 10 + globalRank + 1);
      EXPECT_EQ(errs, 0) << "rank " << globalRank << " chunk from peer " << i;
    }

    verifyGpeLeak(comm->ctran_.get());

    releaseDataBuf(sendBuf, bufNbytes, /*doDeregister=*/true);
    releaseDataBuf(recvBuf, bufNbytes, /*doDeregister=*/true);
  }

  void runAllToAllP(CtranComm* comm) {
    const size_t maxRecvCount = 1024 * 1024;
    const size_t count = 8192;
    const size_t bufNbytes = maxRecvCount * sizeof(int);
    const int expectedVal = 0;

    int* sendBuf =
        reinterpret_cast<int*>(createDataBuf(bufNbytes, /*doRegister=*/true));
    void* recvBuf = createDataBufCached(bufNbytes);

    meta::comms::Hints hints;
    CtranPersistentRequest* request = nullptr;
    COMMCHECK_TEST(
        ctran::AllToAllPInit(
            recvBuf, maxRecvCount, hints, commInt, comm, testStream, request));

    for (int i = 0; i < numRanks; i++) {
      assignChunkValue<int>(
          sendBuf + i * count, count, expectedVal + globalRank * 100 + i + 1);
    }

    ASSERT_EQ(ctran::AllToAllPExec(sendBuf, count, request), commSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(testStream));

    int* recvbuff = reinterpret_cast<int*>(recvBuf);
    for (int i = 0; i < numRanks; i++) {
      const int errs = checkChunkValue<int>(
          recvbuff + i * count, count, expectedVal + i * 100 + globalRank + 1);
      EXPECT_EQ(errs, 0) << "rank " << globalRank << " chunk from peer " << i;
    }

    ASSERT_EQ(ctran::AllToAllPDestroy(request), commSuccess);
    delete request;

    releaseDataBuf(sendBuf, bufNbytes, /*doDeregister=*/true);
    releaseDataBufCached(recvBuf, bufNbytes);
  }

 protected:
  cudaStream_t testStream{0};
};

TEST_F(CtranTmpbufEagerAllocDistTest, InitContractWithEagerAllocFalse) {
  const size_t base = ctranUsedMemBytes();
  const size_t baseFree = cudaFreeBytes();
  auto comm = makeEagerAllocFalseComm();
  const size_t initUsed = ctranUsedMemBytes() - base;
  const size_t devInitUsed = baseFree - cudaFreeBytes();
  LOG(INFO) << "[CTRAN_MEM] test=InitContract nLocalRanks="
            << comm->statex_->nLocalRanks() << " rank=" << globalRank
            << " initUsedMiB=" << (initUsed / 1024.0 / 1024.0)
            << " devInitUsedMiB=" << (devInitUsed / 1024.0 / 1024.0);
  auto* algo = comm->ctran_->algo.get();
  // devState_d_ is always allocated; the eager slab / nvlTransports_ are not.
  EXPECT_NE(algo->getDevState(), nullptr);
  EXPECT_EQ(algo->getNvlTransportsBase(), nullptr);
}

TEST_F(CtranTmpbufEagerAllocDistTest, AllGatherRing) {
  const size_t base = ctranUsedMemBytes();
  const size_t baseFree = cudaFreeBytes();
  auto comm = makeEagerAllocFalseComm();
  const size_t initUsed = ctranUsedMemBytes() - base;
  const size_t devInitUsed = baseFree - cudaFreeBytes();
  LOG(INFO) << "[CTRAN_MEM] test=AllGatherRing nLocalRanks="
            << comm->statex_->nLocalRanks() << " rank=" << globalRank
            << " initUsedMiB=" << (initUsed / 1024.0 / 1024.0)
            << " devInitUsedMiB=" << (devInitUsed / 1024.0 / 1024.0);
  const auto algo = NCCL_ALLGATHER_ALGO::ctring;
  if (!ctranAllGatherSupport(comm.get(), algo)) {
    GTEST_SKIP() << "ctring AllGather not supported, skip test";
  }
  runAllGather(comm.get(), algo);
  const size_t postUsed = ctranUsedMemBytes() - base;
  const size_t devPostUsed = baseFree - cudaFreeBytes();
  LOG(INFO) << "[CTRAN_MEM] test=AllGatherRing nLocalRanks="
            << comm->statex_->nLocalRanks() << " rank=" << globalRank
            << " postUsedMiB=" << (postUsed / 1024.0 / 1024.0)
            << " devPostUsedMiB=" << (devPostUsed / 1024.0 / 1024.0);
}

TEST_F(CtranTmpbufEagerAllocDistTest, AllGatherCtsrd) {
  const size_t base = ctranUsedMemBytes();
  const size_t baseFree = cudaFreeBytes();
  auto comm = makeEagerAllocFalseComm();
  const size_t initUsed = ctranUsedMemBytes() - base;
  const size_t devInitUsed = baseFree - cudaFreeBytes();
  LOG(INFO) << "[CTRAN_MEM] test=AllGatherCtsrd nLocalRanks="
            << comm->statex_->nLocalRanks() << " rank=" << globalRank
            << " initUsedMiB=" << (initUsed / 1024.0 / 1024.0)
            << " devInitUsedMiB=" << (devInitUsed / 1024.0 / 1024.0);
  const auto algo = NCCL_ALLGATHER_ALGO::ctsrd;
  if (!ctranAllGatherSupport(comm.get(), algo)) {
    GTEST_SKIP() << "ctsrd AllGather requires nLocalRanks=1, skip test";
  }
  runAllGather(comm.get(), algo);
  const size_t postUsed = ctranUsedMemBytes() - base;
  const size_t devPostUsed = baseFree - cudaFreeBytes();
  LOG(INFO) << "[CTRAN_MEM] test=AllGatherCtsrd nLocalRanks="
            << comm->statex_->nLocalRanks() << " rank=" << globalRank
            << " postUsedMiB=" << (postUsed / 1024.0 / 1024.0)
            << " devPostUsedMiB=" << (devPostUsed / 1024.0 / 1024.0);
}

TEST_F(CtranTmpbufEagerAllocDistTest, AllGatherP) {
  const size_t base = ctranUsedMemBytes();
  const size_t baseFree = cudaFreeBytes();
  auto comm = makeEagerAllocFalseComm();
  const size_t initUsed = ctranUsedMemBytes() - base;
  const size_t devInitUsed = baseFree - cudaFreeBytes();
  LOG(INFO) << "[CTRAN_MEM] test=AllGatherP nLocalRanks="
            << comm->statex_->nLocalRanks() << " rank=" << globalRank
            << " initUsedMiB=" << (initUsed / 1024.0 / 1024.0)
            << " devInitUsedMiB=" << (devInitUsed / 1024.0 / 1024.0);
  if (comm->statex_->nLocalRanks() > 1) {
    GTEST_SKIP() << "flag=false: NVL sync/staging maps are null at ppn>1";
  }
  if (!ctran::allGatherPSupport(comm.get())) {
    GTEST_SKIP() << "AllGatherP not supported, skip test";
  }
  if (comm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB backend found, skip test";
  }
  runAllGatherP(comm.get());
  const size_t postUsed = ctranUsedMemBytes() - base;
  const size_t devPostUsed = baseFree - cudaFreeBytes();
  LOG(INFO) << "[CTRAN_MEM] test=AllGatherP nLocalRanks="
            << comm->statex_->nLocalRanks() << " rank=" << globalRank
            << " postUsedMiB=" << (postUsed / 1024.0 / 1024.0)
            << " devPostUsedMiB=" << (devPostUsed / 1024.0 / 1024.0);
}

TEST_F(CtranTmpbufEagerAllocDistTest, AllToAll) {
  const size_t base = ctranUsedMemBytes();
  const size_t baseFree = cudaFreeBytes();
  auto comm = makeEagerAllocFalseComm();
  const size_t initUsed = ctranUsedMemBytes() - base;
  const size_t devInitUsed = baseFree - cudaFreeBytes();
  LOG(INFO) << "[CTRAN_MEM] test=AllToAll nLocalRanks="
            << comm->statex_->nLocalRanks() << " rank=" << globalRank
            << " initUsedMiB=" << (initUsed / 1024.0 / 1024.0)
            << " devInitUsedMiB=" << (devInitUsed / 1024.0 / 1024.0);
  if (comm->statex_->nLocalRanks() > 1) {
    GTEST_SKIP() << "flag=false: NVL sync/staging maps are null at ppn>1";
  }
  if (!ctranAllToAllSupport(
          8192, commInt32, comm.get(), NCCL_ALLTOALL_ALGO::ctran)) {
    GTEST_SKIP() << "AllToAll not supported, skip test";
  }
  runAllToAll(comm.get());
  const size_t postUsed = ctranUsedMemBytes() - base;
  const size_t devPostUsed = baseFree - cudaFreeBytes();
  LOG(INFO) << "[CTRAN_MEM] test=AllToAll nLocalRanks="
            << comm->statex_->nLocalRanks() << " rank=" << globalRank
            << " postUsedMiB=" << (postUsed / 1024.0 / 1024.0)
            << " devPostUsedMiB=" << (devPostUsed / 1024.0 / 1024.0);
}

TEST_F(CtranTmpbufEagerAllocDistTest, AllToAllP) {
  const size_t base = ctranUsedMemBytes();
  const size_t baseFree = cudaFreeBytes();
  auto comm = makeEagerAllocFalseComm();
  const size_t initUsed = ctranUsedMemBytes() - base;
  const size_t devInitUsed = baseFree - cudaFreeBytes();
  LOG(INFO) << "[CTRAN_MEM] test=AllToAllP nLocalRanks="
            << comm->statex_->nLocalRanks() << " rank=" << globalRank
            << " initUsedMiB=" << (initUsed / 1024.0 / 1024.0)
            << " devInitUsedMiB=" << (devInitUsed / 1024.0 / 1024.0);
  if (comm->statex_->nLocalRanks() > 1) {
    GTEST_SKIP() << "flag=false: NVL sync/staging maps are null at ppn>1";
  }
  if (!ctran::AllToAllPSupport(comm.get())) {
    GTEST_SKIP() << "AllToAllP not supported, skip test";
  }
  runAllToAllP(comm.get());
  const size_t postUsed = ctranUsedMemBytes() - base;
  const size_t devPostUsed = baseFree - cudaFreeBytes();
  LOG(INFO) << "[CTRAN_MEM] test=AllToAllP nLocalRanks="
            << comm->statex_->nLocalRanks() << " rank=" << globalRank
            << " postUsedMiB=" << (postUsed / 1024.0 / 1024.0)
            << " devPostUsedMiB=" << (devPostUsed / 1024.0 / 1024.0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
