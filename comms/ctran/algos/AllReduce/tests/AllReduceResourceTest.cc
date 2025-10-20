// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllReduce/AllReduceDevTypes.h"
#include "comms/ctran/algos/AllReduce/AllReduceResourceImpl.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

using ctran::algos::allreduce::AllReduceComm;
using ctran::algos::allreduce::AllReduceDevConn;
using ctran::algos::allreduce::AllReduceResourceBufName;
using ctran::algos::allreduce::AllReduceResourceImpl;

class CtranAllReduceResourceTest : public NcclxBaseTest {
 public:
  CtranAllReduceResourceTest() = default;
  void SetUp() override {
    // This test requires CTRAN to be enabled
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    // TODO: remove this when memCache does not rely on colltrace
    setenv("NCCL_COLLTRACE", "trace", 1);
    NcclxBaseTest::SetUp();
    comm_ = createNcclComm(
        globalRank, numRanks, localRank, false, nullptr, server.get());
  }

  void TearDown() override {
    finalizeNcclComm(globalRank, server.get());
    NCCLCHECK_TEST(ncclCommDestroy(comm_));
    NcclxBaseTest::TearDown();
  }

 protected:
  ncclComm_t comm_{nullptr};
};

namespace {
#define ARGTOSTR(arg) #arg
#define CHECK_VALID_BUF(ref, bufname)                            \
  {                                                              \
    auto regBuf = ref.getBuf(AllReduceResourceBufName::bufname); \
    ASSERT_NE(regBuf.ptr, nullptr)                               \
        << " in regBuf " << ARGTOSTR(bufname) << std::endl;      \
    ASSERT_NE(regBuf.size, 0)                                    \
        << " in regBuf " << ARGTOSTR(bufname) << std::endl;      \
  }

#define CHECK_VALID_REGBUF(ref, bufname)                            \
  {                                                                 \
    auto regBuf = ref.getRegBuf(AllReduceResourceBufName::bufname); \
    ASSERT_NE(regBuf.ptr, nullptr)                                  \
        << " in regBuf " << ARGTOSTR(bufname) << std::endl;         \
    ASSERT_NE(regBuf.size, 0)                                       \
        << " in regBuf " << ARGTOSTR(bufname) << std::endl;         \
    ASSERT_NE(regBuf.regHdl, nullptr)                               \
        << " in regBuf " << ARGTOSTR(bufname) << std::endl;         \
  }

#define CHECK_VALID_REMBUF(ref, bufname, myRank)                            \
  {                                                                         \
    auto remRegBufs = ref.getRemRegBufs(AllReduceResourceBufName::bufname); \
    ASSERT_GT(remRegBufs.size(), 0);                                        \
    for (auto& remRegBuf : remRegBufs) {                                    \
      ASSERT_NE(remRegBuf.ptr, nullptr)                                     \
          << " in remRegBuf " << ARGTOSTR(bufname) << std::endl;            \
      ASSERT_GE(remRegBuf.peerRank, 0)                                      \
          << " in remRegBuf " << ARGTOSTR(bufname) << std::endl;            \
      if (remRegBuf.peerRank != myRank) {                                   \
        ASSERT_NE(remRegBuf.rkey.backend, CtranMapperBackend::UNSET)        \
            << " in remRegBuf " << ARGTOSTR(bufname) << std::endl;          \
      }                                                                     \
    }                                                                       \
  }
}; // namespace

TEST_F(CtranAllReduceResourceTest, InitDestroy) {
  if (!ctranInitialized(comm_->ctranComm_.get())) {
    GTEST_SKIP() << "Skip test because ctranInitialized returns false";
  }

  const int nBlocks = 4; // Number of blocks
  const int numIter = 10;
  const auto myRank = comm_->ctranComm_->statex_->rank();

  auto usedBytesBase =
      ncclx::memory::memCacheAllocator::getInstance()->getUsedMem();
  auto numUsedSegsBeforeInit =
      ncclx::memory::memCacheAllocator::getInstance()->getNumUsedReg();

  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (int x = 0; x < numIter; x++) {
    auto resource = std::make_unique<AllReduceResourceImpl>(
        comm_->ctranComm_->statex_.get(),
        comm_->ctranComm_->ctran_->mapper.get(),
        &comm_->logMetaData);
    ASSERT_NE(resource, nullptr);
    ASSERT_EQ(
        resource->initAllReduceDirectResourceAsync(nBlocks, stream),
        commSuccess);
    cudaStreamSynchronize(stream);

    auto numUsedSegsAfterInit =
        ncclx::memory::memCacheAllocator::getInstance()->getNumUsedReg();

    // memory pool may not release the memory after allreduce destroy, thus get
    // delta based on usage before first allreduce init
    auto usedBytes =
        ncclx::memory::memCacheAllocator::getInstance()->getUsedMem() -
        usedBytesBase;

    // Verify buffer references are all set
    auto& ref = resource->getRef();

    // Check regular buffers
    CHECK_VALID_BUF(ref.bufs, kReduceComm);
    CHECK_VALID_BUF(ref.bufs, kLocalPeerStructures);
    CHECK_VALID_BUF(ref.bufs, kDevPeers);

    // Check registered buffers (with handle)
    CHECK_VALID_REGBUF(ref.bufs, kTmpbuff);
    CHECK_VALID_REGBUF(ref.bufs, kPostFlags);
    CHECK_VALID_REGBUF(ref.bufs, kCompleteFlags);

    // Check remote registered buffers
    CHECK_VALID_REMBUF(ref.bufs, kTmpbuff, myRank);
    CHECK_VALID_REMBUF(ref.bufs, kPostFlags, myRank);
    CHECK_VALID_REMBUF(ref.bufs, kCompleteFlags, myRank);

    // Verify AllReduceComm structure is valid
    ASSERT_NE(ref.allReduceComms, nullptr);

    // Verify send and recv resources (if applicable)
    // Note: AllReduceResourceImpl appears to be for direct intra-node
    // communication so these might not be used in the same way as the MultiRing
    // version

    ASSERT_EQ(resource->destroy(), commSuccess);
    resource.reset();

    auto numUsedSegsAfterDestroy =
        ncclx::memory::memCacheAllocator::getInstance()->getNumUsedReg();

    // Track memory usage from memory pool
    // - After init, expect increased used segments
    EXPECT_LT(numUsedSegsBeforeInit, numUsedSegsAfterInit);
    // - After destory, expect used segments are released
    EXPECT_EQ(numUsedSegsBeforeInit, numUsedSegsAfterDestroy);

    if (myRank == 0) {
      std::cout << "InitDestroy finished iter " << x << ", used segments "
                << numUsedSegsAfterInit - numUsedSegsBeforeInit
                << " total bytes " << usedBytes << std::endl;
    }
  }
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

TEST_F(CtranAllReduceResourceTest, IsInitializedCheck) {
  if (!ctranInitialized(comm_->ctranComm_.get())) {
    GTEST_SKIP() << "Skip test because ctranInitialized returns false";
  }

  const int nBlocks = 2;

  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  auto resource = std::make_unique<AllReduceResourceImpl>(
      comm_->ctranComm_->statex_.get(),
      comm_->ctranComm_->ctran_->mapper.get(),
      &comm_->logMetaData);

  // Before initialization
  EXPECT_FALSE(resource->isInitialized());

  // After initialization
  ASSERT_EQ(
      resource->initAllReduceDirectResourceAsync(nBlocks, stream), commSuccess);
  cudaStreamSynchronize(stream);
  EXPECT_TRUE(resource->isInitialized());

  // After destroy
  ASSERT_EQ(resource->destroy(), commSuccess);
  // Note: isInitialized may still return true after destroy because bufMngr_
  // may still be committed and exchanged even after release

  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
