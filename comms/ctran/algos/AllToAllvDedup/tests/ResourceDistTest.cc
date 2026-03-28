// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAllvDedup/ResourceImpl.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestUtils.h"

// #define VERBOSE
using ctran::alltoallvdedup::PersistArgs;
using ctran::alltoallvdedup::PersistConfig;
using ctran::alltoallvdedup::ResourceBufName;
using ctran::alltoallvdedup::ResourceImpl;

class CtranAllToAllvDedupResourceTest : public ctran::CtranDistTestFixture {
 public:
  CtranAllToAllvDedupResourceTest() = default;
  void SetUp() override {
    // TODO: remove this when memCache does not rely on colltrace
    setenv("NCCL_COLLTRACE", "trace", 1);
    ctran::CtranDistTestFixture::SetUp();
    ctranComm_ = makeCtranComm();
  }

  void TearDown() override {
    ctranComm_.reset();
    ctran::CtranDistTestFixture::TearDown();
  }

 protected:
  std::unique_ptr<CtranComm> ctranComm_;
};

namespace {
#define ARGTOSTR(arg) #arg
#define CHECK_VALID_BUF(ref, bufname)                       \
  {                                                         \
    auto regBuf = ref.getBuf(ResourceBufName::bufname);     \
    ASSERT_NE(regBuf.ptr, nullptr)                          \
        << " in regBuf " << ARGTOSTR(bufname) << std::endl; \
    ASSERT_NE(regBuf.size, 0)                               \
        << " in regBuf " << ARGTOSTR(bufname) << std::endl; \
    ASSERT_NE(regBuf.regHdl, nullptr)                       \
        << " in regBuf " << ARGTOSTR(bufname) << std::endl; \
  }

#define CHECK_VALID_REMBUF(ref, bufname, myRank)                     \
  {                                                                  \
    auto remRegBufs = ref.getRemBufs(ResourceBufName::bufname);      \
    ASSERT_GT(remRegBufs.size(), 0);                                 \
    for (auto& remRegBuf : remRegBufs) {                             \
      ASSERT_NE(remRegBuf.ptr, nullptr)                              \
          << " in remRegBuf " << ARGTOSTR(bufname) << std::endl;     \
      ASSERT_GE(remRegBuf.peerRank, 0)                               \
          << " in remRegBuf " << ARGTOSTR(bufname) << std::endl;     \
      if (remRegBuf.peerRank != myRank) {                            \
        ASSERT_NE(remRegBuf.rkey.backend, CtranMapperBackend::UNSET) \
            << " in remRegBuf " << ARGTOSTR(bufname) << std::endl;   \
      }                                                              \
    }                                                                \
  }
}; // namespace

TEST_F(CtranAllToAllvDedupResourceTest, InitDestroy) {
  if (!ctranInitialized(ctranComm_.get())) {
    GTEST_SKIP() << "Skip test because ctranInitialized returns false";
  }

  PersistArgs pArgs = {
      .totalNumSendBlocks = 16,
      .blockCount = 16,
      .blockNumRecvBuckets = 4,
      .numRecvBuckets = 2,
      .datatype = commBfloat16,
  };

  PersistConfig config = {
      .tmpChunkSize = 128,
      .tmpNumChunks = 2,
  };

  const int numIter = 10;
  const auto myRank = ctranComm_->statex_->rank();

  auto usedBytesBase =
      ncclx::memory::memCacheAllocator::getInstance()->getUsedMem();
  auto numUsedSegsBeforeInit =
      ncclx::memory::memCacheAllocator::getInstance()->getNumUsedReg();

  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));
  for (int x = 0; x < numIter; x++) {
    auto resource = std::make_unique<ResourceImpl>(
        ctranComm_->statex_.get(),
        ctranComm_->ctran_->mapper.get(),
        &ctranComm_->logMetaData_);
    ASSERT_NE(resource, nullptr);
    ASSERT_EQ(resource->initialize(pArgs, config, stream), ncclSuccess);

    auto numUsedSegsAfterInit =
        ncclx::memory::memCacheAllocator::getInstance()->getNumUsedReg();

    // memory pool may not release the memory after dedup destroy, thus get
    // delta based on usage before first dedup init
    auto usedBytes =
        ncclx::memory::memCacheAllocator::getInstance()->getUsedMem() -
        usedBytesBase;

    // Verify buffer references are all set
    auto& ref = resource->getRef();

    CHECK_VALID_BUF(ref, kTmpRecvOffsets);

    CHECK_VALID_BUF(ref, kSendGKSyncs);

    CHECK_VALID_BUF(ref, kTmpSendBuff);

    CHECK_VALID_BUF(ref, kTmpFwdBuff);
    CHECK_VALID_REMBUF(ref, kTmpFwdBuff, myRank);

    CHECK_VALID_BUF(ref, kTmpRecvBuff);
    CHECK_VALID_REMBUF(ref, kTmpRecvBuff, myRank);

    ASSERT_EQ(resource->destroy(), ncclSuccess);
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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
