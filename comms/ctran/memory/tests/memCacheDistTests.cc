// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/Random.h>
#include <folly/init/Init.h>
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

#include "comm.h"
#include "nccl.h"

#include "comms/ctran/memory/Utils.h"
#include "comms/ctran/memory/memCacheAllocator.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/transport/transportProxy.h"

constexpr int kNp2pRegionPerComm = 1;

class NcclxMemDistTestFixture : public NcclxBaseTestFixture {
  static constexpr float kTol = 1e-5;

 public:
  ncclConfig_t config{};
  std::vector<void*> sendBufs;
  std::vector<void*> recvBufs;
  std::vector<cudaStream_t> streams;
  int allocatorRefNum{0};
  enum ncclFunc {
    allReduce = 0,
    allToAll = 1,
    reduceScatter = 2,
    allGather = 3,
    sendRecv = 4
  };

 protected:
  void SetUp() override {
    ctran::utils::commCudaLibraryInit();
    setenv("NCCL_DEBUG_SUBSYS", "ALLOC", 0);
    setenv("NCCL_DEBUG", "WARN", 0);
    NcclxBaseTestFixture::SetUp();
    allocatorRefNum =
        ncclx::memory::memCacheAllocator::getInstance().use_count();
    config = NCCL_CONFIG_INITIALIZER;
    config.commDesc = "NcclxMemDistTestFixture";
    config.splitShare = 0;
  }

  void TearDown() override {
    for (auto& sendBuf : sendBufs) {
      CUDACHECK_TEST(cudaFree(sendBuf));
    }
    for (auto& recvBuf : recvBufs) {
      CUDACHECK_TEST(cudaFree(recvBuf));
    }
    for (auto& stream : streams) {
      CUDACHECK_TEST(cudaStreamDestroy(stream));
    }

    sendBufs.clear();
    recvBufs.clear();
    streams.clear();
    if (NCCL_USE_MEM_CACHE) {
      EXPECT_EQ(
          allocatorRefNum,
          ncclx::memory::memCacheAllocator::getInstance().use_count());
      // reset the global allocator
      ncclx::memory::memCacheAllocator::getInstance()->reset();
    }
    NcclxBaseTestFixture::TearDown();
  }

  inline size_t getDefaultCount() {
    // 1K elements for LL for quick test, 1M otherwise
    return (NCCL_PROTO == "LL") ? 1 << 10 : 1 << 20;
  }

  void createStream() {
    cudaStream_t stream;
    CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    streams.push_back(stream);
  }

  void createBufs() {
    sendBufs.push_back(nullptr);
    recvBufs.push_back(nullptr);
  }

  // create a child comm and its asscoiated streams/buffers
  void splitChildComm(
      ncclComm_t* childComm,
      int groupSize,
      ncclConfig_t* childCommConfig) {
    int nGroups = comm->nRanks / groupSize;
    // split ranks into groupSize round robin style
    NCCLCHECK_TEST(ncclCommSplit(
        comm,
        globalRank % nGroups,
        globalRank / nGroups,
        childComm,
        childCommConfig));
  }

  template <typename T>
  void prepBuffer(
      void** buf,
      size_t count,
      T initVal,
      ncclDataType_t dataType = ncclInt) {
    ASSERT_NE(comm, nullptr);
    CUDACHECK_TEST(cudaMalloc(buf, count * ncclTypeSize(dataType)));
    assignChunkValue((T*)*buf, count, initVal);
    // Ensure value has been set before collective runs on nonblocking stream
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  template <typename T>
  bool
  checkLocalResults(T* recvBuf, int count, ncclComm_t comm, ncclFunc func) {
    int nRanks = comm->nRanks;
    int rank = comm->rank;
    if (func == ncclFunc::allReduce) {
      T expectedVal = nRanks * (nRanks - 1) / 2;
      return checkChunkValue(recvBuf, count, expectedVal) == 0;
    } else if (func == ncclFunc::allToAll) {
      for (int i = 0; i < nRanks; i++) {
        T expectedVal = i;
        if (checkChunkValue(recvBuf + count * i, count, expectedVal) != 0) {
          return false;
        }
      }
      return true;
    } else if (func == ncclFunc::reduceScatter) {
      T expectedVal = nRanks * (nRanks - 1) / 2;
      return checkChunkValue(recvBuf, count, expectedVal) == 0;
    } else if (func == ncclFunc::allGather) {
      for (int i = 0; i < nRanks; i++) {
        T expectedVal = i;
        if (checkChunkValue(recvBuf + count * i, count, expectedVal) != 0) {
          return false;
        }
      }
      return true;
    } else if (func == ncclFunc::sendRecv) {
      T expectedVal = rank - 1;
      return checkChunkValue(recvBuf, count, expectedVal) == 0;
    }
    return false;
  }
};

TEST_P(NcclxMemDistTestFixture, AllReduce) {
  createStream();
  createBufs();
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, ncclUid, globalRank, &config));
  ASSERT_NE(nullptr, comm);
  size_t count = getDefaultCount();
  prepBuffer(&sendBufs.at(0), count, comm->rank);
  prepBuffer(&recvBufs.at(0), count, 0);

  // run baseline allreduce
  EXPECT_EQ(
      ncclAllReduce(
          sendBufs.at(0),
          recvBufs.at(0),
          count,
          ncclInt,
          ncclSum,
          comm,
          streams.at(0)),
      ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));
  // check value correctness
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(0), count, comm, ncclFunc::allReduce));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

// FIXME: this is a unsupported case for now
TEST_P(NcclxMemDistTestFixture, coalescedAllReduce) {
  createStream();
  createBufs();
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, ncclUid, globalRank, &config));
  ASSERT_NE(nullptr, comm);
  size_t count = getDefaultCount();
  prepBuffer(&sendBufs.at(0), count, comm->rank);
  prepBuffer(&recvBufs.at(0), count, 0);
  // posting multiple allreduce in single group, e.g., coalesced
  constexpr int numAR = 10;
  EXPECT_EQ(ncclGroupStart(), ncclSuccess);
  for (int i = 0; i < numAR; ++i) {
    EXPECT_EQ(
        ncclAllReduce(
            sendBufs.at(0),
            recvBufs.at(0),
            count,
            ncclInt,
            ncclSum,
            comm,
            streams.at(0)),
        ncclSuccess);
  }
  EXPECT_EQ(ncclGroupEnd(), ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_P(NcclxMemDistTestFixture, Alltoall) {
  if (NCCL_PROTO == "LL128") {
    GTEST_SKIP()
        << "A2A/send/recv does not use LL128 and NCCL will fallback to LL or Simple, skip to avoid repeated tests";
  }

  createStream();
  createBufs();
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, ncclUid, globalRank, &config));
  ASSERT_NE(nullptr, comm);
  size_t countPerRank = getDefaultCount();
  prepBuffer(&sendBufs.at(0), countPerRank * comm->nRanks, comm->rank);
  prepBuffer(&recvBufs.at(0), countPerRank * comm->nRanks, 0);
  auto res = ncclAllToAll(
      sendBufs.at(0),
      recvBufs.at(0),
      countPerRank,
      ncclInt,
      comm,
      streams.at(0));
  EXPECT_EQ(res, ncclSuccess);
  if (NCCL_USE_MEM_CACHE) {
    EXPECT_GE(comm->memCache->getNumUsedReg(), 0);
  }
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));
  // check value correctness
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(0), countPerRank, comm, ncclFunc::allToAll));
  // make sure all reserved buffers are released
  if (NCCL_USE_MEM_CACHE) {
    comm->transportProxy_->waitAll();
    EXPECT_EQ(comm->memCache->getNumUsedReg(), kNp2pRegionPerComm);
  }
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_P(NcclxMemDistTestFixture, splitCommMemSharingNoOverlap) {
  for (int i = 0; i < 2; ++i) {
    createBufs();
    createStream();
  }
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, ncclUid, globalRank, &config));
  ASSERT_NE(nullptr, comm);
  ncclComm_t childComm = nullptr;
  ncclConfig_t childCommConfig = NCCL_CONFIG_INITIALIZER;
  childCommConfig.commDesc = "child_communicator";
  splitChildComm(
      &childComm, comm->ctranComm_->statex_.get()->nRanks(), &childCommConfig);
  ASSERT_NE(nullptr, childComm);
  // run baseline large allreduce on comm
  size_t count = getDefaultCount();
  prepBuffer(&sendBufs.at(0), count, comm->rank);
  prepBuffer(&recvBufs.at(0), count, 0);
  prepBuffer(&sendBufs.at(1), count, childComm->rank);
  prepBuffer(&recvBufs.at(1), count, 0);
  EXPECT_EQ(
      ncclAllReduce(
          sendBufs.at(0),
          recvBufs.at(0),
          count,
          ncclInt,
          ncclSum,
          comm,
          streams.at(0)),
      ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(0), count, comm, ncclFunc::allReduce));
  size_t rootCommMemUsed = 0;
  if (NCCL_USE_MEM_CACHE) {
    EXPECT_NE(comm->memCache, nullptr);
    rootCommMemUsed = comm->memCache->getNumAllocReg();
    comm->transportProxy_->waitAll();
  }
  // run baseline large allreduce on ChildComm
  EXPECT_EQ(
      ncclAllReduce(
          sendBufs.at(1),
          recvBufs.at(1),
          count,
          ncclInt,
          ncclSum,
          childComm,
          streams.at(1)),
      ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(1)));
  if (NCCL_USE_MEM_CACHE) {
    // memory pool size should not increase
    EXPECT_EQ(
        rootCommMemUsed + kNp2pRegionPerComm,
        childComm->memCache->getNumAllocReg());
  }
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(1), count, childComm, ncclFunc::allReduce));
  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_P(NcclxMemDistTestFixture, splitCommOverlapAR) {
  for (int i = 0; i < 2; ++i) {
    createBufs();
    createStream();
  }
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, ncclUid, globalRank, &config));
  ASSERT_NE(nullptr, comm);
  ncclComm_t childComm = nullptr;
  // create a child comm, duplicate of comm
  ncclConfig_t childCommConfig = NCCL_CONFIG_INITIALIZER;
  childCommConfig.commDesc = "child_communicator";
  splitChildComm(
      &childComm, comm->ctranComm_->statex_.get()->nRanks(), &childCommConfig);
  ASSERT_NE(nullptr, childComm);

  size_t count = getDefaultCount();
  prepBuffer(&sendBufs.at(0), count, comm->rank);
  prepBuffer(&recvBufs.at(0), count, 0);
  prepBuffer(&sendBufs.at(1), count, childComm->rank);
  prepBuffer(&recvBufs.at(1), count, 0);

  // run baseline large allreduce on comm
  //  first allreduce from parent comm, to ensure all required buffers are
  //  allocated
  EXPECT_EQ(
      ncclAllReduce(
          sendBufs.at(0),
          recvBufs.at(0),
          count,
          ncclInt,
          ncclSum,
          comm,
          streams.at(0)),
      ncclSuccess);
  size_t prevNumRegions = 0;
  if (NCCL_USE_MEM_CACHE) {
    prevNumRegions = comm->memCache->getNumAllocReg();
  }
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(0), count, comm, ncclFunc::allReduce));
  // Wait till parent comm releases all buffers
  if (NCCL_USE_MEM_CACHE) {
    comm->transportProxy_->waitAll();
  }
  // child comm post allreduce after parent finishes up allreduce
  // so child comm can reuse same buffers with parent comm
  EXPECT_EQ(
      ncclAllReduce(
          sendBufs.at(1),
          recvBufs.at(1),
          count,
          ncclInt,
          ncclSum,
          childComm,
          streams.at(1)),
      ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(1)));
  // Wait till child comm releases all buffers
  if (NCCL_USE_MEM_CACHE) {
    childComm->transportProxy_->waitAll();
    EXPECT_EQ(
        childComm->memCache->getNumAllocReg(),
        prevNumRegions + kNp2pRegionPerComm);
  }
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(1), count, comm, ncclFunc::allReduce));
  // parent comm post second allreduce, reuse buffers
  EXPECT_EQ(
      ncclAllReduce(
          sendBufs.at(0),
          recvBufs.at(0),
          count,
          ncclInt,
          ncclSum,
          comm,
          streams.at(0)),
      ncclSuccess);
  // child comm post allreduce and overlapping with parent's allreduce (i.e.,
  // w/o cudaStreamSynchronize) new set of buffers should be allocated
  EXPECT_EQ(
      ncclAllReduce(
          sendBufs.at(1),
          recvBufs.at(1),
          count,
          ncclInt,
          ncclSum,
          comm,
          streams.at(1)),
      ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(1)));
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(0), count, comm, ncclFunc::allReduce));
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(1), count, comm, ncclFunc::allReduce));
  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_P(NcclxMemDistTestFixture, splitCommOverlapA2A) {
  if (NCCL_PROTO == "LL128") {
    GTEST_SKIP()
        << "A2A/send/recv does not use LL128 and NCCL will fallback to LL or Simple, skip to avoid repeated tests";
  }
  for (int i = 0; i < 2; ++i) {
    createBufs();
    createStream();
  }
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, ncclUid, globalRank, &config));
  ASSERT_NE(nullptr, comm);
  // create a child comm, duplicate of comm
  ncclComm_t childComm = nullptr;
  ncclConfig_t childCommConfig = NCCL_CONFIG_INITIALIZER;
  childCommConfig.commDesc = "child_communicator";
  splitChildComm(
      &childComm, comm->ctranComm_->statex_.get()->nRanks(), &childCommConfig);
  ASSERT_NE(nullptr, childComm);
  size_t countPerRank = getDefaultCount();
  prepBuffer(&sendBufs.at(0), countPerRank * comm->nRanks, comm->rank);
  prepBuffer(&recvBufs.at(0), countPerRank * comm->nRanks, 0);
  prepBuffer(
      &sendBufs.at(1), countPerRank * childComm->nRanks, childComm->rank);
  prepBuffer(&recvBufs.at(1), countPerRank * childComm->nRanks, 0);
  // first alltoall from parent comm, to ensure all required buffers are
  // allocated
  EXPECT_EQ(
      ncclAllToAll(
          sendBufs.at(0),
          recvBufs.at(0),
          countPerRank,
          ncclInt,
          comm,
          streams.at(0)),
      ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));
  size_t prevNumRegions = 0;
  if (NCCL_USE_MEM_CACHE) {
    comm->transportProxy_->waitAll();
    prevNumRegions = comm->memCache->getNumAllocReg();
  }
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(0), countPerRank, comm, ncclFunc::allToAll));
  // child comm post alltoall after parent finishes up alltoall
  // so child comm can reuse same buffers with parent comm
  EXPECT_EQ(
      ncclAllToAll(
          sendBufs.at(1),
          recvBufs.at(1),
          countPerRank,
          ncclInt,
          childComm,
          streams.at(1)),
      ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(1)));
  if (NCCL_USE_MEM_CACHE) {
    childComm->transportProxy_->waitAll();
    // should have reused 100% of the allocated buffers since child A2A is
    // only scheduled after parent A2A's buffers have been released
    EXPECT_EQ(
        childComm->memCache->getNumAllocReg(),
        prevNumRegions + kNp2pRegionPerComm);
  }
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(1), countPerRank, comm, ncclFunc::allToAll));
  // put two collectives in the same group, to gurrantee the overlap and they
  // cannot share buffers
  // parent comm post second alltoall, reuse buffers
  EXPECT_EQ(
      ncclAllToAll(
          sendBufs.at(0),
          recvBufs.at(0),
          countPerRank,
          ncclInt,
          comm,
          streams.at(0)),
      ncclSuccess);
  // child comm post alltoall and overlapping with parent's alltoall (i.e.,
  // w/o cudaStreamSynchronize) new set of buffers should be allocated
  EXPECT_EQ(
      ncclAllToAll(
          sendBufs.at(1),
          recvBufs.at(1),
          countPerRank,
          ncclInt,
          comm,
          streams.at(1)),
      ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(1)));

  if (NCCL_USE_MEM_CACHE) {
    childComm->transportProxy_->waitAll();
    // child comm should use more memory regions because they cannot share
    // memory due to overlap
    EXPECT_GE(
        childComm->memCache->getNumAllocReg(),
        prevNumRegions + kNp2pRegionPerComm);
  }
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(0), countPerRank, comm, ncclFunc::allToAll));
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(1), countPerRank, comm, ncclFunc::allToAll));

  // parent comm post third alltoall, reuse buffers
  EXPECT_EQ(
      ncclAllToAll(
          sendBufs.at(0),
          recvBufs.at(0),
          countPerRank,
          ncclInt,
          comm,
          streams.at(0)),
      ncclSuccess);
  // child comm post alltoall and overlapping with parent's alltoall (i.e.,
  // w/o cudaStreamSynchronize) new set of buffers should be allocated
  EXPECT_EQ(
      ncclAllToAll(
          sendBufs.at(1),
          recvBufs.at(1),
          countPerRank,
          ncclInt,
          comm,
          streams.at(1)),
      ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(1)));
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(0), countPerRank, comm, ncclFunc::allToAll));
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(1), countPerRank, childComm, ncclFunc::allToAll));

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_P(NcclxMemDistTestFixture, childCommsOverlapAR) {
  if (this->numRanks < 8) {
    GTEST_SKIP() << "This test requires at least 8 ranks";
  }
  EnvRAII<std::string> algo(NCCL_ALGO, "ring");
  for (int i = 0; i < 2; ++i) {
    createBufs();
    createStream();
  }
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, ncclUid, globalRank, &config));
  ASSERT_NE(nullptr, comm);
  // create the first child comm 1x8
  ncclComm_t childComm = nullptr;
  ncclConfig_t childCommConfig1 = NCCL_CONFIG_INITIALIZER;
  childCommConfig1.commDesc = "child_communicator_1x8";
  splitChildComm(&childComm, 8, &childCommConfig1);
  ASSERT_NE(nullptr, childComm);

  // create the second child comm 1x4
  ncclComm_t childComm2 = nullptr;
  ncclConfig_t childCommConfig2 = NCCL_CONFIG_INITIALIZER;
  childCommConfig2.commDesc = "child_communicator_1x4";
  splitChildComm(&childComm2, 4, &childCommConfig2);
  ASSERT_NE(nullptr, childComm2);

  size_t count = getDefaultCount();
  prepBuffer(&sendBufs.at(0), count, childComm->rank);
  prepBuffer(&recvBufs.at(0), count, 0);
  prepBuffer(&sendBufs.at(1), count, childComm2->rank);
  prepBuffer(&recvBufs.at(1), count, 0);

  // run baseline large allreduce on child comm 1x8
  EXPECT_EQ(
      ncclAllReduce(
          sendBufs.at(0),
          recvBufs.at(0),
          count,
          ncclInt,
          ncclSum,
          childComm,
          streams.at(0)),
      ncclSuccess);
  auto prevNumRegions = 0UL;
  if (NCCL_USE_MEM_CACHE) {
    prevNumRegions = comm->memCache->getNumAllocReg();
  }
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));

  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(0), count, childComm, ncclFunc::allReduce));
  if (NCCL_USE_MEM_CACHE) {
    // Wait till all buffers are released
    childComm->transportProxy_->waitAll();
  }
  EXPECT_EQ(
      ncclAllReduce(
          sendBufs.at(1),
          recvBufs.at(1),
          count,
          ncclInt,
          ncclSum,
          childComm2,
          streams.at(1)),
      ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(1)));
  if (NCCL_USE_MEM_CACHE) {
    // Wait till all buffers are released
    childComm2->transportProxy_->waitAll();
    EXPECT_EQ(
        childComm2->memCache->getNumAllocReg(),
        prevNumRegions + kNp2pRegionPerComm);
  }
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(1), count, childComm2, ncclFunc::allReduce));

  constexpr int kNumIters = 5;
  for (int i = 0; i < kNumIters; ++i) {
    // childComm post second allreduce, reuse buffers
    EXPECT_EQ(
        ncclAllReduce(
            sendBufs.at(0),
            recvBufs.at(0),
            count,
            ncclInt,
            ncclSum,
            childComm,
            streams.at(0)),
        ncclSuccess);
    // childComm2 post allreduce and overlapping with childComm's allreduce
    // (i.e., w/o cudaStreamSynchronize), new set of buffers should be
    // allocated
    EXPECT_EQ(
        ncclAllReduce(
            sendBufs.at(1),
            recvBufs.at(1),
            count,
            ncclInt,
            ncclSum,
            childComm2,
            streams.at(1)),
        ncclSuccess);
    if (NCCL_USE_MEM_CACHE) {
      // should use more memory regions because they cannot share memory due
      // to overlap
      EXPECT_GE(
          childComm2->memCache->getNumAllocReg(),
          prevNumRegions + kNp2pRegionPerComm);
    }
  }
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));
  CUDACHECK_TEST(cudaStreamSynchronize(streams.at(1)));
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(0), count, childComm, ncclFunc::allReduce));
  EXPECT_TRUE(checkLocalResults(
      (int*)recvBufs.at(1), count, childComm2, ncclFunc::allReduce));
  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(childComm2));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_P(NcclxMemDistTestFixture, overlapMultipleColl) {
  for (int i = 0; i < 6; ++i) {
    createBufs();
    /*
      comm -> small 1 count AR (float) followed up large AR (int)
      childComm -> large count A2A (int)
      childComm2 -> middle count AG (double)
      childComm3 -> large count RS (int) + small count send/recv (int)
    */
  }
  for (int i = 0; i < 4; ++i) {
    createStream(); // 4 communicators
  }

  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, ncclUid, globalRank, &config));
  ASSERT_NE(nullptr, comm);

  // create 3 child comms, duplicate of comm
  auto totalRanks = comm->ctranComm_->statex_.get()->nRanks();
  ncclComm_t childComm = nullptr;
  ncclConfig_t childCommConfig = NCCL_CONFIG_INITIALIZER;
  childCommConfig.commDesc = "child_communicator";
  splitChildComm(&childComm, totalRanks, &childCommConfig);
  ASSERT_NE(nullptr, childComm);

  ncclComm_t childComm2 = nullptr;
  ncclConfig_t childCommConfig2 = NCCL_CONFIG_INITIALIZER;
  childCommConfig2.commDesc = "child_communicator2";
  splitChildComm(&childComm2, totalRanks / 2, &childCommConfig2);
  ASSERT_NE(nullptr, childComm2);

  ncclComm_t childComm3 = nullptr;
  ncclConfig_t childCommConfig3 = NCCL_CONFIG_INITIALIZER;
  childCommConfig3.commDesc = "child_communicator3";
  splitChildComm(&childComm3, totalRanks / 4, &childCommConfig3);
  ASSERT_NE(nullptr, childComm3);

  size_t lCount = getDefaultCount();
  size_t sCount = 1 << 10; // 1k BF16 elements
  size_t mCount = 1 << 15;
  size_t lCountPerRank = lCount / comm->nRanks;
  size_t mCountPerRank = mCount / childComm2->nRanks;

  // parent comm runs 2 allreduces
  prepBuffer(&sendBufs.at(0), comm->nRanks, (float)comm->rank, ncclFloat32);
  prepBuffer(&recvBufs.at(0), comm->nRanks, (float)0, ncclFloat32);

  prepBuffer(&sendBufs.at(1), comm->nRanks, comm->rank);
  prepBuffer(&recvBufs.at(1), comm->nRanks, 0);

  // child1 runs A2A
  prepBuffer(&sendBufs.at(2), lCount, childComm->rank);
  prepBuffer(&recvBufs.at(2), lCount, 0);

  // child2 runs allgather
  prepBuffer(
      &sendBufs.at(3), mCountPerRank, (double)childComm2->rank, ncclFloat64);
  prepBuffer(&recvBufs.at(3), mCount, (double)0, ncclFloat64);

  // child3 runs reducescatter
  prepBuffer(&sendBufs.at(4), lCount, childComm3->rank);
  prepBuffer(&recvBufs.at(4), lCountPerRank, 0);

  // child3 runs an extra send/recv, even rank to odd ranks
  prepBuffer(&sendBufs.at(5), sCount, childComm3->rank);
  prepBuffer(&recvBufs.at(5), sCount, 0);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Execute multiple types of collective overlapping each other multiple
  // times
  int numIters = 5;
  for (int i = 0; i < numIters; ++i) {
    EXPECT_EQ(
        ncclAllReduce(
            sendBufs.at(0),
            recvBufs.at(0),
            comm->nRanks,
            ncclFloat32,
            ncclSum,
            comm,
            streams.at(0)),
        ncclSuccess);
    EXPECT_EQ(
        ncclAllReduce(
            sendBufs.at(1),
            recvBufs.at(1),
            comm->nRanks,
            ncclInt,
            ncclSum,
            comm,
            streams.at(0)),
        ncclSuccess);
    EXPECT_EQ(
        ncclAllToAll(
            sendBufs.at(2),
            recvBufs.at(2),
            lCountPerRank,
            ncclInt,
            childComm,
            streams.at(1)),
        ncclSuccess);
    EXPECT_EQ(
        ncclAllGather(
            sendBufs.at(3),
            recvBufs.at(3),
            mCountPerRank,
            ncclFloat64,
            childComm2,
            streams.at(2)),
        ncclSuccess);
    EXPECT_EQ(
        ncclReduceScatter(
            sendBufs.at(4),
            recvBufs.at(4),
            lCountPerRank,
            ncclInt,
            ncclSum,
            childComm3,
            streams.at(3)),
        ncclSuccess);
    if (childComm3->rank % 2 == 0) {
      EXPECT_EQ(
          ncclSend(
              sendBufs.at(5),
              sCount,
              ncclInt,
              childComm3->rank + 1,
              childComm3,
              streams.at(3)),
          ncclSuccess);
    } else {
      EXPECT_EQ(
          ncclRecv(
              recvBufs.at(5),
              sCount,
              ncclInt,
              childComm3->rank - 1,
              childComm3,
              streams.at(3)),
          ncclSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));
    CUDACHECK_TEST(cudaStreamSynchronize(streams.at(1)));
    CUDACHECK_TEST(cudaStreamSynchronize(streams.at(2)));
    CUDACHECK_TEST(cudaStreamSynchronize(streams.at(3)));

    EXPECT_TRUE(checkLocalResults(
        (float*)recvBufs.at(0), comm->nRanks, comm, ncclFunc::allReduce));
    EXPECT_TRUE(checkLocalResults(
        (int*)recvBufs.at(1), comm->nRanks, comm, ncclFunc::allReduce));
    EXPECT_TRUE(checkLocalResults(
        (int*)recvBufs.at(2), lCountPerRank, childComm, ncclFunc::allToAll));
    EXPECT_TRUE(checkLocalResults(
        (double*)recvBufs.at(3),
        mCountPerRank,
        childComm2,
        ncclFunc::allGather));
    EXPECT_TRUE(checkLocalResults(
        (int*)recvBufs.at(4),
        lCountPerRank,
        childComm3,
        ncclFunc::reduceScatter));
    if (childComm3->rank % 2 == 1) {
      EXPECT_TRUE(checkLocalResults(
          (int*)recvBufs.at(5), sCount, childComm3, ncclFunc::sendRecv));
    }

    // post second round of collectives with reverse orders, mimicing
    // forward/backward pass
    if (childComm3->rank % 2 == 0) {
      EXPECT_EQ(
          ncclSend(
              sendBufs.at(5),
              sCount,
              ncclInt,
              childComm3->rank + 1,
              childComm3,
              streams.at(3)),
          ncclSuccess);
    } else {
      EXPECT_EQ(
          ncclRecv(
              recvBufs.at(5),
              sCount,
              ncclInt,
              childComm3->rank - 1,
              childComm3,
              streams.at(3)),
          ncclSuccess);
    }
    EXPECT_EQ(
        ncclReduceScatter(
            sendBufs.at(4),
            recvBufs.at(4),
            lCountPerRank,
            ncclInt,
            ncclSum,
            childComm3,
            streams.at(3)),
        ncclSuccess);
    EXPECT_EQ(
        ncclAllGather(
            sendBufs.at(3),
            recvBufs.at(3),
            mCountPerRank,
            ncclFloat64,
            childComm2,
            streams.at(2)),
        ncclSuccess);
    EXPECT_EQ(
        ncclAllToAll(
            sendBufs.at(2),
            recvBufs.at(2),
            lCountPerRank,
            ncclInt,
            childComm,
            streams.at(1)),
        ncclSuccess);
    EXPECT_EQ(
        ncclAllReduce(
            sendBufs.at(1),
            recvBufs.at(1),
            comm->nRanks,
            ncclInt,
            ncclSum,
            comm,
            streams.at(0)),
        ncclSuccess);
    EXPECT_EQ(
        ncclAllReduce(
            sendBufs.at(0),
            recvBufs.at(0),
            comm->nRanks,
            ncclFloat32,
            ncclSum,
            comm,
            streams.at(0)),
        ncclSuccess);

    // verify recvBuf results
    CUDACHECK_TEST(cudaStreamSynchronize(streams.at(0)));
    CUDACHECK_TEST(cudaStreamSynchronize(streams.at(1)));
    CUDACHECK_TEST(cudaStreamSynchronize(streams.at(2)));
    CUDACHECK_TEST(cudaStreamSynchronize(streams.at(3)));

    EXPECT_TRUE(checkLocalResults(
        (float*)recvBufs.at(0), comm->nRanks, comm, ncclFunc::allReduce));
    EXPECT_TRUE(checkLocalResults(
        (int*)recvBufs.at(1), comm->nRanks, comm, ncclFunc::allReduce));
    EXPECT_TRUE(checkLocalResults(
        (int*)recvBufs.at(2), lCountPerRank, childComm, ncclFunc::allToAll));
    EXPECT_TRUE(checkLocalResults(
        (double*)recvBufs.at(3),
        mCountPerRank,
        childComm2,
        ncclFunc::allGather));
    EXPECT_TRUE(checkLocalResults(
        (int*)recvBufs.at(4),
        lCountPerRank,
        childComm3,
        ncclFunc::reduceScatter));
    if (childComm3->rank % 2 == 1) {
      EXPECT_TRUE(checkLocalResults(
          (int*)recvBufs.at(5), sCount, childComm3, ncclFunc::sendRecv));
    }
  }

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(childComm2));
  NCCLCHECK_TEST(ncclCommDestroy(childComm3));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

INSTANTIATE_TEST_SUITE_P(
    MyTestSuite,
    NcclxMemDistTestFixture,
    testing::Values(
        // Baseline
        NcclxEnvs({{"NCCL_USE_MEM_CACHE", "0"}}),
        // MemOpt + force Simple protocol
        NcclxEnvs(
            {{"NCCL_USE_MEM_CACHE", "1"},
             {"NCCL_PROTO", "Simple"},
             {"NCCL_LAZY_SETUP_CHANNELS", "1"}}),
        // MemOpt + force LL128 protocol
        NcclxEnvs(
            {{"NCCL_USE_MEM_CACHE", "1"},
             {"NCCL_PROTO", "LL128"},
             {"NCCL_LAZY_SETUP_CHANNELS", "1"}}),
        // MemOpt + force LL protocol
        NcclxEnvs(
            {{"NCCL_USE_MEM_CACHE", "1"},
             {"NCCL_PROTO", "LL"},
             {"NCCL_LAZY_SETUP_CHANNELS", "1"}}),
        // MemOpt + baseline auto protocol selection
        NcclxEnvs(
            {{"NCCL_USE_MEM_CACHE", "1"}, {"NCCL_LAZY_SETUP_CHANNELS", "1"}})),
    [](const testing::TestParamInfo<NcclxMemDistTestFixture::ParamType>& info) {
      // generate test-name for a given NcclxEnvs
      std::string name;
      for (const auto& [key, val] : info.param) {
        if (key == "NCCL_USE_MEM_CACHE") {
          name += (val == "1") ? "memOpt" : "Baseline";
        }
        if (key == "NCCL_PROTO") {
          name += "_" + val;
        }
      }
      return name;
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
