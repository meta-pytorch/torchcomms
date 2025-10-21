// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/tests/CtranDistAlgoDevUTBase.h"
#include "comms/ctran/algos/tests/CtranDistAlgoDevUTKernels.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

class CtranDistAlgoDevTestParamFixture
    : public CtranDistAlgoDevTest,
      public ::testing::WithParamInterface<
          std::tuple<ElemTestType, int, unsigned int, size_t, unsigned int>> {};

class CtranDistAlgoDevBcastTestParamFixture
    : public CtranDistAlgoDevTest,
      public ::testing::WithParamInterface<std::tuple<
          BcastTestAlgo,
          ElemTestType,
          int,
          unsigned int,
          unsigned int,
          size_t,
          size_t>> {};

class CtranDistAlgoDevReduceTestParamFixture
    : public CtranDistAlgoDevTest,
      public ::testing::WithParamInterface<
          std::tuple<ElemTestType, int, bool, bool, unsigned int, size_t>> {
 public:
  int checkReduceResults(void* buf, size_t totalCount, int root, int nRanks) {
    std::vector<int> reducedVals(totalCount, 0);
    CUDACHECK_TEST(cudaMemcpy(
        reducedVals.data(),
        buf,
        totalCount * sizeof(int),
        cudaMemcpyDeviceToHost));

    int nerrs = 0;
    std::vector<int> expVals(totalCount, 0);
    for (int i = 0; i < totalCount; i++) {
      int expVal = 0;
      for (int r = 0; r < nRanks; r++) {
        expVal += r + i + totalCount * root;
      }
      if (expVal != reducedVals.at(i)) {
        if (nerrs < 10) {
          printf(
              "[%d] observedVals[%d] = %d, expectedVal = %d\n",
              globalRank,
              i,
              reducedVals.at(i),
              expVal);
        }
        nerrs++;
      }
    }
    return nerrs;
  }
};

static void postElemList(KernelElem* elemList) {
  auto elem = elemList;
  while (elem) {
    elem->post();
    elem = elem->next;
  }
}

static void waitElemListWithStep(KernelElem* elemList, int stepId) {
  auto elem = elemList;
  // wait each elem complete one-by-one
  while (elem) {
    elem->wait();
    EXPECT_EQ(stepId, elem->stepDone);
    elem = elem->next;
  }
}

static void checkElemListFree(KernelElem* elemList) {
  KernelElem* elem = elemList;
  while (elem) {
    EXPECT_TRUE(elem->isFree());
    elem = elem->next;
  }
}

static void checkElemListCompleteAndFree(KernelElem* elemList) {
  KernelElem* elem = elemList;
  while (elem) {
    EXPECT_TRUE(elem->isComplete());
    elem->free();
    elem = elem->next;
  }
}

TEST_P(CtranDistAlgoDevTestParamFixture, PutNotify) {
  const auto& [testType, numElems, nPutGroups, count, nWaitNotifyBlockThreads] =
      GetParam();

  cudaStream_t putStream = nullptr;
  cudaStream_t waitStream = nullptr;
  const int localRank = comm_->ctranComm_->statex_->localRank();
  const int localRanks = comm_->ctranComm_->statex_->nLocalRanks();

  int putPeerLocalRank = (localRank + 1) % localRanks;
  int waitPeerLocalRank = (localRank - 1 + localRanks) % localRanks;

  CUDACHECK_TEST(cudaStreamCreateWithFlags(&putStream, cudaStreamNonBlocking));
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&waitStream, cudaStreamNonBlocking));

  // Put kernel requires nPutGroups thread blocks
  KernelElem* putElemList = nullptr;
  COMMCHECK_TEST(comm_->ctranComm_->ctran_->gpe->allocKernelElems(
      numElems, nPutGroups, &putElemList));

  // Wait kernel requires only 1 thread block
  KernelElem* waitElemList = nullptr;
  COMMCHECK_TEST(comm_->ctranComm_->ctran_->gpe->allocKernelElems(
      numElems, 1, &waitElemList));

  initIpcBufs<int>(count * numElems);
  // Use localBuf as source of put, and ipcBuf as recvBuf
  assignVal<int>(localBuf_, count * numElems, localRank, true);
  assignVal<int>(ipcBuf_, count * numElems, rand());
  // Ensure data has been stored before IPC access
  intraNodeBarrier(comm_);

  // Submit put kernel
  {
    auto elem = putElemList;
    int nPut = 0;
    while (elem) {
      // Assign different chunk of sendBuf for each elem
      size_t offset = nPut * count * sizeof(int);
      elem->putNotify.sendbuff = reinterpret_cast<char*>(localBuf_) + offset;
      elem->putNotify.recvbuff =
          reinterpret_cast<uint64_t>(
              ipcRemMem_.at(putPeerLocalRank)->getBase()) +
          offset;
      elem->putNotify.nbytes = count * sizeof(int);
      elem->putNotify.peerLocalRank = putPeerLocalRank;
      elem->putNotify.notify = true;
      elem->putNotify.ngroups = nPutGroups;
      nPut++;
      elem = elem->next;
    }

    dim3 grid = {nPutGroups, 1, 1};
    dim3 blocks = {1024, 1, 1};
    CUDACHECK_TEST(testKernMultiPutNotifyWrapper(
        testType,
        grid,
        blocks,
        putStream,
        putElemList,
        comm_->ctranComm_->ctran_->algo->getDevState()));

    // recvbuff already assigned; just post for kernel to start
    postElemList(putElemList);
  }

  // Submit wait kernel
  {
    auto elem = waitElemList;
    while (elem) {
      elem->waitNotify.peerLocalRank = waitPeerLocalRank;
      elem->waitNotify.ngroups = nPutGroups;
      elem = elem->next;
    }

    dim3 grid = {1, 1, 1};
    dim3 blocks = {nWaitNotifyBlockThreads, 1, 1};
    CUDACHECK_TEST(testKernMultiWaitNotifyWrapper(
        testType,
        grid,
        blocks,
        waitStream,
        waitElemList,
        comm_->ctranComm_->ctran_->algo->getDevState()));

    // recvbuff already assigned; just post for kernel to start
    postElemList(waitElemList);
  }

  // Wait completion on both put and wait streams
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // check received data in local recvBuf
  checkVals<int>(count, waitPeerLocalRank);

  // check elems have freed
  if (testType == kTestElemComplete) {
    checkElemListCompleteAndFree(putElemList);
    checkElemListCompleteAndFree(waitElemList);
  } else {
    checkElemListFree(putElemList);
    checkElemListFree(waitElemList);
  }

  freeIpcBufs();
  CUDACHECK_TEST(cudaStreamDestroy(putStream));
  CUDACHECK_TEST(cudaStreamDestroy(waitStream));
}

TEST_P(CtranDistAlgoDevBcastTestParamFixture, Bcast) {
  const auto& [testAlgo, testType, nSteps, nGroups, nBcasts, count, lastCount] =
      GetParam();

  int nBcastBlocks = nGroups / nBcasts;
  ASSERT_TRUE(nGroups >= nBcasts);

  ASSERT_TRUE(count >= lastCount) << "lastCount should be smaller than count";

  cudaStream_t stream = nullptr;
  const int localRank = comm_->ctranComm_->statex_->localRank();
  const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();

  CUDACHECK_TEST(cudaStreamCreate(&stream));

  KernelElem* bcastElem = nullptr;
  COMMCHECK_TEST(comm_->ctranComm_->ctran_->gpe->allocKernelElems(
      nBcasts, nBcastBlocks, &bcastElem));

  initIpcBufs<int>(count * nSteps * nBcasts * nLocalRanks);
  // Use localBuf as sendbuf, and ipcBuf as recvbuf in an allgather patter
  assignVal<int>(localBuf_, count * nSteps, localRank, true);
  assignVal<int>(ipcBuf_, count * nSteps * nBcasts * nLocalRanks, rand());
  // Ensure data has been stored before IPC access
  CUDACHECK_TEST(cudaDeviceSynchronize());
  intraNodeBarrier(comm_);

  // Submit bcast kernel
  // Always barrier + flush to ensure every rank has finished copy and every
  // rank can check the local received data on host side
  KernelElem* curElem = bcastElem;
  while (curElem) {
    curElem->bcast.barrier = true;
    curElem->bcast.flushMem = true;
    curElem->bcast.nvectors = nLocalRanks;
    curElem = curElem->next;
  }

  dim3 grid = {nGroups, 1, 1};
  dim3 blocks = {256, 1, 1};
  CUDACHECK_TEST(testKernBcastWrapper(
      testAlgo,
      testType,
      grid,
      blocks,
      stream,
      bcastElem,
      nSteps,
      nBcastBlocks,
      comm_->ctranComm_->ctran_->algo->getDevState()));

  curElem = bcastElem;
  for (int j = 0; j < nBcasts; j++) {
    for (int i = 0; i < nSteps; i++) {
      // Repost the same bcast op nSteps times, similar to usage in collective
      // algorithm
      size_t srcStart = count * i * sizeof(int);
      size_t curCount = (i == nSteps - 1) ? lastCount : count;
      size_t bcastIdxOffset = j * count * nSteps * nLocalRanks * sizeof(int);
      size_t dstStart =
          bcastIdxOffset + count * (nLocalRanks * i + localRank) * sizeof(int);

      curElem->bcast.count = curCount;
      curElem->bcast.src = reinterpret_cast<char*>(localBuf_) + srcStart;
      for (int peer = 0; peer < nLocalRanks; peer++) {
        char* dstBase = peer == localRank
            ? reinterpret_cast<char*>(ipcMem_->getBase())
            : reinterpret_cast<char*>(ipcRemMem_.at(peer)->getBase());
        curElem->bcast.dsts[peer] = dstBase + dstStart;
      }
      curElem->post();
      curElem->wait();
    }
    curElem = curElem->next;
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // check received data in local recvbuf from other ranks in each step
  for (int j = 0; j < nBcasts; j++) {
    for (int i = 0; i < nSteps; i++) {
      size_t curCount = (i == nSteps - 1) ? lastCount : count;
      for (int peer = 0; peer < nLocalRanks; peer++) {
        size_t bcastIdxOffset = j * count * nSteps * nLocalRanks * sizeof(int);
        size_t dstStart =
            bcastIdxOffset + count * (nLocalRanks * i + peer) * sizeof(int);
        checkVals<int>(curCount, peer + count * i, dstStart);
      }
    }
  }

  // check elems have freed
  if (testType == kTestElemComplete) {
    checkElemListCompleteAndFree(bcastElem);
  } else {
    checkElemListFree(bcastElem);
  }

  freeIpcBufs();
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

TEST_P(CtranDistAlgoDevReduceTestParamFixture, MultiReduce) {
  const auto& [testType, numElems, inplace, barrierLastElem, nGroups, count] =
      GetParam();

  cudaStream_t stream = nullptr;
  const int localRank = comm_->ctranComm_->statex_->localRank();
  const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();

  CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  initIpcBufs<int>(count * numElems * nLocalRanks);
  // Use ipcBuf as source of reduce, and localBuf as destination
  assignVal<int>(localBuf_, count * numElems * nLocalRanks, rand());
  assignVal<int>(ipcBuf_, count * numElems * nLocalRanks, localRank, true);
  // Ensure data has been stored before IPC access
  intraNodeBarrier(comm_);
  int nSteps = (testType == kTestElemRepost) ? 5 : 1;

  if (testType == kTestElemRepost && inplace) {
    GTEST_SKIP() << "Do not support kTestElemRepost with inplace, skip test";
  }

  void* dstBase = nullptr;
  if (inplace) {
    // For in-place case, each rank reduces to different portion
    dstBase = reinterpret_cast<char*>(ipcMem_->getBase()) +
        count * numElems * localRank * sizeof(int);
  } else {
    dstBase = localBuf_;
  }

  KernelElem* reduceELemList = nullptr;
  COMMCHECK_TEST(comm_->ctranComm_->ctran_->gpe->allocKernelElems(
      numElems, nGroups, &reduceELemList));

  // In-place reduce from all local ranks for each elem. Each elem starts at
  // different offset
  auto elem = reduceELemList;
  // Each rank starts from different portion similar to reduceScatter
  size_t srcStart = count * numElems * localRank * sizeof(int);
  size_t offset = 0;
  while (elem) {
    elem->reduce.count = count;
    for (int i = 0; i < nLocalRanks; i++) {
      char* srcBase = i == localRank
          ? reinterpret_cast<char*>(ipcMem_->getBase())
          : reinterpret_cast<char*>(ipcRemMem_.at(i)->getBase());
      elem->reduce.srcs[i] = srcBase + srcStart + offset;
    }
    elem->reduce.nsrcs = nLocalRanks;
    elem->reduce.dsts[0] = reinterpret_cast<char*>(dstBase) + offset;
    elem->reduce.ndsts = 1;
    // Mark each step as final since each of them handles different portion
    elem->reduce.isFinal = true;
    offset += count * sizeof(int);

    if (elem->next == nullptr) {
      // Flush at last element at each step, so that host side can read
      elem->reduce.flushMem = true;
      // Optionally barrier at last element so that host side can read remote
      // ranks' results too
      if (barrierLastElem) {
        elem->reduce.barrier = true;
      }
    }

    elem = elem->next;
  }

  dim3 grid = {nGroups, 1, 1};
  dim3 blocks = {1024, 1, 1};
  CUDACHECK_TEST(testKernMultiReduceWrapper(
      testType,
      grid,
      blocks,
      stream,
      reduceELemList,
      nSteps,
      comm_->ctranComm_->ctran_->algo->getDevState()));

  if (testType == kTestElemRepost) {
    // recvbuff already assigned; just post for kernel to start
    for (int stepId = 0; stepId < nSteps; stepId++) {
      postElemList(reduceELemList);

      // wait for kernel to finish all reduces for each step
      waitElemListWithStep(reduceELemList, stepId);

      // check reduced data in local recvBuf after each step
      EXPECT_EQ(
          checkReduceResults(dstBase, count * numElems, localRank, nLocalRanks),
          0)
          << " check at dstBase " << dstBase << " on rank " << localRank
          << " at stepId " << stepId;
    }
  } else {
    postElemList(reduceELemList);
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // if flushMem + barrier is performed, all ranks' result should be visible.
  // Let rank0 check inplace results also for other ranks
  if (barrierLastElem && localRank == 0) {
    for (int peer = 1; peer < nLocalRanks; peer++) {
      void* peerDst = reinterpret_cast<char*>(ipcRemMem_.at(peer)->getBase()) +
          count * numElems * peer * sizeof(int);

      EXPECT_EQ(
          checkReduceResults(peerDst, count * numElems, peer, nLocalRanks), 0)
          << " check at peerDst " << peerDst << " from peer " << peer;
    }
  }

  // check elems have freed
  if (testType == kTestElemComplete || testType == kTestElemRepost) {
    checkElemListCompleteAndFree(reduceELemList);
  } else {
    checkElemListFree(reduceELemList);
  }

  freeIpcBufs();
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

INSTANTIATE_TEST_SUITE_P(
    CtranDistAlgoDevTest,
    CtranDistAlgoDevTestParamFixture,
    ::testing::Values(
        // CompleteOrFree, numElems, nPutGroups, count of int,
        // nWaitNotifyBlockThreads
        std::make_tuple(kTestElemComplete, 5, 2, 4096, 1),
        std::make_tuple(kTestElemFree, 5, 4, 4096, 2),
        std::make_tuple(kTestElemFree, 5, 4, 4096, 5),
        // unaligned size
        std::make_tuple(kTestElemComplete, 5, 4, 4093, 10)),
    [&](const testing::TestParamInfo<
        CtranDistAlgoDevTestParamFixture::ParamType>& info) {
      return ((std::get<0>(info.param) == kTestElemComplete)
                  ? "testElemComplte_"
                  : "testElemFree_") +
          std::to_string(std::get<1>(info.param)) + "nElems_" +
          std::to_string(std::get<2>(info.param)) + "putGroups_" +
          std::to_string(std::get<3>(info.param)) + "int_" +
          std::to_string(std::get<4>(info.param)) + "WaitNotifyBlockThreads_";
    });

INSTANTIATE_TEST_SUITE_P(
    CtranDistAlgoDevTest,
    CtranDistAlgoDevReduceTestParamFixture,
    ::testing::Values(
        // CompleteOrFree, numElems, inplace, barrierLastElem, nGroups, count
        std::make_tuple(kTestElemComplete, 5, false, false, 2, 4096),
        std::make_tuple(kTestElemRepost, 1, false, false, 4, 8192),
        std::make_tuple(kTestElemFree, 5, true, false, 4, 8192),
        // unaligned size
        std::make_tuple(kTestElemComplete, 1, false, false, 1, 51),
        // out-of-place
        std::make_tuple(kTestElemFree, 5, false, false, 4, 4096),
        // inplace with barrier
        std::make_tuple(kTestElemComplete, 5, true, true, 4, 4096)),
    [&](const testing::TestParamInfo<
        CtranDistAlgoDevReduceTestParamFixture::ParamType>& info) {
      return ((std::get<0>(info.param) == kTestElemComplete)
                  ? "testElemComplte_"
                  : ((std::get<0>(info.param) == kTestElemRepost)
                         ? "testElemRepost_"
                         : "testElemFree_")) +
          std::to_string(std::get<1>(info.param)) + "nElems_" +
          (std::get<2>(info.param) ? "InPlace_" : "OutOfPlace_") +
          (std::get<3>(info.param) ? "barrierLastElem_"
                                   : "nobarrierLastElem_") +
          std::to_string(std::get<4>(info.param)) + "groups_" +
          std::to_string(std::get<5>(info.param)) + "count";
    });

INSTANTIATE_TEST_SUITE_P(
    CtranDistAlgoDevTest,
    CtranDistAlgoDevBcastTestParamFixture,
    ::testing::Values(
        // BcastAlgo nSteps, nGroups, nBcasts, count of int in each step, count
        // of int in last step
        std::make_tuple(kTestDefaultBcast, kTestElemComplete, 5, 2, 1, 16, 0),
        std::make_tuple(kTestDefaultBcast, kTestElemComplete, 5, 2, 2, 16, 0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            32,
            8,
            1,
            1048576,
            2),
        // unaligned size
        std::make_tuple(kTestDefaultBcast, kTestElemComplete, 5, 1, 1, 15, 0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            4,
            1,
            1048577,
            0),
        // switch to kTestMultiPutBcast
        std::make_tuple(kTestMultiPutBcast, kTestElemComplete, 5, 2, 1, 16, 0),
        std::make_tuple(kTestMultiPutBcast, kTestElemComplete, 5, 2, 2, 16, 0),
        std::make_tuple(
            kTestMultiPutBcast,
            kTestElemComplete,
            32,
            8,
            1,
            1048576,
            2),
        // unaligned size
        std::make_tuple(kTestMultiPutBcast, kTestElemComplete, 5, 1, 1, 15, 0),
        std::make_tuple(
            kTestMultiPutBcast,
            kTestElemComplete,
            5,
            4,
            1,
            1048577,
            0)),
    [&](const testing::TestParamInfo<
        CtranDistAlgoDevBcastTestParamFixture::ParamType>& info) {
      auto algo = std::get<0>(info.param);
      std::string algoStr =
          (algo == kTestDefaultBcast) ? "defaultBcast_" : "multiPutBcast_";
      return algoStr + "testElemComplte_" +
          std::to_string(std::get<2>(info.param)) + "nSteps" +
          std::to_string(std::get<3>(info.param)) + "groups_" +
          std::to_string(std::get<4>(info.param)) + "nBcasts_" +
          std::to_string(std::get<5>(info.param)) + "int_" +
          std::to_string(std::get<6>(info.param)) + "last";
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
