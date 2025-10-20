// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <csignal>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/gpe/CtranGpeImpl.h"
// FIXME [REBASE]: update the path once moved to fbcode/comms
#include "comms/ctran/gpe/tests/KernelElemPoolUTKernels.h"
#include "comms/ctran/tests/CtranXPlatUtUtils.h"

class KernelElemPoolTest : public ::testing::Test {
 public:
  int cudaDev;
  KernelElemPoolTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    CUDACHECK_TEST(cudaSetDevice(cudaDev));
  }
};

class KernelElemPoolAbortTest : public ::testing::Test {
 public:
  int cudaDev;

 protected:
  void SetUp() override {
    // this is needed to avoid segfault in testing::KilledBySignal
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    cudaDev = 0;
    EXPECT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

    // Ensure logging is initialized
    setenv("NCCL_DEBUG", "WARN", 0);
    ncclCvarInit();

    // Intentionally skip pool destory since the stack already be in undefined
    // state after testing.
    auto elemPool = new KernelElemPool(10);
    ASSERT_NE(elemPool, nullptr);

    // Prepare a single elem for abort testing
    elem = elemPool->pop(5);
  }

  void TearDown() override {
    elemPool = nullptr;
    elem = nullptr;
  }

  KernelElemPool* elemPool{nullptr};
  KernelElem* elem{nullptr};
};

TEST_F(KernelElemPoolTest, Initialize) {
  constexpr int poolSize = 1000;
  auto elemPool = std::make_unique<KernelElemPool>(poolSize);

  ASSERT_NE(elemPool, nullptr);
  EXPECT_EQ(elemPool->size(), poolSize);
  EXPECT_EQ(elemPool->capacity(), poolSize);
}

TEST_F(KernelElemPoolTest, InvalidNGroup) {
  constexpr int poolSize = 1000;
  auto elemPool = std::make_unique<KernelElemPool>(poolSize);

  ASSERT_NE(elemPool, nullptr);
  EXPECT_EQ(elemPool->size(), poolSize);
  EXPECT_EQ(elemPool->capacity(), poolSize);

  auto elem = elemPool->pop(CTRAN_ALGO_MAX_THREAD_BLOCKS + 1);
  EXPECT_EQ(elem, nullptr);
}

static KernelElem* popElemList(
    std::unique_ptr<KernelElemPool>& elemPool,
    const int nElems,
    const int ngroups) {
  // Pop some elements from freePool, stored as C-style list for kernel to
  // access
  KernelElem *prevElem = nullptr, *elemList = nullptr;
  for (int i = 0; i < nElems; i++) {
    auto elem = elemPool->pop(ngroups);
    if (prevElem) {
      prevElem->next = elem; // append to existing list
    } else {
      elemList = elem; // head of list
    }
    prevElem = elem;
  }
  return elemList;
}

TEST_F(KernelElemPoolTest, PopReclaim) {
  constexpr int nElems = 5;
  constexpr int poolSize = 1000;
  auto elemPool = std::make_unique<KernelElemPool>(poolSize);
  ASSERT_NE(elemPool, nullptr);

  constexpr int ngroups = 5;

  // Pop some elements from freePool, stored as C-style list for kernel to
  // access
  KernelElem* elemList = popElemList(elemPool, nElems, ngroups);
  auto elem = elemList;
  for (int i = 0; i < nElems; i++) {
    // Expect each has been marked as inuse
    EXPECT_EQ(elem->isFree(), false);
    elem = elem->next;
  }

  // Check current size of freePool
  EXPECT_EQ(elemPool->size(), poolSize - nElems);
  // Check capacity is unchanged
  EXPECT_EQ(elemPool->capacity(), poolSize);

  // Launch kernel to consume these elements, with ngroups gridSize
  dim3 grid = {ngroups, 1, 1};
  dim3 blocks = {1, 1, 1};
  void* args[] = {&elemList};
  CUDACHECK_TEST(cudaLaunchKernel(
      reinterpret_cast<void*>(KElemConsumerKernel), grid, blocks, args, 0, 0));
  CUDACHECK_TEST(cudaStreamSynchronize(0));

  // Reclaim no longer inuse elements, and check pool size has increased back
  elemPool->reclaim();
  EXPECT_EQ(elemPool->size(), poolSize);
}

TEST_F(KernelElemPoolTest, PostRevokeComplete) {
  constexpr int nElems = 10;
  constexpr int poolSize = 1000;
  auto elemPool = std::make_unique<KernelElemPool>(poolSize);
  ASSERT_NE(elemPool, nullptr);

  constexpr int ngroups = 5;
  KernelElem* elemList = popElemList(elemPool, nElems, ngroups);

  // Pass a specific unuseIdx to test case where collective allocates more
  // elements but used only some of them. It requires host to mark it as unuse
  // to be freed; no action need from kernel.
  int unuseIdx = 4;

  // Launch kernel to consume these elements, with ngroups gridSize
  dim3 grid = {ngroups, 1, 1};
  dim3 blocks = {640, 1, 1};
  void* args[] = {&elemList, &unuseIdx};
  CUDACHECK_TEST(cudaLaunchKernel(
      reinterpret_cast<void*>(KElemPostRevokeKernel),
      grid,
      blocks,
      args,
      0,
      0));

  // Now host side posts the elems, revoke only 1 elem in the middle
  constexpr int revokeIdx = 3;
  auto elem = elemList;
  for (int i = 0; i < nElems; i++) {
    if (i == revokeIdx) {
      elem->revoke();
    } else if (i == unuseIdx) {
      elem->unuse();
    } else {
      elem->post();
    }
    elem = elem->next;
  }

  // Before kernel finishes, wait completion for elements that are posted,
  // and free both completed and unused elements.
  elem = elemList;
  for (int i = 0; i < nElems; i++) {
    if (i != revokeIdx) {
      if (i != unuseIdx) {
        elem->wait();
        EXPECT_TRUE(elem->isComplete());
      }

      // Now host side frees the elem
      elem->free();
      EXPECT_TRUE(elem->isFree());
    }
    elem = elem->next;
  }

  CUDACHECK_TEST(cudaStreamSynchronize(0));

  // Check all elems have been freed either by kernel or host
  elem = elemList;
  for (int i = 0; i < nElems; i++) {
    EXPECT_TRUE(elem->isFree());
    elem = elem->next;
  }

  // Reclaim no longer inuse elements, and check pool size has increased back
  elemPool->reclaim();
  EXPECT_EQ(elemPool->size(), poolSize);
}

TEST_F(KernelElemPoolTest, PostWait) {
  constexpr int poolSize = 1000;
  size_t count = 65536;
  auto elemPool = std::make_unique<KernelElemPool>(poolSize);
  ASSERT_NE(elemPool, nullptr);

  constexpr int ngroups = 5;
  KernelElem* elem = popElemList(elemPool, 1, ngroups);
  int *vec1 = nullptr, *vec2 = nullptr;
  CUDACHECK_TEST(
      cudaHostAlloc(&vec1, count * sizeof(int), cudaHostAllocDefault));
  CUDACHECK_TEST(
      cudaHostAlloc(&vec2, count * sizeof(int), cudaHostAllocDefault));

  for (int i = 0; i < count; i++) {
    vec1[i] = i;
    vec2[i] = i;
  }
  // Launch kernel to consume these elements, with ngroups gridSize
  dim3 grid = {ngroups, 1, 1};
  dim3 blocks = {640, 1, 1};
  void* args[] = {&elem, &count, &vec1, &vec2};
  CUDACHECK_TEST(cudaLaunchKernel(
      reinterpret_cast<void*>(KElemPostWaitKernel), grid, blocks, args, 0, 0));

  // Host side posts the elems
  elem->post();

  // Wait for kernel to finish
  elem->wait();

  // Expect kernel has finished sum
  std::vector<int> expVal(count, 0);
  std::vector<int> vec1Val(count, 0);
  for (int i = 0; i < count; i++) {
    expVal[i] = i * 2;
    vec1Val[i] = vec1[i];
  }
  EXPECT_THAT(vec1Val, ::testing::ElementsAreArray(expVal));

  // Host side frees the elem
  elem->free();

  CUDACHECK_TEST(cudaStreamSynchronize(0));

  // Reclaim no longer inuse elements, and check pool size has increased back
  elemPool->reclaim();
  EXPECT_EQ(elemPool->size(), poolSize);
  CUDACHECK_TEST(cudaFreeHost(vec1));
  CUDACHECK_TEST(cudaFreeHost(vec2));
}

TEST_F(KernelElemPoolTest, PostMultiGroupSets) {
  constexpr int poolSize = 1000;
  size_t countPerGroupSet = 65536;
  auto elemPool = std::make_unique<KernelElemPool>(poolSize);
  ASSERT_NE(elemPool, nullptr);

  unsigned int nGroupsSets = 4;
  constexpr unsigned int nGroups = 2;
  // Allocate nGroupsSets number of elements, each used by nGroups thread
  // blocks
  KernelElem* elemList = popElemList(elemPool, nGroupsSets, nGroups);

  size_t totalCount = countPerGroupSet * nGroupsSets;
  int *vec1 = nullptr, *vec2 = nullptr;
  CUDACHECK_TEST(
      cudaHostAlloc(&vec1, totalCount * sizeof(int), cudaHostAllocDefault));
  CUDACHECK_TEST(
      cudaHostAlloc(&vec2, totalCount * sizeof(int), cudaHostAllocDefault));

  for (int i = 0; i < totalCount; i++) {
    vec1[i] = i;
    vec2[i] = i;
  }

  // Launch kernel to consume these elements, with ngroups gridSize
  dim3 grid = {nGroups * nGroupsSets, 1, 1};
  dim3 blocks = {256, 1, 1};
  void* args[] = {&elemList, &countPerGroupSet, &nGroupsSets, &vec1, &vec2};
  CUDACHECK_TEST(cudaLaunchKernel(
      reinterpret_cast<void*>(KElemPostMultiGroupsKernel),
      grid,
      blocks,
      args,
      0,
      0));

  auto elem = elemList;
  for (int i = 0; i < nGroupsSets; i++) {
    // Host side posts the elems
    elem->post();
    elem = elem->next;
  }

  elem = elemList;
  for (int i = 0; i < nGroupsSets; i++) {
    // Wait for kernel to finish
    elem->wait();
    elem = elem->next;
  }

  // Expect kernel has finished sum
  std::vector<int> expVal(totalCount, 0);
  std::vector<int> vec1Val(totalCount, 0);
  for (int i = 0; i < totalCount; i++) {
    expVal[i] = i * 2;
    vec1Val[i] = vec1[i];
  }
  EXPECT_THAT(vec1Val, ::testing::ElementsAreArray(expVal));

  elem = elemList;
  for (int i = 0; i < nGroupsSets; i++) {
    // Host side frees the elem
    elem->free();
    elem = elem->next;
  }

  CUDACHECK_TEST(cudaStreamSynchronize(0));

  // Reclaim no longer inuse elements, and check pool size has increased back
  elemPool->reclaim();
  EXPECT_EQ(elemPool->size(), poolSize);
  CUDACHECK_TEST(cudaFreeHost(vec1));
  CUDACHECK_TEST(cudaFreeHost(vec2));
}

void accessElem(KernelElem* elem) {
  elem->wait();
}

TEST_F(KernelElemPoolAbortTest, PostWithInvalidNGroups) {
  // Make ngroups invalid
  elem->ngroups = -1;
  ASSERT_EXIT(elem->post(), testing::KilledBySignal(SIGABRT), "")
      << "Expect abort when posting with invalid ngroups";
}

TEST_F(KernelElemPoolAbortTest, WaitWithInvalidNGroups) {
  // Make ngroups invalid
  elem->ngroups = -1;
  ASSERT_EXIT(elem->wait(), testing::KilledBySignal(SIGABRT), "")
      << "Expect abort when waiting with invalid ngroups";
}

TEST_F(KernelElemPoolAbortTest, RevokeWithInvalidNGroups) {
  // Make ngroups invalid
  elem->ngroups = -1;
  ASSERT_EXIT(elem->revoke(), testing::KilledBySignal(SIGABRT), "")
      << "Expect abort when revoking with invalid ngroups";
}

TEST_F(KernelElemPoolAbortTest, FreeWithInvalidNGroups) {
  // Make ngroups invalid
  elem->ngroups = -1;
  ASSERT_EXIT(elem->free(), testing::KilledBySignal(SIGABRT), "")
      << "Expect abort when revoking with invalid ngroups";
}

TEST_F(KernelElemPoolAbortTest, CheckCompleteWithInvalidNGroups) {
  // Make ngroups invalid
  elem->ngroups = -1;
  ASSERT_EXIT(elem->isComplete(), testing::KilledBySignal(SIGABRT), "")
      << "Expect abort when revoking with invalid ngroups";
}

TEST_F(KernelElemPoolAbortTest, CheckFreeWithInvalidNGroups) {
  // Make ngroups invalid
  elem->ngroups = -1;
  ASSERT_EXIT(elem->isFree(), testing::KilledBySignal(SIGABRT), "")
      << "Expect abort when revoking with invalid ngroups";
}
