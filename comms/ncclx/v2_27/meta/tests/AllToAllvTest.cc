// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include <cstdint>

#include "checks.h"
#include "comms/ctran/Ctran.h"
#include "comms/testinfra/AlgoTestUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/testinfra/tests_common.cuh"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/CollTrace.h"
#include "meta/hints/GlobalHints.h"

using testinfra::AlgoRAII;

class AllToAllvTest
    : public NcclxBaseTest,
      public ::testing::WithParamInterface<enum NCCL_ALLTOALLV_ALGO> {
 public:
  AllToAllvTest() = default;
  void SetUp() override {
#ifdef TEST_ENABLE_CTRAN
    setenv("NCCL_COLLTRACE", "trace", 0);
#endif

    NcclxBaseTest::SetUp();

    this->comm = createNcclComm(globalRank, numRanks, localRank);

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
  }

  template <typename T>
  void assignChunkValue(T* buf, size_t count, T val) {
    if (count == 0) {
      return;
    }
    std::vector<T> expectedVals(count, val);
    CUDACHECKIGNORE(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyDefault));
  }

  template <typename T>
  int checkChunkValue(T* buf, size_t count, T val) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print failing location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != val) {
        if (errs < 10) {
          std::cout << "[" << this->globalRank << "] observedVals[" << i
                    << "] = " << observedVals[i] << ", expectedVal = " << val
                    << "\n";
        }
        errs++;
      }
    }
    return errs;
  }

  void runReuseSharedBuffer(bool registFlag = false) {
    if (this->globalRank > 1) {
      return;
    }

    // prepare alltoallv arguments
    std::vector<size_t> sendCounts(this->numRanks);
    std::vector<size_t> sendDispls(this->numRanks);
    std::vector<size_t> recvCounts(this->numRanks);
    std::vector<size_t> recvDispls(this->numRanks);
    if (this->globalRank == 0) {
      sendCounts[0] = 60000776;
      sendCounts[1] = 60000172;
      recvCounts[0] = 60000776;
      recvCounts[1] = 60000316;
    } else if (this->globalRank == 1) {
      sendCounts[0] = 60000316;
      sendCounts[1] = 60000564;
      recvCounts[0] = 60000172;
      recvCounts[1] = 60000564;
    }
    sendDispls[0] = 0;
    recvDispls[0] = 0;
    for (int i = 1; i < 2; i++) {
      sendDispls[i] = sendDispls[i - 1] + sendCounts[i - 1];
      recvDispls[i] = recvDispls[i - 1] + recvCounts[i - 1];
    }

    int sendCount = 0;
    int recvCount = 0;
    for (int i = 0; i < 2; i++) {
      sendCount += sendCounts[i];
      recvCount += recvCounts[i];
    }

    // create and register buffers
    int *sendBuf = nullptr, *recvBuf = nullptr;
    void *sendHandle = nullptr, *recvHandle = nullptr;

    CUDACHECK_TEST(cudaMalloc(&sendBuf, sendCount * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, recvCount * sizeof(int)));

    int shared_buf_size = NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE;
    int bufCount = shared_buf_size / sizeof(int);
    assignChunkValue(sendBuf, sendCount, -1);
    if (this->globalRank == 1) {
      for (int i = 0; i < sendCounts[0] / bufCount; i++) {
        assignChunkValue(sendBuf + sendDispls[0] + i * bufCount, bufCount, i);
      }
    }
    assignChunkValue(recvBuf, recvCount, -1);

    if (registFlag) {
      NCCLCHECK_TEST(ncclCommRegister(
          comm, sendBuf, sendCount * sizeof(int), &sendHandle));
      NCCLCHECK_TEST(ncclCommRegister(
          comm, recvBuf, recvCount * sizeof(int), &recvHandle));
    }

    // run alltoallv
    auto res = ncclAllToAllv(
        sendBuf,
        sendCounts.data(),
        sendDispls.data(),
        recvBuf,
        recvCounts.data(),
        recvDispls.data(),
        ncclInt,
        comm,
        stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    if (this->globalRank == 0) {
      for (int i = 0; i < recvCounts[1] / bufCount; i++) {
        int errs = checkChunkValue(
            recvBuf + recvDispls[1] + i * bufCount, bufCount, i);
        EXPECT_EQ(errs, 0) << "failed on rank " << this->globalRank
                           << " and iteration i=" << i;
      }
    }

    if (registFlag) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
      NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    }

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
  }

  void runCanCopy16Mismatch(bool registFlag = false) {
    if (this->numRanks < 4) {
      std::cout << "Need at least four ranks to run this test." << std::endl;
      return;
    }

    if (this->globalRank > 3) {
      return;
    }

    // prepare alltoallv arguments
    std::vector<size_t> sendCounts(this->numRanks);
    std::vector<size_t> sendDispls(this->numRanks);
    std::vector<size_t> recvCounts(this->numRanks);
    std::vector<size_t> recvDispls(this->numRanks);
    if (this->globalRank == 0) {
      sendCounts[0] = 20000775;
      sendCounts[1] = 20000171;
      sendCounts[2] = 20000806;
      sendCounts[3] = 20000365;
      recvCounts[0] = 20000775;
      recvCounts[1] = 20000316;
      recvCounts[2] = 20000575;
      recvCounts[3] = 20000954;
    } else if (this->globalRank == 1) {
      sendCounts[0] = 20000316;
      sendCounts[1] = 20000564;
      sendCounts[2] = 20000432;
      sendCounts[3] = 20000625;
      recvCounts[0] = 20000171;
      recvCounts[1] = 20000564;
      recvCounts[2] = 20000529;
      recvCounts[3] = 20000582;
    } else if (this->globalRank == 2) {
      sendCounts[0] = 20000575;
      sendCounts[1] = 20000529;
      sendCounts[2] = 20000343;
      sendCounts[3] = 20000841;
      recvCounts[0] = 20000806;
      recvCounts[1] = 20000432;
      recvCounts[2] = 20000343;
      recvCounts[3] = 20000763;
    } else if (this->globalRank == 3) {
      sendCounts[0] = 20000954;
      sendCounts[1] = 20000582;
      sendCounts[2] = 20000763;
      sendCounts[3] = 20000142;
      recvCounts[0] = 20000365;
      recvCounts[1] = 20000625;
      recvCounts[2] = 20000841;
      recvCounts[3] = 20000142;
    }
    sendDispls[0] = 0;
    recvDispls[0] = 0;
    for (int i = 1; i < 4; i++) {
      sendDispls[i] = sendDispls[i - 1] + sendCounts[i - 1];
      recvDispls[i] = recvDispls[i - 1] + recvCounts[i - 1];
    }

    int sendCount = 0;
    int recvCount = 0;
    for (int i = 0; i < 4; i++) {
      sendCount += sendCounts[i];
      recvCount += recvCounts[i];
    }

    // create and register buffers
    int *sendBuf = nullptr, *recvBuf = nullptr;
    void *sendHandle = nullptr, *recvHandle = nullptr;

    CUDACHECK_TEST(cudaMalloc(&sendBuf, sendCount * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, recvCount * sizeof(int)));

    assignChunkValue(sendBuf, sendCount, 32);
    assignChunkValue(recvBuf, recvCount, -1);

    if (registFlag) {
      NCCLCHECK_TEST(ncclCommRegister(
          comm, sendBuf, sendCount * sizeof(int), &sendHandle));
      NCCLCHECK_TEST(ncclCommRegister(
          comm, recvBuf, recvCount * sizeof(int), &recvHandle));
    }

    // run alltoallv
    auto res = ncclAllToAllv(
        sendBuf,
        sendCounts.data(),
        sendDispls.data(),
        recvBuf,
        recvCounts.data(),
        recvDispls.data(),
        ncclInt,
        comm,
        stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    for (int r = 0; r < this->numRanks; r++) {
      int expectedVal = 32;
      int errs =
          checkChunkValue(recvBuf + recvDispls[r], recvCounts[r], expectedVal);
      EXPECT_EQ(errs, 0) << "rank " << this->globalRank << " checked chunk "
                         << r << " at " << recvBuf + recvDispls[r] << " with "
                         << errs << " errors";
    }

    if (registFlag) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
      NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    }

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));

#ifdef TEST_ENABLE_CTRAN
    // CollTrace is updated by a separate thread, need wait for it to finish to
    // avoid flaky test
    comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();
    auto dump = comm->ctranComm_->collTrace_->dump();
    EXPECT_EQ(dump.pastColls.size(), 1);

    for (auto& coll : dump.pastColls) {
      if (NCCL_ALLTOALLV_ALGO == NCCL_ALLTOALLV_ALGO::ctran) {
        EXPECT_EQ(coll.dataType, ncclInt);
        EXPECT_EQ(coll.opName, "AllToAllV");
        EXPECT_EQ(coll.codepath, CollTraceColl::Codepath::CTRAN);
      } else {
        EXPECT_EQ(coll.opName, "SendRecv");
        EXPECT_EQ(coll.codepath, CollTraceColl::Codepath::BASELINE);
      }
    }
#endif
  }

  size_t myGetSize(size_t input) {
    return std::max(input, static_cast<size_t>(CTRAN_MIN_REGISTRATION_SIZE));
  }

  void runZeroByteSendRecv(bool registFlag = false) {
    if (this->numRanks < 4) {
      std::cout << "Need at least four ranks to run this test." << std::endl;
      return;
    }

    int kNumBuf = 2;
    int kRepeat = 10;
    // prepare alltoallv arguments
    // send counts:
    // r0 |0, 0, 0, 0|, |0, 0, 0, 0|
    // r1 |0, 9, 0, 0|, |1, 2, 3, 4|
    // r2 |0, 5, 0, 0|, |0, 0, 0, 0|
    // r3 |0, 7, 0, 0|, |0, 0, 0, 0|
    // recv counts:
    // r0 |0, 0, 0, 0|, |0, 1, 0, 0|
    // r1 |0, 9, 5, 7|, |0, 2, 0, 0|
    // r2 |0, 0, 0, 0|, |0, 3, 0, 0|
    // r3 |0, 0, 0, 0|, |0, 4, 0, 0|

    std::vector<std::vector<size_t>> sendCounts(
        kNumBuf, std::vector<size_t>(this->numRanks));
    std::vector<std::vector<size_t>> sendDispls(
        kNumBuf, std::vector<size_t>(this->numRanks));
    std::vector<std::vector<size_t>> recvCounts(
        kNumBuf, std::vector<size_t>(this->numRanks));
    std::vector<std::vector<size_t>> recvDispls(
        kNumBuf, std::vector<size_t>(this->numRanks));
    if (this->globalRank == 0) {
      sendCounts[0][0] = 0;
      sendCounts[0][1] = 0;
      sendCounts[0][2] = 0;
      sendCounts[0][3] = 0;
      sendCounts[1][0] = 0;
      sendCounts[1][1] = 0;
      sendCounts[1][2] = 0;
      sendCounts[1][3] = 0;

      recvCounts[0][0] = 0;
      recvCounts[0][1] = 0;
      recvCounts[0][2] = 0;
      recvCounts[0][3] = 0;
      recvCounts[1][0] = 0;
      recvCounts[1][1] = 1;
      recvCounts[1][2] = 0;
      recvCounts[1][3] = 0;
    } else if (this->globalRank == 1) {
      sendCounts[0][0] = 0;
      sendCounts[0][1] = 9;
      sendCounts[0][2] = 0;
      sendCounts[0][3] = 0;
      sendCounts[1][0] = 1;
      sendCounts[1][1] = 2;
      sendCounts[1][2] = 3;
      sendCounts[1][3] = 4;

      recvCounts[0][0] = 0;
      recvCounts[0][1] = 9;
      recvCounts[0][2] = 5;
      recvCounts[0][3] = 7;
      recvCounts[1][0] = 0;
      recvCounts[1][1] = 2;
      recvCounts[1][2] = 0;
      recvCounts[1][3] = 0;
    } else if (this->globalRank == 2) {
      sendCounts[0][0] = 0;
      sendCounts[0][1] = 5;
      sendCounts[0][2] = 0;
      sendCounts[0][3] = 0;
      sendCounts[1][0] = 0;
      sendCounts[1][1] = 0;
      sendCounts[1][2] = 0;
      sendCounts[1][3] = 0;

      recvCounts[0][0] = 0;
      recvCounts[0][1] = 0;
      recvCounts[0][2] = 0;
      recvCounts[0][3] = 0;
      recvCounts[1][0] = 0;
      recvCounts[1][1] = 3;
      recvCounts[1][2] = 0;
      recvCounts[1][3] = 0;
    } else if (this->globalRank == 3) {
      sendCounts[0][0] = 0;
      sendCounts[0][1] = 7;
      sendCounts[0][2] = 0;
      sendCounts[0][3] = 0;
      sendCounts[1][0] = 0;
      sendCounts[1][1] = 0;
      sendCounts[1][2] = 0;
      sendCounts[1][3] = 0;

      recvCounts[0][0] = 0;
      recvCounts[0][1] = 0;
      recvCounts[0][2] = 0;
      recvCounts[0][3] = 0;
      recvCounts[1][0] = 0;
      recvCounts[1][1] = 4;
      recvCounts[1][2] = 0;
      recvCounts[1][3] = 0;
    }
    sendDispls[0][0] = 0;
    sendDispls[1][0] = 0;
    recvDispls[0][0] = 0;
    recvDispls[1][0] = 0;
    for (int i = 1; i < 4; i++) {
      sendDispls[0][i] = sendDispls[0][i - 1] + sendCounts[0][i - 1];
      sendDispls[1][i] = sendDispls[1][i - 1] + sendCounts[1][i - 1];
      recvDispls[0][i] = recvDispls[0][i - 1] + recvCounts[0][i - 1];
      recvDispls[1][i] = recvDispls[1][i - 1] + recvCounts[1][i - 1];
    }

    std::vector<int> sendCount(kNumBuf, 0);
    std::vector<int> recvCount(kNumBuf, 0);
    for (int i = 0; i < 4; i++) {
      sendCount[0] += sendCounts[0][i];
      sendCount[1] += sendCounts[1][i];
      recvCount[0] += recvCounts[0][i];
      recvCount[1] += recvCounts[1][i];
    }

    // create and register buffers
    std::vector<int64_t*> sendBuf(kNumBuf, nullptr);
    std::vector<int64_t*> recvBuf(kRepeat, nullptr);
    std::vector<void*> sendHandle(kNumBuf, nullptr);
    std::vector<void*> recvHandle(kRepeat, nullptr);

    CUDACHECK_TEST(
        cudaMalloc(&sendBuf[0], myGetSize(sendCount[0] * sizeof(int64_t))));
    CUDACHECK_TEST(
        cudaMalloc(&sendBuf[1], myGetSize(sendCount[1] * sizeof(int64_t))));
    for (int i = 0; i < kRepeat; i++) {
      CUDACHECK_TEST(cudaMalloc(
          &recvBuf[i], myGetSize(recvCount[i % kNumBuf] * sizeof(int64_t))));
    }

    int64_t expectedVal = 32;
    assignChunkValue(sendBuf[0], sendCount[0], expectedVal);
    assignChunkValue(sendBuf[1], sendCount[1], expectedVal);
    for (int i = 0; i < kRepeat; i++) {
      assignChunkValue(
          recvBuf[i], recvCount[i % kNumBuf], static_cast<int64_t>(-1));
    }

    if (registFlag) {
      NCCLCHECK_TEST(ncclCommRegister(
          comm,
          sendBuf[0],
          myGetSize(sendCount[0] * sizeof(int64_t)),
          &sendHandle[0]));
      NCCLCHECK_TEST(ncclCommRegister(
          comm,
          sendBuf[1],
          myGetSize(sendCount[1] * sizeof(int64_t)),
          &sendHandle[1]));
      for (int i = 0; i < kRepeat; i++) {
        NCCLCHECK_TEST(ncclCommRegister(
            comm,
            recvBuf[i],
            myGetSize(recvCount[i % kNumBuf] * sizeof(int64_t)),
            &recvHandle[i]));
      }
    }

    // run alltoallv
    for (int i = 0; i < kRepeat; i++) {
      auto bufIdx = i % kNumBuf;
      auto res = ncclAllToAllv(
          sendBuf[bufIdx],
          sendCounts[bufIdx].data(),
          sendDispls[bufIdx].data(),
          recvBuf[i],
          recvCounts[bufIdx].data(),
          recvDispls[bufIdx].data(),
          ncclInt64,
          comm,
          stream);
      ASSERT_EQ(res, ncclSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    if (this->globalRank > 3) {
      return;
    }

    for (int i = 0; i < kRepeat; i++) {
      auto bufIdx = i % kNumBuf;
      for (int r = 0; r < this->numRanks; r++) {
        expectedVal = 32;
        int errs = checkChunkValue(
            recvBuf[i] + recvDispls[bufIdx][r],
            recvCounts[bufIdx][r],
            expectedVal);
        EXPECT_EQ(errs, 0) << "rank " << this->globalRank << " checked chunk "
                           << r << " at "
                           << recvBuf[bufIdx] + recvDispls[bufIdx][r]
                           << " with " << errs << " errors";
      }
    }

    if (registFlag) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle[0]));
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle[1]));
      for (int i = 0; i < kRepeat; i++) {
        NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle[i]));
      }
    }

    for (int i = 0; i < kNumBuf; i++) {
      CUDACHECK_TEST(cudaFree(sendBuf[i]));
    }
    for (int i = 0; i < kRepeat; i++) {
      CUDACHECK_TEST(cudaFree(recvBuf[i]));
    }
  }

  template <typename T>
  void run(bool registFlag = false) {
    // create and register buffers
    constexpr int count = 1048576;
    T *sendBuf = nullptr, *recvBuf = nullptr;
    void *sendHandle = nullptr, *recvHandle = nullptr;

    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * this->numRanks * sizeof(T)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * this->numRanks * sizeof(T)));

    for (int r = 0; r < this->numRanks; r++) {
      T expectedVal = this->globalRank * 10 + r + 1;
      assignChunkValue(sendBuf + r * count, count, expectedVal);
      assignChunkValue(recvBuf + r * count, count, (T)0);
    }

    if (registFlag) {
      NCCLCHECK_TEST(ncclCommRegister(
          comm, sendBuf, count * this->numRanks * sizeof(T), &sendHandle));
      NCCLCHECK_TEST(ncclCommRegister(
          comm, recvBuf, count * this->numRanks * sizeof(T), &recvHandle));
    }

    // prepare alltoallv arguments
    std::vector<size_t> sendCounts(this->numRanks);
    std::vector<size_t> sendDispls(this->numRanks);
    std::vector<size_t> recvCounts(this->numRanks);
    std::vector<size_t> recvDispls(this->numRanks);
    for (int r = 0; r < this->numRanks; r++) {
      sendCounts[r] = r % 2 ? count : count / 2;
      sendDispls[r] = r * count;
      recvCounts[r] = this->globalRank % 2 ? count : count / 2;
      recvDispls[r] = r * count;
    }

    // run alltoallv
    auto res = ncclAllToAllv(
        sendBuf,
        sendCounts.data(),
        sendDispls.data(),
        recvBuf,
        recvCounts.data(),
        recvDispls.data(),
        getNcclDataType<T>(),
        comm,
        stream);
    ASSERT_EQ(res, commSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    for (int r = 0; r < this->numRanks; r++) {
      T expectedVal = r * 10 + this->globalRank + 1;
      int errs =
          checkChunkValue(recvBuf + recvDispls[r], recvCounts[r], expectedVal);
      EXPECT_EQ(errs, 0) << "rank " << this->globalRank << " checked chunk "
                         << r << " at " << recvBuf + recvDispls[r] << " with "
                         << errs << " errors";
    }

    if (registFlag) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
      NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    }

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));

    // FIXME: Temp disable because causing test to segfault
    /*
    #ifdef TEST_ENABLE_CTRAN
        // CollTrace is updated by a separate thread, need wait for it to finish
    to
        // avoid flaky test
        comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();
        auto dump = comm->ctranComm_->collTrace_->dump();
        EXPECT_EQ(dump.pastColls.size(), 1);

        for (auto& coll : dump.pastColls) {
          if (NCCL_ALLTOALLV_ALGO == NCCL_ALLTOALLV_ALGO::ctran) {
            EXPECT_EQ(coll.dataType, getNcclDataType<T>());
            EXPECT_EQ(coll.opName, "AllToAllV");
            EXPECT_EQ(coll.codepath, CollTraceColl::Codepath::CTRAN);
          } else {
            EXPECT_EQ(coll.opName, "SendRecv");
            EXPECT_EQ(coll.codepath, CollTraceColl::Codepath::BASELINE);
          }
        }
    #endif
    */
  }
  template <typename T>
  void runSparseAlltoallv(bool registFlag = false) {
    // even ranks only send data, odd ranks only recv data
    constexpr int count = 1048576;
    T *sendBuf = nullptr, *recvBuf = nullptr;
    void *sendHandle = nullptr, *recvHandle = nullptr;

    std::vector<size_t> sendCounts(this->numRanks);
    std::vector<size_t> sendDispls(this->numRanks);
    std::vector<size_t> recvCounts(this->numRanks);
    std::vector<size_t> recvDispls(this->numRanks);

    if (this->globalRank % 2 == 0) {
      CUDACHECK_TEST(
          cudaMalloc(&sendBuf, count * (this->numRanks / 2) * sizeof(T)));

      int rankOffset = 0;
      for (int r = 0; r < this->numRanks; r++) {
        if (r % 2 != 0) {
          // send data to odd ranks
          T expectedVal = this->globalRank * 10 + r + 1;
          assignChunkValue(sendBuf + rankOffset * count, count, expectedVal);
          sendCounts[r] = count;
          sendDispls[r] = (rankOffset++) * count;
        } else {
          sendCounts[r] = 0;
          sendDispls[r] = 0;
        }
        recvCounts[r] = 0;
        recvDispls[r] = 0;
      }

      if (registFlag) {
        NCCLCHECK_TEST(ncclCommRegister(
            comm,
            sendBuf,
            count * (this->numRanks / 2) * sizeof(T),
            &sendHandle));
      }

    } else {
      CUDACHECK_TEST(
          cudaMalloc(&recvBuf, count * (this->numRanks / 2) * sizeof(T)));
      assignChunkValue(recvBuf, count * (this->numRanks / 2), (T)0);

      int rankOffset = 0;
      for (int r = 0; r < this->numRanks; r++) {
        if (r % 2 == 0) {
          // receive data from even ranks
          recvCounts[r] = count;
          recvDispls[r] = (rankOffset++) * count;
        } else {
          recvCounts[r] = 0;
          recvDispls[r] = 0;
        }
        sendCounts[r] = 0;
        sendDispls[r] = 0;
      }

      if (registFlag) {
        NCCLCHECK_TEST(ncclCommRegister(
            comm,
            recvBuf,
            count * (this->numRanks / 2) * sizeof(T),
            &recvHandle));
      }
    }

    // run alltoallv
    auto res = ncclAllToAllv(
        sendBuf,
        sendCounts.data(),
        sendDispls.data(),
        recvBuf,
        recvCounts.data(),
        recvDispls.data(),
        getNcclDataType<T>(),
        comm,
        stream);
    ASSERT_EQ(res, commSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    // odd ranks check received data
    if (this->globalRank % 2 != 0) {
      for (int r = 0; r < this->numRanks; r += 2) {
        T expectedVal = r * 10 + this->globalRank + 1;
        int errs = checkChunkValue(
            recvBuf + recvDispls[r], recvCounts[r], expectedVal);
        EXPECT_EQ(errs, 0) << "rank " << this->globalRank << " checked chunk "
                           << r << " at " << recvBuf + recvDispls[r] << " with "
                           << errs << " errors";
      }
    }

    if (sendHandle) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
    }
    if (recvHandle) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    }

    if (sendBuf) {
      CUDACHECK_TEST(cudaFree(sendBuf));
    }
    if (recvBuf) {
      CUDACHECK_TEST(cudaFree(recvBuf));
    }
  }

 protected:
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_F(AllToAllvTest, OutOfPlaceInt) {
  run<int>();
}
TEST_F(AllToAllvTest, OutOfPlaceUint8) {
  run<uint8_t>();
}
TEST_F(AllToAllvTest, OutOfPlaceFloat) {
  run<float>();
}

#ifdef TEST_ENABLE_CTRAN
TEST_F(AllToAllvTest, CtranInt) {
  auto envGuard = AlgoRAII(NCCL_ALLTOALLV_ALGO, NCCL_ALLTOALLV_ALGO::ctran);
  run<int>();
}

TEST_F(AllToAllvTest, CtranUint8) {
  auto envGuard = AlgoRAII(NCCL_ALLTOALLV_ALGO, NCCL_ALLTOALLV_ALGO::ctran);
  run<uint8_t>();
}
#endif

TEST_P(AllToAllvTest, CanCopy16Mismatch) {
  auto envGuard = AlgoRAII(NCCL_ALLTOALLV_ALGO, GetParam());
  runCanCopy16Mismatch();
}

#ifdef TEST_ENABLE_CTRAN
TEST_P(AllToAllvTest, ZeroByteSendRecv) {
  auto envGuard = AlgoRAII(NCCL_ALLTOALLV_ALGO, GetParam());
  runZeroByteSendRecv(GetParam() == NCCL_ALLTOALLV_ALGO::ctran);
}
#endif

TEST_P(AllToAllvTest, ReuseSharedBuffer) {
  auto envGuard = AlgoRAII(NCCL_ALLTOALLV_ALGO, GetParam());
  runReuseSharedBuffer();
}

TEST_P(AllToAllvTest, SparseAlltoallvInt) {
  auto envGuard = AlgoRAII(NCCL_ALLTOALLV_ALGO, GetParam());
  runSparseAlltoallv<int>(true /*registFlag*/);
}

TEST_P(AllToAllvTest, SparseAlltoallvUint8) {
  auto envGuard = AlgoRAII(NCCL_ALLTOALLV_ALGO, GetParam());
  runSparseAlltoallv<uint8_t>(true /*registFlag*/);
}

TEST_F(AllToAllvTest, InvalidSendbuf) {
  constexpr int count = 1048576;
  int* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, count * this->numRanks * sizeof(int)));

  // prepare alltoallv arguments
  std::vector<size_t> sendCounts(this->numRanks, count);
  std::vector<size_t> sendDispls(this->numRanks, 0);
  std::vector<size_t> recvCounts(this->numRanks, count);
  std::vector<size_t> recvDispls(this->numRanks, 0);

  // run alltoallv
  auto res = ncclAllToAllv(
      nullptr,
      sendCounts.data(),
      sendDispls.data(),
      buf,
      recvCounts.data(),
      recvDispls.data(),
      ncclInt,
      comm,
      stream);
  ASSERT_EQ(res, ncclInvalidArgument);
  CUDACHECK_TEST(cudaFree(buf));
}

TEST_F(AllToAllvTest, InvalidRecvbuf) {
  constexpr int count = 1048576;
  int* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, count * this->numRanks * sizeof(int)));

  // prepare alltoallv arguments
  std::vector<size_t> sendCounts(this->numRanks, count);
  std::vector<size_t> sendDispls(this->numRanks, 0);
  std::vector<size_t> recvCounts(this->numRanks, count);
  std::vector<size_t> recvDispls(this->numRanks, 0);

  // run alltoallv
  auto res = ncclAllToAllv(
      buf,
      sendCounts.data(),
      sendDispls.data(),
      nullptr,
      recvCounts.data(),
      recvDispls.data(),
      ncclInt,
      comm,
      stream);
  ASSERT_EQ(res, ncclInvalidArgument);
  CUDACHECK_TEST(cudaFree(buf));
}

TEST_F(AllToAllvTest, InvalidInPlace) {
  constexpr int count = 1048576;
  int* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, count * this->numRanks * sizeof(int)));

  // prepare alltoallv arguments
  std::vector<size_t> sendCounts(this->numRanks, count);
  std::vector<size_t> sendDispls(this->numRanks, 0);
  std::vector<size_t> recvCounts(this->numRanks, count);
  std::vector<size_t> recvDispls(this->numRanks, 0);

  // run alltoallv
  auto res = ncclAllToAllv(
      buf,
      sendCounts.data(),
      sendDispls.data(),
      buf,
      recvCounts.data(),
      recvDispls.data(),
      ncclInt,
      comm,
      stream);
  ASSERT_EQ(res, ncclInvalidArgument);
  CUDACHECK_TEST(cudaFree(buf));
}

TEST_F(AllToAllvTest, ValidInPlace) {
  // prepare alltoallv arguments
  std::vector<size_t> sendCounts(this->numRanks, 0);
  std::vector<size_t> sendDispls(this->numRanks, 0);
  std::vector<size_t> recvCounts(this->numRanks, 0);
  std::vector<size_t> recvDispls(this->numRanks, 0);

  // run alltoallv
  auto res = ncclAllToAllv(
      nullptr,
      sendCounts.data(),
      sendDispls.data(),
      nullptr,
      recvCounts.data(),
      recvDispls.data(),
      ncclInt,
      comm,
      stream);
  ASSERT_EQ(res, ncclSuccess);
}

TEST_F(AllToAllvTest, AllToAllvWithHintOverride) {
  AlgoRAII algoEnv(NCCL_ALLTOALLV_ALGO, NCCL_ALLTOALLV_ALGO::orig);

  ASSERT_TRUE(ncclx::setGlobalHint("algo_alltoallv", "ctran"));
  run<int>();

  ASSERT_TRUE(ncclx::resetGlobalHint("algo_alltoallv"));
  run<int>();
}

INSTANTIATE_TEST_SUITE_P(
    AllToAllvTestWithParamInstantiation,
    AllToAllvTest,
    ::testing::Values(
        NCCL_ALLTOALLV_ALGO::orig
#ifdef TEST_ENABLE_CTRAN
        ,
        NCCL_ALLTOALLV_ALGO::ctran
#endif
        ));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
