// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <nccl.h>
#include <stdlib.h>
#include <cstdio>

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "CtranUtUtils.h"
#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestsDistUtils.h"

// #define VERBOSE

namespace {
enum ArgMemType {
  kArgInHostRegisteredMem,
  kArgInDeviceMem,
};
}

class CtranAllToAllvDedupTest : public CtranDistBaseTest {
 public:
  CtranAllToAllvDedupTest() = default;
  void SetUp() override {
    CtranDistBaseTest::SetUp();

    comm_ = commWorld;
    std::cout << "set comm_ = commWorld " << commWorld << std::endl;
  }

  void TearDown() override {
    CtranDistBaseTest::TearDown();
  }

  void generateDistRandomExpValue() {
    if (globalRank == 0) {
      // expectedVal_ = rand();
      expectedVal_ = 100000000;
    }

    // Use MPI to broadcast the value to all ranks
    if (tcpStoreServer.get() == nullptr) {
      MPI_Bcast(&expectedVal_, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return;
    }

    // Use TCPStore to broadcast the value to all ranks
    if (globalRank == 0) {
      tcpStoreServer->set(
          "expected_val", {static_cast<unsigned char>(expectedVal_)});
    } else {
      tcpStoreServer->wait({"expected_val"});
      expectedVal_ = tcpStoreServer->get("expected_val").at(0);
    }
  }

  void setTestParams(
      int totalNumSendBlocks,
      size_t blockCount,
      int blockNumRecvBuckets,
      int numRecvBuckets) {
    pArgs_.totalNumSendBlocks = totalNumSendBlocks;
    pArgs_.blockCount = blockCount;
    pArgs_.blockNumRecvBuckets = blockNumRecvBuckets;
    pArgs_.numRecvBuckets = numRecvBuckets;
  }

  template <typename T>
  T* createDeviceArg(const std::vector<T>& argVec, bool regist = false) {
    void* devArg = nullptr;
    size_t nbytes = sizeof(T) * argVec.size();
    nbytes = std::max(nbytes, (size_t)CTRAN_MIN_REGISTRATION_SIZE);

    if (argMemType_ == kArgInHostRegisteredMem) {
      CUDACHECK_TEST(cudaHostAlloc(&devArg, nbytes, cudaHostAllocDefault));
      memset(devArg, 0, nbytes);
    } else {
      NCCLCHECK_TEST(ncclMemAlloc(&devArg, nbytes));
      CUDACHECK_TEST(cudaMemset(devArg, 0, nbytes));
    }

    void* handle = nullptr;
    if (regist) {
      NCCLCHECK_TEST(ncclCommRegister(comm_, devArg, nbytes, &handle));
    }

    // store argPtr to release at the end of test
    deviceArgs_.push_back(std::make_pair(devArg, regist ? handle : nullptr));
    return reinterpret_cast<T*>(devArg);
  }

  template <typename T>
  void updateDeviceArg(
      T* devArg,
      const std::vector<T>& argVec,
      cudaStream_t stream) {
    CUDACHECK_TEST(cudaMemcpyAsync(
        devArg,
        argVec.data(),
        sizeof(T) * argVec.size(),
        cudaMemcpyDefault,
        stream));
  }

  template <typename T>
  void
  loadDeviceArg(std::vector<T>& argVec, const T* devArg, cudaStream_t stream) {
    CUDACHECK_TEST(cudaMemcpyAsync(
        argVec.data(),
        devArg,
        sizeof(T) * argVec.size(),
        cudaMemcpyDefault,
        stream));
  }

  void releaseDeviceArgs() {
    for (auto arg : deviceArgs_) {
      auto ptr = arg.first;
      auto handle = arg.second;
      if (handle) {
        NCCLCHECK_TEST(ncclCommDeregister(comm_, handle));
      }

      if (argMemType_ == kArgInHostRegisteredMem) {
        CUDACHECK_TEST(cudaFreeHost(ptr));
      } else {
        NCCLCHECK_TEST(ncclMemFree(ptr));
      }
    }
    deviceArgs_.clear();
  }

  void getExpPrepareOutputs(
      const std::vector<std::vector<int>>& allRankBlockRecvBuckets,
      const int iter,
      std::vector<size_t>& numSendBlocks,
      std::vector<size_t>& numRecvBlocks,
      std::vector<size_t>& recvOffsets,
      std::vector<size_t>& numForwardBlocks,
      size_t& totalNumRecvBlocks,
      std::vector<int>& xnodeInputSplits,
      std::vector<int>& xnodeOutputSplits,
      std::vector<int>& localInputSplits,
      std::vector<int>& localOutputSplits,
      CtranPersistentRequest* request) {
    const auto statex = comm_->ctranComm_->statex_.get();
    const int nRanks = statex->nRanks();
    const int myNode = statex->node();
    const int nNodes = statex->nNodes();
    const int nLocalRanks = statex->nLocalRanks();
    const int myLocalRank = statex->localRank();
    const int myRank = statex->rank();

    numSendBlocks.resize(nRanks, 0);
    numRecvBlocks.resize(nRanks * pArgs_.numRecvBuckets, 0);
    recvOffsets.resize(nRanks * pArgs_.numRecvBuckets, 0);
    numForwardBlocks.resize(nRanks, 0);
    totalNumRecvBlocks = 0;

    const auto blockNumRecvBuckets = pArgs_.blockNumRecvBuckets;

    for (int sendRank = 0; sendRank < nRanks; sendRank++) {
      const auto& blockRecvBuckets = allRankBlockRecvBuckets[sendRank];
      int sendLocalRank = statex->localRank(sendRank);
      int sendNode = statex->node(sendRank);
      for (int i = 0; i < pArgs_.totalNumSendBlocks; i++) {
        bool foundSendRank = false;
        for (int j = 0; j < pArgs_.blockNumRecvBuckets; j++) {
          int recvBucket = blockRecvBuckets[i * blockNumRecvBuckets + j];
          int blockRecvRank = recvBucket / pArgs_.numRecvBuckets;
          int rankRecvBucket = recvBucket % pArgs_.numRecvBuckets;
          int recvNode = statex->node(blockRecvRank);
          if (blockRecvRank == myRank) {
            numRecvBlocks[sendRank * pArgs_.numRecvBuckets + rankRecvBucket]++;
            totalNumRecvBlocks++;
          }
          if (sendRank == myRank) {
            numSendBlocks[blockRecvRank]++;
          }

          // Record number of forwarding blocks going through this rank.
          if (sendLocalRank == myLocalRank) {
            // blocks to be forwarded from send rank in the same rail
            if (sendNode != myNode && recvNode == myNode && !foundSendRank) {
              numForwardBlocks[sendRank]++;
              // only count once if the same block is forwarded to multiple
              // local recvRanks
              foundSendRank = true;
            }
            // blocks to be forwarded to recv ranks in the same node
            if (sendNode != myNode && recvNode == myNode) {
              // blocks to be forwarded to local recvRanks
              numForwardBlocks[blockRecvRank]++;
            }
          }
        }
      }
    }
    int sum = 0;
    for (int bucket = 0; bucket < pArgs_.numRecvBuckets; bucket++) {
      for (int localRank = 0; localRank < nLocalRanks; localRank++) {
        for (int node = 0; node < nNodes; node++) {
          auto rank = node * nLocalRanks + localRank;
          recvOffsets[rank * pArgs_.numRecvBuckets + bucket] = sum;
          sum += numRecvBlocks[rank * pArgs_.numRecvBuckets + bucket] *
              pArgs_.blockCount;
        }
      }
    }

    std::vector<std::vector<int>> xnodeSplits(
        nNodes, std::vector<int>(nNodes, 0));
    for (int node = 0; node < nNodes; node++) {
      const auto sendRank = statex->localRankToRank(myLocalRank, node);
      const auto& blockRecvBuckets = allRankBlockRecvBuckets[sendRank];
      auto& xnodeSplitsRow = xnodeSplits[node];
      for (int i = 0; i < pArgs_.totalNumSendBlocks; i++) {
        std::vector<bool> sentToNode(nNodes, false);
        for (int j = 0; j < pArgs_.blockNumRecvBuckets; j++) {
          int recvBucket = blockRecvBuckets[i * blockNumRecvBuckets + j];
          int blockRecvRank = recvBucket / pArgs_.numRecvBuckets;
          int recvNode = statex->node(blockRecvRank);
          if (!sentToNode[recvNode]) {
            xnodeSplitsRow[recvNode]++;
            sentToNode[recvNode] = true;
          }
        }
      }
    }
    for (int i = 0; i < nNodes; i++) {
      xnodeInputSplits[i * nLocalRanks + myLocalRank] = xnodeSplits[myNode][i];
      xnodeOutputSplits[i * nLocalRanks + myLocalRank] = xnodeSplits[i][myNode];
    }

    std::vector<std::vector<int>> localSplits(
        nLocalRanks, std::vector<int>(nLocalRanks, 0));
    for (int localRank = 0; localRank < nLocalRanks; localRank++) {
      auto& localSplitsRow = localSplits[localRank];
      for (int node = 0; node < nNodes; node++) {
        const auto sendRank = statex->localRankToRank(localRank, node);
        const auto& blockRecvBuckets = allRankBlockRecvBuckets[sendRank];
        for (int i = 0; i < pArgs_.totalNumSendBlocks; i++) {
          std::vector<bool> sentToLocalRank(nLocalRanks, false);
          for (int j = 0; j < pArgs_.blockNumRecvBuckets; j++) {
            int recvBucket = blockRecvBuckets[i * blockNumRecvBuckets + j];
            int blockRecvRank = recvBucket / pArgs_.numRecvBuckets;
            int recvNode = statex->node(blockRecvRank);
            if (recvNode == myNode) {
              int localBlockRecvRank = statex->localRank(blockRecvRank);
              if (!sentToLocalRank[localBlockRecvRank]) {
                localSplitsRow[localBlockRecvRank]++;
                sentToLocalRank[localBlockRecvRank] = true;
              }
            }
          }
        }
      }
    }

    for (int i = 0; i < nLocalRanks; i++) {
      localInputSplits[myNode * nLocalRanks + i] = localSplits[myLocalRank][i];
      localOutputSplits[myNode * nLocalRanks + i] = localSplits[i][myLocalRank];
    }

#ifdef VERBOSE
    std::cout << "TEST rank " << myRank << " iter " << iter
              << " expected totalNumRecvBlocks: " << totalNumRecvBlocks
              << std::endl;
    std::cout << "TEST rank " << myRank << " iter " << iter
              << " expected numSendBlocks: " << folly::join(",", numSendBlocks)
              << std::endl;
    std::cout << "TEST rank " << myRank << " iter " << iter
              << " expected numRecvBlocks: " << folly::join(",", numRecvBlocks)
              << std::endl;
    std::cout << "TEST rank " << myRank << " iter " << iter
              << " expected recvOffsets: " << folly::join(",", recvOffsets)
              << std::endl;
    std::cout << "TEST rank " << myRank << " iter " << iter
              << " expected numForwardBlocks: "
              << folly::join(",", numForwardBlocks) << std::endl;
#endif
  }

  void* createDataBuf(size_t nbytes, void** handle = nullptr) {
    void* buf = nullptr;
    // Allocate data buffer, and assign different value for each send chunk
    CUDACHECK_TEST(cudaMalloc(&buf, nbytes));

    if (buf) {
      CUDACHECK_TEST(cudaMemset(buf, -1, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());
      if (handle) {
        NCCLCHECK_TEST(ncclCommRegister(comm_, buf, nbytes, handle));
      }
    }
    dataBufHdls_.push_back(std::make_pair(buf, handle ? *handle : nullptr));
    return buf;
  }

  void releaseDataBufs() {
    for (auto& [buf, handle] : dataBufHdls_) {
      if (handle) {
        NCCLCHECK_TEST(ncclCommDeregister(comm_, handle));
      }
      CUDACHECK_TEST(cudaFree(buf));
    }
    dataBufHdls_.clear();
  }

  void getBlockRecvBuckets(
      const int sendRank,
      const int iter,
      std::vector<int>& blockRecvBuckets) const {
    const int nRanks = comm_->ctranComm_->statex_->nRanks();
    const int blockNumRecvBuckets = pArgs_.blockNumRecvBuckets;

    for (int i = 0; i < pArgs_.totalNumSendBlocks; i++) {
      // each rank sends to 2 peers from itself, and shift by one for each
      // block. E.g., rank 0: block0->[0,1], block1->[1,2], block2->[2,3],
      // blcok3->[3,4]...). For each iteration, shift the starting block by
      // one.
      int recvRankStart =
          (sendRank + iter + i) % (nRanks * pArgs_.numRecvBuckets);
      for (int j = 0; j < blockNumRecvBuckets; j++) {
        int recvBucket = (recvRankStart + j) % (nRanks * pArgs_.numRecvBuckets);
        if (skippedRecvBuckets_.contains(recvBucket)) {
          // ensure the replaced bucket is not already contained
          recvBucket = (recvRankStart + blockNumRecvBuckets) %
              (nRanks * pArgs_.numRecvBuckets);
        }
        blockRecvBuckets[i * blockNumRecvBuckets + j] = recvBucket;
      }
    }
  }

  std::vector<size_t> getSendBlockIdx(
      const int sendRank,
      const std::vector<int>& blockRecvBuckets,
      const int recvBucket) const {
    std::vector<size_t> sendBlockIdx;
    sendBlockIdx.reserve(pArgs_.totalNumSendBlocks);
    for (int i = 0; i < pArgs_.totalNumSendBlocks; i++) {
      for (int j = 0; j < pArgs_.blockNumRecvBuckets; j++) {
        if (recvBucket ==
            blockRecvBuckets[i * pArgs_.blockNumRecvBuckets + j]) {
          sendBlockIdx.push_back(i);
          // a block should be sent to the receiver only once
          continue;
        }
      }
    }
#ifdef VERBOSE
    std::cout << "TEST recvBucket " << recvBucket
              << " get blockIdx of sendRank " << sendRank
              << " sendBlockIdx: " << folly::join(",", sendBlockIdx)
              << std::endl;
#endif
    return sendBlockIdx;
  }

  template <typename T>
  void assignChunkValue(
      T* buf,
      size_t count,
      const std::vector<T>& expectedVals,
      cudaStream_t stream) {
    FB_CUDACHECKIGNORE(cudaMemcpyAsync(
        buf,
        expectedVals.data(),
        count * sizeof(T),
        cudaMemcpyDefault,
        stream));
  }

  template <typename T>
  void assignChunkValue(T* buf, size_t count, T val, cudaStream_t stream) {
    std::vector<T> expectedVals(count, val);
    FB_CUDACHECKIGNORE(cudaMemcpyAsync(
        buf,
        expectedVals.data(),
        count * sizeof(T),
        cudaMemcpyDefault,
        stream));
  }

  template <typename T>
  std::string valsToStr(
      const std::vector<T>& vals,
      const std::string delim = " ",
      int numToPrint = 2) const {
    std::string more;
    if (vals.size() < numToPrint) {
      numToPrint = vals.size();
    } else if (vals.size() > numToPrint) {
      // more = "...";
    }
    std::vector<T> sub(vals.begin(), vals.begin() + numToPrint);
    return folly::join(delim, sub) + more;
  }

  template <typename T>
  int checkChunkValue(T* buf, size_t count, T val) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print first 10 failing
    // location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != val) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d] = %d, expectedVal_ = %d\n",
              globalRank,
              i,
              int(observedVals[i]),
              int(val));
        }
        errs++;
      }
    }
    return errs;
  }

  template <typename T>
  void assignSendBuffVals(
      int* sendBuff,
      const int iter,
      std::vector<int>& blockRecvBuckets) {
    const size_t totalNumSendBlocks = pArgs_.totalNumSendBlocks;
    const size_t blockCount = pArgs_.blockCount;

    std::vector<std::string> assignedValStrs;
    assignedValStrs.reserve(totalNumSendBlocks);
    for (int i = 0; i < totalNumSendBlocks; ++i) {
      std::vector<T> assignedVals(
          blockCount, T(expectedVal_ + globalRank * 100000 + i));
      assignChunkValue<T>(
          sendBuff + blockCount * i, blockCount, assignedVals, stream_);
      std::vector<int> recvBuckets;
      for (int j = 0; j < pArgs_.blockNumRecvBuckets; j++) {
        recvBuckets.push_back(
            blockRecvBuckets[i * pArgs_.blockNumRecvBuckets + j]);
      }
      assignedValStrs.push_back(
          valsToStr(assignedVals) + " [" + folly::join(",", recvBuckets) + "]");
    }

#ifdef VERBOSE
    const int myRank = comm_->ctranComm_->statex_->rank();
    std::cout << fmt::format(
                     "TEST rank {} iter {} sendBuff 0x{:x} : {}",
                     myRank,
                     iter,
                     (uintptr_t)sendBuff,
                     folly::join(", ", assignedValStrs))
              << std::endl;
#endif
  }

  void setupDeviceArgs(
      int nRanks,
      int*& blockRecvBuckets,
      size_t*& numSendBlocks,
      size_t*& numRecvBlocks,
      size_t*& recvOffsets,
      size_t*& numForwardBlocks,
      size_t*& totalNumRecvBlocks,
      int*& sendIdx,
      int*& fwdIdx,
      int*& recvIdx,
      char*& barrierByte,
      bool allRankBlockRecvBucketsOverride = false) {
#ifdef VERBOSE
    std::cout << "TEST init: rank " << myRank_
              << " totalNumSendBlocks: " << pArgs_.totalNumSendBlocks
              << ", blockCount: " << pArgs_.blockCount
              << ", blockNumRecvBuckets: " << pArgs_.blockNumRecvBuckets
              << std::endl;
#endif

    argMemType_ = kArgInDeviceMem;
    blockRecvBuckets_.resize(
        pArgs_.totalNumSendBlocks * pArgs_.blockNumRecvBuckets);
    std::fill(blockRecvBuckets_.begin(), blockRecvBuckets_.end(), 0);
    numSendBlocks_.resize(nRanks);
    std::fill(numSendBlocks_.begin(), numSendBlocks_.end(), 0);
    numRecvBlocks_.resize(nRanks * pArgs_.numRecvBuckets);
    std::fill(numRecvBlocks_.begin(), numRecvBlocks_.end(), 0);
    recvOffsets_.resize(nRanks * pArgs_.numRecvBuckets);
    std::fill(recvOffsets_.begin(), recvOffsets_.end(), 0);
    numForwardBlocks_.resize(nRanks);
    std::fill(numForwardBlocks_.begin(), numForwardBlocks_.end(), 0);
    totalNumRecvBlocks_.resize(1);
    std::fill(totalNumRecvBlocks_.begin(), totalNumRecvBlocks_.end(), 0);
    sendIdx_.resize(
        pArgs_.totalNumSendBlocks * comm_->ctranComm_->statex_->nNodes());
    std::fill(sendIdx_.begin(), sendIdx_.end(), 0);
    fwdIdx_.resize(
        pArgs_.totalNumSendBlocks * comm_->ctranComm_->statex_->nNodes() *
        comm_->ctranComm_->statex_->nLocalRanks());
    std::fill(fwdIdx_.begin(), fwdIdx_.end(), 0);
    recvIdx_.resize(
        pArgs_.totalNumSendBlocks * comm_->ctranComm_->statex_->nNodes() *
        comm_->ctranComm_->statex_->nLocalRanks() * pArgs_.numRecvBuckets);
    std::fill(recvIdx_.begin(), recvIdx_.end(), 0);
    if (!allRankBlockRecvBucketsOverride) {
      allRankBlockRecvBuckets.resize(nRanks);
      for (auto& vec : allRankBlockRecvBuckets) {
        vec.clear();
      }
    }

    blockRecvBuckets = createDeviceArg<int>(blockRecvBuckets_);
    ASSERT_NE(blockRecvBuckets, nullptr);

    numSendBlocks = createDeviceArg<size_t>(numSendBlocks_, true);
    ASSERT_NE(numSendBlocks, nullptr);

    numRecvBlocks = createDeviceArg<size_t>(numRecvBlocks_);
    ASSERT_NE(numRecvBlocks, nullptr);

    recvOffsets = createDeviceArg<size_t>(recvOffsets_);
    ASSERT_NE(recvOffsets, nullptr);

    numForwardBlocks = createDeviceArg<size_t>(numForwardBlocks_, true);
    ASSERT_NE(numForwardBlocks, nullptr);

    totalNumRecvBlocks = createDeviceArg<size_t>(totalNumRecvBlocks_);
    ASSERT_NE(totalNumRecvBlocks, nullptr);

    sendIdx = createDeviceArg<int>(sendIdx_);
    ASSERT_NE(sendIdx, nullptr);

    fwdIdx = createDeviceArg<int>(fwdIdx_);
    ASSERT_NE(fwdIdx, nullptr);

    recvIdx = createDeviceArg<int>(recvIdx_);
    ASSERT_NE(recvIdx, nullptr);

    std::vector<char> barrierByte_(1);
    barrierByte = createDeviceArg<char>(barrierByte_);
    ASSERT_NE(barrierByte, nullptr);

    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  void setupBlockRecvBucketsRand(int x, int nRanks, int* blockRecvBuckets) {
    // Set up blockRecvBucket input, each rank has the same data which means
    // numSendBlocks will be the same on each rank (e.g. [1 4 5 6] for the 2x2
    // test). numRecvBlocks will look like [1 1 1 1] on rank0, [4 4 4 4] on
    // rank1, ... and recvOffsets will look like [0 1 2 3] on rank0, [0 4 8
    // 12] on rank1, ...
    for (int i = 0; i < pArgs_.totalNumSendBlocks; i++) {
      for (int j = 0; j < pArgs_.blockNumRecvBuckets; j++) {
        blockRecvBuckets_[i * pArgs_.blockNumRecvBuckets + j] =
            (x + rand()) % nRanks;
      }
    }
    // Copy to all ranks since values are the same
    for (int i = 0; i < nRanks; i++) {
      allRankBlockRecvBuckets[i] = blockRecvBuckets_;
    }

    updateDeviceArg(blockRecvBuckets, blockRecvBuckets_, stream);
#ifdef VERBOSE
    std::cout << "TEST rank " << myRank_ << " iter " << x
              << " blockRecvBuckets " << blockRecvBuckets << ": "
              << folly::join(",", blockRecvBuckets_) << std::endl;
#endif
  }

  void setupBlockRecvBuckets(
      int x,
      int myRank,
      int nRanks,
      int* blockRecvBuckets,
      bool allRankBlockRecvBucketsOverride = false) {
    if (allRankBlockRecvBucketsOverride) {
      blockRecvBuckets_ = allRankBlockRecvBuckets[myRank];
    } else {
      // Before prepare:
      // - Generate recvRanks for each sendblock
      getBlockRecvBuckets(myRank, x, blockRecvBuckets_);

      // Generating allRankBlockRecvBuckets for all ranks to get expected
      // prepare outputs
      for (int sendRank = 0; sendRank < nRanks; sendRank++) {
        allRankBlockRecvBuckets[sendRank].resize(blockRecvBuckets_.size());
        auto& buckets = allRankBlockRecvBuckets[sendRank];
        getBlockRecvBuckets(sendRank, x, buckets);
      }
    }

    updateDeviceArg(blockRecvBuckets, blockRecvBuckets_, stream);
#ifdef VERBOSE
    std::cout << "TEST prepare: rank " << myRank << " blockRecvBuckets "
              << blockRecvBuckets << ": "
              << ::ctran::utils::array2DToStr(
                     blockRecvBuckets_.data(),
                     pArgs_.totalNumSendBlocks,
                     pArgs_.blockNumRecvBuckets)
              << std::endl;
#endif
  }

  int bucketToNode(int bucket) {
    const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();
    const auto numRecvBuckets = pArgs_.numRecvBuckets;
    return bucket / (nLocalRanks * numRecvBuckets);
  }

  int bucketToRank(int bucket) {
    const auto numRecvBuckets = pArgs_.numRecvBuckets;
    return bucket / numRecvBuckets;
  }

  void setupDispatchIndices(
      int x,
      int myRank,
      int nRanks,
      int* sendIdx,
      int* fwdIdx,
      int* recvIdx) {
    const auto totalNumSendBlocks = pArgs_.totalNumSendBlocks;
    const auto numRecvBuckets = pArgs_.numRecvBuckets;
    const auto blockNumRecvBuckets = pArgs_.blockNumRecvBuckets;
    const int nNodes = comm_->ctranComm_->statex_->nNodes();
    const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();
    const int myLocalRank = comm_->ctranComm_->statex_->localRank();
    const int myNode = comm_->ctranComm_->statex_->node();
    // setup sendIdx
    for (int node = 0; node < nNodes; node++) {
      int idx = 0;
      for (int b = 0; b < totalNumSendBlocks; b++) {
        bool blockSendToNode = false;
        for (int r = 0; r < blockNumRecvBuckets; r++) {
          int bucketNode = bucketToNode(
              allRankBlockRecvBuckets[myRank][b * blockNumRecvBuckets + r]);
          if (bucketNode == node) {
            blockSendToNode = true;
            break;
          }
        }
        if (blockSendToNode) {
          sendIdx_[node * totalNumSendBlocks + b] = idx;
          idx++;
        } else {
          sendIdx_[node * totalNumSendBlocks + b] = -1;
        }
      }
    }
    // setup fwdIdx
    for (int node = 0; node < nNodes; node++) {
      for (int localRank = 0; localRank < nLocalRanks; localRank++) {
        int idx = 0;
        for (int b = 0; b < totalNumSendBlocks; b++) {
          bool blockSendToRank = false;
          for (int r = 0; r < blockNumRecvBuckets; r++) {
            int bucketRank = bucketToRank(
                allRankBlockRecvBuckets[node * nLocalRanks + myLocalRank]
                                       [b * blockNumRecvBuckets + r]);
            if (bucketRank == myNode * nLocalRanks + localRank) {
              blockSendToRank = true;
              break;
            }
          }
          if (blockSendToRank) {
            fwdIdx_
                [localRank * totalNumSendBlocks * nNodes +
                 node * totalNumSendBlocks + b] = idx;
            idx++;
          } else {
            fwdIdx_
                [localRank * totalNumSendBlocks * nNodes +
                 node * totalNumSendBlocks + b] = -1;
          }
        }
      }
    }
    // setup recvIdx
    for (int rankBucket = 0; rankBucket < numRecvBuckets; rankBucket++) {
      for (int rank = 0; rank < nRanks; rank++) {
        int idx = 0;
        for (int b = 0; b < totalNumSendBlocks; b++) {
          bool blockSendToBucket = false;
          for (int r = 0; r < blockNumRecvBuckets; r++) {
            int sendBucket =
                allRankBlockRecvBuckets[rank][b * blockNumRecvBuckets + r];
            if (sendBucket == myRank * numRecvBuckets + rankBucket) {
              blockSendToBucket = true;
            }
          }
          if (blockSendToBucket) {
            recvIdx_
                [rankBucket * totalNumSendBlocks * nRanks +
                 rank * totalNumSendBlocks + b] = idx;
            idx++;
          } else {
            recvIdx_
                [rankBucket * totalNumSendBlocks * nRanks +
                 rank * totalNumSendBlocks + b] = -1;
          }
        }
      }
    }
    updateDeviceArg(sendIdx, sendIdx_, stream);
    updateDeviceArg(fwdIdx, fwdIdx_, stream);
    updateDeviceArg(recvIdx, recvIdx_, stream);
#ifdef VERBOSE
    std::cout << "TEST prepare: rank " << myRank << " sendIdx " << sendIdx
              << ": "
              << ::ctran::utils::array2DToStr(
                     sendIdx_.data(), nNodes, pArgs_.totalNumSendBlocks)
              << std::endl;
    std::cout << "TEST prepare: rank " << myRank << " fwdIdx " << fwdIdx << ": "
              << ::ctran::utils::array2DToStr(
                     fwdIdx_.data(),
                     nNodes * nLocalRanks,
                     pArgs_.totalNumSendBlocks)
              << std::endl;
    std::cout << "TEST prepare: rank " << myRank << " recvIdx " << recvIdx
              << ": "
              << ::ctran::utils::array2DToStr(
                     recvIdx_.data(),
                     nNodes * nLocalRanks,
                     pArgs_.totalNumSendBlocks * pArgs_.numRecvBuckets)
              << std::endl;
#endif
  }

  void checkPrepareOutputs(
      int x,
      int nRanks,
      CtranPersistentRequest*& request,
      size_t*& numSendBlocks,
      size_t*& numRecvBlocks,
      size_t*& recvOffsets,
      size_t*& numForwardBlocks,
      size_t*& totalNumRecvBlocks,
      std::vector<int>& xnodeInputSplits,
      std::vector<int>& xnodeOutputSplits,
      std::vector<int>& localInputSplits,
      std::vector<int>& localOutputSplits) {
    // Copy results from device
    loadDeviceArg(numSendBlocks_, numSendBlocks, stream_);
    loadDeviceArg(numForwardBlocks_, numForwardBlocks, stream_);
    loadDeviceArg(numRecvBlocks_, numRecvBlocks, stream_);
    loadDeviceArg(recvOffsets_, recvOffsets, stream_);
    loadDeviceArg(totalNumRecvBlocks_, totalNumRecvBlocks, stream_);

    // Generate expected results
    std::vector<size_t> expNumSendBlocks(nRanks, 0);
    std::vector<size_t> expNumForwardBlocks(nRanks, 0);
    std::vector<size_t> expNumRecvBlocks(nRanks, 0);
    std::vector<size_t> expRecvOffsets(nRanks, 0);
    std::vector<int> expXNodeInputSplits(nRanks, 0);
    std::vector<int> expXNodeOutputSplits(nRanks, 0);
    std::vector<int> expLocalInputSplits(nRanks, 0);
    std::vector<int> expLocalOutputSplits(nRanks, 0);
    size_t expTotalNumRecvBlocks = 0;

    getExpPrepareOutputs(
        allRankBlockRecvBuckets,
        x,
        expNumSendBlocks,
        expNumRecvBlocks,
        expRecvOffsets,
        expNumForwardBlocks,
        expTotalNumRecvBlocks,
        expXNodeInputSplits,
        expXNodeOutputSplits,
        expLocalInputSplits,
        expLocalOutputSplits,
        request);

    // Check equality for each results and expected result
    for (int i = 0; i < nRanks; i++) {
      EXPECT_EQ(numSendBlocks_[i], expNumSendBlocks[i])
          << "on rank " << globalRank << " iteration " << x << " at " << i
          << ", numSendBlocks_=" << folly::join(",", numSendBlocks_)
          << ", expected " << folly::join(",", expNumSendBlocks) << std::endl;
      EXPECT_EQ(expNumForwardBlocks[i], numForwardBlocks_[i])
          << "on rank " << globalRank << " iteration " << x << " at " << i
          << ", numForwardBlocks_=" << folly::join(",", numForwardBlocks_)
          << ", expected " << folly::join(",", expNumForwardBlocks)
          << std::endl;
      EXPECT_EQ(expNumRecvBlocks[i], numRecvBlocks_[i])
          << "on rank " << globalRank << " iteration " << x << " at " << i
          << ", numRecvBlocks_=" << folly::join(",", numRecvBlocks_)
          << ", expected " << folly::join(",", expNumRecvBlocks) << std::endl;
      EXPECT_EQ(expRecvOffsets[i], recvOffsets_[i])
          << "on rank " << globalRank << " iteration " << x << " at " << i
          << ", recvOffsets_=" << folly::join(",", recvOffsets_)
          << ", expected " << folly::join(",", expRecvOffsets) << std::endl;
      EXPECT_EQ(expXNodeInputSplits[i], xnodeInputSplits[i])
          << "on rank " << globalRank << " iteration " << x << " at " << i
          << ", xnodeInputSplits=" << folly::join(",", xnodeInputSplits)
          << ", expected " << folly::join(",", expXNodeInputSplits)
          << std::endl;
      EXPECT_EQ(expXNodeOutputSplits[i], xnodeOutputSplits[i])
          << "on rank " << globalRank << " iteration " << x << " at " << i
          << ", xnodeOutputSplits=" << folly::join(",", xnodeOutputSplits)
          << ", expected " << folly::join(",", expXNodeOutputSplits)
          << std::endl;
      EXPECT_EQ(expLocalInputSplits[i], localInputSplits[i])
          << "on rank " << globalRank << " iteration " << x << " at " << i
          << ", localInputSplits=" << folly::join(",", localInputSplits)
          << ", expected " << folly::join(",", expLocalInputSplits)
          << std::endl;
      EXPECT_EQ(expLocalOutputSplits[i], localOutputSplits[i])
          << "on rank " << globalRank << " iteration " << x << " at " << i
          << ", localOutputSplits=" << folly::join(",", localOutputSplits)
          << ", expected " << folly::join(",", expLocalOutputSplits)
          << std::endl;
    }
    EXPECT_EQ(totalNumRecvBlocks_[0], expTotalNumRecvBlocks)
        << "on rank " << globalRank << " iteration " << x
        << " totalNumRecvBlocks_=" << totalNumRecvBlocks_[0] << ", expected "
        << expTotalNumRecvBlocks << std::endl;
#ifdef VERBOSE
    std::cout << "on rank " << globalRank << " iteration " << x
              << ", numSendBlocks_=" << folly::join(",", numSendBlocks_)
              << ", expected " << folly::join(",", expNumSendBlocks)
              << std::endl;
    std::cout << "on rank " << globalRank << " iteration " << x
              << ", numForwardBlocks_=" << folly::join(",", numForwardBlocks_)
              << ", expected " << folly::join(",", expNumForwardBlocks)
              << std::endl;
    std::cout << "on rank " << globalRank << " iteration " << x
              << ", numRecvBlocks_=" << folly::join(",", numRecvBlocks_)
              << ", expected " << folly::join(",", expNumRecvBlocks)
              << std::endl;
    std::cout << "on rank " << globalRank << " iteration " << x
              << ", recvOffsets_=" << folly::join(",", recvOffsets_)
              << ", expected " << folly::join(",", expRecvOffsets) << std::endl;
    std::cout << "on rank " << globalRank << " iteration " << x
              << ", xnodeInputSplits=" << folly::join(",", xnodeInputSplits)
              << ", expected " << folly::join(",", expXNodeInputSplits)
              << std::endl;
    std::cout << "on rank " << globalRank << " iteration " << x
              << ", xnodeOutputSplits=" << folly::join(",", xnodeOutputSplits)
              << ", expected " << folly::join(",", expXNodeOutputSplits)
              << std::endl;
    std::cout << "on rank " << globalRank << " iteration " << x
              << ", localInputSplits=" << folly::join(",", localInputSplits)
              << ", expected " << folly::join(",", expLocalInputSplits)
              << std::endl;
    std::cout << "on rank " << globalRank << " iteration " << x
              << ", localOutputSplits=" << folly::join(",", localOutputSplits)
              << ", expected " << folly::join(",", expLocalOutputSplits)
              << std::endl;
    std::cout << "on rank " << globalRank << " iteration " << x
              << " totalNumRecvBlocks_=" << totalNumRecvBlocks_[0]
              << ", expected " << expTotalNumRecvBlocks << std::endl;
#endif
  }

  void barrier(ncclComm_t comm, cudaStream_t stream, char* barrierByte) {
    // simple Allreduce as barrier before get data from other ranks
    NCCLCHECK_TEST(ncclAllReduce(
        barrierByte, barrierByte, 1, ncclChar, ncclSum, comm, stream));
  }

  template <commDataType_t DataType = commInt>
  void run(
      bool skipExec = false,
      std::optional<int> currNumIter = std::nullopt,
      bool allRankBlockRecvBucketsOverride = false) {
    using DT = typename CommTypeTraits<DataType>::T;
    size_t dataTypeSize = sizeof(DT);
    meta::comms::Hints hints; // unused

    std::cout << "calling support check with comm_ " << comm_ << std::endl;
    if (!::ctran::allToAllvDedupSupport(comm_->ctranComm_.get(), hints)) {
      GTEST_SKIP() << "Skip test because allToAllvDedupSupport returns false";
    }

    // generateDistRandomExpValue();
    expectedVal_ = 0;

    const int myRank = comm_->ctranComm_->statex_->rank();
    myRank_ = myRank;
    const int nRanks = comm_->ctranComm_->statex_->nRanks();
    const int nNodes = comm_->ctranComm_->statex_->nNodes();
    const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();

    int* blockRecvBuckets = nullptr;
    size_t* numSendBlocks = nullptr;
    size_t* numRecvBlocks = nullptr;
    size_t* recvOffsets = nullptr;
    size_t* numForwardBlocks = nullptr;
    size_t* totalNumRecvBlocks = nullptr;
    int* sendIdx = nullptr;
    int* fwdIdx = nullptr;
    int* recvIdx = nullptr;
    char* barrierByte = nullptr;
    std::vector<int> xnodeInputSplits(nRanks, 0);
    std::vector<int> xnodeOutputSplits(nRanks, 0);
    std::vector<int> xnodeGatherIndices(nNodes * pArgs_.totalNumSendBlocks, 0);
    std::vector<int> localInputSplits(nRanks, 0);
    std::vector<int> localOutputSplits(nRanks, 0);
    std::vector<int> localGatherIndices(
        nLocalRanks * nNodes * pArgs_.totalNumSendBlocks, 0);
    std::vector<int> eGatherIndices(
        nLocalRanks * nNodes * pArgs_.totalNumSendBlocks *
            pArgs_.numRecvBuckets,
        0);

    setupDeviceArgs(
        nRanks,
        blockRecvBuckets,
        numSendBlocks,
        numRecvBlocks,
        recvOffsets,
        numForwardBlocks,
        totalNumRecvBlocks,
        sendIdx,
        fwdIdx,
        recvIdx,
        barrierByte,
        allRankBlockRecvBucketsOverride);

    CtranPersistentRequest* request = nullptr;
    ASSERT_EQ(
        ::ctran::allToAllvDedupInit(
            pArgs_.totalNumSendBlocks,
            pArgs_.blockCount,
            pArgs_.blockNumRecvBuckets,
            pArgs_.numRecvBuckets,
            hints,
            DataType,
            comm_->ctranComm_.get(),
            stream_,
            request),
        commSuccess);
    ASSERT_NE(request, nullptr);

    const auto numIter =
        currNumIter.has_value() ? currNumIter.value() : defaultNumIters;
    std::vector<double> dispatchTimes;
    std::vector<double> combineTimes;
    dispatchTimes.reserve(numIter);
    combineTimes.reserve(numIter);
    cudaEvent_t dispatchStart, dispatchStop;
    cudaEvent_t combineStart, combineStop;
    ASSERT_EQ(cudaEventCreate(&dispatchStart), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&dispatchStop), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&combineStart), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&combineStop), cudaSuccess);

    for (int x = 0; x < numIter; x++) {
      setupBlockRecvBuckets(
          x, myRank, nRanks, blockRecvBuckets, allRankBlockRecvBucketsOverride);

      setupDispatchIndices(x, myRank, nRanks, sendIdx, fwdIdx, recvIdx);

      ASSERT_EQ(
          ::ctran::allToAllvDedupPrepare(
              blockRecvBuckets,
              numSendBlocks,
              numRecvBlocks,
              recvOffsets,
              numForwardBlocks,
              totalNumRecvBlocks,
              xnodeInputSplits.data(),
              xnodeOutputSplits.data(),
              xnodeGatherIndices.data(),
              localInputSplits.data(),
              localOutputSplits.data(),
              localGatherIndices.data(),
              eGatherIndices.data(),
              request),
          commSuccess);

      ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess)
          << "errored on rank " << myRank;

      checkPrepareOutputs(
          x,
          nRanks,
          request,
          numSendBlocks,
          numRecvBlocks,
          recvOffsets,
          numForwardBlocks,
          totalNumRecvBlocks,
          xnodeInputSplits,
          xnodeOutputSplits,
          localInputSplits,
          localOutputSplits);

      if (skipExec) {
        releaseDataBufs();
        continue;
      }

      DT* sendBuff = nullptr;
      DT* recvBuff = nullptr;

      // Update expectedVal to use different value per iteration
      expectedVal_ += 100000000;

      // - Allocate and assign values for sendBuff
      sendBuff = (DT*)createDataBuf(
          pArgs_.totalNumSendBlocks * pArgs_.blockCount * dataTypeSize);
      assignSendBuffVals<DT>(sendBuff, x, blockRecvBuckets_);

      // Before exec:
      // - Allocate recvBuff based on returned totalNumRecvBlocks from prepare
      recvBuff = (DT*)createDataBuf(
          totalNumRecvBlocks_[0] * pArgs_.blockCount * dataTypeSize);
      assignChunkValue<DT>(
          recvBuff,
          totalNumRecvBlocks_[0] * pArgs_.blockCount,
          DT(-1),
          stream_);

      int* blockSendRanks =
          createDeviceArg<int>(std::vector<int>(pArgs_.totalNumSendBlocks, -1));
      ASSERT_NE(blockSendRanks, nullptr);
      std::cout << "dispatch on rank " << myRank << " iteration " << x
                << std::endl;
      barrier(comm_, stream_, barrierByte);

      // start timing
      ASSERT_EQ(cudaEventRecord(dispatchStart, stream_), cudaSuccess);

      ASSERT_EQ(
          ::ctran::allToAllvDedupExec(
              sendBuff,
              blockRecvBuckets,
              numSendBlocks,
              numRecvBlocks,
              recvOffsets,
              numForwardBlocks,
              totalNumRecvBlocks_[0],
              sendIdx,
              fwdIdx,
              recvIdx,
              recvBuff,
              blockSendRanks,
              request),
          commSuccess);

      // end timing
      ASSERT_EQ(cudaEventRecord(dispatchStop, stream_), cudaSuccess);
      ASSERT_EQ(cudaEventSynchronize(dispatchStop), cudaSuccess);
      float dispatchMs;
      ASSERT_EQ(
          cudaEventElapsedTime(&dispatchMs, dispatchStart, dispatchStop),
          cudaSuccess);
      dispatchTimes.push_back(dispatchMs);

      ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess)
          << "errored on rank " << myRank << " iter " << x << std::endl;

#ifdef VERBOSE
      std::vector<DT> allRecvBuffVals(
          pArgs_.blockCount * totalNumRecvBlocks_[0], -1);
      CUDACHECK_TEST(cudaMemcpy(
          allRecvBuffVals.data(),
          recvBuff,
          dataTypeSize * pArgs_.blockCount * totalNumRecvBlocks_[0],
          cudaMemcpyDefault));

      std::cout << fmt::format(
                       "TEST rank {} iter {} allRecvBuffVals: {}",
                       myRank,
                       x,
                       folly::join(",", allRecvBuffVals))
                << std::endl;
#endif

      // Check received value
      size_t recvOffset = 0;
      for (int e = 0; e < pArgs_.numRecvBuckets; e++) {
        for (int localRank = 0; localRank < nLocalRanks; localRank++) {
          for (int node = 0; node < nNodes; node++) {
            auto rank = node * nLocalRanks + localRank;
            if (numRecvBlocks_[rank * pArgs_.numRecvBuckets + e]) {
              // get sendBlockIdx for all recvBlocks from this rank to compute
              // expectedVal
              const int myRecvBucket = myRank * pArgs_.numRecvBuckets + e;
              std::vector<size_t> sendBlockIdx = getSendBlockIdx(
                  rank, allRankBlockRecvBuckets[rank], myRecvBucket);
              ASSERT_EQ(
                  sendBlockIdx.size(),
                  numRecvBlocks_[rank * pArgs_.numRecvBuckets + e]);

              std::vector<std::string> observedValStrs;
              std::vector<DT> expectedVals;
              observedValStrs.reserve(sendBlockIdx.size());
              expectedVals.reserve(sendBlockIdx.size());
              for (auto b = 0; b < sendBlockIdx.size(); b++) {
                auto expectedVal =
                    DT(expectedVal_ + rank * 100000 + sendBlockIdx[b]);
                expectedVals.push_back(expectedVal);
#ifdef VERBOSE
                std::vector<DT> observedVals(1, -1);
                CUDACHECK_TEST(cudaMemcpy(
                    observedVals.data(),
                    recvBuff + recvOffset,
                    dataTypeSize,
                    cudaMemcpyDefault));
                observedValStrs.push_back(valsToStr(observedVals));
#endif

                int errs = checkChunkValue<DT>(
                    recvBuff + recvOffset, pArgs_.blockCount, expectedVal);
                EXPECT_EQ(errs, 0) << fmt::format(
                    "rank {} iter {} checked recvBuff[{}][{}] recvBlock {} sendBlockIdx {} offset {} with {} errors ",
                    myRank,
                    x,
                    e,
                    rank,
                    b,
                    folly::join(",", sendBlockIdx),
                    recvOffset,
                    errs);
                recvOffset += pArgs_.blockCount;
              }
#ifdef VERBOSE
              std::cout
                  << fmt::format(
                         "TEST rank {} iter {} recvBuff[{}]: {} (expected {}, sendBlockIdx: {})",
                         myRank,
                         x,
                         e,
                         folly::join(", ", observedValStrs),
                         folly::join(",", expectedVals),
                         folly::join(",", sendBlockIdx))
                  << std::endl;
#endif
            }
          }
        }
      }

      releaseDataBufs();
    } // end of iteration of prepare-exec
    ASSERT_EQ(::ctran::allToAllvDedupDestroy(request), commSuccess);
    delete request;
    releaseDeviceArgs();

    double avgDispatch =
        std::accumulate(dispatchTimes.begin(), dispatchTimes.end(), 0.0) /
        dispatchTimes.size();
    double avgCombine =
        std::accumulate(combineTimes.begin(), combineTimes.end(), 0.0) /
        combineTimes.size();
    double size = dataTypeSize * pArgs_.blockCount * pArgs_.totalNumSendBlocks;
    double sizeGB = size / (1 << 30);
    double dispatchBW = sizeGB / (avgDispatch / 1e3);
    double combineBW = sizeGB / (avgCombine / 1e3);
    std::cout
        << fmt::format(
               "rank {}, avg dispatch time: {} ({} GB/s), avg combine time: {} ({} GB/s)",
               myRank,
               avgDispatch,
               dispatchBW,
               avgCombine,
               combineBW)
        << std::endl;
    std::cout << fmt::format(
                     "rank {}, dispatch times: {}", myRank, dispatchTimes)
              << std::endl;
    std::cout << fmt::format("rank {}, combine times: {}", myRank, combineTimes)
              << std::endl;
  }

 protected:
  cudaStream_t stream_{0};
  ncclComm_t comm_{nullptr};
  int expectedVal_{0};
  int myRank_;

  std::unordered_set<int> skippedRecvBuckets_ = {};
  ArgMemType argMemType_{ArgMemType::kArgInHostRegisteredMem};
  std::vector<std::pair<void*, void*>> deviceArgs_;
  std::vector<std::pair<void*, void*>> dataBufHdls_;
  struct {
    int totalNumSendBlocks{0}; // number of tokens
    size_t blockCount{0}; // elements per token
    int blockNumRecvBuckets{0}; // topK
    int numRecvBuckets{0}; // number of experts per rank
  } pArgs_;
  std::vector<int> blockRecvBuckets_;
  std::vector<size_t> numSendBlocks_;
  std::vector<size_t> numRecvBlocks_;
  std::vector<size_t> recvOffsets_;
  std::vector<size_t> numForwardBlocks_;
  std::vector<size_t> totalNumRecvBlocks_;
  std::vector<int> sendIdx_;
  std::vector<int> fwdIdx_;
  std::vector<int> recvIdx_;
  std::vector<std::vector<int>> allRankBlockRecvBuckets;

  const int defaultNumIters = 10;
};

TEST_F(CtranAllToAllvDedupTest, InitDestroy) {
  meta::comms::Hints hints; // unused

  setTestParams(16, 8192, 4, 2);

  std::cout << "calling support check with comm_ " << comm_ << std::endl;
  if (!::ctran::allToAllvDedupSupport(comm_->ctranComm_.get(), hints)) {
    GTEST_SKIP() << "Skip test because allToAllvDedupSupport returns false";
  }

  auto usedBytesBase =
      ncclx::memory::memCacheAllocator::getInstance()->getUsedMem();

  // Multiple times init / destory to ensure no memory leak nor race condition
  const int myRank = comm_->ctranComm_->statex_->rank();
  const auto numIters = defaultNumIters;
  for (int x = 0; x < numIters; x++) {
    if (myRank == 0) {
      std::cout << " InitDestroy starts iter " << x << std::endl;
    }
    CtranPersistentRequest* request = nullptr;
    auto numUsedSegsBeforeInit =
        ncclx::memory::memCacheAllocator::getInstance()->getNumUsedReg();

    ASSERT_EQ(
        ::ctran::allToAllvDedupInit(
            pArgs_.totalNumSendBlocks,
            pArgs_.blockCount,
            pArgs_.blockNumRecvBuckets,
            pArgs_.numRecvBuckets,
            hints,
            commInt,
            comm_->ctranComm_.get(),
            stream_,
            request),
        commSuccess);

    auto numUsedSegsAfterInit =
        ncclx::memory::memCacheAllocator::getInstance()->getNumUsedReg();
    ASSERT_NE(request, nullptr);

    // memory pool may not release the memory after dedup destroy, thus get
    // delta based on usage before first dedup init
    auto usedBytes =
        ncclx::memory::memCacheAllocator::getInstance()->getUsedMem() -
        usedBytesBase;

    ASSERT_EQ(::ctran::allToAllvDedupDestroy(request), commSuccess);
    delete request;
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
}

TEST_F(CtranAllToAllvDedupTest, PrepareAsymmetric) {
  const int totalNumSendBlocks = 4; // number of tokens
  const int blockNumRecvBuckets = 3; // topK
  const size_t blockCount = 8192; // elements per token
  const int numRecvBuckets = 2; // number of experts per rank
  ncclDataType_t DataType = ncclInt;
  meta::comms::Hints hints; // unused

  setTestParams(
      totalNumSendBlocks, blockCount, blockNumRecvBuckets, numRecvBuckets);

  const int myRank = comm_->ctranComm_->statex_->rank();
  const int nRanks = comm_->ctranComm_->statex_->nRanks();
  const int nNodes = comm_->ctranComm_->statex_->nNodes();
  const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();

  if (nRanks != 4 || nLocalRanks != 2) {
    GTEST_SKIP() << "Skip test because special 2x2 config test";
  }

  int* blockRecvBuckets = nullptr;
  size_t* numSendBlocks = nullptr;
  size_t* numRecvBlocks = nullptr;
  size_t* recvOffsets = nullptr;
  size_t* numForwardBlocks = nullptr;
  size_t* totalNumRecvBlocks = nullptr;
  int* sendIdx = nullptr;
  int* fwdIdx = nullptr;
  int* recvIdx = nullptr;
  char* barrierByte = nullptr;
  std::vector<int> xnodeInputSplits(nRanks, 0);
  std::vector<int> xnodeOutputSplits(nRanks, 0);
  std::vector<int> xnodeGatherIndices(nNodes * totalNumSendBlocks, 0);
  std::vector<int> localInputSplits(nRanks, 0);
  std::vector<int> localOutputSplits(nRanks, 0);
  std::vector<int> localGatherIndices(
      nLocalRanks * nNodes * totalNumSendBlocks, 0);
  std::vector<int> eGatherIndices(
      nLocalRanks * nNodes * totalNumSendBlocks * numRecvBuckets, 0);

  setupDeviceArgs(
      nRanks,
      blockRecvBuckets,
      numSendBlocks,
      numRecvBlocks,
      recvOffsets,
      numForwardBlocks,
      totalNumRecvBlocks,
      sendIdx,
      fwdIdx,
      recvIdx,
      barrierByte);

  CtranPersistentRequest* request = nullptr;
  ASSERT_EQ(
      ::ctran::allToAllvDedupInit(
          totalNumSendBlocks,
          blockCount,
          blockNumRecvBuckets,
          numRecvBuckets,
          hints,
          ncclToMetaComm(DataType),
          comm_->ctranComm_.get(),
          stream_,
          request),
      commSuccess);

  allRankBlockRecvBuckets[0] = {3, 1, 2, 0, 3, 2, 5, 6, 7, 5, 6, 7};
  allRankBlockRecvBuckets[1] = {1, 0, 2, 3, 0, 1, 4, 1, 3, 5, 1, 5};
  allRankBlockRecvBuckets[2] = {0, 2, 1, 3, 0, 1, 5, 0, 2, 3, 0, 1};
  allRankBlockRecvBuckets[3] = {2, 0, 1, 1, 2, 0, 7, 7, 7, 7, 7, 7};
  blockRecvBuckets_ = allRankBlockRecvBuckets[myRank];
  updateDeviceArg(blockRecvBuckets, blockRecvBuckets_, stream);

  ASSERT_EQ(
      ::ctran::allToAllvDedupPrepare(
          blockRecvBuckets,
          numSendBlocks,
          numRecvBlocks,
          recvOffsets,
          numForwardBlocks,
          totalNumRecvBlocks,
          xnodeInputSplits.data(),
          xnodeOutputSplits.data(),
          xnodeGatherIndices.data(),
          localInputSplits.data(),
          localOutputSplits.data(),
          localGatherIndices.data(),
          eGatherIndices.data(),
          request),
      commSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess)
      << "errored on rank " << myRank;

  checkPrepareOutputs(
      0,
      nRanks,
      request,
      numSendBlocks,
      numRecvBlocks,
      recvOffsets,
      numForwardBlocks,
      totalNumRecvBlocks,
      xnodeInputSplits,
      xnodeOutputSplits,
      localInputSplits,
      localOutputSplits);

  ASSERT_EQ(::ctran::allToAllvDedupDestroy(request), commSuccess);
  delete request;
  releaseDeviceArgs();
}

TEST_F(CtranAllToAllvDedupTest, Prepare) {
  const int totalNumSendBlocks = 8; // number of tokens
  const int blockNumRecvBuckets = 2; // topK
  const size_t blockCount = 8192; // elements per token
  // TODO add support for numRecvBuckets > 1
  const int numRecvBuckets = 1; // number of experts per rank
  commDataType_t DataType = commInt;
  meta::comms::Hints hints; // unused

  setTestParams(
      totalNumSendBlocks, blockCount, blockNumRecvBuckets, numRecvBuckets);

  const int myRank = comm_->ctranComm_->statex_->rank();
  const int nRanks = comm_->ctranComm_->statex_->nRanks();
  const int nNodes = comm_->ctranComm_->statex_->nNodes();
  const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();

  int* blockRecvBuckets = nullptr;
  size_t* numSendBlocks = nullptr;
  size_t* numRecvBlocks = nullptr;
  size_t* recvOffsets = nullptr;
  size_t* numForwardBlocks = nullptr;
  size_t* totalNumRecvBlocks = nullptr;
  int* sendIdx = nullptr;
  int* fwdIdx = nullptr;
  int* recvIdx = nullptr;
  char* barrierByte = nullptr;
  std::vector<int> xnodeInputSplits(nRanks, 0);
  std::vector<int> xnodeOutputSplits(nRanks, 0);
  std::vector<int> xnodeGatherIndices(nNodes * totalNumSendBlocks, 0);
  std::vector<int> localInputSplits(nRanks, 0);
  std::vector<int> localOutputSplits(nRanks, 0);
  std::vector<int> localGatherIndices(
      nLocalRanks * nNodes * totalNumSendBlocks, 0);
  std::vector<int> eGatherIndices(
      nLocalRanks * nNodes * totalNumSendBlocks * numRecvBuckets, 0);

  setupDeviceArgs(
      nRanks,
      blockRecvBuckets,
      numSendBlocks,
      numRecvBlocks,
      recvOffsets,
      numForwardBlocks,
      totalNumRecvBlocks,
      sendIdx,
      fwdIdx,
      recvIdx,
      barrierByte);

  CtranPersistentRequest* request = nullptr;
  ASSERT_EQ(
      ::ctran::allToAllvDedupInit(
          totalNumSendBlocks,
          blockCount,
          blockNumRecvBuckets,
          numRecvBuckets,
          hints,
          DataType,
          comm_->ctranComm_.get(),
          stream_,
          request),
      commSuccess);

  const auto numIters = defaultNumIters;
  for (int x = 0; x < numIters; x++) {
    // setupBlockRecvBucketsRand(x, nRanks, blockRecvBuckets);
    setupBlockRecvBuckets(x, myRank, nRanks, blockRecvBuckets);

    ASSERT_EQ(
        ::ctran::allToAllvDedupPrepare(
            blockRecvBuckets,
            numSendBlocks,
            numRecvBlocks,
            recvOffsets,
            numForwardBlocks,
            totalNumRecvBlocks,
            xnodeInputSplits.data(),
            xnodeOutputSplits.data(),
            xnodeGatherIndices.data(),
            localInputSplits.data(),
            localOutputSplits.data(),
            localGatherIndices.data(),
            eGatherIndices.data(),
            request),
        commSuccess);

    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess)
        << "errored on rank " << myRank;

    checkPrepareOutputs(
        x,
        nRanks,
        request,
        numSendBlocks,
        numRecvBlocks,
        recvOffsets,
        numForwardBlocks,
        totalNumRecvBlocks,
        xnodeInputSplits,
        xnodeOutputSplits,
        localInputSplits,
        localOutputSplits);
  }

  ASSERT_EQ(::ctran::allToAllvDedupDestroy(request), commSuccess);
  delete request;
  releaseDeviceArgs();
}

class CtranTestAllToAllvDedupFixture
    : public CtranAllToAllvDedupTest,
      public ::testing::WithParamInterface<std::tuple<
          int,
          size_t,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          bool>> {};

TEST_P(CtranTestAllToAllvDedupFixture, Basic) {
  const auto& [totalNumSendBlocks, blockCount, blockNumRecvBuckets, numRecvBuckets, chunkSizeMb, sendG, sendW, fwdG, fwdW, recvG, recvW, skipBucket] =
      GetParam();
  EnvRAII<int> env1(
      NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE, chunkSizeMb * 1 << 20);
  EnvRAII<int> envSendG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, sendG);
  EnvRAII<int> envSendW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP, sendW);
  EnvRAII<int> envFwdG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCK_GROUPS, fwdG);
  EnvRAII<int> envFwdW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS_PER_GROUP, fwdW);
  EnvRAII<int> envRecvG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, recvG);
  EnvRAII<int> envRecvW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP, recvW);

  const auto nRanks = comm_->ctranComm_->statex_->nRanks();

  int actualBlockNumRecvBuckets = std::min(nRanks, blockNumRecvBuckets);
  // optionally skip one bucket
  if (skipBucket) {
    skippedRecvBuckets_.insert(nRanks - 1);
    actualBlockNumRecvBuckets = std::min(nRanks - 1, blockNumRecvBuckets);
  }

  setTestParams(
      totalNumSendBlocks,
      blockCount,
      actualBlockNumRecvBuckets,
      numRecvBuckets);
  run<commInt32>();
  skippedRecvBuckets_.clear();
}

TEST_F(CtranTestAllToAllvDedupFixture, TracingPrepare) {
  const auto& [totalNumSendBlocks, blockCount, blockNumRecvBuckets, numRecvBuckets, chunkSizeMb, sendG, sendW, fwdG, fwdW, recvG, recvW, skipBucket] =
      std::make_tuple(8192, 8192, 2, 1, 4 /* MB */, 1, 1, 1, 1, 1, 1, false);
  EnvRAII<int> envPrepare(
      NCCL_CTRAN_ALLTOALLV_DEDUP_PREPARE_NUM_THREAD_BLOCK_GROUPS, 1);
  EnvRAII<bool> envTraceLogger(NCCL_CTRAN_ENABLE_TRACE_LOGGER, true);
  EnvRAII<int> env1(
      NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE, chunkSizeMb * 1 << 20);
  EnvRAII<int> envSendG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, sendG);
  EnvRAII<int> envSendW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP, sendW);
  EnvRAII<int> envFwdG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCK_GROUPS, fwdG);
  EnvRAII<int> envFwdW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS_PER_GROUP, fwdW);
  EnvRAII<int> envRecvG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, recvG);
  EnvRAII<int> envRecvW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP, recvW);

  const auto nRanks = comm_->ctranComm_->statex_->nRanks();
  int actualBlockNumRecvBuckets = std::min(nRanks, blockNumRecvBuckets);
  setTestParams(
      totalNumSendBlocks,
      blockCount,
      actualBlockNumRecvBuckets,
      numRecvBuckets);

  run<commInt32>(true /* skip exec */, 10 /* numIters */);
  skippedRecvBuckets_.clear();
}

TEST_F(CtranTestAllToAllvDedupFixture, TracingExec) {
  const auto& [totalNumSendBlocks, blockCount, blockNumRecvBuckets, numRecvBuckets, chunkSizeMb, sendG, sendW, fwdG, fwdW, recvG, recvW, skipBucket] =
      std::make_tuple(8192, 8192, 2, 2, 8 /* MB */, 1, 4, 1, 16, 1, 8, false);
  EnvRAII<int> envPrepare(
      NCCL_CTRAN_ALLTOALLV_DEDUP_PREPARE_NUM_THREAD_BLOCK_GROUPS, 1);
  EnvRAII<bool> envTraceLogger(NCCL_CTRAN_ENABLE_TRACE_LOGGER, true);
  EnvRAII<int> env1(
      NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE, chunkSizeMb * 1 << 20);
  EnvRAII<int> envSendG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, sendG);
  EnvRAII<int> envSendW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP, sendW);
  EnvRAII<int> envFwdG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCK_GROUPS, fwdG);
  EnvRAII<int> envFwdW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS_PER_GROUP, fwdW);
  EnvRAII<int> envRecvG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, recvG);
  EnvRAII<int> envRecvW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP, recvW);
  EnvRAII<int> envNumChunks(NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS, 8);

  const auto nRanks = comm_->ctranComm_->statex_->nRanks();

  int actualBlockNumRecvBuckets = std::min(nRanks, blockNumRecvBuckets);
  // optionally skip one bucket
  if (skipBucket) {
    skippedRecvBuckets_.insert(nRanks - 1);
    actualBlockNumRecvBuckets = std::min(nRanks - 1, blockNumRecvBuckets);
  }

  setTestParams(
      totalNumSendBlocks,
      blockCount,
      actualBlockNumRecvBuckets,
      numRecvBuckets);
  run<commInt32>();
  skippedRecvBuckets_.clear();
}

TEST_F(CtranTestAllToAllvDedupFixture, TracingExecInternode) {
  const auto& [totalNumSendBlocks, blockCount, blockNumRecvBuckets, numRecvBuckets, chunkSizeMb, sendG, sendW, fwdG, fwdW, recvG, recvW, skipBucket] =
      std::make_tuple(8192, 8192, 2, 2, 4 /* MB */, 1, 4, 3, 4, 8, 4, false);
  EnvRAII<int> envPrepare(
      NCCL_CTRAN_ALLTOALLV_DEDUP_PREPARE_NUM_THREAD_BLOCK_GROUPS, 1);
  EnvRAII<bool> envTraceLogger(NCCL_CTRAN_ENABLE_TRACE_LOGGER, true);
  EnvRAII<int> env1(
      NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE, chunkSizeMb * 1 << 20);
  EnvRAII<int> envSendG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, sendG);
  EnvRAII<int> envSendW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP, sendW);
  EnvRAII<int> envFwdG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCK_GROUPS, fwdG);
  EnvRAII<int> envFwdW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS_PER_GROUP, fwdW);
  EnvRAII<int> envRecvG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, recvG);
  EnvRAII<int> envRecvW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP, recvW);
  EnvRAII<int> envNumChunks(NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS, 1);

  const auto nRanks = comm_->ctranComm_->statex_->nRanks();
  const auto nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();
  const auto nNodes = comm_->ctranComm_->statex_->nNodes();

  int actualBlockNumRecvBuckets = std::min(nRanks, blockNumRecvBuckets);
  // optionally skip one bucket
  if (skipBucket) {
    skippedRecvBuckets_.insert(nRanks - 1);
    actualBlockNumRecvBuckets = std::min(nRanks - 1, blockNumRecvBuckets);
  }

  allRankBlockRecvBuckets.resize(nRanks);
  const auto numBucketsPerNode = nLocalRanks * numRecvBuckets;

  for (int i = 0; i < nRanks; i++) {
    const auto node = comm_->ctranComm_->statex_->node(i);
    std::vector<int> candidates;
    for (int peernode = 0; peernode < nNodes; peernode++) {
      if (peernode == node)
        continue;
      for (int j = 0; j < numBucketsPerNode; j++) {
        candidates.push_back(j + numBucketsPerNode * peernode);
      }
    }

    for (int j = 0; j < totalNumSendBlocks; j++) {
      for (int k = 0; k < blockNumRecvBuckets; k++) {
        allRankBlockRecvBuckets[i].push_back(
            candidates[(j + k) % candidates.size()]);
      }
    }
  }

  for (int i = 0; i < nRanks; i++) {
    std::cout << "allrankblockrecvbuckets " << i << ": "
              << allRankBlockRecvBuckets[i].size() << " :"
              << allRankBlockRecvBuckets[i] << std::endl;
  }

  setTestParams(
      totalNumSendBlocks,
      blockCount,
      actualBlockNumRecvBuckets,
      numRecvBuckets);
  run<commInt32>(false, 10, true);
  skippedRecvBuckets_.clear();
}

TEST_F(CtranTestAllToAllvDedupFixture, TracingExecSmall) {
  const auto& [totalNumSendBlocks, blockCount, blockNumRecvBuckets, numRecvBuckets, chunkSizeMb, sendG, sendW, fwdG, fwdW, recvG, recvW, skipBucket] =
      std::make_tuple(2, 8192, 2, 2, 4 /* MB */, 1, 4, 1, 4, 1, 4, false);
  EnvRAII<int> envPrepare(
      NCCL_CTRAN_ALLTOALLV_DEDUP_PREPARE_NUM_THREAD_BLOCK_GROUPS, 1);
  EnvRAII<bool> envTraceLogger(NCCL_CTRAN_ENABLE_TRACE_LOGGER, true);
  EnvRAII<int> env1(
      NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE, chunkSizeMb * 1 << 20);
  EnvRAII<int> envSendG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, sendG);
  EnvRAII<int> envSendW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP, sendW);
  EnvRAII<int> envFwdG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCK_GROUPS, fwdG);
  EnvRAII<int> envFwdW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS_PER_GROUP, fwdW);
  EnvRAII<int> envRecvG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, recvG);
  EnvRAII<int> envRecvW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP, recvW);

  const auto nRanks = comm_->ctranComm_->statex_->nRanks();

  if (nRanks != 4) {
    GTEST_SKIP() << "Skip test because special 4 rank test";
  }

  int actualBlockNumRecvBuckets = std::min(nRanks, blockNumRecvBuckets);

  allRankBlockRecvBuckets.resize(nRanks);
  allRankBlockRecvBuckets[0] = {1, 5, 5, 6};
  allRankBlockRecvBuckets[1] = {0, 1, 3, 5};
  allRankBlockRecvBuckets[2] = {3, 5, 0, 7};
  allRankBlockRecvBuckets[3] = {5, 6, 5, 6};

  setTestParams(
      totalNumSendBlocks,
      blockCount,
      actualBlockNumRecvBuckets,
      numRecvBuckets);
  run<commInt32>(false, 1, true);
  skippedRecvBuckets_.clear();
}

TEST_F(CtranTestAllToAllvDedupFixture, SmallChunkSize) {
  const auto& [totalNumSendBlocks, blockCount, blockNumRecvBuckets, numRecvBuckets, chunkSizeKb, sendG, sendW, fwdG, fwdW, recvG, recvW, numChunks] =
      std::make_tuple(32, 8192, 2, 1, 64 /* KB */, 1, 1, 1, 1, 8, 1, 8);
  EnvRAII<int> env1(
      NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE, chunkSizeKb * 1 << 10);
  EnvRAII<int> envSendG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, sendG);
  EnvRAII<int> envSendW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP, sendW);
  EnvRAII<int> envFwdG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCK_GROUPS, fwdG);
  EnvRAII<int> envFwdW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS_PER_GROUP, fwdW);
  EnvRAII<int> envRecvG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, recvG);
  EnvRAII<int> envRecvW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP, recvW);
  EnvRAII<int> envNumChunks(NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS, numChunks);

  const auto nRanks = comm_->ctranComm_->statex_->nRanks();
  const auto nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();
  const auto nNodes = comm_->ctranComm_->statex_->nNodes();

  int actualBlockNumRecvBuckets = std::min(nRanks, blockNumRecvBuckets);

  allRankBlockRecvBuckets.resize(nRanks);
  const auto numBucketsPerNode = nLocalRanks * numRecvBuckets;

  for (int i = 0; i < nRanks; i++) {
    const auto node = comm_->ctranComm_->statex_->node(i);
    std::vector<int> candidates;
    for (int peernode = 0; peernode < nNodes; peernode++) {
      if (peernode == node)
        continue;
      for (int j = 0; j < numBucketsPerNode; j++) {
        candidates.push_back(j + numBucketsPerNode * peernode);
      }
    }

    for (int j = 0; j < totalNumSendBlocks; j++) {
      for (int k = 0; k < blockNumRecvBuckets; k++) {
        allRankBlockRecvBuckets[i].push_back(
            candidates[(j + k) % candidates.size()]);
      }
    }
  }

  setTestParams(
      totalNumSendBlocks,
      blockCount,
      actualBlockNumRecvBuckets,
      numRecvBuckets);
  run<commInt32>(false /* skip exec */, 1, true);
  skippedRecvBuckets_.clear();
}

TEST_F(CtranTestAllToAllvDedupFixture, InvalidExecBuffs) {
  meta::comms::Hints hints; // unused

  pArgs_.totalNumSendBlocks = 16;
  pArgs_.blockCount = 8192;
  pArgs_.blockNumRecvBuckets = 4;
  pArgs_.numRecvBuckets = 2;

  std::cout << "calling support check with comm_ " << comm_ << std::endl;
  if (!::ctran::allToAllvDedupSupport(comm_->ctranComm_.get(), hints)) {
    GTEST_SKIP() << "Skip test because allToAllvDedupSupport returns false";
  }

  CtranPersistentRequest* request = nullptr;
  ASSERT_EQ(
      ::ctran::allToAllvDedupInit(
          pArgs_.totalNumSendBlocks,
          pArgs_.blockCount,
          pArgs_.blockNumRecvBuckets,
          pArgs_.numRecvBuckets,
          hints,
          commInt,
          comm_->ctranComm_.get(),
          stream_,
          request),
      commSuccess);
  ASSERT_NE(request, nullptr);

  const int nRanks = comm_->ctranComm_->statex_->nRanks();
  const int nNodes = comm_->ctranComm_->statex_->nNodes();
  const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();
  int* blockRecvBuckets = createDeviceArg<int>(std::vector<int>(
      pArgs_.totalNumSendBlocks * pArgs_.blockNumRecvBuckets, 0));
  size_t* numSendBlocks = createDeviceArg<size_t>(std::vector<size_t>(nRanks));
  size_t* numRecvBlocks = createDeviceArg<size_t>(std::vector<size_t>(nRanks));
  size_t* recvOffsets = createDeviceArg<size_t>(std::vector<size_t>(nRanks));
  size_t* numForwardBlocks =
      createDeviceArg<size_t>(std::vector<size_t>(nRanks));
  size_t totalNumRecvBlocks = pArgs_.totalNumSendBlocks;
  int* sendIdx = createDeviceArg<int>(
      std::vector<int>(pArgs_.totalNumSendBlocks * nNodes));
  int* fwdIdx = createDeviceArg<int>(
      std::vector<int>(pArgs_.totalNumSendBlocks * nNodes * nLocalRanks));
  int* recvIdx = createDeviceArg<int>(std::vector<int>(
      pArgs_.totalNumSendBlocks * nNodes * nLocalRanks *
      pArgs_.numRecvBuckets));

  int* blockSendRanks =
      createDeviceArg<int>(std::vector<int>(pArgs_.totalNumSendBlocks, -1));

  int* dataBuff = (int*)createDataBuf(
      pArgs_.totalNumSendBlocks * pArgs_.blockCount * sizeof(int));

  // Invalid sendBuff
  ASSERT_EQ(
      ::ctran::allToAllvDedupExec(
          nullptr,
          blockRecvBuckets,
          numSendBlocks,
          numRecvBlocks,
          recvOffsets,
          numForwardBlocks,
          totalNumRecvBlocks,
          sendIdx,
          fwdIdx,
          recvIdx,
          dataBuff,
          blockSendRanks,
          request),
      commInvalidArgument);

  // Invalid recvBuff
  ASSERT_EQ(
      ::ctran::allToAllvDedupExec(
          dataBuff,
          blockRecvBuckets,
          numSendBlocks,
          numRecvBlocks,
          recvOffsets,
          numForwardBlocks,
          totalNumRecvBlocks,
          sendIdx,
          fwdIdx,
          recvIdx,
          nullptr,
          blockSendRanks,
          request),
      commInvalidArgument);

  //  other invalid argumetns
  ASSERT_EQ(
      ::ctran::allToAllvDedupExec(
          dataBuff,
          nullptr,
          numSendBlocks,
          numRecvBlocks,
          recvOffsets,
          numForwardBlocks,
          totalNumRecvBlocks,
          sendIdx,
          fwdIdx,
          recvIdx,
          dataBuff,
          blockSendRanks,
          request),
      commInvalidArgument);

  ASSERT_EQ(
      ::ctran::allToAllvDedupExec(
          dataBuff,
          blockRecvBuckets,
          nullptr,
          numRecvBlocks,
          recvOffsets,
          numForwardBlocks,
          totalNumRecvBlocks,
          sendIdx,
          fwdIdx,
          recvIdx,
          dataBuff,
          blockSendRanks,
          request),
      commInvalidArgument);

  ASSERT_EQ(
      ::ctran::allToAllvDedupExec(
          dataBuff,
          blockRecvBuckets,
          numSendBlocks,
          nullptr,
          recvOffsets,
          numForwardBlocks,
          totalNumRecvBlocks,
          sendIdx,
          fwdIdx,
          recvIdx,
          dataBuff,
          blockSendRanks,
          request),
      commInvalidArgument);

  ASSERT_EQ(
      ::ctran::allToAllvDedupExec(
          dataBuff,
          blockRecvBuckets,
          numSendBlocks,
          numRecvBlocks,
          nullptr,
          numForwardBlocks,
          totalNumRecvBlocks,
          sendIdx,
          fwdIdx,
          recvIdx,
          dataBuff,
          blockSendRanks,
          request),
      commInvalidArgument);

  ASSERT_EQ(
      ::ctran::allToAllvDedupExec(
          dataBuff,
          blockRecvBuckets,
          numSendBlocks,
          numRecvBlocks,
          recvOffsets,
          nullptr,
          totalNumRecvBlocks,
          sendIdx,
          fwdIdx,
          recvIdx,
          dataBuff,
          blockSendRanks,
          request),
      commInvalidArgument);

  ASSERT_EQ(
      ::ctran::allToAllvDedupExec(
          dataBuff,
          blockRecvBuckets,
          numSendBlocks,
          numRecvBlocks,
          recvOffsets,
          numForwardBlocks,
          totalNumRecvBlocks,
          sendIdx,
          fwdIdx,
          recvIdx,
          dataBuff,
          nullptr,
          request),
      commInvalidArgument);

  ASSERT_EQ(::ctran::allToAllvDedupDestroy(request), commSuccess);
  delete request;
  releaseDataBufs();
  releaseDeviceArgs();
}

TEST_F(CtranTestAllToAllvDedupFixture, InvalidEnvConfig) {
  meta::comms::Hints hints; // unused

  EnvRAII<int> envSendG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, 16);
  EnvRAII<int> envFwdG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCK_GROUPS, 16);
  EnvRAII<int> envFwdW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS_PER_GROUP, 16);
  EnvRAII<int> envRecvG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, 16);

  pArgs_.totalNumSendBlocks = 16;
  pArgs_.blockCount = 16;
  pArgs_.blockNumRecvBuckets = 4;
  pArgs_.numRecvBuckets = 2;

  CtranPersistentRequest* request = nullptr;
  ASSERT_EQ(
      ::ctran::allToAllvDedupInit(
          pArgs_.totalNumSendBlocks,
          pArgs_.blockCount,
          pArgs_.blockNumRecvBuckets,
          pArgs_.numRecvBuckets,
          hints,
          commInt,
          comm_->ctranComm_.get(),
          stream_,
          request),
      commInvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranTestAllToAllvDedupFixture,
    ::testing::Values(
        std::make_tuple(8, 2048, 2, 1, 4 /* MB */, 1, 1, 1, 1, 1, 1, false),
        std::make_tuple(8192, 8192, 2, 1, 4 /* MB */, 1, 1, 1, 1, 1, 1, false),
        // FIXME temporarily skip tests using different buff size
        // std::make_tuple(256, 8192, 4, 1, 1 /* MB */, 1, 1, 2, 1, 1, 1,
        // false),
        std::make_tuple(8192, 8192, 4, 1, 4 /* MB */, 2, 1, 2, 2, 2, 1, false),
        std::make_tuple(8192, 8192, 4, 1, 4 /* MB */, 2, 2, 2, 2, 2, 1, false),
        std::make_tuple(8192, 8192, 4, 1, 4 /* MB */, 2, 2, 2, 2, 2, 2, false),
        std::make_tuple(8, 2048, 2, 2, 4 /* MB */, 1, 1, 1, 1, 1, 1, false),
        std::make_tuple(8192, 8192, 2, 2, 4 /* MB */, 1, 1, 1, 1, 1, 1, false),
        // FIXME temporarily skip tests using different buff size
        // std::make_tuple(256, 8192, 4, 2, 1 /* MB */, 1, 1, 2, 1, 1, 1,
        // false),
        std::make_tuple(8192, 8192, 4, 2, 4 /* MB */, 2, 1, 2, 2, 2, 1, false),
        std::make_tuple(8, 2048, 2, 4, 4 /* MB */, 1, 1, 1, 1, 1, 1, false),
        // skip last bucket
        std::make_tuple(8192, 8192, 4, 1, 4 /* MB */, 1, 1, 2, 2, 1, 1, true),
        std::make_tuple(8192, 8192, 4, 1, 4 /* MB */, 2, 1, 2, 2, 2, 1, true)),
    [&](const testing::TestParamInfo<CtranTestAllToAllvDedupFixture::ParamType>&
            info) {
      const bool skipBucket = std::get<11>(info.param);
      return std::to_string(std::get<0>(info.param)) + "numblocks_" +
          std::to_string(std::get<1>(info.param)) + "count_" +
          std::to_string(std::get<2>(info.param)) + "bBuckets_" +
          std::to_string(std::get<3>(info.param)) + "buckets_" +
          std::to_string(std::get<4>(info.param)) + "MBchunkSz_" +
          std::to_string(std::get<5>(info.param)) + "sendG_" +
          std::to_string(std::get<6>(info.param)) + "sendW_" +
          std::to_string(std::get<7>(info.param)) + "fwdG_" +
          std::to_string(std::get<8>(info.param)) + "fwdW_" +
          std::to_string(std::get<9>(info.param)) + "recvG" +
          std::to_string(std::get<10>(info.param)) + "recvW" +
          (skipBucket ? "_skipBucket" : "");
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
