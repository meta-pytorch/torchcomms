// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/commstate/CommStateX.h"

using ncclx::CommStateX;

struct TestPArgsParam {
  int totalNumSendBlocks;
  int blockCount;
  int blockNumRecvBuckets;
  int numRecvBuckets;
  inline std::string toString() const {
    return fmt::format(
        "totalNumSendBlocks {} blockCount {} blockNumRecvBuckets {} numRecvBuckets{}",
        totalNumSendBlocks,
        blockCount,
        blockNumRecvBuckets,
        numRecvBuckets);
  }

  inline std::string toTestName() const {
    return fmt::format(
        "{}nBlocks_{}count_{}blockNumBkts_{}nBkts",
        totalNumSendBlocks,
        blockCount,
        blockNumRecvBuckets,
        numRecvBuckets);
  }
};

struct TestTmpChunkParam {
  int numChunks;
  int chunkSizeMb;
  inline std::string toTestName() const {
    return fmt::format("{}nChks_{}chkSzMb", numChunks, chunkSizeMb);
  }
};

struct TestExecWgParam {
  int sendG;
  int sendW;
  int fwdW;
  int recvG;
  int recvW;
  int intraFwdW;
  int intraRecvW;
  inline std::string toTestName() const {
    return fmt::format(
        "send{}g{}w_fwd{}w_recv{}g{}w_intraFwd{}w_intraRecv{}w",
        sendG,
        sendW,
        fwdW,
        recvG,
        recvW,
        intraFwdW,
        intraRecvW);
  }
};

#define SET_TMPCHK_ENVRAII(param)                                          \
  EnvRAII<int> envChkSz(                                                   \
      NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE, param.chunkSizeMb * 1 << 20); \
  EnvRAII<int> envNumChks(                                                 \
      NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS, param.numChunks);

#define SET_EXEC_WG_ENVRAII(param)                                           \
  EnvRAII<int> envSendG(                                                     \
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, param.sendG); \
  EnvRAII<int> envSendW(                                                     \
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP,           \
      param.sendW);                                                          \
  EnvRAII<int> envFwdW(                                                      \
      NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS, param.fwdW);         \
  EnvRAII<int> envRecvG(                                                     \
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, param.recvG); \
  EnvRAII<int> envRecvW(                                                     \
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP,           \
      param.recvW);                                                          \
  EnvRAII<int> envIntraFwdW(                                                 \
      NCCL_CTRAN_ALLTOALLV_DEDUP_INTRA_FWD_NUM_THREAD_BLOCKS,                \
      param.intraFwdW);                                                      \
  EnvRAII<int> envIntraRecvW(                                                \
      NCCL_CTRAN_ALLTOALLV_DEDUP_INTRA_RECV_NUM_THREAD_BLOCKS,               \
      param.intraRecvW);

class AllToAllvDedupTestBase {
 public:
  AllToAllvDedupTestBase() = default;
  ~AllToAllvDedupTestBase() = default;

 protected:
  void setPersistArgs(
      const TestPArgsParam& pArgs,
      const int numSkippedBuckets = 0) {
    const auto nRanks = statex_->nRanks();
    pArgs_ = pArgs;
    // update blockNumRecvBuckets to reflect skipped buckets
    pArgs_.blockNumRecvBuckets = std::min(
        nRanks * pArgs.numRecvBuckets - numSkippedBuckets,
        pArgs.blockNumRecvBuckets);
  }

  std::vector<size_t> getRecvBlockIds(const int* recvIdx, const int count)
      const;
  // generate receive buckets for all send ranks; optional skippedBuckets
  // argument allows to customize the generation of special block distribution
  // such as no block for a set of buckets
  void genAllRankIndices(
      const int seed,
      const std::unordered_set<int>& skippedBuckets);
  // generate receive buckets for a single send rank; can be used to generate
  // special block distribution such as inter-node only
  void genRankBlockRecvBuckets(
      const int sendRank,
      const int iter,
      const std::unordered_set<int>& skippedBuckets,
      std::vector<int>& buckets) const;

  void setupIndices(
      std::vector<int>& sendIdxH,
      std::vector<int>& fwdIdxH,
      std::vector<int>& recvIdxH);
  int getTotalNumRecvBlocks() const;

  void logIndices(
      const std::vector<int>& sendIdxH,
      const std::vector<int>& fwdIdxH,
      const std::vector<int>& recvIdxH,
      const int iter);

 private:
  inline int bucketToNode(int bucket) const {
    const int nLocalRanks = statex_->nLocalRanks();
    const auto numRecvBuckets = pArgs_.numRecvBuckets;
    return bucket / (nLocalRanks * numRecvBuckets);
  }

  inline int bucketToRank(int bucket) const {
    const auto numRecvBuckets = pArgs_.numRecvBuckets;
    return bucket / numRecvBuckets;
  }

 protected:
  CommStateX* statex_{nullptr};
  void setStatex(CommStateX* statex) {
    statex_ = statex;
  }

  TestPArgsParam pArgs_;

  std::vector<std::vector<int>> allRankBlkBkts_;
};
