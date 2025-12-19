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
  ~AllToAllvDedupTestBase() {
    allRankBlkBkts_.clear();
  };

 protected:
  void setPersistArgs(
      const TestPArgsParam& pArgs,
      const int numAllowedBuckets = -1) {
    const auto nRanks = statex_->nRanks();
    pArgs_ = pArgs;
    // update blockNumRecvBuckets based on allowed buckets
    // if numAllowedBuckets is -1, use all buckets (nRanks * numRecvBuckets)
    const auto totalBuckets = nRanks * pArgs.numRecvBuckets;
    const auto effectiveAllowedBuckets =
        numAllowedBuckets == -1 ? totalBuckets : numAllowedBuckets;

    pArgs_.blockNumRecvBuckets =
        std::min(effectiveAllowedBuckets, pArgs.blockNumRecvBuckets);
  }

  std::vector<size_t> getRecvBlockIds(const int* recvIdx, const int count)
      const;

  // Generate a vector of all bucket indices from 0 to totalBuckets-1, excluding
  // buckets in excludeBuckets
  std::vector<int> genAllowBuckets(
      const int totalBuckets,
      const std::unordered_set<int>& excludeBuckets) const;

  // Generate receive buckets for a single send rank based on the provided
  // allowBuckets filter. The allowBuckets parameter specifies which buckets
  // are allowed to be assigned, and the results are stored in the buckets
  // output parameter. If seed is provided, uses deterministic bucket selection.
  void genRankBlockRecvBuckets(
      const std::vector<int>& allowBuckets,
      int* chosenBuckets,
      std::optional<unsigned int> seed = std::nullopt) const;

  void setupIndices(
      std::vector<int>& sendIdxH,
      std::vector<int>& fwdIdxH,
      std::vector<int>& recvIdxH);
  int getTotalNumRecvBlocks() const;

  void logIndices(
      const std::vector<int>& sendIdxH,
      const std::vector<int>& fwdIdxH,
      const std::vector<int>& recvIdxH,
      const int iter) const;

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
  std::vector<std::vector<int>> allRankBlkBkts_;

  void setStatex(CommStateX* statex) {
    statex_ = statex;
  }

  void setAllRankBlkBkts(const std::vector<std::vector<int>>& allRankBlkBkts) {
    allRankBlkBkts_ = allRankBlkBkts;
  };

  TestPArgsParam pArgs_;
};
