// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/commstate/CommStateX.h"

using ncclx::CommStateX;

class AllToAllvDedupTestBase {
 public:
  AllToAllvDedupTestBase() = default;
  ~AllToAllvDedupTestBase() = default;

 protected:
  void setPersistArgs(
      const int totalNumSendBlocks,
      const int blockCount,
      const int blockNumRecvBuckets,
      const int numRecvBuckets) {
    pArgs_.totalNumSendBlocks = totalNumSendBlocks;
    pArgs_.blockCount = blockCount;
    pArgs_.blockNumRecvBuckets = blockNumRecvBuckets;
    pArgs_.numRecvBuckets = numRecvBuckets;
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

  struct {
    int totalNumSendBlocks{0}; // number of tokens
    int blockCount{0}; // elements per token
    int blockNumRecvBuckets{0}; // topK
    int numRecvBuckets{0}; // number of experts per rank
    inline std::string toString() const {
      return fmt::format(
          "totalNumSendBlocks {} blockCount {} blockNumRecvBuckets {} numRecvBuckets{}",
          totalNumSendBlocks,
          blockCount,
          blockNumRecvBuckets,
          numRecvBuckets);
    }
  } pArgs_;

  std::vector<std::vector<int>> allRankBlkBkts_;
};
