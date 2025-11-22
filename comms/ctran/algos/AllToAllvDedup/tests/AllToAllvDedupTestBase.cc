// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAllvDedup/tests/AllToAllvDedupTestBase.h"
#include <iostream>
#include "comms/ctran/utils/Utils.h"

void AllToAllvDedupTestBase::genRankBlockRecvBuckets(
    const int sendRank,
    const int iter,
    const std::unordered_set<int>& skippedBuckets,
    std::vector<int>& buckets) const {
  const int nRanks = statex_->nRanks();
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
      if (skippedBuckets.contains(recvBucket)) {
        // ensure the replaced bucket is not already contained
        recvBucket = (recvRankStart + blockNumRecvBuckets) %
            (nRanks * pArgs_.numRecvBuckets);
      }
      buckets[i * blockNumRecvBuckets + j] = recvBucket;
    }
  }
}

void AllToAllvDedupTestBase::genAllRankIndices(
    const int iter,
    const std::unordered_set<int>& skippedBuckets) {
  const auto numBlocksFromRank =
      pArgs_.totalNumSendBlocks * pArgs_.blockNumRecvBuckets;
  const int nRanks = statex_->nRanks();
  allRankBlkBkts_.resize(nRanks);

  // Generating allRankBlkBkts_ for all ranks to get expected
  // prepare outputs
  for (int sendRank = 0; sendRank < nRanks; sendRank++) {
    allRankBlkBkts_[sendRank].resize(numBlocksFromRank);
    auto& buckets = allRankBlkBkts_[sendRank];
    genRankBlockRecvBuckets(sendRank, iter, skippedBuckets, buckets);
  }
}

void AllToAllvDedupTestBase::setupIndices(
    std::vector<int>& sendIdxH,
    std::vector<int>& fwdIdxH,
    std::vector<int>& recvIdxH) {
  const auto totalNumSendBlocks = pArgs_.totalNumSendBlocks;
  const auto numRecvBuckets = pArgs_.numRecvBuckets;
  const auto blockNumRecvBuckets = pArgs_.blockNumRecvBuckets;

  const int nNodes = statex_->nNodes();
  const int nLocalRanks = statex_->nLocalRanks();
  const int myLocalRank = statex_->localRank();
  const int myNode = statex_->node();
  const int myRank = statex_->rank();
  const int nRanks = statex_->nRanks();

  sendIdxH.resize(totalNumSendBlocks * nNodes);
  std::fill(sendIdxH.begin(), sendIdxH.end(), -1);
  fwdIdxH.resize(totalNumSendBlocks * nNodes * nLocalRanks);
  std::fill(fwdIdxH.begin(), fwdIdxH.end(), -1);
  recvIdxH.resize(totalNumSendBlocks * nRanks * numRecvBuckets);
  std::fill(recvIdxH.begin(), recvIdxH.end(), -1);

  // setup sendIdx (nNodes * totalNumSendBlocks)
  for (int n = 0; n < nNodes; n++) {
    const auto idxOffset = n * totalNumSendBlocks;
    int idx = 0;
    for (int b = 0; b < totalNumSendBlocks; b++) {
      const auto bIdx = b * blockNumRecvBuckets;
      for (int r = 0; r < blockNumRecvBuckets; r++) {
        const auto recvNode = bucketToNode(allRankBlkBkts_[myRank][bIdx + r]);
        if (recvNode == n) {
          sendIdxH[idxOffset + b] = idx++;
          break;
        }
      }
    }
  }

  // setup fwdIdx (nLocalRanks * nNodes * totalNumSendBlocks)
  for (int n = 0; n < nNodes; n++) {
    const auto railSendRank = n * nLocalRanks + myLocalRank;
    for (int r = 0; r < nLocalRanks; r++) {
      const auto localRecvRank = myNode * nLocalRanks + r;
      const auto idxOffset =
          r * totalNumSendBlocks * nNodes + n * totalNumSendBlocks;
      int idx = 0;
      for (int b = 0; b < totalNumSendBlocks; b++) {
        const auto bIdx = b * blockNumRecvBuckets;
        for (int blk = 0; blk < blockNumRecvBuckets; blk++) {
          const auto expRecvRank =
              bucketToRank(allRankBlkBkts_[railSendRank][bIdx + blk]);
          if (expRecvRank == localRecvRank) {
            fwdIdxH[idxOffset + b] = idx++;
            break;
          }
        }
      }
    }
  }

  // setup recvIdx (numRecvBuckets * nRanks * totalNumSendBlocks)
  for (int bkt = 0; bkt < numRecvBuckets; bkt++) {
    for (int r = 0; r < nRanks; r++) {
      const auto idxOffset = (bkt * nRanks + r) * totalNumSendBlocks;
      int idx = 0;
      for (int b = 0; b < totalNumSendBlocks; b++) {
        for (int i = 0; i < blockNumRecvBuckets; i++) {
          const auto dstBkt = allRankBlkBkts_[r][b * blockNumRecvBuckets + i];
          if (dstBkt == myRank * numRecvBuckets + bkt) {
            recvIdxH[idxOffset + b] = idx++;
            break;
          }
        }
      }
    }
  }
}

void AllToAllvDedupTestBase::logIndices(
    const std::vector<int>& sendIdxH,
    const std::vector<int>& fwdIdxH,
    const std::vector<int>& recvIdxH,
    const int iter) {
  const int nNodes = statex_->nNodes();
  const int nLocalRanks = statex_->nLocalRanks();
  const int myRank = statex_->rank();
  const int nRanks = statex_->nRanks();

  std::cout << "TEST prepare iteration " << iter << ": rank " << myRank
            << " sendIdx: "
            << ::ctran::utils::array2DToStr(
                   sendIdxH.data(), nNodes, pArgs_.totalNumSendBlocks, 20, 20)
            << std::endl;
  for (int r = 0; r < nLocalRanks; r++) {
    const auto offset = pArgs_.totalNumSendBlocks * nNodes * r;
    std::cout << "TEST prepare iteration " << iter << ": rank " << myRank
              << fmt::format(
                     " fwdIdx[localRank {}][sendNode 0:{}]: ", r, nNodes)
              << ::ctran::utils::array2DToStr(
                     fwdIdxH.data() + offset,
                     nNodes,
                     pArgs_.totalNumSendBlocks,
                     20,
                     20)
              << std::endl;
  }
  for (int bkt = 0; bkt < pArgs_.numRecvBuckets; bkt++) {
    std::cout << "TEST prepare iteration " << iter << ": rank " << myRank
              << fmt::format(
                     " recvIdx[bucket {}][sendRank 0:{}]: ", bkt, nRanks)
              << ::ctran::utils::array2DToStr(
                     recvIdxH.data() + bkt * nRanks * pArgs_.totalNumSendBlocks,
                     nRanks,
                     pArgs_.totalNumSendBlocks,
                     10,
                     10)
              << std::endl;
  }
}

int AllToAllvDedupTestBase::getTotalNumRecvBlocks() const {
  const auto numRecvBuckets = pArgs_.numRecvBuckets;
  const auto blockNumRecvBuckets = pArgs_.blockNumRecvBuckets;
  const auto totalNumSendBlocks = pArgs_.totalNumSendBlocks;
  const auto myRank = statex_->rank();
  const auto nRanks = statex_->nRanks();
  int total = 0;
  for (int bkt = 0; bkt < numRecvBuckets; bkt++) {
    for (int r = 0; r < nRanks; r++) {
      for (int b = 0; b < totalNumSendBlocks; b++) {
        for (int i = 0; i < blockNumRecvBuckets; i++) {
          const auto dstBkt = allRankBlkBkts_[r][b * blockNumRecvBuckets + i];
          if (dstBkt == myRank * numRecvBuckets + bkt) {
            total++;
            break;
          }
        }
      }
    }
  }
  return total;
}

std::vector<size_t> AllToAllvDedupTestBase::getRecvBlockIds(
    const int* recvIdx,
    const int count) const {
  std::vector<size_t> recvBlockIds;
  recvBlockIds.reserve(count);
  for (int i = 0; i < count; i++) {
    if (recvIdx[i] != -1) {
      // all non -1 recvIdx[i] are contiguous
      recvBlockIds.push_back(i);
    }
  }
  return recvBlockIds;
}
