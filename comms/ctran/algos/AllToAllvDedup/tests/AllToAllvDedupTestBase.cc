// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAllvDedup/tests/AllToAllvDedupTestBase.h"
#include <iostream>
#include <optional>
#include <random>
#include "comms/ctran/utils/Utils.h"

namespace {

// Returns numBuckets of unique integers chosen from allowBuckets
// If seed is provided, uses deterministic bucket assignment
inline std::unordered_set<int> getUniqueBuckets(
    const int numBuckets,
    const std::vector<int>& allowBuckets,
    std::optional<unsigned int> seed = std::nullopt) {
  // Validate input
  if (numBuckets < 0) {
    throw std::invalid_argument("numBuckets cannot be negative");
  }
  const auto totalBuckets = allowBuckets.size();
  if (static_cast<size_t>(numBuckets) > totalBuckets) {
    throw std::invalid_argument("numBuckets cannot exceed allowBuckets.size()");
  }

  // Create a shuffled copy of allowBuckets for O(n) selection
  std::vector<int> shuffledBuckets = allowBuckets;
  std::mt19937 gen(seed.has_value() ? seed.value() : std::random_device{}());
  std::shuffle(shuffledBuckets.begin(), shuffledBuckets.end(), gen);

  // Take first numBuckets elements from shuffled vector
  std::unordered_set<int> chosenBuckets(
      shuffledBuckets.begin(), shuffledBuckets.begin() + numBuckets);
  return chosenBuckets;
}
} // namespace

std::vector<int> AllToAllvDedupTestBase::genAllowBuckets(
    const int totalBuckets,
    const std::unordered_set<int>& excludeBuckets) const {
  std::vector<int> buckets;
  buckets.reserve(totalBuckets - excludeBuckets.size());
  for (int i = 0; i < totalBuckets; i++) {
    if (excludeBuckets.find(i) == excludeBuckets.end()) {
      buckets.push_back(i);
    }
  }
  return buckets;
}

void AllToAllvDedupTestBase::genRankBlockRecvBuckets(
    const std::vector<int>& allowBuckets,
    int* buckets,
    std::optional<unsigned int> seed) const {
  const int blockNumRecvBuckets = pArgs_.blockNumRecvBuckets;

  for (int i = 0; i < pArgs_.totalNumSendBlocks; i++) {
    // Create unique seed per block to ensure different bucket selection
    // across blocks while maintaining reproducibility when seed is provided
    std::optional<unsigned int> blockSeed =
        seed.has_value() ? std::make_optional(seed.value() + i) : std::nullopt;
    auto chosenBuckets =
        getUniqueBuckets(blockNumRecvBuckets, allowBuckets, blockSeed);
    for (int j = 0; j < blockNumRecvBuckets; j++) {
      buckets[i * blockNumRecvBuckets + j] =
          *std::next(chosenBuckets.begin(), j);
    }
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
    const int iter) const {
  const int nNodes = statex_->nNodes();
  const int nLocalRanks = statex_->nLocalRanks();
  const int myRank = statex_->rank();
  const int nRanks = statex_->nRanks();

  std::cout << "TEST prepare iteration " << iter << ": rank " << myRank
            << " allRankBlkBkts[myRank]: "
            << ::ctran::utils::array2DToStr(
                   allRankBlkBkts_[myRank].data(),
                   pArgs_.totalNumSendBlocks,
                   pArgs_.blockNumRecvBuckets,
                   20,
                   10)
            << std::endl;
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
