// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranCommSplit.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <folly/String.h>

#include "comms/ctran/bootstrap/ICtranBootstrap.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/Utils.h"

namespace {

class SplitBootstrapAdapter final : public meta::comms::ICtranBootstrap {
 public:
  SplitBootstrapAdapter(
      meta::comms::ICtranBootstrap* bootstrap,
      std::vector<int> childRankToParentRank)
      : bootstrap_(bootstrap),
        childRankToParentRank_(std::move(childRankToParentRank)) {}

  folly::SemiFuture<int> allGather(void* buf, int len, int rank, int nranks)
      override {
    return bootstrap_->allGatherNvlDomain(
        buf, len, rank, nranks, childRankToParentRank_);
  }

  folly::SemiFuture<int> barrier(int rank, int nranks) override {
    return bootstrap_->barrierNvlDomain(rank, nranks, childRankToParentRank_);
  }

  folly::SemiFuture<int> allGatherNvlDomain(
      void* buf,
      int len,
      int nvlLocalRank,
      int nvlNranks,
      std::vector<int> nvlRankToCommRank) override {
    return bootstrap_->allGatherNvlDomain(
        buf,
        len,
        nvlLocalRank,
        nvlNranks,
        mapChildRanks(std::move(nvlRankToCommRank)));
  }

  folly::SemiFuture<int> barrierNvlDomain(
      int nvlLocalRank,
      int nvlNranks,
      std::vector<int> nvlRankToCommRank) override {
    return bootstrap_->barrierNvlDomain(
        nvlLocalRank, nvlNranks, mapChildRanks(std::move(nvlRankToCommRank)));
  }

  folly::SemiFuture<int> send(void* buf, int len, int peer, int tag) override {
    return bootstrap_->send(buf, len, childRankToParentRank_.at(peer), tag);
  }

  folly::SemiFuture<int> recv(void* buf, int len, int peer, int tag) override {
    return bootstrap_->recv(buf, len, childRankToParentRank_.at(peer), tag);
  }

 private:
  std::vector<int> mapChildRanks(std::vector<int> childRanks) const {
    for (auto& rank : childRanks) {
      rank = childRankToParentRank_.at(rank);
    }
    return childRanks;
  }

  meta::comms::ICtranBootstrap* bootstrap_{nullptr};
  std::vector<int> childRankToParentRank_;
};

uint64_t makeSplitCommHash(
    const ncclx::CommStateX* statex,
    const std::vector<int>& parentRanks,
    const std::string& commDesc) {
  const std::string hashInput = folly::to<std::string>(
      statex->commHash(), ":", commDesc, ":", folly::join(",", parentRanks));
  return ctran::utils::getHash(
      hashInput.data(), static_cast<int>(hashInput.size()));
}

commResult_t buildSplitShareChild(
    CtranComm* parent,
    std::vector<int> parentRanks,
    const std::string& descSuffix,
    std::shared_ptr<CtranComm>* childOut) {
  if (parent == nullptr || parent->statex_ == nullptr ||
      parent->bootstrap_ == nullptr || childOut == nullptr) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Cannot splitShare CtranComm without parent statex/bootstrap or child output");
  }
  if (parent->isSplitShare()) {
    FB_ERRORRETURN(
        commInvalidArgument, "Cannot splitShare a splitShare CtranComm");
  }
  if (parentRanks.empty()) {
    FB_ERRORRETURN(
        commInvalidArgument, "Cannot splitShare CtranComm with no ranks");
  }

  auto* parentStatex = parent->statex_.get();
  const int parentRank = parentStatex->rank();
  int childRank = -1;
  for (int r = 0; r < static_cast<int>(parentRanks.size()); ++r) {
    if (parentRanks[r] == parentRank) {
      childRank = r;
      break;
    }
  }
  if (childRank == -1) {
    FB_ERRORRETURN(
        commInternalError,
        "splitShare ranks do not contain this parent rank {}",
        parentRank);
  }

  const auto& parentTopologies = parentStatex->rankTopologiesRef();
  std::vector<ncclx::RankTopology> childTopologies;
  childTopologies.reserve(parentRanks.size());
  for (int r = 0; r < static_cast<int>(parentRanks.size()); ++r) {
    auto topology = parentTopologies.at(parentRanks[r]);
    topology.rank = r;
    childTopologies.push_back(topology);
  }

  // Compose world ranks via parent's rank-to-world map so child->gRank() is
  // honest. Falls back to parent rank when parent has no world-rank map set
  // (lazy/non-eager init).
  std::vector<int> worldRanks(parentRanks.size());
  const bool parentHasWorldRanks =
      !parentStatex->commRanksToWorldRanksRef().empty();
  for (int r = 0; r < static_cast<int>(parentRanks.size()); ++r) {
    worldRanks[r] = parentHasWorldRanks ? parentStatex->gRank(parentRanks[r])
                                        : parentRanks[r];
  }

  const std::string commDesc = parentStatex->commDesc() + descSuffix;
  const uint64_t commHash =
      makeSplitCommHash(parentStatex, parentRanks, commDesc);

  auto child = std::make_shared<CtranComm>(parent->getAbort(), parent->config_);
  child->config_.commDesc = commDesc;
  child->logMetaData_ = parent->logMetaData_;
  child->logMetaData_.commHash = commHash;
  child->logMetaData_.commDesc = commDesc;
  child->logMetaData_.rank = childRank;
  child->logMetaData_.nRanks = parentRanks.size();
  child->opCount_ = parent->opCount_;
  child->runtimeConn_ = parent->runtimeConn_;
  child->colltraceNew_ = parent->colltraceNew_;
  child->memCache_ = parent->memCache_;
  child->isSplitShare_ = true;
  child->resourceComm_ = parent;
  child->parentRanks_ = parentRanks;
  child->bootstrap_ = std::make_unique<SplitBootstrapAdapter>(
      parent->bootstrap_.get(), parentRanks);
  child->statex_ = std::make_unique<ncclx::CommStateX>(
      childRank,
      static_cast<int>(parentRanks.size()),
      parentStatex->cudaDev(),
      parentStatex->cudaArch(),
      parentStatex->busId(),
      commHash,
      std::move(childTopologies),
      std::move(worldRanks),
      commDesc,
      false /* noLocal */,
      static_cast<int>(parentRanks.size()) /* vCliqueSize */);

  *childOut = std::move(child);
  return commSuccess;
}

} // namespace

commResult_t ctranCommSplitShare(
    CtranComm* parent,
    int color,
    int key,
    std::shared_ptr<CtranComm>* childOut) {
  if (parent == nullptr || parent->statex_ == nullptr ||
      parent->bootstrap_ == nullptr) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Cannot splitShare CtranComm without parent statex/bootstrap");
  }

  struct Entry {
    int color;
    int key;
  };

  const int parentNRanks = parent->statex_->nRanks();
  const int parentRank = parent->statex_->rank();
  std::vector<Entry> entries(parentNRanks);
  entries[parentRank] = Entry{color, key};
  FB_COMMCHECK(
      static_cast<commResult_t>(
          parent->bootstrap_
              ->allGather(
                  entries.data(), sizeof(Entry), parentRank, parentNRanks)
              .get()));

  // Ranks with matching color, sorted by (key, parent_rank).
  std::vector<std::pair<int, int>> matches;
  matches.reserve(parentNRanks);
  for (int r = 0; r < parentNRanks; ++r) {
    if (entries[r].color == color) {
      matches.emplace_back(entries[r].key, r);
    }
  }
  std::sort(matches.begin(), matches.end());

  std::vector<int> parentRanks;
  parentRanks.reserve(matches.size());
  for (const auto& m : matches) {
    parentRanks.push_back(m.second);
  }

  return buildSplitShareChild(
      parent, std::move(parentRanks), "/splitShare", childOut);
}

commResult_t ctranCommSplitLocalNvl(
    CtranComm* parent,
    std::shared_ptr<CtranComm>* childOut) {
  if (parent == nullptr || parent->statex_ == nullptr) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "Cannot splitShare local NVL CtranComm without parent statex");
  }
  return buildSplitShareChild(
      parent, parent->statex_->localRankToRanks(), "/local-nvl", childOut);
}
