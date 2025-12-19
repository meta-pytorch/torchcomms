// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/tests/bootstrap/IntraProcessBootstrap.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::testing {

void IntraProcessBootstrap::barrierNamed(
    int rank,
    int nRanks,
    int timeoutSeconds,
    const std::string& name) {
  CLOGF(INFO, "rank [{}/{}] barrier '{}' enter", rank, nRanks, name);
  // Each thread gets its own sense
  bool local_sense = !state_->sense.load();
  // Atomically increment the count
  int arrived = state_->nArrivals.fetch_add(1);
  if (arrived == nRanks - 1) {
    // Last thread to arrive: reset count and flip sense
    state_->nArrivals.store(0);
    state_->sense.store(local_sense);
  } else {
    const auto timeout = std::chrono::seconds(timeoutSeconds);
    const auto startTs = std::chrono::high_resolution_clock::now();
    // Spin-wait for sense to change
    while (state_->sense.load() != local_sense) {
      auto now = std::chrono::high_resolution_clock::now();
      if (now - startTs > timeout) {
        throw ctran::utils::Exception(
            fmt::format(
                "rank [{}/{}] barrier '{}' timeout after {}s",
                rank,
                nRanks,
                name,
                timeoutSeconds),
            commInternalError);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  CLOGF(INFO, "rank [{}/{}] barrier '{}' leave", rank, nRanks, name);
}

folly::SemiFuture<int>
IntraProcessBootstrap::allGather(void* buf, int len, int rank, int nRanks) {
  CHECK(len <= kMaxPerRankAllGatherSize);

  char* src = reinterpret_cast<char*>(buf);
  char* dst = reinterpret_cast<char*>(state_->tmpBuf.data());
  std::memcpy(dst + rank * len, src + rank * len, len);

  // barrier all ranks
  barrierNamed(rank, nRanks, /*timeoutSeconds=*/3, "AG");

  std::memcpy(buf, dst, len * nRanks);

  // barrier all ranks
  barrierNamed(rank, nRanks, /*timeoutSeconds=*/3, "AG-complete");

  return commSuccess;
}
folly::SemiFuture<int> IntraProcessBootstrap::allGatherIntraNode(
    void* buf,
    int len,
    int localRank,
    int localNRanks,
    std::vector<int> localRankToCommRank) {
  if (localNRanks == 1) {
    // Topo::nolocal
    return commSuccess;
  }

  CHECK(len <= kMaxPerRankAllGatherSize);

  // Topo::system
  char* src = reinterpret_cast<char*>(buf);
  char* dst = reinterpret_cast<char*>(state_->tmpBuf.data());
  std::memcpy(dst + localRank * len, src + localRank * len, len);

  barrierNamed(
      localRank,
      localNRanks,
      /*timeoutSeconds=*/3,
      "AGIntraNode");

  std::memcpy(buf, dst, len * localNRanks);

  barrierNamed(
      localRank,
      localNRanks,
      /*timeoutSeconds=*/3,
      "AGIntraNode-complete");

  return commSuccess;
}
folly::SemiFuture<int> IntraProcessBootstrap::barrier(int rank, int nRanks) {
  // global barrier, use the last barrier.
  barrierNamed(rank, nRanks, /*timeoutSeconds=*/3, "barrier");
  return commSuccess;
}
folly::SemiFuture<int> IntraProcessBootstrap::barrierIntraNode(
    int localRank,
    int localNRanks,
    std::vector<int> localRankToCommRank) {
  if (localNRanks == 1) {
    // Topo::nolocal
    return commSuccess;
  }

  // Topo::system
  barrierNamed(
      localRank,
      localNRanks,
      /*timeoutSeconds=*/3,
      "barrierIntraNode");
  return commSuccess;
}
folly::SemiFuture<int>
IntraProcessBootstrap::send(void* buf, int len, int peer, int tag) {
  throw ctran::utils::Exception(
      "IntraProcessBootstrap::send not implemented", commInvalidUsage);
}
folly::SemiFuture<int>
IntraProcessBootstrap::recv(void* buf, int len, int peer, int tag) {
  throw ctran::utils::Exception(
      "IntraProcessBootstrap::recv not implemented", commInvalidUsage);
}

} // namespace ctran::testing
