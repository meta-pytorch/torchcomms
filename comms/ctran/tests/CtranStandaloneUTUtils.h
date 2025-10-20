// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/tests/bootstrap/IntraProcessBootstrap.h"
#include "comms/ctran/utils/Abort.h"

namespace ctran::testing {

class CtranStandaloneBaseTest : public ::testing::Test {
 protected:
  const std::string kCommDesc{"ut_comm_desc"};

  std::unique_ptr<CtranComm> ctranComm{nullptr};
  int rank{0};
  int cudaDev{0};

  void setupBase();

  // by default turn on fault tolerance code path for new tests
  void initCtranComm(
      std::shared_ptr<::ctran::utils::Abort> abort =
          ctran::utils::createAbort(/*enabled=*/true));
};

class CtranStandaloneMultiRankBaseTest : public ::testing::Test {
 public:
  static constexpr size_t kBufferSize = 128 * 1024;

  struct PerRankState;
  using Work = std::function<void(PerRankState&)>;

  struct PerRankState {
    // Ideally we could use the IBootstrap interface, but it makes UT debugging
    // hard since the barriers are not named. We use the specific
    // IntraProcessBootstrap class for namedBarriers.
    ::ctran::testing::IntraProcessBootstrap* getBootstrap() {
      return reinterpret_cast<::ctran::testing::IntraProcessBootstrap*>(
          ctranComm->bootstrap_.get());
    }

    std::shared_ptr<::ctran::testing::IntraProcessBootstrap::State>
        sharedBootstrapState;
    std::unique_ptr<CtranComm> ctranComm{nullptr};
    int nRanks{1};
    int rank{0};
    int cudaDev{0};
    cudaStream_t stream{nullptr};

    // device buffer for collectives
    void* srcBuffer{nullptr};
    void* dstBuffer{nullptr};

    folly::Promise<Work> workPromise;
    folly::SemiFuture<Work> workSemiFuture{workPromise.getSemiFuture()};
  };

 protected:
  std::vector<PerRankState> perRankStates_;
  std::vector<std::thread> workers_;

  void SetUp() override;

  void startWorkers(
      int nRanks,
      const std::vector<std::shared_ptr<::ctran::utils::Abort>>& aborts);

  void run(int rank, const Work& work) {
    perRankStates_[rank].workPromise.setValue(work);
  }

  void TearDown() override;
};

} // namespace ctran::testing
