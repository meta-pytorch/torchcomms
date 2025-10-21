// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ProfilerTest.hpp"

#include <gtest/gtest.h>
#include <json/value.h>
#include <filesystem>
#include <vector>

/***
 * Test class to verify tracing output  across all operations in TorchComms.
 */
TEST_F(ProfilerTest, AllTests) {
  namespace fs = std::filesystem;
  fs::path trace_file;

  {
    ProfilerGuard profilerGuard;

    // Set rank and size information
    rank_ = torchcomm_->getRank();
    num_ranks_ = torchcomm_->getSize();

    if (rank_ == 0) {
      trace_file = fs::temp_directory_path() /
          ("torchcomms_profiler_test_rank" + std::to_string(rank_) + "_" +
           std::to_string(std::time(nullptr)) + ".json");
      profilerGuard.setEnableTracingSaving(trace_file);
    }

    auto work = runAllCollectiveOperations();
    work->wait();

    torchcomm_->finalize();
  }

  if (rank_ == 0) {
    Json::Value json_value = readTraceFile(trace_file);
    std::map<std::string, std::vector<Json::Value>> events;
    sanityCheckProfilerMeta(json_value, events);

    ASSERT_EQ(events["barrier"].size(), 1);
    ASSERT_EQ(events["wait"].size(), 1);
    ASSERT_EQ(events["send"].size(), 1);
    ASSERT_EQ(events["recv"].size(), 1);
    ASSERT_EQ(events["all_reduce"].size(), 1);
    ASSERT_EQ(events["reduce"].size(), 1);
    ASSERT_EQ(events["all_gather_single"].size(), 1);
    ASSERT_EQ(events["all_gather"].size(), 1);
    ASSERT_EQ(events["gather"].size(), 1);
    ASSERT_EQ(events["reduce_scatter"].size(), 1);
    ASSERT_EQ(events["reduce_scatter_single"].size(), 1);
    ASSERT_EQ(events["scatter"].size(), 1);
    ASSERT_EQ(events["all_to_all"].size(), 1);
    ASSERT_EQ(events["all_to_all_single"].size(), 1);
    ASSERT_EQ(events["all_to_all_v_single"].size(), 1);
    ASSERT_EQ(events["broadcast"].size(), 1);

    std::filesystem::remove(trace_file);
  }
}

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
