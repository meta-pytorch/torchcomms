// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/tests/integration/cpp/ProfilerTest.hpp"

#include <gtest/gtest.h>
#include <json/value.h>
#include <filesystem>
#include <vector>

// ncclx-specific validation function
void validateNcclxProfilerEvents(
    const std::map<std::string, std::vector<Json::Value>>& events) {
  ASSERT_EQ(events.at("barrier").size(), 1);
  ASSERT_EQ(events.at("wait").size(), 1);
  ASSERT_EQ(events.at("send").size(), 1);
  ASSERT_EQ(events.at("recv").size(), 1);
  ASSERT_EQ(events.at("all_reduce").size(), 1);
  ASSERT_EQ(events.at("reduce").size(), 1);
  ASSERT_EQ(events.at("all_gather_single").size(), 1);
  ASSERT_EQ(events.at("all_gather").size(), 1);
  ASSERT_EQ(events.at("gather").size(), 1);
  ASSERT_EQ(events.at("reduce_scatter").size(), 1);
  ASSERT_EQ(events.at("reduce_scatter_single").size(), 1);
  ASSERT_EQ(events.at("scatter").size(), 1);
  ASSERT_EQ(events.at("all_to_all").size(), 1);
  ASSERT_EQ(events.at("all_to_all_single").size(), 1);
  ASSERT_EQ(events.at("all_to_all_v_single").size(), 1);
  ASSERT_EQ(events.at("broadcast").size(), 1);
}

class ProfilerNcclxTest : public ProfilerTest {
 public:
  ProfilerNcclxTest()
      : ProfilerTest(
            "ncclx",
            c10::DeviceType::CUDA,
            validateNcclxProfilerEvents) {}
};

TEST_F(ProfilerNcclxTest, AllTests) {
  namespace fs = std::filesystem;
  fs::path trace_file;

  {
    ProfilerGuard profilerGuard;

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
    sanityCheckProfilerMeta(json_value, events, "comms_test_name");

    // Call the validation function
    validation_func_(events);

    std::filesystem::remove(trace_file);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
