// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/json/dynamic.h>
#include <folly/json/json.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/ncclx/NcclxGlobalApi.hpp"
#include "comms/torchcomms/ncclx/tests/integration/cpp/CommDumpTest.hpp"

#define LOG_TEST(msg) XLOG(INFO) << "[Rank " << rank_ << "] " << msg

TEST_F(CommDumpTest, BasicCommDump) {
  LOG_TEST("Starting BasicCommDump test");

  auto dump = ncclx_comm_->comm_dump();

  // Verify the dump is non-empty
  EXPECT_GT(dump.size(), 0) << "comm_dump() returned empty map";

  // Verify expected metadata keys exist
  EXPECT_EQ(dump.count("commHash"), 1) << "Missing commHash key";
  EXPECT_EQ(dump.count("rank"), 1) << "Missing rank key";
  EXPECT_EQ(dump.count("nRanks"), 1) << "Missing nRanks key";

  // Verify rank and nRanks values are correct
  EXPECT_EQ(dump["rank"], std::to_string(rank_));
  EXPECT_EQ(dump["nRanks"], std::to_string(num_ranks_));

  // Verify all values are parseable as JSON
  for (const auto& [key, val] : dump) {
    EXPECT_NO_THROW(folly::parseJson(val))
        << "Value for key '" << key << "' is not valid JSON: " << val;
  }

  if (rank_ == 0) {
    LOG_TEST("comm_dump() returned " << dump.size() << " entries");
    for (const auto& [key, val] : dump) {
      LOG_TEST("  " << key << ": " << val);
    }
  }

  LOG_TEST("BasicCommDump test passed");
}

TEST_F(CommDumpTest, CommDumpAfterCollective) {
  LOG_TEST("Starting CommDumpAfterCollective test");

  // Run an all_reduce to generate collective trace data
  auto tensor =
      at::ones({64}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  torchcomm_->all_reduce(tensor, torch::comms::ReduceOp::SUM, false);

  auto dump = ncclx_comm_->comm_dump();

  // Verify dump has expected keys
  EXPECT_EQ(dump.count("commHash"), 1);
  EXPECT_EQ(dump.count("rank"), 1);
  EXPECT_EQ(dump.count("nRanks"), 1);

  // Verify CT_pastColls exists and is non-empty after running a collective
  if (dump.count("CT_pastColls") == 1) {
    auto pastColls = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_GT(pastColls.size(), 0)
        << "CT_pastColls should be non-empty after all_reduce";
    LOG_TEST("CT_pastColls has " << pastColls.size() << " entries");
  }

  LOG_TEST("CommDumpAfterCollective test passed");
}

TEST_F(CommDumpTest, CommDumpAll) {
  LOG_TEST("Starting CommDumpAll test");

  torch::comms::DefaultNcclxGlobalApi global_api;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      all_dumps;
  auto result = global_api.commDumpAll(all_dumps);
  EXPECT_EQ(result, ncclSuccess) << "ncclCommDumpAll failed";

  // At least 1 communicator should exist (created during SetUp)
  EXPECT_GE(all_dumps.size(), 1)
      << "Expected at least 1 communicator in dump_all";

  for (const auto& [comm_key, dump] : all_dumps) {
    EXPECT_GT(dump.size(), 0) << "Dump for comm " << comm_key << " is empty";
    EXPECT_EQ(dump.count("rank"), 1) << "Missing rank for comm " << comm_key;
    EXPECT_EQ(dump.count("nRanks"), 1)
        << "Missing nRanks for comm " << comm_key;
  }

  if (rank_ == 0) {
    LOG_TEST(
        "comm_dump_all() returned " << all_dumps.size() << " communicators");
    for (const auto& [comm_key, dump] : all_dumps) {
      LOG_TEST("  Comm " << comm_key << ": " << dump.size() << " entries");
    }
  }

  LOG_TEST("CommDumpAll test passed");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
