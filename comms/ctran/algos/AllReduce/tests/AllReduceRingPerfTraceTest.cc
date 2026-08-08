// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <filesystem>
#include <fstream>
#include <iterator>

#include <folly/ScopeGuard.h>
#include <folly/json.h>
#include <folly/testing/TestUtil.h>
#include <gtest/gtest.h>

#include "comms/ctran/algos/AllReduce/AllReduceRingPerfTrace.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::allreduce::ring {
namespace {

TEST(AllReduceRingPerfTraceTest, SelectsConfiguredRanks) {
  EXPECT_TRUE(shouldTraceRank(0, ""));
  EXPECT_TRUE(shouldTraceRank(0, "0,1,48,95"));
  EXPECT_TRUE(shouldTraceRank(48, "0,1,48,95"));
  EXPECT_TRUE(shouldTraceRank(95, "0, 1, 48, 95"));
  EXPECT_FALSE(shouldTraceRank(2, "0,1,48,95"));
  EXPECT_FALSE(shouldTraceRank(7, "0,garbage,95"));
}

TEST(AllReduceRingPerfTraceTest, WritesChunkStageMetadata) {
  folly::test::TemporaryDirectory traceDir("ring_perftrace");
  setenv("NCCL_CTRAN_ENABLE_PERFTRACE", "1", 1);
  setenv("NCCL_CTRAN_PERFTRACE_DIR", traceDir.path().c_str(), 1);
  ncclCvarInit();

  {
    RingPerfTrace trace(7, 96, 512ULL * 1024 * 1024, 1024 * 1024, 32, 8, 1234);
    ChunkTraceMetadata chunk{
        .partition = 2,
        .step = 11,
        .round = 37,
        .chunkId = 5,
        .shardId = 42,
        .shardDataChunkId = 3,
        .offsetBytes = 4096,
        .bytes = 1024 * 1024,
        .phase = "reduce_scatter",
    };
    trace.startChunkStage("send_trans", 101, 8, chunk);
    trace.endChunkStage("send_trans", 101);
    trace.addChunkPoint(
        "flush_complete",
        102,
        7,
        chunk,
        {{"poll_count", "4"}, {"poll_cpu_us", "19"}});
  }

  std::filesystem::path traceFile;
  for (const auto& entry :
       std::filesystem::directory_iterator(traceDir.path().string())) {
    traceFile = entry.path();
    break;
  }
  ASSERT_FALSE(traceFile.empty());

  std::ifstream input(traceFile);
  std::string json{
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  auto events = folly::parseJson(json);
  ASSERT_EQ(events.size(), 3);

  const auto& collective = events[0];
  EXPECT_EQ(collective["name"], "AllReduceRing");
  EXPECT_EQ(collective["args"]["rank"], "7");
  EXPECT_EQ(collective["args"]["nranks"], "96");
  EXPECT_EQ(collective["args"]["message_bytes"], "536870912");
  EXPECT_EQ(collective["args"]["chunk_bytes"], "1048576");
  EXPECT_EQ(collective["args"]["num_chunks"], "32");
  EXPECT_EQ(collective["args"]["num_blocks"], "8");
  EXPECT_EQ(collective["args"]["op_count"], "1234");

  const auto stage =
      std::find_if(events.begin(), events.end(), [](const auto& event) {
        return event["name"] == "send_trans";
      });
  ASSERT_NE(stage, events.end());
  EXPECT_EQ((*stage)["tid"], 8);
  EXPECT_EQ((*stage)["args"]["seqNum"], "101");
  EXPECT_EQ((*stage)["args"]["partition"], "2");
  EXPECT_EQ((*stage)["args"]["step"], "11");
  EXPECT_EQ((*stage)["args"]["round"], "37");
  EXPECT_EQ((*stage)["args"]["chunk_id"], "5");
  EXPECT_EQ((*stage)["args"]["shard_id"], "42");
  EXPECT_EQ((*stage)["args"]["shard_chunk_id"], "3");
  EXPECT_EQ((*stage)["args"]["offset_bytes"], "4096");
  EXPECT_EQ((*stage)["args"]["bytes"], "1048576");
  EXPECT_EQ((*stage)["args"]["phase"], "reduce_scatter");

  const auto point =
      std::find_if(events.begin(), events.end(), [](const auto& event) {
        return event["name"] == "flush_complete";
      });
  ASSERT_NE(point, events.end());
  EXPECT_EQ((*point)["args"]["poll_count"], "4");
  EXPECT_EQ((*point)["args"]["poll_cpu_us"], "19");
  EXPECT_EQ((*point)["args"]["bytes"], "1048576");
}

TEST(AllReduceRingPerfTraceTest, IgnoresUnmatchedStageCompletion) {
  RingPerfTrace trace(0, 96, 256ULL * 1024 * 1024, 512 * 1024, 64, 8, 0, "0");

  EXPECT_NO_THROW(trace.endChunkStage("recv_trans", 7));
}

} // namespace
} // namespace ctran::allreduce::ring
