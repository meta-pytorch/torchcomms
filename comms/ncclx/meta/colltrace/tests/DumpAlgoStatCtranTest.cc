// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/testinfra/DistEnvironmentBase.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/colltrace/AlgoStats.h"
#include "meta/colltrace/CollTraceWrapper.h"

#include "comm.h"
#include "nccl.h"

using namespace meta::comms::ncclx;

class DumpAlgoStatCtranTest : public NcclxBaseTestFixture {
 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    setenv("NCCL_COLLTRACE", "algostat", 1);
    NcclxBaseTestFixture::SetUp();
    comm_.emplace(globalRank, numRanks, localRank, bootstrap_.get());
    ASSERT_NE(comm()->ctranComm_, nullptr);
    ASSERT_TRUE(comm()->ctranComm_->dumpAlgoStats().has_value());
  }

  ncclComm_t comm() const {
    return comm_->get();
  }

  std::optional<ncclx::test::NcclCommRAII> comm_;
};

TEST_F(DumpAlgoStatCtranTest, CtranStatsAppearInDump) {
  comm()->ctranComm_->recordAlgoStat("SendRecv", "CtranSendRecv");
  comm()->ctranComm_->recordAlgoStat("SendRecv", "CtranSendRecv");
  comm()->ctranComm_->recordAlgoStat("SendRecv", "CtranSendRecv");

  std::unordered_map<std::string, std::unordered_map<std::string, int64_t>> map;
  ncclx::colltrace::dumpAlgoStat(comm(), map);

  ASSERT_NE(map.find("SendRecv"), map.end());
  EXPECT_EQ(map["SendRecv"]["CtranSendRecv"], 3);
}

TEST_F(DumpAlgoStatCtranTest, MergeBaselineAndCtranStats) {
  ASSERT_NE(comm()->algoStats, nullptr);

  comm()->algoStats->record("AllReduce", "Baseline_Simple_Ring_8");
  comm()->algoStats->record("AllReduce", "Baseline_Simple_Ring_8");

  comm()->ctranComm_->recordAlgoStat("AllReduce", "CTRing_NVL_8");
  comm()->ctranComm_->recordAlgoStat("SendRecv", "CtranSendRecv");

  std::unordered_map<std::string, std::unordered_map<std::string, int64_t>> map;
  ncclx::colltrace::dumpAlgoStat(comm(), map);

  ASSERT_NE(map.find("AllReduce"), map.end());
  EXPECT_EQ(map["AllReduce"]["Baseline_Simple_Ring_8"], 2);
  EXPECT_EQ(map["AllReduce"]["CTRing_NVL_8"], 1);

  ASSERT_NE(map.find("SendRecv"), map.end());
  EXPECT_EQ(map["SendRecv"]["CtranSendRecv"], 1);
}

TEST_F(DumpAlgoStatCtranTest, AdditiveMergeSameAlgo) {
  ASSERT_NE(comm()->algoStats, nullptr);

  comm()->algoStats->record("AllGather", "SharedAlgo");
  comm()->algoStats->record("AllGather", "SharedAlgo");

  comm()->ctranComm_->recordAlgoStat("AllGather", "SharedAlgo");
  comm()->ctranComm_->recordAlgoStat("AllGather", "SharedAlgo");
  comm()->ctranComm_->recordAlgoStat("AllGather", "SharedAlgo");

  std::unordered_map<std::string, std::unordered_map<std::string, int64_t>> map;
  ncclx::colltrace::dumpAlgoStat(comm(), map);

  ASSERT_NE(map.find("AllGather"), map.end());
  EXPECT_EQ(map["AllGather"]["SharedAlgo"], 5);
}

TEST_F(DumpAlgoStatCtranTest, NullCtranComm) {
  ASSERT_NE(comm()->algoStats, nullptr);
  comm()->algoStats->record("ReduceScatter", "Baseline_LL_Tree_4");

  comm()->ctranComm_.reset();

  std::unordered_map<std::string, std::unordered_map<std::string, int64_t>> map;
  ncclx::colltrace::dumpAlgoStat(comm(), map);

  ASSERT_NE(map.find("ReduceScatter"), map.end());
  EXPECT_EQ(map["ReduceScatter"]["Baseline_LL_Tree_4"], 1);
}

TEST_F(DumpAlgoStatCtranTest, NullComm) {
  std::unordered_map<std::string, std::unordered_map<std::string, int64_t>> map;
  ncclx::colltrace::dumpAlgoStat(nullptr, map);
  EXPECT_TRUE(map.empty());
}

TEST_F(DumpAlgoStatCtranTest, BothNull) {
  comm()->algoStats.reset();
  comm()->ctranComm_.reset();

  std::unordered_map<std::string, std::unordered_map<std::string, int64_t>> map;
  ncclx::colltrace::dumpAlgoStat(comm(), map);
  EXPECT_TRUE(map.empty());
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
