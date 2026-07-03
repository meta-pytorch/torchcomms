// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include <folly/init/Init.h>

#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual
#include "meta/tuner/MetaTuner.h"

namespace {

using ncclx::tuner::kMetaTuner;
using ncclx::tuner::metaTunerEnabled;

// Mirrors the cost-table shape NCCL core hands to the tuner: a row per
// algorithm, a column per protocol. NCCL_ALGO_PROTO_IGNORE (-1.0) marks
// unsupported algo/proto combos.
class CostTable {
 public:
  CostTable() {
    for (auto& row : table_) {
      row.fill(1.0F);
    }
  }

  void setIgnore(const int algo, const int proto) {
    table_[algo][proto] = NCCL_ALGO_PROTO_IGNORE;
  }

  float at(const int algo, const int proto) const {
    return table_[algo][proto];
  }

  float** asCallbackArg() {
    return reinterpret_cast<float**>(table_.data());
  }

 private:
  std::array<std::array<float, NCCL_NUM_PROTOCOLS>, NCCL_NUM_ALGORITHMS> table_;
};

// RAII helper that writes a temp config file and points NCCLX_TUNER_CONFIG_FILE
// at it via EnvRAII, restoring the cvar and removing the file on teardown.
class TunerConfigFile {
 public:
  TunerConfigFile(const std::string& suffix, const std::string& contents)
      : path_(writeTempFile(suffix, contents)),
        envGuard_(NCCLX_TUNER_CONFIG_FILE, path_) {}

  ~TunerConfigFile() {
    std::remove(path_.c_str());
  }

  const std::string& path() const {
    return path_;
  }

 private:
  static std::string writeTempFile(
      const std::string& suffix,
      const std::string& contents) {
    std::string path = std::string(std::tmpnam(nullptr)) + suffix;
    std::ofstream out(path);
    out << contents;
    return path;
  }

  std::string path_;
  EnvRAII<std::string> envGuard_;
};

// Drives the v6 init callback for a given topology and returns its result,
// writing the opaque context (nullptr on failure) through `context`. Caller
// must finalize a successfully-created context.
ncclResult_t
tryInitTuner(void** context, const size_t nRanks, const size_t nNodes) {
  return kMetaTuner.init(
      context,
      /* commId */ 0,
      nRanks,
      nNodes,
      /* logFunction */ nullptr,
      /* nvlDomainInfo */ nullptr,
      /* constants */ nullptr);
}

// Initializes the tuner via the v6 init callback for a given topology and
// returns the opaque context, asserting init succeeds. Caller must finalize.
void* initTuner(const size_t nRanks, const size_t nNodes) {
  void* context = nullptr;
  EXPECT_EQ(tryInitTuner(&context, nRanks, nNodes), ncclSuccess);
  EXPECT_NE(context, nullptr);
  return context;
}

// Resolves the runtime path of a checked-in example config. The BUCK rule maps
// each example file to an env var via env = {VAR: "$(location :target)"}, so
// the path is materialized for us at test time.
std::string exampleConfigPath(const char* const envVar) {
  const char* const path = std::getenv(envVar);
  EXPECT_NE(path, nullptr) << "resource env var " << envVar << " is unset";
  return path == nullptr ? std::string{} : std::string{path};
}

class MetaTunerTest : public ::testing::Test {};

// Case 8: cvar empty -> tuner disabled.
TEST_F(MetaTunerTest, DisabledWhenCvarEmpty) {
  const EnvRAII<std::string> envGuard(NCCLX_TUNER_CONFIG_FILE, std::string{});
  EXPECT_FALSE(metaTunerEnabled());
}

TEST_F(MetaTunerTest, EnabledWhenCvarSet) {
  const TunerConfigFile config(
      ".csv", "allreduce,[0,1024],ring,ll128,-1,*,*\n");
  EXPECT_TRUE(metaTunerEnabled());
}

// Case 2: size-range match sets the targeted entry to 0.0, others unchanged;
// out-of-range leaves the whole table unchanged. nRanks=1 so bytesPerRank ==
// nBytes here, isolating the Int64Range match from the per-rank division (the
// division itself is covered by BytesPerRankFilter).
TEST_F(MetaTunerTest, SizeRangeMatch) {
  const TunerConfigFile config(
      ".csv", "allreduce,[1024,4096],ring,ll128,-1,*,*\n");
  void* context = initTuner(/* nRanks */ 1, /* nNodes */ 1);

  CostTable inRange;
  int nChannels = 1;
  EXPECT_EQ(
      kMetaTuner.getCollInfo(
          context,
          ncclFuncAllReduce,
          /* nBytes */ 2048,
          /* numPipeOps */ 1,
          inRange.asCallbackArg(),
          NCCL_NUM_ALGORITHMS,
          NCCL_NUM_PROTOCOLS,
          /* regBuff */ 0,
          &nChannels),
      ncclSuccess);
  EXPECT_EQ(inRange.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), 0.0F);
  EXPECT_EQ(inRange.at(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE), 1.0F);

  CostTable outOfRange;
  EXPECT_EQ(
      kMetaTuner.getCollInfo(
          context,
          ncclFuncAllReduce,
          /* nBytes */ 8192,
          /* numPipeOps */ 1,
          outOfRange.asCallbackArg(),
          NCCL_NUM_ALGORITHMS,
          NCCL_NUM_PROTOCOLS,
          /* regBuff */ 0,
          &nChannels),
      ncclSuccess);
  EXPECT_EQ(outOfRange.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), 1.0F);

  kMetaTuner.finalize(context);
}

// Case 3: topology filter (nNodes/nLocalRanks) gates the match. The context
// derives nLocalRanks = nRanks / nNodes, so a rule keyed nNodes=2,nLocalRanks=8
// matches a 16-rank/2-node comm (local 8) but not an 8-rank/1-node comm (local
// 8 but wrong nNodes).
TEST_F(MetaTunerTest, TopologyFilter) {
  const TunerConfigFile config(
      ".csv", "allreduce,[0,4096],ring,ll128,-1,2,8\n");

  void* matching = initTuner(/* nRanks */ 16, /* nNodes */ 2);
  CostTable matchTable;
  int nChannels = 1;
  EXPECT_EQ(
      kMetaTuner.getCollInfo(
          matching,
          ncclFuncAllReduce,
          1024,
          1,
          matchTable.asCallbackArg(),
          NCCL_NUM_ALGORITHMS,
          NCCL_NUM_PROTOCOLS,
          0,
          &nChannels),
      ncclSuccess);
  EXPECT_EQ(matchTable.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), 0.0F);
  kMetaTuner.finalize(matching);

  void* nonMatching = initTuner(/* nRanks */ 8, /* nNodes */ 1);
  CostTable noMatchTable;
  EXPECT_EQ(
      kMetaTuner.getCollInfo(
          nonMatching,
          ncclFuncAllReduce,
          1024,
          1,
          noMatchTable.asCallbackArg(),
          NCCL_NUM_ALGORITHMS,
          NCCL_NUM_PROTOCOLS,
          0,
          &nChannels),
      ncclSuccess);
  EXPECT_EQ(noMatchTable.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), 1.0F);
  kMetaTuner.finalize(nonMatching);
}

// Case 3b: nLocalRanks gating, isolated from nNodes. A rule keyed only on
// nLocalRanks=4 (nNodes wildcard) matches a 16-rank/4-node comm (local 4) but
// NOT a 16-rank/2-node comm (local 8) -- proving the match keys on ranks per
// node, not total ranks (both comms have nRanks=16).
TEST_F(MetaTunerTest, LocalRanksFilter) {
  const TunerConfigFile config(
      ".csv", "allreduce,[0,4096],ring,ll128,-1,*,4\n");

  void* matching = initTuner(/* nRanks */ 16, /* nNodes */ 4);
  CostTable matchTable;
  int nChannels = 1;
  kMetaTuner.getCollInfo(
      matching,
      ncclFuncAllReduce,
      1024,
      1,
      matchTable.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      0,
      &nChannels);
  EXPECT_EQ(matchTable.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), 0.0F);
  kMetaTuner.finalize(matching);

  void* nonMatching = initTuner(/* nRanks */ 16, /* nNodes */ 2);
  CostTable noMatchTable;
  kMetaTuner.getCollInfo(
      nonMatching,
      ncclFuncAllReduce,
      1024,
      1,
      noMatchTable.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      0,
      &nChannels);
  EXPECT_EQ(noMatchTable.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), 1.0F);
  kMetaTuner.finalize(nonMatching);
}

// Helper: drive getCollInfo once and report whether the rule matched (the
// targeted ring/ll128 entry was forced to 0.0).
bool ruleMatched(
    void* context,
    const ncclFunc_t collType,
    const size_t nBytes) {
  CostTable table;
  int nChannels = 1;
  kMetaTuner.getCollInfo(
      context,
      collType,
      nBytes,
      /* numPipeOps */ 1,
      table.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      /* regBuff */ 0,
      &nChannels);
  return table.at(NCCL_ALGO_RING, NCCL_PROTO_LL128) == 0.0F;
}

// Int64Range bytesPerRank matching: a half-open interval [1024,4096) matches
// its lower endpoint, an interior value, and excludes its upper endpoint and
// below-range. nRanks=1 so bytesPerRank == nBytes (per-rank division covered by
// BytesPerRankFilter).
TEST_F(MetaTunerTest, BytesIntervalEndpoints) {
  const TunerConfigFile config(
      ".csv", "allreduce,[1024,4096),ring,ll128,-1,*,*\n");
  void* context = initTuner(/* nRanks */ 1, /* nNodes */ 1);

  EXPECT_FALSE(ruleMatched(context, ncclFuncAllReduce, 1023));
  EXPECT_TRUE(ruleMatched(context, ncclFuncAllReduce, 1024));
  EXPECT_TRUE(ruleMatched(context, ncclFuncAllReduce, 2048));
  EXPECT_FALSE(ruleMatched(context, ncclFuncAllReduce, 4096));

  kMetaTuner.finalize(context);
}

// Open-ended bytesPerRank interval (1024,) matches everything strictly above
// 1024. nRanks=1 so bytesPerRank == nBytes (per-rank division covered by
// BytesPerRankFilter).
TEST_F(MetaTunerTest, BytesOpenEndedAbove) {
  const TunerConfigFile config(".csv", "allreduce,(1024,),ring,ll128,-1,*,*\n");
  void* context = initTuner(/* nRanks */ 1, /* nNodes */ 1);

  EXPECT_FALSE(ruleMatched(context, ncclFuncAllReduce, 1024));
  EXPECT_TRUE(ruleMatched(context, ncclFuncAllReduce, 1025));
  EXPECT_TRUE(ruleMatched(context, ncclFuncAllReduce, 1073741824));

  kMetaTuner.finalize(context);
}

// Wildcard bytesPerRank (*) matches any size.
TEST_F(MetaTunerTest, BytesWildcardMatchesAll) {
  const TunerConfigFile config(".csv", "allreduce,*,ring,ll128,-1,*,*\n");
  void* context = initTuner(/* nRanks */ 8, /* nNodes */ 1);

  EXPECT_TRUE(ruleMatched(context, ncclFuncAllReduce, 0));
  EXPECT_TRUE(ruleMatched(context, ncclFuncAllReduce, 1073741824));

  kMetaTuner.finalize(context);
}

// Range-keyed nNodes: a rule keyed nNodes=(1,) (more than one node) matches a
// 2-node comm but not a 1-node comm; nLocalRanks/bytes left wildcard.
TEST_F(MetaTunerTest, NodesRangeKeyed) {
  const TunerConfigFile config(".csv", "allreduce,*,ring,ll128,-1,(1,),*\n");

  void* multiNode = initTuner(/* nRanks */ 16, /* nNodes */ 2);
  EXPECT_TRUE(ruleMatched(multiNode, ncclFuncAllReduce, 1024));
  kMetaTuner.finalize(multiNode);

  void* singleNode = initTuner(/* nRanks */ 8, /* nNodes */ 1);
  EXPECT_FALSE(ruleMatched(singleNode, ncclFuncAllReduce, 1024));
  kMetaTuner.finalize(singleNode);
}

// Range-keyed nLocalRanks: a rule keyed nLocalRanks=[4,8] matches a comm with 8
// ranks per node (16 ranks / 2 nodes) but not one with 2 ranks per node.
TEST_F(MetaTunerTest, LocalRanksRangeKeyed) {
  const TunerConfigFile config(".csv", "allreduce,*,ring,ll128,-1,*,[4,8]\n");

  void* inRange = initTuner(/* nRanks */ 16, /* nNodes */ 2);
  EXPECT_TRUE(ruleMatched(inRange, ncclFuncAllReduce, 1024));
  kMetaTuner.finalize(inRange);

  void* outOfRange = initTuner(/* nRanks */ 8, /* nNodes */ 4);
  EXPECT_FALSE(ruleMatched(outOfRange, ncclFuncAllReduce, 1024));
  kMetaTuner.finalize(outOfRange);
}

// bytesPerRank filter: a single rule keyed on bytesPerRank=[8MiB,16MiB] (every
// other field wildcard) matches a collective iff nBytes/nRanks is in range --
// across DIFFERENT topologies that share a per-rank size. nRanks = nNodes *
// nLocalRanks, so one per-rank rule spans rank counts where a total-bytes rule
// would need one rule per topology (the whole point of bytesPerRank).
TEST_F(MetaTunerTest, BytesPerRankFilter) {
  constexpr size_t kMiB = 1024 * 1024;
  const TunerConfigFile config(
      ".csv", "allgather,[8388608,16777216],ring,ll128,-1,*,*\n");

  // Topology A: 8 ranks / 1 node -> nRanks = 8.
  void* topoA = initTuner(/* nRanks */ 8, /* nNodes */ 1);
  // Topology B: 16 ranks / 2 nodes -> nRanks = 16.
  void* topoB = initTuner(/* nRanks */ 16, /* nNodes */ 2);

  // Same per-rank size (10 MiB, in [8,16]) on both topologies -> both match,
  // even though the total bytes differ -- one rule spans both rank counts.
  EXPECT_TRUE(ruleMatched(topoA, ncclFuncAllGather, 10 * kMiB * 8));
  EXPECT_TRUE(ruleMatched(topoB, ncclFuncAllGather, 10 * kMiB * 16));

  // Endpoint inclusivity: per-rank == 8 MiB (lower bound) matches; below
  // misses.
  EXPECT_TRUE(ruleMatched(topoA, ncclFuncAllGather, 8 * kMiB * 8));
  EXPECT_FALSE(ruleMatched(topoA, ncclFuncAllGather, 4 * kMiB * 8));
  // Above the upper bound misses too (32 MiB/rank).
  EXPECT_FALSE(ruleMatched(topoB, ncclFuncAllGather, 32 * kMiB * 16));

  // The discriminating case: ONE total size (80 MiB) maps to 10 MiB/rank on
  // topoA (8 ranks, in range -> match) but 5 MiB/rank on topoB (16 ranks, below
  // range -> no match). A total-bytes rule could not tell these apart.
  EXPECT_TRUE(ruleMatched(topoA, ncclFuncAllGather, 80 * kMiB));
  EXPECT_FALSE(ruleMatched(topoB, ncclFuncAllGather, 80 * kMiB));

  kMetaTuner.finalize(topoA);
  kMetaTuner.finalize(topoB);
}

// Per-rank rules generated with nLocalRanks=(1,) must NOT match nolocal /
// IB-only comms (nLocalRanks==1), which use PAT/Simple rather than ring/ll128,
// even when the per-rank size is in range. Guards the gen --per-rank nolocal
// carve-out: a wildcard nLocalRanks would otherwise force ll128 on nolocal.
TEST_F(MetaTunerTest, PerRankExcludesNolocal) {
  constexpr size_t kMiB = 1024 * 1024;
  const TunerConfigFile config(
      ".csv", "allgather,[8388608,16777216],ring,ll128,-1,(1,),(1,)\n");

  // NVLink topology (16 ranks / 2 nodes -> nLocalRanks=8 > 1): per-rank 10 MiB
  // is in range and nLocalRanks=(1,) is satisfied -> match.
  void* nvlink = initTuner(/* nRanks */ 16, /* nNodes */ 2);
  EXPECT_TRUE(ruleMatched(nvlink, ncclFuncAllGather, 10 * kMiB * 16));
  kMetaTuner.finalize(nvlink);

  // nolocal topology (8 ranks / 8 nodes -> nLocalRanks=1): same per-rank size
  // (10 MiB, in range) and nNodes>1, but nLocalRanks=(1,) excludes it -> no
  // match.
  void* nolocal = initTuner(/* nRanks */ 8, /* nNodes */ 8);
  EXPECT_FALSE(ruleMatched(nolocal, ncclFuncAllGather, 10 * kMiB * 8));
  kMetaTuner.finalize(nolocal);
}

// Bracket-aware CSV split: commas inside the bytesPerRank interval do not split
// the column, so a row with [0,4096] and topology fields parses into the right
// columns. The rule matches the topology it targets and not another.
TEST_F(MetaTunerTest, BracketAwareCsvSplit) {
  const TunerConfigFile config(
      ".csv", "allreduce,[0,4096],ring,ll128,-1,2,8\n");

  void* matching = initTuner(/* nRanks */ 16, /* nNodes */ 2);
  EXPECT_TRUE(ruleMatched(matching, ncclFuncAllReduce, 1024));
  // nBytes/nRanks = 131072/16 = 8192 > 4096 -> above the per-rank range.
  EXPECT_FALSE(ruleMatched(matching, ncclFuncAllReduce, 131072));
  kMetaTuner.finalize(matching);

  void* wrongTopo = initTuner(/* nRanks */ 8, /* nNodes */ 1);
  EXPECT_FALSE(ruleMatched(wrongTopo, ncclFuncAllReduce, 1024));
  kMetaTuner.finalize(wrongTopo);
}

// With NCCLX_TUNER_IGNORE_CONFIG_ERRORS set, a rule with an invalid interval
// (lo > hi) is skipped; a valid rule on the following line still loads and
// takes effect. (Strict mode rejecting the bad rule is covered separately.)
TEST_F(MetaTunerTest, InvalidIntervalRuleSkipped) {
  const EnvRAII<bool> ignoreGuard(NCCLX_TUNER_IGNORE_CONFIG_ERRORS, true);
  const TunerConfigFile config(
      ".csv",
      "allreduce,(4096,1024],ring,ll128,-1,*,*\n"
      "allgather,[0,4096],tree,simple,-1,*,*\n");
  void* context = initTuner(/* nRanks */ 8, /* nNodes */ 1);

  // The bad allreduce rule was dropped: its target is not forced.
  EXPECT_FALSE(ruleMatched(context, ncclFuncAllReduce, 1024));

  // The valid allgather rule still loaded.
  CostTable table;
  int nChannels = 1;
  kMetaTuner.getCollInfo(
      context,
      ncclFuncAllGather,
      1024,
      1,
      table.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      0,
      &nChannels);
  EXPECT_EQ(table.at(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE), 0.0F);

  kMetaTuner.finalize(context);
}

// Asserts that, in strict mode (NCCLX_TUNER_IGNORE_CONFIG_ERRORS unset), a
// config with the given contents fails tuner init with ncclInvalidUsage and
// leaves no context to finalize. Drives the same init callback the tests use.
void expectStrictInitFails(const std::string& suffix, const std::string& csv) {
  const EnvRAII<bool> strictGuard(NCCLX_TUNER_IGNORE_CONFIG_ERRORS, false);
  const TunerConfigFile config(suffix, csv);
  void* context = nullptr;
  EXPECT_EQ(
      tryInitTuner(&context, /* nRanks */ 8, /* nNodes */ 1), ncclInvalidUsage);
  EXPECT_EQ(context, nullptr);
}

// Strict mode (default): an unknown collective token fails comm init rather
// than silently defaulting to allreduce.
TEST_F(MetaTunerTest, StrictUnknownCollectiveFailsInit) {
  expectStrictInitFails(".csv", "allReducee,[0,1024],ring,ll128,-1,*,*\n");
}

// Strict mode (default): an unknown algorithm token fails comm init.
TEST_F(MetaTunerTest, StrictUnknownAlgorithmFailsInit) {
  expectStrictInitFails(".csv", "allreduce,[0,1024],rinng,ll128,-1,*,*\n");
}

// Strict mode (default): an unknown protocol token fails comm init.
TEST_F(MetaTunerTest, StrictUnknownProtocolFailsInit) {
  expectStrictInitFails(".csv", "allreduce,[0,1024],ring,ll-128,-1,*,*\n");
}

// Strict mode (default): an invalid Int64Range (lo > hi) fails comm init.
TEST_F(MetaTunerTest, StrictInvalidIntervalFailsInit) {
  expectStrictInitFails(".csv", "allreduce,(4096,1024],ring,ll128,-1,*,*\n");
}

// Strict mode (default): a present-but-unparseable numeric column (channels)
// fails comm init rather than silently falling back to the default.
TEST_F(MetaTunerTest, StrictInvalidNumericFailsInit) {
  expectStrictInitFails(".csv", "allreduce,[0,1024],ring,ll128,abc,*,*\n");
}

// Strict mode (default): channels is always an int, so a present-but-out-of-
// int-range value fails comm init rather than silently truncating to int.
TEST_F(MetaTunerTest, StrictOutOfRangeChannelsFailsInit) {
  expectStrictInitFails(
      ".csv", "allreduce,[0,1024],ring,ll128,99999999999,*,*\n");
}

// Strict mode (default): a negative chunkSize would wrap to a huge size_t, so
// it must be rejected like any other present-but-invalid numeric.
TEST_F(MetaTunerTest, StrictNegativeChunkSizeFailsInit) {
  expectStrictInitFails(".csv", "allreduce,[0,1024],ring,ll128,-1,*,*,-5\n");
}

// Strict mode (default): a non-existent config file fails comm init.
TEST_F(MetaTunerTest, StrictMissingFileFailsInit) {
  const EnvRAII<bool> strictGuard(NCCLX_TUNER_IGNORE_CONFIG_ERRORS, false);
  const EnvRAII<std::string> envGuard(
      NCCLX_TUNER_CONFIG_FILE, std::string{"/nonexistent/path/to/tuner.csv"});
  void* context = nullptr;
  EXPECT_EQ(
      tryInitTuner(&context, /* nRanks */ 8, /* nNodes */ 1), ncclInvalidUsage);
  EXPECT_EQ(context, nullptr);
}

#ifdef NCCLX_TUNER_WITH_FOLLY_JSON
// Strict mode (default): a JSON rule missing its filter/config object fails
// comm init.
TEST_F(MetaTunerTest, StrictJsonMissingFilterConfigFailsInit) {
  expectStrictInitFails(
      ".json",
      R"({"rules": [{"config": {"algorithm": "ring", "protocol": "ll128"}}]})");
}

// Strict mode (default): a malformed JSON value type must surface as a clean
// init failure (ncclInvalidUsage), never an escaping folly::TypeError. Covers
// both a non-numeric array ("channels": [4]) and a non-numeric string
// ("channels": "abc").
TEST_F(MetaTunerTest, StrictJsonBadValueTypeArrayFailsInit) {
  expectStrictInitFails(
      ".json",
      R"({"rules": [{"filter": {"collective": "allreduce"},
          "config": {"algorithm": "ring", "protocol": "ll128",
                     "channels": [4]}}]})");
}

TEST_F(MetaTunerTest, StrictJsonBadValueTypeStringFailsInit) {
  expectStrictInitFails(
      ".json",
      R"({"rules": [{"filter": {"collective": "allreduce"},
          "config": {"algorithm": "ring", "protocol": "ll128",
                     "channels": "abc"}}]})");
}

// Ignore mode: the same malformed JSON value type is skipped (not fatal),
// init succeeds with no rule applied, and no exception escapes.
TEST_F(MetaTunerTest, IgnoreJsonBadValueTypeSkipsRule) {
  const EnvRAII<bool> ignoreGuard(NCCLX_TUNER_IGNORE_CONFIG_ERRORS, true);
  const TunerConfigFile config(
      ".json",
      R"({"rules": [{"filter": {"collective": "allreduce"},
          "config": {"algorithm": "ring", "protocol": "ll128",
                     "channels": [4]}}]})");
  void* context = initTuner(/* nRanks */ 8, /* nNodes */ 1);

  // The bad rule was dropped: its target is not forced.
  EXPECT_FALSE(ruleMatched(context, ncclFuncAllReduce, 512));

  kMetaTuner.finalize(context);
}
#endif

// Case 4: when two rows match, the first wins (row order = priority).
TEST_F(MetaTunerTest, RowOrderPriority) {
  const TunerConfigFile config(
      ".csv",
      "allreduce,[0,4096],ring,ll128,-1,*,*\n"
      "allreduce,[0,4096],tree,simple,-1,*,*\n");
  void* context = initTuner(8, 1);

  CostTable table;
  int nChannels = 1;
  kMetaTuner.getCollInfo(
      context,
      ncclFuncAllReduce,
      1024,
      1,
      table.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      0,
      &nChannels);
  EXPECT_EQ(table.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), 0.0F);
  EXPECT_EQ(table.at(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE), 1.0F);

  kMetaTuner.finalize(context);
}

// Case 5: nChannels override -- -1 leaves the value, a positive value sets it.
TEST_F(MetaTunerTest, ChannelsOverride) {
  const TunerConfigFile config(
      ".csv",
      "allreduce,[0,1024],ring,ll128,-1,*,*\n"
      "allgather,[0,1024],ring,simple,4,*,*\n");
  void* context = initTuner(8, 1);

  CostTable defaultTable;
  int defaultChannels = 7;
  kMetaTuner.getCollInfo(
      context,
      ncclFuncAllReduce,
      512,
      1,
      defaultTable.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      0,
      &defaultChannels);
  EXPECT_EQ(defaultChannels, 7);

  CostTable overrideTable;
  int overrideChannels = 7;
  kMetaTuner.getCollInfo(
      context,
      ncclFuncAllGather,
      512,
      1,
      overrideTable.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      0,
      &overrideChannels);
  EXPECT_EQ(overrideChannels, 4);

  kMetaTuner.finalize(context);
}

// Case 6: an entry preset to NCCL_ALGO_PROTO_IGNORE is never force-selected.
TEST_F(MetaTunerTest, IgnoreProtection) {
  const TunerConfigFile config(
      ".csv", "allreduce,[0,1024],ring,ll128,-1,*,*\n");
  void* context = initTuner(8, 1);

  CostTable table;
  table.setIgnore(NCCL_ALGO_RING, NCCL_PROTO_LL128);
  int nChannels = 1;
  kMetaTuner.getCollInfo(
      context,
      ncclFuncAllReduce,
      512,
      1,
      table.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      0,
      &nChannels);
  // Stays IGNORE; the rule did not force the unsupported combo to 0.0.
  EXPECT_EQ(table.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), NCCL_ALGO_PROTO_IGNORE);

  kMetaTuner.finalize(context);
}

// Case 7: with NCCLX_TUNER_IGNORE_CONFIG_ERRORS set, a missing file -> init
// succeeds, no rules, outputs untouched. (Strict mode failing on a missing
// file is covered separately.)
TEST_F(MetaTunerTest, MissingFileIsNoOp) {
  const EnvRAII<bool> ignoreGuard(NCCLX_TUNER_IGNORE_CONFIG_ERRORS, true);
  const EnvRAII<std::string> envGuard(
      NCCLX_TUNER_CONFIG_FILE, std::string{"/nonexistent/path/to/tuner.csv"});
  EXPECT_TRUE(metaTunerEnabled());

  void* context = initTuner(8, 1);
  CostTable table;
  int nChannels = 5;
  kMetaTuner.getCollInfo(
      context,
      ncclFuncAllReduce,
      512,
      1,
      table.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      0,
      &nChannels);
  EXPECT_EQ(table.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), 1.0F);
  EXPECT_EQ(nChannels, 5);

#ifdef NCCLX_TUNER_HAS_GETCHUNKSIZE
  size_t chunkSize = 16384;
  kMetaTuner.getChunkSize(
      context,
      ncclFuncAllReduce,
      512,
      NCCL_ALGO_RING,
      NCCL_PROTO_LL128,
      1,
      &chunkSize);
  EXPECT_EQ(chunkSize, 16384);
#endif

  kMetaTuner.finalize(context);
}

#ifdef NCCLX_TUNER_HAS_GETCHUNKSIZE
// Case 9: getChunkSize override -- chunkSize != 0 sets it; 0/omitted leaves it.
TEST_F(MetaTunerTest, ChunkSizeOverride) {
  const TunerConfigFile config(
      ".csv",
      "allreduce,[0,4096],ring,ll128,-1,*,*,65536\n"
      "allgather,[0,4096],ring,ll128,-1,*,*,0\n");
  void* context = initTuner(8, 1);

  size_t overridden = 16384;
  kMetaTuner.getChunkSize(
      context,
      ncclFuncAllReduce,
      1024,
      NCCL_ALGO_RING,
      NCCL_PROTO_LL128,
      1,
      &overridden);
  EXPECT_EQ(overridden, 65536);

  size_t untouched = 16384;
  kMetaTuner.getChunkSize(
      context,
      ncclFuncAllGather,
      1024,
      NCCL_ALGO_RING,
      NCCL_PROTO_LL128,
      1,
      &untouched);
  EXPECT_EQ(untouched, 16384);

  kMetaTuner.finalize(context);
}

// Case 10: getChunkSize keys on algo/proto; only the matching row overrides.
TEST_F(MetaTunerTest, ChunkSizeAlgoProtoKey) {
  const TunerConfigFile config(
      ".csv",
      "allreduce,[0,4096],ring,ll128,-1,*,*,65536\n"
      "allreduce,[0,4096],tree,simple,-1,*,*,32768\n");
  void* context = initTuner(8, 1);

  size_t ringChunk = 16384;
  kMetaTuner.getChunkSize(
      context,
      ncclFuncAllReduce,
      1024,
      NCCL_ALGO_RING,
      NCCL_PROTO_LL128,
      1,
      &ringChunk);
  EXPECT_EQ(ringChunk, 65536);

  size_t treeChunk = 16384;
  kMetaTuner.getChunkSize(
      context,
      ncclFuncAllReduce,
      1024,
      NCCL_ALGO_TREE,
      NCCL_PROTO_SIMPLE,
      1,
      &treeChunk);
  EXPECT_EQ(treeChunk, 32768);

  // An algo/proto not present in any row leaves the default untouched.
  size_t otherChunk = 16384;
  kMetaTuner.getChunkSize(
      context,
      ncclFuncAllReduce,
      1024,
      NCCL_ALGO_NVLS,
      NCCL_PROTO_SIMPLE,
      1,
      &otherChunk);
  EXPECT_EQ(otherChunk, 16384);

  kMetaTuner.finalize(context);
}
#endif // NCCLX_TUNER_HAS_GETCHUNKSIZE

// Case 1: CSV parsing with comments, blank lines and omitted optional columns.
TEST_F(MetaTunerTest, CsvParsingWithCommentsAndOmittedColumns) {
  const TunerConfigFile config(
      ".csv",
      "# leading comment\n"
      "\n"
      "allreduce,[0,1024],ring,ll128,-1,*,*\n"
      "  allgather, [0, 2048], tree, simple, 4, 1, 8 \n");
  void* context = initTuner(/* nRanks */ 8, /* nNodes */ 1);

  // First rule (omitted chunkSize defaults to 0 = no override).
  CostTable allreduceTable;
  int allreduceChannels = 9;
  kMetaTuner.getCollInfo(
      context,
      ncclFuncAllReduce,
      512,
      1,
      allreduceTable.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      0,
      &allreduceChannels);
  EXPECT_EQ(allreduceTable.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), 0.0F);
  EXPECT_EQ(allreduceChannels, 9);

  // Second rule: whitespace trimmed, channels overridden, topology honored.
  CostTable allgatherTable;
  int allgatherChannels = 9;
  kMetaTuner.getCollInfo(
      context,
      ncclFuncAllGather,
      1024,
      1,
      allgatherTable.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      0,
      &allgatherChannels);
  EXPECT_EQ(allgatherTable.at(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE), 0.0F);
  EXPECT_EQ(allgatherChannels, 4);

  kMetaTuner.finalize(context);
}

// Observable effect of one parsed rule, gathered by driving the public tuner
// hooks. Two contexts loaded from equivalent tables (CSV vs JSON) must yield
// the same probe for every authored rule.
struct RuleProbe {
  float matchedCost;
  int channels;
#ifdef NCCLX_TUNER_HAS_GETCHUNKSIZE
  size_t chunkSize;
#endif

  bool operator==(const RuleProbe& other) const {
    return matchedCost == other.matchedCost && channels == other.channels
#ifdef NCCLX_TUNER_HAS_GETCHUNKSIZE
        && chunkSize == other.chunkSize
#endif
        ;
  }
};

// Probes a context for the rule keyed on (collType, nBytes, algo, proto):
// matchedCost / channels come from getCollInfo, chunkSize from getChunkSize.
RuleProbe probeRule(
    void* context,
    const ncclFunc_t collType,
    const size_t nBytes,
    const int algo,
    const int proto) {
  CostTable table;
  int channels = -1;
  kMetaTuner.getCollInfo(
      context,
      collType,
      nBytes,
      /* numPipeOps */ 1,
      table.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      /* regBuff */ 0,
      &channels);

#ifdef NCCLX_TUNER_HAS_GETCHUNKSIZE
  size_t chunkSize = 0;
  kMetaTuner.getChunkSize(
      context, collType, nBytes, algo, proto, /* nChannels */ 1, &chunkSize);

  return RuleProbe{table.at(algo, proto), channels, chunkSize};
#else
  return RuleProbe{table.at(algo, proto), channels};
#endif
}

#ifdef NCCLX_TUNER_WITH_FOLLY_JSON
// Case 1b: the SAME tuning table authored as CSV and as JSON must parse to
// equivalent configs -- verified field-by-field through identical probes.
TEST_F(MetaTunerTest, JsonParsingEquivalentToCsv) {
  void* csvContext = nullptr;
  {
    const TunerConfigFile csv(
        ".csv",
        "allreduce,[0,4096],ring,ll128,4,1,8,65536\n"
        "allgather,[0,2048],tree,simple,-1,*,*,0\n"
        "reducescatter,[8388608,16777216],ring,ll128,-1,*,*\n");
    csvContext = initTuner(/* nRanks */ 8, /* nNodes */ 1);
  }

  void* jsonContext = nullptr;
  {
    // The allreduce row exercises a JSON `bytesPerRank` interval string with a
    // pinned topology; the allgather row omits `bytesPerRank` (defaults to the
    // * wildcard). The reducescatter row keys purely on `bytesPerRank`. The CSV
    // authors the equivalent rules.
    const TunerConfigFile json(
        ".json",
        R"({
          "rules": [
            {"filter": {"collective": "allreduce", "bytesPerRank": "[0,4096]",
                        "nNodes": 1, "nLocalRanks": 8},
             "config": {"algorithm": "ring", "protocol": "ll128",
                        "channels": 4, "chunkSize": 65536}},
            {"filter": {"collective": "allgather", "bytesPerRank": "[0,2048]"},
             "config": {"algorithm": "tree", "protocol": "simple"}},
            {"filter": {"collective": "reducescatter",
                        "bytesPerRank": "[8388608,16777216]"},
             "config": {"algorithm": "ring", "protocol": "ll128"}}
          ]
        })");
    jsonContext = initTuner(/* nRanks */ 8, /* nNodes */ 1);
  }

  // Probe each authored row in both contexts; the observable per-row config
  // (matched cost, channels, chunkSize) must be identical.
  const RuleProbe csvRow0 = probeRule(
      csvContext, ncclFuncAllReduce, 1024, NCCL_ALGO_RING, NCCL_PROTO_LL128);
  const RuleProbe jsonRow0 = probeRule(
      jsonContext, ncclFuncAllReduce, 1024, NCCL_ALGO_RING, NCCL_PROTO_LL128);
  EXPECT_EQ(csvRow0, jsonRow0);
#ifdef NCCLX_TUNER_HAS_GETCHUNKSIZE
  EXPECT_EQ(csvRow0, (RuleProbe{0.0F, 4, 65536}));
#else
  EXPECT_EQ(csvRow0, (RuleProbe{0.0F, 4}));
#endif

  const RuleProbe csvRow1 = probeRule(
      csvContext, ncclFuncAllGather, 1024, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE);
  const RuleProbe jsonRow1 = probeRule(
      jsonContext, ncclFuncAllGather, 1024, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE);
  EXPECT_EQ(csvRow1, jsonRow1);
#ifdef NCCLX_TUNER_HAS_GETCHUNKSIZE
  EXPECT_EQ(csvRow1, (RuleProbe{0.0F, -1, 0}));
#else
  EXPECT_EQ(csvRow1, (RuleProbe{0.0F, -1}));
#endif

  // Row 2 keys on bytesPerRank=[8MiB,16MiB]; at nRanks=8 a 80 MiB total is
  // 10 MiB/rank (in range), so CSV and JSON must force ring/ll128 identically.
  const RuleProbe csvRow2 = probeRule(
      csvContext,
      ncclFuncReduceScatter,
      80 * 1024 * 1024,
      NCCL_ALGO_RING,
      NCCL_PROTO_LL128);
  const RuleProbe jsonRow2 = probeRule(
      jsonContext,
      ncclFuncReduceScatter,
      80 * 1024 * 1024,
      NCCL_ALGO_RING,
      NCCL_PROTO_LL128);
  EXPECT_EQ(csvRow2, jsonRow2);
#ifdef NCCLX_TUNER_HAS_GETCHUNKSIZE
  EXPECT_EQ(csvRow2, (RuleProbe{0.0F, -1, 0}));
#else
  EXPECT_EQ(csvRow2, (RuleProbe{0.0F, -1}));
#endif

  kMetaTuner.finalize(jsonContext);
  kMetaTuner.finalize(csvContext);
}
#endif

#ifndef NCCLX_TUNER_WITH_FOLLY_JSON
// Case 1c: in a no-folly build, a .json config is unsupported -- init still
// succeeds with EMPTY configs (no override applied), while CSV still loads.
// Guarded by the same macro, so this body compiles only in the OSS no-folly
// build (the buck build defines the macro and compiles it out).
TEST_F(MetaTunerTest, JsonUnsupportedInNoFollyBuild) {
  {
    // A .json path is unsupported here; with ignore set it is skipped (empty
    // table, init succeeds) rather than failing comm init.
    const EnvRAII<bool> ignoreGuard(NCCLX_TUNER_IGNORE_CONFIG_ERRORS, true);
    const TunerConfigFile json(
        ".json",
        R"({"rules": [{"filter": {"collective": "allreduce", "bytesPerRank": "[0,4096]"},
            "config": {"algorithm": "ring", "protocol": "ll128"}}]})");
    void* context = initTuner(/* nRanks */ 8, /* nNodes */ 1);

    // No rule was parsed: the targeted entry is left untouched.
    CostTable table;
    int channels = 7;
    kMetaTuner.getCollInfo(
        context,
        ncclFuncAllReduce,
        1024,
        1,
        table.asCallbackArg(),
        NCCL_NUM_ALGORITHMS,
        NCCL_NUM_PROTOCOLS,
        0,
        &channels);
    EXPECT_EQ(table.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), 1.0F);
    EXPECT_EQ(channels, 7);
    kMetaTuner.finalize(context);
  }

  // CSV still works in the same no-folly build.
  {
    const TunerConfigFile csv(".csv", "allreduce,[0,4096],ring,ll128,-1,*,*\n");
    void* context = initTuner(/* nRanks */ 8, /* nNodes */ 1);
    CostTable table;
    int channels = 1;
    kMetaTuner.getCollInfo(
        context,
        ncclFuncAllReduce,
        1024,
        1,
        table.asCallbackArg(),
        NCCL_NUM_ALGORITHMS,
        NCCL_NUM_PROTOCOLS,
        0,
        &channels);
    EXPECT_EQ(table.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), 0.0F);
    kMetaTuner.finalize(context);
  }
}
#endif

// The checked-in CSV example config parses and its rules take effect. Drives
// getCollInfo to confirm the small-allreduce rule resolves; the large-allreduce
// chunkSize override is checked only where getChunkSize exists.
TEST_F(MetaTunerTest, ExampleCsvConfigLoads) {
  const std::string path = exampleConfigPath("EXAMPLE_TUNER_CONFIG_CSV");
  const EnvRAII<std::string> envGuard(NCCLX_TUNER_CONFIG_FILE, path);
  EXPECT_TRUE(metaTunerEnabled());

  void* context = initTuner(/* nRanks */ 8, /* nNodes */ 1);

  // Small allreduce: ring/ll128 forced to 0.0.
  CostTable smallTable;
  int channels = -1;
  kMetaTuner.getCollInfo(
      context,
      ncclFuncAllReduce,
      /* nBytes */ 4096,
      /* numPipeOps */ 1,
      smallTable.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      /* regBuff */ 0,
      &channels);
  EXPECT_EQ(smallTable.at(NCCL_ALGO_RING, NCCL_PROTO_LL128), 0.0F);

#ifdef NCCLX_TUNER_HAS_GETCHUNKSIZE
  // Large allreduce: per-rank shard above 1 MiB (16 MiB total / 8 ranks =
  // 2 MiB/rank) -> ring/simple carries a 524288-byte chunkSize override.
  size_t chunkSize = 16384;
  kMetaTuner.getChunkSize(
      context,
      ncclFuncAllReduce,
      /* nBytes */ 16777216,
      NCCL_ALGO_RING,
      NCCL_PROTO_SIMPLE,
      /* nChannels */ 1,
      &chunkSize);
  EXPECT_EQ(chunkSize, 524288);
#endif

  kMetaTuner.finalize(context);
}

#ifdef NCCLX_TUNER_WITH_FOLLY_JSON
// The checked-in JSON example config parses and its allgather rule (channel +
// topology override) takes effect on a matching topology.
TEST_F(MetaTunerTest, ExampleJsonConfigLoads) {
  const std::string path = exampleConfigPath("EXAMPLE_TUNER_CONFIG_JSON");
  const EnvRAII<std::string> envGuard(NCCLX_TUNER_CONFIG_FILE, path);
  EXPECT_TRUE(metaTunerEnabled());

  void* context = initTuner(/* nRanks */ 16, /* nNodes */ 2);

  // Allgather on the 2-node / 16-rank topology: tree/simple forced, 4 channels.
  CostTable table;
  int channels = 9;
  kMetaTuner.getCollInfo(
      context,
      ncclFuncAllGather,
      /* nBytes */ 1048576,
      /* numPipeOps */ 1,
      table.asCallbackArg(),
      NCCL_NUM_ALGORITHMS,
      NCCL_NUM_PROTOCOLS,
      /* regBuff */ 0,
      &channels);
  EXPECT_EQ(table.at(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE), 0.0F);
  EXPECT_EQ(channels, 4);

  kMetaTuner.finalize(context);
}
#endif

} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
