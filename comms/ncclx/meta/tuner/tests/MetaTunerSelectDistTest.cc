// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <folly/init/Init.h>

#include <comm.h>
#include <nccl.h>
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/ncclx/meta/tests/VerifyAlgoStatsUtil.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual

/**
 * Comprehensive distributed test proving the NCCLX built-in meta tuner controls
 * the algorithm, protocol, and channel count NCCL actually selects for real
 * AllReduce, AllGather, and ReduceScatter collectives, across three virtual
 * topologies and with topology- and size-keyed config matching.
 *
 * The tuner is engaged purely by the NCCLX_TUNER_CONFIG_FILE cvar, read during
 * comm init (metaTunerInit). Each case writes a temp CSV forcing a collective
 * to a specific algo/proto (and optionally nChannels and a topology/size key),
 * points the cvar at it BEFORE comm creation, runs a real collective, and uses
 * VerifyAlgoStatsHelper to confirm the algorithm NCCL actually used matches
 * (or, for negative cases, does NOT change) the forced config. AlgoStats
 * reports the algo as "Baseline_{PROTO}_{ALGO}_{nChannels}" (e.g.
 * "Baseline_SIMPLE_RING_8"); verifyExact() asserts every recorded algorithm
 * contains the expected substring (e.g. "SIMPLE_RING" or "SIMPLE_RING_2").
 * Negative cases compare a comm carrying a non-matching rule against a no-tuner
 * comm via verifyEqual() to prove the rule changed nothing.
 *
 * Topology: launched on a single physical 8-GPU host. The EFFECTIVE
 * comm->nNodes is driven by the NCCL_HOSTID env var, which overrides each
 * rank's hostHash that NCCL core counts to compute nNodes. Each rank is its own
 * process and sets its own NCCL_HOSTID in SetUp() BEFORE any comm init (i.e.
 * before getHostHash()'s std::call_once locks the value). The compile macro
 * TUNER_TOPO_NNODES selects the topology: default 1 => (nNodes,nRanks)=(1,8);
 * 8 => (8,8) (nolocal, PAT-eligible since nNodes==nRanks); 2 => (2,8) (vnode2).
 * The BUCK suite generates one target per topology config. kExpectedNodes/
 * kExpectedRanks below are derived from the macro so topology-keyed CSV rows
 * can match the comm, and each comm asserts its realized (nNodes, nRanks).
 *
 * All assertions are hard (EXPECT/ASSERT, no GTEST_SKIP): unsupported combos
 * (e.g. LL128 on a GPU that lacks it) are expected to surface as real failures
 * so we observe ground truth rather than silently skipping.
 */

// Effective topology of the comm under the active compile macro. Each rank
// drives comm->nNodes by setting NCCL_HOSTID before comm creation (see
// SetUp()). TUNER_TOPO_NNODES is set per BUCK config; default 1 (all ranks
// share the real host).
#ifndef TUNER_TOPO_NNODES
#define TUNER_TOPO_NNODES 1
#endif
constexpr int kExpectedNodes = TUNER_TOPO_NNODES;
constexpr int kExpectedRanks = 8;
// Ranks per node derived from the topology (1x8 -> 8, nolocal -> 1, vnode2 ->
// 4). This is the value the tuner context computes as nLocalRanks.
constexpr int kExpectedLocalRanks = kExpectedRanks / kExpectedNodes;

namespace {

// Collective under test. Drives both the AlgoStats lookup name and which NCCL
// API the helper invokes.
enum class Collective { kAllReduce, kAllGather, kReduceScatter };

// AlgoStats collective-name string (from ncclFuncToString in collectives.cc).
const char* algoStatsName(Collective coll) {
  switch (coll) {
    case Collective::kAllReduce:
      return "AllReduce";
    case Collective::kAllGather:
      return "AllGather";
    case Collective::kReduceScatter:
      return "ReduceScatter";
  }
  return "AllReduce";
}

// Tuner CSV collective token (from MetaTuner.cc nameToColl).
const char* tunerName(Collective coll) {
  switch (coll) {
    case Collective::kAllReduce:
      return "allreduce";
    case Collective::kAllGather:
      return "allgather";
    case Collective::kReduceScatter:
      return "reducescatter";
  }
  return "allreduce";
}

// RAII helper that writes a temp config file and points NCCLX_TUNER_CONFIG_FILE
// at it via EnvRAII, restoring the cvar and removing the file on teardown.
// Mirrors the TunerConfigFile helper in MetaTunerTest.cpp. In a multi-process
// dist test each rank runs this independently and writes its own unique temp
// file via std::tmpnam.
class TunerConfigFile {
 public:
  explicit TunerConfigFile(const std::string& contents)
      : path_(writeTempFile(contents)),
        envGuard_(NCCLX_TUNER_CONFIG_FILE, path_) {}

  ~TunerConfigFile() {
    std::remove(path_.c_str());
  }

 private:
  static std::string writeTempFile(const std::string& contents) {
    std::string path = std::string(std::tmpnam(nullptr)) + ".csv";
    std::ofstream out(path);
    out << contents;
    return path;
  }

  std::string path_;
  EnvRAII<std::string> envGuard_;
};

class MetaTunerSelectDistTest : public NcclxBaseTestFixture {
 protected:
  ncclx::test::VerifyAlgoStatsHelper algoStats_;

  void SetUp() override {
    NcclxBaseTestFixture::SetUp();

    // Drive the comm's effective nNodes by overriding each rank's hostHash via
    // NCCL_HOSTID. This must happen after the base SetUp (so
    // globalRank/numRanks are populated by distSetUp) but before any comm is
    // created -- getHostHash caches the value via std::call_once on the first
    // comm init, so setting it here guarantees it takes effect.
    if (kExpectedNodes > 1) {
      const int ranksPerNode = numRanks / kExpectedNodes;
      const int nodeId = globalRank / ranksPerNode;
      // Overwrite=1 so the value applies even if NCCL_HOSTID was already set.
      setenv("NCCL_HOSTID", fmt::format("tunertest-node{}", nodeId).c_str(), 1);
    }

    // AlgoStats tracing must be enabled before comm creation.
    algoStats_.enable();
  }

  // Build a tuner CSV rule for one collective covering a broad size range.
  // Format (MetaTuner CSV):
  // collective,bytesPerRank,algo,proto,nChannels,nNodes, nLocalRanks. The size
  // range is emitted as the closed per-rank interval [minBytes,maxBytes]
  // (bytesPerRank = nBytes / nRanks). A channels of -1 means "keep the NCCL
  // default"; nNodes/nLocalRanks default to -1, emitted as the
  // "*" wildcard (match any). nNodes/nLocalRanks accept any Int64Range
  // expression string via the makeRangeRule overload below.
  static std::string makeRule(
      Collective coll,
      const std::string& algo,
      const std::string& proto,
      int channels = -1,
      int64_t minBytes = 0,
      int64_t maxBytes = 1073741824,
      int nNodes = -1,
      int nLocalRanks = -1) {
    return makeRangeRule(
        coll,
        algo,
        proto,
        channels,
        fmt::format("[{},{}]", minBytes, maxBytes),
        nNodes == -1 ? std::string("*") : std::to_string(nNodes),
        nLocalRanks == -1 ? std::string("*") : std::to_string(nLocalRanks));
  }

  // Like makeRule, but bytesPerRank / nNodes / nLocalRanks are arbitrary
  // Int64Range expression strings (e.g. "(1,)", "[0,1048576]"), exercising
  // interval matching end-to-end.
  static std::string makeRangeRule(
      Collective coll,
      const std::string& algo,
      const std::string& proto,
      int channels,
      const std::string& bytesPerRank,
      const std::string& nNodes,
      const std::string& nLocalRanks) {
    return fmt::format(
        "{},{},{},{},{},{},{}\n",
        tunerName(coll),
        bytesPerRank,
        algo,
        proto,
        channels,
        nNodes,
        nLocalRanks);
  }

  // Assert a freshly created comm realized the topology this config targets.
  // If NCCL_HOSTID failed to take effect (e.g. set too late), comm->nNodes
  // would silently fall back to 1; this hard assert makes that fail loudly.
  // Called once per comm creation, immediately after the comm is created.
  static void assertTopology(ncclComm_t comm) {
    ASSERT_EQ(comm->nNodes, kExpectedNodes);
    ASSERT_EQ(comm->nRanks, kExpectedRanks);
    // The tuner derives nLocalRanks = nRanks / nNodes; assert the comm realizes
    // the ranks-per-node count the topology-keyed rules below target.
    ASSERT_EQ(comm->nRanks / comm->nNodes, kExpectedLocalRanks);
  }

  // Create a comm (loading the tuner from the cvar set by the active
  // TunerConfigFile), run the given collective once at the given element count,
  // and synchronize. Returns the comm via the RAII guard's lifetime, so the
  // caller verifies AlgoStats before the guard goes out of scope.
  void runCollective(ncclComm_t comm, Collective coll, size_t count) {
    cudaStream_t stream = nullptr;
    CUDACHECK_TEST(cudaStreamCreate(&stream));

    // ReduceScatter sends count*nRanks and receives count; AllGather sends
    // count and receives count*nRanks; AllReduce is count in/out. Allocate the
    // larger count*nRanks for both buffers to cover all three.
    const size_t maxElems = count * numRanks;
    float* sendBuf = nullptr;
    float* recvBuf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&sendBuf, maxElems * sizeof(float)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, maxElems * sizeof(float)));

    ncclResult_t res = ncclSuccess;
    switch (coll) {
      case Collective::kAllReduce:
        res = ncclAllReduce(
            sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream);
        break;
      case Collective::kAllGather:
        res = ncclAllGather(sendBuf, recvBuf, count, ncclFloat, comm, stream);
        break;
      case Collective::kReduceScatter:
        res = ncclReduceScatter(
            sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream);
        break;
    }
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
  }

  // Force a collective to (algo, proto) over a broad size range, create a fresh
  // comm (the tuner parses the config at comm init, so every forced combo needs
  // its own comm), run the collective at a mid-range size, and verify AlgoStats
  // reports exactly expectProtoAlgo (proto+algo, e.g. "LL_RING").
  void forceAndVerify(
      Collective coll,
      const std::string& algo,
      const std::string& proto,
      const std::string& expectProtoAlgo) {
    TunerConfigFile config(makeRule(coll, algo, proto));
    ncclx::test::NcclCommRAII commGuard(
        globalRank, numRanks, localRank, bootstrap_.get());
    ncclComm_t comm = commGuard.get();
    assertTopology(comm);

    // 4096 floats = 16 KiB, well inside the configured [0, 1 GiB) range.
    runCollective(comm, coll, 4096);
    algoStats_.verifyExact(comm, algoStatsName(coll), expectProtoAlgo);
  }
};

// One algo x proto sweep case: force a collective to (algo, proto) and verify
// AlgoStats reports the expected proto+algo token. The name field becomes the
// gtest instance name (e.g. "AllReduce_ring_LL").
struct SweepCase {
  Collective collective;
  const char* algo; // "ring"/"tree"/"pat"
  const char* proto; // "ll"/"ll128"/"simple"
  std::string expectProtoAlgo; // e.g. "LL_RING", "SIMPLE_PAT"
  std::string name; // test instance name, e.g. "AllReduce_ring_LL"
};

// Build the algo x proto sweep cases:
//   AllReduce:     {ring, tree} x {LL, LL128, Simple}
//   AllGather:     ring x {LL, LL128, Simple}
//   ReduceScatter: ring x {LL, LL128, Simple}
// Forcing each distinct combo and verifying its proto+algo token proves the
// tuner controls both dimensions independently of NCCL's default selection.
//
// PAT x Simple is appended for AllGather and ReduceScatter only under the
// nolocal topology. ncclPatEnable (tuning.cc) returns 0 unless comm->nNodes ==
// comm->nRanks, which holds only for the nolocal topology (8 nodes / 8 ranks);
// PAT also supports only the Simple protocol (tuning.cc). Elsewhere PAT is
// physically disabled, so these cases are gated to nolocal.
std::vector<SweepCase> makeSweepCases() {
  std::vector<SweepCase> cases = {
      {Collective::kAllReduce, "ring", "ll", "LL_RING", "AllReduce_ring_LL"},
      {Collective::kAllReduce,
       "ring",
       "ll128",
       "LL128_RING",
       "AllReduce_ring_LL128"},
      {Collective::kAllReduce,
       "ring",
       "simple",
       "SIMPLE_RING",
       "AllReduce_ring_Simple"},
      {Collective::kAllReduce, "tree", "ll", "LL_TREE", "AllReduce_tree_LL"},
      {Collective::kAllReduce,
       "tree",
       "ll128",
       "LL128_TREE",
       "AllReduce_tree_LL128"},
      {Collective::kAllReduce,
       "tree",
       "simple",
       "SIMPLE_TREE",
       "AllReduce_tree_Simple"},
      {Collective::kAllGather, "ring", "ll", "LL_RING", "AllGather_ring_LL"},
      {Collective::kAllGather,
       "ring",
       "ll128",
       "LL128_RING",
       "AllGather_ring_LL128"},
      {Collective::kAllGather,
       "ring",
       "simple",
       "SIMPLE_RING",
       "AllGather_ring_Simple"},
      {Collective::kReduceScatter,
       "ring",
       "ll",
       "LL_RING",
       "ReduceScatter_ring_LL"},
      {Collective::kReduceScatter,
       "ring",
       "ll128",
       "LL128_RING",
       "ReduceScatter_ring_LL128"},
      {Collective::kReduceScatter,
       "ring",
       "simple",
       "SIMPLE_RING",
       "ReduceScatter_ring_Simple"},
  };

#if TUNER_TOPO_NNODES == 8
  cases.push_back(
      {Collective::kAllGather,
       "pat",
       "simple",
       "SIMPLE_PAT",
       "AllGather_pat_Simple"});
  cases.push_back(
      {Collective::kReduceScatter,
       "pat",
       "simple",
       "SIMPLE_PAT",
       "ReduceScatter_pat_Simple"});
#endif

  return cases;
}

// Parameterized fixture over the algo x proto sweep. Inherits SetUp/TearDown
// and all helpers from MetaTunerSelectDistTest.
class MetaTunerSweepTest : public MetaTunerSelectDistTest,
                           public ::testing::WithParamInterface<SweepCase> {};

TEST_P(MetaTunerSweepTest, ForcesAlgoProto) {
  const auto& sweepCase = GetParam();
  forceAndVerify(
      sweepCase.collective,
      sweepCase.algo,
      sweepCase.proto,
      sweepCase.expectProtoAlgo);
}

INSTANTIATE_TEST_SUITE_P(
    Sweep,
    MetaTunerSweepTest,
    ::testing::ValuesIn(makeSweepCases()),
    [](const ::testing::TestParamInfo<SweepCase>& info) {
      return info.param.name;
    });

// Force AllReduce to ring+simple with an explicit 2-channel (SM) count and
// confirm NCCL used exactly 2 channels. Verifying the full "SIMPLE_RING_2"
// token confirms algo, proto, AND channel count simultaneously.
TEST_F(MetaTunerSelectDistTest, ForcesTwoChannels) {
  TunerConfigFile config(
      makeRule(Collective::kAllReduce, "ring", "simple", /*channels=*/2));
  ncclx::test::NcclCommRAII commGuard(
      globalRank, numRanks, localRank, bootstrap_.get());
  ncclComm_t comm = commGuard.get();
  assertTopology(comm);
  runCollective(comm, Collective::kAllReduce, 4096);
  algoStats_.verifyExact(comm, "AllReduce", "SIMPLE_RING_2");
}

// Topology-keyed matching, positive case: a rule keyed to the comm's ACTUAL
// (nNodes, nLocalRanks) forces tree+LL, so AllReduce uses LL_TREE.
TEST_F(MetaTunerSelectDistTest, TopologyKeyMatches) {
  TunerConfigFile config(makeRule(
      Collective::kAllReduce,
      "tree",
      "ll",
      /*channels=*/-1,
      /*minBytes=*/0,
      /*maxBytes=*/1073741824,
      /*nNodes=*/kExpectedNodes,
      /*nLocalRanks=*/kExpectedLocalRanks));
  ncclx::test::NcclCommRAII commGuard(
      globalRank, numRanks, localRank, bootstrap_.get());
  ncclComm_t comm = commGuard.get();
  assertTopology(comm);
  runCollective(comm, Collective::kAllReduce, 4096);
  algoStats_.verifyExact(comm, "AllReduce", "LL_TREE");
}

// Topology-keyed matching, negative case: a rule keyed to a DIFFERENT
// ranks-per-node count (nLocalRanks = kExpectedLocalRanks+1) does NOT match
// this comm, so the tuner does not apply it and NCCL falls back to its default
// selection.
//
// "Unchanged vs default" model: create a no-tuner comm (NCCLX_TUNER_CONFIG_FILE
// unset, tuner disabled) and run AllReduce on it, then create a second comm
// carrying a topo-MISMATCHED rule (nLocalRanks+1) forcing tree+LL and run the
// SAME collective at the SAME size on it. verifyEqual asserts both comms
// selected the same set of algorithms, proving the non-matching rule changed
// nothing.
TEST_F(MetaTunerSelectDistTest, TopologyKeyMismatchIgnored) {
  // No-tuner comm: no TunerConfigFile in scope, so the tuner is disabled and
  // NCCL uses its default selection.
  ncclx::test::NcclCommRAII noTunerGuard(
      globalRank, numRanks, localRank, bootstrap_.get());
  ncclComm_t noTunerComm = noTunerGuard.get();
  assertTopology(noTunerComm);
  runCollective(noTunerComm, Collective::kAllReduce, 4096);

  // Rule comm: a topo-MISMATCHED rule (nLocalRanks+1) forcing tree+LL does not
  // match this comm, so the tuner ignores it. Keep both comms alive until
  // verifyEqual reads their AlgoStats.
  TunerConfigFile config(makeRule(
      Collective::kAllReduce,
      "tree",
      "ll",
      /*channels=*/-1,
      /*minBytes=*/0,
      /*maxBytes=*/1073741824,
      /*nNodes=*/kExpectedNodes,
      /*nLocalRanks=*/kExpectedLocalRanks + 1));
  ncclx::test::NcclCommRAII ruleGuard(
      globalRank, numRanks, localRank, bootstrap_.get());
  ncclComm_t ruleComm = ruleGuard.get();
  assertTopology(ruleComm);
  runCollective(ruleComm, Collective::kAllReduce, 4096);

  algoStats_.verifyEqual(noTunerComm, ruleComm, "AllReduce");
}

// Range-keyed topology matching via an open-ended nNodes interval. A rule keyed
// nNodes=(1,) (strictly more than one node) forces tree+LL. It MATCHES the
// multi-node topologies (nolocal: 8 nodes; vnode2: 2 nodes), so AllReduce uses
// LL_TREE there. On the single-node 1x8 topology (nNodes==1) the interval does
// NOT match, so the tuner is ignored and NCCL falls back to its default
// selection -- verified against a no-tuner comm via verifyEqual.
TEST_F(MetaTunerSelectDistTest, NodesRangeKeyMatchesMultiNode) {
  if (kExpectedNodes > 1) {
    // Multi-node topology: nNodes=(1,) matches, so the rule forces LL_TREE.
    TunerConfigFile config(makeRangeRule(
        Collective::kAllReduce,
        "tree",
        "ll",
        /*channels=*/-1,
        /*bytesPerRank=*/"*",
        /*nNodes=*/"(1,)",
        /*nLocalRanks=*/"*"));
    ncclx::test::NcclCommRAII commGuard(
        globalRank, numRanks, localRank, bootstrap_.get());
    ncclComm_t comm = commGuard.get();
    assertTopology(comm);
    runCollective(comm, Collective::kAllReduce, 4096);
    algoStats_.verifyExact(comm, "AllReduce", "LL_TREE");
  } else {
    // Single-node 1x8 topology: nNodes=(1,) does NOT match (nNodes==1), so the
    // rule is ignored. Compare a comm carrying the rule against a no-tuner
    // comm.
    ncclx::test::NcclCommRAII noTunerGuard(
        globalRank, numRanks, localRank, bootstrap_.get());
    ncclComm_t noTunerComm = noTunerGuard.get();
    assertTopology(noTunerComm);
    runCollective(noTunerComm, Collective::kAllReduce, 4096);

    TunerConfigFile config(makeRangeRule(
        Collective::kAllReduce,
        "tree",
        "ll",
        /*channels=*/-1,
        /*bytesPerRank=*/"*",
        /*nNodes=*/"(1,)",
        /*nLocalRanks=*/"*"));
    ncclx::test::NcclCommRAII ruleGuard(
        globalRank, numRanks, localRank, bootstrap_.get());
    ncclComm_t ruleComm = ruleGuard.get();
    assertTopology(ruleComm);
    runCollective(ruleComm, Collective::kAllReduce, 4096);

    algoStats_.verifyEqual(noTunerComm, ruleComm, "AllReduce");
  }
}

// Size+topology-keyed matching: a rule keyed by BOTH a [0, 1 MiB] per-rank size
// range AND the actual topology forces tree+LL. An in-range AllReduce (16 KiB
// total, 2 KiB per rank) uses LL_TREE; an out-of-range AllReduce (2 MiB per
// rank > 1 MiB max) does not match the rule, so the forced LL_TREE is absent
// (NCCL default applies).
TEST_F(MetaTunerSelectDistTest, SizeAndTopologyKeyInRange) {
  TunerConfigFile config(makeRule(
      Collective::kAllReduce,
      "tree",
      "ll",
      /*channels=*/-1,
      /*minBytes=*/0,
      /*maxBytes=*/1048576,
      /*nNodes=*/kExpectedNodes,
      /*nLocalRanks=*/kExpectedLocalRanks));
  ncclx::test::NcclCommRAII commGuard(
      globalRank, numRanks, localRank, bootstrap_.get());
  ncclComm_t comm = commGuard.get();
  assertTopology(comm);
  // 4096 floats = 16 KiB total -> 2 KiB per rank, inside [0, 1 MiB].
  runCollective(comm, Collective::kAllReduce, 4096);
  algoStats_.verifyExact(comm, "AllReduce", "LL_TREE");
}

// Size+topology-keyed matching, out-of-range case ("unchanged vs default"
// model): a rule with a [0, 1 MiB] per-rank size range does NOT match a 16 MiB
// AllReduce (2 MiB per rank), so the tuner ignores it. Create a no-tuner comm
// and run a 16 MiB AllReduce on it, then create a comm carrying the
// out-of-range rule and run the SAME 16 MiB AllReduce on it. verifyEqual
// asserts both comms selected the same set of algorithms, proving the
// non-matching rule changed nothing.
TEST_F(MetaTunerSelectDistTest, SizeAndTopologyKeyOutOfRange) {
  // 4194304 floats = 16 MiB total -> 2 MiB per rank (16 MiB / 8 ranks), above
  // the rule's 1 MiB per-rank max.
  constexpr size_t kOutOfRangeCount = 4194304;

  // No-tuner comm: no TunerConfigFile in scope, so the tuner is disabled and
  // NCCL uses its default selection.
  ncclx::test::NcclCommRAII noTunerGuard(
      globalRank, numRanks, localRank, bootstrap_.get());
  ncclComm_t noTunerComm = noTunerGuard.get();
  assertTopology(noTunerComm);
  runCollective(noTunerComm, Collective::kAllReduce, kOutOfRangeCount);

  // Rule comm: a rule covering only [0, 1 MiB] per-rank forcing tree+LL does
  // not match the 16 MiB run (2 MiB per rank), so the tuner ignores it. Keep
  // both comms alive until verifyEqual reads their AlgoStats.
  TunerConfigFile config(makeRule(
      Collective::kAllReduce,
      "tree",
      "ll",
      /*channels=*/-1,
      /*minBytes=*/0,
      /*maxBytes=*/1048576,
      /*nNodes=*/kExpectedNodes,
      /*nLocalRanks=*/kExpectedLocalRanks));
  ncclx::test::NcclCommRAII ruleGuard(
      globalRank, numRanks, localRank, bootstrap_.get());
  ncclComm_t ruleComm = ruleGuard.get();
  assertTopology(ruleComm);
  runCollective(ruleComm, Collective::kAllReduce, kOutOfRangeCount);

  algoStats_.verifyEqual(noTunerComm, ruleComm, "AllReduce");
}

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
