// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * Production-shape tests for the relay control plane.
 *
 * RelayControlTest covers the protocol in isolation, with forked children
 * standing in for peers. This file covers what that deliberately cannot: the
 * communicator-bound layer, on a real 8-rank comm, driving real relay
 * collectives.
 *
 * The shape is the deployment's, not a convenience:
 *
 *   ranks 0..nActive-1   ACTIVE   run the model, know the plan
 *   rank  0              PUBLISHER  the one rank that writes it
 *   ranks nActive..7     HELPER   know nothing until they consume a plan
 *
 * A helper here is given no counts, no opcode and no call count by the test. It
 * learns all three from the segment and then enqueues collectives on that basis
 * alone. That is the property under test: if the plan did not arrive intact,
 * the helper's relay calls would not match its peers' and the collective would
 * hang or corrupt rather than quietly pass.
 *
 * Correctness of the collectives themselves is covered by the four
 * ShardedRelay*Test suites. What is checked here is that a published plan
 * drives them -- so allreduce is verified numerically as the representative
 * case, and the other three are required to complete, which for a symmetric
 * collective across 8 ranks already means every rank agreed on counts and
 * route.
 *
 * NOTHING IN THIS FILE USES A FATAL ASSERTION. ASSERT_* returns from the test
 * body, which on one rank of eight means the other seven are left waiting
 * inside a collective that rank will never join: a single-rank control-plane
 * failure would present as an eight-rank hang with no attribution instead of a
 * named failure on the rank that caused it. Every check is therefore EXPECT_*,
 * the decision to enter the collectives is taken unanimously through a
 * reduction, and loop bounds come from values every rank holds locally.
 *
 * That applies to SetUp and TearDown too, which is easy to miss because the
 * FAIL()-based *CHECK_TEST macros read like the ones every other test in this
 * directory uses. They are deliberately NOT defined here so the rule cannot be
 * broken by habit -- only the ADD_FAILURE()-based *EXPECT_TEST forms exist.
 * SetUp records its outcome in setupOk instead, and requireEightRanks() gates
 * every body on it.
 *
 * One case is beyond rescue: a rank that fails to CREATE its communicator can
 * never join the collective its peers are already in, and nothing checkable
 * from inside a single rank can detect a peer that will never arrive. Non-fatal
 * handling still turns that into an attributed skip plus a clean teardown
 * rather than a null dereference.
 */

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "comm.h"
#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"
#include "meta/relay/relay_control.h"
#include "nccl.h"

// FAIL() is only usable in a void function, and the reductions below have to
// return a value to every rank.
#define HIPEXPECT_TEST(cmd)                                                \
  do {                                                                     \
    hipError_t error = cmd;                                                \
    if (error != hipSuccess) {                                             \
      ADD_FAILURE() << "HIP error: " << hipGetErrorString(error) << " at " \
                    << __FILE__ << ":" << __LINE__;                        \
    }                                                                      \
  } while (0)

#define NCCLEXPECT_TEST(cmd)                                                  \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    if (result != ncclSuccess) {                                              \
      ADD_FAILURE() << "NCCL error: " << ncclGetErrorString(result) << " at " \
                    << __FILE__ << ":" << __LINE__;                           \
    }                                                                         \
  } while (0)

#define CUDAEXPECT_TEST(cmd)                                                 \
  do {                                                                       \
    cudaError_t error = cmd;                                                 \
    if (error != cudaSuccess) {                                              \
      ADD_FAILURE() << "CUDA error: " << cudaGetErrorString(error) << " at " \
                    << __FILE__ << ":" << __LINE__;                          \
    }                                                                        \
  } while (0)

namespace {

constexpr int64_t kMs = 1000LL * 1000LL;
// Generous, because a helper waits here while its peers do real GPU work.
constexpr int64_t kForwardTimeoutNs = 60LL * 1000LL * kMs;
// Used only where a timeout is the expected outcome.
constexpr int64_t kShortTimeoutNs = 300LL * kMs;

constexpr uint32_t kMaxCallsPerPlan = 8;
constexpr size_t kElemsPerCall = 64ULL * 1024;

} // namespace

/**
 * Parameterised on the number of ACTIVE ranks: 4 active + 4 helper, and 2
 * active
 * + 6 helper. Both are real deployment shapes, they select different relay
 * routes, and the 2-active case puts six ranks on the consume path rather than
 * four -- so a bug in how helpers are counted or waited on shows up in one and
 * not the other.
 *
 * gtest runs cases in a fixed order, so every rank walks the two configurations
 * in the same sequence. The parameter is never communicated and never needs to
 * be: it is derived identically on every rank.
 */
class ShardedRelayControlPlaneTest : public ::testing::TestWithParam<int> {
 public:
  void SetUp() override {
    int localSize;
    std::tie(this->localRank, this->globalRank, this->numRanks, localSize) =
        getTcpStoreOrMpiInfo();
    const bool isServer = (this->globalRank == 0);
    if (checkTcpStoreEnv()) {
      server = createTcpStore(isServer);
    } else if (isServer) {
      server = createTcpStore(true);
    }
    this->comm = createNcclComm(
        this->globalRank,
        this->numRanks,
        this->localRank,
        false,
        nullptr,
        server.get());
    // Non-fatal from here down. This file's whole premise is that no rank may
    // abandon a collective its peers are entering, and FAIL() returns from
    // SetUp() on one rank while the other seven walk into the test body's first
    // all-reduce -- producing exactly the eight-rank hang the rule exists to
    // prevent, with the real cause buried behind a harness timeout.
    CUDAEXPECT_TEST(cudaStreamCreate(&stream));
    // Two dedicated, persistent scratch words, never aliased with each other.
    //
    // Allocating these per call is a real hazard rather than a style point:
    // hipFree followed by hipMalloc readily returns the SAME device address, so
    // one all-reduce's write -- not guaranteed to be retired when the host
    // stages the next value -- can land on top of that value. That is the
    // documented flaky "expected X but got 0" mode called out in
    // ShardedRelayAllReduceTest's barrierSyncOn, and an earlier version of this
    // file hit exactly it.
    HIPEXPECT_TEST(hipMalloc(&barrierScratch, sizeof(int32_t)));
    HIPEXPECT_TEST(hipMalloc(&reduceScratch, sizeof(int32_t)));

    setupOk = this->comm != nullptr && this->stream != nullptr &&
        barrierScratch != nullptr && reduceScratch != nullptr;
  }

  void TearDown() override {
    // Non-fatal for the same reason as SetUp: a rank that returns early from
    // teardown skips ncclCommDestroy, and a comm destroyed on seven of eight
    // ranks is its own hang.
    if (barrierScratch != nullptr) {
      HIPEXPECT_TEST(hipFree(barrierScratch));
    }
    if (reduceScratch != nullptr) {
      HIPEXPECT_TEST(hipFree(reduceScratch));
    }
    if (server && checkTcpStoreEnv()) {
      finalizeNcclComm(this->globalRank, server.get());
    }
    // Comm BEFORE stream. A comm holds internal references to the streams it
    // was used with, so destroying the stream first can leave teardown touching
    // a freed stream; the reverse order has nothing depending on it.
    if (this->comm != nullptr) {
      NCCLEXPECT_TEST(ncclCommDestroy(this->comm));
      this->comm = nullptr;
    }
    if (this->stream != nullptr) {
      CUDAEXPECT_TEST(cudaStreamDestroy(this->stream));
      this->stream = nullptr;
    }
    server.reset();
  }

 protected:
  // False if any part of SetUp() failed. Every test body checks it before
  // entering a collective, so a rank whose setup failed reports that cause and
  // skips instead of dereferencing a null comm.
  //
  // This does NOT rescue a comm-creation failure: a rank that never built a
  // comm cannot join the collective its peers are already in, and no gate
  // reachable from inside one rank can detect a peer that will never arrive.
  // What it does buy is a clean, attributed skip instead of a segfault, and a
  // teardown that still runs.
  bool setupOk{false};
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm{nullptr};
  cudaStream_t stream{nullptr};
  int32_t* barrierScratch{nullptr};
  int32_t* reduceScratch{nullptr};
  std::shared_ptr<c10d::TCPStore> server;

  bool requireEightRanks() {
    if (this->numRanks != 8) {
      return false;
    }
    return setupOk;
  }

  // Cross-rank barrier. Also the mechanism for the unanimity checks below: a
  // sum over a per-rank 0/1 tells every rank what every other rank saw, which
  // is the only way to assert a property of the whole comm from inside one
  // rank.
  void barrier() {
    HIPEXPECT_TEST(
        hipMemsetAsync(barrierScratch, 0, sizeof(int32_t), this->stream));
    NCCLEXPECT_TEST(ncclAllReduce(
        barrierScratch,
        barrierScratch,
        1,
        ncclInt32,
        ncclSum,
        this->comm,
        this->stream));
    HIPEXPECT_TEST(hipStreamSynchronize(this->stream));
  }

  // Everything is enqueued on the one stream, so the staging copy, the
  // reduction and the read-back are ordered against each other by construction
  // rather than by the host happening to be ahead.
  int sumAcrossRanks(int value) {
    const int32_t host = static_cast<int32_t>(value);
    int32_t out = 0;
    HIPEXPECT_TEST(hipMemcpyAsync(
        reduceScratch,
        &host,
        sizeof(int32_t),
        hipMemcpyHostToDevice,
        this->stream));
    NCCLEXPECT_TEST(ncclAllReduce(
        reduceScratch,
        reduceScratch,
        1,
        ncclInt32,
        ncclSum,
        this->comm,
        this->stream));
    HIPEXPECT_TEST(hipMemcpyAsync(
        &out,
        reduceScratch,
        sizeof(int32_t),
        hipMemcpyDeviceToHost,
        this->stream));
    HIPEXPECT_TEST(hipStreamSynchronize(this->stream));
    return static_cast<int>(out);
  }

  static int32_t contribution(int activeIndex) {
    return static_cast<int32_t>(activeIndex + 1);
  }

  static int32_t expectedAllReduceSum(int nActive) {
    return static_cast<int32_t>(nActive * (nActive + 1) / 2);
  }

  // One relay call, dispatched on the opcode the plan carried. Buffers are
  // sized for the widest of the four (count * nActive elements), so the same
  // pair serves every opcode.
  ncclResult_t enqueueRelayCall(
      uint32_t opCode,
      const void* sendBuff,
      void* recvBuff,
      size_t count,
      const int* const* allActiveRanks,
      int nActive) {
    const void* sendPtrs[1] = {sendBuff};
    void* recvPtrs[1] = {recvBuff};
    size_t counts[1] = {count};
    switch (opCode) {
      case rcclx::relay::kRelayOpAllReduce:
        return ncclShardedRelayMultiGroupAllReduce(
            sendPtrs,
            recvPtrs,
            counts,
            ncclInt32,
            ncclSum,
            this->comm,
            this->stream,
            allActiveRanks,
            nActive,
            1,
            /*lowPrecision=*/0);
      case rcclx::relay::kRelayOpReduceScatter:
        return ncclShardedRelayMultiGroupReduceScatter(
            sendPtrs,
            recvPtrs,
            counts,
            ncclInt32,
            ncclSum,
            this->comm,
            this->stream,
            allActiveRanks,
            nActive,
            1,
            /*lowPrecision=*/0);
      case rcclx::relay::kRelayOpAllGather:
        return ncclShardedRelayMultiGroupAllGather(
            sendPtrs,
            recvPtrs,
            counts,
            ncclInt32,
            this->comm,
            this->stream,
            allActiveRanks,
            nActive,
            1,
            /*lowPrecision=*/0);
      case rcclx::relay::kRelayOpAllToAll:
        return ncclShardedRelayMultiGroupAllToAll(
            sendPtrs,
            recvPtrs,
            counts,
            ncclInt32,
            this->comm,
            this->stream,
            allActiveRanks,
            nActive,
            1,
            /*lowPrecision=*/0);
      default:
        return ncclInvalidArgument;
    }
  }

  /**
   * One forward, exactly as the deployment runs it.
   *
   * `counts` is what the ACTIVE side decided. The helper side is handed
   * nothing: it is passed only the epoch and must recover opcode, call count
   * and counts from the segment. `verifySum` is honoured on active ranks for
   * allreduce.
   *
   * Each call gets its own buffers. Sharing one pair across the calls of a plan
   * would be a write-after-read hazard between ranks -- a rank that finished
   * call i racing ahead to overwrite the send buffer a peer is still reading --
   * which is an artifact of the test, not of the deployment, where every relay
   * call has its own tensor.
   */
  void runForward(
      uint64_t epoch,
      uint32_t opCode,
      const std::vector<size_t>& counts,
      int nActive,
      bool verifySum) {
    std::vector<int> activeStorage(nActive);
    for (int i = 0; i < nActive; i++) {
      activeStorage[i] = i;
    }
    const int* allActiveRanks[1] = {activeStorage.data()};
    const bool isActive = this->globalRank < nActive;
    const bool isPublisher = (this->globalRank == 0);

    bool localOk = true;
    ncclRelayPlanInfo info{};
    info.nCalls = static_cast<uint32_t>(counts.size());
    info.opCode = opCode;
    info.dtype = ncclInt32;
    info.redOp = ncclSum;

    if (isPublisher) {
      const ncclResult_t res = ncclRelayControlPublish(
          this->comm, epoch, &info, counts.data(), kForwardTimeoutNs);
      EXPECT_EQ(res, ncclSuccess) << "publish failed at epoch " << epoch;
      localOk = (res == ncclSuccess);
    }

    if (!isActive) {
      // The helper's entire knowledge of this forward comes from here.
      ncclRelayPlanInfo got{};
      std::vector<size_t> gotCounts(kMaxCallsPerPlan, 0);
      const ncclResult_t res = ncclRelayControlConsume(
          this->comm,
          epoch,
          &got,
          gotCounts.data(),
          kMaxCallsPerPlan,
          kForwardTimeoutNs);
      EXPECT_EQ(res, ncclSuccess) << "consume failed at epoch " << epoch
                                  << " on rank " << this->globalRank;
      if (res == ncclSuccess) {
        EXPECT_EQ(got.opCode, opCode);
        EXPECT_EQ(got.nCalls, counts.size());
        EXPECT_EQ(got.dtype, static_cast<uint32_t>(ncclInt32));
        if (got.nCalls == counts.size()) {
          gotCounts.resize(got.nCalls);
          // Only meaningful because the test knows the answer; the helper does
          // not. This is the assertion the whole file exists for.
          EXPECT_EQ(gotCounts, counts) << "plan did not survive the segment";
        }
      } else {
        localOk = false;
      }
    }

    // Deciding to proceed has to be unanimous. Every assertion above is
    // non-fatal on purpose: a fatal one returns from the test body on the rank
    // that tripped it, leaving the other seven waiting in a collective it will
    // never join, so a one-rank control-plane failure would present as an
    // eight-rank hang instead of a named failure. This gate is itself a
    // collective every rank reaches, and either all ranks continue or all
    // return.
    if (sumAcrossRanks(localOk ? 1 : 0) != this->numRanks) {
      return;
    }

    // Loop control comes from the test's own plan, which every rank holds
    // locally, rather than from the consumed copy: the number of collectives a
    // rank enters must never be able to differ from its peers'. Whether the
    // consumed copy matched is asserted above, where a mismatch is a named
    // failure rather than a divergence.
    for (size_t i = 0; i < counts.size(); i++) {
      const size_t count = counts[i];
      const size_t bytes = count * nActive * sizeof(int32_t);
      int32_t* sendBuff = nullptr;
      int32_t* recvBuff = nullptr;
      HIPEXPECT_TEST(hipMalloc(&sendBuff, bytes));
      HIPEXPECT_TEST(hipMalloc(&recvBuff, bytes));
      HIPEXPECT_TEST(hipMemset(sendBuff, 0, bytes));
      HIPEXPECT_TEST(hipMemset(recvBuff, 0, bytes));

      if (isActive) {
        std::vector<int32_t> host(count, contribution(this->globalRank));
        HIPEXPECT_TEST(hipMemcpy(
            sendBuff,
            host.data(),
            count * sizeof(int32_t),
            hipMemcpyHostToDevice));
      }
      // Every rank is at the same call before any of them starts it, so a
      // mismatch is a plan-delivery failure rather than a straggler.
      barrier();

      // Non-fatal for the same reason as above: bailing here would leave the
      // buffers unfreed and, worse, the peers inside the next barrier.
      EXPECT_EQ(
          enqueueRelayCall(
              opCode,
              sendBuff,
              isActive ? recvBuff : sendBuff,
              count,
              allActiveRanks,
              nActive),
          ncclSuccess)
          << "relay call " << i << " failed at epoch " << epoch;
      HIPEXPECT_TEST(hipStreamSynchronize(this->stream));

      if (verifySum && isActive && opCode == rcclx::relay::kRelayOpAllReduce) {
        std::vector<int32_t> out(count, 0);
        HIPEXPECT_TEST(hipMemcpy(
            out.data(),
            recvBuff,
            count * sizeof(int32_t),
            hipMemcpyDeviceToHost));
        const int32_t expected = expectedAllReduceSum(nActive);
        // Spot-check the ends and the middle rather than every element: a
        // control-plane fault shows up as a whole-buffer disagreement, and the
        // element-exact case is already covered by the allreduce suite.
        EXPECT_EQ(out[0], expected) << "epoch " << epoch << " call " << i;
        EXPECT_EQ(out[count / 2], expected) << "epoch " << epoch;
        EXPECT_EQ(out[count - 1], expected) << "epoch " << epoch;
      }

      HIPEXPECT_TEST(hipFree(sendBuff));
      HIPEXPECT_TEST(hipFree(recvBuff));
    }
  }
};

/**
 * The setup path, asserted across the whole comm rather than locally.
 *
 * A control plane that came up on only some ranks is the failure mode the
 * unanimity vote exists to prevent, and it is precisely the one a per-rank
 * EXPECT_TRUE cannot see: the ranks that have a segment would wait for peers
 * that took the other path. Summing the flag is what turns "I am ready" into
 * "we are all ready".
 */
TEST_P(ShardedRelayControlPlaneTest, ControlPlaneComesUpOnEveryRankOrNone) {
  if (!requireEightRanks()) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  const bool ready = rcclx::relay::relayControlReady(this->comm);
  EXPECT_TRUE(ready) << "rank " << this->globalRank << " has no control plane";
  const int readyCount = sumAcrossRanks(ready ? 1 : 0);
  EXPECT_EQ(readyCount, this->numRanks)
      << "control plane is up on " << readyCount << " of " << this->numRanks
      << " ranks, which is the split state unanimity is meant to exclude";
}

TEST_P(ShardedRelayControlPlaneTest, AllReduceForwardDrivenByThePublishedPlan) {
  if (!requireEightRanks()) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  const int nActive = GetParam();
  barrier();
  runForward(
      /*epoch=*/0,
      rcclx::relay::kRelayOpAllReduce,
      {kElemsPerCall, kElemsPerCall / 2, kElemsPerCall / 4},
      nActive,
      /*verifySum=*/true);
  barrier();
}

/**
 * Counts change every forward, which is the case the TCP-store design existed
 * to serve and the reason none of this can be hoisted to init.
 */
TEST_P(ShardedRelayControlPlaneTest, CountsAndCallCountVaryAcrossForwards) {
  if (!requireEightRanks()) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  const int nActive = GetParam();
  barrier();

  const std::vector<std::vector<size_t>> plans = {
      {kElemsPerCall},
      {1024, 2048, 4096},
      {kElemsPerCall / 8},
      {512, 512, 512, 512, 512},
      {kElemsPerCall, 64},
  };
  for (uint64_t epoch = 0; epoch < plans.size(); epoch++) {
    runForward(
        epoch,
        rcclx::relay::kRelayOpAllReduce,
        plans[epoch],
        nActive,
        /*verifySum=*/true);
  }
  barrier();
}

/**
 * All four entry points, at whichever width the parameter selects. Only the
 * allreduce is verified numerically; the other three are required to complete,
 * which for a symmetric collective across 8 ranks already means every rank
 * agreed on counts and route.
 */
TEST_P(ShardedRelayControlPlaneTest, AllFourCollectives) {
  if (!requireEightRanks()) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  const int nActive = GetParam();
  barrier();

  const uint32_t opCodes[] = {
      rcclx::relay::kRelayOpAllReduce,
      rcclx::relay::kRelayOpReduceScatter,
      rcclx::relay::kRelayOpAllGather,
      rcclx::relay::kRelayOpAllToAll};
  for (uint64_t epoch = 0; epoch < 4; epoch++) {
    runForward(
        epoch,
        opCodes[epoch],
        {kElemsPerCall / 4, kElemsPerCall / 8},
        nActive,
        /*verifySum=*/epoch == 0);
  }
  barrier();
}

/**
 * Shutdown is an opcode, not a third entry point, so this is the check that a
 * graceful stop needs no extra API.
 */
TEST_P(ShardedRelayControlPlaneTest, ShutdownOpCodeReachesEveryHelper) {
  if (!requireEightRanks()) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  const int nActive = GetParam();
  barrier();

  int sawShutdown = 0;
  if (this->globalRank == 0) {
    ncclRelayPlanInfo info{};
    info.opCode = ncclRelayOpShutdown;
    info.nCalls = 0;
    EXPECT_EQ(
        ncclRelayControlPublish(
            this->comm, 0, &info, nullptr, kForwardTimeoutNs),
        ncclSuccess);
  }
  if (this->globalRank >= nActive) {
    ncclRelayPlanInfo got{};
    std::vector<size_t> counts(kMaxCallsPerPlan, 0);
    const ncclResult_t res = ncclRelayControlConsume(
        this->comm,
        0,
        &got,
        counts.data(),
        kMaxCallsPerPlan,
        kForwardTimeoutNs);
    EXPECT_EQ(res, ncclSuccess);
    if (res == ncclSuccess) {
      EXPECT_EQ(got.opCode, static_cast<uint32_t>(ncclRelayOpShutdown));
      EXPECT_EQ(got.nCalls, 0u);
      sawShutdown = 1;
    }
  }

  EXPECT_EQ(sumAcrossRanks(sawShutdown), this->numRanks - nActive);
  barrier();
}

/**
 * Only rank 0 may publish. There is one ring per communicator, so a second
 * publisher would race the same seqlock -- and because every active rank knows
 * the same token count their plans are byte-identical, which means the damage
 * would be invisible in the data and would surface as a spurious desync
 * somewhere else entirely. Rejecting it makes the misuse an error at the call
 * site.
 */
TEST_P(ShardedRelayControlPlaneTest, OnlyOneRankMayPublish) {
  if (!requireEightRanks()) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  barrier();

  ncclRelayPlanInfo info{};
  info.nCalls = 1;
  info.opCode = ncclRelayOpAllReduce;
  info.dtype = ncclInt32;
  size_t counts[1] = {kElemsPerCall};

  if (this->globalRank == 0) {
    EXPECT_EQ(
        ncclRelayControlPublish(
            this->comm, 0, &info, counts, kForwardTimeoutNs),
        ncclSuccess);
  }
  barrier();
  if (this->globalRank == 1) {
    // Active, and it knows the plan, but it is not the publisher.
    EXPECT_EQ(
        ncclRelayControlPublish(this->comm, 1, &info, counts, kShortTimeoutNs),
        ncclInvalidArgument);
  }
  barrier();
}

/**
 * Capacity is a runtime parameter, so exceeding it must be a clean rejection
 * naming the parameter rather than a truncated plan -- a truncated plan is a
 * hang, since the helper would enqueue fewer calls than its peers.
 */
TEST_P(ShardedRelayControlPlaneTest, PlanBeyondCapacityIsRejected) {
  if (!requireEightRanks()) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  barrier();
  if (this->globalRank == 0) {
    const uint32_t capacity = rcclx::relay::relayControlConfiguredMaxCalls();
    std::vector<size_t> counts(capacity + 1, 64);
    ncclRelayPlanInfo info{};
    info.nCalls = capacity + 1;
    info.opCode = ncclRelayOpAllReduce;
    info.dtype = ncclInt32;
    EXPECT_EQ(
        ncclRelayControlPublish(
            this->comm, 0, &info, counts.data(), kShortTimeoutNs),
        ncclInvalidArgument);
  }
  barrier();
}

/**
 * Test: the exported wrappers enforce the contract their own header states.
 *
 * flags and reserved are both documented "must be 0". flags was validated
 * downstream, so a violation was at least reported; reserved was inspected by
 * nothing at all, so non-zero bytes in a field reserved for future meaning were
 * accepted and then silently dropped on the way to the internal record. Once
 * something does start using them, that silence is the bug: a caller compiled
 * against a newer header would set a field the library quietly discards.
 *
 * Only rank 0 publishes, and the calls are rejected before touching the
 * segment, so no epoch is consumed and the barriers keep every other rank in
 * step.
 */
TEST_P(
    ShardedRelayControlPlaneTest,
    ExportedPublishRejectsReservedFieldMisuse) {
  if (!requireEightRanks()) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  barrier();
  if (this->globalRank == 0) {
    size_t counts[1] = {64};
    ncclRelayPlanInfo base{};
    base.nCalls = 1;
    base.opCode = ncclRelayOpAllReduce;
    base.dtype = ncclInt32;

    ncclRelayPlanInfo badFlags = base;
    badFlags.flags = 1;
    EXPECT_EQ(
        ncclRelayControlPublish(
            this->comm, 0, &badFlags, counts, kShortTimeoutNs),
        ncclInvalidArgument);

    for (size_t i = 0; i < sizeof(base.reserved) / sizeof(base.reserved[0]);
         i++) {
      ncclRelayPlanInfo badReserved = base;
      badReserved.reserved[i] = 1;
      EXPECT_EQ(
          ncclRelayControlPublish(
              this->comm, 0, &badReserved, counts, kShortTimeoutNs),
          ncclInvalidArgument)
          << "reserved[" << i << "] must be rejected, not silently dropped";
    }

    // nCalls > 0 with no counts array, caught at the boundary so the diagnostic
    // names the exported symbol the caller actually called.
    EXPECT_EQ(
        ncclRelayControlPublish(this->comm, 0, &base, nullptr, kShortTimeoutNs),
        ncclInvalidArgument);
  }
  barrier();
}

/**
 * Fault injection: the publisher stops. The store this replaced had a wait()
 * timeout, so removing the store must not remove that property -- the helpers
 * must fail, bounded, rather than wait forever.
 *
 * Runs last (Z_ prefix) because a timeout deliberately poisons the segment for
 * every rank, which is the point: one stuck rank should produce a single
 * attributed cause rather than eight independent timeouts.
 */
TEST_P(ShardedRelayControlPlaneTest, Z_HelpersTimeOutWhenThePublisherStops) {
  if (!requireEightRanks()) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  const int nActive = GetParam();
  barrier();

  // Epoch 0 is published and consumed normally, so the helpers are registered
  // and the failure below is a stalled publisher rather than a cold start.
  if (this->globalRank == 0) {
    ncclRelayPlanInfo info{};
    info.nCalls = 1;
    info.opCode = ncclRelayOpAllReduce;
    info.dtype = ncclInt32;
    size_t counts[1] = {kElemsPerCall};
    EXPECT_EQ(
        ncclRelayControlPublish(
            this->comm, 0, &info, counts, kForwardTimeoutNs),
        ncclSuccess);
  }
  if (this->globalRank >= nActive) {
    ncclRelayPlanInfo got{};
    std::vector<size_t> counts(kMaxCallsPerPlan, 0);
    EXPECT_EQ(
        ncclRelayControlConsume(
            this->comm,
            0,
            &got,
            counts.data(),
            kMaxCallsPerPlan,
            kForwardTimeoutNs),
        ncclSuccess);
  }
  barrier();

  // Epoch 1 is never published.
  int failed = 0;
  if (this->globalRank >= nActive) {
    ncclRelayPlanInfo got{};
    std::vector<size_t> counts(kMaxCallsPerPlan, 0);
    const ncclResult_t res = ncclRelayControlConsume(
        this->comm, 1, &got, counts.data(), kMaxCallsPerPlan, kShortTimeoutNs);
    EXPECT_EQ(res, ncclInternalError)
        << "rank " << this->globalRank << " should have timed out";
    failed = (res == ncclInternalError) ? 1 : 0;
  }
  EXPECT_EQ(sumAcrossRanks(failed), this->numRanks - nActive);
  barrier();
}

// Both shapes, so a failure names the configuration it happened in.
/**
 * Test: a consume that produces no plan leaves the caller's record ALONE.
 *
 * The failure mode this pins down is specific. A zeroed ncclRelayPlanInfo is
 * nCalls 0 with opCode 0, and opCode 0 is ncclRelayOpShutdown -- a perfectly
 * valid instruction to stop. So a wrapper that copies out an untouched local on
 * every failure hands a timed-out or aborted helper something indistinguishable
 * from "the publisher told you to shut down". A caller that logged the plan
 * before checking the result would report an orderly shutdown for what was
 * actually a stalled peer.
 *
 * The sentinel below is deliberately not a plausible plan, so the assertion
 * fails whether the wrapper zeroes the record or writes a real one.
 *
 * Z_ prefixed and declared LAST in this file: it times out on purpose, which
 * poisons the segment for every rank, so nothing that expects a healthy segment
 * may run after it.
 */
TEST_P(
    ShardedRelayControlPlaneTest,
    Z_ExportedConsumeLeavesThePlanUntouchedOnFailure) {
  if (!requireEightRanks()) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  barrier();

  // Nobody publishes this epoch, so every rank's consume must time out. Rank 0
  // is the publisher and does not consume, so it just waits at the barrier.
  if (this->globalRank != 0) {
    constexpr uint32_t kSentinel = 0xABCDu;
    ncclRelayPlanInfo info{};
    info.nCalls = kSentinel;
    info.opCode = kSentinel;
    info.dtype = kSentinel;
    info.redOp = kSentinel;
    size_t counts[kMaxCallsPerPlan] = {};

    const ncclResult_t res = ncclRelayControlConsume(
        this->comm,
        /*epoch=*/0,
        &info,
        counts,
        kMaxCallsPerPlan,
        kShortTimeoutNs);
    EXPECT_NE(res, ncclSuccess);
    EXPECT_EQ(info.nCalls, kSentinel)
        << "a failed consume must not overwrite the caller's plan record";
    EXPECT_EQ(info.opCode, kSentinel)
        << "opCode 0 would be ncclRelayOpShutdown, i.e. a failure that reads as "
           "a valid instruction to stop";
    EXPECT_EQ(info.dtype, kSentinel);
    EXPECT_EQ(info.redOp, kSentinel);
  }
  barrier();
}

INSTANTIATE_TEST_SUITE_P(
    ActiveWidth,
    ShardedRelayControlPlaneTest,
    ::testing::Values(2, 4),
    [](const ::testing::TestParamInfo<int>& info) {
      return std::to_string(info.param) + "Active";
    });

int main(int argc, char* argv[]) {
  // The control plane is built at commInitRank under the same switch as eager
  // one-shot region creation, and NCCL caches the parameter on first read, so
  // this has to happen before any comm exists.
  setenv("NCCL_SHARDED_RELAY_MODE_ENABLE", "1", /*overwrite=*/1);
  setenv(
      "NCCL_RELAY_CONTROL_MAX_CALLS",
      std::to_string(kMaxCallsPerPlan).c_str(),
      /*overwrite=*/1);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
