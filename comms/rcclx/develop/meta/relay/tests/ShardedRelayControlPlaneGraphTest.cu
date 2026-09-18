// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * The relay control plane under HIP graph capture.
 *
 * The control plane is pure host code -- shm reads and writes, no stream work
 * -- so under capture it behaves in a way that is easy to get wrong: it
 * executes at CAPTURE time and is not recorded, which means a replay re-runs
 * the collectives but not the publish or the consume that chose their
 * arguments. Put a consume inside a captured region and it happens exactly
 * once, no matter how many times the graph runs. That is the same family as the
 * stale-baked-epoch bugs ShardedRelayGraphCaptureTest guards, arriving from the
 * host side.
 *
 * Hence the contract this file pins:
 *
 *   1. publish and consume belong OUTSIDE the captured region, once per
 * forward.
 *   2. a captured graph is pinned to the plan SHAPE it was captured with,
 * because a graph bakes buffer pointers and sizes. Only the data may vary per
 * replay.
 *   3. the plan is how a serving loop finds out the shape changed, and
 * therefore that it must re-capture.
 *
 * Own target rather than cases added to ShardedRelayGraphCaptureTest, because
 * the control plane requires NCCL_SHARDED_RELAY_MODE_ENABLE at comm creation
 * and only the eager half of that suite's binary pair sets it. Bolting these on
 * would leave them silently inert in the lazy half.
 *
 * As in ShardedRelayControlPlaneTest, nothing here uses a fatal assertion:
 * ASSERT_* returns from the test body, and on one rank of eight that leaves the
 * other seven waiting in a collective it will never join, turning a one-rank
 * failure into an eight-rank hang.
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

namespace {

constexpr int64_t kMs = 1000LL * 1000LL;
constexpr int64_t kForwardTimeoutNs = 60LL * 1000LL * kMs;
constexpr uint32_t kMaxCallsPerPlan = 8;
constexpr size_t kCount = 32ULL * 1024;
// Buffers are sized for the widest configuration so one allocation serves both.
constexpr int kMaxActive = 4;

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
 * be: it is derived identically everywhere.
 */
class ShardedRelayControlPlaneGraphTest : public ::testing::TestWithParam<int> {
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
    CUDACHECK_TEST(cudaStreamCreate(&stream));
    // Dedicated and never aliased: sharing one word between the barrier and the
    // reduction, or reallocating per call, reintroduces the address-reuse
    // write-after-write that ShardedRelayAllReduceTest's barrierSyncOn
    // documents and that ShardedRelayControlPlaneTest hit.
    HIPEXPECT_TEST(hipMalloc(&barrierScratch, sizeof(int32_t)));
    HIPEXPECT_TEST(hipMalloc(&reduceScratch, sizeof(int32_t)));
    const size_t bytes = kCount * kMaxActive * sizeof(int32_t);
    HIPEXPECT_TEST(hipMalloc(&sendBuff, bytes));
    HIPEXPECT_TEST(hipMalloc(&recvBuff, bytes));
    HIPEXPECT_TEST(hipMemset(sendBuff, 0, bytes));
    HIPEXPECT_TEST(hipMemset(recvBuff, 0, bytes));
  }

  void TearDown() override {
    HIPEXPECT_TEST(hipFree(sendBuff));
    HIPEXPECT_TEST(hipFree(recvBuff));
    HIPEXPECT_TEST(hipFree(barrierScratch));
    HIPEXPECT_TEST(hipFree(reduceScratch));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    if (server && checkTcpStoreEnv()) {
      finalizeNcclComm(this->globalRank, server.get());
    }
    NCCLEXPECT_TEST(ncclCommDestroy(this->comm));
    server.reset();
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm{nullptr};
  cudaStream_t stream{nullptr};
  int32_t* barrierScratch{nullptr};
  int32_t* reduceScratch{nullptr};
  int32_t* sendBuff{nullptr};
  int32_t* recvBuff{nullptr};
  std::shared_ptr<c10d::TCPStore> server;

  bool isActive() const {
    return this->globalRank < nActive();
  }
  int nActive() const {
    return GetParam();
  }
  bool isPublisher() const {
    return this->globalRank == 0;
  }

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

  // Every rank must agree before entering or skipping a collective, or a
  // one-rank failure becomes an eight-rank hang.
  bool allRanksOk(bool localOk) {
    return sumAcrossRanks(localOk ? 1 : 0) == this->numRanks;
  }

  static int32_t contribution(int activeIndex, int round) {
    return static_cast<int32_t>(activeIndex + 1 + 10 * round);
  }

  int32_t expectedSum(int round) const {
    int32_t sum = 0;
    for (int i = 0; i < nActive(); i++) {
      sum += contribution(i, round);
    }
    return sum;
  }

  void stageInput(int round) {
    if (!isActive()) {
      return;
    }
    const std::vector<int32_t> host(
        kCount, contribution(this->globalRank, round));
    HIPEXPECT_TEST(hipMemcpy(
        sendBuff,
        host.data(),
        kCount * sizeof(int32_t),
        hipMemcpyHostToDevice));
  }

  void expectReduced(int round, const char* what) {
    if (!isActive()) {
      return;
    }
    std::vector<int32_t> out(kCount, 0);
    HIPEXPECT_TEST(hipMemcpy(
        out.data(), recvBuff, kCount * sizeof(int32_t), hipMemcpyDeviceToHost));
    const int32_t expected = expectedSum(round);
    EXPECT_EQ(out[0], expected) << what << " (round " << round << ")";
    EXPECT_EQ(out[kCount / 2], expected) << what;
    EXPECT_EQ(out[kCount - 1], expected) << what;
  }

  ncclResult_t publishOneCall(uint64_t epoch, size_t count) {
    ncclRelayPlanInfo info{};
    info.nCalls = 1;
    info.opCode = ncclRelayOpAllReduce;
    info.dtype = ncclInt32;
    info.redOp = ncclSum;
    const size_t counts[1] = {count};
    return ncclRelayControlPublish(
        this->comm, epoch, &info, counts, kForwardTimeoutNs);
  }

  ncclResult_t consumeOneCall(uint64_t epoch, size_t* countOut) {
    ncclRelayPlanInfo got{};
    std::vector<size_t> counts(kMaxCallsPerPlan, 0);
    const ncclResult_t res = ncclRelayControlConsume(
        this->comm,
        epoch,
        &got,
        counts.data(),
        kMaxCallsPerPlan,
        kForwardTimeoutNs);
    if (res == ncclSuccess) {
      EXPECT_EQ(got.nCalls, 1u);
      EXPECT_EQ(got.opCode, static_cast<uint32_t>(ncclRelayOpAllReduce));
      if (countOut != nullptr && got.nCalls >= 1) {
        *countOut = counts[0];
      }
    }
    return res;
  }

  void enqueueAllReduce(size_t count) {
    std::vector<int> activeRanks(nActive());
    for (int i = 0; i < nActive(); i++) {
      activeRanks[i] = i;
    }
    const int* allActiveRanks[1] = {activeRanks.data()};
    const void* sendPtrs[1] = {sendBuff};
    void* recvPtrs[1] = {isActive() ? recvBuff : sendBuff};
    const size_t counts[1] = {count};
    NCCLEXPECT_TEST(ncclShardedRelayMultiGroupAllReduce(
        sendPtrs,
        recvPtrs,
        counts,
        ncclInt32,
        ncclSum,
        this->comm,
        this->stream,
        allActiveRanks,
        nActive(),
        1,
        /*lowPrecision=*/0));
  }

  // EndCapture always runs, so a failure inside `body` cannot leave the stream
  // stuck in capture mode and wedge every later case.
  template <typename Body>
  hipGraphExec_t captureGraph(Body&& body) {
    HIPEXPECT_TEST(hipStreamSynchronize(this->stream));
    hipGraph_t graph = nullptr;
    HIPEXPECT_TEST(
        hipStreamBeginCapture(this->stream, hipStreamCaptureModeRelaxed));
    body();
    const hipError_t endErr = hipStreamEndCapture(this->stream, &graph);
    if (endErr != hipSuccess || graph == nullptr) {
      ADD_FAILURE() << "hipStreamEndCapture failed: "
                    << hipGetErrorString(endErr);
      return nullptr;
    }
    hipGraphExec_t exec = nullptr;
    const hipError_t instErr =
        hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    HIPEXPECT_TEST(hipGraphDestroy(graph));
    if (instErr != hipSuccess) {
      ADD_FAILURE() << "hipGraphInstantiate failed: "
                    << hipGetErrorString(instErr);
      return nullptr;
    }
    return exec;
  }

  void replay(hipGraphExec_t exec) {
    HIPEXPECT_TEST(hipGraphLaunch(exec, this->stream));
    HIPEXPECT_TEST(hipStreamSynchronize(this->stream));
  }
};

/**
 * The supported pattern: one publish and one consume per forward, both outside
 * the captured region, with a single graph replayed for each.
 *
 * Only the data changes between replays -- the plan shape is constant, because
 * a graph bakes sizes. Varying the data is what makes this a real check: if the
 * replay were not doing the work, the reduced value would stay at the previous
 * round's sum.
 */
TEST_P(ShardedRelayControlPlaneGraphTest, PerForwardPlansAroundAReplayedGraph) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  barrier();

  // Warm up outside capture first: the first relay call on a comm can build
  // one-shot state, which does not belong inside a graph.
  bool ok = true;
  if (isPublisher()) {
    const ncclResult_t res = publishOneCall(0, kCount);
    EXPECT_EQ(res, ncclSuccess);
    ok = (res == ncclSuccess);
  }
  if (!isActive()) {
    size_t got = 0;
    const ncclResult_t res = consumeOneCall(0, &got);
    EXPECT_EQ(res, ncclSuccess);
    EXPECT_EQ(got, kCount);
    ok = (res == ncclSuccess);
  }
  if (!allRanksOk(ok)) {
    return;
  }
  stageInput(0);
  barrier();
  enqueueAllReduce(kCount);
  HIPEXPECT_TEST(hipStreamSynchronize(this->stream));
  expectReduced(0, "eager warm-up");

  barrier();
  hipGraphExec_t exec = captureGraph([&]() { enqueueAllReduce(kCount); });
  if (!allRanksOk(exec != nullptr)) {
    if (exec != nullptr) {
      HIPEXPECT_TEST(hipGraphExecDestroy(exec));
    }
    return;
  }

  for (int round = 1; round <= 3; round++) {
    bool roundOk = true;
    if (isPublisher()) {
      const ncclResult_t res =
          publishOneCall(static_cast<uint64_t>(round), kCount);
      EXPECT_EQ(res, ncclSuccess) << "publish failed at round " << round;
      roundOk = (res == ncclSuccess);
    }
    if (!isActive()) {
      size_t got = 0;
      const ncclResult_t res =
          consumeOneCall(static_cast<uint64_t>(round), &got);
      EXPECT_EQ(res, ncclSuccess) << "consume failed at round " << round;
      // The shape the graph was captured with; a change here is the signal to
      // re-capture, which the next case covers.
      EXPECT_EQ(got, kCount);
      roundOk = (res == ncclSuccess);
    }
    if (!allRanksOk(roundOk)) {
      break;
    }
    stageInput(round);
    barrier();
    replay(exec);
    expectReduced(round, "graph replay");
  }

  HIPEXPECT_TEST(hipGraphExecDestroy(exec));
  barrier();
}

/**
 * The control plane is safe to call while a capture is in progress on the
 * stream.
 *
 * This is about API legality, not about replay. A capture is invalidated if the
 * region touches a HIP API that is illegal during capture, and publish/consume
 * are host-only today -- shm, atomics, nanosleep -- so they must not disturb
 * it. Pinning that means a later change which adds an allocation, an event or a
 * memcpy to either one fails here rather than in a serving loop that
 * interleaves them with capture.
 *
 * Note what this deliberately does NOT claim: that the consume is replayed.
 * Host code in a captured region executes once, at capture time, and is not
 * recorded, which is exactly why publish and consume belong outside the region
 * as in the case above. That is a HIP property rather than one of ours, so it
 * is stated in this file's contract instead of asserted here.
 */
TEST_P(
    ShardedRelayControlPlaneGraphTest,
    ControlPlaneCallsDoNotInvalidateAnActiveCapture) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  barrier();

  // Warm up outside capture: the first relay call on a comm can build one-shot
  // state, which does not belong inside a graph.
  bool ok = true;
  if (isPublisher()) {
    const ncclResult_t res = publishOneCall(0, kCount);
    EXPECT_EQ(res, ncclSuccess);
    ok = (res == ncclSuccess);
  }
  if (!isActive()) {
    size_t got = 0;
    const ncclResult_t res = consumeOneCall(0, &got);
    EXPECT_EQ(res, ncclSuccess);
    ok = (res == ncclSuccess);
  }
  if (!allRanksOk(ok)) {
    return;
  }
  stageInput(0);
  barrier();
  enqueueAllReduce(kCount);
  HIPEXPECT_TEST(hipStreamSynchronize(this->stream));
  barrier();

  // Publish and consume for the NEXT forward from inside the captured region.
  bool insideOk = true;
  hipGraphExec_t exec = captureGraph([&]() {
    if (isPublisher()) {
      const ncclResult_t res = publishOneCall(1, kCount);
      EXPECT_EQ(res, ncclSuccess) << "publish during capture";
      insideOk = insideOk && (res == ncclSuccess);
    }
    if (!isActive()) {
      size_t got = 0;
      const ncclResult_t res = consumeOneCall(1, &got);
      EXPECT_EQ(res, ncclSuccess) << "consume during capture";
      EXPECT_EQ(got, kCount);
      insideOk = insideOk && (res == ncclSuccess);
    }
    enqueueAllReduce(kCount);
  });

  // A non-null exec is the real assertion: had either call touched an API that
  // is illegal during capture, EndCapture or Instantiate would have failed.
  EXPECT_NE(exec, nullptr)
      << "a control-plane call inside the region invalidated the capture";
  if (!allRanksOk(exec != nullptr && insideOk)) {
    if (exec != nullptr) {
      HIPEXPECT_TEST(hipGraphExecDestroy(exec));
    }
    return;
  }

  // And the graph it produced is still a working graph.
  stageInput(1);
  barrier();
  replay(exec);
  expectReduced(1, "replay of a graph captured around control-plane calls");

  HIPEXPECT_TEST(hipGraphExecDestroy(exec));
  barrier();
}

/**
 * A captured graph is pinned to the plan shape it was captured with, and the
 * plan is how a serving loop learns the shape changed.
 *
 * The control plane cannot detect this itself -- it knows nothing about graphs
 * -- so what matters is that the new shape is delivered faithfully and early,
 * while the caller can still choose to re-capture instead of replaying. That is
 * the property checked here: consume reports the new count, before any
 * collective is enqueued.
 */
TEST_P(ShardedRelayControlPlaneGraphTest, AShapeChangeIsVisibleBeforeReplay) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "requires exactly 8 ranks, got " << this->numRanks;
  }
  barrier();

  const size_t capturedCount = kCount;
  const size_t newCount = kCount / 2;

  bool ok = true;
  if (isPublisher()) {
    const ncclResult_t res = publishOneCall(0, capturedCount);
    EXPECT_EQ(res, ncclSuccess);
    ok = (res == ncclSuccess);
  }
  if (!isActive()) {
    size_t got = 0;
    const ncclResult_t res = consumeOneCall(0, &got);
    EXPECT_EQ(res, ncclSuccess);
    EXPECT_EQ(got, capturedCount);
    ok = (res == ncclSuccess);
  }
  if (!allRanksOk(ok)) {
    return;
  }
  stageInput(0);
  barrier();
  enqueueAllReduce(capturedCount);
  HIPEXPECT_TEST(hipStreamSynchronize(this->stream));
  barrier();

  hipGraphExec_t exec =
      captureGraph([&]() { enqueueAllReduce(capturedCount); });
  if (!allRanksOk(exec != nullptr)) {
    if (exec != nullptr) {
      HIPEXPECT_TEST(hipGraphExecDestroy(exec));
    }
    return;
  }

  // A forward whose shape no longer matches the graph.
  bool changeOk = true;
  if (isPublisher()) {
    const ncclResult_t res = publishOneCall(1, newCount);
    EXPECT_EQ(res, ncclSuccess);
    changeOk = (res == ncclSuccess);
  }
  if (!isActive()) {
    size_t got = 0;
    const ncclResult_t res = consumeOneCall(1, &got);
    EXPECT_EQ(res, ncclSuccess);
    // The whole point: the helper sees the new shape while it still has the
    // choice, rather than discovering it by replaying a graph built for the old
    // one.
    EXPECT_EQ(got, newCount)
        << "consume must report the published shape, not the captured one";
    EXPECT_NE(got, capturedCount);
    changeOk = (res == ncclSuccess);
  }
  if (!allRanksOk(changeOk)) {
    HIPEXPECT_TEST(hipGraphExecDestroy(exec));
    return;
  }

  // Re-capture for the new shape, which is what the signal is for, and confirm
  // the result is correct at the new size.
  HIPEXPECT_TEST(hipGraphExecDestroy(exec));
  barrier();
  stageInput(1);
  barrier();
  enqueueAllReduce(newCount);
  HIPEXPECT_TEST(hipStreamSynchronize(this->stream));

  if (isActive()) {
    std::vector<int32_t> out(newCount, 0);
    HIPEXPECT_TEST(hipMemcpy(
        out.data(),
        recvBuff,
        newCount * sizeof(int32_t),
        hipMemcpyDeviceToHost));
    const int32_t expected = expectedSum(1);
    EXPECT_EQ(out[0], expected) << "re-captured shape";
    EXPECT_EQ(out[newCount - 1], expected) << "re-captured shape";
  }
  barrier();
}

// Both shapes, so a failure names the configuration it happened in.
INSTANTIATE_TEST_SUITE_P(
    ActiveWidth,
    ShardedRelayControlPlaneGraphTest,
    ::testing::Values(2, 4),
    [](const ::testing::TestParamInfo<int>& info) {
      return std::to_string(info.param) + "Active";
    });

int main(int argc, char* argv[]) {
  // Must precede the first communicator: the control plane is built at
  // commInitRank under this switch, and NCCL caches the parameter on first
  // read.
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
