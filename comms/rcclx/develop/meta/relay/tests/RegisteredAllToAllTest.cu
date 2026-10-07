// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <tuple>
#include <vector>

#include "bootstrap.h"
#include "comm.h"
#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"
#include "meta/relay/registered_alltoall.h"
#include "nccl.h"

#define HIPCHECK_TEST(cmd)                                          \
  do {                                                              \
    hipError_t error = cmd;                                         \
    if (error != hipSuccess) {                                      \
      FAIL() << "HIP error: " << hipGetErrorString(error) << " at " \
             << __FILE__ << ":" << __LINE__;                        \
    }                                                               \
  } while (0)

#define HIPEXPECT_TEST(cmd)                                                \
  do {                                                                     \
    hipError_t error = cmd;                                                \
    if (error != hipSuccess) {                                             \
      ADD_FAILURE() << "HIP error: " << hipGetErrorString(error) << " at " \
                    << __FILE__ << ":" << __LINE__;                        \
    }                                                                      \
  } while (0)

namespace {

constexpr int kActive = 4;
constexpr int kSyncTimeoutSec = 30;
constexpr size_t kElement = sizeof(uint16_t);

// The [redacted] Ulysses layouts at a reduced row count. Scatter: 18432-element
// source rows, destination d's 2560 elements at d * 4608; destination receives
// [s * rows + t][2560]. Gather: source [d * rows + t][2048]; destination
// receives [t][s * 2048 + e].
enum class Kind { Scatter, Gather };

struct Shape {
  Kind kind;
  size_t rows;
};

ncclRegisteredAllToAllLayout layoutFor(const Shape& shape) {
  ncclRegisteredAllToAllLayout layout{};
  layout.rows = shape.rows;
  if (shape.kind == Kind::Scatter) {
    layout.rowBytes = 2560 * kElement;
    layout.sendRowStride = 18432 * kElement;
    layout.sendPeerStride = 4608 * kElement;
    layout.recvRowStride = 2560 * kElement;
    layout.recvPeerStride = shape.rows * 2560 * kElement;
  } else {
    layout.rowBytes = 2048 * kElement;
    layout.sendRowStride = 2048 * kElement;
    layout.sendPeerStride = shape.rows * 2048 * kElement;
    layout.recvRowStride = kActive * 2048 * kElement;
    layout.recvPeerStride = 2048 * kElement;
  }
  return layout;
}

size_t sendBytes(const ncclRegisteredAllToAllLayout& l) {
  return (kActive - 1) * l.sendPeerStride + (l.rows - 1) * l.sendRowStride +
      l.rowBytes;
}

size_t recvBytes(const ncclRegisteredAllToAllLayout& l) {
  return (kActive - 1) * l.recvPeerStride + (l.rows - 1) * l.recvRowStride +
      l.rowBytes;
}

uint16_t tag(int source, int dest, size_t row, size_t element, int generation) {
  return static_cast<uint16_t>(
      (source * 0x3001) ^ (dest * 0x0503) ^ (row * 131) ^ (element * 7) ^
      (generation * 0x1111));
}

// Fills rank `me`'s send buffer: every (me, d) row gets its tag, the gaps in
// the strided layout get a poison value that must never be copied.
std::vector<uint16_t>
makeSend(const ncclRegisteredAllToAllLayout& l, int me, int generation) {
  std::vector<uint16_t> host(sendBytes(l) / kElement, 0xdead);
  for (int d = 0; d < kActive; ++d) {
    for (size_t t = 0; t < l.rows; ++t) {
      const size_t base =
          (d * l.sendPeerStride + t * l.sendRowStride) / kElement;
      for (size_t e = 0; e < l.rowBytes / kElement; ++e) {
        host[base + e] = tag(me, d, t, e, generation);
      }
    }
  }
  return host;
}

// Returns the index of the first wrong element, or -1.
long firstMismatch(
    const ncclRegisteredAllToAllLayout& l,
    const std::vector<uint16_t>& recv,
    int me,
    int generation) {
  for (int s = 0; s < kActive; ++s) {
    for (size_t t = 0; t < l.rows; ++t) {
      const size_t base =
          (s * l.recvPeerStride + t * l.recvRowStride) / kElement;
      for (size_t e = 0; e < l.rowBytes / kElement; ++e) {
        if (recv[base + e] != tag(s, me, t, e, generation)) {
          return static_cast<long>(base + e);
        }
      }
    }
  }
  return -1;
}

// Every layout the suite registers; the shared buffers are sized for the
// largest of them.
constexpr Shape kSuiteShapes[] = {
    {Kind::Scatter, 256},
    {Kind::Gather, 256},
    {Kind::Gather, 250},
    {Kind::Gather, 64},
    {Kind::Scatter, 2048},
    {Kind::Gather, 2048}};

} // namespace

class RegisteredAllToAllTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    int localSize = 0;
    std::tie(localRank, globalRank, numRanks, localSize) =
        getTcpStoreOrMpiInfo();
    const bool isServer = globalRank == 0;
    if (checkTcpStoreEnv()) {
      server = createTcpStore(isServer);
    } else if (isServer) {
      server = createTcpStore(true);
    }
    world = createNcclComm(
        globalRank, numRanks, localRank, false, nullptr, server.get());
    // Groups of four; with eight ranks the two groups run concurrently and
    // each relays through the other group's GPUs.
    if (numRanks % kActive == 0) {
      NCCLCHECK_TEST(ncclCommSplit(
          world, globalRank / kActive, globalRank, &comm, nullptr));
    }
    HIPCHECK_TEST(hipGetDeviceCount(&deviceCount));
    // Allocated once and registered by every case, as a serving process
    // registers its fixed buffers once: freeing registered buffers and
    // registering new ones in the same process is not a supported pattern
    // (see verifyPeerMappings in registered_alltoall.cc).
    {
      size_t sendMax = 0;
      size_t recvMax = 0;
      for (const Shape& shape : kSuiteShapes) {
        const auto l = layoutFor(shape);
        sendMax = std::max(sendMax, sendBytes(l));
        recvMax = std::max(recvMax, recvBytes(l));
      }
      HIPCHECK_TEST(hipMalloc(&sendPool, sendMax));
      HIPCHECK_TEST(hipMalloc(&recvPool, recvMax));
      HIPCHECK_TEST(hipMalloc(&sendPool2, sendMax));
      HIPCHECK_TEST(hipMalloc(&recvPool2, recvMax));
    }
  }

  static void TearDownTestSuite() {
    if (server && checkTcpStoreEnv()) {
      finalizeNcclComm(globalRank, server.get());
    }
    for (void** pool : {&sendPool, &recvPool, &sendPool2, &recvPool2}) {
      if (*pool != nullptr) {
        (void)hipFree(*pool);
        *pool = nullptr;
      }
    }
    for (ncclComm_t* c : {&comm, &world}) {
      if (*c != nullptr) {
        ncclCommDestroy(*c);
        *c = nullptr;
      }
    }
    server.reset();
  }

  void SetUp() override {
    ASSERT_NE(comm, nullptr);
    HIPCHECK_TEST(hipStreamCreate(&stream));
  }

  void TearDown() override {
    if (request != nullptr) {
      EXPECT_EQ(ncclRegisteredAllToAllFinalize(request, stream), ncclSuccess);
      request = nullptr;
    }
    send = recv = nullptr;
    HIPEXPECT_TEST(hipStreamDestroy(stream));
  }

  bool supported() const {
    return comm != nullptr && comm->nRanks == kActive && comm->nNodes == 1 &&
        comm->archName != nullptr &&
        std::strncmp(comm->archName, "gfx950", 6) == 0 && comm->isAllDirectP2p;
  }

  // Relay needs GPUs outside the group that every rank can see.
  bool haveHelperGpus() const {
    return deviceCount > kActive;
  }

  static bool isActive() {
    return true;
  }

  bool allVote(bool ok) {
    std::vector<uint8_t> votes(kActive, 0);
    votes[comm->rank] = ok ? 1 : 0;
    if (bootstrapAllGather(comm->bootstrap, votes.data(), 1) != ncclSuccess) {
      return false;
    }
    return std::all_of(
        votes.begin(), votes.end(), [](uint8_t v) { return v != 0; });
  }

  void syncStream(const char* what) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(kSyncTimeoutSec);
    for (;;) {
      const hipError_t status = hipStreamQuery(stream);
      if (status == hipSuccess) {
        return;
      }
      if (status != hipErrorNotReady) {
        FAIL() << "R" << globalRank << " " << what << ": "
               << hipGetErrorString(status);
      }
      if (std::chrono::steady_clock::now() > deadline) {
        std::cerr << "R" << globalRank << " " << what
                  << ": stream did not drain\n";
        std::_Exit(EXIT_FAILURE);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  // Points send/recv at the suite buffers (active ranks only).
  void usePools(const Shape& shape) {
    layout = layoutFor(shape);
    if (isActive()) {
      send = sendPool;
      recv = recvPool;
    }
  }

  void init(const Shape& shape, float relayFraction) {
    usePools(shape);
    ncclRegisteredAllToAllConfig config{};
    config.relayFraction = relayFraction;
    ASSERT_EQ(
        ncclRegisteredAllToAllInit(
            send, recv, &layout, &config, comm, &request),
        ncclSuccess);
  }

  void fillSend(int generation) {
    if (!isActive()) {
      return;
    }
    const std::vector<uint16_t> host = makeSend(layout, comm->rank, generation);
    HIPCHECK_TEST(hipMemcpyAsync(
        send,
        host.data(),
        host.size() * kElement,
        hipMemcpyHostToDevice,
        stream));
    HIPCHECK_TEST(hipMemsetAsync(recv, 0, recvBytes(layout), stream));
  }

  void exec() {
    ASSERT_EQ(
        ncclRegisteredAllToAllExec(send, recv, stream, request), ncclSuccess);
  }

  bool recvMatches(int generation, const char* what) {
    if (!isActive()) {
      return true;
    }
    std::vector<uint16_t> host(recvBytes(layout) / kElement);
    if (hipMemcpy(
            host.data(), recv, host.size() * kElement, hipMemcpyDeviceToHost) !=
        hipSuccess) {
      return false;
    }
    const long bad = firstMismatch(layout, host, comm->rank, generation);
    if (bad >= 0) {
      ADD_FAILURE() << "R" << globalRank << " " << what
                    << ": first wrong element " << bad;
    }
    return bad < 0;
  }

  void runExchanges(const Shape& shape, float relayFraction, int calls) {
    init(shape, relayFraction);
    for (int generation = 1; generation <= calls; ++generation) {
      fillSend(generation);
      exec();
      syncStream("exchange");
      EXPECT_TRUE(allVote(recvMatches(generation, "exchange")));
    }
  }

  static inline ncclComm_t world{nullptr};
  static inline ncclComm_t comm{nullptr};
  static inline int deviceCount{0};
  static inline int localRank{0};
  static inline int globalRank{0};
  static inline int numRanks{0};
  static inline std::unique_ptr<c10d::TCPStore> server{nullptr};
  static inline void* sendPool{nullptr};
  static inline void* recvPool{nullptr};
  static inline void* sendPool2{nullptr};
  static inline void* recvPool2{nullptr};

  hipStream_t stream{nullptr};
  void* send{nullptr};
  void* recv{nullptr};
  void* request{nullptr};
  ncclRegisteredAllToAllLayout layout{};
};

TEST_F(RegisteredAllToAllTest, AbiVersion) {
  EXPECT_EQ(
      ncclRegisteredAllToAllAbiVersion(),
      NCCL_REGISTERED_ALL_TO_ALL_ABI_VERSION);
}

TEST_F(RegisteredAllToAllTest, ScatterDirectOnly) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  runExchanges({Kind::Scatter, 256}, 0.0f, 2);
}

TEST_F(RegisteredAllToAllTest, GatherDirectOnly) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  runExchanges({Kind::Gather, 256}, 0.0f, 2);
}

TEST_F(RegisteredAllToAllTest, ScatterRelayed) {
  if (!supported() || !haveHelperGpus()) {
    GTEST_SKIP() << "requires GPUs outside the group";
  }
  runExchanges({Kind::Scatter, 256}, 0.35f, 3);
}

TEST_F(RegisteredAllToAllTest, GatherRelayedWithRowRemainder) {
  if (!supported() || !haveHelperGpus()) {
    GTEST_SKIP() << "requires GPUs outside the group";
  }
  // 250 rows at 0.5 over the helpers leaves a remainder that stays direct and
  // a last relay chunk shorter than the chunk size.
  runExchanges({Kind::Gather, 250}, 0.5f, 3);
}

TEST_F(RegisteredAllToAllTest, BackToBackWithoutHostSync) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  const float relay = haveHelperGpus() ? 0.35f : 0.0f;
  init({Kind::Gather, 256}, relay);
  for (int generation = 1; generation <= 20; ++generation) {
    fillSend(generation);
    exec();
  }
  syncStream("back-to-back");
  EXPECT_TRUE(allVote(recvMatches(20, "back-to-back")));
}

TEST_F(RegisteredAllToAllTest, GraphReplayWithChangedInput) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  const float relay = haveHelperGpus() ? 0.35f : 0.0f;
  init({Kind::Scatter, 256}, relay);
  fillSend(1);
  syncStream("pre-capture");
  hipGraph_t graph = nullptr;
  hipGraphExec_t graphExec = nullptr;
  HIPCHECK_TEST(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
  exec();
  HIPCHECK_TEST(hipStreamEndCapture(stream, &graph));
  HIPCHECK_TEST(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  for (int generation = 2; generation <= 6; ++generation) {
    fillSend(generation);
    HIPCHECK_TEST(hipGraphLaunch(graphExec, stream));
    syncStream("graph replay");
    EXPECT_TRUE(allVote(recvMatches(generation, "graph replay")));
  }
  HIPEXPECT_TEST(hipGraphExecDestroy(graphExec));
  HIPEXPECT_TEST(hipGraphDestroy(graph));
}

TEST_F(RegisteredAllToAllTest, MismatchedConfigRejectedOnEveryRank) {
  if (!supported() || !haveHelperGpus()) {
    GTEST_SKIP() << "requires GPUs outside the group";
  }
  usePools({Kind::Gather, 64});
  ncclRegisteredAllToAllConfig config{};
  config.relayFraction = comm->rank == 1 ? 0.25f : 0.35f;
  EXPECT_EQ(
      ncclRegisteredAllToAllInit(send, recv, &layout, &config, comm, &request),
      ncclInvalidArgument);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(rcclx::relay::registeredAllToAllLivePeerMappingsForTest(), 0u);
}

TEST_F(RegisteredAllToAllTest, InvalidLayoutRejected) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  usePools({Kind::Gather, 64});
  auto misaligned = layout;
  misaligned.rowBytes += 2;
  EXPECT_EQ(
      ncclRegisteredAllToAllInit(
          send, recv, &misaligned, nullptr, comm, &request),
      ncclInvalidArgument);
  // Larger than the shared suite buffers, whatever layouts size them.
  auto tooLarge = layout;
  tooLarge.sendPeerStride = size_t{1} << 30;
  EXPECT_EQ(
      ncclRegisteredAllToAllInit(
          send, recv, &tooLarge, nullptr, comm, &request),
      ncclInvalidArgument);
  ncclRegisteredAllToAllConfig config{};
  config.relayFraction = 1.0f;
  EXPECT_EQ(
      ncclRegisteredAllToAllInit(send, recv, &layout, &config, comm, &request),
      ncclInvalidArgument);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(rcclx::relay::registeredAllToAllLivePeerMappingsForTest(), 0u);
}

TEST_F(RegisteredAllToAllTest, ExecRejectsOtherBuffers) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  init({Kind::Gather, 64}, 0.0f);
  EXPECT_EQ(
      ncclRegisteredAllToAllExec(recv, send, stream, request),
      ncclInvalidArgument);
}

// Eager exchanges and replays of a captured one share the device-resident
// call counter, so they can interleave freely.
TEST_F(RegisteredAllToAllTest, EagerAndGraphReplaysInterleave) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  init({Kind::Gather, 256}, haveHelperGpus() ? 0.35f : 0.0f);
  fillSend(1);
  syncStream("pre-capture");
  hipGraph_t graph = nullptr;
  hipGraphExec_t graphExec = nullptr;
  HIPCHECK_TEST(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
  exec();
  HIPCHECK_TEST(hipStreamEndCapture(stream, &graph));
  HIPCHECK_TEST(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  for (int generation = 1; generation <= 8; ++generation) {
    fillSend(generation);
    if (generation % 2 == 0) {
      HIPCHECK_TEST(hipGraphLaunch(graphExec, stream));
    } else {
      exec();
    }
    syncStream("interleaved");
    EXPECT_TRUE(allVote(recvMatches(generation, "interleaved")));
  }
  HIPEXPECT_TEST(hipGraphExecDestroy(graphExec));
  HIPEXPECT_TEST(hipGraphDestroy(graph));
}

// Registration probes every page of each send buffer through the peers'
// mappings; the owner's bytes must be exactly what they were before Init.
TEST_F(RegisteredAllToAllTest, RegistrationPreservesSendBufferContents) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  usePools({Kind::Scatter, 256});
  std::vector<uint16_t> before;
  if (isActive()) {
    before = makeSend(layout, comm->rank, 7);
    HIPCHECK_TEST(hipMemcpy(
        send, before.data(), before.size() * kElement, hipMemcpyHostToDevice));
  }
  ncclRegisteredAllToAllConfig config{};
  config.relayFraction = haveHelperGpus() ? 0.35f : 0.0f;
  ASSERT_EQ(
      ncclRegisteredAllToAllInit(send, recv, &layout, &config, comm, &request),
      ncclSuccess);
  if (isActive()) {
    std::vector<uint16_t> after(before.size());
    HIPCHECK_TEST(hipMemcpy(
        after.data(), send, after.size() * kElement, hipMemcpyDeviceToHost));
    EXPECT_TRUE(before == after) << "registration changed the send buffer";
  }
}

// The stale-mapping check, driven deterministically: one rank's check sees
// stale data in a peer's send buffer from byte 2 MiB on (the symptom observed
// with re-registered buffers). Registration must fail with ncclSystemError on
// every rank, return no request, leave no peer mappings, and a normal
// registration on the same buffers afterwards must work.
TEST_F(RegisteredAllToAllTest, StaleMappingFailsRegistrationOnEveryRank) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  usePools({Kind::Scatter, 256});
  ASSERT_GT(sendBytes(layout), size_t{2} << 20);
  const size_t mappingsBefore =
      rcclx::relay::registeredAllToAllLivePeerMappingsForTest();
  rcclx::relay::registeredAllToAllSetStaleMappingForTest(2, 0);
  ncclRegisteredAllToAllConfig config{};
  config.relayFraction = haveHelperGpus() ? 0.35f : 0.0f;
  void* rejected = nullptr;
  const ncclResult_t result =
      ncclRegisteredAllToAllInit(send, recv, &layout, &config, comm, &rejected);
  rcclx::relay::registeredAllToAllSetStaleMappingForTest(-1, -1);
  EXPECT_EQ(result, ncclSystemError);
  EXPECT_EQ(rejected, nullptr);
  EXPECT_EQ(
      rcclx::relay::registeredAllToAllLivePeerMappingsForTest(),
      mappingsBefore);
  EXPECT_TRUE(allVote(result == ncclSystemError));

  init({Kind::Scatter, 256}, config.relayFraction);
  fillSend(1);
  exec();
  syncStream("after rejected registration");
  EXPECT_TRUE(allVote(recvMatches(1, "after rejected registration")));
}

// The pattern that exposed stale mappings: freshly allocated buffers of
// changing sizes registered, used, finalized and freed over and over. Each
// registration must either exchange byte-exact data or be rejected on every
// rank; wrong data is never acceptable.
TEST_F(RegisteredAllToAllTest, ReRegistrationNeverExchangesStaleData) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  const Shape shapes[] = {
      {Kind::Scatter, 256},
      {Kind::Gather, 64},
      {Kind::Gather, 256},
      {Kind::Scatter, 200},
      {Kind::Gather, 250}};
  const float relay = haveHelperGpus() ? 0.35f : 0.0f;
  const size_t pooledAtStart =
      rcclx::relay::registeredAllToAllPooledMappingsForTest();
  int accepted = 0;
  int rejected = 0;
  for (int iteration = 0; iteration < 15; ++iteration) {
    layout = layoutFor(shapes[iteration % 5]);
    void* freshSend = nullptr;
    void* freshRecv = nullptr;
    HIPCHECK_TEST(hipMalloc(&freshSend, sendBytes(layout)));
    HIPCHECK_TEST(hipMalloc(&freshRecv, recvBytes(layout)));
    send = freshSend;
    recv = freshRecv;
    ncclRegisteredAllToAllConfig config{};
    config.relayFraction = relay;
    const ncclResult_t result = ncclRegisteredAllToAllInit(
        send, recv, &layout, &config, comm, &request);
    ASSERT_TRUE(result == ncclSuccess || result == ncclSystemError)
        << "iteration " << iteration << ": " << result;
    ASSERT_TRUE(
        allVote(result == ncclSuccess) || allVote(result != ncclSuccess))
        << "ranks disagreed on registration " << iteration;
    if (result == ncclSuccess) {
      ++accepted;
      for (int generation = 1; generation <= 2; ++generation) {
        fillSend(100 * iteration + generation);
        exec();
        syncStream("re-registration");
        EXPECT_TRUE(allVote(
            recvMatches(100 * iteration + generation, "re-registration")))
            << "iteration " << iteration;
      }
      EXPECT_EQ(ncclRegisteredAllToAllFinalize(request, stream), ncclSuccess);
    } else {
      ++rejected;
      EXPECT_EQ(request, nullptr);
    }
    request = nullptr;
    HIPCHECK_TEST(hipFree(freshSend));
    HIPCHECK_TEST(hipFree(freshRecv));
    send = recv = nullptr;
  }
  std::cout << "[re-registration] rank " << globalRank << ": " << accepted
            << " accepted, " << rejected << " rejected as stale\n";
  EXPECT_EQ(rcclx::relay::registeredAllToAllLivePeerMappingsForTest(), 0u);
  // Relay staging grows at most a few times over these shapes; flags never.
  EXPECT_LE(
      rcclx::relay::registeredAllToAllPooledMappingsForTest(),
      pooledAtStart + 3 * 4 * 3)
      << "pooled imports should not scale with registrations";
}

// Relay shares from none to most of every pair, with row counts that leave a
// direct remainder and short last chunks, bitwise on both layouts.
TEST_F(RegisteredAllToAllTest, RelayFractionSweep) {
  if (!supported() || !haveHelperGpus()) {
    GTEST_SKIP() << "requires GPUs outside the group";
  }
  int generation = 0;
  for (const float fraction : {0.1f, 0.25f, 0.5f, 0.75f, 0.95f}) {
    for (const Shape shape :
         {Shape{Kind::Gather, 1},
          Shape{Kind::Gather, 17},
          Shape{Kind::Scatter, 250},
          Shape{Kind::Gather, 2048}}) {
      init(shape, fraction);
      for (int k = 0; k < 2; ++k) {
        fillSend(++generation);
        exec();
        syncStream("relay sweep");
        EXPECT_TRUE(allVote(recvMatches(generation, "relay sweep")))
            << "fraction " << fraction << " rows " << shape.rows;
      }
      EXPECT_EQ(ncclRegisteredAllToAllFinalize(request, stream), ncclSuccess);
      request = nullptr;
    }
  }
}

// Relay GPU selection: capped at one helper, restricted to a device list that
// matches nothing (relay requested, exchange falls back to direct), and the
// default. Every configuration must be bitwise correct.
TEST_F(RegisteredAllToAllTest, HelperSelectionStaysCorrect) {
  if (!supported() || !haveHelperGpus()) {
    GTEST_SKIP() << "requires GPUs outside the group";
  }
  struct Case {
    const char* name;
    const char* value;
  };
  const Case cases[] = {
      {"NCCL_RELAY_MAX_HELPERS", "1"},
      {"NCCL_RELAY_HELPER_DEVICES", "0000:ff:1f.7"},
      {"NCCL_RELAY_MAX_HELPERS", "0"}};
  int generation = 0;
  for (const Case& c : cases) {
    setenv(c.name, c.value, 1);
    init({Kind::Scatter, 256}, 0.5f);
    unsetenv(c.name);
    for (int k = 0; k < 2; ++k) {
      fillSend(++generation);
      exec();
      syncStream("helper selection");
      EXPECT_TRUE(allVote(recvMatches(generation, "helper selection")))
          << c.name << "=" << c.value;
    }
    EXPECT_EQ(ncclRegisteredAllToAllFinalize(request, stream), ncclSuccess);
    request = nullptr;
  }
}

// Relay staging and flags belong to the communicator: the first registration
// allocates them, later registrations of the same shape reuse them (no helper
// HBM growth, no new imports), Finalize keeps them, and ncclCommDestroy frees
// them. Uses a private communicator so its destroy can be observed.
TEST_F(RegisteredAllToAllTest, RelayMemoryPooledPerCommunicator) {
  if (!supported() || !haveHelperGpus() || numRanks != kActive) {
    GTEST_SKIP() << "needs one 4-rank group and idle GPUs outside it";
  }
  const int helperDevice = kActive;
  auto freeOn = [&](size_t* free) {
    size_t total = 0;
    HIPCHECK_TEST(hipSetDevice(helperDevice));
    HIPCHECK_TEST(hipDeviceSynchronize());
    HIPCHECK_TEST(hipMemGetInfo(free, &total));
    HIPCHECK_TEST(hipSetDevice(localRank));
  };
  ncclComm_t own = nullptr;
  NCCLCHECK_TEST(ncclCommSplit(world, 0, globalRank, &own, nullptr));
  usePools({Kind::Gather, 2048});
  ncclRegisteredAllToAllConfig config{};
  config.relayFraction = 0.35f;
  const size_t pooledBefore =
      rcclx::relay::registeredAllToAllPooledMappingsForTest();
  allVote(true);
  size_t before = 0, first = 0, again = 0, finalized = 0, destroyed = 0;
  freeOn(&before);

  void* req = nullptr;
  ASSERT_EQ(
      ncclRegisteredAllToAllInit(send, recv, &layout, &config, own, &req),
      ncclSuccess);
  const size_t pooledFirst =
      rcclx::relay::registeredAllToAllPooledMappingsForTest();
  EXPECT_EQ(ncclRegisteredAllToAllFinalize(req, stream), ncclSuccess);
  allVote(true);
  freeOn(&first);
  for (int k = 0; k < 3; ++k) {
    ASSERT_EQ(
        ncclRegisteredAllToAllInit(send, recv, &layout, &config, own, &req),
        ncclSuccess);
    EXPECT_EQ(ncclRegisteredAllToAllFinalize(req, stream), ncclSuccess);
  }
  allVote(true);
  freeOn(&again);
  EXPECT_EQ(
      rcclx::relay::registeredAllToAllPooledMappingsForTest(), pooledFirst)
      << "re-registration imported RCCLX memory again";
  finalized = again;
  NCCLCHECK_TEST(ncclCommDestroy(own));
  allVote(true);
  freeOn(&destroyed);

  EXPECT_GT(pooledFirst, pooledBefore);
  EXPECT_LE(before - first, size_t{64} << 20)
      << "helper HBM grew by " << ((before - first) >> 20) << " MiB";
  EXPECT_GE(first + (size_t{2} << 20), again)
      << "re-registration grew helper HBM again";
  EXPECT_LE(finalized, first + (size_t{2} << 20))
      << "staging must stay pooled after Finalize";
  EXPECT_GE(destroyed + (size_t{2} << 20), before)
      << "ncclCommDestroy did not free the relay staging";
  EXPECT_EQ(
      rcclx::relay::registeredAllToAllPooledMappingsForTest(), pooledBefore);
}

// Each live request owns a flags slot; a communicator holds 16. The 17th
// registration fails on every rank, and finalizing one frees its slot.
TEST_F(RegisteredAllToAllTest, SlotsBoundLiveRequests) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  usePools({Kind::Gather, 64});
  ncclRegisteredAllToAllConfig config{};
  std::vector<void*> live;
  for (int k = 0; k < 16; ++k) {
    void* req = nullptr;
    ASSERT_EQ(
        ncclRegisteredAllToAllInit(send, recv, &layout, &config, comm, &req),
        ncclSuccess)
        << "request " << k;
    live.push_back(req);
  }
  void* extra = nullptr;
  EXPECT_EQ(
      ncclRegisteredAllToAllInit(send, recv, &layout, &config, comm, &extra),
      ncclInvalidUsage);
  EXPECT_EQ(extra, nullptr);
  EXPECT_EQ(ncclRegisteredAllToAllFinalize(live.back(), stream), ncclSuccess);
  live.pop_back();
  ASSERT_EQ(
      ncclRegisteredAllToAllInit(send, recv, &layout, &config, comm, &extra),
      ncclSuccess);
  live.push_back(extra);
  request = live.front();
  fillSend(1);
  exec();
  syncStream("slot reuse");
  EXPECT_TRUE(allVote(recvMatches(1, "slot reuse")));
  request = nullptr;
  for (void* req : live) {
    EXPECT_EQ(ncclRegisteredAllToAllFinalize(req, stream), ncclSuccess);
  }
}

// Two requests live at once (the Ulysses scatter and gather pair), each on its
// own buffers, alternating exchanges on one stream.
TEST_F(RegisteredAllToAllTest, TwoLiveRequestsAlternate) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  const float relay = haveHelperGpus() ? 0.35f : 0.0f;
  const auto scatterLayout = layoutFor({Kind::Scatter, 256});
  const auto gatherLayout = layoutFor({Kind::Gather, 256});
  ncclRegisteredAllToAllConfig config{};
  config.relayFraction = relay;
  void* scatter = nullptr;
  void* gather = nullptr;
  ASSERT_EQ(
      ncclRegisteredAllToAllInit(
          sendPool, recvPool, &scatterLayout, &config, comm, &scatter),
      ncclSuccess);
  ASSERT_EQ(
      ncclRegisteredAllToAllInit(
          sendPool2, recvPool2, &gatherLayout, &config, comm, &gather),
      ncclSuccess);
  for (int generation = 1; generation <= 6; ++generation) {
    const bool isScatter = generation % 2 == 1;
    layout = isScatter ? scatterLayout : gatherLayout;
    send = isScatter ? sendPool : sendPool2;
    recv = isScatter ? recvPool : recvPool2;
    request = isScatter ? scatter : gather;
    fillSend(generation);
    exec();
    syncStream("two live requests");
    EXPECT_TRUE(allVote(recvMatches(generation, "two live requests")))
        << (isScatter ? "scatter" : "gather") << " " << generation;
  }
  request = nullptr;
  send = recv = nullptr;
  EXPECT_EQ(ncclRegisteredAllToAllFinalize(scatter, stream), ncclSuccess);
  EXPECT_EQ(ncclRegisteredAllToAllFinalize(gather, stream), ncclSuccess);
}

// A captured exchange replayed many times with the input changing between
// replays, spot-checked throughout.
TEST_F(RegisteredAllToAllTest, LongGraphReplay) {
  if (!supported()) {
    GTEST_SKIP() << "requires a 4-rank gfx950 group with direct P2P";
  }
  init({Kind::Gather, 256}, haveHelperGpus() ? 0.35f : 0.0f);
  fillSend(1);
  syncStream("pre-capture");
  hipGraph_t graph = nullptr;
  hipGraphExec_t graphExec = nullptr;
  HIPCHECK_TEST(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
  exec();
  HIPCHECK_TEST(hipStreamEndCapture(stream, &graph));
  HIPCHECK_TEST(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  for (int replay = 1; replay <= 1000; ++replay) {
    if (replay % 100 == 0) {
      fillSend(replay);
    }
    HIPCHECK_TEST(hipGraphLaunch(graphExec, stream));
    if (replay % 100 == 0) {
      syncStream("long replay");
      EXPECT_TRUE(allVote(recvMatches(replay, "long replay")))
          << "replay " << replay;
    }
  }
  HIPEXPECT_TEST(hipGraphExecDestroy(graphExec));
  HIPEXPECT_TEST(hipGraphDestroy(graph));
}

// Timing at the exact [redacted] prefill shapes (8192-token chunk over 4 ranks:
// 2048 rows per pair). Prints per-exchange microseconds; set
// REGISTERED_A2A_PERF=1 to run. RELAY_FRACTION overrides the relay share.
TEST_F(RegisteredAllToAllTest, Z_PerfRedactedShapes) {
  const char* on = getenv("REGISTERED_A2A_PERF");
  if (!supported() || on == nullptr || on[0] != '1') {
    GTEST_SKIP() << "set REGISTERED_A2A_PERF=1";
  }
  const char* fractionEnv = getenv("RELAY_FRACTION");
  const float fraction =
      fractionEnv != nullptr ? std::strtof(fractionEnv, nullptr) : 0.35f;
  for (Kind kind : {Kind::Scatter, Kind::Gather}) {
    for (float relay : {0.0f, fraction}) {
      if (relay > 0.0f && !haveHelperGpus()) {
        continue;
      }
      init({kind, 2048}, relay);
      fillSend(1);
      syncStream("perf warmup fill");
      for (int i = 0; i < 5; ++i) {
        exec();
      }
      syncStream("perf warmup");
      hipEvent_t start = nullptr, stop = nullptr;
      HIPCHECK_TEST(hipEventCreate(&start));
      HIPCHECK_TEST(hipEventCreate(&stop));
      constexpr int kIters = 30;
      std::vector<float> samples;
      for (int i = 0; i < kIters; ++i) {
        allVote(true); // align every rank before each exchange
        HIPCHECK_TEST(hipEventRecord(start, stream));
        exec();
        HIPCHECK_TEST(hipEventRecord(stop, stream));
        syncStream("perf");
        float ms = 0;
        HIPCHECK_TEST(hipEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
      }
      std::sort(samples.begin(), samples.end());
      float ms = samples[samples.size() / 2];
      if (getenv("A2A_PERF_B2B") != nullptr) {
        allVote(true);
        HIPCHECK_TEST(hipEventRecord(start, stream));
        for (int i = 0; i < kIters; ++i) {
          exec();
        }
        HIPCHECK_TEST(hipEventRecord(stop, stream));
        syncStream("perf b2b");
        HIPCHECK_TEST(hipEventElapsedTime(&ms, start, stop));
        ms /= kIters;
      }
      std::vector<float> all(kActive, 0.0f);
      all[comm->rank] = ms * 1000.0f;
      ASSERT_EQ(
          bootstrapAllGather(comm->bootstrap, all.data(), sizeof(float)),
          ncclSuccess);
      if (comm->rank == 0) {
        std::cout << "[a2a-perf] group " << globalRank / kActive << " "
                  << (kind == Kind::Scatter ? "scatter" : "gather")
                  << " relay=" << relay << " world=" << numRanks << ": "
                  << *std::max_element(all.begin(), all.end()) << " (min "
                  << *std::min_element(all.begin(), all.end()) << ")"
                  << " us per exchange (median of 30, max over active ranks)\n";
      }
      HIPEXPECT_TEST(hipEventDestroy(start));
      HIPEXPECT_TEST(hipEventDestroy(stop));
      EXPECT_EQ(ncclRegisteredAllToAllFinalize(request, stream), ncclSuccess);
      request = nullptr;
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
