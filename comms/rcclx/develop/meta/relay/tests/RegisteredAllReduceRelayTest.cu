// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Relay route of the two-rank registered all-reduce. Runs at 2 ranks (one
// pair relaying through the other six GPUs) and at 8 ranks, where the world
// comm is split into four pairs that run concurrently, each relaying through
// the other six GPUs.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "comm.h"
#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"
#include "meta/relay/registered_allreduce.h"
#include "nccl.h"

#define HIPCHECK_TEST(cmd)                                          \
  do {                                                              \
    hipError_t error = cmd;                                         \
    if (error != hipSuccess) {                                      \
      FAIL() << "HIP error: " << hipGetErrorString(error) << " at " \
             << __FILE__ << ":" << __LINE__;                        \
    }                                                               \
  } while (0)

namespace {

constexpr size_t kMiB = 1024 * 1024;
constexpr size_t kCapacity = 144 * kMiB;
constexpr size_t kDefaultRelayMinBytes = 2 * kMiB;

uint16_t toBf16(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    return 0x7fc0u;
  }
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

float fromBf16(uint16_t v) {
  const uint32_t u = static_cast<uint32_t>(v) << 16;
  float f = 0;
  std::memcpy(&f, &u, 4);
  return f;
}

float mixed(uint64_t seed, size_t i) {
  uint64_t x = (seed << 32) ^ i;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return static_cast<float>(x % 2000001ull) / 1000000.0f - 1.0f;
}

std::vector<uint16_t> makeInput(size_t count, int rank, uint32_t salt) {
  std::vector<uint16_t> out(count);
  const uint64_t seed = 1000ull * salt + rank + 1;
  for (size_t i = 0; i < count; ++i) {
    out[i] = toBf16(mixed(seed, i) * (1.0f + 3.0f * rank));
  }
  return out;
}

// Rank-order float sum rounded to bf16.
std::vector<uint16_t> expected(size_t count, uint32_t salt) {
  const auto r0 = makeInput(count, 0, salt);
  const auto r1 = makeInput(count, 1, salt);
  std::vector<uint16_t> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = toBf16(fromBf16(r0[i]) + fromBf16(r1[i]));
  }
  return out;
}

size_t firstMismatch(
    const std::vector<uint16_t>& got,
    const std::vector<uint16_t>& want) {
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      return i;
    }
  }
  return got.size();
}

} // namespace

class RegisteredAllReduceRelayTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    int localSize = 0;
    std::tie(localRank, globalRank, numRanks, localSize) =
        getTcpStoreOrMpiInfo();
    HIPCHECK_TEST(hipSetDevice(localRank));
    const bool isServer = globalRank == 0;
    if (checkTcpStoreEnv()) {
      server = createTcpStore(isServer);
    } else if (isServer) {
      server = createTcpStore(true);
    }
    world = createNcclComm(
        globalRank, numRanks, localRank, false, nullptr, server.get());
    pair = split(globalRank / 2);
    pairRank = globalRank % 2;
    HIPCHECK_TEST(hipMalloc(&input, kCapacity));
    HIPCHECK_TEST(hipMalloc(&output, kCapacity));
    NCCLCHECK_TEST(
        ncclRegisteredAllReduceInit(input, output, kCapacity, pair, &request));
  }

  static void TearDownTestSuite() {
    if (request != nullptr) {
      NCCLCHECK_TEST(ncclRegisteredAllReduceFinalize(request, nullptr));
      request = nullptr;
    }
    for (void** p : {&input, &output}) {
      if (*p != nullptr) {
        (void)hipFree(*p);
        *p = nullptr;
      }
    }
    if (server && checkTcpStoreEnv()) {
      finalizeNcclComm(globalRank, server.get());
    }
    for (ncclComm_t* c : {&pair, &world}) {
      if (*c != nullptr) {
        NCCLCHECK_TEST(ncclCommDestroy(*c));
        *c = nullptr;
      }
    }
    server.reset();
  }

  // Collective over the world comm.
  static ncclComm_t split(int color) {
    ncclComm_t out = nullptr;
    NCCLCHECK_TEST(ncclCommSplit(world, color, globalRank, &out, nullptr));
    return out;
  }

  void SetUp() override {
    ASSERT_NE(request, nullptr);
    HIPCHECK_TEST(hipStreamCreate(&stream));
  }

  void TearDown() override {
    HIPCHECK_TEST(hipStreamSynchronize(stream));
    HIPCHECK_TEST(hipStreamDestroy(stream));
  }

  void worldBarrier() {
    float* one = nullptr;
    HIPCHECK_TEST(hipMalloc(&one, sizeof(float)));
    NCCLCHECK_TEST(
        ncclAllReduce(one, one, 1, ncclFloat32, ncclSum, world, stream));
    HIPCHECK_TEST(hipStreamSynchronize(stream));
    HIPCHECK_TEST(hipFree(one));
  }

  // This rank's input for `salt` in a fresh device buffer.
  void* upload(size_t count, uint32_t salt) {
    const auto host = makeInput(count, pairRank, salt);
    void* dev = nullptr;
    EXPECT_EQ(hipMalloc(&dev, count * 2), hipSuccess);
    EXPECT_EQ(
        hipMemcpy(dev, host.data(), count * 2, hipMemcpyHostToDevice),
        hipSuccess);
    return dev;
  }

  void exec(void* req, void* in, void* out, size_t count) {
    NCCLCHECK_TEST(ncclRegisteredAllReduceExec(
        in, out, count, ncclBfloat16, ncclSum, nullptr, stream, req));
  }

  std::vector<uint16_t> download(const void* dev, size_t count) {
    std::vector<uint16_t> host(count);
    EXPECT_EQ(
        hipMemcpy(host.data(), dev, count * 2, hipMemcpyDeviceToHost),
        hipSuccess);
    return host;
  }

  // One execution of `count` elements on the suite request, checked bitwise
  // against the rank-order reference.
  void runAndCheck(size_t count, uint32_t salt) {
    void* src = upload(count, salt);
    HIPCHECK_TEST(
        hipMemcpyAsync(input, src, count * 2, hipMemcpyDeviceToDevice, stream));
    exec(request, input, output, count);
    HIPCHECK_TEST(hipStreamSynchronize(stream));
    EXPECT_EQ(
        firstMismatch(download(output, count), expected(count, salt)), count)
        << "count=" << count << ": first mismatching element";
    HIPCHECK_TEST(hipFree(src));
  }

  static int visibleDevices() {
    int n = 0;
    EXPECT_EQ(hipGetDeviceCount(&n), hipSuccess);
    return n;
  }

  // Bus IDs of visible GPUs outside this rank's pair.
  static std::vector<std::string> otherBusIds() {
    std::vector<std::string> out;
    const int n = visibleDevices();
    const int peerDevice = localRank ^ 1;
    for (int d = 0; d < n; ++d) {
      if (d == localRank || d == peerDevice) {
        continue;
      }
      char id[64] = {};
      EXPECT_EQ(hipDeviceGetPCIBusId(id, sizeof(id), d), hipSuccess);
      out.emplace_back(id);
    }
    return out;
  }

  static inline int localRank{0};
  static inline int globalRank{0};
  static inline int numRanks{0};
  static inline int pairRank{0};
  static inline ncclComm_t world{nullptr};
  static inline ncclComm_t pair{nullptr};
  static inline void* input{nullptr};
  static inline void* output{nullptr};
  static inline void* request{nullptr};
  static inline std::unique_ptr<c10d::TCPStore> server{nullptr};
  hipStream_t stream{nullptr};
};

TEST_F(RegisteredAllReduceRelayTest, RelayUsesEveryOtherGpu) {
  const int want = std::min(visibleDevices() - 2, 7);
  EXPECT_EQ(
      rcclx::relay::registeredAllReduceRelayHelpersForTest(request), want);
}

// Both sides of the cutover, bitwise against the rank-order reference; the
// relay route is taken exactly from the cutover up.
TEST_F(RegisteredAllReduceRelayTest, CorrectAcrossTheCutover) {
  const std::vector<size_t> bytes = {
      16,
      72 * 1024,
      576 * 1024,
      kMiB,
      kDefaultRelayMinBytes - 16,
      kDefaultRelayMinBytes,
      kDefaultRelayMinBytes + 16,
      9 * kMiB,
      72 * kMiB,
      kCapacity};
  uint32_t salt = 10;
  for (size_t b : bytes) {
    const uint64_t before =
        rcclx::relay::registeredAllReduceRelayLaunchesForTest();
    runAndCheck(b / 2, ++salt);
    const uint64_t relayed =
        rcclx::relay::registeredAllReduceRelayLaunchesForTest() - before;
    EXPECT_EQ(relayed, b >= kDefaultRelayMinBytes ? 1u : 0u) << b << " bytes";
  }
}

TEST_F(RegisteredAllReduceRelayTest, MatchesNcclAllReduceBitwise) {
  const size_t count = 72 * kMiB / 2;
  void* src = upload(count, 300);
  void* ref = nullptr;
  HIPCHECK_TEST(hipMalloc(&ref, count * 2));
  HIPCHECK_TEST(
      hipMemcpyAsync(input, src, count * 2, hipMemcpyDeviceToDevice, stream));
  exec(request, input, output, count);
  NCCLCHECK_TEST(
      ncclAllReduce(src, ref, count, ncclBfloat16, ncclSum, pair, stream));
  HIPCHECK_TEST(hipStreamSynchronize(stream));
  EXPECT_EQ(
      firstMismatch(download(output, count), download(ref, count)), count);
  HIPCHECK_TEST(hipFree(src));
  HIPCHECK_TEST(hipFree(ref));
}

// Decode- and prefill-sized executions interleaved on one request with no
// host sync. Each call's producer overwrites the registered input right after
// the previous call, so every call must be done with both inputs on return.
TEST_F(RegisteredAllReduceRelayTest, DecodeAndPrefillSizesBackToBack) {
  const std::vector<size_t> counts = {
      36864, 72 * kMiB / 2, 294912, 9 * kMiB / 2, 8, kCapacity / 2, 147456};
  constexpr int kCalls = 42;
  std::vector<void*> srcs, outs;
  for (int i = 0; i < kCalls; ++i) {
    const size_t count = counts[i % counts.size()];
    srcs.push_back(upload(count, 400 + i));
    void* out = nullptr;
    HIPCHECK_TEST(hipMalloc(&out, count * 2));
    outs.push_back(out);
  }
  for (int i = 0; i < kCalls; ++i) {
    const size_t count = counts[i % counts.size()];
    HIPCHECK_TEST(hipMemcpyAsync(
        input, srcs[i], count * 2, hipMemcpyDeviceToDevice, stream));
    exec(request, input, output, count);
    HIPCHECK_TEST(hipMemcpyAsync(
        outs[i], output, count * 2, hipMemcpyDeviceToDevice, stream));
  }
  HIPCHECK_TEST(hipStreamSynchronize(stream));
  for (int i = 0; i < kCalls; ++i) {
    const size_t count = counts[i % counts.size()];
    EXPECT_EQ(
        firstMismatch(download(outs[i], count), expected(count, 400 + i)),
        count)
        << "call " << i;
    HIPCHECK_TEST(hipFree(srcs[i]));
    HIPCHECK_TEST(hipFree(outs[i]));
  }
}

// A relay-sized execution captured with its producer copy, replayed with
// changed inputs and interleaved with eager decode- and prefill-sized calls.
TEST_F(RegisteredAllReduceRelayTest, GraphReplayInterleavedWithEagerCalls) {
  const size_t count = 72 * kMiB / 2;
  void* staged = nullptr;
  HIPCHECK_TEST(hipMalloc(&staged, count * 2));
  hipGraph_t graph = nullptr;
  hipGraphExec_t graphExec = nullptr;
  HIPCHECK_TEST(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
  HIPCHECK_TEST(hipMemcpyAsync(
      input, staged, count * 2, hipMemcpyDeviceToDevice, stream));
  exec(request, input, output, count);
  HIPCHECK_TEST(hipStreamEndCapture(stream, &graph));
  HIPCHECK_TEST(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  for (uint32_t round = 0; round < 6; ++round) {
    const uint32_t salt = 500 + round;
    void* src = upload(count, salt);
    HIPCHECK_TEST(hipMemcpy(staged, src, count * 2, hipMemcpyDeviceToDevice));
    HIPCHECK_TEST(hipGraphLaunch(graphExec, stream));
    HIPCHECK_TEST(hipStreamSynchronize(stream));
    EXPECT_EQ(
        firstMismatch(download(output, count), expected(count, salt)), count)
        << "replay " << round;
    HIPCHECK_TEST(hipFree(src));
    runAndCheck(round % 2 == 0 ? 294912 : 9 * kMiB / 2, 600 + round);
  }
  HIPCHECK_TEST(hipGraphExecDestroy(graphExec));
  HIPCHECK_TEST(hipGraphDestroy(graph));
  HIPCHECK_TEST(hipFree(staged));
}

// The per-lane slot sequence counters are 32-bit. Starting them just below the
// wrap, relay calls that cross it must still wait for each slot's credit and
// produce exact sums.
TEST_F(RegisteredAllReduceRelayTest, SequenceCountersWrapSafely) {
  if (rcclx::relay::registeredAllReduceRelayHelpersForTest(request) <= 0) {
    GTEST_SKIP() << "no relay route";
  }
  HIPCHECK_TEST(hipStreamSynchronize(stream));
  worldBarrier();
  ASSERT_EQ(
      rcclx::relay::registeredAllReduceSetRelaySequenceForTest(
          request, 0xFFFFFFFFu - 5u),
      ncclSuccess);
  worldBarrier();
  const size_t counts[] = {36 * kMiB / 2, 72 * kMiB / 2, 9 * kMiB / 2};
  for (uint32_t round = 0; round < 9; ++round) {
    runAndCheck(counts[round % 3], 700 + round);
  }
}

// Pair 0 only; the other pairs split off and idle. Requests restricted to two
// helper GPUs, with no reachable helper (one-shot at every size), and with a
// capacity below the cutover. Helper HBM grows by the staging only and is
// returned at Finalize.
TEST_F(RegisteredAllReduceRelayTest, HelperSelectionCapacityAndMemory) {
  const auto others = otherBusIds();
  if (others.size() < 2) {
    GTEST_SKIP() << "needs at least two visible GPUs outside the pair";
  }
  const bool participate = globalRank / 2 == 0;
  ncclComm_t sub = split(participate ? 0 : 1);
  int helperDevice = -1;
  HIPCHECK_TEST(hipDeviceGetByPCIBusId(&helperDevice, others[0].c_str()));
  const size_t capacity = 72 * kMiB;
  void* in = nullptr;
  void* out = nullptr;
  HIPCHECK_TEST(hipMalloc(&in, capacity));
  HIPCHECK_TEST(hipMalloc(&out, capacity));

  auto helperFree = [&](size_t* free) {
    size_t total = 0;
    HIPCHECK_TEST(hipSetDevice(helperDevice));
    HIPCHECK_TEST(hipMemGetInfo(free, &total));
    HIPCHECK_TEST(hipSetDevice(localRank));
  };
  auto registerAndRun = [&](const char* devices,
                            size_t cap,
                            int wantHelpers,
                            uint32_t salt,
                            void** req) {
    setenv("NCCL_RELAY_HELPER_DEVICES", devices, 1);
    NCCLCHECK_TEST(ncclRegisteredAllReduceInit(in, out, cap, sub, req));
    unsetenv("NCCL_RELAY_HELPER_DEVICES");
    EXPECT_EQ(
        rcclx::relay::registeredAllReduceRelayHelpersForTest(*req),
        wantHelpers);
    const size_t n = cap / 2;
    void* src = upload(n, salt);
    HIPCHECK_TEST(hipMemcpy(in, src, cap, hipMemcpyDeviceToDevice));
    exec(*req, in, out, n);
    HIPCHECK_TEST(hipStreamSynchronize(stream));
    EXPECT_EQ(firstMismatch(download(out, n), expected(n, salt)), n);
    HIPCHECK_TEST(hipFree(src));
  };

  worldBarrier();
  size_t before = 0, during = 0, finalized = 0, again = 0;
  if (participate) {
    const std::string two = others[0] + "," + others[1];
    void* req = nullptr;
    helperFree(&before);
    registerAndRun(two.c_str(), capacity, 2, 700, &req);
    helperFree(&during);
    NCCLCHECK_TEST(ncclRegisteredAllReduceFinalize(req, stream));
    helperFree(&finalized);
    // Relay memory belongs to the communicator: re-registering reuses it.
    registerAndRun(two.c_str(), capacity, 2, 703, &req);
    NCCLCHECK_TEST(ncclRegisteredAllReduceFinalize(req, stream));
    helperFree(&again);
    EXPECT_LE(before - during, 64 * kMiB)
        << "helper HBM grew by " << ((before - during) >> 20) << " MiB";
    EXPECT_LE(finalized, during + 2 * kMiB)
        << "relay staging must stay pooled after Finalize";
    EXPECT_GE(again + 2 * kMiB, finalized)
        << "re-registration allocated relay staging again";

    registerAndRun("0000:ff:1f.7", capacity, 0, 701, &req);
    NCCLCHECK_TEST(ncclRegisteredAllReduceFinalize(req, stream));

    registerAndRun(two.c_str(), kMiB, 0, 702, &req);
    NCCLCHECK_TEST(ncclRegisteredAllReduceFinalize(req, stream));
  }
  worldBarrier();
  HIPCHECK_TEST(hipFree(in));
  HIPCHECK_TEST(hipFree(out));
  NCCLCHECK_TEST(ncclCommDestroy(sub));
  worldBarrier();
  if (participate) {
    size_t destroyed = 0;
    helperFree(&destroyed);
    EXPECT_GE(destroyed + 2 * kMiB, before)
        << "ncclCommDestroy did not free the relay staging";
  }
}

// Relay requests registered, used and finalized on freshly allocated buffers
// over and over: every exchange must be exact, and a fresh buffer written
// after each Finalize must read back intact (freed peer-written memory used to
// corrupt a page of later allocations).
TEST_F(RegisteredAllReduceRelayTest, ReRegistrationOnFreshBuffers) {
  const bool participate = globalRank / 2 == 0;
  ncclComm_t sub = split(participate ? 0 : 1);
  worldBarrier();
  if (participate) {
    for (int iteration = 0; iteration < 12; ++iteration) {
      const size_t capacity = (iteration % 2 == 0 ? 72 : 9) * kMiB;
      void* in = nullptr;
      void* out = nullptr;
      HIPCHECK_TEST(hipMalloc(&in, capacity));
      HIPCHECK_TEST(hipMalloc(&out, capacity));
      void* req = nullptr;
      NCCLCHECK_TEST(ncclRegisteredAllReduceInit(in, out, capacity, sub, &req));
      const size_t n = capacity / 2;
      void* src = upload(n, 800 + iteration);
      HIPCHECK_TEST(hipMemcpy(in, src, capacity, hipMemcpyDeviceToDevice));
      exec(req, in, out, n);
      HIPCHECK_TEST(hipStreamSynchronize(stream));
      EXPECT_EQ(
          firstMismatch(download(out, n), expected(n, 800 + iteration)), n)
          << "iteration " << iteration;
      NCCLCHECK_TEST(ncclRegisteredAllReduceFinalize(req, stream));
      HIPCHECK_TEST(hipFree(src));

      constexpr size_t kCanaryBytes = size_t{8} << 20;
      constexpr uint32_t kCanary = 0xa5a5a5a5u;
      void* canary = nullptr;
      HIPCHECK_TEST(hipMalloc(&canary, kCanaryBytes));
      HIPCHECK_TEST(hipMemsetD32(
          static_cast<hipDeviceptr_t>(canary), kCanary, kCanaryBytes / 4));
      HIPCHECK_TEST(hipDeviceSynchronize());
      std::vector<uint32_t> words(kCanaryBytes / 4);
      HIPCHECK_TEST(
          hipMemcpy(words.data(), canary, kCanaryBytes, hipMemcpyDeviceToHost));
      EXPECT_EQ(
          std::count(words.begin(), words.end(), kCanary),
          static_cast<std::ptrdiff_t>(words.size()))
          << "iteration " << iteration << ": fresh buffer lost writes";
      HIPCHECK_TEST(hipFree(canary));
      HIPCHECK_TEST(hipFree(out));
      HIPCHECK_TEST(hipFree(in));
    }
  }
  worldBarrier();
  NCCLCHECK_TEST(ncclCommDestroy(sub));
}

// Crossover sweep, 2 ranks: the relay route against the same request shape
// with no helpers (one-shot at every size) and ncclAllReduce, bf16 SUM,
// back-to-back calls. REGISTERED_AR_RELAY_PERF=1 to run; NCCL_RELAY_MAX_HELPERS
// limits the relay GPUs (2 on [redacted]'s 4-GPU reservation).
TEST_F(RegisteredAllReduceRelayTest, Z_PerfSweep) {
  const char* on = getenv("REGISTERED_AR_RELAY_PERF");
  if (on == nullptr || on[0] != '1' || numRanks != 2) {
    GTEST_SKIP() << "set REGISTERED_AR_RELAY_PERF=1 at 2 ranks";
  }
  void* oneShotIn = nullptr;
  void* oneShotOut = nullptr;
  void* oneShot = nullptr;
  HIPCHECK_TEST(hipMalloc(&oneShotIn, kCapacity));
  HIPCHECK_TEST(hipMalloc(&oneShotOut, kCapacity));
  setenv("NCCL_RELAY_HELPER_DEVICES", "0000:ff:1f.7", 1);
  NCCLCHECK_TEST(ncclRegisteredAllReduceInit(
      oneShotIn, oneShotOut, kCapacity, pair, &oneShot));
  unsetenv("NCCL_RELAY_HELPER_DEVICES");
  HIPCHECK_TEST(hipMemset(input, 0x3c, kCapacity));
  HIPCHECK_TEST(hipMemset(oneShotIn, 0x3c, kCapacity));

  auto timeUs = [&](const auto& call, int iters, float* result) {
    for (int i = 0; i < 5; ++i) {
      call();
    }
    HIPCHECK_TEST(hipStreamSynchronize(stream));
    float best = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
      worldBarrier();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        call();
      }
      HIPCHECK_TEST(hipStreamSynchronize(stream));
      const float perCall = std::chrono::duration<float, std::micro>(
                                std::chrono::steady_clock::now() - t0)
                                .count() /
          iters;
      best = perCall < best ? perCall : best;
    }
    *result = best;
  };
  const std::vector<size_t> sizes = {
      72 * 1024,
      288 * 1024,
      576 * 1024,
      kMiB,
      2 * kMiB,
      4 * kMiB,
      9 * kMiB,
      36 * kMiB,
      72 * kMiB,
      kCapacity};
  for (size_t bytes : sizes) {
    const size_t count = bytes / 2;
    const int iters = bytes <= 4 * kMiB ? 200 : 20;
    float registered = 0, oneShotUs = 0, nccl = 0;
    timeUs([&] { exec(request, input, output, count); }, iters, &registered);
    timeUs(
        [&] { exec(oneShot, oneShotIn, oneShotOut, count); },
        iters,
        &oneShotUs);
    timeUs(
        [&] {
          NCCLCHECK_TEST(ncclAllReduce(
              oneShotIn,
              oneShotOut,
              count,
              ncclBfloat16,
              ncclSum,
              pair,
              stream));
        },
        iters,
        &nccl);
    if (pairRank == 0) {
      printf(
          "[relay-perf] %zu KiB: registered %.2f us, one-shot %.2f us, nccl %.2f us\n",
          bytes / 1024,
          registered,
          oneShotUs,
          nccl);
    }
  }
  NCCLCHECK_TEST(ncclRegisteredAllReduceFinalize(oneShot, stream));
  HIPCHECK_TEST(hipFree(oneShotIn));
  HIPCHECK_TEST(hipFree(oneShotOut));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
