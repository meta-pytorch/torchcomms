// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Exercises GpuMemHandler::exchangeMulticast() boundary semantics over a real
// NVL-domain bootstrap on a GPU host:
//
//   - Happy path on the auto-selected VMM mode (fabric on GB200-class hosts,
//     POSIX FD on H100) and on the explicit kPosixFd mode, confirming that the
//     pointer is non-null + stable across a matching re-entry.
//   - Failed-exchange retry must not become a silent success: if the first
//     exchangeMulticast throws, multimem_ stays null and a follow-up call must
//     actually re-attempt (not hit the if-multimem_-return early-exit).
//   - Re-entry with mismatched (commRank, nvlRankToCommRank, cudaDevice) must
//     throw deterministically with a "re-entered with different" message rather
//     than silently rebinding the existing overlay to a new topology.
//
// The AMD non-NVIDIA stubs (`exchangeMulticast` / `getMultimemDeviceMemPtr`
// throwing "not supported on AMD") are verified by code inspection plus the
// existing AMD conda CI compile run; they cannot easily be exercised here
// because this target is `disable_amd_ci = True` (the rest of the GPU multicast
// surface is NVIDIA-only).

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/futures/Future.h>
#include <folly/init/Init.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "comms/common/bootstrap/tests/MockBootstrap.h"
#include "comms/prims/memory/GpuMemHandler.h"
#include "comms/prims/memory/MultimemHandler.h"
#include "comms/testinfra/DistEnvironmentBase.h"
#include "comms/testinfra/DistTestBase.h"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::prims::tests {

namespace {

std::vector<int> identityRankMap(int size) {
  std::vector<int> rankMap(static_cast<std::size_t>(size));
  for (int rank = 0; rank < size; ++rank) {
    rankMap[static_cast<std::size_t>(rank)] = rank;
  }
  return rankMap;
}

std::shared_ptr<meta::comms::IBootstrap> makeBootstrap(
    const std::string& prefix) {
  return std::shared_ptr<meta::comms::IBootstrap>(
      meta::comms::createBootstrap(prefix));
}

// Mirrors the gate used by MultimemHandlerTest: if any rank reports the device
// can't host a multimem object, skip on every rank. Avoids partial-team setup
// where some ranks would throw inside exchange().
bool allRanksMultimemSupported(
    const std::shared_ptr<meta::comms::IBootstrap>& bootstrap,
    int rank,
    int nRanks,
    int cudaDevice) {
  std::vector<int> supported(static_cast<std::size_t>(nRanks));
  supported[static_cast<std::size_t>(rank)] =
      MultimemHandler::isMultimemSupported(cudaDevice) ? 1 : 0;
  auto rc =
      bootstrap->allGather(supported.data(), sizeof(int), rank, nRanks).get();
  EXPECT_EQ(rc, 0);
  if (rc != 0) {
    return false;
  }
  for (const auto value : supported) {
    if (value == 0) {
      return false;
    }
  }
  return true;
}

// Picks the strongest VMM-capable sharing mode for the multicast tests:
// kFabric on GB200 (and on H100 with IMEX), kPosixFd on H100 NVLS without
// IMEX. Returns std::nullopt when detectBestMode() can only offer kCudaIpc;
// cudaIpc has no physical handle for a multicast object to bind into, so the
// test must skip in that case. Centralizing this avoids hardcoding kFabric
// (which would fail on H100 NVLS) at every test site.
std::optional<MemSharingMode> bestVmmModeForMulticast() {
  const MemSharingMode mode = GpuMemHandler::detectBestMode();
  if (mode == MemSharingMode::kFabric || mode == MemSharingMode::kPosixFd) {
    return mode;
  }
  return std::nullopt;
}

// alignFloor for the constructor's VMM allocation must satisfy
// max(local granularity, multicast granularity); MultimemHandler exposes that
// via backingGranularity(). Falls back to 4096 only on the unsupported path,
// which we never reach (the test skips first).
std::size_t multicastBackingSize(int cudaDevice, int nvlRanks) {
  const std::size_t gran =
      MultimemHandler::backingGranularity(cudaDevice, nvlRanks);
  return gran ? gran : 4096;
}

using meta::comms::testing::MockBootstrap;
using StrictMockBootstrap = ::testing::StrictMock<MockBootstrap>;

// Builds a StrictMock that, by default, forwards every IBootstrap API to
// `real`. Tests override individual call ordinals via EXPECT_CALL to inject
// failures without losing the default delegation. Same pattern as
// MultimemHandlerTest's `makeDelegatingMock`.
std::shared_ptr<StrictMockBootstrap> makeDelegatingMock(
    const std::shared_ptr<meta::comms::IBootstrap>& real) {
  using ::testing::_;
  auto mock = std::make_shared<StrictMockBootstrap>();
  ON_CALL(*mock, allGather(_, _, _, _))
      .WillByDefault([real](void* buf, int len, int rank, int nRanks) {
        return real->allGather(buf, len, rank, nRanks);
      });
  ON_CALL(*mock, barrier(_, _)).WillByDefault([real](int rank, int nRanks) {
    return real->barrier(rank, nRanks);
  });
  ON_CALL(*mock, allGatherNvlDomain(_, _, _, _, _))
      .WillByDefault(
          [real](void* buf, int len, int r, int n, const std::vector<int>& m) {
            return real->allGatherNvlDomain(buf, len, r, n, m);
          });
  ON_CALL(*mock, barrierNvlDomain(_, _, _))
      .WillByDefault([real](int r, int n, const std::vector<int>& m) {
        return real->barrierNvlDomain(r, n, m);
      });
  ON_CALL(*mock, send(_, _, _, _))
      .WillByDefault([real](void* buf, int len, int peer, int tag) {
        return real->send(buf, len, peer, tag);
      });
  ON_CALL(*mock, recv(_, _, _, _))
      .WillByDefault([real](void* buf, int len, int peer, int tag) {
        return real->recv(buf, len, peer, tag);
      });
  return mock;
}

} // namespace

class GpuMemHandlerMulticastTestFixture : public ::testing::Test,
                                          public meta::comms::DistBaseTest {
 protected:
  void SetUp() override {
    distSetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    distTearDown();
  }
};

// Happy path on the auto-selected VMM mode. detectBestMode picks kFabric when
// the driver + IMEX support fabric (GB200-class), otherwise kPosixFd (H100
// single-host NVLS). Either route drives exchangeMulticast through the same
// MultimemHandler integration, exercising the local pending->multimem_
// commit-on-success path. We also verify pointer stability on a matching
// re-entry (the early `if (multimem_) return` branch must not invalidate the
// previously-vended pointer).
TEST_F(GpuMemHandlerMulticastTestFixture, ExchangeMulticastHappyPathAutoMode) {
  if (numRanks < 3) {
    GTEST_SKIP() << "multimem requires >=3 NVL ranks";
  }
  auto bootstrap = makeBootstrap("gpumem_multicast_happy_auto");
  if (!allRanksMultimemSupported(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "multimem unsupported on this host";
  }
  const auto modeOpt = bestVmmModeForMulticast();
  if (!modeOpt) {
    GTEST_SKIP() << "multicast requires VMM mode (kFabric/kPosixFd); "
                    "detectBestMode returned kCudaIpc";
  }
  const std::size_t allocSize = multicastBackingSize(localRank, numRanks);
  GpuMemHandler handler(
      bootstrap,
      globalRank,
      numRanks,
      allocSize,
      *modeOpt,
      /*alignFloor=*/allocSize);
  handler.exchangeMemPtrs();

  EXPECT_THROW(handler.getMultimemDeviceMemPtr(), std::runtime_error);

  handler.exchangeMulticast(globalRank, identityRankMap(numRanks), localRank);
  auto* mcPtr = handler.getMultimemDeviceMemPtr();
  EXPECT_NE(mcPtr, nullptr);

  // Matching re-entry: must hit the early `if (multimem_) return` and leave
  // the existing overlay's pointer intact (no rebind, no throw).
  handler.exchangeMulticast(globalRank, identityRankMap(numRanks), localRank);
  EXPECT_EQ(mcPtr, handler.getMultimemDeviceMemPtr());

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

// A failed exchangeMulticast must NOT make subsequent calls a silent no-op.
//
// The V1.6 fix in GpuMemHandler::exchangeMulticast constructs into a local
// `pending` unique_ptr, calls pending->exchange(), then assigns multimem_ only
// on success. The expected behavior chain:
//   1. First call: inject a failure into the NVL-domain bootstrap → throws.
//   2. multimem_ is still null (the local pending was destroyed by the throw).
//   3. Second call: must construct a new pending, retry exchange(), and on
//      success produce a valid pointer. It must NOT hit `if (multimem_) return`
//      because that branch is reserved for previously-successful exchanges.
//
// Failure injection point: agreeOnHandleType (the first allGatherNvlDomain
// inside MultimemHandler::exchange()) returns -1. exchangeMemPtrs() runs first
// and only uses the non-NVL-domain `allGather` / `barrier` APIs, so the
// injection does not corrupt the pre-multicast setup.
TEST_F(
    GpuMemHandlerMulticastTestFixture,
    ExchangeMulticastFailedRetryIsNotSilentSuccess) {
  using ::testing::_;
  using ::testing::AnyNumber;
  using ::testing::InSequence;
  if (numRanks < 3) {
    GTEST_SKIP() << "multimem requires >=3 NVL ranks";
  }
  auto real = makeBootstrap("gpumem_multicast_failed_retry");
  if (!allRanksMultimemSupported(real, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "multimem unsupported on this host";
  }
  auto mock = makeDelegatingMock(real);

  // Drive the first NVL-domain allGather of the first exchangeMulticast call
  // to fail with -1; every subsequent NVL-domain allGather delegates to `real`
  // so the retry can succeed. barrierNvlDomain is allowed AnyNumber of times
  // (always delegating); StrictMock would otherwise flag any incidental call
  // as an uninteresting expectation.
  {
    InSequence s;
    EXPECT_CALL(*mock, allGatherNvlDomain(_, _, _, _, _))
        .WillOnce([](void*, int, int, int, const std::vector<int>&) {
          return folly::makeSemiFuture(-1);
        });
    EXPECT_CALL(*mock, allGatherNvlDomain(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(
            [real](void* b, int l, int r, int n, const std::vector<int>& m) {
              return real->allGatherNvlDomain(b, l, r, n, m);
            });
  }
  EXPECT_CALL(*mock, barrierNvlDomain(_, _, _))
      .Times(AnyNumber())
      .WillRepeatedly([real](int r, int n, const std::vector<int>& m) {
        return real->barrierNvlDomain(r, n, m);
      });
  // GpuMemHandler::exchangeMemPtrs (which runs before exchangeMulticast) uses
  // the non-NVL-domain `allGather` and `barrier` APIs via NvlMemExchange. With
  // StrictMock, ON_CALL alone does NOT authorize unexpected calls -- we need
  // explicit EXPECT_CALL with AnyNumber so the pre-multicast setup doesn't
  // trip the uninteresting-call check.
  EXPECT_CALL(*mock, allGather(_, _, _, _))
      .Times(AnyNumber())
      .WillRepeatedly([real](void* buf, int len, int rank, int nRanks) {
        return real->allGather(buf, len, rank, nRanks);
      });
  EXPECT_CALL(*mock, barrier(_, _))
      .Times(AnyNumber())
      .WillRepeatedly(
          [real](int rank, int nRanks) { return real->barrier(rank, nRanks); });

  const auto modeOpt = bestVmmModeForMulticast();
  if (!modeOpt) {
    GTEST_SKIP() << "multicast requires VMM mode (kFabric/kPosixFd); "
                    "detectBestMode returned kCudaIpc";
  }
  const std::size_t allocSize = multicastBackingSize(localRank, numRanks);
  GpuMemHandler handler(
      mock,
      globalRank,
      numRanks,
      allocSize,
      *modeOpt,
      /*alignFloor=*/allocSize);
  handler.exchangeMemPtrs();

  EXPECT_THROW(
      handler.exchangeMulticast(
          globalRank, identityRankMap(numRanks), localRank),
      std::runtime_error);
  // Pointer must still throw - the failed exchange must not vend a stale ptr.
  EXPECT_THROW(handler.getMultimemDeviceMemPtr(), std::runtime_error);

  // Retry: must actually re-attempt, not silently return success. A non-null
  // pointer here is the proof that the early `if (multimem_) return` branch
  // was NOT hit (multimem_ was left null by the failed first attempt).
  handler.exchangeMulticast(globalRank, identityRankMap(numRanks), localRank);
  EXPECT_NE(handler.getMultimemDeviceMemPtr(), nullptr);

  ASSERT_EQ(real->barrier(globalRank, numRanks).get(), 0);
}

// After a successful exchangeMulticast, re-entry with a different commRank
// must throw rather than silently no-op. Without the V1.6 mismatch check the
// second call would just hit `if (multimem_) return` and the caller would
// think they had bound the multicast object to a different topology.
TEST_F(
    GpuMemHandlerMulticastTestFixture,
    ExchangeMulticastMismatchedReentryCommRankThrows) {
  if (numRanks < 3) {
    GTEST_SKIP() << "multimem requires >=3 NVL ranks";
  }
  auto bootstrap = makeBootstrap("gpumem_multicast_mismatch_commrank");
  if (!allRanksMultimemSupported(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "multimem unsupported on this host";
  }
  const auto modeOpt = bestVmmModeForMulticast();
  if (!modeOpt) {
    GTEST_SKIP() << "multicast requires VMM mode (kFabric/kPosixFd); "
                    "detectBestMode returned kCudaIpc";
  }
  const std::size_t allocSize = multicastBackingSize(localRank, numRanks);
  GpuMemHandler handler(
      bootstrap,
      globalRank,
      numRanks,
      allocSize,
      *modeOpt,
      /*alignFloor=*/allocSize);
  handler.exchangeMemPtrs();

  handler.exchangeMulticast(globalRank, identityRankMap(numRanks), localRank);
  try {
    // The intra-rank validation runs BEFORE any bootstrap call, so passing a
    // commRank not in the original team is observable locally without
    // requiring cooperating peers to also call.
    handler.exchangeMulticast(
        /*commRank=*/globalRank + numRanks,
        identityRankMap(numRanks),
        localRank);
    FAIL() << "expected std::runtime_error on commRank mismatch";
  } catch (const std::runtime_error& ex) {
    EXPECT_NE(
        std::string(ex.what()).find("re-entered with different"),
        std::string::npos)
        << ex.what();
  }

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    GpuMemHandlerMulticastTestFixture,
    ExchangeMulticastMismatchedReentryRankMapThrows) {
  if (numRanks < 3) {
    GTEST_SKIP() << "multimem requires >=3 NVL ranks";
  }
  auto bootstrap = makeBootstrap("gpumem_multicast_mismatch_rankmap");
  if (!allRanksMultimemSupported(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "multimem unsupported on this host";
  }
  const auto modeOpt = bestVmmModeForMulticast();
  if (!modeOpt) {
    GTEST_SKIP() << "multicast requires VMM mode (kFabric/kPosixFd); "
                    "detectBestMode returned kCudaIpc";
  }
  const std::size_t allocSize = multicastBackingSize(localRank, numRanks);
  GpuMemHandler handler(
      bootstrap,
      globalRank,
      numRanks,
      allocSize,
      *modeOpt,
      /*alignFloor=*/allocSize);
  handler.exchangeMemPtrs();

  handler.exchangeMulticast(globalRank, identityRankMap(numRanks), localRank);

  // Append an extra rank id to make the second map differ from the first.
  // Local-only check, no bootstrap traffic.
  std::vector<int> altMap = identityRankMap(numRanks);
  altMap.push_back(numRanks);
  try {
    handler.exchangeMulticast(globalRank, altMap, localRank);
    FAIL() << "expected std::runtime_error on nvlRankToCommRank mismatch";
  } catch (const std::runtime_error& ex) {
    EXPECT_NE(
        std::string(ex.what()).find("re-entered with different"),
        std::string::npos)
        << ex.what();
  }

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(
    GpuMemHandlerMulticastTestFixture,
    ExchangeMulticastMismatchedReentryCudaDeviceThrows) {
  if (numRanks < 3) {
    GTEST_SKIP() << "multimem requires >=3 NVL ranks";
  }
  auto bootstrap = makeBootstrap("gpumem_multicast_mismatch_cudadev");
  if (!allRanksMultimemSupported(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "multimem unsupported on this host";
  }
  const auto modeOpt = bestVmmModeForMulticast();
  if (!modeOpt) {
    GTEST_SKIP() << "multicast requires VMM mode (kFabric/kPosixFd); "
                    "detectBestMode returned kCudaIpc";
  }
  const std::size_t allocSize = multicastBackingSize(localRank, numRanks);
  GpuMemHandler handler(
      bootstrap,
      globalRank,
      numRanks,
      allocSize,
      *modeOpt,
      /*alignFloor=*/allocSize);
  handler.exchangeMemPtrs();

  handler.exchangeMulticast(globalRank, identityRankMap(numRanks), localRank);
  try {
    // -1 is guaranteed != localRank for every valid CUDA device ordinal; the
    // local mismatch check fires before any device touch.
    handler.exchangeMulticast(
        globalRank, identityRankMap(numRanks), /*cudaDevice=*/-1);
    FAIL() << "expected std::runtime_error on cudaDevice mismatch";
  } catch (const std::runtime_error& ex) {
    EXPECT_NE(
        std::string(ex.what()).find("re-entered with different"),
        std::string::npos)
        << ex.what();
  }

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::DistEnvironmentBase());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
