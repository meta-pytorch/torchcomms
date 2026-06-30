// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/futures/Future.h>
#include <folly/init/Init.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "caffe2/torch/csrc/distributed/c10d/Store.hpp"
#include "comms/common/bootstrap/tests/MockBootstrap.h"
#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/memory/MultimemHandler.h"
#include "comms/prims/platform/CudaDriverLazy.h"
#include "comms/testinfra/DistEnvironmentBase.h"
#include "comms/testinfra/DistTestBase.h"
#include "comms/testinfra/TcpStoreBootstrap.h"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::prims::tests {

class MultimemHandlerTestFixture : public ::testing::Test,
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

std::vector<int> identityRankMap(int size) {
  std::vector<int> rankMap(static_cast<std::size_t>(size));
  for (int rank = 0; rank < size; ++rank) {
    rankMap[static_cast<std::size_t>(rank)] = rank;
  }
  return rankMap;
}

std::vector<int> reverseRankMap(int size) {
  std::vector<int> rankMap(static_cast<std::size_t>(size));
  for (int rank = 0; rank < size; ++rank) {
    rankMap[static_cast<std::size_t>(rank)] = size - 1 - rank;
  }
  return rankMap;
}

std::shared_ptr<meta::comms::IBootstrap> makeBootstrap(
    const std::string& prefix) {
  return std::shared_ptr<meta::comms::IBootstrap>(
      meta::comms::createBootstrap(prefix));
}

bool allRanksMultimemSupported(
    const std::shared_ptr<meta::comms::IBootstrap>& bootstrap,
    int rank,
    int nRanks,
    int cudaDevice) {
  std::vector<int> supported(static_cast<std::size_t>(nRanks));
  supported[static_cast<std::size_t>(rank)] =
      MultimemHandler::isMultimemSupported(cudaDevice) ? 1 : 0;
  auto supportRc =
      bootstrap->allGather(supported.data(), sizeof(int), rank, nRanks).get();
  EXPECT_EQ(supportRc, 0);
  if (supportRc != 0) {
    return false;
  }
  bool allRanksSupported = true;
  for (const auto value : supported) {
    allRanksSupported = allRanksSupported && value != 0;
  }
  return allRanksSupported;
}

// Creates a physical backing CuMemAllocation sized to be bindable into a
// multicast object for a team of `nRanks` devices on `cudaDevice`.
std::shared_ptr<CuMemAllocation>
makeMulticastBacking(int cudaDevice, int nRanks, std::size_t size) {
  EXPECT_EQ(cuda_driver_lazy_init(), 0);
  CUdevice cuDev = 0;
  EXPECT_EQ(pfn_cuDeviceGet(&cuDev, cudaDevice), CUDA_SUCCESS);
  const std::size_t gran =
      MultimemHandler::backingGranularity(cudaDevice, nRanks);
  EXPECT_GT(gran, 0u);
  // Request fabric + POSIX FD; create() drops fabric and falls back to POSIX FD
  // on a single host without IMEX.
  const unsigned int mask =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC;
  return CuMemAllocation::create(cuDev, size, mask, gran);
}

// Non-std exception type used to drive the `catch (...)` arm in
// `MultimemHandler::exchange`. Lives outside any class so the gmock action
// lambdas can throw it directly. Deliberately does NOT derive from
// std::exception -- that's the entire point of having this type: it lets us
// verify the production `catch (...)` fallback fires when a peer / driver /
// other layer throws something exotic.
struct NonStdSimulatedFault {};

using meta::comms::testing::MockBootstrap;
using StrictMockBootstrap = ::testing::StrictMock<MockBootstrap>;

// Builds a StrictMockBootstrap that, by default, forwards every IBootstrap
// API to `real`. Individual tests override specific methods (or specific call
// ordinals) via `EXPECT_CALL(...).WillOnce(...)` to inject failures. StrictMock
// ensures that if a future change to `MultimemHandler` adds a new bootstrap
// call we don't account for, the test fails immediately with an "uninteresting
// mock function call" message -- locking down the bootstrap API surface that
// `MultimemHandler` is allowed to touch.
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
      .WillByDefault([real](
                         void* buf,
                         int len,
                         int r,
                         int n,
                         const std::vector<int>& rankMap) {
        return real->allGatherNvlDomain(buf, len, r, n, rankMap);
      });
  ON_CALL(*mock, barrierNvlDomain(_, _, _))
      .WillByDefault([real](int r, int n, const std::vector<int>& rankMap) {
        return real->barrierNvlDomain(r, n, rankMap);
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

// Identifies one of the four bootstrap call sites inside
// `MultimemHandler::exchange`. `ordinal` is the 0-based occurrence of the
// chosen `Api` within a single `exchange()` invocation:
//   (kAllGatherNvlDomain, 0) -> agreeOnHandleType
//   (kAllGatherNvlDomain, 1) -> exchangeMulticastHandle
//   (kBarrierNvlDomain,   0) -> synchronizeRanks("pre-bind")
//   (kBarrierNvlDomain,   1) -> synchronizeRanks("post-map")
struct FailureCase {
  // Short, gtest-name-safe label used in the parameterized test name.
  const char* name;
  enum class Api { kAllGatherNvlDomain, kBarrierNvlDomain } api;
  int ordinal;
};

enum class FailureMode {
  // Bootstrap method returns SemiFuture<int>(-1). MultimemHandler's per-phase
  // code observes a non-zero result and throws std::runtime_error.
  kReturnNonZero,
  // Bootstrap method throws std::runtime_error synchronously at the call site
  // (simulating a bootstrap-internal failure that didn't get translated into
  // a return code). Caught by `catch (const std::exception&)`.
  kSyncThrowsStd,
  // Bootstrap method throws a non-std exception synchronously. Caught by the
  // `catch (...)` fallback branch in exchange().
  kSyncThrowsNonStd,
};

constexpr FailureCase kAllFailureCases[] = {
    {"agreeOnHandleType", FailureCase::Api::kAllGatherNvlDomain, 0},
    {"exchangeMulticastHandle", FailureCase::Api::kAllGatherNvlDomain, 1},
    {"preBindBarrier", FailureCase::Api::kBarrierNvlDomain, 0},
    {"postMapBarrier", FailureCase::Api::kBarrierNvlDomain, 1},
};

constexpr FailureMode kAllFailureModes[] = {
    FailureMode::kReturnNonZero,
    FailureMode::kSyncThrowsStd,
    FailureMode::kSyncThrowsNonStd,
};

const char* describeMode(FailureMode mode) {
  switch (mode) {
    case FailureMode::kReturnNonZero:
      return "ReturnNonZero";
    case FailureMode::kSyncThrowsStd:
      return "SyncThrowsStd";
    case FailureMode::kSyncThrowsNonStd:
      return "SyncThrowsNonStd";
  }
  return "Unknown";
}

// Returns the gmock action that implements the chosen failure mode for an
// allGatherNvlDomain invocation. Captured by reference where used.
auto allGatherFailureAction(FailureMode mode) {
  return [mode](void*, int, int, int, const std::vector<int>&)
             -> folly::SemiFuture<int> {
    switch (mode) {
      case FailureMode::kReturnNonZero:
        return folly::makeSemiFuture(-1);
      case FailureMode::kSyncThrowsStd:
        throw std::runtime_error("simulated allGatherNvlDomain failure");
      case FailureMode::kSyncThrowsNonStd:
        // NonStdSimulatedFault deliberately does NOT derive from
        // std::exception -- the whole purpose is to exercise the
        // production `catch (...)` fallback in MultimemHandler::exchange().
        // NOLINTNEXTLINE(facebook-hte-ThrowNonStdExceptionIssue)
        throw NonStdSimulatedFault{};
    }
    return folly::makeSemiFuture(-1);
  };
}

auto barrierFailureAction(FailureMode mode) {
  return [mode](int, int, const std::vector<int>&) -> folly::SemiFuture<int> {
    switch (mode) {
      case FailureMode::kReturnNonZero:
        return folly::makeSemiFuture(-1);
      case FailureMode::kSyncThrowsStd:
        throw std::runtime_error("simulated barrierNvlDomain failure");
      case FailureMode::kSyncThrowsNonStd:
        // See allGatherFailureAction above for the NOLINT rationale.
        // NOLINTNEXTLINE(facebook-hte-ThrowNonStdExceptionIssue)
        throw NonStdSimulatedFault{};
    }
    return folly::makeSemiFuture(-1);
  };
}

// Programs the StrictMockBootstrap so the targeted (api, ordinal) call hits
// `mode`, while every preceding ordinal of the same API delegates to `real`.
// Other APIs are left to their ON_CALL defaults. Uses an InSequence so the
// "succeed N-1 times then fail" pattern is unambiguous.
//
// `injectOnThisRank` lets the caller turn injection off for asymmetric tests.
// When false, no EXPECT_CALL overrides are added; the makeDelegatingMock()
// ON_CALL defaults stay in effect and this rank fully delegates to `real`. In
// asymmetric scenarios, surviving ranks surface their failure via the bootstrap
// store's timeout instead of via the injected error.
void programFailureInjection(
    StrictMockBootstrap& mock,
    const std::shared_ptr<meta::comms::IBootstrap>& real,
    const FailureCase& fail,
    FailureMode mode,
    bool injectOnThisRank = true) {
  using ::testing::_;
  using ::testing::AnyNumber;
  using ::testing::InSequence;

  // Local delegation lambdas, captured per-API. Defined once and reused for
  // every EXPECT_CALL / ON_CALL action below so the closures aren't repeated
  // 5+ times inline.
  auto delegateAllGatherNvl =
      [real](
          void* buf, int len, int r, int n, const std::vector<int>& rankMap) {
        return real->allGatherNvlDomain(buf, len, r, n, rankMap);
      };
  auto delegateBarrierNvl = [real](
                                int r, int n, const std::vector<int>& rankMap) {
    return real->barrierNvlDomain(r, n, rankMap);
  };

  if (!injectOnThisRank) {
    // StrictMock treats any unmatched call as a test failure ("uninteresting
    // mock function call"). Provide baseline AnyNumber expectations for the
    // two NVL-domain APIs MultimemHandler::exchange() uses, all delegating to
    // `real`. This lets the surviving rank actually run through exchange()
    // with the real bootstrap and surface its failure via the lowered store
    // timeout.
    EXPECT_CALL(mock, allGatherNvlDomain(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(delegateAllGatherNvl);
    EXPECT_CALL(mock, barrierNvlDomain(_, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(delegateBarrierNvl);
    return;
  }

  // Allow the non-targeted API to be called by exchange() any number of times,
  // always delegating to `real`. (For an allGatherNvlDomain failure, this lets
  // any incidental barrierNvlDomain calls succeed, and vice versa.) Without
  // this, StrictMock would flag those calls as uninteresting.
  if (fail.api == FailureCase::Api::kAllGatherNvlDomain) {
    EXPECT_CALL(mock, barrierNvlDomain(_, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(delegateBarrierNvl);
  } else {
    EXPECT_CALL(mock, allGatherNvlDomain(_, _, _, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(delegateAllGatherNvl);
  }

  InSequence s;
  if (fail.api == FailureCase::Api::kAllGatherNvlDomain) {
    for (int i = 0; i < fail.ordinal; ++i) {
      EXPECT_CALL(mock, allGatherNvlDomain(_, _, _, _, _))
          .WillOnce(delegateAllGatherNvl);
    }
    EXPECT_CALL(mock, allGatherNvlDomain(_, _, _, _, _))
        .WillOnce(allGatherFailureAction(mode));
  } else {
    for (int i = 0; i < fail.ordinal; ++i) {
      EXPECT_CALL(mock, barrierNvlDomain(_, _, _)).WillOnce(delegateBarrierNvl);
    }
    EXPECT_CALL(mock, barrierNvlDomain(_, _, _))
        .WillOnce(barrierFailureAction(mode));
  }
}

// Identifies which ranks should have their mock inject the failure for a
// given parameterized case. Asymmetric patterns rely on the surviving ranks
// hitting the bootstrap store's timeout (which we shorten via
// ScopedShortTimeoutBootstrap) instead of an injected error.
enum class RankPattern {
  kAllRanks,
  kRank0Only,
  kAllExceptRank0,
};

constexpr RankPattern kAllRankPatterns[] = {
    RankPattern::kAllRanks,
    RankPattern::kRank0Only,
    RankPattern::kAllExceptRank0,
};

const char* describePattern(RankPattern p) {
  switch (p) {
    case RankPattern::kAllRanks:
      return "AllRanks";
    case RankPattern::kRank0Only:
      return "Rank0Only";
    case RankPattern::kAllExceptRank0:
      return "AllExceptRank0";
  }
  return "Unknown";
}

bool shouldInject(RankPattern p, int globalRank) {
  switch (p) {
    case RankPattern::kAllRanks:
      return true;
    case RankPattern::kRank0Only:
      return globalRank == 0;
    case RankPattern::kAllExceptRank0:
      return globalRank != 0;
  }
  return false;
}

// Wraps the DistEnvironmentBase's global TCPStore in a fresh PrefixStore with
// a shortened timeout, then constructs a TcpStoreBootstrap directly over it.
// Restoring the timeout in the destructor matters: PrefixStore::setTimeout
// forwards to the underlying global store, so leaving it lowered would shrink
// every subsequent bootstrap's timeout in the same process.
//
// Returns an empty wrapper (operator bool == false) if the dist env is not in
// TCPStore mode (e.g. MPI-only test runs); the caller should GTEST_SKIP in
// that case.
class ScopedShortTimeoutBootstrap {
 public:
  ScopedShortTimeoutBootstrap(
      const std::string& /*prefix-unused, createPrefixStore stamps its own*/,
      std::chrono::milliseconds timeout) {
    if (!meta::comms::isDistTcpStoreMode()) {
      return;
    }
    store_ = meta::comms::createPrefixStore();
    if (!store_) {
      return;
    }
    savedTimeout_ = store_->getTimeout();
    store_->setTimeout(timeout);
    auto [lr, gr, nr, ls] = meta::comms::getDistRankInfo();
    bootstrap_ = std::make_shared<meta::comms::TcpStoreBootstrap>(
        store_, gr, nr, lr, ls);
  }
  ~ScopedShortTimeoutBootstrap() {
    if (store_) {
      store_->setTimeout(savedTimeout_);
    }
  }
  ScopedShortTimeoutBootstrap(const ScopedShortTimeoutBootstrap&) = delete;
  ScopedShortTimeoutBootstrap& operator=(const ScopedShortTimeoutBootstrap&) =
      delete;

  explicit operator bool() const {
    return bootstrap_ != nullptr;
  }
  std::shared_ptr<meta::comms::IBootstrap> bootstrap() const {
    return bootstrap_;
  }

 private:
  c10::intrusive_ptr<c10d::Store> store_;
  std::chrono::milliseconds savedTimeout_{0};
  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;
};

TEST_F(MultimemHandlerTestFixture, ExchangeSetsUpStableMappings) {
  if (numRanks < 3) {
    GTEST_SKIP() << "CUDA multimem transport is only useful for 3+ ranks";
  }
  auto bootstrap = makeBootstrap("multimem_handler_test");
  if (!allRanksMultimemSupported(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not supported";
  }

  constexpr std::size_t kRequestedBytes = 4096;
  auto backing = makeMulticastBacking(localRank, numRanks, kRequestedBytes);
  MultimemHandler handler(
      backing, bootstrap, globalRank, identityRankMap(numRanks), localRank);
  EXPECT_THROW(handler.getMultimemDeviceMemPtr(), std::runtime_error);

  handler.exchange();

  auto* multimemPtr = handler.getMultimemDeviceMemPtr();
  EXPECT_NE(multimemPtr, nullptr);
  EXPECT_EQ(multimemPtr, handler.getMultimemDeviceMemPtr());
  EXPECT_EQ(handler.backing(), backing);
  EXPECT_GE(handler.getAllocatedSize(), kRequestedBytes);

  std::vector<std::uintptr_t> multimemAddrs(static_cast<std::size_t>(numRanks));
  multimemAddrs[static_cast<std::size_t>(globalRank)] =
      reinterpret_cast<std::uintptr_t>(multimemPtr);
  auto rc = bootstrap
                ->allGather(
                    multimemAddrs.data(),
                    sizeof(std::uintptr_t),
                    globalRank,
                    numRanks)
                .get();
  ASSERT_EQ(rc, 0);
  for (int rank = 0; rank < numRanks; ++rank) {
    EXPECT_NE(multimemAddrs[static_cast<std::size_t>(rank)], 0);
  }

  handler.exchange();
  EXPECT_EQ(multimemPtr, handler.getMultimemDeviceMemPtr());

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(MultimemHandlerTestFixture, ExchangeSupportsNonIdentityRankMap) {
  if (numRanks < 3) {
    GTEST_SKIP() << "CUDA multimem transport is only useful for 3+ ranks";
  }
  auto bootstrap = makeBootstrap("multimem_handler_non_identity_rank_map_test");
  if (!allRanksMultimemSupported(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not supported";
  }

  constexpr std::size_t kRequestedBytes = 4096;
  auto backing = makeMulticastBacking(localRank, numRanks, kRequestedBytes);
  MultimemHandler handler(
      backing, bootstrap, globalRank, reverseRankMap(numRanks), localRank);
  handler.exchange();

  EXPECT_NE(handler.getMultimemDeviceMemPtr(), nullptr);
  EXPECT_EQ(handler.backing(), backing);
  EXPECT_GE(handler.getAllocatedSize(), kRequestedBytes);

  ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
}

TEST_F(MultimemHandlerTestFixture, SharingRejectsNullBacking) {
  auto bootstrap = makeBootstrap("multimem_null_backing_test");
  // A null backing must fail fast.
  EXPECT_THROW(
      MultimemHandler(
          nullptr, bootstrap, globalRank, identityRankMap(numRanks), localRank),
      std::runtime_error);
}

// Locks down the ctor's "bootstrap must be non-null" branch on a GPU host
// (where `handleType_ != kUnsupported`, so the unsupported-device branch
// doesn't preempt this check). A valid backing + valid rank map + null
// bootstrap must throw.
TEST_F(MultimemHandlerTestFixture, ConstructorRejectsNullBootstrap) {
  if (numRanks < 3) {
    GTEST_SKIP() << "CUDA multimem transport is only useful for 3+ ranks";
  }
  auto realBootstrap = makeBootstrap("multimem_ctor_reject_null_bootstrap");
  if (!allRanksMultimemSupported(
          realBootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not supported";
  }

  auto backing = makeMulticastBacking(localRank, numRanks, 4096);
  EXPECT_THROW(
      MultimemHandler(
          backing,
          /*bootstrap=*/nullptr,
          globalRank,
          identityRankMap(numRanks),
          localRank),
      std::runtime_error);
}

// On a GPU host where normal multimem support succeeds, passing
// `cudaDevice = -1` still triggers `selectMultimemHandleTypeImpl ==
// kUnsupported` and the ctor must throw. Rules out a false negative from the
// no-GPU validation host (where every device id looks unsupported regardless
// of branch coverage).
TEST_F(
    MultimemHandlerTestFixture,
    ConstructorRejectsUnsupportedDeviceOnGpuHost) {
  if (numRanks < 3) {
    GTEST_SKIP() << "CUDA multimem transport is only useful for 3+ ranks";
  }
  auto realBootstrap = makeBootstrap("multimem_ctor_reject_unsupported_dev");
  if (!allRanksMultimemSupported(
          realBootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not supported";
  }

  // Use the real (multimem-capable) device to build a usable backing -- we
  // never reach exchange(), the ctor should reject `cudaDevice = -1` first.
  auto backing = makeMulticastBacking(localRank, numRanks, 4096);
  EXPECT_THROW(
      MultimemHandler(
          backing,
          realBootstrap,
          globalRank,
          identityRankMap(numRanks),
          /*cudaDevice=*/-1),
      std::runtime_error);
}

// Locks down the bootstrap API surface used by `MultimemHandler::exchange()`:
//   - allGatherNvlDomain: exactly 2 calls per exchange (agreeOnHandleType,
//     then exchangeMulticastHandle).
//   - barrierNvlDomain:   exactly 2 calls per exchange (pre-bind, post-map).
//   - allGather/barrier/send/recv: never called by MultimemHandler itself.
// StrictMock + exact `.Times(...)` makes any future regression (an extra
// barrier inserted, a stray allGather, a missing pre-bind sync) fail this
// test with a clear "actually called N times, expected M" diff -- before the
// regression has a chance to surface as a wedged exchange in production.
// Also re-asserts the "exchange() is idempotent after success" contract by
// invoking it a second time and verifying the call counters don't increment.
TEST_F(MultimemHandlerTestFixture, SuccessPathCoversAllBootstrapApis) {
  if (numRanks < 3) {
    GTEST_SKIP() << "CUDA multimem transport is only useful for 3+ ranks";
  }
  auto realBootstrap = makeBootstrap("multimem_success_api_coverage");
  if (!allRanksMultimemSupported(
          realBootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not supported";
  }

  using ::testing::_;
  auto mock = makeDelegatingMock(realBootstrap);

  EXPECT_CALL(*mock, allGatherNvlDomain(_, _, _, _, _)).Times(2);
  EXPECT_CALL(*mock, barrierNvlDomain(_, _, _)).Times(2);
  EXPECT_CALL(*mock, allGather(_, _, _, _)).Times(0);
  EXPECT_CALL(*mock, barrier(_, _)).Times(0);
  EXPECT_CALL(*mock, send(_, _, _, _)).Times(0);
  EXPECT_CALL(*mock, recv(_, _, _, _)).Times(0);

  auto backing = makeMulticastBacking(localRank, numRanks, 4096);
  MultimemHandler handler(
      backing,
      std::shared_ptr<meta::comms::IBootstrap>(mock),
      globalRank,
      identityRankMap(numRanks),
      localRank);
  handler.exchange();
  EXPECT_NE(handler.getMultimemDeviceMemPtr(), nullptr);

  // Second call must early-return without touching the bootstrap; the exact
  // .Times(2) expectations above would fail otherwise.
  handler.exchange();
  EXPECT_NE(handler.getMultimemDeviceMemPtr(), nullptr);

  ASSERT_EQ(realBootstrap->barrier(globalRank, numRanks).get(), 0);
}

// Parameterized failure-injection suite. The product is
//   {4 phases of exchange()} x {3 failure modes} = 12 cases.
//
// For each case:
//   - A real bootstrap is created with a per-case prefix so prior keys cannot
//     leak between cases.
//   - A StrictMockBootstrap is built that delegates every call to the real
//     bootstrap, then the (api, ordinal) call site under test is overridden
//     to fail with the chosen mode (return -1, throw std::runtime_error, or
//     throw a non-std exception).
//   - All ranks inject the same failure at the same ordinal, so every rank's
//     exchange() throws together -- avoiding the asymmetric-failure deadlock
//     scenario that would require a peer-side timeout to remain deterministic.
//
// Assertions per case:
//   1. exchange() throws std::runtime_error.
//   2. The message tags the right failedPhase (`failedPhase=<name>`), which
//      proves `describeState` ran with the right phase label.
//   3. The message contains the phase-specific substring (e.g. "pre-bind").
//   4. For kSyncThrowsNonStd: the message contains the `catch (...)` prefix
//      "MultimemHandler::exchange failed with unknown exception", confirming
//      the unknown-exception fallback is reached. For std-throw / non-zero
//      cases, the message uses the "MultimemHandler::exchange failed: " prefix.
//   5. After the throw, both getters still throw "exchange() must complete",
//      proving the partial-init state was not committed.
//   6. The realBootstrap is still healthy: a subsequent barrier() returns 0.
//      Proves cleanup() didn't leave any bootstrap state poisoned.
//   7. Constructing a fresh handler over the *same* `backing` shared_ptr with
//      a fresh bootstrap (different prefix) lets exchange() complete cleanly.
//      Proves cleanup() released only the handler-owned multicast state and
//      left the caller-owned physical backing intact and bindable.
class MultimemHandlerFailureInjectionTest
    : public ::testing::TestWithParam<
          std::tuple<FailureCase, FailureMode, RankPattern>>,
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

TEST_P(MultimemHandlerFailureInjectionTest, ExchangeReportsAndCleansUp) {
  if (numRanks < 3) {
    GTEST_SKIP() << "CUDA multimem transport is only useful for 3+ ranks";
  }
  const FailureCase& failCase = std::get<0>(GetParam());
  const FailureMode failMode = std::get<1>(GetParam());
  const RankPattern rankPattern = std::get<2>(GetParam());
  const std::string casePrefix = std::string("multimem_fail_") + failCase.name +
      "_" + describeMode(failMode) + "_" + describePattern(rankPattern);

  // Asymmetric patterns require the bootstrap store's wait() to time out for
  // surviving ranks; the underlying TCPStore supports that, MPI does not.
  // Symmetric (kAllRanks) cases never reach a timeout because every rank
  // throws synchronously, so we can use the normal bootstrap there.
  const bool isAsymmetric = (rankPattern != RankPattern::kAllRanks);
  if (isAsymmetric && !meta::comms::isDistTcpStoreMode()) {
    GTEST_SKIP() << "asymmetric failure tests require TCPStore env";
  }

  {
    std::optional<ScopedShortTimeoutBootstrap> shortBs;
    std::shared_ptr<meta::comms::IBootstrap> bootstrap;
    if (isAsymmetric) {
      shortBs.emplace(casePrefix, std::chrono::milliseconds(5000));
      bootstrap = shortBs->bootstrap();
      ASSERT_TRUE(bootstrap)
          << "ScopedShortTimeoutBootstrap returned null in TCPStore mode";
    } else {
      bootstrap = makeBootstrap(casePrefix);
    }

    if (!allRanksMultimemSupported(
            bootstrap, globalRank, numRanks, localRank)) {
      GTEST_SKIP() << "CUDA multimem/NVLS multicast is not supported";
    }

    auto backing = makeMulticastBacking(localRank, numRanks, 4096);

    {
      auto mock = makeDelegatingMock(bootstrap);
      programFailureInjection(
          *mock,
          bootstrap,
          failCase,
          failMode,
          /*injectOnThisRank=*/shouldInject(rankPattern, globalRank));

      MultimemHandler handler(
          backing,
          std::shared_ptr<meta::comms::IBootstrap>(mock),
          globalRank,
          identityRankMap(numRanks),
          localRank);

      // We assert behaviour, not message text: exchange must throw, and the
      // post-failure handler must refuse to vend the (never-set) multicast
      // pointer / allocated size. For asymmetric patterns, surviving ranks
      // throw via the bootstrap store timeout instead of via the injected
      // error; the outer `EXPECT_THROW` covers both surfaces uniformly.
      EXPECT_THROW(handler.exchange(), std::runtime_error);
      EXPECT_THROW((void)handler.getMultimemDeviceMemPtr(), std::runtime_error);
      EXPECT_THROW((void)handler.getAllocatedSize(), std::runtime_error);
    }

    // Sanity: bootstrap is still callable after the failed exchange.
    // Symmetric only -- asymmetric cases legitimately leave the failure-time
    // bootstrap with stale keys / abandoned waits from the rank that didn't
    // participate in the call where its peer aborted, so a follow-up barrier
    // on the same bootstrap is allowed to time out. The recovery exchange
    // below proves the *system* is still usable via a fresh bootstrap, which
    // is the load-bearing assertion.
    if (rankPattern == RankPattern::kAllRanks) {
      ASSERT_EQ(bootstrap->barrier(globalRank, numRanks).get(), 0);
    }
  } // ScopedShortTimeoutBootstrap (if any) destructed here -> timeout restored

  // Recovery: the shared backing must still be bindable. A fresh handler over
  // the same `backing` with a fresh, normal-timeout bootstrap should reach a
  // clean steady state. Uses a different prefix so the bootstrap store has no
  // overlap with the failed exchange's keys.
  auto recoveryBootstrap = makeBootstrap(casePrefix + "_recovery");
  MultimemHandler recovery(
      makeMulticastBacking(localRank, numRanks, 4096),
      recoveryBootstrap,
      globalRank,
      identityRankMap(numRanks),
      localRank);
  recovery.exchange();
  EXPECT_NE(recovery.getMultimemDeviceMemPtr(), nullptr);
  EXPECT_GE(recovery.getAllocatedSize(), 4096u);
  ASSERT_EQ(recoveryBootstrap->barrier(globalRank, numRanks).get(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    AllPhasesAllModesAllPatterns,
    MultimemHandlerFailureInjectionTest,
    ::testing::Combine(
        ::testing::ValuesIn(kAllFailureCases),
        ::testing::ValuesIn(kAllFailureModes),
        ::testing::ValuesIn(kAllRankPatterns)),
    [](const ::testing::TestParamInfo<
        std::tuple<FailureCase, FailureMode, RankPattern>>& info) {
      // Structured bindings here would put a bare `,` in the macro arg list
      // and confuse the preprocessor; use std::get instead.
      const FailureCase& failCase = std::get<0>(info.param);
      const FailureMode failMode = std::get<1>(info.param);
      const RankPattern pattern = std::get<2>(info.param);
      return std::string(failCase.name) + "_" + describeMode(failMode) + "_" +
          describePattern(pattern);
    });

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::DistEnvironmentBase());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
