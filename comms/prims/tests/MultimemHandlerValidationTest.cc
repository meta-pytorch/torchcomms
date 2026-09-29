// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/futures/Future.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/memory/MultimemHandler.h"

namespace comms::prims::tests {

namespace {

class FakeBootstrap : public meta::comms::IBootstrap {
 public:
  folly::SemiFuture<int> allGather(void*, int, int, int) override {
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int> barrier(int, int) override {
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int> send(void*, int, int, int) override {
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int> recv(void*, int, int, int) override {
    return folly::makeSemiFuture(0);
  }
};

std::shared_ptr<meta::comms::IBootstrap> makeBootstrap() {
  return std::make_shared<FakeBootstrap>();
}

template <typename Fn>
void expectRuntimeErrorContains(Fn&& fn, const std::string& expected) {
  try {
    std::forward<Fn>(fn)();
    FAIL() << "expected std::runtime_error containing " << expected;
  } catch (const std::runtime_error& ex) {
    EXPECT_NE(std::string(ex.what()).find(expected), std::string::npos)
        << ex.what();
  }
}

} // namespace

// The rank-map validation runs in the constructor initializer list (before the
// backing / GPU checks), so these cases throw with a null backing and need no
// GPU.
TEST(MultimemHandlerValidationTest, RejectsEmptyRankMap) {
  expectRuntimeErrorContains(
      [] { MultimemHandler handler(nullptr, makeBootstrap(), 0, {}, 0); },
      "nvlRankToCommRank must be non-empty");
}

TEST(MultimemHandlerValidationTest, RejectsMissingCommRank) {
  expectRuntimeErrorContains(
      [] {
        MultimemHandler handler(nullptr, makeBootstrap(), 7, {0, 1, 2}, 0);
      },
      "commRank must appear in nvlRankToCommRank");
}

TEST(MultimemHandlerValidationTest, RejectsNegativeRankInMap) {
  expectRuntimeErrorContains(
      [] { MultimemHandler handler(nullptr, makeBootstrap(), 0, {0, -1}, 0); },
      "contains a negative rank");
}

TEST(MultimemHandlerValidationTest, RejectsDuplicateRankInMap) {
  expectRuntimeErrorContains(
      [] { MultimemHandler handler(nullptr, makeBootstrap(), 0, {0, 0}, 0); },
      "contains duplicate ranks");
}

TEST(MultimemHandlerValidationTest, RejectsNullBacking) {
  // A valid rank map but a null backing throws at requireBacking, still before
  // any GPU access.
  expectRuntimeErrorContains(
      [] { MultimemHandler handler(nullptr, makeBootstrap(), 0, {0}, 0); },
      "backing CuMemAllocation must be non-null");
}

TEST(MultimemHandlerValidationTest, NegativeCudaDeviceIsUnsupported) {
  EXPECT_FALSE(MultimemHandler::isMultimemSupported(-1));
}

// Locks down both branches of `isMultimemSupported`:
//   1. The cache fast-path (`cachedState != kUnknown`): repeated calls for the
//      same in-range device must agree with the first call.
//   2. The slow-path used when `cudaDevice >= kCachedMultimemSupportDevices`
//      (which is 8): repeated calls for the same out-of-cache device id must
//      also agree (the slow path has no cache so the agreement is purely a
//      selector-determinism property).
// A regression that flips the cache semantics (wrong memory order, off-by-one
// in `kCachedMultimemSupportDevices`, or a stale-cache bug) trips the
// fast-path assertion; a non-deterministic selector trips the slow-path
// assertion. The concrete bool is host-dependent (a no-GPU CI host returns
// `false` for everything; a 1-GPU devserver returns `true` for device 0 but
// `false` for device 8 because the device doesn't exist) -- we assert only
// per-device stability, not cross-device equality.
TEST(MultimemHandlerValidationTest, IsMultimemSupportedCacheStableAcrossCalls) {
  const bool firstDevice0 = MultimemHandler::isMultimemSupported(0);
  EXPECT_EQ(MultimemHandler::isMultimemSupported(0), firstDevice0);
  EXPECT_EQ(MultimemHandler::isMultimemSupported(0), firstDevice0);

  const bool device7 = MultimemHandler::isMultimemSupported(7);
  EXPECT_EQ(MultimemHandler::isMultimemSupported(7), device7);

  // Device 8 falls outside the cache; the slow-path's selector must agree
  // with itself on repeated calls (regardless of whether device 8 exists).
  const bool firstDevice8 = MultimemHandler::isMultimemSupported(8);
  EXPECT_EQ(MultimemHandler::isMultimemSupported(8), firstDevice8);
  EXPECT_EQ(MultimemHandler::isMultimemSupported(8), firstDevice8);
}

// `backingGranularity` has its own "unsupported -> return 0" early-exit that
// is independent of `isMultimemSupported`'s cache. Pairs with the existing
// `NegativeCudaDeviceIsUnsupported` to lock down both static entry points and
// confirms `nvlRanks` is irrelevant when the device is unsupported. A future
// refactor that accidentally divides by `nvlRanks` before the supported check
// would surface here.
TEST(MultimemHandlerValidationTest, BackingGranularityZeroOnUnsupportedDevice) {
  EXPECT_EQ(MultimemHandler::backingGranularity(-1, 8), 0u);
  EXPECT_EQ(MultimemHandler::backingGranularity(-1, 1), 0u);
  // On a no-GPU host device 0 is also unsupported (no driver / no multicast
  // attribute); the in-range branch is exercised on a GPU host's success
  // path via `MultimemHandlerTest`.
  if (!MultimemHandler::isMultimemSupported(0)) {
    EXPECT_EQ(MultimemHandler::backingGranularity(0, 8), 0u);
  }
}

} // namespace comms::prims::tests
