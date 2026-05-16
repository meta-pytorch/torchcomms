// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/futures/Future.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/pipes/MultimemHandler.h"

namespace comms::pipes::tests {

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

TEST(MultimemHandlerValidationTest, RejectsEmptyRankMap) {
  expectRuntimeErrorContains(
      [] { MultimemHandler handler(makeBootstrap(), 0, {}, 0, 4096); },
      "nvlRankToCommRank must be non-empty");
}

TEST(MultimemHandlerValidationTest, RejectsMissingCommRank) {
  expectRuntimeErrorContains(
      [] { MultimemHandler handler(makeBootstrap(), 7, {0, 1, 2}, 0, 4096); },
      "commRank must appear in nvlRankToCommRank");
}

TEST(MultimemHandlerValidationTest, RejectsNegativeRankInMap) {
  expectRuntimeErrorContains(
      [] { MultimemHandler handler(makeBootstrap(), 0, {0, -1}, 0, 4096); },
      "contains a negative rank");
}

TEST(MultimemHandlerValidationTest, RejectsDuplicateRankInMap) {
  expectRuntimeErrorContains(
      [] { MultimemHandler handler(makeBootstrap(), 0, {0, 0}, 0, 4096); },
      "contains duplicate ranks");
}

TEST(MultimemHandlerValidationTest, RejectsZeroSize) {
  expectRuntimeErrorContains(
      [] { MultimemHandler handler(makeBootstrap(), 0, {0}, 0, 0); },
      "size must be non-zero");
}

TEST(MultimemHandlerValidationTest, RejectsNullBootstrap) {
  expectRuntimeErrorContains(
      [] { MultimemHandler handler(nullptr, 0, {0}, 0, 4096); },
      "bootstrap must be non-null");
}

TEST(MultimemHandlerValidationTest, NegativeCudaDeviceIsUnsupported) {
  EXPECT_FALSE(MultimemHandler::isMultimemSupported(-1));
}

} // namespace comms::pipes::tests
