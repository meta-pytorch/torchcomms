// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "comms/prims/bootstrap/TeamStatusAgreement.h"

namespace comms::prims::detail {
namespace {

auto gatherWithStatuses(
    std::vector<uint32_t> statuses,
    int result = 0,
    int* callCount = nullptr) {
  return [statuses = std::move(statuses), result, callCount](
             void* data, int recordSize) {
    if (callCount != nullptr) {
      ++*callCount;
    }
    EXPECT_EQ(recordSize, static_cast<int>(sizeof(TeamStatus)));
    auto* records = static_cast<TeamStatus*>(data);
    for (std::size_t rank = 0; rank < statuses.size(); ++rank) {
      records[rank].success = statuses[rank];
    }
    return result;
  };
}

void expectNestedRuntimeError(
    const std::runtime_error& error,
    std::string_view expected) {
  try {
    std::rethrow_if_nested(error);
    ADD_FAILURE() << "expected nested runtime_error";
  } catch (const std::runtime_error& nested) {
    EXPECT_EQ(nested.what(), expected);
  } catch (...) {
    ADD_FAILURE() << "unexpected nested exception type";
  }
}

TEST(TeamStatusAgreementTest, AllRanksSucceed) {
  std::vector<TeamStatus> records(3);
  int gatherCalls = 0;

  EXPECT_NO_THROW(gatherAndAgree(
      /*rank=*/1,
      /*nranks=*/3,
      records,
      std::exception_ptr{},
      "test operation",
      gatherWithStatuses({1, 1, 1}, /*result=*/0, &gatherCalls)));
  EXPECT_EQ(gatherCalls, 1);
}

TEST(TeamStatusAgreementTest, RemoteFailureUsesLowestRank) {
  std::vector<TeamStatus> records(3);

  try {
    gatherAndAgree(
        /*rank=*/0,
        /*nranks=*/3,
        records,
        std::exception_ptr{},
        "test operation",
        gatherWithStatuses({1, 0, 0}));
    FAIL() << "expected team failure";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "test operation failed on rank 1");
    EXPECT_NO_THROW(std::rethrow_if_nested(error));
  }
}

TEST(TeamStatusAgreementTest, PrimaryLocalFailureRetainsCause) {
  std::vector<TeamStatus> records(3);
  const auto localError =
      std::make_exception_ptr(std::runtime_error("rank 0 local error"));

  try {
    gatherAndAgree(
        /*rank=*/0,
        /*nranks=*/3,
        records,
        localError,
        "test operation",
        gatherWithStatuses({0, 1, 0}));
    FAIL() << "expected team failure";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(), "test operation failed on rank 0: rank 0 local error");
    expectNestedRuntimeError(error, "rank 0 local error");
  }
}

TEST(TeamStatusAgreementTest, LocalFailureRetainedWhenLowerRankFails) {
  std::vector<TeamStatus> records(3);
  const auto localError =
      std::make_exception_ptr(std::runtime_error("rank 2 local error"));

  try {
    gatherAndAgree(
        /*rank=*/2,
        /*nranks=*/3,
        records,
        localError,
        "test operation",
        gatherWithStatuses({0, 1, 0}));
    FAIL() << "expected team failure";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(),
        "test operation failed on rank 0; local rank 2 also failed: "
        "rank 2 local error");
    expectNestedRuntimeError(error, "rank 2 local error");
  }
}

TEST(TeamStatusAgreementTest, AllGatherFailureIsReported) {
  std::vector<TeamStatus> records(2);
  int gatherCalls = 0;

  try {
    gatherAndAgree(
        /*rank=*/0,
        /*nranks=*/2,
        records,
        std::exception_ptr{},
        "test operation",
        gatherWithStatuses({1, 1}, /*result=*/1, &gatherCalls));
    FAIL() << "expected allGather failure";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "test operation status allGather failed");
  }
  EXPECT_EQ(gatherCalls, 1);
}

TEST(TeamStatusAgreementTest, RejectsWrongRecordCount) {
  std::vector<TeamStatus> records(1);
  bool gatherCalled = false;

  try {
    gatherAndAgree(
        /*rank=*/0,
        /*nranks=*/2,
        records,
        std::exception_ptr{},
        "test operation",
        [&](void*, int) {
          gatherCalled = true;
          return 0;
        });
    FAIL() << "expected invalid record count";
  } catch (const std::invalid_argument& error) {
    EXPECT_STREQ(error.what(), "gatherAndAgree records size must equal nranks");
  }
  EXPECT_FALSE(gatherCalled);
}

} // namespace
} // namespace comms::prims::detail
