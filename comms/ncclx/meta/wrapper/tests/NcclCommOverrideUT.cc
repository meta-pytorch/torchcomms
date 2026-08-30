// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <memory>

#include "meta/wrapper/NcclCommOverride.h" // @manual

using ::ncclx::wrapper::ncclCommOverrideDesc;
using ::ncclx::wrapper::ncclCommUsePatAvg;

namespace {

// ncclComm is far too large for the stack and has no public constructor that
// takes the fields under test, so tests value-initialize one on the heap and
// set only what they assert on.
std::unique_ptr<ncclComm> makeComm() {
  return std::make_unique<ncclComm>();
}

} // namespace

TEST(NcclCommOverrideTest, DescIsNullSentinelForNullComm) {
  EXPECT_EQ(ncclCommOverrideDesc(nullptr), "null");
}

TEST(NcclCommOverrideTest, DescReturnsCommDescHint) {
  auto comm = makeComm();
  comm->config.commDesc = "training_dp_group";
  EXPECT_EQ(ncclCommOverrideDesc(comm.get()), "training_dp_group");
}

TEST(NcclCommOverrideTest, DescIsEmptyWhenHintUnset) {
  auto comm = makeComm();

  comm->config.commDesc = nullptr;
  EXPECT_EQ(ncclCommOverrideDesc(comm.get()), "");

  // NCCL spells "unset" as a non-null sentinel, which must not be returned as
  // if it were a string the caller can read.
  comm->config.commDesc = NCCL_CONFIG_UNDEF_PTR;
  EXPECT_EQ(ncclCommOverrideDesc(comm.get()), "");
}

TEST(NcclCommOverrideTest, UsePatAvgIsFalseForNullComm) {
  EXPECT_FALSE(ncclCommUsePatAvg(nullptr));
}

TEST(NcclCommOverrideTest, UsePatAvgReflectsCommState) {
  auto comm = makeComm();

#if defined(IS_NCCLX)
  EXPECT_FALSE(ncclCommUsePatAvg(comm.get()));
  comm->usePatAvg_ = true;
  EXPECT_TRUE(ncclCommUsePatAvg(comm.get()));
#else
  // Pristine NCCL has no usePatAvg_ field, so the accessor is always false.
  EXPECT_FALSE(ncclCommUsePatAvg(comm.get()));
#endif
}
