#include <gtest/gtest.h>

#include "meta/wrapper/DataTypeConv.h" // @manual=//comms/ncclx:nccl2.27-internal
#include "meta/wrapper/MetaFactory.h" // @manual=//comms/ncclx:nccl2.27-internal

TEST(MetaFactoryTest, CommToNccl) {
  auto commResult = commInProgress;
  ASSERT_EQ(metaCommToNccl(commResult), ncclInProgress);
}

TEST(MetaFactoryTest, NcclToComm) {
  auto ncclResult = ncclInProgress;
  ASSERT_EQ(ncclToMetaComm(ncclResult), commInProgress);
}

// These two tests will fail on dev mode due to static check, always run them in
// opt mode
TEST(MetaFactoryTest, SmallerThanZeroToNccl) {
  auto commResult = static_cast<commResult_t>(-1);
  EXPECT_THROW(metaCommToNccl(commResult), std::runtime_error);
}

TEST(MetaFactoryTest, SmallerThanZeroToMetaComm) {
  auto ncclResult = static_cast<ncclResult_t>(-1);
  EXPECT_THROW(ncclToMetaComm(ncclResult), std::runtime_error);
}

TEST(MetaFactoryTest, BiggerThanLimitToNccl) {
  auto commResult =
      static_cast<commResult_t>(static_cast<int>(commNumResults) + 1);
  EXPECT_THROW(metaCommToNccl(commResult), std::runtime_error);
}

TEST(MetaFactoryTest, BiggerThanLimitToMetaComm) {
  auto ncclResult =
      static_cast<ncclResult_t>(static_cast<int>(ncclNumResults) + 1);
  EXPECT_THROW(ncclToMetaComm(ncclResult), std::runtime_error);
}

// Test ncclToCommPattern conversion
TEST(DataTypeConvTest, NcclToCommPattern) {
  using namespace meta::comms;

  // Test all pattern conversions
  EXPECT_EQ(ncclToCommPattern(ncclPatternRing), CommPattern::Ring);
  EXPECT_EQ(ncclToCommPattern(ncclPatternRingTwice), CommPattern::RingTwice);
  EXPECT_EQ(
      ncclToCommPattern(ncclPatternPipelineFrom), CommPattern::PipelineFrom);
  EXPECT_EQ(ncclToCommPattern(ncclPatternPipelineTo), CommPattern::PipelineTo);
  EXPECT_EQ(ncclToCommPattern(ncclPatternTreeUp), CommPattern::TreeUp);
  EXPECT_EQ(ncclToCommPattern(ncclPatternTreeDown), CommPattern::TreeDown);
  EXPECT_EQ(ncclToCommPattern(ncclPatternTreeUpDown), CommPattern::TreeUpDown);
  EXPECT_EQ(
      ncclToCommPattern(ncclPatternCollnetChain), CommPattern::CollnetChain);
  EXPECT_EQ(
      ncclToCommPattern(ncclPatternCollnetDirect), CommPattern::CollnetDirect);
  EXPECT_EQ(ncclToCommPattern(ncclPatternNvls), CommPattern::Nvls);
  EXPECT_EQ(ncclToCommPattern(ncclPatternNvlsTree), CommPattern::NvlsTree);
  EXPECT_EQ(ncclToCommPattern(ncclPatternPatUp), CommPattern::PatUp);
  EXPECT_EQ(ncclToCommPattern(ncclPatternPatDown), CommPattern::PatDown);
  EXPECT_EQ(ncclToCommPattern(ncclPatternSend), CommPattern::Send);
  EXPECT_EQ(ncclToCommPattern(ncclPatternRecv), CommPattern::Recv);
}

// Test ncclToCommFunc conversion
TEST(DataTypeConvTest, NcclToCommFunc) {
  using namespace meta::comms;

  // Test all function conversions
  EXPECT_EQ(ncclToCommFunc(ncclFuncBroadcast), CommFunc::Broadcast);
  EXPECT_EQ(ncclToCommFunc(ncclFuncReduce), CommFunc::Reduce);
  EXPECT_EQ(ncclToCommFunc(ncclFuncAllGather), CommFunc::AllGather);
  EXPECT_EQ(ncclToCommFunc(ncclFuncReduceScatter), CommFunc::ReduceScatter);
  EXPECT_EQ(ncclToCommFunc(ncclFuncAllReduce), CommFunc::AllReduce);
  EXPECT_EQ(ncclToCommFunc(ncclFuncSendRecv), CommFunc::SendRecv);
  EXPECT_EQ(ncclToCommFunc(ncclFuncSend), CommFunc::Send);
  EXPECT_EQ(ncclToCommFunc(ncclFuncRecv), CommFunc::Recv);

  // Test NumFuncs
  EXPECT_EQ(
      static_cast<int>(CommFunc::NumFuncs), static_cast<int>(ncclNumFuncs));
}

// Test ncclToCommDataType conversion
TEST(DataTypeConvTest, NcclToCommDataType) {
  using namespace meta::comms;

  // Test all data type conversions
  EXPECT_EQ(ncclToCommDataType(ncclInt8), commInt8);
  EXPECT_EQ(ncclToCommDataType(ncclUint8), commUint8);
  EXPECT_EQ(ncclToCommDataType(ncclInt32), commInt32);
  EXPECT_EQ(ncclToCommDataType(ncclUint32), commUint32);
  EXPECT_EQ(ncclToCommDataType(ncclInt64), commInt64);
  EXPECT_EQ(ncclToCommDataType(ncclUint64), commUint64);
  EXPECT_EQ(ncclToCommDataType(ncclFloat16), commFloat16);
  EXPECT_EQ(ncclToCommDataType(ncclFloat32), commFloat32);
  EXPECT_EQ(ncclToCommDataType(ncclFloat64), commFloat64);
  EXPECT_EQ(ncclToCommDataType(ncclBfloat16), commBfloat16);

  // Test NumTypes
  EXPECT_EQ(static_cast<int>(commNumTypes), static_cast<int>(ncclNumTypes));
}

// Test ncclToCommRedOp conversion
TEST(DataTypeConvTest, NcclToCommRedOp) {
  using namespace meta::comms;

  // Test all reduction operation conversions
  EXPECT_EQ(ncclToCommRedOp(ncclSum), commSum);
  EXPECT_EQ(ncclToCommRedOp(ncclProd), commProd);
  EXPECT_EQ(ncclToCommRedOp(ncclMax), commMax);
  EXPECT_EQ(ncclToCommRedOp(ncclMin), commMin);

  // Test NumOps
  EXPECT_EQ(static_cast<int>(commNumOps), static_cast<int>(ncclNumOps));
}

TEST(DataTypeConvTest, NcclToCommCmpOp) {
  using namespace meta::comms;

  // Test all comparison operation conversions
  EXPECT_EQ(ncclToMetaComm(ncclCmpEQ), commCmpEQ);
  EXPECT_EQ(ncclToMetaComm(ncclCmpGE), commCmpGE);
  EXPECT_EQ(ncclToMetaComm(ncclCmpLE), commCmpLE);

  // Test NumOps
  EXPECT_EQ(static_cast<int>(commNumCmpOps), static_cast<int>(ncclNumCmpOps));
}

TEST(DataTypeConvTest, NcclToCommHints) {
  ncclx::Hints from;
  meta::comms::Hints to;

  const std::string key = "ncclx_alltoallp_skip_ctrl_msg_exchange";
  std::string val;

  to = ncclToMetaComm(from);
  EXPECT_EQ(to.get(key, val), commSuccess);
  EXPECT_EQ(val, "false");

  from.set(key, "True");
  to = ncclToMetaComm(from);
  EXPECT_EQ(to.get(key, val), commSuccess);
  EXPECT_EQ(val, "true");
}

// Test that static assertions are working correctly by verifying a few key enum
// values
TEST(DataTypeConvTest, StaticAssertionsVerification) {
  using namespace meta::comms;

  // Verify a few key enum values match between NCCL and Comm types
  EXPECT_EQ(
      static_cast<int>(CommPattern::Ring), static_cast<int>(ncclPatternRing));
  EXPECT_EQ(
      static_cast<int>(CommFunc::AllReduce),
      static_cast<int>(ncclFuncAllReduce));
  EXPECT_EQ(static_cast<int>(commFloat32), static_cast<int>(ncclFloat32));
  EXPECT_EQ(static_cast<int>(commSum), static_cast<int>(ncclSum));
}
