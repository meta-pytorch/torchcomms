// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <gtest/gtest.h>

#include "comms/utils/Conversion.h"
#include "comms/utils/commSpecs.h"

// Test the commTypeSize function for all data types
TEST(CommSpecsTest, CommTypeSize) {
  EXPECT_EQ(commTypeSize(commInt8), 1);
  EXPECT_EQ(commTypeSize(commChar), 1);
  EXPECT_EQ(commTypeSize(commUint8), 1);
  EXPECT_EQ(commTypeSize(commFloat8e4m3), 1);
  EXPECT_EQ(commTypeSize(commFloat8e5m2), 1);

  EXPECT_EQ(commTypeSize(commFloat16), 2);
  EXPECT_EQ(commTypeSize(commHalf), 2);
  EXPECT_EQ(commTypeSize(commBfloat16), 2);

  EXPECT_EQ(commTypeSize(commInt32), 4);
  EXPECT_EQ(commTypeSize(commInt), 4);
  EXPECT_EQ(commTypeSize(commUint32), 4);
  EXPECT_EQ(commTypeSize(commFloat32), 4);
  EXPECT_EQ(commTypeSize(commFloat), 4);

  EXPECT_EQ(commTypeSize(commInt64), 8);
  EXPECT_EQ(commTypeSize(commUint64), 8);
  EXPECT_EQ(commTypeSize(commFloat64), 8);
  EXPECT_EQ(commTypeSize(commDouble), 8);

  // Test invalid type
  EXPECT_EQ(commTypeSize(static_cast<commDataType_t>(100)), -1);
}

// Test the commResultToString function
TEST(CommSpecsTest, CommResultToString) {
  EXPECT_STREQ(commResultToString(commSuccess), "commSuccess");
  EXPECT_STREQ(
      commResultToString(commUnhandledCudaError), "commUnhandledCudaError");
  EXPECT_STREQ(commResultToString(commSystemError), "commSystemError");
  EXPECT_STREQ(commResultToString(commInternalError), "commInternalError");
  EXPECT_STREQ(commResultToString(commInvalidArgument), "commInvalidArgument");
  EXPECT_STREQ(commResultToString(commInvalidUsage), "commInvalidUsage");
  EXPECT_STREQ(commResultToString(commRemoteError), "commRemoteError");
  EXPECT_STREQ(commResultToString(commInProgress), "commInProgress");

  // Test invalid result code
  EXPECT_STREQ(commResultToString(static_cast<commResult_t>(100)), "Unknown");
}

// Test the commOpToString function
TEST(CommSpecsTest, CommOpToString) {
  EXPECT_STREQ(commOpToString(commSum), "commSum");
  EXPECT_STREQ(commOpToString(commProd), "commProd");
  EXPECT_STREQ(commOpToString(commMax), "commMax");
  EXPECT_STREQ(commOpToString(commMin), "commMin");
  EXPECT_STREQ(commOpToString(commAvg), "commAvg");

  // Test invalid op code
  EXPECT_STREQ(commOpToString(static_cast<commRedOp_t>(100)), "Unknown");
}

// Test the CommLogData struct and its hash function
TEST(CommSpecsTest, CommLogData) {
  CommLogData data1{1, 2, "test", 3, 4};
  CommLogData data2{1, 2, "test", 3, 4};
  CommLogData data3{5, 6, "other", 7, 8};

  // Test equality operator
  EXPECT_TRUE(data1 == data2);
  EXPECT_FALSE(data1 == data3);

  // Test hash function
  EXPECT_EQ(data1.hash(), data2.hash());
  EXPECT_NE(data1.hash(), data3.hash());
}

// Test the CommsError class in the meta::comms namespace
TEST(CommSpecsTest, CommsError) {
  using namespace meta::comms;

  CommsError error1("error message", commInvalidArgument);
  CommsError error2("error message", commInvalidArgument);
  CommsError error3("different message", commInvalidArgument);
  CommsError error4("error message", commSystemError);

  // Test equality operator
  EXPECT_TRUE(error1 == error2);
  EXPECT_FALSE(error1 == error3);
  EXPECT_FALSE(error1 == error4);

  // Test name method
  EXPECT_STREQ(CommsError::name(), "CommsError");

  // Test message and errorCode fields
  EXPECT_EQ(error1.message, "error message");
  EXPECT_EQ(error1.errorCode, commInvalidArgument);
}

// Test the CommsMaybe template in the meta::comms namespace
TEST(CommSpecsTest, CommsMaybe) {
  using namespace meta::comms;

  // Test successful case
  CommsMaybe<int> success = 42;
  EXPECT_TRUE(success.hasValue());
  EXPECT_EQ(success.value(), 42);

  // Test error case
  CommsError error("test error", commInvalidArgument);
  CommsMaybe<int> failure = folly::makeUnexpected(error);
  EXPECT_FALSE(failure.hasValue());
  EXPECT_EQ(failure.error().message, "test error");
  EXPECT_EQ(failure.error().errorCode, commInvalidArgument);

  // Test CommsMaybeVoid
  CommsMaybeVoid voidSuccess = folly::unit;
  EXPECT_TRUE(voidSuccess.hasValue());

  CommsMaybeVoid voidFailure = folly::makeUnexpected(error);
  EXPECT_FALSE(voidFailure.hasValue());
  EXPECT_EQ(voidFailure.error().message, "test error");
}

// Test the string conversion functions for CommPattern
TEST(ConversionTest, CommPatternStringConversion) {
  using namespace meta::comms;

  // Test commPatternToString
  EXPECT_EQ(commPatternToString(CommPattern::Ring), "Ring");
  EXPECT_EQ(commPatternToString(CommPattern::RingTwice), "RingTwice");
  EXPECT_EQ(commPatternToString(CommPattern::PipelineFrom), "PipelineFrom");
  EXPECT_EQ(commPatternToString(CommPattern::PipelineTo), "PipelineTo");
  EXPECT_EQ(commPatternToString(CommPattern::TreeUp), "TreeUp");
  EXPECT_EQ(commPatternToString(CommPattern::TreeDown), "TreeDown");
  EXPECT_EQ(commPatternToString(CommPattern::TreeUpDown), "TreeUpDown");
  EXPECT_EQ(commPatternToString(CommPattern::CollnetChain), "CollnetChain");
  EXPECT_EQ(commPatternToString(CommPattern::CollnetDirect), "CollnetDirect");
  EXPECT_EQ(commPatternToString(CommPattern::Nvls), "Nvls");
  EXPECT_EQ(commPatternToString(CommPattern::NvlsTree), "NvlsTree");
  EXPECT_EQ(commPatternToString(CommPattern::PatUp), "PatUp");
  EXPECT_EQ(commPatternToString(CommPattern::PatDown), "PatDown");
  EXPECT_EQ(commPatternToString(CommPattern::Send), "Send");
  EXPECT_EQ(commPatternToString(CommPattern::Recv), "Recv");
  EXPECT_EQ(commPatternToString(static_cast<CommPattern>(100)), "Unknown");

  // Test stringToCommPattern
  EXPECT_EQ(stringToCommPattern("Ring"), CommPattern::Ring);
  EXPECT_EQ(stringToCommPattern("RingTwice"), CommPattern::RingTwice);
  EXPECT_EQ(stringToCommPattern("PipelineFrom"), CommPattern::PipelineFrom);
  EXPECT_EQ(stringToCommPattern("PipelineTo"), CommPattern::PipelineTo);
  EXPECT_EQ(stringToCommPattern("TreeUp"), CommPattern::TreeUp);
  EXPECT_EQ(stringToCommPattern("TreeDown"), CommPattern::TreeDown);
  EXPECT_EQ(stringToCommPattern("TreeUpDown"), CommPattern::TreeUpDown);
  EXPECT_EQ(stringToCommPattern("CollnetChain"), CommPattern::CollnetChain);
  EXPECT_EQ(stringToCommPattern("CollnetDirect"), CommPattern::CollnetDirect);
  EXPECT_EQ(stringToCommPattern("Nvls"), CommPattern::Nvls);
  EXPECT_EQ(stringToCommPattern("NvlsTree"), CommPattern::NvlsTree);
  EXPECT_EQ(stringToCommPattern("PatUp"), CommPattern::PatUp);
  EXPECT_EQ(stringToCommPattern("PatDown"), CommPattern::PatDown);
  EXPECT_EQ(stringToCommPattern("Send"), CommPattern::Send);
  EXPECT_EQ(stringToCommPattern("Recv"), CommPattern::Recv);
  EXPECT_EQ(stringToCommPattern("Unknown"), CommPattern::NumPatterns);
}

// Test the string conversion functions for CommFunc
TEST(ConversionTest, CommFuncStringConversion) {
  using namespace meta::comms;

  // Test commFuncToString
  EXPECT_EQ(commFuncToString(CommFunc::Broadcast), "Broadcast");
  EXPECT_EQ(commFuncToString(CommFunc::Reduce), "Reduce");
  EXPECT_EQ(commFuncToString(CommFunc::AllGather), "AllGather");
  EXPECT_EQ(commFuncToString(CommFunc::ReduceScatter), "ReduceScatter");
  EXPECT_EQ(commFuncToString(CommFunc::AllReduce), "AllReduce");
  EXPECT_EQ(commFuncToString(CommFunc::SendRecv), "SendRecv");
  EXPECT_EQ(commFuncToString(CommFunc::Send), "Send");
  EXPECT_EQ(commFuncToString(CommFunc::Recv), "Recv");
  EXPECT_EQ(commFuncToString(static_cast<CommFunc>(100)), "Unknown");

  // Test stringToCommFunc
  EXPECT_EQ(stringToCommFunc("Broadcast"), CommFunc::Broadcast);
  EXPECT_EQ(stringToCommFunc("Reduce"), CommFunc::Reduce);
  EXPECT_EQ(stringToCommFunc("AllGather"), CommFunc::AllGather);
  EXPECT_EQ(stringToCommFunc("ReduceScatter"), CommFunc::ReduceScatter);
  EXPECT_EQ(stringToCommFunc("AllReduce"), CommFunc::AllReduce);
  EXPECT_EQ(stringToCommFunc("SendRecv"), CommFunc::SendRecv);
  EXPECT_EQ(stringToCommFunc("Send"), CommFunc::Send);
  EXPECT_EQ(stringToCommFunc("Recv"), CommFunc::Recv);
  EXPECT_EQ(stringToCommFunc("Unknown"), CommFunc::NumFuncs);
}

// Test the string conversion functions for commRedOp_t
TEST(ConversionTest, CommRedOpStringConversion) {
  using namespace meta::comms;

  // Test commRedOpToString
  EXPECT_EQ(commRedOpToString(commSum), "Sum");
  EXPECT_EQ(commRedOpToString(commProd), "Prod");
  EXPECT_EQ(commRedOpToString(commMax), "Max");
  EXPECT_EQ(commRedOpToString(commMin), "Min");
  EXPECT_EQ(commRedOpToString(commAvg), "Avg");
  EXPECT_EQ(commRedOpToString(static_cast<commRedOp_t>(100)), "Unknown");

  // Test stringToCommRedOp
  EXPECT_EQ(stringToCommRedOp("Sum"), commSum);
  EXPECT_EQ(stringToCommRedOp("Prod"), commProd);
  EXPECT_EQ(stringToCommRedOp("Max"), commMax);
  EXPECT_EQ(stringToCommRedOp("Min"), commMin);
  EXPECT_EQ(stringToCommRedOp("Avg"), commAvg);
  EXPECT_EQ(stringToCommRedOp("Unknown"), commNumOps);
}

// Test the string conversion functions for CommAlgo
TEST(ConversionTest, CommAlgoStringConversion) {
  using namespace meta::comms;

  // Test commAlgoToString
  EXPECT_EQ(commAlgoToString(CommAlgo::Tree), "Tree");
  EXPECT_EQ(commAlgoToString(CommAlgo::Ring), "Ring");
  EXPECT_EQ(commAlgoToString(CommAlgo::CollNetDirect), "CollNetDirect");
  EXPECT_EQ(commAlgoToString(CommAlgo::CollNetChain), "CollNetChain");
  EXPECT_EQ(commAlgoToString(CommAlgo::NVLS), "NVLS");
  EXPECT_EQ(commAlgoToString(CommAlgo::NVLSTree), "NVLSTree");
  EXPECT_EQ(commAlgoToString(CommAlgo::PAT), "PAT");
  EXPECT_EQ(commAlgoToString(static_cast<CommAlgo>(100)), "Unknown");

  // Test stringToCommAlgo
  EXPECT_EQ(stringToCommAlgo("Tree"), CommAlgo::Tree);
  EXPECT_EQ(stringToCommAlgo("Ring"), CommAlgo::Ring);
  EXPECT_EQ(stringToCommAlgo("CollNetDirect"), CommAlgo::CollNetDirect);
  EXPECT_EQ(stringToCommAlgo("CollNetChain"), CommAlgo::CollNetChain);
  EXPECT_EQ(stringToCommAlgo("NVLS"), CommAlgo::NVLS);
  EXPECT_EQ(stringToCommAlgo("NVLSTree"), CommAlgo::NVLSTree);
  EXPECT_EQ(stringToCommAlgo("PAT"), CommAlgo::PAT);
  EXPECT_EQ(stringToCommAlgo("Unknown"), CommAlgo::NumAlgorithms);
}

// Test the string conversion functions for CommProtocol
TEST(ConversionTest, CommProtocolStringConversion) {
  using namespace meta::comms;

  // Test commProtocolToString
  EXPECT_EQ(commProtocolToString(CommProtocol::LL), "LL");
  EXPECT_EQ(commProtocolToString(CommProtocol::LL128), "LL128");
  EXPECT_EQ(commProtocolToString(CommProtocol::Simple), "Simple");
  EXPECT_EQ(commProtocolToString(static_cast<CommProtocol>(100)), "Unknown");

  // Test stringToCommProtocol
  EXPECT_EQ(stringToCommProtocol("LL"), CommProtocol::LL);
  EXPECT_EQ(stringToCommProtocol("LL128"), CommProtocol::LL128);
  EXPECT_EQ(stringToCommProtocol("Simple"), CommProtocol::Simple);
  EXPECT_EQ(stringToCommProtocol("Unknown"), CommProtocol::NumProtocols);
}

// Test the string conversion functions for commDataType_t
TEST(ConversionTest, CommDataTypeStringConversion) {
  using namespace meta::comms;

  // Test getCommsDatatypeStr
  EXPECT_EQ(getCommsDatatypeStr(commInt8), "commInt8");
  EXPECT_EQ(getCommsDatatypeStr(commChar), "commInt8");
  EXPECT_EQ(getCommsDatatypeStr(commUint8), "commUint8");
  EXPECT_EQ(getCommsDatatypeStr(commInt32), "commInt32");
  EXPECT_EQ(getCommsDatatypeStr(commInt), "commInt32");
  EXPECT_EQ(getCommsDatatypeStr(commUint32), "commUint32");
  EXPECT_EQ(getCommsDatatypeStr(commInt64), "commInt64");
  EXPECT_EQ(getCommsDatatypeStr(commUint64), "commUint64");
  EXPECT_EQ(getCommsDatatypeStr(commFloat16), "commFloat16");
  EXPECT_EQ(getCommsDatatypeStr(commHalf), "commFloat16");
  EXPECT_EQ(getCommsDatatypeStr(commFloat32), "commFloat32");
  EXPECT_EQ(getCommsDatatypeStr(commFloat), "commFloat32");
  EXPECT_EQ(getCommsDatatypeStr(commFloat64), "commFloat64");
  EXPECT_EQ(getCommsDatatypeStr(commDouble), "commFloat64");
  EXPECT_EQ(getCommsDatatypeStr(commBfloat16), "commBfloat16");
  EXPECT_EQ(getCommsDatatypeStr(commFloat8e4m3), "commFloat8e4m3");
  EXPECT_EQ(getCommsDatatypeStr(commFloat8e5m2), "commFloat8e5m2");
  EXPECT_EQ(
      getCommsDatatypeStr(static_cast<commDataType_t>(100)), "Unknown type");

  // Test stringToCommsDatatype
  EXPECT_EQ(stringToCommsDatatype("commInt8"), commInt8);
  EXPECT_EQ(stringToCommsDatatype("commUint8"), commUint8);
  EXPECT_EQ(stringToCommsDatatype("commInt32"), commInt32);
  EXPECT_EQ(stringToCommsDatatype("commUint32"), commUint32);
  EXPECT_EQ(stringToCommsDatatype("commInt64"), commInt64);
  EXPECT_EQ(stringToCommsDatatype("commUint64"), commUint64);
  EXPECT_EQ(stringToCommsDatatype("commFloat16"), commFloat16);
  EXPECT_EQ(stringToCommsDatatype("commFloat32"), commFloat32);
  EXPECT_EQ(stringToCommsDatatype("commFloat64"), commFloat64);
  EXPECT_EQ(stringToCommsDatatype("commBfloat16"), commBfloat16);
  EXPECT_EQ(stringToCommsDatatype("commFloat8e4m3"), commFloat8e4m3);
  EXPECT_EQ(stringToCommsDatatype("commFloat8e5m2"), commFloat8e5m2);
  EXPECT_EQ(stringToCommsDatatype("Unknown"), commNumTypes);
}

// Test the commCodeToName and commCodeToString functions
TEST(ConversionTest, CommCodeStringConversion) {
  using namespace meta::comms;

  // Test commCodeToName
  EXPECT_STREQ(commCodeToName(commSuccess), "commSuccess");
  EXPECT_STREQ(
      commCodeToName(commUnhandledCudaError), "commUnhandledCudaError");
  EXPECT_STREQ(commCodeToName(commSystemError), "commSystemError");
  EXPECT_STREQ(commCodeToName(commInternalError), "commInternalError");
  EXPECT_STREQ(commCodeToName(commInvalidArgument), "commInvalidArgument");
  EXPECT_STREQ(commCodeToName(commInvalidUsage), "commInvalidUsage");
  EXPECT_STREQ(commCodeToName(commRemoteError), "commRemoteError");
  EXPECT_STREQ(commCodeToName(commInProgress), "commInProgress");
  EXPECT_STREQ(commCodeToName(commNumResults), "commNumResults");
  EXPECT_STREQ(
      commCodeToName(static_cast<commResult_t>(100)), "unknown result code");

  // Test commCodeToString
  EXPECT_STREQ(commCodeToString(commSuccess), "no error");
  EXPECT_STREQ(
      commCodeToString(commUnhandledCudaError),
      "unhandled cuda error (run with NCCL_DEBUG=INFO for details)");
  EXPECT_STREQ(
      commCodeToString(commSystemError),
      "unhandled system error (run with NCCL_DEBUG=INFO for details)");
  EXPECT_STREQ(
      commCodeToString(commInternalError),
      "internal error - please report this issue to the NCCL developers");
  EXPECT_STREQ(
      commCodeToString(commInvalidArgument),
      "invalid argument (run with NCCL_DEBUG=WARN for details)");
  EXPECT_STREQ(
      commCodeToString(commInvalidUsage),
      "invalid usage (run with NCCL_DEBUG=WARN for details)");
  EXPECT_STREQ(
      commCodeToString(commRemoteError),
      "remote process exited or there was a network error");
  EXPECT_STREQ(commCodeToString(commInProgress), "NCCL operation in progress");
  EXPECT_STREQ(commCodeToString(commNumResults), "numericall error");
  EXPECT_STREQ(
      commCodeToString(static_cast<commResult_t>(100)), "unknown result code");
}
