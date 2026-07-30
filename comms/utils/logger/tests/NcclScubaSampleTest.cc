// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/NcclScubaSample.h"

#include <string>
#include <vector>

#include <folly/json/dynamic.h>
#include <folly/json/json.h>
#include <folly/portability/GTest.h>

namespace {

// setError() with a native stack records the frames in the stack_trace
// normvector column and the plain top-level message in exception_message,
// without duplicating the stack into the message.
TEST(NcclScubaSampleTest, SetErrorRecordsStackAndMessage) {
  const std::vector<std::string> stack{"frameA", "frameB"};
  NcclScubaSample sample("ERROR");
  sample.setError("boom", stack);

  const auto json = folly::parseJson(sample.toJson());

  const auto expectedStack = folly::dynamic::array("frameA", "frameB");
  EXPECT_EQ(json["normvector"]["stack_trace"], expectedStack);

  // The message column carries only the top-level message, not the stack.
  const auto& message = json["normal"]["exception_message"].asString();
  EXPECT_EQ(message, "boom");
  EXPECT_EQ(message.find("frameA"), std::string::npos);
  EXPECT_EQ(message.find("frameB"), std::string::npos);

  EXPECT_EQ(json["int"]["exception_set"].asInt(), 1);
}

// setError() with an empty stack leaves stack_trace an empty normvector and
// still records the message verbatim.
TEST(NcclScubaSampleTest, SetErrorEmptyStack) {
  NcclScubaSample sample("ERROR");
  sample.setError("lonely error", {});

  const auto json = folly::parseJson(sample.toJson());

  EXPECT_TRUE(json["normvector"]["stack_trace"].empty());
  EXPECT_EQ(json["normal"]["exception_message"].asString(), "lonely error");
  EXPECT_EQ(json["int"]["exception_set"].asInt(), 1);
}

} // namespace
