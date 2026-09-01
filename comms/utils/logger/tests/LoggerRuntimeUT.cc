// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/LoggerRuntime.h"

#include <string>

#include <folly/logging/LogCategory.h>
#include <gtest/gtest.h>

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/DataTableWrapper.h"
#include "comms/utils/logger/LogTypes.h"
#include "comms/utils/logger/Logger.h"

namespace meta::comms::logger {
namespace {

class LoggerRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    originalDebugSubsys_ = NCCL_DEBUG_SUBSYS;
    NcclLogger::close();
  }

  void TearDown() override {
    NcclLogger::close();
    NCCL_DEBUG_SUBSYS = originalDebugSubsys_;
  }

 private:
  std::string originalDebugSubsys_;
};

TEST_F(LoggerRuntimeTest, ShutdownBeforeInitializationIsNoOp) {
  shutdownCommLoggerRuntime();

  EXPECT_EQ(SCUBA_nccl_structured_logging_ptr, nullptr);
  EXPECT_EQ(SCUBA_nccl_memory_logging_ptr, nullptr);
}

TEST_F(LoggerRuntimeTest, LegacyCloseIsSafeWhenNotInitialized) {
  NcclLogger::close();
}

TEST_F(LoggerRuntimeTest, LegacyInitPreservesSharedUtilsHandler) {
  NcclLogger::init({.contextName = "comms.test.first", .logPrefix = "FIRST"});
  const auto utilsCategory = folly::LoggerDB::get().getCategory("comms.utils");
  if (utilsCategory == nullptr) {
    ADD_FAILURE() << "NcclLogger did not register the shared utils category";
    return;
  }
  EXPECT_EQ(utilsCategory->getHandlers().size(), 1);

  NcclLogger::init({.contextName = "comms.test.second", .logPrefix = "SECOND"});
  const auto updatedUtilsCategory =
      folly::LoggerDB::get().getCategory("comms.utils");
  if (updatedUtilsCategory == nullptr) {
    ADD_FAILURE() << "NcclLogger removed the shared utils category";
    return;
  }
  EXPECT_EQ(utilsCategory, updatedUtilsCategory);
  EXPECT_EQ(updatedUtilsCategory->getHandlers().size(), 1);
}

TEST_F(LoggerRuntimeTest, OwnsSharedStateLifecycle) {
  NCCL_DEBUG_SUBSYS = "INIT";
  initCommLoggerRuntime();

  EXPECT_TRUE(isEnabledSubSystemBitwise(INIT));
  EXPECT_FALSE(isEnabledSubSystemBitwise(COLL));
  ASSERT_NE(SCUBA_nccl_structured_logging_ptr, nullptr);
  ASSERT_NE(SCUBA_nccl_memory_logging_ptr, nullptr);

  NCCL_DEBUG_SUBSYS = "COLL";
  initCommLoggerRuntime();
  EXPECT_TRUE(isEnabledSubSystemBitwise(INIT));
  EXPECT_FALSE(isEnabledSubSystemBitwise(COLL));

  shutdownCommLoggerRuntime();
  initCommLoggerRuntime();
  EXPECT_FALSE(isEnabledSubSystemBitwise(INIT));
  EXPECT_TRUE(isEnabledSubSystemBitwise(COLL));
  ASSERT_NE(SCUBA_nccl_structured_logging_ptr, nullptr);
  ASSERT_NE(SCUBA_nccl_memory_logging_ptr, nullptr);
}

} // namespace
} // namespace meta::comms::logger
