// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * Integration test for ncclDebugLogger_t ABI compatibility.
 *
 * This test verifies that network plugins compiled against baseline NCCL
 * headers can correctly call ncclDebugLog through the ncclDebugLogger_t
 * function pointer.
 *
 * The test simulates what happens when an external plugin (like OFI/EFA)
 * receives the ncclDebugLog function pointer and attempts to call it.
 * If the signature doesn't match, this would crash at runtime.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <cstring>

#include "debug.h"
#include "nccl_common.h"

namespace {

// Track if our mock plugin successfully called the logger
std::atomic<bool> g_loggerCalled{false};
std::atomic<int> g_lastLogLevel{-1};
const char* g_lastMessage = nullptr;

// Simulates a plugin's init function that receives ncclDebugLogger_t
// This is what happens inside OFI plugin's nccl_net_ofi_init_v11()
ncclResult_t
mockPluginInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  // Plugin stores the logger for later use
  if (logFunction == nullptr) {
    return ncclInternalError;
  }

  // Plugin calls the logger - this is where ABI mismatch would crash
  // Baseline NCCL signature: (level, flags, filefunc, line, fmt, ...)
  logFunction(
      NCCL_LOG_INFO,
      NCCL_NET,
      __FILE__, // filefunc parameter (combined file+func in baseline)
      __LINE__,
      "Mock plugin initialized with commId=%lu",
      commId);

  g_loggerCalled = true;
  *ctx = reinterpret_cast<void*>(0xDEADBEEF);
  return ncclSuccess;
}

// Test fixture to reset state between tests
class DebugLoggerIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_loggerCalled = false;
    g_lastLogLevel = -1;
    g_lastMessage = nullptr;
  }
};

TEST_F(DebugLoggerIntegrationTest, PluginCanCallDebugLogger) {
  // Simulate NCCLx passing ncclDebugLog to a plugin
  // This is what happens in ncclNetInit() -> plugin->init()
  void* ctx = nullptr;
  uint64_t commId = 12345;

  // Pass ncclDebugLog as the logger function pointer
  // If signature doesn't match ncclDebugLogger_t, this won't compile
  ncclResult_t result = mockPluginInit(&ctx, commId, &ncclDebugLog);

  EXPECT_EQ(result, ncclSuccess);
  EXPECT_NE(ctx, nullptr);
  EXPECT_TRUE(g_loggerCalled.load())
      << "Plugin should have successfully called the debug logger";
}

TEST_F(DebugLoggerIntegrationTest, PluginReceivesValidFunctionPointer) {
  // Verify that taking address of ncclDebugLog produces valid pointer
  ncclDebugLogger_t logger = &ncclDebugLog;

  EXPECT_NE(logger, nullptr);

  // Verify we can call it without crashing (basic smoke test)
  // Note: actual logging output depends on NCCL_DEBUG env var
  logger(NCCL_LOG_INFO, NCCL_NET, "test.cc", 1, "Integration test message");
}

TEST_F(DebugLoggerIntegrationTest, LoggerSignatureMatchesBaselineNccl) {
  // This test documents the expected baseline NCCL signature
  // and verifies our ncclDebugLogger_t matches it

  // Baseline NCCL 2.28 signature from nccl_common.h:
  // typedef void (*ncclDebugLogger_t)(
  //     ncclDebugLogLevel level,
  //     unsigned long flags,
  //     const char *filefunc,  // Combined file+func (NOT separate)
  //     int line,
  //     const char *fmt,
  //     ...);

  using ExpectedSignature = void (*)(
      ncclDebugLogLevel,
      unsigned long,
      const char*, // filefunc
      int, // line
      const char*, // fmt
      ...);

  // This static_assert ensures compile-time verification
  static_assert(
      std::is_same_v<ncclDebugLogger_t, ExpectedSignature>,
      "ncclDebugLogger_t must have 5-parameter baseline NCCL signature");

  SUCCEED();
}

// Test that simulates the exact calling pattern from OFI plugin
TEST_F(DebugLoggerIntegrationTest, SimulateOfiPluginLogging) {
  ncclDebugLogger_t logger = &ncclDebugLog;

  // OFI plugin uses these exact patterns in nccl_net_ofi_create_plugin()
  // and throughout its codebase

  // Pattern 1: Simple info message
  logger(NCCL_LOG_INFO, NCCL_NET, "net_ofi.c", 100, "NET/OFI Initialized");

  // Pattern 2: Message with format args
  logger(
      NCCL_LOG_INFO,
      NCCL_NET,
      "net_ofi.c",
      200,
      "NET/OFI Using provider: %s",
      "efa");

  // Pattern 3: Warning message
  logger(
      NCCL_LOG_WARN,
      NCCL_ALL,
      "net_ofi.c",
      300,
      "NET/OFI Unexpected status: %d",
      -1);

  // If we get here without crashing, the ABI is compatible
  SUCCEED();
}

} // namespace
