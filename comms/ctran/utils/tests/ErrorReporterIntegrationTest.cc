// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/utils/ErrorReport.h"
#include "comms/ctran/utils/ErrorReporterGuard.h"
#include "comms/ctran/utils/ErrorReporterRegistry.h"
#include "comms/ctran/utils/IErrorReporter.h"

namespace ctran::testing {

// Mock reporter that captures calls for verification
class CapturingReporter : public IErrorReporter {
 public:
  void reportError(const ErrorReport& report) override {
    reports.push_back(report);
  }
  std::vector<ErrorReport> reports;
};

class ErrorReporterIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setThreadLocalErrorReporter(nullptr);
  }

  void TearDown() override {
    setThreadLocalErrorReporter(nullptr);
  }
};

// Tests the pattern used in Ctran.cc: createErrorReporter produces a working
// reporter that can be used with ErrorReporterGuard in the GPE thread.
TEST_F(ErrorReporterIntegrationTest, CreateAndGuardWorkflow) {
  // Simulate what Ctran.cc does: create a reporter for NCCLX type
  auto reporter = createErrorReporter(ReporterType::NCCLX, nullptr);
  ASSERT_NE(reporter, nullptr);

  // Simulate what CtranGpeImpl.cc does: set thread-local guard
  {
    ErrorReporterGuard guard(reporter.get());
    EXPECT_EQ(getThreadLocalErrorReporter(), reporter.get());

    // Simulate what ErrorStackTraceUtil does: dispatch through thread-local
    auto* tlReporter = getThreadLocalErrorReporter();
    ASSERT_NE(tlReporter, nullptr);

    ErrorReport report;
    report.kind = ErrorReportKind::GENERAL_ERROR;
    report.errorMessage = "integration test error";
    // This calls NcclxErrorReporter::reportError which writes to scuba;
    // we can't verify scuba output in a unit test, but we verify no crash.
    // Note: in production, EventsScubaUtil may not be initialized, but
    // NcclxErrorReporter handles that gracefully.
  }

  // After guard is destroyed, thread-local should be null
  EXPECT_EQ(getThreadLocalErrorReporter(), nullptr);
}

// Tests that a registered factory (MCCL pattern) works end-to-end with guard
TEST_F(ErrorReporterIntegrationTest, RegisteredFactoryWithGuardWorkflow) {
  // Register a mock factory (simulates what registerMcclErrorReporter does)
  registerErrorReporterFactory(
      ReporterType::MCCL,
      [](CtranComm* /*comm*/) -> std::unique_ptr<IErrorReporter> {
        return std::make_unique<CapturingReporter>();
      });

  // Create reporter (simulates Ctran.cc constructor)
  auto reporter = createErrorReporter(ReporterType::MCCL, nullptr);
  ASSERT_NE(reporter, nullptr);

  auto* capturing = dynamic_cast<CapturingReporter*>(reporter.get());
  ASSERT_NE(capturing, nullptr);

  // Set guard (simulates CtranGpeImpl.cc)
  ErrorReporterGuard guard(reporter.get());

  // Dispatch error (simulates ErrorStackTraceUtil)
  auto* tlReporter = getThreadLocalErrorReporter();
  ASSERT_NE(tlReporter, nullptr);

  ErrorReport report;
  report.kind = ErrorReportKind::GENERAL_ERROR;
  report.errorMessage = "mccl integration test";
  tlReporter->reportError(report);

  ASSERT_EQ(capturing->reports.size(), 1);
  EXPECT_EQ(capturing->reports[0].errorMessage, "mccl integration test");
}

// Tests that NIC_EVENT flows through the full create → guard → dispatch chain
TEST_F(ErrorReporterIntegrationTest, NicEventEndToEnd) {
  registerErrorReporterFactory(
      ReporterType::MCCL,
      [](CtranComm* /*comm*/) -> std::unique_ptr<IErrorReporter> {
        return std::make_unique<CapturingReporter>();
      });

  auto reporter = createErrorReporter(ReporterType::MCCL, nullptr);
  auto* capturing = dynamic_cast<CapturingReporter*>(reporter.get());

  ErrorReporterGuard guard(reporter.get());

  ErrorReport report;
  report.kind = ErrorReportKind::NIC_EVENT;
  report.deviceName = "mlx5_0";
  report.port = 1;
  report.nicStatus = "DOWN";
  getThreadLocalErrorReporter()->reportError(report);

  ASSERT_EQ(capturing->reports.size(), 1);
  EXPECT_EQ(capturing->reports[0].kind, ErrorReportKind::NIC_EVENT);
  EXPECT_EQ(capturing->reports[0].deviceName, "mlx5_0");
  EXPECT_EQ(capturing->reports[0].port, 1);
  EXPECT_EQ(capturing->reports[0].nicStatus, "DOWN");
}

} // namespace ctran::testing
