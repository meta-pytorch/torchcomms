// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/utils/ErrorReport.h"
#include "comms/ctran/utils/ErrorReporterGuard.h"
#include "comms/ctran/utils/ErrorReporterRegistry.h"
#include "comms/ctran/utils/IErrorReporter.h"
#include "comms/ctran/utils/NcclxErrorReporter.h"

namespace ctran::testing {

// Mock reporter that captures calls for verification
class MockErrorReporter : public IErrorReporter {
 public:
  void reportError(const ErrorReport& report) override {
    reports.push_back(report);
  }

  std::vector<ErrorReport> reports;
};

class ErrorReporterGuardTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure thread-local is clean at start
    setThreadLocalErrorReporter(nullptr);
  }

  void TearDown() override {
    setThreadLocalErrorReporter(nullptr);
  }
};

TEST_F(ErrorReporterGuardTest, NoReporterByDefault) {
  EXPECT_EQ(getThreadLocalErrorReporter(), nullptr);
}

TEST_F(ErrorReporterGuardTest, GuardSetsAndRestoresReporter) {
  MockErrorReporter reporter;
  EXPECT_EQ(getThreadLocalErrorReporter(), nullptr);

  {
    ErrorReporterGuard guard(&reporter);
    EXPECT_EQ(getThreadLocalErrorReporter(), &reporter);
  }

  EXPECT_EQ(getThreadLocalErrorReporter(), nullptr);
}

TEST_F(ErrorReporterGuardTest, NestedGuardsRestoreCorrectly) {
  MockErrorReporter outer;
  MockErrorReporter inner;

  {
    ErrorReporterGuard outerGuard(&outer);
    EXPECT_EQ(getThreadLocalErrorReporter(), &outer);

    {
      ErrorReporterGuard innerGuard(&inner);
      EXPECT_EQ(getThreadLocalErrorReporter(), &inner);
    }

    EXPECT_EQ(getThreadLocalErrorReporter(), &outer);
  }

  EXPECT_EQ(getThreadLocalErrorReporter(), nullptr);
}

TEST_F(ErrorReporterGuardTest, DispatchThroughThreadLocal) {
  MockErrorReporter reporter;
  ErrorReporterGuard guard(&reporter);

  ErrorReport report;
  report.kind = ErrorReportKind::GENERAL_ERROR;
  report.errorMessage = "test error";

  auto* threadReporter = getThreadLocalErrorReporter();
  ASSERT_NE(threadReporter, nullptr);
  threadReporter->reportError(report);

  ASSERT_EQ(reporter.reports.size(), 1);
  EXPECT_EQ(reporter.reports[0].errorMessage, "test error");
  EXPECT_EQ(reporter.reports[0].kind, ErrorReportKind::GENERAL_ERROR);
}

TEST_F(ErrorReporterGuardTest, NicEventDispatch) {
  MockErrorReporter reporter;
  ErrorReporterGuard guard(&reporter);

  ErrorReport report;
  report.kind = ErrorReportKind::NIC_EVENT;
  report.deviceName = "mlx5_0";
  report.port = 1;
  report.nicStatus = "DOWN";

  getThreadLocalErrorReporter()->reportError(report);

  ASSERT_EQ(reporter.reports.size(), 1);
  EXPECT_EQ(reporter.reports[0].kind, ErrorReportKind::NIC_EVENT);
  EXPECT_EQ(reporter.reports[0].deviceName, "mlx5_0");
  EXPECT_EQ(reporter.reports[0].port, 1);
  EXPECT_EQ(reporter.reports[0].nicStatus, "DOWN");
}

class ErrorReporterRegistryTest : public ::testing::Test {};

TEST_F(ErrorReporterRegistryTest, DefaultFallsBackToNcclx) {
  // Requesting an unregistered type should return NcclxErrorReporter
  auto reporter = createErrorReporter(ReporterType::NCCLX, nullptr);
  ASSERT_NE(reporter, nullptr);
  // Verify it's a NcclxErrorReporter via dynamic_cast
  EXPECT_NE(dynamic_cast<NcclxErrorReporter*>(reporter.get()), nullptr);
}

TEST_F(ErrorReporterRegistryTest, RegisterAndCreate) {
  bool factoryCalled = false;

  registerErrorReporterFactory(
      ReporterType::MCCL,
      [&](CtranComm* /*comm*/) -> std::unique_ptr<IErrorReporter> {
        factoryCalled = true;
        return std::make_unique<MockErrorReporter>();
      });

  auto reporter = createErrorReporter(ReporterType::MCCL, nullptr);
  EXPECT_TRUE(factoryCalled);
  ASSERT_NE(reporter, nullptr);
  EXPECT_NE(dynamic_cast<MockErrorReporter*>(reporter.get()), nullptr);
}

} // namespace ctran::testing
