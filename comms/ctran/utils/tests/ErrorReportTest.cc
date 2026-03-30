// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/utils/ErrorReport.h"
#include "comms/ctran/utils/IErrorReporter.h"
#include "comms/ctran/utils/ReporterType.h"

namespace ctran::testing {

// Verify ErrorReport struct has correct defaults
TEST(ErrorReportTest, DefaultConstruction) {
  ErrorReport report;
  EXPECT_EQ(report.kind, ErrorReportKind::GENERAL_ERROR);
  EXPECT_TRUE(report.errorMessage.empty());
  EXPECT_TRUE(report.stackTrace.empty());
  EXPECT_TRUE(report.deviceName.empty());
  EXPECT_EQ(report.port, 0);
  EXPECT_TRUE(report.nicStatus.empty());
}

TEST(ErrorReportTest, GeneralErrorFields) {
  ErrorReport report;
  report.kind = ErrorReportKind::GENERAL_ERROR;
  report.errorMessage = "test error";
  report.stackTrace = {"frame1", "frame2"};

  EXPECT_EQ(report.kind, ErrorReportKind::GENERAL_ERROR);
  EXPECT_EQ(report.errorMessage, "test error");
  ASSERT_EQ(report.stackTrace.size(), 2);
  EXPECT_EQ(report.stackTrace[0], "frame1");
  EXPECT_EQ(report.stackTrace[1], "frame2");
}

TEST(ErrorReportTest, NicEventFields) {
  ErrorReport report;
  report.kind = ErrorReportKind::NIC_EVENT;
  report.deviceName = "mlx5_0";
  report.port = 1;
  report.nicStatus = "DOWN";

  EXPECT_EQ(report.kind, ErrorReportKind::NIC_EVENT);
  EXPECT_EQ(report.deviceName, "mlx5_0");
  EXPECT_EQ(report.port, 1);
  EXPECT_EQ(report.nicStatus, "DOWN");
}

// Verify ReporterType enum values exist
TEST(ReporterTypeTest, EnumValues) {
  auto ncclx = ReporterType::NCCLX;
  auto mccl = ReporterType::MCCL;
  EXPECT_NE(ncclx, mccl);
}

// Verify IErrorReporter can be implemented and used polymorphically
class TestReporter : public IErrorReporter {
 public:
  void reportError(const ErrorReport& report) override {
    lastReport = report;
    callCount++;
  }

  ErrorReport lastReport;
  int callCount{0};
};

TEST(IErrorReporterTest, PolymorphicUsage) {
  auto reporter = std::make_unique<TestReporter>();
  IErrorReporter* base = reporter.get();

  ErrorReport report;
  report.kind = ErrorReportKind::GENERAL_ERROR;
  report.errorMessage = "polymorphic test";
  base->reportError(report);

  EXPECT_EQ(reporter->callCount, 1);
  EXPECT_EQ(reporter->lastReport.errorMessage, "polymorphic test");
}

TEST(IErrorReporterTest, VirtualDestructorWorks) {
  // Ensure deleting through base pointer doesn't leak
  std::unique_ptr<IErrorReporter> reporter = std::make_unique<TestReporter>();
  reporter.reset(); // Should not crash
}

} // namespace ctran::testing
