// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/profiler/DefaultGpeProfilerReporter.h"

#include <folly/portability/GTest.h>

#include <memory>
#include <string>
#include <string_view>

#include "comms/ctran/profiler/GpeProfilerReport.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/DataTableWrapper.h"

namespace ctran {

namespace {

constexpr uint64_t kCommId = 11111;
constexpr uint64_t kCommHash = 22222;
constexpr int kRank = 3;
constexpr int kNRanks = 8;
const char* kCommDesc = "test_pg:0";

CommLogData makeCommLogData() {
  return CommLogData{kCommId, kCommHash, kCommDesc, kRank, kNRanks};
}

GpeProfilerReport makeReport(
    GpeTracePoint p = GpeTracePoint::HOST_ALGO,
    bool aborted = false,
    std::string_view message = std::string_view{}) {
  GpeProfilerReport r;
  r.commHash = kCommHash;
  r.rank = kRank;
  r.opCount = 42;
  r.opType = 7;
  r.tracePoint = p;
  r.iterUs = 1234;
  r.durationUs = 567;
  r.aborted = aborted;
  r.message = message;
  return r;
}

class DefaultGpeProfilerReporterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Wire SCUBA_*_ptr globals so getTablePtrFromEvent() does not throw on
    // dispatch. Lazy lookup of the backing scuba table may still warn
    // (test env doesn't register dataset) but addSample is a no-op when
    // table_ stays null — report() must return cleanly either way.
    DataTableWrapper::init();
  }
  void TearDown() override {
    DataTableWrapper::shutdown();
  }
};

// Happy-path report with all fields set must dispatch via the Scuba layer
// without throwing or crashing.
TEST_F(DefaultGpeProfilerReporterTest, ReportRunsCleanlyForSampledRow) {
  DefaultGpeProfilerReporter reporter;
  const auto md = makeCommLogData();
  auto report = makeReport();
  report.logMetaData = &md;

  EXPECT_NO_THROW(reporter.report(report));
}

// Aborted-row dispatch: aborted=true + non-empty message. Verify the
// reporter does not throw on the marker path.
TEST_F(DefaultGpeProfilerReporterTest, ReportAbortedRowRunsCleanly) {
  DefaultGpeProfilerReporter reporter;
  const auto md = makeCommLogData();
  auto report = makeReport(
      GpeTracePoint::ALGO_ABORTED,
      /*aborted=*/true,
      std::string_view{"timeout"});
  report.logMetaData = &md;

  EXPECT_NO_THROW(reporter.report(report));
}

// nullptr logMetaData: the reporter forwards logMetaData straight into
// CtranProfilerGpeEvent, which uses CommEvent's nullptr-tolerant ctor
// (defaults rank/commHash/commDesc). Must not crash.
TEST_F(DefaultGpeProfilerReporterTest, ReportTolerantOfNullLogMetaData) {
  DefaultGpeProfilerReporter reporter;
  auto report = makeReport();
  report.logMetaData = nullptr;

  EXPECT_NO_THROW(reporter.report(report));
}

// Each GpeTracePoint enum value must dispatch without throwing — the
// reporter calls tracePointName(p) and the switch must cover every case
// (a missing case would either fall through to "<invalid>" or trigger UB
// depending on compiler).
TEST_F(DefaultGpeProfilerReporterTest, ReportCoversEveryTracePoint) {
  DefaultGpeProfilerReporter reporter;
  const auto md = makeCommLogData();
  for (int i = 0; i < static_cast<int>(GpeTracePoint::NUM_TRACE_POINTS); ++i) {
    auto report = makeReport(static_cast<GpeTracePoint>(i));
    report.logMetaData = &md;
    EXPECT_NO_THROW(reporter.report(report))
        << "tracePoint enum value " << i << " threw";
  }
}

} // namespace
} // namespace ctran
