// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/EventMgr.h"

#include <folly/json/dynamic.h>
#include <folly/json/json.h>
#include <folly/portability/GTest.h>

#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/DataTableWrapper.h"
#include "comms/utils/logger/ScubaLogger.h"
#include "comms/utils/trainer/TrainerContext.h"

namespace {

constexpr uint64_t kCommId = 11111;
constexpr uint64_t kCommHash = 22222;
constexpr int kRank = 3;
constexpr int kNRanks = 8;
constexpr const char* kCommDesc = "test_pg:0";

constexpr uint64_t kOpCount = 42;
constexpr int kOpType = 7;
const std::string kTracePoint = "host_algo";
constexpr uint64_t kIterUs = 1234;
constexpr uint64_t kDurationUs = 567;
constexpr bool kAborted = true;
const std::string kMessage = "timeout";
constexpr int64_t kIteration = 99;

CommLogData makeCommLogData() {
  return CommLogData{kCommId, kCommHash, kCommDesc, kRank, kNRanks};
}

std::unique_ptr<CtranProfilerGpeEvent> makeGpeEvent(
    const CommLogData* logMetaData) {
  return std::make_unique<CtranProfilerGpeEvent>(
      logMetaData,
      /*stage=*/"ctranProfilerGpeV1",
      kOpCount,
      kOpType,
      kTracePoint,
      kIterUs,
      kDurationUs,
      kAborted,
      kMessage);
}

} // namespace

// Construct a CtranProfilerGpeEvent and verify every GPE-specific column
// added by its toSample() is present and round-trips through JSON.
TEST(CtranProfilerGpeEventTest, ToSamplePopulatesAllNewColumns) {
  ncclxSetIteration(kIteration);
  const auto md = makeCommLogData();
  auto event = makeGpeEvent(&md);

  auto sample = event->toSample();
  const auto json = folly::parseJson(sample.toJson());

  EXPECT_EQ(json["normal"]["type"].asString(), "CtranProfilerGpeEvent");
  EXPECT_EQ(json["int"]["opCount"].asInt(), static_cast<int64_t>(kOpCount));
  EXPECT_EQ(json["int"]["opType"].asInt(), kOpType);
  EXPECT_EQ(json["normal"]["tracePoint"].asString(), kTracePoint);
  EXPECT_EQ(json["int"]["iterUs"].asInt(), static_cast<int64_t>(kIterUs));
  EXPECT_EQ(
      json["int"]["durationUs"].asInt(), static_cast<int64_t>(kDurationUs));
  EXPECT_EQ(json["int"]["aborted"].asInt(), kAborted ? 1 : 0);
  EXPECT_EQ(json["normal"]["message"].asString(), kMessage);
  EXPECT_EQ(json["int"]["iteration"].asInt(), kIteration);
}

// CommEvent base-class columns (commId, commHash, commDesc, rank, nranks)
// must be populated from the injected CommLogData. The GPE subclass must
// NOT re-add them with conflicting values.
TEST(CtranProfilerGpeEventTest, ToSampleBaseColumnsFromCommLogData) {
  const auto md = makeCommLogData();
  auto event = makeGpeEvent(&md);

  auto sample = event->toSample();
  const auto json = folly::parseJson(sample.toJson());

  EXPECT_EQ(json["int"]["commId"].asInt(), static_cast<int64_t>(kCommId));
  EXPECT_EQ(json["int"]["commHash"].asInt(), static_cast<int64_t>(kCommHash));
  EXPECT_EQ(json["normal"]["commDesc"].asString(), kCommDesc);
  EXPECT_EQ(json["int"]["rank"].asInt(), kRank);
  EXPECT_EQ(json["int"]["nranks"].asInt(), kNRanks);
}

// Aborted=false / empty message path: aborted column reads 0, message is
// the empty string.
TEST(CtranProfilerGpeEventTest, ToSampleNonAbortedRow) {
  const auto md = makeCommLogData();
  auto event = std::make_unique<CtranProfilerGpeEvent>(
      &md,
      /*stage=*/"ctranProfilerGpeV1",
      kOpCount,
      kOpType,
      std::string("cmd_dequeue"),
      kIterUs,
      kDurationUs,
      /*aborted=*/false,
      /*message=*/"");

  auto sample = event->toSample();
  const auto json = folly::parseJson(sample.toJson());

  EXPECT_EQ(json["int"]["aborted"].asInt(), 0);
  EXPECT_EQ(json["normal"]["message"].asString(), "");
  EXPECT_EQ(json["normal"]["tracePoint"].asString(), "cmd_dequeue");
}

// getEventType() must return the GPE variant so ScubaLogger's dispatch
// routes the event to nccl_profiler_gpe (and not to the algo or another
// sibling table).
TEST(CtranProfilerGpeEventTest, GetEventTypeIsGpeVariant) {
  const auto md = makeCommLogData();
  auto event = makeGpeEvent(&md);
  EXPECT_EQ(event->getEventType(), LoggerEventType::CtranProfilerGpeEventType);
}

// ScubaLogger's getTablePtrFromEvent() must route the GPE event type to
// the wrapper added in this diff (SCUBA_nccl_profiler_gpe_ptr). Without
// the correct routing, GPE rows would land in algo (or worse, throw on
// the default branch).
TEST(CtranProfilerGpeEventTest, ScubaDispatchTargetsGpeTablePtr) {
  DataTableWrapper::init();
  ASSERT_NE(SCUBA_nccl_profiler_gpe_ptr, nullptr);
  EXPECT_EQ(
      getTablePtrFromEvent(LoggerEventType::CtranProfilerGpeEventType),
      SCUBA_nccl_profiler_gpe_ptr.get());
  // Algo path must not return the GPE pointer (sanity check).
  EXPECT_NE(
      getTablePtrFromEvent(LoggerEventType::CtranProfilerAlgoEventType),
      SCUBA_nccl_profiler_gpe_ptr.get());
  // shutdown must be safe with no addSample calls (lazy table_ stays
  // null; the SHUTDOWN_scuba_table macro guards on table_).
  DataTableWrapper::shutdown();
}
