// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "comms/ctran/colltrace/MapperTrace.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/testinfra/BenchUtils.h"
#include "comms/utils/colltrace/CollRecord.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ncclx::colltrace;

// Lightweight ICollRecord for benchmarks (no gmock overhead)
struct NopCollRecord : public meta::comms::colltrace::ICollRecord {
  uint64_t getCollId() const noexcept override {
    return 0;
  }
  folly::dynamic toDynamic() const noexcept override {
    return folly::dynamic::object();
  }
};

static void BM_RecordPutStart(
    uint32_t iters,
    int eventsPerColl,
    folly::UserCounters& counters) {
  folly::BenchmarkSuspender suspender;
  ncclCvarInit();
  MapperTrace trace;
  auto coll = std::make_shared<NopCollRecord>();
  suspender.dismiss();

  for (uint32_t i = 0; i < iters; ++i) {
    trace.recordMapperEvent(CollStart{coll});
    for (int j = 0; j < eventsPerColl; ++j) {
      CtranMapperRequest req;
      trace.recordMapperEvent(
          PutStart{
              .sendBuffer = nullptr,
              .remoteBuffer = nullptr,
              .length = 100,
              .peerRank = j % 8,
              .sourceHandle = nullptr,
              .remoteAccessKey = {},
              .req = &req,
          });
    }
    trace.recordMapperEvent(CollEnd{});
  }
  counters["events_per_coll"] = eventsPerColl;
}

BENCHMARK_SINGLE_PARAM_COUNTERS(BM_RecordPutStart, 100);
BENCHMARK_SINGLE_PARAM_COUNTERS(BM_RecordPutStart, 1000);

static void BM_RecordRecvNotified(
    uint32_t iters,
    int eventsPerColl,
    folly::UserCounters& counters) {
  folly::BenchmarkSuspender suspender;
  ncclCvarInit();
  MapperTrace trace;
  auto coll = std::make_shared<NopCollRecord>();
  suspender.dismiss();

  for (uint32_t i = 0; i < iters; ++i) {
    trace.recordMapperEvent(CollStart{coll});
    for (int j = 0; j < eventsPerColl; ++j) {
      trace.recordMapperEvent(RecvNotified{.peerRank = j % 8});
    }
    trace.recordMapperEvent(CollEnd{});
  }
  counters["events_per_coll"] = eventsPerColl;
}

BENCHMARK_SINGLE_PARAM_COUNTERS(BM_RecordRecvNotified, 100);
BENCHMARK_SINGLE_PARAM_COUNTERS(BM_RecordRecvNotified, 1000);

static void BM_RecordCopyStart(
    uint32_t iters,
    int eventsPerColl,
    folly::UserCounters& counters) {
  folly::BenchmarkSuspender suspender;
  ncclCvarInit();
  MapperTrace trace;
  auto coll = std::make_shared<NopCollRecord>();
  suspender.dismiss();

  for (uint32_t i = 0; i < iters; ++i) {
    trace.recordMapperEvent(CollStart{coll});
    for (int j = 0; j < eventsPerColl; ++j) {
      CtranMapperRequest req;
      trace.recordMapperEvent(
          CopyStart{
              .sourceBuffer = nullptr,
              .destBuffer = nullptr,
              .length = 100,
              .stream = nullptr,
              .req = &req,
          });
    }
    trace.recordMapperEvent(CollEnd{});
  }
  counters["events_per_coll"] = eventsPerColl;
}

BENCHMARK_SINGLE_PARAM_COUNTERS(BM_RecordCopyStart, 100);
BENCHMARK_SINGLE_PARAM_COUNTERS(BM_RecordCopyStart, 1000);

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}

// clang-format off

/*
=============================================================================================
[...]ltrace/benchmarks/MapperTraceBench.cc     relative  time/iter   iters/s  events_per_coll
=============================================================================================
BM_RecordPutStart(100)                                      5.33us   187.70K              100
BM_RecordPutStart(1000)                                    69.23us    14.45K             1000
BM_RecordRecvNotified(100)                                755.02ns     1.32M              100
BM_RecordRecvNotified(1000)                                 7.97us   125.53K             1000
BM_RecordCopyStart(100)                                     8.39us   119.20K              100
BM_RecordCopyStart(1000)                                   65.53us    15.26K             1000
=============================================================================================

So roughly: 1 putStart = 53ns
            1 recvNotified = 7ns
            1 copyStart = 80ns
*/
