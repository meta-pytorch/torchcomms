// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/perftrace/Tracer.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include <folly/String.h>
#include <folly/logging/xlog.h>
#include <fstream>
#include <sstream>

namespace ctran::perftrace {

Tracer::Tracer(int rank) {
  rank_ = rank;
  records_.wlock()->clear();
  if (NCCL_CTRAN_ENABLE_PERFTRACE) {
    traceEnabled_ = true;
  } else {
    traceEnabled_ = false;
  }
}

void Tracer::addRecord(std::unique_ptr<Record> record) {
  if (isTraceEnabled()) {
    record->end();
    records_.wlock()->push_back(std::move(record));
  }
}

// To make the logic simple, now it is only called when this object is
// destroyed.
void Tracer::reportTracing() {
  if (!isTraceEnabled()) {
    return;
  }
  auto lockedRecords = records_.wlock();
  if (lockedRecords->empty()) {
    XLOG(INFO) << "No trace records to report; skip reporting";
    return;
  }

  // flush trace records.
  // Default output dir is /tmp, and the manifold path example is
  // "manifold://reference_cycle_detector/tree/Mosaic"
  // Then can view the trace in url
  // https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/Mosaic/{filename}&bucket=reference_cycle_detector
  const auto& outputDir = NCCL_CTRAN_PERFTRACE_DIR;
  auto pid = getpid();
  static uint64_t reportCnt = 0;
  std::stringstream stream;
  std::string filename(
      outputDir + "/" + std::string("ctran_trace_log.") + std::to_string(pid) +
      std::string(".rank") + std::to_string(this->rank_) + std::string(".") +
      std::to_string(reportCnt++) + std::string(".json"));
  XLOG(INFO) << "Dumping logging output to " << filename;

  int id = 0;
  stream << "[" << std::endl;
  std::vector<std::string> jsonEntries;
  for (auto& ts : *lockedRecords) {
    jsonEntries.push_back(ts->toJsonEntry(id, rank_));
  }
  stream << folly::join(",", jsonEntries) << std::endl;
  stream << "]" << std::endl;

  XLOG(INFO) << "Dumped logging output to " << filename << " with id " << id;
  std::ofstream f(filename);
  f << stream.str();
  f.close();

  lockedRecords->clear();
}

Tracer::~Tracer() {
  reportTracing();
}

} // namespace ctran::perftrace
