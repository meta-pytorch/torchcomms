// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/CtranTraceLogger.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include <folly/String.h>
#include <folly/logging/xlog.h>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace ctran::utils {

namespace {
// Convert tp to dur in us to be visible in trace
const int kPointToDurUs = 5;
} // namespace
std::string TimestampPoint::toJsonEntry(
    const std::string& name,
    int id,
    const int pid,
    const int seqNum) const {
  std::vector<std::string> argStrs;
  argStrs.push_back(fmt::format("\"seqNum\": \"{}\"", seqNum));
  for (const auto& [key, value] : metaData_) {
    argStrs.push_back(fmt::format("\"{}\": \"{}\"", key, value));
  }
  std::string argStr = folly::join(", ", argStrs);
  return "{\"name\": \"" + name + "\", " + "\"cat\": \"COL\", " +
      "\"id\": " + std::to_string(id) + ", " + "\"ph\": \"X\", " +
      "\"pid\": " + std::to_string(pid) + ", " + "\"args\": {" + argStr + "}," +
      "\"tid\": " + std::to_string(peer_) + ", " + "\"ts\": " +
      std::to_string(
             std::chrono::duration_cast<std::chrono::microseconds>(
                 now_.time_since_epoch())
                 .count()) +
      ", \"dur\": " + std::to_string(kPointToDurUs) + "}";
}

std::string TimeInterval::toJsonEntry(
    const std::string& name,
    int id,
    const int pid,
    const int seqNum) const {
  std::vector<std::string> argStrs;
  argStrs.push_back(fmt::format("\"seqNum\": \"{}\"", seqNum));
  for (const auto& [key, value] : metaData_) {
    argStrs.push_back(fmt::format("\"{}\": \"{}\"", key, value));
  }
  std::string argStr = folly::join(", ", argStrs);
  auto dur = durationUs();
  if (dur == 0) {
    dur = 1; // Ensure always visible in trace
  }
  return "{\"name\": \"" + name + "\", " + "\"cat\": \"COL\", " +
      "\"id\": " + std::to_string(id) + ", " + "\"ph\": \"X\", " +
      "\"pid\": " + std::to_string(pid) + ", " + "\"args\": {" + argStr + "}," +
      "\"dur\": " + std::to_string(dur) + ", " +
      "\"tid\": " + std::to_string(peer_) + ", " + "\"ts\": " +
      std::to_string(
             std::chrono::duration_cast<std::chrono::microseconds>(
                 start_.time_since_epoch())
                 .count()) +
      "}";
}

TraceRecord::TraceRecord(const std::string& algo, const int rank)
    : algo(algo), rank_(rank) {
  mapSeqNumTimePoints.clear();
  mapSeqNumTimeIntervals.clear();
  if (NCCL_CTRAN_ENABLE_TRACE_LOGGER) {
    enabled = true;
    start_ = std::chrono::system_clock::now();
  } else {
    enabled = false;
  }
}

void TraceRecord::addMetadata(const std::string& key, const std::string& val) {
  if (enabled) {
    metaDataKeys_.push_back(key);
    metaDataVals_.push_back(val);
  }
}

std::string TraceRecord::toJsonEntry(int& id, const int pid) const {
  std::vector<std::string> jsonEntries;
  std::vector<std::string> argStrs;
  argStrs.reserve(metaDataKeys_.size());
  for (auto i = 0; i < metaDataKeys_.size(); i++) {
    argStrs.push_back(
        fmt::format(
            "\"{}\": \"{}\"", metaDataKeys_.at(i), metaDataVals_.at(i)));
  }
  std::string argStr = folly::join(", ", argStrs);

  std::string recordEntry = "{\"name\": \"" + algo + "\", " +
      "\"cat\": \"COL\", " + "\"id\": " + std::to_string(id) + ", " +
      "\"ph\": \"X\", " + "\"pid\": " + std::to_string(pid) + ", " +
      "\"dur\": " + std::to_string(durationUs()) + ", " + "\"args\": {" +
      argStr + "}," + "\"tid\": -1, " + "\"ts\": " +
      std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                         start_.time_since_epoch())
                         .count()) +
      "}";
  id++;
  jsonEntries.push_back(recordEntry);

  for (const auto& entry : mapSeqNumTimePoints) {
    const auto& [name, seqNum] = entry.first;
    auto& tsp = entry.second;
    id++;
    jsonEntries.push_back(tsp.toJsonEntry(name, id, pid, seqNum));
  }
  for (const auto& entry : mapSeqNumTimeIntervals) {
    const auto& [name, seqNum] = entry.first;
    auto& timeInterval = entry.second;
    id++;
    jsonEntries.push_back(timeInterval.toJsonEntry(name, id, pid, seqNum));
  }
  return folly::join(",", jsonEntries);
}

TraceLogger::TraceLogger(int rank) {
  rank_ = rank;
  traceRecords_.wlock()->clear();
  if (NCCL_CTRAN_ENABLE_TRACE_LOGGER) {
    traceEnabled_ = true;
  } else {
    traceEnabled_ = false;
  }
}

void TraceLogger::addTraceRecord(std::unique_ptr<TraceRecord> traceRecord) {
  if (isTraceEnabled()) {
    traceRecord->end();
    traceRecords_.wlock()->push_back(std::move(traceRecord));
  }
}

// To make the logic simple, now it is only called when this object is
// destroyed.
void TraceLogger::reportTracing() {
  if (!isTraceEnabled()) {
    return;
  }
  auto lockedTraceRecords = traceRecords_.wlock();
  if (lockedTraceRecords->empty()) {
    XLOG(INFO) << "No trace records to report; skip reporting";
    return;
  }

  // flush trace records.
  // Default output dir is /tmp, and the manifold path example is
  // "manifold://reference_cycle_detector/tree/Mosaic"
  // Then can view the trace in url
  // https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/Mosaic/{filename}&bucket=reference_cycle_detector
  auto outputDir = NCCL_CTRAN_TRACE_LOGGER_LOCAL_DIR;
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
  for (auto& ts : *lockedTraceRecords) {
    jsonEntries.push_back(ts->toJsonEntry(id, rank_));
  }
  stream << folly::join(",", jsonEntries) << std::endl;
  stream << "]" << std::endl;

  XLOG(INFO) << "Dumped logging output to " << filename << " with id " << id;
  std::ofstream f(filename);
  f << stream.str();
  f.close();

  lockedTraceRecords->clear();
}

TraceLogger::~TraceLogger() {
  reportTracing();
}

} // namespace ctran::utils
