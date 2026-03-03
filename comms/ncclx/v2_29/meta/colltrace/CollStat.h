// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <fmt/format.h>

#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/ScubaLogger.h"
#include "meta/colltrace/CollTraceColl.h"

namespace ncclx::colltrace {

struct CollStatSingature {
  std::string opName;
  ncclDataType_t dataType;
  std::optional<uint64_t> count;

  bool operator==(const CollStatSingature& other) const {
    return opName == other.opName && dataType == other.dataType &&
        count == other.count;
  }

  std::string toString() const {
    return fmt::format(
        "{}_{}_{}", opName, static_cast<int>(dataType), count.value_or(-1));
  }
};

} // namespace ncclx::colltrace

// Implement hash function for CollStatSingature before using it in
// unordered_map
template <>
struct std::hash<ncclx::colltrace::CollStatSingature> {
  size_t operator()(
      const ncclx::colltrace::CollStatSingature& sig) const noexcept {
    std::size_t hashVal = 0xfaceb00c;
    hash_combine(hashVal, sig.opName);
    hash_combine(hashVal, sig.count.value_or(-1));
    hash_combine(hashVal, sig.dataType);
    return hashVal;
  }
};

namespace ncclx::colltrace {

struct CollStatData {
  std::chrono::microseconds p5;
  std::chrono::microseconds p25;
  std::chrono::microseconds p50;
  std::chrono::microseconds p75;
  std::chrono::microseconds p95;
  std::chrono::microseconds min;
  std::chrono::microseconds max;
  std::chrono::microseconds avg;

  static CollStatData fromDurationList(
      std::vector<std::chrono::microseconds> durations);

  void addScubaSampleWithPrefix(
      NcclScubaSample& sample,
      std::string_view prefix);
};

class CollTimingRecord {
 public:
  void insertRecord(const CollTraceColl& coll);
  NcclScubaSample toScubaSample() const;

 private:
  struct SingleCollTimingRecord {
    std::chrono::microseconds executionTime;
    std::chrono::microseconds interCollTime;
    std::chrono::microseconds queueingTime;
  };
  std::vector<SingleCollTimingRecord> collStats_;
};

// This is class is not thread safe and only supposed to be used in a single
// thread (currently it should only be used by CollTrace thread).
class CollStat {
 public:
  CollStat(const CommLogData& logMetaData) : logMetaData_(logMetaData) {};

  void recordColl(const CollTraceColl& coll);

 private:
  void reportCollsToScuba();

  // The latest iteration of the collectives ran by this communicator
  // used to decide when to report CollStat. Currently we report every
  // iterations.
  int curIter_{0};
  const CommLogData logMetaData_;
  std::unordered_map<CollStatSingature, CollTimingRecord> collStatMap_;
};
} // namespace ncclx::colltrace
