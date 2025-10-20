// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CollStat.h"

#include <fmt/core.h>
#include <cmath>
#include <numeric>

#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/wrapper/DataTypeStrUtils.h"

namespace ncclx::colltrace {

namespace {
template <typename T>
T getNthPercentile(std::vector<T> vec, unsigned int percentile) {
  if (vec.empty()) {
    throw std::invalid_argument("The data vector is empty.");
  }
  if (percentile < 0 || percentile > 100) {
    throw std::invalid_argument("Percentile must be between 0 and 100");
  }
  int index = std::round((vec.size() - 1) * (percentile / 100.0));
  return vec[index];
}

} // namespace

CollStatData CollStatData::fromDurationList(
    std::vector<std::chrono::microseconds> durations) {
  if (durations.empty()) {
    return {};
  }

  std::ranges::sort(durations);

  auto durationSum = std::accumulate(
      durations.begin(), durations.end(), std::chrono::microseconds::zero());

  return CollStatData{
      .p5 = getNthPercentile(durations, 5),
      .p25 = getNthPercentile(durations, 25),
      .p50 = getNthPercentile(durations, 50),
      .p75 = getNthPercentile(durations, 75),
      .p95 = getNthPercentile(durations, 95),
      .min = durations.front(),
      .max = durations.back(),
      .avg = durationSum / durations.size(),
  };
}

void CollStatData::addScubaSampleWithPrefix(
    NcclScubaSample& sample,
    std::string_view prefix) {
  sample.addInt(fmt::format("{}_us_p5", prefix), this->p5.count());
  sample.addInt(fmt::format("{}_us_p25", prefix), this->p25.count());
  sample.addInt(fmt::format("{}_us_p50", prefix), this->p50.count());
  sample.addInt(fmt::format("{}_us_p75", prefix), this->p75.count());
  sample.addInt(fmt::format("{}_us_p95", prefix), this->p95.count());
  sample.addInt(fmt::format("{}_us_avg", prefix), this->avg.count());
  sample.addInt(fmt::format("{}_us_min", prefix), this->min.count());
  sample.addInt(fmt::format("{}_us_max", prefix), this->max.count());
}

void CollTimingRecord::insertRecord(const CollTraceColl& coll) {
  auto latencyMs = std::chrono::duration<float, std::milli>(coll.latency);

  collStats_.emplace_back(SingleCollTimingRecord{
      .executionTime =
          std::chrono::duration_cast<std::chrono::microseconds>(latencyMs),
      .interCollTime = coll.interCollTime,
      .queueingTime = std::chrono::duration_cast<std::chrono::microseconds>(
          coll.startTs - coll.enqueueTs),
  });
}

NcclScubaSample CollTimingRecord::toScubaSample() const {
  NcclScubaSample sample(
      "coll_stats_report", NcclScubaSample::ScubaLogType::LITE);

  if (collStats_.size() == 0) {
    return sample;
  }

  sample.addInt("CollCount", collStats_.size());

  std::vector<std::chrono::microseconds> executionTimes{};
  std::vector<std::chrono::microseconds> interCollTimes{};
  std::vector<std::chrono::microseconds> queueingTimes{};

  std::ranges::for_each(
      collStats_,
      [&executionTimes, &interCollTimes, &queueingTimes](
          SingleCollTimingRecord record) {
        executionTimes.emplace_back(record.executionTime);
        interCollTimes.emplace_back(record.interCollTime);
        queueingTimes.emplace_back(record.queueingTime);
      });

  auto executionTimeStats = CollStatData::fromDurationList(executionTimes);
  auto interCollTimeStats = CollStatData::fromDurationList(interCollTimes);
  auto queueingTimeStats = CollStatData::fromDurationList(queueingTimes);

  executionTimeStats.addScubaSampleWithPrefix(sample, "executionTime");
  interCollTimeStats.addScubaSampleWithPrefix(sample, "interCollTime");
  queueingTimeStats.addScubaSampleWithPrefix(sample, "queueingTime");

  return sample;
}

void CollStat::reportCollsToScuba() {
  for (const auto& [sig, record] : collStatMap_) {
    // Add collective stat fields
    auto sample = record.toScubaSample();

    // Add collective signature fields
    sample.addNormal("opName", sig.opName);
    sample.addNormal("dataType", getDatatypeStr(sig.dataType));
    sample.addInt("count", sig.count.value_or(0));

    sample.addInt("iteration", curIter_);
    sample.addInt("reportIteration", NCCL_COLLSTAT_REPORT_INTERVAL);

    // Add communicator fields
    sample.setCommunicatorMetadata(&this->logMetaData_);

    SCUBA_nccl_collective_stats.addSample(std::move(sample));
  }
}

void CollStat::recordColl(const CollTraceColl& coll) {
  if (coll.iteration > curIter_) {
    if (NCCL_COLLSTAT_REPORT_INTERVAL > 0 &&
        curIter_ % NCCL_COLLSTAT_REPORT_INTERVAL == 0) {
      reportCollsToScuba();
      collStatMap_.clear();
    }
    curIter_ = coll.iteration;
  }

  auto& collStatData = collStatMap_[CollStatSingature{
      .opName = coll.opName,
      .dataType = coll.dataType,
      .count = coll.count,
  }];
  collStatData.insertRecord(coll);
}

} // namespace ncclx::colltrace
