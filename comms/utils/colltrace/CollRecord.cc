// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/CollRecord.h"

namespace meta::comms::colltrace {

// CollTimingRecord implementation
void CollTimingRecord::setPreviousCollEndTs(system_clock_time_point ts) {
  previousCollEndTs_ = ts;
}

CollTimingRecord::system_clock_time_point
CollTimingRecord::getPreviousCollEndTs() const {
  return previousCollEndTs_.load(std::memory_order::relaxed);
}

void CollTimingRecord::setCollEnqueueTs(system_clock_time_point ts) {
  collEnqueueTs_ = ts;
}

CollTimingRecord::system_clock_time_point CollTimingRecord::getCollEnqueueTs()
    const {
  return collEnqueueTs_.load();
}

void CollTimingRecord::setCollStartTs(system_clock_time_point ts) {
  collStartTs_ = ts;
}

CollTimingRecord::system_clock_time_point CollTimingRecord::getCollStartTs()
    const {
  return collStartTs_.load();
}

void CollTimingRecord::setCollEndTs(system_clock_time_point ts) {
  collEndTs_ = ts;
}

CollTimingRecord::system_clock_time_point CollTimingRecord::getCollEndTs()
    const {
  return collEndTs_.load();
}

bool CollTimingRecord::clockInitialized(const system_clock_time_point& time) {
  return time.time_since_epoch().count() != 0;
}

std::size_t CollTimingRecord::hash() const {
  return folly::hash::hash_combine(
      previousCollEndTs_.load().time_since_epoch().count(),
      collEnqueueTs_.load().time_since_epoch().count(),
      collStartTs_.load().time_since_epoch().count(),
      collEndTs_.load().time_since_epoch().count());
}

bool CollTimingRecord::equals(const CollTimingRecord& other) const noexcept {
  // We don't guarantee that if another thread is updating the timestamps, we
  // will get the correct result. This use pattern doesn't make sense anyway.
  return previousCollEndTs_.load() == other.previousCollEndTs_.load() &&
      collEnqueueTs_.load() == other.collEnqueueTs_.load() &&
      collStartTs_.load() == other.collStartTs_.load() &&
      collEndTs_.load() == other.collEndTs_.load();
}

folly::dynamic CollTimingRecord::toDynamic() const noexcept {
  folly::dynamic result = folly::dynamic::object();

  // Time points are in microseconds
  result["enqueueTs"] = std::chrono::duration_cast<std::chrono::microseconds>(
                            collEnqueueTs_.load().time_since_epoch())
                            .count();

  result["startTs"] = std::chrono::duration_cast<std::chrono::microseconds>(
                          collStartTs_.load().time_since_epoch())
                          .count();
  // Newly added fields
  result["endTs"] = std::chrono::duration_cast<std::chrono::microseconds>(
                        collEndTs_.load().time_since_epoch())
                        .count();
  result["previousEndTs"] =
      std::chrono::duration_cast<std::chrono::microseconds>(
          previousCollEndTs_.load().time_since_epoch())
          .count();

  // Durations in microseconds
  result["latencyUs"] = std::chrono::duration_cast<std::chrono::microseconds>(
                            collEndTs_.load() - collStartTs_.load())
                            .count();
  // Execution time us is basically an alias for latency. They both exist for
  // legacy reasons. We should remove one of them.
  result["ExecutionTimeUs"] = result["latencyUs"];

  result["QueueingTimeUs"] =
      std::chrono::duration_cast<std::chrono::microseconds>(
          collStartTs_.load() - collEnqueueTs_.load())
          .count();

  result["InterCollTimeUs"] =
      std::chrono::duration_cast<std::chrono::microseconds>(
          collStartTs_.load() - previousCollEndTs_.load())
          .count();

  return result;
}

bool CollTimingRecord::operator==(const CollTimingRecord& other) const {
  return equals(other);
}

// CollRecord implementation
CollRecord::CollRecord(
    uint64_t collId,
    std::unique_ptr<ICollMetadata> ICollMetadata)
    : collId_(collId), collMetadata_(std::move(ICollMetadata)) {}

uint64_t CollRecord::getCollId() const noexcept {
  return collId_;
}

const ICollMetadata* CollRecord::getCollMetadata() const {
  return collMetadata_.get();
}

CollTimingRecord& CollRecord::getTimingInfo() {
  return timingInfo_;
}

std::size_t CollRecord::hash() const {
  // Hash the collId and timingInfo
  std::size_t seed = folly::hash::hash_combine(collId_, timingInfo_.hash());

  // Add the hash of collMetadata if it exists
  if (collMetadata_) {
    seed = folly::hash::hash_combine(seed, collMetadata_->hash());
  }

  return seed;
}

bool CollRecord::equals(const CollRecord& other) const noexcept {
  // Compare collId and timingInfo
  if (collId_ != other.collId_ || !(timingInfo_ == other.timingInfo_)) {
    return false;
  }

  // Compare collMetadata if both exist
  if (collMetadata_ && other.collMetadata_) {
    return collMetadata_->equals(*other.collMetadata_);
  }

  // If one exists and the other doesn't, they're not equal
  if (collMetadata_ || other.collMetadata_) {
    return false;
  }

  // Both are null, so they're equal
  return true;
}

folly::dynamic CollRecord::toDynamic() const noexcept {
  folly::dynamic result = folly::dynamic::object();

  result.update(timingInfo_.toDynamic());

  // Add collMetadata if it exists
  if (collMetadata_) {
    result.update(collMetadata_->toDynamic());
  }

  // Add collId and opCount to be collID. Because we used opCount as the unique
  // id in Analyzer code and now opCount is not unique, we implicitly use collId
  // and call it "opCount". We should switch to use collId in the future.
  result["collId"] = collId_;
  result["opCount"] = collId_;

  return result;
}

bool CollRecord::operator==(const CollRecord& other) const {
  return equals(other);
}

} // namespace meta::comms::colltrace
