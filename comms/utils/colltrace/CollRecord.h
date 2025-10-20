// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/dynamic.h>

#include "comms/utils/colltrace/CollMetadata.h"
#include "comms/utils/colltrace/CollWaitEvent.h"

namespace meta::comms::colltrace {

class CollTimingRecord {
  using system_clock_time_point = ICollWaitEvent::system_clock_time_point;

 public:
  // Setter and getter for previousCollEndTs_
  void setPreviousCollEndTs(system_clock_time_point ts);
  system_clock_time_point getPreviousCollEndTs() const;

  // Setter and getter for collEnqueueTs_
  void setCollEnqueueTs(system_clock_time_point ts);
  system_clock_time_point getCollEnqueueTs() const;

  // Setter and getter for collStartTs_
  void setCollStartTs(system_clock_time_point ts);
  system_clock_time_point getCollStartTs() const;

  // Setter and getter for collEndTs_
  void setCollEndTs(system_clock_time_point ts);
  system_clock_time_point getCollEndTs() const;

  std::size_t hash() const;
  bool equals(const CollTimingRecord& other) const noexcept;
  folly::dynamic toDynamic() const noexcept;

  bool operator==(const CollTimingRecord& other) const;

 private:
  bool clockInitialized(const system_clock_time_point& time);
  std::atomic<system_clock_time_point> previousCollEndTs_;
  std::atomic<system_clock_time_point> collEnqueueTs_;
  std::atomic<system_clock_time_point> collStartTs_;
  std::atomic<system_clock_time_point> collEndTs_;
};

// This is being used to make transition from old to new colltrace easier
// We might remove it in the future. We only define a very simple interface.
class ICollRecord {
 public:
  virtual ~ICollRecord() = default;

  virtual uint64_t getCollId() const noexcept = 0;
  virtual folly::dynamic toDynamic() const noexcept = 0;
};

class CollRecord : public ICollRecord {
 public:
  CollRecord(uint64_t collId, std::unique_ptr<ICollMetadata> ICollMetadata);

  uint64_t getCollId() const noexcept override;
  const ICollMetadata* getCollMetadata() const;

  // Timing info is the only field that we could modify after init
  CollTimingRecord& getTimingInfo();

  std::size_t hash() const;
  bool equals(const CollRecord& other) const noexcept;
  folly::dynamic toDynamic() const noexcept override;

  bool operator==(const CollRecord& other) const;

 private:
  uint64_t collId_; // CollTrace internal collective id. Should always increment
                    // monotonically
  std::unique_ptr<ICollMetadata>
      collMetadata_; // Using unique ptr as we might get an inherited class
  CollTimingRecord
      timingInfo_; // This is the only field that might get changed after init
};
} // namespace meta::comms::colltrace
