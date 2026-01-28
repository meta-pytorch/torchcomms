// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <set>
#include <string>
#include <vector>

#include <folly/json/dynamic.h>

namespace comms {

// Base class for Scuba samples in comms libraries (NCCL, MCCL, etc.)
// Provides common field setters and JSON serialization.
// See rfe/scubadata/ScubaDataSample.h for the underlying Scuba format.
class CommsScubaSample {
 public:
  // Only allow moves not copies (public interface)
  CommsScubaSample(CommsScubaSample&&) = default;
  CommsScubaSample& operator=(CommsScubaSample&&) = default;
  virtual ~CommsScubaSample() = default;

  // Core field setters
  void addNormal(const std::string& key, std::string value);
  void addInt(const std::string& key, int64_t value);
  void addDouble(const std::string& key, double value);
  void addNormVector(const std::string& key, std::vector<std::string> value);
  void addTagSet(const std::string& key, const std::set<std::string>& value);

  std::string toJson() const;

  // Error handling - captures stack trace and sets exception fields
  // Subclasses may override to customize behavior
  virtual void setError(const std::string& error) noexcept;

  // Extra attributes for subsequent retrieval. We do so to avoid retrieval
  // from dynamic object which may pose undesired exceptions e.g. type
  // conversion if not set properly
  bool hasException{false};
  std::string exceptionMessage;
  std::vector<std::string> stackTrace;

 protected:
  explicit CommsScubaSample(std::string type);

  // Allow subclasses to copy
  CommsScubaSample(const CommsScubaSample&) = default;
  CommsScubaSample& operator=(const CommsScubaSample&) = default;

 private:
  // Override this to control stack trace capture behavior
  virtual bool shouldCaptureStackTrace() const {
    return true;
  }

  folly::dynamic sample_;
};

} // namespace comms
