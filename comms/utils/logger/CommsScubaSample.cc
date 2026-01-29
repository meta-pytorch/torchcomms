// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/logger/CommsScubaSample.h"

#include <sstream>

#include <folly/debugging/symbolizer/Symbolizer.h>
#include <folly/json/json.h>

namespace comms {

CommsScubaSample::CommsScubaSample(std::string type)
    : sample_(folly::dynamic::object()) {
  sample_["int"] = folly::dynamic::object;
  sample_["normal"] = folly::dynamic::object;
  sample_["normvector"] = folly::dynamic::object;
  sample_["tags"] = folly::dynamic::object;
  sample_["double"] = folly::dynamic::object;
  sample_["normal"]["type"] = std::move(type);
}

void CommsScubaSample::addNormal(const std::string& key, std::string value) {
  sample_["normal"][key] = std::move(value);
}

void CommsScubaSample::addInt(const std::string& key, int64_t value) {
  sample_["int"][key] = value;
}

void CommsScubaSample::addDouble(const std::string& key, double value) {
  sample_["double"][key] = value;
}

void CommsScubaSample::addNormVector(
    const std::string& key,
    std::vector<std::string> value) {
  sample_["normvector"][key] =
      folly::dynamic::array(value.begin(), value.end());
}

void CommsScubaSample::addTagSet(
    const std::string& key,
    const std::set<std::string>& value) {
  sample_["tags"][key] = folly::dynamic::array(value.begin(), value.end());
}

std::string CommsScubaSample::toJson() const {
  return folly::toJson(sample_);
}

void CommsScubaSample::setError(const std::string& error) noexcept {
  try {
    if (shouldCaptureStackTrace()) {
      std::stringstream ss;
      ss << folly::symbolizer::getStackTraceStr();
      std::vector<std::string> stackTraceMangled;
      // @lint-ignore CLANGTIDY
      folly::split('\n', ss.str(), stackTraceMangled);
      for (auto& line : stackTraceMangled) {
        auto demangledLine = folly::demangle(line.c_str()).toStdString();
        line.swap(demangledLine);
      }
      this->stackTrace = stackTraceMangled;
      addNormVector("stack_trace", std::move(stackTraceMangled));
    }

    this->hasException = true;
    this->exceptionMessage = error;

    addInt("exception_set", 1);
    addNormal("exception_message", error);
  } catch (...) {
  }
}

} // namespace comms
