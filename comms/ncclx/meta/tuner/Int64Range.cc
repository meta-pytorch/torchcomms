// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/tuner/Int64Range.h"

#include <cstdlib>
#include <exception>

namespace ncclx::tuner {

namespace {

// Strips leading/trailing ASCII whitespace from value.
std::string_view trim(std::string_view value) {
  const auto isSpace = [](const char character) {
    return character == ' ' || character == '\t' || character == '\r' ||
        character == '\n';
  };
  while (!value.empty() && isSpace(value.front())) {
    value.remove_prefix(1);
  }
  while (!value.empty() && isSpace(value.back())) {
    value.remove_suffix(1);
  }
  return value;
}

// Parses a decimal int64 from a (trimmed) string. Returns nullopt if value is
// empty or not a fully-consumed integer.
std::optional<int64_t> parseInt64(std::string_view value) {
  const std::string_view trimmed = trim(value);
  if (trimmed.empty()) {
    return std::nullopt;
  }
  try {
    size_t consumed = 0;
    const int64_t parsed = std::stoll(std::string(trimmed), &consumed);
    if (consumed != trimmed.size()) {
      return std::nullopt;
    }
    return parsed;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

} // namespace

bool Int64Range::matches(int64_t n) const {
  if (loBounded) {
    if (loInclusive ? (n < lo) : (n <= lo)) {
      return false;
    }
  }
  if (hiBounded) {
    if (hiInclusive ? (n > hi) : (n >= hi)) {
      return false;
    }
  }
  return true;
}

std::string Int64Range::toString() const {
  if (!loBounded && !hiBounded) {
    return "*";
  }
  if (loBounded && hiBounded && loInclusive && hiInclusive && lo == hi) {
    return std::to_string(lo);
  }
  std::string result;
  result += loInclusive ? '[' : '(';
  if (loBounded) {
    result += std::to_string(lo);
  }
  result += ',';
  if (hiBounded) {
    result += std::to_string(hi);
  }
  result += hiInclusive ? ']' : ')';
  return result;
}

std::optional<Int64Range> parseInt64Range(std::string_view value) {
  const std::string_view trimmed = trim(value);
  if (trimmed.empty()) {
    return std::nullopt;
  }

  const char front = trimmed.front();
  const bool isInterval = front == '(' || front == '[';
  if (!isInterval) {
    // Bare value: "*" is the wildcard; any integer N is exact [N, N].
    if (trimmed == "*") {
      return Int64Range{};
    }
    const std::optional<int64_t> parsed = parseInt64(trimmed);
    if (!parsed.has_value()) {
      return std::nullopt;
    }
    return Int64Range{/* lo */ *parsed,
                      /* hi */ *parsed,
                      /* loInclusive */ true,
                      /* hiInclusive */ true,
                      /* loBounded */ true,
                      /* hiBounded */ true};
  }

  const char back = trimmed.back();
  const bool hiInclusive = back == ']';
  const bool hiBracketValid = back == ']' || back == ')';
  if (!hiBracketValid) {
    return std::nullopt;
  }
  const bool loInclusive = front == '[';

  // Strip the surrounding brackets and split on the single comma.
  const std::string_view body = trim(trimmed.substr(1, trimmed.size() - 2));
  const size_t comma = body.find(',');
  if (comma == std::string_view::npos) {
    return std::nullopt;
  }
  const std::string_view loStr = trim(body.substr(0, comma));
  const std::string_view hiStr = trim(body.substr(comma + 1));
  // A second comma is malformed.
  if (hiStr.find(',') != std::string_view::npos) {
    return std::nullopt;
  }

  Int64Range range{};
  range.loInclusive = loInclusive;
  range.hiInclusive = hiInclusive;

  if (!loStr.empty()) {
    const std::optional<int64_t> lo = parseInt64(loStr);
    if (!lo.has_value()) {
      return std::nullopt;
    }
    range.lo = *lo;
    range.loBounded = true;
  }
  if (!hiStr.empty()) {
    const std::optional<int64_t> hi = parseInt64(hiStr);
    if (!hi.has_value()) {
      return std::nullopt;
    }
    range.hi = *hi;
    range.hiBounded = true;
  }

  // An empty interval body on a side means unbounded; lo > hi (when both
  // bounded) is an empty set, treated as a parse error.
  if (range.loBounded && range.hiBounded && range.lo > range.hi) {
    return std::nullopt;
  }

  return range;
}

} // namespace ncclx::tuner
