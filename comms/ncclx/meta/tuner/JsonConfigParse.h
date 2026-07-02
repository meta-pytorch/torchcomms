// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// JSON-format front-end for the meta tuner config: bridges folly::dynamic rule
// objects to the shared ConfigParse.h column parsers. Compiled only when folly
// is available (NCCLX_TUNER_WITH_FOLLY_JSON); included by MetaTuner.cc under
// that gate.

#include <optional>
#include <string>
#include <vector>

#include <folly/json/dynamic.h>
#include <folly/json/json.h>

#include "meta/tuner/ConfigParse.h"

namespace ncclx::tuner {

inline std::string jsonStringOr(
    const folly::dynamic& object,
    const char* key,
    const std::string& fallback) {
  const auto* value = object.get_ptr(key);
  if (value == nullptr) {
    return fallback;
  }
  if (value->isString()) {
    return value->asString();
  }
  return folly::toJson(*value);
}

inline std::string jsonIntStr(const folly::dynamic& object, const char* key) {
  const auto* value = object.get_ptr(key);
  if (value == nullptr) {
    return std::string();
  }
  return std::to_string(value->asInt());
}

// bytesPerRank / nNodes / nLocalRanks may be a JSON integer (exact), the
// string "*" (wildcard), OR an interval string (e.g. "[0,1048576]", "(1,)").
// Render either form to the string the Int64Range parser consumes. An omitted
// key maps to an empty string, which buildConfig treats as the wildcard
// (matches any) -- identical to an omitted CSV column.
inline std::string jsonRangeStr(const folly::dynamic& object, const char* key) {
  const auto* value = object.get_ptr(key);
  if (value == nullptr) {
    return std::string();
  }
  if (value->isString()) {
    return value->asString();
  }
  return std::to_string(value->asInt());
}

// Parses one rule object into a TuningConfig. Returns nullopt for any
// per-rule problem (missing filter/config, an unknown enum, an invalid
// interval/numeric, or a bad value type whose asInt() throws). The whole body
// is wrapped so a folly::TypeError (e.g. "channels": [4] or "abc") never
// escapes this function.
inline std::optional<TuningConfig> parseJsonRule(const folly::dynamic& rule) {
  try {
    // Each rule must carry a "filter" object (match conditions) and a
    // "config" object (overrides). Reject any rule missing either.
    const auto* filterObj = rule.get_ptr("filter");
    const auto* configObj = rule.get_ptr("config");
    if (filterObj == nullptr || !filterObj->isObject() ||
        configObj == nullptr || !configObj->isObject()) {
      return std::nullopt;
    }

    // Reuse the CSV column builder so JSON and CSV always parse to identical
    // TuningConfig values for an equivalent table.
    const std::string collType = jsonStringOr(*filterObj, "collective", "");
    const std::string bytesPerRank = jsonRangeStr(*filterObj, "bytesPerRank");
    const std::string algorithm = jsonStringOr(*configObj, "algorithm", "");
    const std::string protocol = jsonStringOr(*configObj, "protocol", "");
    const std::string channels = jsonIntStr(*configObj, "channels");
    const std::string nNodes = jsonRangeStr(*filterObj, "nNodes");
    const std::string nLocalRanks = jsonRangeStr(*filterObj, "nLocalRanks");
    const std::string numPipeOps = jsonIntStr(*filterObj, "numPipeOps");
    const std::string regBuff = jsonIntStr(*filterObj, "regBuff");
    const std::string chunkSize = jsonIntStr(*configObj, "chunkSize");

    // Kept in lock-step with buildConfig's index order; bytesPerRank is the
    // size field at index 1 (where total bytes used to be).
    const std::vector<std::string_view> fields{
        collType,
        bytesPerRank,
        algorithm,
        protocol,
        channels,
        nNodes,
        nLocalRanks,
        numPipeOps,
        regBuff,
        chunkSize};
    return buildConfig(fields);
  } catch (const std::exception&) {
    // A non-numeric / non-string value (object/array/null) reaching asInt()
    // throws folly::TypeError; treat it as a per-rule parse failure.
    return std::nullopt;
  }
}

} // namespace ncclx::tuner
