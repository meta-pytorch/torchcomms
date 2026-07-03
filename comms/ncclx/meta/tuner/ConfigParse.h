// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// Generic, dependency-light parsers shared by the CSV and JSON config loaders
// of the meta tuner: whitespace trimming, collective/algorithm/protocol enum
// parsing, strict integer parsing, and the TuningConfig column builder. These
// helpers have NO folly dependency, so they can be reused from any config
// front-end. The CSV line splitter lives in CsvConfigParse.h and the
// folly::dynamic adapter in JsonConfigParse.h.

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "comm.h"
#include "meta/tuner/Int64Range.h"
#include "meta/tuner/MetaTuner.h"

namespace ncclx::tuner {

inline std::string_view trim(std::string_view value) {
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

// Returns nullopt for an unknown OR empty token so the caller treats an
// unrecognized collective (e.g. a typo) as a parse failure rather than
// silently defaulting.
inline std::optional<ncclFunc_t> parseCollType(const std::string_view value) {
  if (value == "broadcast") {
    return ncclFuncBroadcast;
  } else if (value == "reduce") {
    return ncclFuncReduce;
  } else if (value == "allgather") {
    return ncclFuncAllGather;
  } else if (value == "reducescatter") {
    return ncclFuncReduceScatter;
  } else if (value == "allreduce") {
    return ncclFuncAllReduce;
  } else {
    return std::nullopt;
  }
}

// Returns nullopt for an unknown OR empty token (see parseCollType).
inline std::optional<int> parseAlgorithm(const std::string_view value) {
  if (value == "tree") {
    return NCCL_ALGO_TREE;
  } else if (value == "ring") {
    return NCCL_ALGO_RING;
  } else if (value == "collnet_direct") {
    return NCCL_ALGO_COLLNET_DIRECT;
  } else if (value == "collnet_chain") {
    return NCCL_ALGO_COLLNET_CHAIN;
  } else if (value == "nvls") {
    return NCCL_ALGO_NVLS;
  } else if (value == "nvls_tree") {
    return NCCL_ALGO_NVLS_TREE;
  } else if (value == "pat") {
    return NCCL_ALGO_PAT;
  } else {
    return std::nullopt;
  }
}

// Returns nullopt for an unknown OR empty token (see parseCollType).
inline std::optional<int> parseProtocol(const std::string_view value) {
  if (value == "ll") {
    return NCCL_PROTO_LL;
  } else if (value == "ll128") {
    return NCCL_PROTO_LL128;
  } else if (value == "simple") {
    return NCCL_PROTO_SIMPLE;
  } else {
    return std::nullopt;
  }
}

// Strictly parses a whole integer column as long long. Returns nullopt only
// when the field is non-empty AND does not fully parse as an integer; an empty
// field returns nullopt too, so the caller must check emptiness first (empty =
// default). The full-consumption check rejects trailing garbage like "12x".
inline std::optional<long long> parseIntStrict(const std::string_view value) {
  if (value.empty()) {
    return std::nullopt;
  }
  try {
    size_t consumed = 0;
    const std::string text(value);
    const long long parsed = std::stoll(text, &consumed);
    if (consumed != text.size()) {
      return std::nullopt;
    }
    return parsed;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

// Strictly parses a whole integer column as int. Like parseIntStrict, but
// std::stoi throws std::out_of_range for a value outside int range, so an
// out-of-int-range value (e.g. "99999999999") becomes a parse error (nullopt)
// rather than being silently truncated.
inline std::optional<int> parseIntStrictAsInt(const std::string_view value) {
  if (value.empty()) {
    return std::nullopt;
  }
  try {
    size_t consumed = 0;
    const std::string text(value);
    const int parsed = std::stoi(text, &consumed);
    if (consumed != text.size()) {
      return std::nullopt;
    }
    return parsed;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

inline bool isFieldSet(
    const std::vector<std::string_view>& fields,
    size_t pos) {
  return fields.size() > pos && !fields[pos].empty();
}

// Builds a TuningConfig from the already-trimmed CSV/JSON column strings. Order
// of fields:
//   collType,bytesPerRank,algorithm,protocol,channels,nNodes,nLocalRanks,
//   numPipeOps,regBuff,chunkSize
// bytesPerRank / nNodes / nLocalRanks are Int64Range expressions (interval,
// exact, or "*" wildcard). channels, numPipeOps, regBuff and chunkSize are
// optional; absent fields are passed as empty strings (or omitted entirely)
// and fall back to their wildcard / no-override defaults. Returns nullopt when
// any column fails to parse (an Int64Range that
// does not parse, an unknown collective/algorithm/protocol token, or a numeric
// column that is present but not a valid integer), so the caller can log an
// ERROR and reject the rule.
inline std::optional<TuningConfig> buildConfig(
    const std::vector<std::string_view>& fields) {
  // An empty Int64Range column defaults to the wildcard (matches any).
  const auto parseRangeField =
      [](const std::string_view field) -> std::optional<Int64Range> {
    return field.empty() ? Int64Range{} : parseInt64Range(field);
  };

  const std::optional<Int64Range> bytesPerRank = parseRangeField(fields[1]);
  const std::optional<Int64Range> nNodes = parseRangeField(fields[5]);
  const std::optional<Int64Range> nLocalRanks = parseRangeField(fields[6]);
  if (!bytesPerRank.has_value() || !nNodes.has_value() ||
      !nLocalRanks.has_value()) {
    return std::nullopt;
  }

  const std::optional<ncclFunc_t> collType = parseCollType(fields[0]);
  const std::optional<int> algorithm = parseAlgorithm(fields[2]);
  const std::optional<int> protocol = parseProtocol(fields[3]);
  if (!collType.has_value() || !algorithm.has_value() ||
      !protocol.has_value()) {
    return std::nullopt;
  }

  const std::optional<int> channels =
      isFieldSet(fields, 4) ? parseIntStrictAsInt(fields[4]) : -1;
  const std::optional<int> numPipeOps =
      isFieldSet(fields, 7) ? parseIntStrictAsInt(fields[7]) : -1;
  const std::optional<int> regBuff =
      isFieldSet(fields, 8) ? parseIntStrictAsInt(fields[8]) : -1;
  const std::optional<long long> chunkSize =
      isFieldSet(fields, 9) ? parseIntStrict(fields[9]) : 0;
  if (!channels.has_value() || !numPipeOps.has_value() ||
      !regBuff.has_value() || !chunkSize.has_value()) {
    return std::nullopt;
  }
  // chunkSize is a byte size cast to size_t; a negative value would wrap to a
  // huge size_t, so reject it like any other present-but-invalid numeric.
  if (*chunkSize < 0) {
    return std::nullopt;
  }

  TuningConfig config{};
  config.collType = *collType;
  config.bytesPerRank = *bytesPerRank;
  config.algorithm = *algorithm;
  config.protocol = *protocol;
  config.nChannels = *channels;
  config.nNodes = *nNodes;
  config.nLocalRanks = *nLocalRanks;
  config.numPipeOps = *numPipeOps;
  config.regBuff = *regBuff;
  config.chunkSize = static_cast<size_t>(*chunkSize);
  return config;
}

} // namespace ncclx::tuner
