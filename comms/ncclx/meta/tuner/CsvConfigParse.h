// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// CSV-format front-end for the meta tuner config: bracket-aware line splitter +
// CSV column-count constants; builds on the shared ConfigParse.h parsers.

#include "meta/tuner/ConfigParse.h"

namespace ncclx::tuner {

// Minimum number of CSV columns required to form a valid rule (up to
// nLocalRanks).
inline constexpr size_t kMinCsvFields = 7;
// Maximum number of CSV columns honored: the 7 required columns plus the
// optional trailing chunkSize (1 more).
inline constexpr size_t kMaxCsvFields = 8;

// Splits a CSV line into trimmed fields, honoring interval brackets: a comma
// inside () or [] (e.g. "[0,1048576]" or "(1,)") is part of an interval and
// does NOT separate columns. Only commas at bracket depth 0 split. Stops after
// kMaxCsvFields fields. The returned views point into `line`.
inline std::vector<std::string_view> splitCsvLine(std::string_view line) {
  std::vector<std::string_view> fields;
  fields.reserve(kMaxCsvFields);
  size_t start = 0;
  int depth = 0;
  for (size_t index = 0; index <= line.size(); index++) {
    const bool atEnd = index == line.size();
    const char character = atEnd ? '\0' : line[index];
    if (!atEnd && (character == '(' || character == '[')) {
      depth++;
    } else if (!atEnd && (character == ')' || character == ']')) {
      if (depth > 0) {
        depth--;
      }
    }
    const bool isSplit = atEnd || (character == ',' && depth == 0);
    if (isSplit) {
      fields.push_back(trim(line.substr(start, index - start)));
      start = index + 1;
      if (fields.size() == kMaxCsvFields) {
        break;
      }
    }
  }
  return fields;
}

} // namespace ncclx::tuner
