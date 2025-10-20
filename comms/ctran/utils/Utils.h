// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <folly/String.h>
#include <string>
#include <utility>
#include <vector>

#define ARGTOSTR(arg) #arg

#define BUFOFFSET(addr, offset) \
  reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(addr) + offset)

namespace ctran::utils {

// copy-pasted from ncclx, it's used in CommHash
inline uint64_t getHash(const char* string, int n) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

// generate a unique hash for a given sorted global ranks
// NOTE: if called on the same set of ranks, each generated hash will be
// different this is to ensure we can have unique hash if we split the same rank
// sets multiple times
uint64_t generateCommHash(const std::vector<int>& sortedGlobalRanks);
uint64_t generateCommHash(const int nRanks);

// generate a list of range as pair<start, end> (both inclusive) for a given
// sorted numbers
std::vector<std::pair<int, int>> getRanges(const std::vector<int>& sortedNums);

// dump ranges to string
std::string rangesToStr(const std::vector<std::pair<int, int>>& ranges);

// extract commDesc from communicator
const char* parseCommDesc(const char* commDesc);

// dump a 2D array to string
template <typename T>
inline const std::string array2DToStr(
    const T* ptr,
    const size_t dimY,
    const size_t dimX,
    const size_t maxReportDimY = 10,
    const size_t maxReportDimX = 5,
    const bool showLast = false) {
  size_t maxRows = std::min(dimY, maxReportDimY);
  size_t maxCols = std::min(dimX, maxReportDimX);
  std::vector<std::string> rowStrs;

  // extra 2 spaces for suspension points and last element if showLast is true
  rowStrs.reserve(maxRows + 2);

  std::string colEnd = "";
  if (dimX > maxCols) {
    colEnd = "...";
  }

  auto appendRows = [&](const int y) {
    std::vector<T> vec;
    vec.reserve(maxCols);
    for (auto j = 0; j < maxCols; j++) {
      vec.push_back(ptr[y * dimX + j]);
    }

    // Optionally print last element
    std::string lastElem = "";
    if (showLast && maxCols < dimX) {
      lastElem = std::to_string(ptr[y * dimX + dimX - 1]);
    }

    // Print "..." if there are omitted columns
    rowStrs.push_back(folly::join(" ", vec) + colEnd + lastElem);
    vec.clear();
  };

  for (auto i = 0; i < maxRows; i++) {
    appendRows(i);
  }

  // Print "..." if there are omitted rows
  if (dimY > maxRows) {
    rowStrs.push_back("...");
  }

  // Optionally print last row
  if (showLast && maxRows < dimY) {
    appendRows(dimY - 1);
  }
  return folly::join(", ", rowStrs);
}

template <typename T>
inline const T align(const T size, const T alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

inline const std::string getFullHostname() {
  char hostname[1024];

  // cannot get hostname for system reason
  if (gethostname(hostname, 1024) != 0) {
    return "";
  }
  return std::string(hostname);
}

inline const std::string getHostname() {
  const std::string hostnameStr = getFullHostname();

  std::vector<std::string> hostnameStrVec;
  folly::split('.', hostnameStr, hostnameStrVec);

  // return first part of hostname
  // e.g. twshared1234.01.abc1.facebook.com -> twshared1234
  if (hostnameStrVec.size() > 0) {
    return hostnameStrVec[0];
  } else {
    return hostnameStr;
  }
}

template <typename T, typename U>
T getConfigValue(const U* config, T U::*member, T defaultValue) {
  return config ? config->*member : defaultValue;
}
} // namespace ctran::utils
