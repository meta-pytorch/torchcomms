// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <string>

#include <folly/Optional.h>
#include <folly/Range.h>
#include <folly/logging/xlog.h>

#ifdef ENABLE_OTEL
#include "comms/utils/logger/OTelLogger.h"
using DataSink = OTelLogger;
#else

/**
 * FB Infra is not fully in conda environment and we build NCCLX in conda
 * with CMake. To log to scuba, we write to files, and a separate
 * process reads the files and uploads to scuba.
 * This defines a MOCK scuba interface for non-FB infra.
 */
#ifdef MOCK_SCUBA_DATA

/**
 * Mock DataSink for unit tests and non-FB infra
 */
class DataSink {
 public:
  explicit DataSink(folly::StringPiece dataset) {
    XLOG(WARNING) << "Empty sink for dataset: " << dataset << ". "
                  << "No logging will be done.";
  }

  size_t addRawData(
      const std::string& dataset,
      const std::string& message,
      folly::Optional<std::chrono::milliseconds> timeout) {
    return 0;
  }
};

#else

#include "rfe/scubadata/ScubaData.h"
using DataSink = facebook::rfe::ScubaData;

#endif // End MOCK_SCUBA_DATA
#endif // End ENABLE_OTEL
