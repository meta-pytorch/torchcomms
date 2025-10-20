// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <optional>
#include <string>

#include "comms/utils/commSpecs.h"

struct CommLogData;

void logMemoryEvent(
    const CommLogData& logMetaData,
    const std::string& callsite,
    const std::string& use,
    uintptr_t memoryAddr,
    std::optional<int64_t> bytes = std::nullopt,
    std::optional<int> numSegments = std::nullopt,
    std::optional<int64_t> durationUs = std::nullopt,
    bool isRegMemEvent = false);
