// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "comms/utils/colltrace/CollTraceInterface.h"

namespace meta::comms::ncclx {

using DumpFieldSet = std::unordered_set<std::string>;

DumpFieldSet parseRequestFields(
    const std::optional<std::string>& requestFieldsStr);

bool isKeyRequested(const DumpFieldSet& fields, std::string_view key);

std::unordered_map<std::string, std::string> dumpNewCollTrace(
    meta::comms::colltrace::ICollTrace& colltrace,
    const DumpFieldSet& requestFields = {});

bool waitForCollTraceDrain(
    meta::comms::colltrace::ICollTrace& colltrace,
    int timeoutMs = 3000);
} // namespace meta::comms::ncclx
