// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <unordered_map>

#include "comms/utils/colltrace/CollTraceInterface.h"

namespace meta::comms::ncclx {
std::unordered_map<std::string, std::string> dumpNewCollTrace(
    meta::comms::colltrace::ICollTrace& colltrace);
}
