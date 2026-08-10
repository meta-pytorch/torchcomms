// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/logger/LogUtils.h"
#include "meta/NcclxLogger.h"

#define NCCLX_LOG_SUBSYS(level, subsys, ...) \
  NCCLX_LOG_IF(level, CLOGF_ENABLED(subsys), __VA_ARGS__)
