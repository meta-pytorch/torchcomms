// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "nccl.h"

// Re-export AlgoStats from common location
#include "comms/utils/colltrace/AlgoStats.h"

namespace ncclx::colltrace {

// Re-export types from common namespace for backwards compatibility
using meta::comms::colltrace::AlgoStats;

// Note: AlgoStatDump and dumpAlgoStat are declared in nccl.h
// (see NCCL_HAS_DUMP_ALGO_STAT)

} // namespace ncclx::colltrace
