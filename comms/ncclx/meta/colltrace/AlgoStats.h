// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "nccl.h"

// TODO: Remove this guard once v2_27 is deprecated — v2_27 does not have the
// comms/utils/colltrace:algostats dependency
#if NCCL_MINOR >= 28

// Re-export AlgoStats from common location
#include "comms/utils/colltrace/AlgoStats.h"

namespace ncclx::colltrace {

// Re-export types from common namespace for backwards compatibility
using meta::comms::colltrace::AlgoStats;

// Note: AlgoStatDump and dumpAlgoStat are declared in nccl.h
// (see NCCL_HAS_DUMP_ALGO_STAT)

} // namespace ncclx::colltrace

#endif // NCCL_MINOR >= 28
