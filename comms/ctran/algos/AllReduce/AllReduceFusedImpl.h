// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <optional>

#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/utils/commSpecs.h"

/*
 * Note: the Prims-backed fused AllReduce algorithms ("ctree"/ctranAllReduceTree
 * and "cthierarchical_ring"/ctranAllReduceHierarchicalRing) were moved into
 * MCCL (comms/mccl/collectives/allreduce) as the single owner; CTRAN no longer
 * declares or implements them.
 */
