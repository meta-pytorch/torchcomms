// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

// comm.h must precede AlgoFactory.cuh: device.h (reached via comm.h ->
// collectives.h) owns the hip_bf16.h workaround and #errors out if something
// has already pulled in the old hip_bfloat16.h, which AlgoFactory.cuh does.
// Alphabetical order happens to be the correct order, so sorting is safe.
#include "comm.h"
#include "comms/common/algorithms/AlgoFactory.cuh"
#include "nccl.h"

// Creates the AlgoFactory holding Meta's custom collective algorithms for
// `comm`, or returns nullptr when DDA cannot be used for its topology.
//
// Defined in AlgoInit.cc rather than here: the RCCL_PARAM configs it reads
// expand to global definitions, so keeping them out of this header is what
// lets it be included from more than one translation unit.
std::unique_ptr<meta::comms::AlgoFactoryDev> initAlgoFactory(ncclComm_t comm);
