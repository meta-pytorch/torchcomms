// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/cvars/nccl_cvars.h"

namespace ncclx::algoconf {
// Setup global hints for AlgoConfig module. This is called once at first
// communicator creation (see initEnv()). Any setHint|getHint|resetHint call to
// AlgoConfig hints before that will be invalid.
void setupGlobalHints();

enum NCCL_SENDRECV_ALGO getSendRecvAlgo();

std::string getAlgoHintValue(enum NCCL_SENDRECV_ALGO algo);

void testOnlyResetAlgoConfig();

void testOnlySetAlgo(enum NCCL_SENDRECV_ALGO algo);
} // namespace ncclx::algoconf
