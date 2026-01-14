// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/hints/GlobalHints.h"

namespace ncclx::algos {

const enum NCCL_SENDRECV_ALGO getSendRecvAlgo();

void regGlobalHints(GlobalHints* hintsMngr);

void testOnlyResetAlgoGlobalHints();
} // namespace ncclx::algos
