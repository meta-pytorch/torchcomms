// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/CtranComm.h"

commResult_t waitSignalSpinningKernel(
    int peer,
    ctran::CtranWin* win,
    cudaStream_t stream,
    uint64_t waitOpCount);
