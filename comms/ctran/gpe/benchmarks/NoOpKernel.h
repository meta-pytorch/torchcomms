// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/algos/AllGather/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeRing.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/utils/colltrace/common/ColltraceDeviceHandle.h"

__global__ void NoOpKernel(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState,
    meta::comms::colltrace::ColltraceDeviceHandle colltraceHdr,
    ctran::allgather::KernelArgs args);
