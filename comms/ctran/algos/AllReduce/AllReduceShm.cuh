// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/AllReduce/AllReduceDevTypes.h"

#if defined(__HIP_PLATFORM_AMD__)
__shared__ ctran::algos::allreduce::AllReducShm allReduceShmem;
#else
extern __shared__ ctran::algos::allreduce::AllReducShm allReduceShmem;
#endif
