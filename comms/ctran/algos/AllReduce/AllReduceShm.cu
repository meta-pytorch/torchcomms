// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllReduce/AllReduceShm.cuh"

#if defined(__HIP_PLATFORM_AMD__)
#else
__shared__ ctran::algos::allreduce::AllReducShm allReduceShmem;
#endif
