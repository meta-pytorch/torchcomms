// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_ALL_TO_ALL_DEDUP_IMPL_H_
#define CTRAN_ALL_TO_ALL_DEDUP_IMPL_H_

#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/cvars/nccl_cvars.h"

commResult_t ctranAllToAllDedupExecImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup);

#endif
