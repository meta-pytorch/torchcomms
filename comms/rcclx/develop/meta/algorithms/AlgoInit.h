// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "BaselineBootstrap.h"
#include "comms/common/algorithms/AlgoFactory.cuh"
#include "nccl.h"
#include "param.h"

// Meta custom algorithm configs
RCCL_PARAM(EnableDdaAllReduce, "ENABLE_DDA_ALL_REDUCE", 0);
RCCL_PARAM(
    DdaAllReduceSendbufBytes,
    "DDA_ALL_REDUCE_SENDBUF_BYTES",
    29 * 1024 * 1024);
RCCL_PARAM(
    DdaAllReduceFlatMaxBytes,
    "DDA_ALL_REDUCE_FLAT_MAX_BYTES",
    200 * 1024);
RCCL_PARAM(
    DdaAllReduceTreeMaxBytes,
    "DDA_ALL_REDUCE_TREE_MAX_BYTES",
    29 * 1024 * 1024);

RCCL_PARAM(DdaAllReduceMaxBlocks, "DDA_ALL_REDUCE_MAX_BLOCKS", 24);

std::unique_ptr<meta::comms::AlgoFactory> initAlgoFactory(ncclComm_t comm) {
  return std::make_unique<::meta::comms::AlgoFactory>(
      std::make_shared<::rcclx::BaselineBootstrap>(comm),
      comm->nRanks,
      comm->rank,
      rcclParamDdaAllReduceMaxBlocks(),
      ::meta::comms::AlgoFactory::AllReduceOptions{
          .enableDda = static_cast<bool>(rcclParamEnableDdaAllReduce()),
          .ddaSendbufSizeBytes =
              static_cast<int>(rcclParamDdaAllReduceSendbufBytes()),
          .ddaFlatMaxThresholdBytes =
              static_cast<int>(rcclParamDdaAllReduceFlatMaxBytes()),
          .ddaTreeMaxThresholdBytes =
              static_cast<int>(rcclParamDdaAllReduceTreeMaxBytes())});
}
