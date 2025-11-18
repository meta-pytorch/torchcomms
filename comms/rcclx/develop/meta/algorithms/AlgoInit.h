// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "BaselineBootstrap.h"
#include "comms/common/algorithms/AlgoFactory.cuh"
#include "nccl.h"
#include "param.h"

// Meta custom algorithm configs
RCCL_PARAM(DdaMaxBlocks, "DDA_MAX_BLOCKS", 24);
RCCL_PARAM(DdaSendbufBytes, "DDA_SENDBUF_BYTES", 32 * 1024 * 1024);

RCCL_PARAM(EnableDdaAllReduce, "ENABLE_DDA_ALL_REDUCE", 0);
RCCL_PARAM(
    DdaAllReduceFlatMaxBytes,
    "DDA_ALL_REDUCE_FLAT_MAX_BYTES",
    200 * 1024);
RCCL_PARAM(
    DdaAllReduceTreeMaxBytes,
    "DDA_ALL_REDUCE_TREE_MAX_BYTES",
    29 * 1024 * 1024);

RCCL_PARAM(EnableDdaAllGather, "ENABLE_DDA_ALL_GATHER", 0);
RCCL_PARAM(DdaAllGatherMaxBytes, "DDA_ALL_GATHER_MAX_BYTES", 16 * 1024 * 1024);

RCCL_PARAM(EnableDdaReduceScatter, "ENABLE_DDA_REDUCE_SCATTER", 0);
RCCL_PARAM(
    DdaReduceScatterMaxBytes,
    "DDA_REDUCE_SCATTER_MAX_BYTES",
    8 * 1024 * 1024);

RCCL_PARAM(EnableDdaAllToAll, "ENABLE_DDA_ALL_TO_ALL", 0);
RCCL_PARAM(DdaAllToAllMaxBytes, "DDA_ALL_TO_ALL_MAX_BYTES", 2 * 1024 * 1024);

std::unique_ptr<meta::comms::AlgoFactory> initAlgoFactory(ncclComm_t comm) {
  return std::make_unique<::meta::comms::AlgoFactory>(
      std::make_shared<::rcclx::BaselineBootstrap>(comm),
      comm->nRanks,
      comm->rank,
      rcclParamDdaMaxBlocks(),
      rcclParamDdaSendbufBytes(),
      ::meta::comms::AlgoFactory::AllReduceOptions{
          .enableDda = static_cast<bool>(rcclParamEnableDdaAllReduce()),
          .ddaFlatMaxThresholdBytes =
              static_cast<int>(rcclParamDdaAllReduceFlatMaxBytes()),
          .ddaTreeMaxThresholdBytes =
              static_cast<int>(rcclParamDdaAllReduceTreeMaxBytes())},
      ::meta::comms::AlgoFactory::AllGatherOptions{
          .enableDda = static_cast<bool>(rcclParamEnableDdaAllGather()),
          .ddaMaxThresholdBytes =
              static_cast<int>(rcclParamDdaAllGatherMaxBytes())},
      ::meta::comms::AlgoFactory::ReduceScatterOptions{
          .enableDda = static_cast<bool>(rcclParamEnableDdaReduceScatter()),
          .ddaMaxThresholdBytes =
              static_cast<int>(rcclParamDdaReduceScatterMaxBytes())},
      ::meta::comms::AlgoFactory::AllToAllOptions{
          .enableDda = static_cast<bool>(rcclParamEnableDdaAllToAll()),
          .ddaMaxThresholdBytes =
              static_cast<int>(rcclParamDdaAllToAllMaxBytes())});
}
