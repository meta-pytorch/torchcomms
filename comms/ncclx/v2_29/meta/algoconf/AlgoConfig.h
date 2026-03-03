// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/cvars/nccl_cvars.h"

namespace ncclx::algoconf {
// Setup global hints for AlgoConfig module. This is called once at first
// communicator creation (see initEnv()). Any setHint|getHint|resetHint call to
// AlgoConfig hints before that will be invalid.
void setupGlobalHints();

enum NCCL_SENDRECV_ALGO getSendRecvAlgo();
enum NCCL_ALLGATHER_ALGO getAllGatherAlgo();
enum NCCL_ALLREDUCE_ALGO getAllReduceAlgo();
enum NCCL_ALLTOALLV_ALGO getAllToAllVAlgo();
enum NCCL_RMA_ALGO getRmaAlgo();

std::string getAlgoHintValue(enum NCCL_SENDRECV_ALGO algo);
std::string getAlgoHintValue(enum NCCL_ALLGATHER_ALGO algo);
std::string getAlgoHintValue(enum NCCL_ALLREDUCE_ALGO algo);
std::string getAlgoHintValue(enum NCCL_ALLTOALLV_ALGO algo);
std::string getAlgoHintValue(enum NCCL_RMA_ALGO algo);

void testOnlyResetAlgoConfig();

void testOnlySetAlgo(enum NCCL_SENDRECV_ALGO algo);
void testOnlySetAlgo(enum NCCL_ALLGATHER_ALGO algo);
void testOnlySetAlgo(enum NCCL_ALLREDUCE_ALGO algo);
void testOnlySetAlgo(enum NCCL_ALLTOALLV_ALGO algo);
void testOnlySetAlgo(enum NCCL_RMA_ALGO algo);
} // namespace ncclx::algoconf
