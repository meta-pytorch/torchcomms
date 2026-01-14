// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <folly/init/Init.h>

#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"

int main(int argc, char* argv[]) {
  folly::Init init(&argc, &argv);
  int size = 32 * 1024 * 1024;

  int localRank, globalRank, numRanks;

  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;

  std::tie(localRank, globalRank, numRanks, comm) = setupNccl(argc, argv);

  CUDACHECK_TEST(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK_TEST(cudaStreamCreate(&s));

  // communicating using NCCL
  NCCLCHECK_TEST(ncclAllGather(
      (const void*)sendbuff,
      (void*)recvbuff,
      size / numRanks,
      ncclFloat,
      comm,
      s));

  // completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK_TEST(cudaStreamSynchronize(s));

  // free device buffers
  CUDACHECK_TEST(cudaFree(sendbuff));
  CUDACHECK_TEST(cudaFree(recvbuff));

  cleanupNccl(comm);

  printf("[MPI Rank %d] Success \n", globalRank);
  return 0;
}
