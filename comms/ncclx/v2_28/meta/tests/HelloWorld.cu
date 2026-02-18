// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <tuple>

#include <folly/init/Init.h>

#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"
#include "tests_common.cuh"

static std::tuple<int, int, int, ncclComm_t> setupNccl(int argc, char** argv) {
  MPICHECK_TEST(MPI_Init(&argc, &argv));

  int localRank, globalRank, numRanks;
  MPICHECK_TEST(MPI_Comm_rank(MPI_COMM_WORLD, &globalRank));
  MPICHECK_TEST(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

  MPI_Comm localComm;
  MPICHECK_TEST(MPI_Comm_split_type(
      MPI_COMM_WORLD,
      MPI_COMM_TYPE_SHARED,
      globalRank,
      MPI_INFO_NULL,
      &localComm));
  MPICHECK_TEST(MPI_Comm_rank(localComm, &localRank));
  MPICHECK_TEST(MPI_Comm_free(&localComm));

  CUDACHECK_TEST(cudaSetDevice(localRank));

  ncclUniqueId id;
  if (globalRank == 0) {
    NCCLCHECK_TEST(ncclGetUniqueId(&id));
  }
  MPICHECK_TEST(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm;
  NCCLCHECK_TEST(ncclCommInitRank(&comm, numRanks, id, globalRank));

  return std::make_tuple(localRank, globalRank, numRanks, comm);
}

static void cleanupNccl(ncclComm_t comm) {
  ncclCommDestroy(comm);
  int initialized = 0;
  MPICHECK_TEST(MPI_Initialized(&initialized));
  if (initialized) {
    MPICHECK_TEST(MPI_Finalize());
  }
}

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
