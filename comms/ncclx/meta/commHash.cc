// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "argcheck.h"
#include "comm.h"
#include "nccl.h"

NCCL_API(ncclResult_t, ncclCommGetUniqueHash, ncclComm_t comm, uint64_t* hash);
ncclResult_t ncclCommGetUniqueHash(ncclComm_t comm, uint64_t* hash) {
  NCCLCHECK(CommCheck(comm, "CommGetUniqueHash", "comm"));

  *hash = comm->commHash;
  return ncclSuccess;
}
