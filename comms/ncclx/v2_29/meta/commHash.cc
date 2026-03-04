// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "argcheck.h"
#include "comm.h"
#include "nccl.h"

#include "comms/ctran/commstate/CommStateX.h"

NCCL_API(ncclResult_t, ncclCommGetUniqueHash, ncclComm_t comm, uint64_t* hash);
ncclResult_t ncclCommGetUniqueHash(ncclComm_t comm, uint64_t* hash) {
  NCCLCHECK(PtrCheck(comm, "CommGetUniqueHash", "comm"));
  NCCLCHECK(PtrCheck(comm->ctranComm_.get(), "CommGetUniqueHash", "ctranComm"));
  NCCLCHECK(
      PtrCheck(comm->ctranComm_->statex_.get(), "CommGetUniqueHash", "statex"));

  *hash = comm->ctranComm_->statex_->commHash();
  return ncclSuccess;
}
