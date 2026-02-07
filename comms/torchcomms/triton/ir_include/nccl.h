/*************************************************************************
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Minimal NCCL type definitions for IR/bitcode compilation.
 * This header provides forward declarations and type aliases needed
 * by the NCCLX device headers when compiled with clang for LLVM bitcode.
 ************************************************************************/

#ifndef NCCL_IR_H_
#define NCCL_IR_H_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error type */
typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6
} ncclResult_t;

/* Opaque handle to communicator */
typedef struct ncclComm* ncclComm_t;

/* Window type - pointer to ncclWindow_vidmem struct defined in core__types.h */
struct ncclWindow_vidmem;
typedef struct ncclWindow_vidmem* ncclWindow_t;

#ifdef __cplusplus
}
#endif

#endif /* NCCL_IR_H_ */
