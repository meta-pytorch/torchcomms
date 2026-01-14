// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "all_reduce_sparse_block.cuh"
#include "argcheck.h"
#include "enqueue.h"
#include "nccl.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/wrapper/MetaFactory.h"

#define NCCLARGCHECK(statement, ...)                                        \
  do {                                                                      \
    if (!(statement)) {                                                     \
      ERR_WITH_SCUBA(__VA_ARGS__);                                          \
      return metaCommToNccl(ErrorStackTraceUtil::log(commInvalidArgument)); \
    }                                                                       \
  } while (0);

static void* unpackSendBlocksKerns[ncclNumTypes] = {
    (void*)ncclKernel_AllReduceSparseBlock_Unpack<int8_t>,
    (void*)ncclKernel_AllReduceSparseBlock_Unpack<uint8_t>,
    (void*)ncclKernel_AllReduceSparseBlock_Unpack<int32_t>,
    (void*)ncclKernel_AllReduceSparseBlock_Unpack<uint32_t>,
    (void*)ncclKernel_AllReduceSparseBlock_Unpack<int64_t>,
    (void*)ncclKernel_AllReduceSparseBlock_Unpack<uint64_t>,
    (void*)ncclKernel_AllReduceSparseBlock_Unpack<half>,
    (void*)ncclKernel_AllReduceSparseBlock_Unpack<float>,
    (void*)ncclKernel_AllReduceSparseBlock_Unpack<double>,
#if defined(__CUDA_BF16_TYPES_EXIST__) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
    (void*)ncclKernel_AllReduceSparseBlock_Unpack<__nv_bfloat16>,
#endif
};

static int unpackMinGridSize = -1;
static int unpackBlockSize = -1;

static inline ncclResult_t unpackSendBlocks(
    const void* sendbuff,
    const int64_t* recv_indices,
    size_t block_count,
    size_t block_length,
    void* recvbuff,
    ncclDataType_t datatype,
    ncclComm* comm,
    cudaStream_t stream) {
  // If first time call into unpack, query cuda recommended block setup
  if (unpackMinGridSize < 0) {
    CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
        &unpackMinGridSize,
        &unpackBlockSize,
        unpackSendBlocksKerns[ncclUint8]));
  }

  // Allow user to customize if specified
  unsigned int num_blocks_x = unpackMinGridSize,
               num_threads_x = unpackBlockSize;
  if (NCCL_ALL_REDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS > 0) {
    num_blocks_x = NCCL_ALL_REDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS;
  }
  if (NCCL_ALL_REDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE > 0) {
    num_threads_x = NCCL_ALL_REDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE;
  }

  INFO(
      NCCL_COLL,
      "ncclAllReduceSparseBlock - Rank %d: unpackSendBlocks num_blocks_x=%d, num_threads_x=%d (cuda recommended: %d, %d)\n",
      comm->rank,
      num_blocks_x,
      num_threads_x,
      unpackMinGridSize,
      unpackBlockSize);

  dim3 grid = {num_blocks_x, 1, 1};
  dim3 block = {num_threads_x, 1, 1};

  void* fn = unpackSendBlocksKerns[datatype];
  void* args[5] = {
      &recvbuff, &sendbuff, &block_count, &recv_indices, &block_length};
  CUDACHECK(cudaLaunchKernel(fn, grid, block, args, 0, stream));
  return ncclSuccess;
}

NCCL_API(
    ncclResult_t,
    ncclAllReduceSparseBlock,
    const void* sendbuff,
    const int64_t* recv_indices,
    size_t block_count,
    size_t block_length,
    void* recvbuff,
    size_t recv_count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream);

/* The firsr version of ncclAllReduceSparseBlock is based on an unpack copy
 * kernel + regular allreduce. The following steps are included:
 * 1. The recvbuff is memset to 0
 * 2. The unpack copy kernel will "unpack" the blocks in sendbuff to local
 * recvbuff based on recv_indices, block_count, block_length.
 * 3. Apply in-place allreduce on the local recvbuff
 *
 * Functionality limitations of the first version:
 * 1. It can support only out-place ncclAllReduceSparseBlock, because we
 * internally copy sendbuff data to recvbuff.
 * 2. It can support only sum operation, since it is not easy to reset recvbuff
 * for other ops. Note that the above limitations are not concern of our use
 * cases. However, we plan to support more general cases as next step.
 *
 * Performance notes:
 * 1. The main overhead compared to regular allreduce is the additional memset
 * and unpack copy. Such overhead is negligible compared to the consequent
 * multinode allreduce.
 * 2. When sendbuff data becomes too sparse in comaprision to recvbuff, the
 * current version will cause significant data exchange for the padding data. We
 * will optimize such case in the future.
 */
ncclResult_t ncclAllReduceSparseBlock(
    const void* sendbuff,
    const int64_t* recv_indices,
    size_t block_count,
    size_t block_length,
    void* recvbuff,
    size_t recv_count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  // Restriction check (see functionality limitation in function description)
  // TODO: remove such restrictions if needed)
  NCCLARGCHECK(
      op == ncclSum,
      "ncclAllReduceSparseBlock only supports ncclSum (%d) as reduction operation, but received (%d).",
      ncclSum,
      op);
  NCCLARGCHECK(
      sendbuff != recvbuff,
      "ncclAllReduceSparseBlock only supports out-place, but received sendbuff=%p, recvbuff=%p.",
      sendbuff,
      recvbuff);
  // Argument validation
  NCCLARGCHECK(
      block_count * block_length <= recv_count,
      "ncclAllReduceSparseBlock expects sendbuff size (block_count * block_length) is less or equal to recv_count, but received block_count=%ld, block_length=%ld, recv_count=%ld",
      block_count,
      block_length,
      recv_count);

  INFO(
      NCCL_COLL,
      "ncclAllReduceSparseBlock - Rank %d: sendbuff=%p, recv_indices=%p, block_count=%ld, block_length=%ld,"
      "recvbuff=%p, recv_count=%ld, datatype=%d\n",
      comm->rank,
      sendbuff,
      recv_indices,
      block_count,
      block_length,
      recvbuff,
      recv_count,
      datatype);

  int devOld = -1;
  size_t element_sz = ncclTypeSize(datatype);
  // To ensure only data copied from sendbuf will change reduced results
  // (e.g., non-zero for sum), we need reset recvbuff if any rank in the
  // communicator has send data (recv_count > 0)
  bool resetFlag = recv_count > 0 ? true : false;
  // Skip local unpack if the local sendbuff is empty. Note we still need reset
  // recvbuff since the other ranks may have valid data and may be reduced with
  // the local recvbuff
  bool unpackFlag = (block_count > 0 && block_length > 0) ? true : false;

  if (resetFlag || unpackFlag) {
    CUDACHECK(cudaGetDevice(&devOld));
    CUDACHECK(cudaSetDevice(comm->cudaDev));
  }

  // Additional pointer checks to ensure valid device pointers are used in
  // reset/unpack
  if (comm->checkPointers) {
    if (resetFlag || unpackFlag) {
      NCCLCHECK(
          CudaPtrCheck(recvbuff, comm, "recvbuff", "ncclAllReduceSparseBlock"));
    }
    if (unpackFlag) {
      NCCLCHECK(CudaPtrCheck(
          recv_indices, comm, "recv_indices", "ncclAllReduceSparseBlock"));
      NCCLCHECK(
          CudaPtrCheck(sendbuff, comm, "sendbuff", "ncclAllReduceSparseBlock"));
    }
  }
  // TODO: check valid block ranges defined in recv_indices

  if (resetFlag) {
    // Assume only support ncclSum, thus reset recvbuff with zero works
    CUDACHECK(cudaMemsetAsync(recvbuff, 0x0, recv_count * element_sz, stream));
  }

  if (unpackFlag) {
    NCCLCHECK(unpackSendBlocks(
        sendbuff,
        recv_indices,
        block_count,
        block_length,
        recvbuff,
        datatype,
        comm,
        stream));
  }

  if (devOld != -1) {
    CUDACHECK(cudaSetDevice(devOld));
  }

  struct ncclInfo info = {
      ncclFuncAllReduce,
      "AllReduce",
      recvbuff,
      recvbuff,
      recv_count,
      datatype,
      op,
      0,
      comm,
      stream, /* Args */
      ALLREDUCE_CHUNKSTEPS,
      ALLREDUCE_SLICESTEPS};
  return ncclEnqueueCheck(&info);
}
