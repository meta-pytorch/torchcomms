// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// NCCLX-only public C++ API.
//
// This header holds the NCCLX extensions to the public NCCL API that are not
// part of upstream NVIDIA NCCL. It is included at the tail of the generated
// nccl.h (behind IS_NCCLX) so every consumer of nccl.h sees these
// declarations, while keeping the forked upstream nccl.h.in free of the NCCLX
// API surface. It depends on the NCCL types (ncclComm_t, ncclResult_t, ...)
// declared earlier in nccl.h and must only be reached through nccl.h.
#ifndef NCCL_H_
#error "ncclx.h must be included through nccl.h, not directly."
#endif

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// NCCLX-added extern-C collective entry points, moved out of the forked
// nccl.h.in. Wrapped in extern "C" so the exported symbols keep the unmangled
// C linkage that libnccl.map exports and that the NCCL_API definitions use.
#ifdef __cplusplus
extern "C" {
#endif

/*
 * All-Reduce-Sparse-block (out-place)
 *
 * Reduces data arrays of variable length count in sendbuff using op operation, and
 * leaves identical copies of result on each recvbuff.
 *
 * Arguments:
 *    IN  sendbuff      - Pointer to sendbuf containing data with block_count * block_length number of elements.
 *                        Only out-place is supported at this time. Thus sendbuff must be different from recvbuff.
 *    IN  recvIndices   - List of indices for data blocks in sendbuff. Each index corresponds to the element-wise relative offset of a data block in the recvbuff
 *    IN  blockCount    - Number of blocks in sendbuff
 *    IN  blockLength   - Length of each block in sendbuff
 *    OUT recvbuff      - Pointer to recvbuf that will receive recvcount number of elements
 *    IN  recvCount     - Number of elements in recvbuff. recvcount must be equal or larger than blockcount * blocklength
 *    IN  datatype      - Type of each data element
 *    IN  ncclRedOp_t op - Reduce operation. Only ncclSum is supported at this time.
 *    IN  ncclComm* comm
 *    IN  cudaStream_t stream
 *
 * Example:
 * INPUT:
 *    rank0: sendbuff = [1,1,  2,2,  3,3               6,6,    7,7], recv_indices= [0,2,4,10,12], block_count=5, block_length=2, recv_count=14
 *    rank1: sendbuff = [      2,2,  3,3,        5,5,  6,6],         recv_indices= [2,4,8,10], block_count=4, block_length=2, recv_count=14
 *    rank2: sendbuff = [1,1,        3,3   4,4,        6,6],         recv_indices= [0,4,6,10], block_count=4, block_length=2, recv_count=14
 *    rank3: sendbuff = [1,1,  2,2,  3,3,  4,4,  5,5,  6,6,    7,7], recv_indices= [0,2,4,6,8,10,12], block_count=7, block_length=2, recv_count=14
 * OUTPUT:
 *    rank0: recvbuff = [3,3,  6,6,  12,12, 8,8, 10,10, 24,24, 14,14]
 *    rank1: recvbuff = [3,3,  6,6,  12,12, 8,8, 10,10, 24,24, 14,14]
 *    rank2: recvbuff = [3,3,  6,6,  12,12, 8,8, 10,10, 24,24, 14,14]
 *    rank3: recvbuff = [3,3,  6,6,  12,12, 8,8, 10,10, 24,24, 14,14]
 */
ncclResult_t  ncclAllReduceSparseBlock(const void* sendbuff, const int64_t* recvIndices, size_t blockCount,
    size_t blockLength, void* recvbuff, size_t recvCount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  pncclAllReduceSparseBlock(const void* sendbuff, const int64_t* recvIndices, size_t blockCount,
    size_t blockLength, void* recvbuff, size_t recvCount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

/*
 * Reduce-Scatter with stochastic rounding.
 *
 * See ncclReduceScatter for how reduce scatter works in general. Data would
 * be rounded using stochastic rounding to BF16, transported to a peer, reduce
 * with the peer's existing data in FP32, repeat, until the algorithm finishes
 * execution.
 *
 * More generically, the flow is:
 * I = InputType, T = Transport Type
 * ----> I (---> T ---------> T -----> FP32)repeat -----> I
 * Input   SR     Transport    Reduce            SR? + Output
 *
 * Notes:
 * The PAT algorithm may do multiple rounds of rounding. The
 * random number used for the rounding of each element would be
 * Philox(seed, element_id + num_elements * current_round_number), so we will
 * not be reusing the same random numbers.
 *
 * Limitations:
 * inputType: must be FP32, this will be the type of input/output
 * transportType: must be BF16, provided to make extension easier.
 * op: Only supports ncclSum and ncclAvg operations.
 * seedPtr: must be a GPU MEMORY pointer. The pointer must outlive the collective.
 * algo: Set NCCL_REDUCESCATTER_QUANTIZED_ALGO=ctdirect_ib to use DirectIB
 *       for ncclSum when supported. The default and fallback algorithm is PAT.
 */
ncclResult_t  ncclReduceScatterQuantize(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t inputType, ncclDataType_t transportType,
    ncclRedOp_t op, uint64_t* seedPtr, ncclComm_t comm, cudaStream_t stream);

/*
 * [NCCLX] All-To-Allv
 * Device (i) sends sendcounts[j] of data from offset sdispls[j] to device (j).
 * At the same time, device (i) receives recvcounts[j] of data from device (j)
 * to be placed at rdispls[j]. sendcounts, sdispls, recvcounts and rdispls are
 * all measured in the units of datatype, not bytes. Only out-of-place operation
 * is allowed (i.e., sendbuff != recvbuff).
 * Arguments:
 *    IN  sendbuff    - Data array to send (contains blocks for each other rank)
 *    IN  sendcounts  - Length of each block in sendbuff
 *    IN  sdispls     - Offsets into sendbuff for each participating rank
 *    OUT recvbuff    - Pointer to recvbuf that will receive blocks from other ranks
 *    IN  recvcounts  - Length of each block in recvbuff
 *    IN  rdispls     - Offsets into recvbuff for each participating rank
 *    IN  datatype      - Type of each data element
 *    IN  ncclComm* comm
 *    IN  cudaStream_t stream
 */
ncclResult_t  ncclAllToAllv(const void *sendbuff, const size_t sendcounts[],
    const size_t sdispls[], void *recvbuff, const size_t recvcounts[],
    const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclAllToAllv(const void *sendbuff, const size_t sendcounts[],
    const size_t sdispls[], void *recvbuff, const size_t recvcounts[],
    const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

/*
 * [NCCLX] All-To-All
 * Device (i) sends count of data from offset sendbuff+count*j to device (j).
 * At the same time, device (i) receives count of data from device (j)
 * to be placed at recvbuff+count*j. Only out-of-place operation is allowed
 * (i.e., sendbuff != recvbuff).
 * Arguments:
 *    IN  sendbuff    - Pointer to sendbuf
 *    OUT recvbuff    - Pointer to recvbuf
 *    IN  count       - count of elements to send to (receive from) each rank
 *    IN  datatype      - Type of each data element
 *    IN  ncclComm* comm
 *    IN  cudaStream_t stream
 */
ncclResult_t  ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  pncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

/* Pointer-based registration for all buffer types - no handle returned.
 * Supports both single-segment and multi-segment (expandable) buffers.
 * Use ncclGlobalDeregisterWithPtr(buf, len) to deregister.
 * "Global" because registration is stored in a global RegCache, not per-comm.
 * Does not require a communicator - uses global backends set by first comm.
 * cudaDev is auto-detected from the buffer pointer. */
ncclResult_t  ncclGlobalRegisterWithPtr(void* buff, size_t size);
ncclResult_t pncclGlobalRegisterWithPtr(void* buff, size_t size);

/* Pointer-based deregistration for all buffer types.
 * Uses pinRange to discover all physical segments and frees each.
 * "Global" because registration is stored in a global RegCache, not per-comm.
 * Does not require a communicator - uses global backends set by first comm.
 * cudaDev is auto-detected from the buffer pointer. */
ncclResult_t  ncclGlobalDeregisterWithPtr(void* buff, size_t size);
ncclResult_t pncclGlobalDeregisterWithPtr(void* buff, size_t size);

/* Return the unique hash of the communicator.
 * For all ranks in a given communicator, this hash will be the same.
 */
ncclResult_t  ncclCommGetUniqueHash(ncclComm_t comm, uint64_t* uniqueHash);
ncclResult_t  pncclCommGetUniqueHash(ncclComm_t comm, uint64_t* uniqueHash);

#ifdef __cplusplus
} // extern "C"
#endif

#define NCCL_COMM_DUMP
#define NCCL_COMM_DUMP_ALL

/* Dump NCCL current internal state for a given communicator in a key-value store format.
 * define outside extern "C"{} to pass C++ template */
ncclResult_t  ncclCommDump(ncclComm_t comm, std::unordered_map<std::string, std::string>& map);

/* Dump NCCL current internal state for all the communicators.
 * The returned map is in the format {commHash: {key: value}} where
 * {key: value} is the result of ncclCommDump in the communicator with hash commHash.
 * hints: key-value map of options. Supported hints:
 *   "comm_dump::requestFields" — semicolon-separated list of field names to include.
 *   "comm_dump::flush" — "1" to flush ring buffers before dumping.
 *   Empty map (default) dumps all fields without flushing.
 */
ncclResult_t ncclCommDumpAll(std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& map,
    const std::unordered_map<std::string, std::string>& hints = {});

/* Snapshot of a communicator's collective tuning model.
 *
 * The bandwidths/latencies tables are the raw model built once at communicator
 * init by ncclTopoTuneModel(): algorithm bandwidth in GB/s (0 means the
 * (function, algorithm, protocol) combination is disabled or unavailable,
 * including NCCL_ALGO/NCCL_PROTO masks) and base latency in microseconds.
 * They are uncorrected: the per-size correction factors that
 * ncclTopoGetAlgoTime() applies are NOT reflected in them.
 *
 * bestBySize[f][s] is the evaluated model at messageSizes[s] bytes: the
 * algorithm/protocol/channel/thread selection a real collective of that size
 * would run with, and its predicted execution time in microseconds with all
 * correction factors applied. It is produced by the same selection path real
 * calls take (including an external tuner plugin when one is loaded), with
 * numPipeOps = 1 and unregistered buffers (regBuff = 0). algorithm = -1 and
 * timeUs = -1 mean no combination is available. Within an interval where the
 * selection does not change, predicted time is affine in nBytes, so linear
 * interpolation between adjacent entries reproduces the model.
 *
 * Not modeled: the fp8 ring relegation for comms larger than 8 ranks (a
 * precision preference, not a time estimate), and per-call op aggregation
 * (numPipeOps > 1 scales only the latency term).
 */
#define NCCL_COLL_TUNING_VERSION 1
#define NCCL_TUNING_MAX_FUNCTIONS 8
#define NCCL_TUNING_MAX_ALGORITHMS 16
#define NCCL_TUNING_MAX_PROTOCOLS 8
#define NCCL_TUNING_SIZE_POINTS 31
#define NCCL_TUNING_NAME_LEN 16 /* incl. NUL; longest today is "ReduceScatter" */

typedef struct {
  int8_t algorithm;  /* index into algorithmNames, -1 if none available */
  int8_t protocol;   /* index into protocolNames */
  int16_t nChannels;
  int16_t nThreads;
  float timeUs;
} ncclCollTuningEntry;

typedef struct {
  int version; /* NCCL_COLL_TUNING_VERSION */

  int nRanks, nNodes, nChannels;
  int minCompCap, maxCompCap;

  int numFunctions, numAlgorithms, numProtocols;
  char functionNames[NCCL_TUNING_MAX_FUNCTIONS][NCCL_TUNING_NAME_LEN];
  char algorithmNames[NCCL_TUNING_MAX_ALGORITHMS][NCCL_TUNING_NAME_LEN];
  char protocolNames[NCCL_TUNING_MAX_PROTOCOLS][NCCL_TUNING_NAME_LEN];

  float bandwidths[NCCL_TUNING_MAX_FUNCTIONS][NCCL_TUNING_MAX_ALGORITHMS][NCCL_TUNING_MAX_PROTOCOLS];
  float latencies[NCCL_TUNING_MAX_FUNCTIONS][NCCL_TUNING_MAX_ALGORITHMS][NCCL_TUNING_MAX_PROTOCOLS];

  uint64_t messageSizes[NCCL_TUNING_SIZE_POINTS]; /* messageSizes[s] = 1 << s */
  ncclCollTuningEntry bestBySize[NCCL_TUNING_MAX_FUNCTIONS][NCCL_TUNING_SIZE_POINTS];
} ncclCollTuning;

/* Fill `tuning` from the communicator's init-time tuning state. Read-only
 * with respect to the communicator and callable from any thread once the
 * communicator is initialized. */
ncclResult_t ncclQueryCollTuning(ncclComm_t comm, ncclCollTuning* tuning);

namespace ncclx::colltrace {

inline constexpr uint64_t kInvalidReplayId = UINT64_MAX;

enum class LifecycleEventType : uint8_t {
  Enqueue,
  Start,
  End,
};

struct LifecycleEvent {
  uint64_t replayId{kInvalidReplayId};
  uint64_t commId{0};
  // Stable one-based submission identity. Zero is reserved for no record.
  // Matches getLatestCollTraceCollectiveId().
  uint64_t collId{0};
  // One-based per-execution identity. Differs from collId for graph replays.
  uint64_t executionCollId{0};
  LifecycleEventType eventType{LifecycleEventType::Enqueue};
  double timestamp{0};
};

// Returns the process-local communicator identity used by lifecycle events.
// Returns ncclInvalidArgument for a null communicator or ncclInvalidUsage if
// the lifecycle feed is disabled.
ncclResult_t getCollTraceCommId(ncclComm_t comm, uint64_t& commId);

// Returns the latest one-based collective ID submitted on this communicator.
// Writes zero and returns ncclSuccess if no collective has been submitted on
// this communicator. Returns ncclInvalidArgument for a null communicator or
// ncclInvalidUsage if the lifecycle feed is disabled.
ncclResult_t getLatestCollTraceCollectiveId(ncclComm_t comm, uint64_t& collId);

// Destructively drains lifecycle events across all lifecycle-enabled
// communicators in this process. This call blocks until every registered
// colltrace instance finishes a flush. Events are ordered by timestamp.
ncclResult_t drainUnreadLifecycleEvents(std::vector<LifecycleEvent>& events);

} // namespace ncclx::colltrace

// NCCL_HAS_DUMP_ALGO_STAT controls whether dumpAlgoStat() is available.
// To disable (e.g., when using a shim with a
// different ncclComm layout), compile with -DNCCL_HAS_DUMP_ALGO_STAT=0.
#if !defined(NCCL_HAS_DUMP_ALGO_STAT)
#define NCCL_HAS_DUMP_ALGO_STAT
#elif NCCL_HAS_DUMP_ALGO_STAT == 0
#undef NCCL_HAS_DUMP_ALGO_STAT
#endif

#ifdef NCCL_HAS_DUMP_ALGO_STAT
namespace ncclx::colltrace {

// Dump collective algorithm statistics for a communicator.
// Output map format: collective name -> algorithm name -> call count.
// Requires NCCL_COLLTRACE=algostat to be enabled.
// Clears and populates the output map. Empty if algostat not enabled or comm is null.
void dumpAlgoStat(ncclComm_t comm, std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>& map);

} // namespace ncclx::colltrace
#endif // NCCL_HAS_DUMP_ALGO_STAT

namespace ncclx {

/*
 * Window-based RMA API (NCCLX extension)
 *
 * These functions use the window-based model with ncclWindow_t handles.
 * They are placed in the ncclx namespace to avoid conflicts with the
 * comm-based baseline API above.
 */

/*
 * One-side put operation from a local buffer to a remote peer's pre-allocated
 * and registered buffer within a NCCL window.
 */
ncclResult_t ncclPutSignal(
    const void* originBuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t targetDisp,
    ncclWindow_t win,
    cudaStream_t stream);

/*
 * One-side put operation from a local buffer to a remote peer's pre-allocated
 * and registered buffer within a NCCL window. Without signaling.
 */
ncclResult_t ncclPut(
    const void* originBuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    size_t targetDisp,
    ncclWindow_t win,
    cudaStream_t stream);

/*
 * One-side get operation from a remote peer's pre-allocated and registered buffer
 * to a local buffer within a NCCL window. Without signaling.
 */
ncclResult_t ncclGet(
    void* targetBuff,
    size_t targetDisp,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclWindow_t win,
    cudaStream_t stream);

/*
 * Wait for a signal from remote peer to complete the put operation.
 */
ncclResult_t ncclWaitSignal(int peer, ncclWindow_t win, cudaStream_t stream);

/*
 * One-sided signal operation to a remote peer's signal region at an offset
 * corresponding to the local rank ID.
 */
ncclResult_t ncclSignal(
    int peer,
    ncclWindow_t win,
    cudaStream_t stream);

/*
 * All-To-Allv Dynamic
 * Device (i) sends scounts[j] of data from sbuf[j] to device (j).
 * At the same time, device (i) receives rcounts[j] of data from device (j)
 * to be placed at rbuf[j]. scounts and rcounts are
 * measured in the units of datatype, not bytes. Only out-of-place operation
 * is allowed (i.e., sbufs and rbufs cannot overlap).
 * Arguments:
 *    IN  sbufs       - Data array to send (contains blocks for each other rank)
 *    IN  scounts     - Length of each block in sbuf
 *    OUT rbufs       - Data array to receive (contains blocks for each other rank)
 *    IN  max_rcounts - Max length of each block in rbuf
 *    OUT actual_rcounts - Actual length of each block in rbuf
 *    IN  hints       - Hints for performance
 *    IN  datatype    - Type of each data element
 *    IN  comm
 *    IN  stream
 *
 * Accepted hints:
 *   ncclx_alltoallv_dynamic_sendbuffs_contig: {true, false} (default: false)
 *   --- indicating whether all sendbuffs are part of a single contiguous memory allocation.
 *   ncclx_alltoallv_dynamic_recvbuffs_contig: {true, false} (default: false)
 *   --- indicating whether all recvbuffs are part of a single contiguous memory allocation.
 *   ncclx_alltoallv_dynamic_sendbuffs_location: {cpu, gpu, auto} (default: auto)
 *   --- indicating the location of the pointers of sendbuffs (not the location of each sendbuff).
 *   ncclx_alltoallv_dynamic_sendcounts_location: {cpu, gpu, auto} (default: auto)
 *   --- indicating the location of sendcounts.
 *   ncclx_alltoallv_dynamic_recvbuffs_location: {cpu, gpu, auto} (default: auto)
 *   --- indicating the location of the pointers of recvbuffs (not the location of each recvbuff).
 *   ncclx_alltoallv_dynamic_max_sendcounts_location: {cpu, gpu, auto} (default: auto)
 *   --- indicating the location of maxSendcounts.
 *   ncclx_alltoallv_dynamic_max_recvcounts_location: {cpu, gpu, auto} (default: auto)
 *   --- indicating the location of maxRecvcounts.
 *   ncclx_alltoallv_dynamic_actual_recvcounts_location: {cpu, gpu, auto} (default: auto)
 *   --- indicating the location of actualRecvcounts.
 */
ncclResult_t alltoallvDynamic(const void * const* sendbuffs, const size_t* sendcounts, void * const* recvbuffs,
    size_t maxSendcount, size_t maxRecvcount, size_t* actualRecvcounts,
    const Hints& hints, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t alltoallvDynamicSplit(const void* sendbuff, const size_t* sendSplitLengths, void* const* recvbuffs,
    size_t maxSendcount, size_t maxRecvcount, size_t* actualRecvcounts, const ncclx::Hints& hints,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t alltoallvDynamicSplitNonContig( const void* sendbuff, const size_t* sendSplitLengths,
    size_t numSendSplitLengths, const size_t* sendIndices, const size_t* sendIndicesBlockLengths, void* const* recvbuffs,
    size_t* recvAllSplitLengths, size_t* recvIndices, size_t* recvIndicesBlockLengths, size_t maxSendcount,
    size_t maxRecvcount, const ncclx::Hints& hints, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t alltoallvDynamicDispatch( const void* sendbuff, const size_t* sendSplitLengths,
    size_t numSendSplitLengths, const size_t* sendIndices, const size_t* sendIndicesBlockLengths, void* const* recvbuffs,
    size_t* recvAllSplitLengths, size_t maxSendcount, size_t maxRecvcount, const ncclx::Hints& hints,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t alltoallvDynamicCombine( const void* sendbuff, const size_t* sendSplitLengths,
    size_t numSendSplitLengths, const size_t* sendIndices, const size_t* sendIndicesBlockLengths, void* recvbuff,
    size_t maxSendcount, size_t maxRecvcount, const ncclx::Hints& hints, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream);

/*
 * Device AllToAllv: AllToAllv where split sizes are device tensors.
 * Unlike regular AllToAllv where counts/displacements are host arrays,
 * this variant takes device pointers for sendcounts and recvcounts.
 * Displacements are computed internally as exclusive prefix sums of
 * the counts. This is useful when split sizes are computed on the GPU
 * and not known on the host beforehand.
 */
// Per-collective hint keys: numBlocks, numThreads, blockScheduling
ncclResult_t deviceAllToAllv(const void* sendbuff, void* recvbuff,
    const int64_t* sendcounts_d, const int64_t* recvcounts_d,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,
    int64_t sendcountsMultiplier = 1, int64_t recvcountsMultiplier = 1,
    const std::unordered_map<std::string, std::string>& hints = {});

/*
 * Persistent All-Gather similar to ncclAllgather, the key difference is that
 * the execution will be deferred until allGatherExec is called
 * Arguments:
 *    OUT recvbuff     - Pointer to recvbuf that will receive blocks from other ranks
 *    IN  maxRecvCount - Count of elements of recvbuff
 *    IN  hints        - Hints for skipping control msg
 *    IN  datatype     - NCCL data type
 *    IN  comm         - NCCL communicator
 *    IN  stream       - CUDA stream
 *    OUT request      - Request to be used in ncclCommExec to trigger the execution
 */
ncclResult_t allGatherInit(void* recvbuff, const size_t maxRecvCount, const Hints& hints,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, void** request);


/* Execute the persistent collective operation created by ncclAllGatherInit.
 * Arguments:
 *    IN  sendbuff    - Pointer to sendbuf
 *    IN  count       - count of elements to send to (receive from) each rank
 *    IN  datatype    - NCCL data type used for the allgather execution. It may be different
 *                      from the datatype used in ncclAllGatherInit.
 *    IN  stream      - CUDA stream
 *    IN  request     - Request created by ncclAllGatherInit
 */
ncclResult_t allGatherExec(
    const void* sendbuff,
    const size_t count,
    const ncclDataType_t datatype,
    void* request);

ncclResult_t allToAllvDedupInit(
    const size_t totalNumSendBlocks, // number of blocks (tokens) per batch
    const size_t blockCount, // number of elements per block (token)
    const size_t blockNumRecvBuckets, // number of receiving buckets for each
                                      // block (experts per token, topK)
    const int numRecvBuckets, // number of receiving buckets per rank (expert
                              // per rank)
    const ncclx::Hints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    void** request);

ncclResult_t allToAllvDedupExec(
    const void* sendBuff,
    const int* sendIdx,
    const int* fwdIdx,
    const int* recvIdx,
    void* recvBuff,
    int recvBlockIds[],
    void* request);

/*
 * Trigger the execution of a request of persistent collective operation
 * created by ncclAllGatherInit or ncclAllToAllDedupInit
 */
ncclResult_t pExec(void* request);

/*
 * Persistent AllToAll similar to ncclAlltoAll, the key difference is that
 * AllToAllP requires user to stick with the same recvbuff so that NCCL can exchange
 * recvbuff and hdl once and skip control msg in the future.
 * Arguments:
 *    IN  recvbuff       - Pointer to recvbuf that will receive blocks from other ranks
 *    IN  maxRecvCount   - Max count of elements recved from all ranks
 *    IN  hints          - Hints for skipping control msg
 *    IN  datatype       - NCCL data type
 *    IN  comm           - NCCL communicator
 *    IN  stream         - CUDA stream
 *    OUT request        - Request to be used in ncclCommExec to trigger the execution
 */
ncclResult_t AllToAllInit(
    void* recvbuff,
    const size_t maxRecvCount,
    const Hints& hints,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    void*& request);

/*
 * Trigger the execution of a request of persistent collective operation
 * created by AllToAllInit.
 */
ncclResult_t AllToAllExec(
    const void* sendbuff,
    const size_t count,
    void* request);

/*
 * Free the Persistent collective request
 */
ncclResult_t  pFree(void* request);

std::shared_ptr<const std::unordered_map<std::string, std::string>> getNcclxInfo();

// Set up hints that are supposed to be global to all NCCL communicators. How
// exactly the hints are used and what will happen if the hints are set after
// initialization depends on the receiver of the hints.
ncclResult_t setGlobalHint(std::string key, std::string val);

/*
 * Comm Set Config
 *
 * Update mutable configuration fields on a live communicator. Only
 * algorithm-selection hint keys (sendrecvAlgo, allgatherAlgo, allreduceAlgo,
 * alltoallvAlgo, rmaAlgo) may be changed after communicator creation.
 *
 * The config parameter must be initialized with NCCL_CONFIG_INITIALIZER. All
 * flat ncclConfig_t fields must remain at their undefined defaults; only the
 * hints pointer is read. Setting any flat field or any non-algo hint key
 * returns ncclInvalidUsage. Passing an uninitialized config (missing magic)
 * returns ncclInvalidArgument.
 *
 * The caller must ensure no concurrent collective or config update is in
 * progress on the same communicator.
 */
// [META:COMM_SET_CONFIG]
ncclResult_t commSetConfig(ncclComm_t comm, const ncclConfig_t* config);

} // namespace ncclx
