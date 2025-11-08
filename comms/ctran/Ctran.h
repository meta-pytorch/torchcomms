// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_COMM_H_
#define CTRAN_COMM_H_

#include <chrono>
#include <memory>
#include <optional>

#include "comms/ctran/interfaces/ICtran.h"

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/commstate/CommStateX.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/ctran/interfaces/IBootstrap.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/memory/memCacheAllocator.h"
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/cvars/nccl_cvars.h"

class Ctran : public ICtran {
 public:
  Ctran(CtranComm* comm);
  ~Ctran();

  bool isInitialized() const override;

  commResult_t commRegister(void* buff, size_t size, void** handle) override;
  commResult_t commDeregister(void* handle) override;

  void updateOpCount() override;
  uint64_t getOpCount() const override;
  uint64_t getCtranOpCount() const override;

 private:
  CtranComm* comm_{nullptr};
};

bool ctranSendRecvSupport(int peer, CtranComm* comm);
commResult_t ctranSendSchedule(
    const void* sendbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream,
    std::deque<OpElem*>& opGroup);
commResult_t ctranSend(
    const void* sendbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream);
commResult_t ctranRecvSchedule(
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream,
    std::deque<OpElem*>& opGroup);
commResult_t ctranRecv(
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream);
void ctranSendRecvCleanOpGroup();

commResult_t ctranGroupEndHook(
    std::deque<OpElem*>& opGroup,
    std::optional<std::chrono::milliseconds> timeout = std::nullopt);
commResult_t ctranGroupEndHook(
    std::optional<std::chrono::milliseconds> timeout = std::nullopt);

bool ctranAllGatherSupport(CtranComm* comm);
commResult_t ctranAllGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);

bool ctranReduceScatterSupport(CtranComm* comm);
commResult_t ctranReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream);

bool ctranAllReduceSupport(CtranComm* comm);
commResult_t ctranAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<const enum NCCL_ALLREDUCE_ALGO> algoSpecified = std::nullopt,
    std::optional<std::chrono::milliseconds> timeout = std::nullopt);

bool ctranAllToAllSupport(
    const size_t count,
    commDataType_t datatype,
    CtranComm* comm);

commResult_t ctranAllToAll(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);

bool ctranAllToAllvSupport(CtranComm* comm);

commResult_t ctranAllToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);

commResult_t ctranAllToAllvDynamic(
    const void* const* sendbuffs,
    const size_t* sendcounts,
    void* const* recvbuffs,
    size_t maxSendcount,
    size_t maxRecvcount,
    size_t* actualRecvcounts,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);

commResult_t ctranAlltoallvDynamicSplit(
    const void* sendbuff,
    const size_t* sendSplitLengths,
    void* const* recvbuffs,
    size_t maxSendcount,
    size_t maxRecvcount,
    size_t* actualRecvcounts,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);

commResult_t ctranAlltoallvDynamicSplitNonContig(
    const void* sendbuff,
    const size_t* sendSplitLengths,
    size_t numSendSplitLengths,
    const size_t* sendIndices,
    const size_t* sendIndicesBlockLengths,
    void* const* recvbuffs,
    size_t maxSendcount,
    size_t maxRecvcount,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    bool combine,
    size_t* recvAllSplitLengths = nullptr);

commResult_t ctranAllToAllvDynamicSupport(
    CtranComm* comm,
    const meta::comms::Hints& hints,
    size_t maxSendcount,
    size_t maxRecvcount,
    commDataType_t datatype);

bool ctranAllToAllDedupSupport(CtranComm* comm);

commResult_t ctranAllToAllDedupInit(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    const size_t maxSendCount,
    void*& recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    const size_t maxRecvCount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request);

commResult_t ctranAllToAllDedupExec(CtranPersistentRequest* request);

commResult_t ctranAllToAllDedupDestroy(CtranPersistentRequest* request);

bool ctranBroadcastSupport(
    CtranComm* comm,
    std::optional<CtranMapperBackend> specifiedBackend = std::nullopt);
commResult_t ctranBroadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int root,
    CtranComm* comm,
    cudaStream_t stream);

commResult_t ctranPutSignal(
    const void* origin_buff,
    size_t count,
    commDataType_t datatype,
    int peer,
    size_t target_disp,
    ctran::CtranWin* win,
    CtranComm* comm,
    cudaStream_t stream,
    bool signal = true);
commResult_t ctranSignal(
    size_t signalDisp,
    uint64_t signalVal,
    int peer,
    ctran::CtranWin* win,
    cudaStream_t stream);
commResult_t ctranWaitSignal(
    int peer,
    ctran::CtranWin* win,
    CtranComm* comm,
    cudaStream_t stream);
commResult_t ctranPutSignal_v2(
    const void* origin_buff,
    size_t target_disp,
    size_t count,
    commDataType_t datatype,
    size_t signal_disp,
    uint64_t signal_val,
    int peer,
    ctran::CtranWin* win,
    cudaStream_t stream,
    bool signal);
commResult_t ctranWaitSignal_v2(
    size_t signal_disp,
    uint64_t cmp_val,
    commCmpOp_t cmp_type,
    ctran::CtranWin* win,
    cudaStream_t stream);
commResult_t ctranGet(
    void* recvBuff,
    size_t targetDisp,
    size_t count,
    commDataType_t datatype,
    int peer,
    ctran::CtranWin* win,
    CtranComm* comm,
    cudaStream_t stream);

void ctranGroupTrackDefaultOp(CtranComm* comm);

namespace ctran {

/* Initialize persistent allgather.
 * Blocking until the ctrl messages are exchanged.
 * Output arguments:
 *   - request: stores ctrl messages. Input argument for allgatherPExec()
 */
commResult_t allGatherPInit(
    void* recvbuff,
    const size_t maxRecvCount,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request);

/* Selector for different allgatherp algorithms.
 * Currently only the direct algorithm is implemented for allgatherp
 */
commResult_t allGatherPExec(
    const void* sendbuff,
    const size_t count,
    commDataType_t datatype,
    CtranPersistentRequest* request);
commResult_t allGatherPDestroy(CtranPersistentRequest* request);
bool allGatherPSupport(CtranComm* comm);

// All array inout arguments are merely pointer without value at init time;
// value will be updated at execution
commResult_t allToAllvDedupInit(
    const size_t totalNumSendBlocks, // number of blocks (tokens) per batch
    const size_t blockCount, // number of elements per block (token)
    const size_t blockNumRecvBuckets, // number of receiving buckets for each
                                      // block (experts per token, topK)
    const int numRecvBuckets, // number of receiving buckets per rank (expert
                              // per rank)
    meta::comms::Hints hints, // unused
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request);

commResult_t allToAllvDedupDestroy(CtranPersistentRequest* request);

bool allToAllvDedupSupport(CtranComm* comm, meta::comms::Hints hints);

commResult_t allToAllvDedupPrepare(
    // Receiving buckets for each send block (experts for each token).
    // Total totalNumSendBlocks * blockNumRecvBuckets elements, stored in a 1D
    // array as:
    //  [blk0_bucket0, blk0_bucket1, ..., blk0_bucketN-1
    //  blk1_bucket0, blk1_bucket1, ..., blk1_bucketN-1, ...]
    const int blockRecvBuckets[],

    // Number of blocks (tokens) sent to each rank.
    // Total nRanks elements, stored in a 1D array.
    size_t numSendBlocks[],
    // TODO: add size_t numSendBlocksToNode[], with nNodes elements

    // Number of blocks (tokens) received from each rank.
    // Total numRecvBuckets * nRanks elements, stored in a 1D array as:
    // [bucket0_rank0, bucket0_rank1, ..., bucket0_rankN-1,
    // bucket1_rank0, bucket1_rank1, ..., bucket1_rankN-1, ...]
    size_t numRecvBlocks[],

    // Offset in elements of received blocks from each rank in receive buffer.
    // Same shape as numRecvBlocks.
    size_t recvOffsets[],

    // Number of blocks (tokens) forwarded through each rank. Total nRanks
    // elements.
    // - For cross-node rail rank, it indicates the number of blocks forwarded
    // from the peer. The values are used to calculate the pipeline steps in the
    // consequent exec.
    // - For local rank, it indicates the number of blocks forwarded to the
    // peer. The values are used to calculate the pipeline steps in later
    // combine.
    // - For rest ranks, the value is 0.
    size_t numForwardBlocks[],

    // Total number of blocks (tokens) received from all ranks.
    size_t* totalNumRecvBlocks,

    // Pytorch metadata
    int xnodeInputSplits[],
    int xnodeOutputSplits[],
    int xnodeGatherIndices[],
    int localInputSplits[],
    int localOutputSplits[],
    int localGatherIndices[],
    int eGatherIndices[],

    CtranPersistentRequest* request);

commResult_t allToAllvDedupExec(
    const void* sendBuff,
    const int blockRecvBuckets[], // generated by prepare
    const size_t numSendBlocks[], // generated by prepare
    const size_t numRecvBlocks[], // generated by prepare
    const size_t recvOffsets[], // generated by prepare
    const size_t numForwardBlocks[], // generated by prepare
    const size_t totalNumRecvBlocks, // generated by prepare
    const int* sendIdx,
    const int* fwdIdx,
    const int* recvIdx,
    void* recvBuff,
    int blockSendRanks[],
    CtranPersistentRequest* request);

bool AllToAllPSupport(CtranComm* comm);

commResult_t AllToAllPInit(
    void* recvbuff,
    const size_t maxRecvCount,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request);

commResult_t AllToAllPExec(
    const void* sendbuff,
    const size_t count,
    CtranPersistentRequest* request);

commResult_t AllToAllPDestroy(CtranPersistentRequest* request);

} // namespace ctran
#endif // CTRAN_COMM_H_
