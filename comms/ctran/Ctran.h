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

commResult_t ctranSend(
    const void* sendbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream);

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

bool ctranAllGatherSupport(CtranComm* comm, enum NCCL_ALLGATHER_ALGO algo);
commResult_t ctranAllGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLGATHER_ALGO algo);

bool ctranReduceScatterSupport(
    CtranComm* comm,
    enum NCCL_REDUCESCATTER_ALGO algo);
commResult_t ctranReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_REDUCESCATTER_ALGO algo);

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

/**
 * Note: we support the combine and dispatch APIs by keeping both recvbuffs (for
 * dispatch) and recvbuff (for combine), for implementation simplicity, as we
 * now only have two moving variables for the two APIs: recvbuffs and
 * recvAllSplitLengths.
 * In the future, if the moving varibles increase and it requires more
 * extendiblity, we should implement a new class/struct like Hints, to support
 * various metadata type.
 *
 * All-to-all communication with dynamic split lengths and non-contiguous
 * receive buffers. Designed for expert-parallel workloads (e.g., Mixture of
 * Experts) where data needs to be routed to different experts across ranks.
 *
 * EXAMPLE SCENARIO:
 * ----------------
 * 4 ranks
 *
 * Rank 0 wants to send data to other ranks with different amounts:
 *
 * SEND-SIDE (What Rank 0 sends):
 * ------------------------------
 * sendbuff: [100 ints | 200 ints | 150 ints | 50 ints | 300 ints | 100 ints]
 * chunk-id:       0        1         2         3         4         5
 *
 * sendSplitLengths = [100, 200, 150, 50, 300, 100]
 *   Meaning: how many elements are in each chunk
 * numSendSplitLengths = 6 (total number of chunks)
 *
 * sendIndices = [0, 1, 4, 3, 5]
 *   Meaning: A list of chunk ids to send
 * (NOTE: we don't send chunk2 at all)
 *
 * sendIndicesBlockLengths = [2, 0, 1, 2] (#entries = #ranks)
 *   Meaning: chunk 0,1 goes to rank0
 *            NOTHING goes to rank1
 *            chunk 4 goes to rank2
 *            chunk 3,5 goes to rank3
 *
 *
 * RECEIVE-SIDE (What Rank 0 receives):
 * ------------------------------------
 * recvbuffs[0]: Data from Rank 0
 * recvbuffs[1]: Data from Rank 1
 * recvbuffs[2]: Data from Rank 2
 * recvbuffs[3]: Data from Rank 3
 *
 * recvAllSplitLengths (output): allgathered sendSplitLengths from all ranks.
 * e.g [sendSplitLengths-rank0, sendSplitLengths-rank1, sendSplitLengths-rank2,
 * sendSplitLengths-rank3]
 * this is used in Dispatch(), but not in Combine().
 *
 *
 * PARAMETERS:
 * -----------
 * @param sendbuff               GPU buffer containing all data to send
 *                               Layout: concatenated chunks in order of
 *                               inputChunkSizes
 *
 * @param inputChunkSizes        GPU array of size inputChunkSizesCount
 *                               Specifies number of elements in each chunk
 *
 * @param inputChunkSizesCount   Total number of chunks
 *
 *
 * @param inputChunkIndices      GPU array of chunk ids to send
 *
 *
 * @param inputChunkCountPerRank GPU array of size numRanks
 *                               Number of chunks sent to each rank
 *                               Partitions inputChunkIndices array by
 * destination
 *
 * @param recvbuffs              Array of numRanks GPU buffer pointers
 *                               recvbuffs[i] receives all data from rank i
 *                               Non-contiguous: separate buffer per sender
 *
 * @param maxSendcount           Maximum total elements that can be sent
 *                               (buffer capacity)
 *
 * @param maxRecvcount           Maximum total elements that can be received
 *                               (buffer capacity)
 *
 * @param hints                  Communication hints for optimization
 *
 * @param datatype               Data type of elements (e.g., commInt32)
 *
 * @param comm                   Ctran communicator
 *
 * @param stream                 CUDA stream for async operations
 *
 * @param combine                false = dispatch mode (scatter TO experts)
 *                               true = combine mode (gather FROM experts)
 *
 * @param outputChunkSizesPerRank Optional GPU output buffer of size
 *                               (numRanks * inputChunkSizesCount)
 *                               Stores actual received sizes for each chunk
 *                               from each sender
 */
commResult_t ctranAlltoallvDynamicSplitNonContig(
    const void* sendbuff,
    const size_t* inputChunkSizes,
    size_t inputChunkSizesCount,
    const size_t* inputChunkIndices,
    const size_t* inputChunkCountPerRank,
    void* const* recvbuffs,
    void* recvbuff,
    size_t maxSendcount,
    size_t maxRecvcount,
    const meta::comms::Hints& hints,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    bool combine,
    size_t* outputChunkSizesPerRank = nullptr);

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
    cudaStream_t stream,
    bool signal = true);
commResult_t ctranSignal(int peer, ctran::CtranWin* win, cudaStream_t stream);
commResult_t
ctranWaitSignal(int peer, ctran::CtranWin* win, cudaStream_t stream);
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
    const int totalNumSendBlocks, // number of blocks (tokens) per batch
    const int blockCount, // number of elements per block (token)
    const int blockNumRecvBuckets, // number of receiving buckets for each
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

commResult_t allToAllvDedupExec(
    const void* sendBuff,
    const int* sendIdx,
    const int* fwdIdx,
    const int* recvIdx,
    void* recvBuff,
    // number of blocks in recvBuff
    int recvBlockIds[],
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
