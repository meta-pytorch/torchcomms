// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_COMM_H_
#define CTRAN_COMM_H_

#include <chrono>
#include <memory>
#include <optional>

#include "comms/ctran/interfaces/ICtran.h"

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/commstate/CommStateX.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/memory/memCacheAllocator.h"
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/cvars/nccl_cvars.h"

class Ctran : public ICtran {
 public:
  Ctran(
      CtranComm* comm,
      std::unique_ptr<ctran::IProfilerReporter> reporter = nullptr,
      std::unique_ptr<ctran::IGpeProfilerReporter> gpeReporter = nullptr);
  ~Ctran();

  bool isInitialized() const override;

  // [DEPRECATED] Handle-based memory registration (baseline NCCL pattern).
  // Returns a handle that must be passed to commDeregister.
  // Prefer globalRegisterWithPtr for new code - it supports both single and
  // multi-segment buffers without requiring handle management.
  commResult_t commRegister(void* buff, size_t size, void** handle) override;

  // [DEPRECATED] Handle-based memory deregistration.
  // Prefer globalDeregisterWithPtr free functions for new code.
  commResult_t commDeregister(void* handle) override;

  void updateOpCount() override;
  uint64_t getOpCount() const override;
  uint64_t getCtranOpCount() const override;

 private:
  CtranComm* comm_{nullptr};
};

bool ctranSendRecvSupport(
    int peer,
    CtranComm* comm,
    enum NCCL_SENDRECV_ALGO algo = NCCL_SENDRECV_ALGO::ctran,
    cudaStream_t stream = nullptr);

commResult_t ctranSend(
    const void* sendbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_SENDRECV_ALGO algo);

commResult_t ctranRecv(
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_SENDRECV_ALGO algo);
void ctranSendRecvCleanOpGroup();

commResult_t ctranGroupEndHook(
    std::deque<OpElem*>& opGroup,
    enum NCCL_SENDRECV_ALGO algo,
    std::optional<std::chrono::milliseconds> timeout = std::nullopt);
commResult_t ctranGroupEndHook(
    std::optional<std::chrono::milliseconds> timeout = std::nullopt);

bool ctranAllGatherSupport(
    CtranComm* comm,
    enum NCCL_ALLGATHER_ALGO algo,
    cudaStream_t stream = nullptr);
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

bool ctranAllReduceSupport(CtranComm* comm, enum NCCL_ALLREDUCE_ALGO algo);
commResult_t ctranAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLREDUCE_ALGO algo,
    std::optional<std::chrono::milliseconds> timeout = std::nullopt);

bool ctranAllToAllSupport(
    const size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    enum NCCL_ALLTOALL_ALGO algo);

commResult_t ctranAllToAll(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLTOALL_ALGO algo);

bool ctranAllToAllvSupport(CtranComm* comm);

bool ctranDeviceAllToAllvSupport(CtranComm* comm);

commResult_t ctranDeviceAllToAllv(
    const void* sendbuff,
    void* recvbuff,
    const int64_t* sendcounts_d,
    const int64_t* recvcounts_d,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    int64_t sendcountsMultiplier = 1,
    int64_t recvcountsMultiplier = 1,
    const std::unordered_map<std::string, std::string>& hints = {});

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
    enum NCCL_BROADCAST_ALGO algo,
    std::optional<CtranMapperBackend> specifiedBackend = std::nullopt);
commResult_t ctranBroadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int root,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_BROADCAST_ALGO algo);

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

commResult_t ctranFetchAdd(
    void* resultBuf,
    uint64_t addVal,
    size_t targetIndex,
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

/* Execute a persistent allgather operation.
 *
 * Operations are submitted on the stream provided to allGatherPInit
 * (request->stream). To capture into a CUDA graph, fork request->stream
 * into the capture before calling this function, then join back after:
 *
 *   cudaEventRecord(forkEv, captureStream);
 *   cudaStreamWaitEvent(request->stream, forkEv, 0);
 *   allGatherPExec(sendbuff, count, datatype, request);
 *   cudaEventRecord(joinEv, request->stream);
 *   cudaStreamWaitEvent(captureStream, joinEv, 0);
 */
commResult_t allGatherPExec(
    const void* sendbuff,
    const size_t count,
    commDataType_t datatype,
    CtranPersistentRequest* request);
commResult_t allGatherPDestroy(CtranPersistentRequest* request);
bool allGatherPSupport(CtranComm* comm);

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

/* Window-based alltoall using the same CE+IB algorithm as AllToAllP.
 * Buffer metadata is sourced from CtranWin (post-exchange) instead of
 * PersistArgs. The window must have been allocated and exchanged before init.
 */
commResult_t AllToAllWinInit(
    CtranWin* win,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request);

commResult_t AllToAllWinExec(
    const void* sendbuff,
    const size_t count,
    CtranPersistentRequest* request);

commResult_t AllToAllWinDestroy(CtranPersistentRequest* request);

commResult_t BroadcastWinInit(
    CtranWin* win,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    CtranPersistentRequest*& request);

commResult_t BroadcastWinExec(
    const void* sendbuff,
    const size_t count,
    int root,
    CtranPersistentRequest* request);

commResult_t BroadcastWinDestroy(CtranPersistentRequest* request);

// Global pointer-based memory registration (does not require a comm).
// If forceReg is true, registration happens even in async/lazy mode.
commResult_t globalRegisterWithPtr(
    void* buff,
    size_t size,
    bool forceReg = false,
    bool ncclManaged = false);

// Global pointer-based memory deregistration (does not require a comm).
// If skipRemRelease is true, skip remote IPC release notifications and
// just remove from exportRegCache. Use this during shutdown when remote
// peers may have already exited.
commResult_t
globalDeregisterWithPtr(void* buff, size_t size, bool skipRemRelease = false);

// Global APIs for bulk registration/deregistration of cached segments.
// These are global operations that work on the singleton RegCache.
commResult_t registerAll();
commResult_t deregisterAll();

} // namespace ctran
#endif // CTRAN_COMM_H_
