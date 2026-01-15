// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/RMA/Types.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

using namespace ctran;

// Forward declarations for RMA operations
commResult_t ctranPutSignal(
    const void* originBuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    size_t targetDisp,
    CtranWin* win,
    cudaStream_t stream,
    bool signal = true);

commResult_t ctranWaitSignal(int peer, CtranWin* win, cudaStream_t stream);

/**
 * Window-based AllGather implementation using RMA put+signal operations.
 *
 * Algorithm:
 * 1. Each rank copies its send data to its slot in the window data buffer
 * 2. Each rank puts its data to all other ranks using ctranPutSignal
 * 3. Each rank waits for signals from all other ranks using ctranWaitSignal
 *
 * The window layout is: [rank0_data | rank1_data | ... | rankN-1_data]
 * Each rank writes to its designated slot in all peer windows.
 *
 * @param sendbuff Source buffer containing this rank's data
 * @param sendcount Number of elements to send
 * @param datatype Data type of elements
 * @param win Pre-created window with recv buffer (win->winDataPtr is the recv
 * buffer)
 * @param stream CUDA stream for async operations
 */
commResult_t ctranAllGatherWindow(
    const void* sendbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranWin* win,
    cudaStream_t stream) {
  CtranComm* comm = win->comm;
  const auto statex = comm->statex_.get();
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  const size_t sendSize = sendcount * commTypeSize(datatype);

  // The recv buffer is the window's data buffer
  void* recvbuff = win->winDataPtr;

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "ctranAllGatherWindow: sendbuff {} recvbuff {} sendcount {} datatype {} "
      "win {} comm {} commHash {:x} [nranks={}, localRanks={}] stream={}",
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      (void*)win,
      (void*)comm,
      statex->commHash(),
      nRanks,
      statex->nLocalRanks(),
      (void*)stream);

  // Handle trivial case
  if (nRanks == 1) {
    char* mySlot = static_cast<char*>(recvbuff);
    if (sendbuff != mySlot) {
      FB_CUDACHECK(cudaMemcpyAsync(
          mySlot, sendbuff, sendSize, cudaMemcpyDefault, stream));
    }
    return commSuccess;
  }

  // Copy local data to the recv buffer first (in-place or not)
  // Each rank's data goes at offset rank * sendSize
  char* mySlot = static_cast<char*>(recvbuff) + rank * sendSize;
  if (sendbuff != mySlot) {
    FB_CUDACHECK(
        cudaMemcpyAsync(mySlot, sendbuff, sendSize, cudaMemcpyDefault, stream));
  }

  // Phase 1: Put our data to all remote peers
  // Each rank writes its data to its designated slot in each peer's window
  for (int p = 1; p < nRanks; p++) {
    const int peer = (rank + p) % nRanks;
    // targetDisp is in elements, pointing to this rank's slot in the peer's
    // window
    const size_t targetDisp = rank * sendcount;

    FB_COMMCHECK(ctranPutSignal(
        mySlot, // Source: our local data (already in recv buffer)
        sendcount, // Count in elements
        datatype, // Data type
        peer, // Target rank
        targetDisp, // Displacement in elements (our slot in peer's window)
        win, // Window handle
        stream, // CUDA stream
        true)); // Signal after put completes
  }

  // Phase 2: Wait for all remote peers to complete their puts to us
  for (int p = 1; p < nRanks; p++) {
    const int peer = (rank + p) % nRanks;
    FB_COMMCHECK(ctranWaitSignal(peer, win, stream));
  }

  return commSuccess;
}
