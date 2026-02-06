// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_MAPPER_H_
#define CTRAN_MAPPER_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/CtranAux.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/nvl/CtranNvl.h"
#include "comms/ctran/backends/socket/CtranSocket.h"
#include "comms/ctran/colltrace/MapperTrace.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/mapper/CtranMapperImpl.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/regcache/IpcRegCache.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CtranPerf.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/utils/commSpecs.h"

#ifdef CTRAN_DISABLE_TCPDM
#include "comms/ctran/backends/mock/CtranTcpDmMock.h"
#else
#include "comms/ctran/backends/tcpdevmem/CtranTcpDm.h"
#endif

// Forward declaration for MapperTrace due to NVCC's lack of concepts support
namespace ncclx::colltrace {
class MapperTrace;
}

class CtranMapperImpl;

const std::string getReqTypeStr(CtranMapperRequest::ReqType type);

class CtranMapper {
 public:
  CtranMapper(CtranComm* comm);
  ~CtranMapper();

  // Allow caller to mapper to mark the mapper object is being destroyed.
  // Thus, internal resource release would need avoid any extra communication.
  void setAtDestruction();

  // Lock the entire mapper instance for exclusive access from the current
  // thread to internal backend and resources. Any concurrent lock request will
  // be blocked till current thread unlock.
  // See CtranMapperEpochRAII for convenient RAII class to guard epoch lock.
  commResult_t epochLock();

  // Unlock the entire mapper instance.
  // See CtranMapperEpochRAII for convenient RAII class to guard epoch lock.
  commResult_t epochUnlock();

  /* Cache and may register the given buffer and return a segment handle.
   * Input arguments:
   *   - buf: the local buffer to be cached, registration will happen only when
   * the eager mode is used
   *   - len: number of bytes of 'buf' to be cached and registered
   *   - forceRegist: force to register the buffer even if set lazy registration
   *   - ncclManaged: whether the buffer is managed by NCCL
   * Output arguments:
   *   - segHdl: a handle of the cached segment
   *   - regHdl: a handle of the registration; valid only when forceRegist or
   *             eager mode is set.
   */
  commResult_t regMem(
      const void* buf,
      std::size_t len,
      void** segHdl,
      bool forceRegist = false,
      bool ncclManaged = false,
      void** regHdl = nullptr);

  /* Deregister and remove the handle in the registration cache.
   * Input arguments:
   *   - regHdl: a handle of the cached segment
   *   - skipRemRelease: whether or not to skip releasing remote imported
   *                     registration. E.g., we'd skip if it is in mapper
   *                     destruction or the caller is manually control remote
   *                     release.
   */
  commResult_t deregMem(void* segHdl, const bool skipRemRelease = false);

  /* Get the handle of the given buffer if it is cached.
   * Input arguments:
   *   - buf: the local buffer to be searched in the cache
   *   - len: number of bytes of 'buf' to be searched
   *   - allowDynamic: whether or not allow to dynamic register if the
   *                   segment is not cached
   * Output arguments:
   *   - regHdl: a handle of the registration
   *   - dynamicRegist: whether or not this buffer is dynamically cached and
   * registered
   */
  commResult_t searchRegHandle(
      const void* buf,
      std::size_t len,
      void** regHdl,
      bool* dynamicRegist,
      bool allowDynamic = true);

  DevMemType segmentType(void* segHdl);

  /* Deregister a dynamic registration.
   * Input arguments:
   *   - regHdl: a handle of the dynamic registration
   */
  commResult_t deregDynamic(void* regHdl);

  /* Deregister an imported buffer registration from remote peer.
   * Input arguments:
   *  - rkey: the remoteAccessKey of the imported remote buffer received in
   *          irecvCtrl.
   */
  commResult_t deregRemReg(struct CtranMapperRemoteAccessKey* rkey);

  /* Asyncrhohnous register the given buffer. If already registered, return
   * commSuccess; otherwise, return commInProgress. If the buffer is not cached
   * by user, async thread will skip it and let GPE thread handle via dynamic
   * registration.
   * Input arguments:
   *   - buf: the local buffer to be registered
   *   - len: number of bytes of 'buf' to be registered
   */
  commResult_t regAsync(const void* buf, const size_t len);

  /* Get all active handles.
   * Output arguments:
   *   - handles: a vector of active handles
   */
  std::vector<void*> getAllRegHandles();

  /* Post a copy op and return a reqest object.
   * Input arguments:
   *   - dbuf: destination buffer to copy the data to
   *   - sbuf: source buffer to copy the data from
   *   - len: number of bytes to copy
   *   - stream: the CUDA stream to execute the copy on
   * Output arguments:
   *   - req (optional): a request object to track the progress of the copy
   */
  commResult_t icopy(
      void* dbuf,
      const void* sbuf,
      std::size_t len,
      cudaStream_t stream,
      CtranMapperRequest** req = nullptr);

  /* Pre-connect peers on the associated backend.
   * This enables the backend to optimize follow-up operations faster.
   * Input arguments:
   *   - peerRanks: the ranks of the peers to be connected
   */
  commResult_t preConnect(const std::unordered_set<int>& peerRanks);

  /* Post a send control op to associated backend.
   * Input arguments:
   *   - buf: the local buffer to be remotely accessed by future iput from the
   * remote peer
   *   - hdl: the handle of the buffer
   *   - peerRank: the rank of the remote peer in the current communicator
   * Output arguments:
   *   - req: the request object to track the progress of the control msg, which
   * will be dynamically allocated by the mapper
   */
  inline commResult_t isendCtrl(
      const void* buf,
      void* hdl,
      int peerRank,
      CtranMapperRequest** req) {
    FB_COMMCHECK(this->checkValidReq(req, __func__));

    *req = new CtranMapperRequest();
    return isendCtrlImpl(buf, hdl, peerRank, *req);
  }

  /* Post a receive control op to associated backend.
   * Input arguments:
   *   - buf: the buffer to receive the control message. It is often a buffer to
   * hold the virtual address of the remote buffer that will be accessed by
   * iput.
   *   - key: the remoteAccessKey of the remote buffer that will be updated by
   * iput. Multiple keys may exist for multiple backend transports.
   *   - peerRank: the rank of the remote peer in the current communicator
   * Output arguments:
   *   - req: the request object to track the progress of the control msg, which
   * will be dynamically allocated by the mapper
   */
  inline commResult_t irecvCtrl(
      void** buf,
      struct CtranMapperRemoteAccessKey* key,
      int peerRank,
      CtranMapperRequest** req) {
    FB_COMMCHECK(this->checkValidReq(req, __func__));

    *req = new CtranMapperRequest();
    return irecvCtrlImpl(buf, key, peerRank, *req);
  }

  /* Post a send control op to associated backend.
   * Input arguments:
   *   - buf: the local buffer to be remotely accessed by future iput from the
   * remote peer
   *   - hdl: the handle of the buffer
   *   - peerRank: the rank of the remote peer in the current communicator
   *   - req: the request object to track the progress of the control msg, which
   * has been allocated by the caller. Note that the req ptr need to remain
   * valid until the req completed. If you store `req` in a std::vector, please
   * preallocate sufficient capacity so that further insertions do not trigger a
   * reallocation, as reallocation invalidates pointers to the stored elements.
   */
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t
  isendCtrl(const void* buf, void* hdl, int peerRank, CtranMapperRequest* req) {
    FB_COMMCHECK(this->checkValidReq(req, __func__));
    return isendCtrlImpl<PerfConfig>(buf, hdl, peerRank, req);
  }

  /* Post a receive control op to associated backend.
   * Input arguments:
   *   - buf: the buffer to receive the control message. It is often a buffer to
   * hold the virtual address of the remote buffer that will be accessed by
   * iput.
   *   - key: the remoteAccessKey of the remote buffer that will be updated by
   * iput. Multiple keys may exist for multiple backend transports.
   *   - peerRank: the rank of the remote peer in the current communicator
   *   - req: the request object to track the progress of the control msg, which
   * has been allocated by the caller. Note that the req ptr need to remain
   * valid until the req completed. If you store `req` in a std::vector, please
   * preallocate sufficient capacity so that further insertions do not trigger a
   * reallocation, as reallocation invalidates pointers to the stored elements.
   */
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t irecvCtrl(
      void** buf,
      struct CtranMapperRemoteAccessKey* key,
      int peerRank,
      CtranMapperRequest* req) {
    FB_COMMCHECK(this->checkValidReq(req, __func__));
    return irecvCtrlImpl<PerfConfig>(buf, key, peerRank, req);
  }

  /* Post a send control message op with raw payload to associated backend.
   * Input arguments:
   *   - type: the type of the control message
   *   - payload: the raw payload buffer to be sent
   *   - size: the size of the payload in bytes
   *   - peerRank: the rank of the remote peer in the current communicator
   *   - req: the request object to track the progress of the control msg, which
   * has been allocated by the caller. Note that the req ptr need to remain
   * valid until the req completed. If you store `req` in a std::vector, please
   * preallocate sufficient capacity so that further insertions do not trigger a
   * reallocation, as reallocation invalidates pointers to the stored elements.
   */
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t isendCtrlMsg(
      const void* payload,
      const size_t size,
      int peerRank,
      CtranMapperRequest* req) {
    FB_COMMCHECK(this->checkValidReq(req, __func__));
    return isendCtrlMsgImpl<PerfConfig>(payload, size, peerRank, req);
  }

  /* Post a receive control message op with raw payload to associated backend.
   * Input arguments:
   *   - payload: the raw payload buffer to receive the control message data
   *   - size: the size of the payload buffer in bytes
   *   - peerRank: the rank of the remote peer in the current communicator
   *   - req: the request object to track the progress of the control msg, which
   * has been allocated by the caller. Note that the req ptr need to remain
   * valid until the req completed. If you store `req` in a std::vector, please
   * preallocate sufficient capacity so that further insertions do not trigger a
   * reallocation, as reallocation invalidates pointers to the stored elements.
   */
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t irecvCtrlMsg(
      void* payload,
      const size_t size,
      int peerRank,
      CtranMapperRequest* req) {
    FB_COMMCHECK(this->checkValidReq(req, __func__));
    return irecvCtrlMsgImp<PerfConfig>(payload, size, peerRank, req);
  }

  /* Batch send control ops to all peerRanks associated backend. All
   * peerRanks should use the same backend. Input arguments:
   *   - buf: the local buffers to be remotely accessed by future iput from the
   * remote peers
   *   - hdl: the handle of the buffer
   *   - peerRanks: the ranks of the remote peers in the current communicator
   *   - backend: the backend to be used for comm. If not IB, fallback to slow
   * path to issue sendctrl peer-by-peer. Output arguments:
   *   - reqs: the requests vector to track the progress of the control msg.
   * User is responsible for allocating the vector to the size of peerRanks.
   */
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t isendCtrlBatch(
      const std::vector<void*>& bufs,
      void* hdl,
      const std::vector<int>& peerRanks,
      std::vector<CtranMapperRequest>& reqs,
      CtranMapperBackend backend = CtranMapperBackend::UNSET) {
    switch (backend) {
      case CtranMapperBackend::IB:
        return isendCtrlBatchImplIB<PerfConfig>(bufs, hdl, peerRanks, reqs);
      default:
        // Fall back to slow path if backend is not IB.
        int idx = 0;
        for (int peer : peerRanks) {
          FB_COMMCHECK(
              isendCtrlImpl<PerfConfig>(bufs[peer], hdl, peer, &reqs[idx++]));
        }
        break;
    }
    return commSuccess;
  }

  /* Post a sync-only send control op to associated backend.
   * Input arguments:
   *   - peerRank: the rank of the remote peer in the current communicator
   * Output arguments:
   *   - req: the request object to track the progress of the control msg
   */
  inline commResult_t isendCtrl(int peerRank, CtranMapperRequest** req) {
    FB_COMMCHECK(this->checkValidReq(req, __func__));

    *req = new CtranMapperRequest();
    return isendCtrlImpl(peerRank, *req);
  }

  inline commResult_t isendCtrl(int peerRank, CtranMapperRequest* req) {
    FB_COMMCHECK(this->checkValidReq(req, __func__));
    return isendCtrlImpl(peerRank, req);
  }

  /* Post a receive sync-only control op to associated backend.
   * Input arguments:
   *   - peerRank: the rank of the remote peer in the current communicator
   * Output arguments:
   *   - req: the request object to track the progress of the control msg
   */
  inline commResult_t irecvCtrl(int peerRank, CtranMapperRequest** req) {
    FB_COMMCHECK(this->checkValidReq(req, __func__));

    *req = new CtranMapperRequest();
    return irecvCtrlImpl(peerRank, *req);
  }

  inline commResult_t irecvCtrl(int peerRank, CtranMapperRequest* req) {
    FB_COMMCHECK(this->checkValidReq(req, __func__));
    return irecvCtrlImpl(peerRank, req);
  }

  /* Blocking allgather control messages among all local ranks of the mapper
   * associated communicator. It returns after sent and received all control
   * messages.
   * Input argument:
   *   - buf: the local buffer to be remotely accessed
   *   - hdl: the handle of the local buffer
   *   - backend: the backend to be used for data transfer. If not specified,
   *              use internal default based on peer rank and memory type.
   * Output arguments:
   *   - remoteBufs: the allgathered remote buffers from all local ranks
   *                 including the rank itself
   *   - remoteAccessKeys: the allgathered remote access keys from all local
   *                       ranks
   */
  commResult_t intraAllGatherCtrl(
      const void* buf,
      void* hdl,
      std::vector<void*>& remoteBufs,
      std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
      CtranMapperBackend backend = CtranMapperBackend::UNSET);

  /* Blocking allgather control messages among ranks of the mapper associated
   * communicator specified in ranks. It returns after sent and received all
   * control messages.
   * Input argument:
   *   - buf: the local buffer to be remotely accessed
   *   - hdl: the handle of the local buffer
   *   - ranks: the ranks to be used for the AllGather
   *   - backend: the backend to be used for data transfer. If not specified,
   *              use internal default based on peer rank and memory type.
   * Output arguments:
   *   - remoteBufs: the allgathered remote buffers from all local ranks
   *                 including the rank itself
   *   - remoteAccessKeys: the allgathered remote access keys from all local
   *                       ranks
   */
  commResult_t allGatherCtrl(
      const void* buf,
      void* hdl,
      const std::vector<int>& ranks,
      std::vector<void*>& remoteBufs,
      std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
      CtranMapperBackend backend = CtranMapperBackend::UNSET);

  /* Blocking allgather control messages among ranks of the mapper associated
   * communicator specified in ranks. It returns after sent and received all
   * control messages.
   * This is a wrapper of allGatherCtrl on all ranks of the communicator
   * Input argument:
   *   - buf: the local buffer to be remotely accessed
   *   - hdl: the handle of the local buffer
   *   - backend: the backend to be used for data transfer. If not specified,
   *              use internal default based on peer rank and memory type.
   * Output arguments:
   *   - remoteBufs: the allgathered remote buffers from all local ranks
   *                 including the rank itself
   *   - remoteAccessKeys: the allgathered remote access keys from all local
   *                       ranks
   */
  commResult_t allGatherCtrl(
      const void* buf,
      void* hdl,
      std::vector<void*>& remoteBufs,
      std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
      CtranMapperBackend backend = CtranMapperBackend::UNSET);

  /* Convenient wrapper of isendCtrl/irecvCtrl to post a blocking barrier among
   * all local ranks of the mapper associated communicator.
   */
  commResult_t intraBarrier();

  /* Convenient wrapper of isendCtrl/irecvCtrl to post a blocking barrier among
   * all ranks of the mapper associated communicator. */
  commResult_t barrier();

  /* Post a put op to associated backend.
   * Input arguments:
   *   - sbuf: local buffer to put data from
   *   - dbuf: virtual address of the remote buffer to receive data. It is
   *           exchanged via isendCtrl|irecvCtrl called by the algorithm
   *           layer.
   *   - len: number of bytes
   *   - peerRank: the rank of the remote peer in the current communicator
   *   - config: the config of the iput for the backends, it may include
   *        - shdl: the handle of the source buffer
   *        - remoteAccessKey: the remote access key of the remote buffer
   *        - notify: whether notify the remote peer when finished the
   * outstanding put.
   *        - kernElem: the kernel element to be used for a NVL put
   *        - ibFastPath: whether use fast path for ib put.
   * Output arguments:
   *   - req: the request object to track the progress of the iput
   */
  // ** NOTICE **
  // Upon ibFastPath=true, we immediately post a single RDMA_WRITE_WITH_IMM on
  // data QP 0.
  //
  // To be eligible for the fast path, the following conditions must be met,
  // otherwise the put will return error:
  // 1. There is no outstanding or pending regular put requests (there can be
  // outstanding fast put requests)
  // 2. The message size is less than or equal to the maximum WQE size so the
  // put can fit in a single WQE (the maximum WQE size is currently equal to
  // NCCL_CTRAN_IB_QP_SCALING_THRESHOLD)
  // 3. The number of pending WQEs on the fast QP is less than the maximum
  // number of QP messages (NCCL_CTRAN_IB_QP_MAX_MSGS)
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iput(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      int peerRank,
      CtranMapperConfig config,
      CtranMapperRequest** req) {
    *req = new CtranMapperRequest();
    return iputImpl<PerfConfig>(sbuf, dbuf, len, peerRank, config, *req);
  }

  /* Post a put op to associated backend.
   * Input arguments:
   *   - sbuf: local buffer to put data from
   *   - dbuf: virtual address of the remote buffer to receive data. It is
   *           exchanged via isendCtrl|irecvCtrl called by the algorithm
   *           layer.
   *   - len: number of bytes
   *   - peerRank: the rank of the remote peer in the current communicator
   *   - config: the config of the iput for the backends, it may include
   *        - shdl: the handle of the source buffer
   *        - remoteAccessKey: the remote access key of the remote buffer
   *        - notify: whether notify the remote peer when finished the
   * outstanding put.
   *        - kernElem: the kernel element to be used for a NVL put
   *        - ibFastPath: whether use fast path for ib put.
   * Output arguments:
   *   - req: the request object to track the progress of the iput. Note that
   * the req can be nullptr, which indicates the put will be posted without
   * local notifiction.
   */
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iput(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      int peerRank,
      CtranMapperConfig config,
      CtranMapperRequest* req) {
    return iputImpl<PerfConfig>(sbuf, dbuf, len, peerRank, config, req);
  }

  /*
   * Batch multiple put operations to efficiently transfer data chunks
   * from local buffers to remote buffers in a single peer rank. This method
   * optimizes performance by reducing the overhead of individual put
   * operations when transferring multiple data segments to the same peer.
   *
   * This method is currently only supported for IB (InfiniBand)
   * backends. For IB backends, it will automatically fallback to issuing
   * individual iput calls sequentially in case batched operations are not
   * possible.
   *
   * Input arguments:
   *   - puts: vector of CtranMapperPutMsg structures containing the details
   *           of each put operation (source buffer, destination buffer,
   *           length, configuration, request tracking, etc.)
   *   - peerRank: the rank of the remote peer in the current communicator
   *
   * Output arguments:
   *   - return: commResult_t indicating success or failure of the batch
   *             operation
   */
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iputBatch(
      const std::vector<CtranMapperPutMsg>& puts,
      int peerRank) {
    return iputBatchImpl<PerfConfig>(puts, peerRank);
  }

  /* Post a get op to associated backend.
   * Input arguments:
   *   - dbuf: local buffer to get data to
   *   - sbuf: virtual address of the remote buffer to get data. It is
   *           exchanged via isendCtrl|irecvCtrl called by the algorithm
   *           layer.
   *   - len: number of bytes
   *   - peerRank: the rank of the remote peer in the current communicator
   *   - config: the config of the iget for the backends, it may include
   *        - shdl: the handle of the source buffer
   *        - remoteAccessKey: the remote access key of the remote buffer
   *        - kernElem: [TODO] the kernel element to be used for a NVL get, not
   * used yet since NVL get is not supported yet.
   *        - ibFastPath: whether use fast path for ib get.
   * Output arguments:
   *   - req: the request object to track the progress of the iget
   */
  // ** NOTICE **
  // Upon ibFastPath=true, we immediately post a single RDMA_READ on
  // data QP 0.
  //
  // To be eligible for the fast path, the following conditions must be met,
  // otherwise the get will return error:
  // 1. There is no outstanding or pending regular get requests (there can be
  // outstanding fast get requests)
  // 2. The message size is less than or equal to the maximum WQE size so the
  // get can fit in a single WQE (the maximum WQE size is currently equal to
  // NCCL_CTRAN_IB_QP_SCALING_THRESHOLD)
  // 3. The number of pending WQEs on the fast QP is less than the maximum
  // number of QP messages (NCCL_CTRAN_IB_QP_MAX_MSGS)
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iget(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      int peerRank,
      CtranMapperConfig config,
      CtranMapperRequest** req) {
    *req = new CtranMapperRequest();
    return igetImpl<PerfConfig>(sbuf, dbuf, len, peerRank, config, *req);
  }

  inline commResult_t atomicSet(
      void* dbuf,
      uint64_t val,
      int peerRank,
      CtranMapperConfig config,
      CtranMapperRequest* req) {
    return atomicSetImpl(dbuf, val, peerRank, config, req);
  }

  /* Post a Flush op to ensure data received from the network is visible from
   * all GPU threads.
   * Input arguments:
   *   - buf: the local buffer to flush
   *   - regHdl: the local registration handle of the local buffer
   * Output arguments:
   *   - req: the request object to track the progress of the flush.
   */
  inline commResult_t
  iflush(const void* buf, const void* regHdl, CtranMapperRequest** req) {
    if (!this->ctranIb) {
      CLOGF(WARN, "CTRAN-MAPPER: IB backend not enabled, skip flush");
      return commSuccess;
    }

    CtranIbRequest* ibReq = nullptr;
    if (req) {
      *req = new CtranMapperRequest();
      ibReq = &((*req)->ibReq);
    }
    FB_COMMCHECK(this->ctranIb->iflush(buf, regHdl, ibReq));
    return commSuccess;
  }

  /* Initialize the notify instance for receiving remote notification from a
   * peer rank. The backend of flag is determined based on the peer's backend
   * and the local receive buffer. NVL-based notify instance will be initialized
   * only when the peer backend includes NVL, and the local receive buffer is
   * allocated by cumem. Otherwise, the notify instance will be initialized as
   * via IB.
   * Input arguments:
   *   - peerRank: the rank of the peer to send the notification
   *   - recvHdl: the handle of the local receive buffer
   *   - kernElem: the kernel element to be used for a NVL notify
   *   - notifyCnt: the number of notifies to receive from the peer
   * Output arguments:
   *   - notify: the notify instance to be initialized
   */
  inline commResult_t initNotify(
      int peerRank,
      void* recvHdl,
      KernelElem* kernElem,
      CtranMapperNotify* notify,
      int notifyCnt = 1) {
    return initNotifyImpl(peerRank, recvHdl, kernElem, notify, notifyCnt);
  }

  /* Wrapper of initNotify with nullptr kernElem for explicitly initializing IB
   * notify. I.e., the notify instance will be initialized as via IB even if the
   * peer can be connected via NVL.
   */
  inline commResult_t initNotify(
      int peerRank,
      void* recvHdl,
      CtranMapperNotify* notify,
      int notifyCnt = 1) {
    return initNotify(peerRank, recvHdl, nullptr, notify, notifyCnt);
  }

  /* Initialize the notify instances for receiving remote notification from
   * peer ranks using IB backend. Input arguments:
   *   - peerRanks: the peer ranks to send the notification
   *   - notifies: the notify instance to be initialized
   */
  inline commResult_t initNotifyBatchIB(
      const std::vector<int>& peerRanks,
      std::vector<CtranMapperNotify>& notifies) {
    return initNotifyBatchImplIB(peerRanks, notifies);
  }

  /* Check the notification from a peer rank, i.e. the completion of any
   * outstanding put. This will check all backend transports.
   * Input arguments:
   *   - notify: the notify instance to check the notification
   * Output arguments:
   *   - done: whether the notification has arrived
   */
  inline commResult_t checkNotify(CtranMapperNotify* notify, bool* done) {
    return checkNotifyImpl(notify, done);
  }

  /* Waiting for the notification from a peer rank. This will wait for all
   * backend transports. Input arguments:
   *   - notify: the notify instance to wait for the notification
   */
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t waitNotify(CtranMapperNotify* notify) {
    return waitNotifyImpl<PerfConfig>(notify);
  }

  /* Waiting for the notification from a vector of notify instances. This will
   * wait for all backend transports. Input arguments:
   *   - notifies: the notify instances to wait for the notification
   */
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t waitAllNotifies(
      std::vector<CtranMapperNotify>& notifies) {
    return waitAllNotifiesImpl<PerfConfig>(notifies);
  }

  /* Test completion of a request. Input arguments:
   *   - req: the request to test
   *   - isComplete: whether the request is completed
   */
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t testRequest(CtranMapperRequest* req, bool* isComplete) {
    return testRequestImpl<PerfConfig>(req, isComplete);
  }

  /* Wait for the completion of a request. Input arguments:
   *   - req: the request to wait
   */
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t waitRequest(CtranMapperRequest* req) {
    return waitRequestImpl<PerfConfig>(req);
  }

  /* Test completion of a vector of requests. Any completed requests will be
   * removed from the vector. User can query the size of the vector to check the
   * number of completed requests. Input arguments:
   *   - reqs: a vector of request objects to test
   *   - tps: a vector of timestamp points to record the completion time of each
   *          request
   *   - recordTime: whether to record the completion time of each request
   */
  inline commResult_t testSomeRequests(
      std::vector<std::unique_ptr<CtranMapperRequest>>& reqs,
      std::vector<CtranMapperTimestampPoint>& tps,
      bool recordTime = true) {
    return testSomeRequestsImpl(reqs, tps, recordTime);
  }

  inline commResult_t testSomeRequests(
      std::vector<std::unique_ptr<CtranMapperRequest>>& reqs) {
    std::vector<CtranMapperTimestampPoint> dummyTps;
    return this->testSomeRequests(reqs, dummyTps, false);
  }

  /* Wait for completion of a vector of requests. Return when all requests have
   * completed. Input arguments:
   *   - reqs: a vector of request objects to wait
   *   - tps: a vector of timestamp points to record the completion time of each
   *          request. If nullptr, the completion time will not be recorded.
   */
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t waitAllRequests(
      std::vector<CtranMapperRequest>& reqs,
      std::vector<CtranMapperTimestampPoint>* tps) {
    return waitAllRequestsImpl<PerfConfig>(reqs, tps);
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t waitAllRequests(std::vector<CtranMapperRequest>& reqs) {
    return this->waitAllRequests<PerfConfig>(reqs, nullptr);
  }

  /* Test notification of a vector of peers. Any completed notification will be
   * removed from the vector. User can query the size of the vector to check the
   * number of completed notification. Input arguments:
   *   - notifyVec: a vector of notify instances to check the notification
   */
  inline commResult_t checkSomeNotify(
      std::vector<std::unique_ptr<CtranMapperNotify>>& notifyVec) {
    for (auto it = notifyVec.begin(); it != notifyVec.end();) {
      auto& notify = *it;
      bool completed = false;
      FB_COMMCHECK(this->checkNotify(notify.get(), &completed));
      if (completed) {
        // Remove completed notify peer
        it = notifyVec.erase(it);
      } else {
        it++;
      }
    }
    return commSuccess;
  }

  /* For certain transports, such as TCP Device Memory, incoming data is
   * received into a bounce buffer in an out-of-order scatter-gather manner.
   * This requires GPU-to-GPU reassembly, known as "unpacking," to organize
   * the data into its final destination. This method prepares the producer
   * of the scatter-gather chunks.
   */
  template <typename T = OpElem*>
  commResult_t prepareUnpackConsumer(
      SQueues* sqs,
      size_t blocks,
      const std::vector<T>& opGroup,
      KernelConfig& config) {
    if (this->ctranTcpDm != nullptr) {
      auto ret = this->ctranTcpDm->prepareUnpackConsumer(
          sqs, blocks, &config.unpackPool);
      for (auto& op : opGroup) {
        op->unpackPool = config.unpackPool;
      }
      return ret;
    }
    return commSuccess;
  }

  commResult_t teardownUnpackConsumer(void* pool) {
    if (this->ctranTcpDm != nullptr) {
      return this->ctranTcpDm->teardownUnpackConsumer(pool);
    }
    return commSuccess;
  }

  /* report the Ctran profiling results
   * Input arguments:
   *   - flush: force flushing the profiling result
   */
  void reportProfiling(bool flush = false);

  /* Get the available backend that supports the peer rank
   * Input arguments:
   *   - rank: the rank of the peer
   * Output arguments:
   *   - backend: the backend that supports the peer
   */
  CtranMapperBackend getBackend(int rank);

  /* Check whether the local rank has a valid backend to communicate with all
   * other peers in the mapper associated communicator. The check excludes the
   * local rank itself, since self copy doesn't require any backend. Output
   * arguments:
   *   - hasBackend: true if all peers have valid backends; otherwise false
   */
  bool hasBackend();

  /* Check whether the local rank has the specified backend to communicate with
   * the given peer ranks.
   * Input arguments:
   *   - rank: the rank of the peer
   *   - specified: the specified backend to check
   * Output arguments:
   *   - hasBackend: true if the specified backend is available; otherwise false
   */
  bool hasBackend(int rank, CtranMapperBackend specified);

  /* Some backends, notably NVL and TCPDM, require notifiers on the receiver
   * side. This method indicates whether the notifier is needed for
   * the specified peer rank.
   */
  inline bool requiresRecvNotify(int rank) {
    return getBackend(rank) == CtranMapperBackend::NVL ||
        getBackend(rank) == CtranMapperBackend::TCPDM;
  }

  /* Some backends, notably TCPDM, don't post the notifier elements
   * to the GPE thread and use the notifiers only for the receive-side
   * accounting. This method indicates whether to add KernelElem
   * to the GPE wait list.
   */
  inline bool requiresPostRecvNotify(int rank) {
    return getBackend(rank) != CtranMapperBackend::TCPDM;
  }

  inline void setContext(CtranMapperContext mapperContext) {
    this->context = std::move(mapperContext);
  }

  inline const CtranMapperContext& getContext() const {
    return context;
  }

  CtranIb* ctranIbPtr();

  CtranSocket* ctranSockPtr();

  ctran::IpcRegCache* ipcRegCachePtr();

  // number of iput requests made for each backend
  std::vector<int> iPutCount;
  std::vector<int> iGetCount;

  // number of iCopy
  int iCopyCount{0};

  int rank;
  struct CommLogData logMetaData_;
  std::vector<std::unique_ptr<CtranMapperTimestamp>> timestamps;
  CtranMapperContext context;

  std::shared_ptr<ncclx::colltrace::MapperTrace> mapperTrace{nullptr};

  static std::string inline backendToStr(const CtranMapperBackend backend) {
    switch (backend) {
      case CtranMapperBackend::IB:
        return "IB";
      case CtranMapperBackend::NVL:
        return "NVL";
      case CtranMapperBackend::SOCKET:
        return "SOCKET";
      case CtranMapperBackend::TCPDM:
        return "TCPDM";
      default:
        return "UNKNOWN";
    }
  }

  // Dump exported registration cache, for testing only
  std::unordered_map<ctran::regcache::RegElem*, std::unordered_set<int>>
  dumpExportRegCache() const;

 protected:
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t progress() {
    if (this->ctranIb != nullptr) {
      FB_COMMCHECK(this->ctranIb->progress<PerfConfig>());
    } else if (this->ctranSock != nullptr) {
      FB_COMMCHECK(this->ctranSock->progress());
    } else if (this->ctranTcpDm != nullptr) {
      FB_COMMCHECK(this->ctranTcpDm->progress());
    }

    // Check if any posted IPC release requests are completed and cleanup
    for (auto it = this->postedCbCtrlReqs_.begin();
         it != this->postedCbCtrlReqs_.end();) {
      auto& req = *it;
      if (req->completed.load()) {
        it = this->postedCbCtrlReqs_.erase(it);
      } else {
        it++;
      }
    }

    return commSuccess;
  }

 private:
  inline commResult_t checkComplete(CtranMapperRequest* req, bool* isComplete) {
    if (req->isComplete()) {
      *isComplete = true;
      return commSuccess;
    }
    *isComplete = false;
    switch (req->type) {
        // Whenver a mapper request is created for SEND_CTRL, RECV_CTRL, or
        // IB_PUT, it relies on the internal IB request to track completion.
      case CtranMapperRequest::ReqType::SEND_CTRL:
      case CtranMapperRequest::ReqType::IB_PUT:
      case CtranMapperRequest::ReqType::IB_GET:
      case CtranMapperRequest::ReqType::ATOMIC_SET:
      case CtranMapperRequest::ReqType::TCPDM_PUT:
      case CtranMapperRequest::ReqType::SEND_SYNC_CTRL:
      case CtranMapperRequest::ReqType::RECV_SYNC_CTRL:
      case CtranMapperRequest::ReqType::SEND_CTRL_MSG:
      case CtranMapperRequest::ReqType::RECV_CTRL_MSG:
        if (req->backend == CtranMapperBackend::IB) {
          *isComplete = req->ibReq.isComplete();
        } else if (req->backend == CtranMapperBackend::SOCKET) {
          *isComplete = req->sockReq.isComplete();
        } else {
          *isComplete = req->tcpDmReq.isComplete();
        }
        break;
      case CtranMapperRequest::ReqType::RECV_CTRL:
        if (req->backend == CtranMapperBackend::IB) {
          *isComplete = req->ibReq.isComplete();
        } else if (req->backend == CtranMapperBackend::SOCKET) {
          *isComplete = req->sockReq.isComplete();
        } else {
          *isComplete = req->tcpDmReq.isComplete();
        }
        // TCP device memory does not export receiver's memory to the sender.
        if (*isComplete && req->backend != CtranMapperBackend::TCPDM) {
          FB_COMMCHECK(importMem(
              req->peer,
              req->recvCtrl.msg,
              req->recvCtrl.buf,
              req->recvCtrl.key));
        }
        break;
      case CtranMapperRequest::ReqType::NVL_PUT: {
        // Kernel handles data copy and notify
        *isComplete = req->getConfig().kernElem_->isComplete();
        break;
      }
      case CtranMapperRequest::ReqType::COPY:
        if (req->workStream.has_value()) {
          auto cudaErr = cudaStreamQuery(req->workStream.value());
          if (cudaErr == cudaSuccess) {
            *isComplete = true;
          } else if (cudaErr != cudaErrorNotReady) {
            CLOGF(
                ERR,
                "CTRAN: cudaStreamQuery returned error '{}'",
                cudaGetErrorString(cudaErr));
            return commSystemError;
          }
        } else {
          *isComplete = true;
        }
        break;
    }

    if (*isComplete) {
      req->setComplete();
      CLOGF_TRACE(
          COLL,
          "CTRAN-MAPPER: request {} completed, reqType {}, peer {}",
          (void*)this,
          getReqTypeStr(req->type),
          req->peer);
      if (this->mapperTrace) {
        this->mapperTrace->recordMapperEvent(
            ncclx::colltrace::MapperRequestEnd{.req = req});
      }
    }
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t waitNotifyImpl(CtranMapperNotify* notify) {
    if (notify->backend == CtranMapperBackend::NVL) {
      while (!notify->kernElem->isComplete() && !comm->testAbort()) {
      }
    } else if (notify->backend == CtranMapperBackend::IB) {
      FB_COMMCHECK(this->ctranIb->waitNotify<PerfConfig>(
          notify->peer, notify->notifyCnt));
    } else if (notify->backend == CtranMapperBackend::TCPDM) {
      // TODO(T239012482): enable and test TCPDM FT
      while (!notify->tcpDmReq.isComplete()) {
        FB_COMMCHECK(this->ctranTcpDm->progress());
      }
    } else {
      CLOGF(ERR, "CTRAN-MAPPER: unexpected backend {}", notify->backend);
      return commInternalError;
    }

    if (comm->testAbort()) {
      // TODO(T238821628): re-evaluate error code
      throw ctran::utils::Exception("comm aborted", commRemoteError);
    }

    CLOGF_TRACE(
        COLL, "CTRAN-MAPPER: check notify({}) completed", notify->toString());
    if (this->mapperTrace) {
      this->mapperTrace->recordMapperEvent(
          ncclx::colltrace::RecvNotified{.peerRank = notify->peer});
    }
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t waitAllNotifiesImpl(
      std::vector<CtranMapperNotify>& notifies) {
    for (auto it = notifies.begin(); it != notifies.end(); it++) {
      auto& notify = *it;
      FB_COMMCHECK(waitNotifyImpl<PerfConfig>(&notify));
    }
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t testRequestImpl(
      CtranMapperRequest* req,
      bool* isComplete) {
    FB_COMMCHECK(this->progress<PerfConfig>());
    FB_COMMCHECK(this->checkComplete(req, isComplete));

    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t waitRequestImpl(CtranMapperRequest* req) {
    bool isComplete = false;

    while (!isComplete && !comm->testAbort()) {
      FB_COMMCHECK(testRequestImpl<PerfConfig>(req, &isComplete));
    }

    if (comm->testAbort()) {
      // TODO(T238821628): re-evaluate error code
      throw ctran::utils::Exception("comm aborted", commRemoteError);
    }

    return commSuccess;
  }

  inline commResult_t exportMem(
      int rank,
      const void* buf,
      void* hdl,
      ControlMsg& msg,
      CtranMapperBackend backend = CtranMapperBackend::UNSET) {
    auto regElem = reinterpret_cast<ctran::regcache::RegElem*>(hdl);

    // For a NVL peer, send NVL registration if the buffer has been registered
    // as NVL sharable buffer (i.e., allocated by cuMem). Otherwise pass IB
    // registration as fallback.
    if (backend == CtranMapperBackend::UNSET) {
      backend = this->queryPeerBackend(regElem, rank);
    }

    if (backend == CtranMapperBackend::NVL) {
      msg.setType(ControlMsgType::NVL_EXPORT_MEM);
      FB_COMMCHECK(
          ipcRegCache_->exportMem(buf, regElem->ipcRegElem, msg.ipcDesc));

      // Record the exported remote rank to notify at deregistration
      exportRegCache_.wlock()->record(regElem, rank);

    } else if (backend == CtranMapperBackend::IB) {
      FB_COMMCHECK(CtranIb::exportMem(buf, regElem->ibRegElem, msg));
    } else if (backend == CtranMapperBackend::TCPDM) {
      // No need to export the buffers, TCP device memory is steered by
      // the receiver.
    } else {
      CLOGF(
          ERR,
          "CTRAN-MAPPER: Cannot export buffer {} to rank {}. The rank may not be "
          "reachable by any backend, or buffer type is not supported.",
          (void*)buf,
          rank);
      return commInvalidUsage;
    }
    return commSuccess;
  }

  inline commResult_t importMem(
      int rank,
      const ControlMsg& msg,
      void** buf,
      CtranMapperRemoteAccessKey* remKey) {
    switch (msg.type) {
      case ControlMsgType::IB_EXPORT_MEM:
        if (!this->ctranIb) {
          CLOGF(
              ERR,
              "CTRAN-MAPPER: IB backend is disabled but received unexpected internal control msg ({})",
              msg.toString());
          return commInternalError;
        }
        remKey->backend = CtranMapperBackend::IB;
        FB_COMMCHECK(CtranIb::importMem(buf, &(remKey->ibKey), msg));
        break;
      case ControlMsgType::NVL_EXPORT_MEM: {
        if (!this->ctranNvl) {
          CLOGF(
              ERR,
              "CTRAN-MAPPER: NVL backend is disabled but received unexpected internal control msg ({})",
              msg.toString());
          return commInternalError;
        }
        remKey->backend = CtranMapperBackend::NVL;
        const std::string peerId = comm->statex_->gPid(rank);
        FB_COMMCHECK(ipcRegCache_->importMem(
            peerId, msg.ipcDesc, buf, &(remKey->nvlKey)));
        break;
      }
      default:
        CLOGF(
            ERR,
            "CTRAN-MAPPER: Received unexpected control message type {}",
            msg.type);
        return commInternalError;
    }
    return commSuccess;
  }

  inline CtranMapperBackend queryPeerBackend(
      ctran::regcache::RegElem* regElem,
      int rank) {
    if (this->ctranNvl && this->ctranNvl->isSupported(rank) &&
        regElem->ipcRegElem) {
      return CtranMapperBackend::NVL;
    } else if (this->ctranIb && regElem->ibRegElem) {
      return CtranMapperBackend::IB;
    } else if (this->ctranTcpDm && regElem->tcpRegElem) {
      return CtranMapperBackend::TCPDM;
    }
    return CtranMapperBackend::UNSET;
  }

  inline CtranMapperBackend getCtrlBackend(
      CtranMapperBackend backend = CtranMapperBackend::UNSET) {
    if (backend != CtranMapperBackend::UNSET) {
      return backend;
    }

    if (this->ctranIb) {
      return CtranMapperBackend::IB;
    } else if (this->ctranSock) {
      return CtranMapperBackend::SOCKET;
    } else if (this->ctranTcpDm) {
      return CtranMapperBackend::TCPDM;
    }

    CLOGF(ERR, "No backend is available, please specify at least one backend.");

    return CtranMapperBackend::UNSET;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t isendCtrlImpl(
      const void* buf,
      void* hdl,
      int peerRank,
      CtranMapperRequest* req,
      CtranMapperBackend backend = CtranMapperBackend::UNSET) {
    req->type = CtranMapperRequest::ReqType::SEND_CTRL;
    req->peer = peerRank;
    req->backend = getCtrlBackend(backend);
    FB_COMMCHECK(
        this->exportMem(peerRank, buf, hdl, req->sendCtrl.msg, backend));
    auto& msg = req->sendCtrl.msg;
    if (this->mapperTrace) {
      this->mapperTrace->recordMapperEvent(
          ncclx::colltrace::SendCtrlStart{
              .buffer = const_cast<void*>(buf),
              .mapperHandle = hdl,
              .peerRank = peerRank,
              .req = req});
    }
    CLOGF_TRACE(
        COLL,
        "CTRAN-MAPPER: Post {} SEND ctrlmsg to rank {} with req {} {} {}: {}",
        ctranIb ? "IB" : "SOCKET",
        peerRank,
        (void*)req,
        ctranIb ? "ibReq " : "sockReq ",
        ctranIb ? (void*)&req->ibReq : (void*)&req->sockReq,
        msg.toString());
    if (ctranIb) {
      return ctranIb->isendCtrlMsg<PerfConfig>(
          msg.type, &msg, sizeof(ControlMsg), peerRank, req->ibReq);
    } else if (ctranSock) {
      return ctranSock->isendCtrlMsg(msg, peerRank, req->sockReq);
    } else {
      return ctranTcpDm->isendCtrlMsg(msg, peerRank, req->tcpDmReq);
    }
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t irecvCtrlImpl(
      void** buf,
      struct CtranMapperRemoteAccessKey* key,
      int peerRank,
      CtranMapperRequest* req) {
    req->type = CtranMapperRequest::ReqType::RECV_CTRL;
    req->peer = peerRank;
    req->backend = getCtrlBackend();
    req->recvCtrl.msg.setType(ControlMsgType::UNSPECIFIED);
    req->recvCtrl.buf = buf;
    req->recvCtrl.key = key;
    auto& msg = req->recvCtrl.msg;
    if (this->mapperTrace) {
      this->mapperTrace->recordMapperEvent(
          ncclx::colltrace::RecvCtrlStart{
              .recvBufferPtr = buf,
              .accessKeyPtr = key,
              .peerRank = peerRank,
              .req = req});
    }
    CLOGF_TRACE(
        COLL,
        "CTRAN-MAPPER: Post {} RECV ctrlmsg from rank {} with req {} {} {}: {}",
        ctranIb ? "IB" : "SOCKET",
        peerRank,
        (void*)req,
        ctranIb ? "ibReq " : "sockReq ",
        ctranIb ? (void*)&req->ibReq : (void*)&req->sockReq,
        msg.toString());
    if (ctranIb) {
      return ctranIb->irecvCtrlMsg<PerfConfig>(
          &msg, sizeof(msg), peerRank, req->ibReq);
    } else if (ctranSock) {
      return ctranSock->irecvCtrlMsg(msg, peerRank, req->sockReq);
    } else {
      return ctranTcpDm->irecvCtrlMsg(msg, peerRank, req->tcpDmReq);
    }
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t isendCtrlMsgImpl(
      const void* payload,
      const size_t size,
      int peerRank,
      CtranMapperRequest* req,
      CtranMapperBackend backend = CtranMapperBackend::UNSET) {
    req->type = CtranMapperRequest::ReqType::SEND_CTRL_MSG;
    req->peer = peerRank;
    req->backend =
        ctranIb ? CtranMapperBackend::IB : CtranMapperBackend::SOCKET;
    if (this->mapperTrace) {
      this->mapperTrace->recordMapperEvent(
          ncclx::colltrace::SendCtrlStart{
              .buffer = const_cast<void*>(payload),
              .peerRank = peerRank,
              .req = req});
    }
    CLOGF_TRACE(
        COLL,
        "CTRAN-MAPPER: Post {} SEND msg to rank {} with req {} {} {}: {}",
        ctranIb ? "IB" : "SOCKET",
        peerRank,
        (void*)req,
        ctranIb ? "ibReq " : "sockReq ",
        ctranIb ? (void*)&req->ibReq : (void*)&req->sockReq,
        payload);
    if (ctranIb) {
      return ctranIb->isendCtrlMsg(
          ControlMsgType::SYNC, payload, size, peerRank, req->ibReq);
    }
    // only support ib for now
    return commInvalidArgument;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t irecvCtrlMsgImp(
      void* payload,
      const size_t size,
      int peerRank,
      CtranMapperRequest* req,
      CtranMapperBackend backend = CtranMapperBackend::UNSET) {
    req->type = CtranMapperRequest::ReqType::RECV_CTRL_MSG;
    req->peer = peerRank;
    req->backend =
        ctranIb ? CtranMapperBackend::IB : CtranMapperBackend::SOCKET;
    if (this->mapperTrace) {
      this->mapperTrace->recordMapperEvent(
          ncclx::colltrace::RecvCtrlStart{
              .recvBufferPtr = (void**)&payload,
              .peerRank = peerRank,
              .req = req});
    }
    CLOGF_TRACE(
        COLL,
        "CTRAN-MAPPER: Post {} RECV msg from rank {} with req {} {} {}: {}",
        ctranIb ? "IB" : "SOCKET",
        peerRank,
        (void*)req,
        ctranIb ? "ibReq " : "sockReq ",
        ctranIb ? (void*)&req->ibReq : (void*)&req->sockReq,
        payload);
    if (ctranIb) {
      return ctranIb->irecvCtrlMsg<PerfConfig>(
          payload, size, peerRank, req->ibReq);
    }
    return commInvalidArgument;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t isendCtrlBatchImplIB(
      const std::vector<void*>& bufs,
      void* hdl,
      const std::vector<int>& peerRanks,
      std::vector<CtranMapperRequest>& reqs) {
    if (peerRanks.empty()) {
      return commSuccess;
    }

    // Use first request to export the contiguous memory
    auto& sendCtrlMsg = reqs.front().sendCtrl.msg;
    int idx = 0;
    FB_COMMCHECK(this->exportMem(
        peerRanks[0], bufs[0], hdl, sendCtrlMsg, CtranMapperBackend::IB));
    for (auto peer : peerRanks) {
      auto& req = reqs[idx++];
      req.sendCtrl.msg = sendCtrlMsg;
      auto& msg = req.sendCtrl.msg;
      req.peer = peer;
      msg.ibExp.remoteAddr = reinterpret_cast<uint64_t>(bufs[peer]);
      msg.aux = reqs[idx - 1].aux;
      CLOGF_TRACE(
          COLL,
          "CTRAN-MAPPER: Post SEND ctrlmsg to rank {} with req {} ibReq {}: {}",
          req.peer,
          (void*)&req,
          (void*)&req.ibReq,
          msg.toString());
      if (this->mapperTrace) {
        this->mapperTrace->recordMapperEvent(
            ncclx::colltrace::SendCtrlStart{
                .buffer = const_cast<void*>(bufs[req.peer]),
                .mapperHandle = hdl,
                .peerRank = req.peer,
                .req = &req});
      }
      this->ctranIb->isendCtrlMsg<PerfConfig>(
          msg.type, &msg, sizeof(ControlMsg), req.peer, req.ibReq);
    }

    return commSuccess;
  }

  inline commResult_t isendCtrlImpl(int peerRank, CtranMapperRequest* req) {
    req->type = CtranMapperRequest::ReqType::SEND_SYNC_CTRL;
    req->peer = peerRank;
    req->backend =
        ctranIb ? CtranMapperBackend::IB : CtranMapperBackend::SOCKET;
    auto& msg = req->sendSyncCtrl.msg;
    msg.setType(ControlMsgType::SYNC);

    if (this->mapperTrace) {
      this->mapperTrace->recordMapperEvent(
          ncclx::colltrace::SendSyncCtrlStart{
              .peerRank = peerRank, .req = req});
    }
    CLOGF_TRACE(
        COLL,
        "CTRAN-MAPPER: Post {} SEND(SYNC) ctrlmsg to rank {} with req {} {} {}: {}",
        ctranIb ? "IB" : "SOCKET",
        peerRank,
        (void*)req,
        ctranIb ? "ibReq " : "sockReq ",
        ctranIb ? (void*)&req->ibReq : (void*)&req->sockReq,
        msg.toString());
    if (ctranIb) {
      return ctranIb->isendCtrlMsg(
          msg.type, &msg, sizeof(ControlMsg), peerRank, req->ibReq);
    } else {
      return ctranSock->isendCtrlMsg(msg, peerRank, req->sockReq);
    }
  }

  inline commResult_t irecvCtrlImpl(int peerRank, CtranMapperRequest* req) {
    req->type = CtranMapperRequest::ReqType::RECV_SYNC_CTRL;
    req->peer = peerRank;
    req->backend =
        ctranIb ? CtranMapperBackend::IB : CtranMapperBackend::SOCKET;
    auto& msg = req->recvSyncCtrl.msg;
    msg.setType(ControlMsgType::SYNC);

    if (this->mapperTrace) {
      this->mapperTrace->recordMapperEvent(
          ncclx::colltrace::RecvSyncCtrlStart{
              .peerRank = peerRank, .req = req});
    }
    CLOGF_TRACE(
        COLL,
        "CTRAN-MAPPER: Post {} RECV(SYNC) ctrlmsg from rank {} with req {} {} {}: {}",
        ctranIb ? "IB" : "SOCKET",
        peerRank,
        (void*)req,
        ctranIb ? "ibReq " : "sockReq ",
        ctranIb ? (void*)&req->ibReq : (void*)&req->sockReq,
        msg.toString());
    if (ctranIb) {
      return ctranIb->irecvCtrlMsg(&msg, sizeof(msg), peerRank, req->ibReq);
    } else {
      return ctranSock->irecvCtrlMsg(msg, peerRank, req->sockReq);
    }
  }

  // allow put messages to be stack allocated
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iputBatchImpl(
      const std::vector<CtranMapperPutMsg>& puts,
      int peerRank) {
    if (this->ctranIb == nullptr) {
      CLOGF(ERR, "CTRAN-MAPPER: ctranIB is null in batch put.");
      return commInternalError;
    }
    std::vector<PutIbMsg> msgs;
    msgs.reserve(puts.size());
    for (auto& put : puts) {
      const auto& remoteAccessKey = put.config.remoteAccessKey_;
      if (remoteAccessKey.backend != CtranMapperBackend::IB) {
        CLOGF(
            ERR,
            "CTRAN-MAPPER: Unsupported remote access key backend {}",
            backendToStr(remoteAccessKey.backend));
        return commInternalError;
      }

      const auto shdl = put.config.memHdl_;
      const auto kernElem = put.config.kernElem_;
      if (put.req != nullptr) {
        put.req->type = CtranMapperRequest::ReqType::IB_PUT;
        put.req->peer = peerRank;
        put.req->backend = CtranMapperBackend::IB;
        put.req->setConfig(put.config);
      }

      struct ctran::regcache::RegElem* regElem =
          reinterpret_cast<struct ctran::regcache::RegElem*>(shdl);
      auto regLk = regElem->stateMnger.rlock();
      CLOGF_TRACE(
          COLL,
          "CTRAN-MAPPER: Post IB PUT to rank {}: sbuf {} -> dbuf {} len {} kernElem {}",
          peerRank,
          put.sbuf,
          put.dbuf,
          put.len,
          (void*)kernElem);
      iPutCount[CtranMapperBackend::IB]++;
      auto msg = PutIbMsg{
          .sbuf = const_cast<void*>(put.sbuf),
          .dbuf = put.dbuf,
          .len = put.len,
          .ibRegElem = regElem->ibRegElem,
          .remoteAccessKey = remoteAccessKey.ibKey,
          .notify = put.config.notify_,
          .config = put.config.ibConfig_,
          .req = put.req == nullptr ? nullptr : &(put.req->ibReq),
          .ibFastPath = put.config.ibFastPath_,
      };
      msgs.emplace_back(std::move(msg));

      if (this->mapperTrace) {
        this->mapperTrace->recordMapperEvent(
            ncclx::colltrace::PutStart{
                .sendBuffer = const_cast<void*>(put.sbuf),
                .remoteBuffer = put.dbuf,
                .length = put.len,
                .peerRank = peerRank,
                .sourceHandle = shdl,
                .remoteAccessKey = remoteAccessKey,
                .req = put.req});
      }
    }

    FB_COMMCHECK(
        this->ctranIb->iputBatch<PerfConfig>(std::move(msgs), peerRank));
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iputImpl(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      int peerRank,
      CtranMapperConfig config,
      CtranMapperRequest* req) {
    const auto shdl = config.memHdl_;
    const auto kernElem = config.kernElem_;
    const auto& remoteAccessKey = config.remoteAccessKey_;
    const auto& notify = config.notify_;
    const auto& ibConfig = config.ibConfig_;
    if (remoteAccessKey.backend == CtranMapperBackend::IB ||
        // If kernElem is not provided, falls back to IB
        // NOTE: it requires a match with the receiver side waitNotify, where it
        // should also be initialized with a nullptr kernElem to fallback to IB
        // based path; otherwise it will hang.
        kernElem == nullptr) {
      if (req != nullptr) {
        req->peer = peerRank;
        if (this->ctranIb != nullptr) {
          req->type = CtranMapperRequest::ReqType::IB_PUT;
          req->backend = CtranMapperBackend::IB;
          req->setConfig(config);
        } else {
          req->type = CtranMapperRequest::ReqType::TCPDM_PUT;
          req->backend = CtranMapperBackend::TCPDM;
        }
      }

      struct ctran::regcache::RegElem* regElem =
          reinterpret_cast<struct ctran::regcache::RegElem*>(shdl);
      auto regLk = regElem->stateMnger.rlock();

      if (this->ctranIb != nullptr) {
        CtranIbRequest* ibReqPtr = (req == nullptr ? nullptr : &(req->ibReq));
        CLOGF_TRACE(
            COLL,
            "CTRAN-MAPPER: Post IB PUT to rank {}: sbuf {} -> dbuf {} len {} kernElem {}",
            peerRank,
            sbuf,
            dbuf,
            len,
            (void*)kernElem);
        iPutCount[CtranMapperBackend::IB]++;
        // IB PUT combines data and notify together and tracked by single ibReq
        FB_COMMCHECK(this->ctranIb->iput<PerfConfig>(
            sbuf,
            dbuf,
            len,
            peerRank,
            regElem->ibRegElem,
            remoteAccessKey.ibKey,
            notify,
            ibConfig,
            ibReqPtr,
            config.ibFastPath_));
      } else {
        if (req == nullptr) {
          CLOGF(ERR, "CTRAN-MAPPER: Unsupported TCPDM iput without request");
          return commInternalError;
        }
        if (!notify) {
          CLOGF(ERR, "CTRAN-MAPPER: Unsupported TCPDM iput without notify");
          return commInternalError;
        }

        FB_COMMCHECK(this->ctranTcpDm->iput(
            (void*)sbuf,
            dbuf,
            len,
            peerRank,
            regElem->tcpRegElem,
            notify,
            nullptr,
            &req->tcpDmReq));
      }

      if (kernElem) {
        // NVL peer but failed to import rbuf. Revoke the elem.
        kernElem->revoke();
      }
    } else if (remoteAccessKey.backend == CtranMapperBackend::NVL) {
      iPutCount[CtranMapperBackend::NVL]++;
      CLOGF_TRACE(
          COLL,
          "CTRAN-MAPPER: Post NVL PUT to rank {}: sbuf {} -> dbuf {} len {}",
          peerRank,
          sbuf,
          dbuf,
          len);

      if (!kernElem) {
        CLOGF(
            ERR,
            "CTRAN-MAPPER: NVL PUT requires valid kernElem. It indicates a COMM internal bug");
        return commInternalError;
      }
      if (req != nullptr) {
        req->type = CtranMapperRequest::ReqType::NVL_PUT;
        req->peer = peerRank;
        req->setConfig(config);
      }

      // Assigned remote recvbuff addr and post to kernel
      // Expect other info has been set before launching kernel
      kernElem->putNotify.recvbuff = reinterpret_cast<uint64_t>(dbuf);
      kernElem->putNotify.notify = notify;
      kernElem->post();
    } else {
      auto backendStr = backendToStr(remoteAccessKey.backend);
      CLOGF(
          ERR,
          "CTRAN-MAPPER: Unsupported remote access key backend {}",
          backendStr.c_str());
      return commInternalError;
    }

    if (this->mapperTrace) {
      this->mapperTrace->recordMapperEvent(
          ncclx::colltrace::PutStart{
              .sendBuffer = const_cast<void*>(sbuf),
              .remoteBuffer = dbuf,
              .length = len,
              .peerRank = peerRank,
              .sourceHandle = shdl,
              .remoteAccessKey = remoteAccessKey,
              .req = req});
    }
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t igetImpl(
      const void* sbuf, // remote buf
      void* dbuf, // local buf
      std::size_t len,
      int peerRank,
      CtranMapperConfig config,
      CtranMapperRequest* req) {
    const auto shdl = config.memHdl_;
    const auto& remoteAccessKey = config.remoteAccessKey_;
    const auto& ibConfig = config.ibConfig_;
    if (remoteAccessKey.backend == CtranMapperBackend::IB) {
      if (req != nullptr) {
        if (this->ctranIb != nullptr) {
          req->type = CtranMapperRequest::ReqType::IB_GET;
          req->peer = peerRank;
          req->backend = CtranMapperBackend::IB;
          req->setConfig(config);
        } else {
          CLOGF(
              ERR, "CTRAN-MAPPER: Unsupported backend iget without IB backend");
          return commInternalError;
        }
      }

      struct ctran::regcache::RegElem* regElem =
          reinterpret_cast<struct ctran::regcache::RegElem*>(shdl);
      auto regLk = regElem->stateMnger.rlock();

      if (this->ctranIb != nullptr) {
        CtranIbRequest* ibReqPtr = (req == nullptr ? nullptr : &(req->ibReq));
        CLOGF_TRACE(
            COLL,
            "CTRAN-MAPPER: Post IB Get to rank {}: sbuf {} -> dbuf {} len {}",
            peerRank,
            sbuf,
            dbuf,
            len);
        iGetCount[CtranMapperBackend::IB]++;
        // IB GET combines data and notify together and tracked by single ibReq
        FB_COMMCHECK(this->ctranIb->iget<PerfConfig>(
            sbuf,
            dbuf,
            len,
            peerRank,
            regElem->ibRegElem,
            remoteAccessKey.ibKey,
            ibConfig,
            ibReqPtr,
            config.ibFastPath_));
      } else {
        CLOGF(
            ERR,
            "CTRAN-MAPPER: IB backend required for iget operation, but not available.");

        return commInternalError;
      }
    } else {
      auto backendStr = backendToStr(remoteAccessKey.backend);
      CLOGF(
          ERR,
          "CTRAN-MAPPER: Unsupported remote access key backend {}",
          backendStr.c_str());
      return commInternalError;
    }

    // TODO: add collTrace Support
    return commSuccess;
  }

  inline commResult_t atomicSetImpl(
      void* dbuf,
      uint64_t val,
      int peerRank,
      CtranMapperConfig config,
      CtranMapperRequest* req) {
    const auto& remoteAccessKey = config.remoteAccessKey_;
    if (remoteAccessKey.backend != CtranMapperBackend::IB) {
      CLOGF(ERR, "CTRAN-MAPPER: atomicSet only supports IB backend");
      return commInternalError;
    }
    if (req != nullptr) {
      req->type = CtranMapperRequest::ReqType::ATOMIC_SET;
      req->peer = peerRank;
      req->backend = CtranMapperBackend::IB;
      req->setConfig(config);
    }
    CtranIbRequest* ibReqPtr = (req == nullptr ? nullptr : &(req->ibReq));
    CLOGF_TRACE(
        COLL,
        "CTRAN-MAPPER: Post IB ATOMIC_SET to rank {}: val {} dbuf {}",
        peerRank,
        val,
        dbuf);
    FB_COMMCHECK(this->ctranIb->iatomicSet(
        dbuf, val, peerRank, remoteAccessKey.ibKey, ibReqPtr));
    return commSuccess;
  }

  inline commResult_t initNotifyImpl(
      int peerRank,
      void* recvHdl,
      KernelElem* kernElem,
      CtranMapperNotify* notify,
      int notifyCnt) {
    auto regElem = reinterpret_cast<ctran::regcache::RegElem*>(recvHdl);
    auto regLk = regElem->stateMnger.rlock();

    auto backend = this->queryPeerBackend(regElem, peerRank);

    // If kernElem is not provided, falls back to IB
    // NOTE: it requires a match with the sender side put, where it should also
    // pass a nullptr kernElem to fallback to IB based put; otherwise it will
    // hange.
    if (backend == CtranMapperBackend::NVL && !kernElem) {
      backend = CtranMapperBackend::IB;
    } else if (backend == CtranMapperBackend::NVL && kernElem) {
      // Kernel start checking
      kernElem->post();
    } else if (backend == CtranMapperBackend::IB && kernElem) {
      // Revoke elem if switched to use IB for a NVL peer
      kernElem->revoke();
      kernElem = nullptr;
    } else if (backend == CtranMapperBackend::TCPDM) {
      // We are mapping two-sided communications (i.e., send/receive) to
      // one-sided communications (i.e., iPUT/waitNotify). Due to this, we
      // handle TCP differently. Specifically, for iPUT, TCP performs the send
      // operation, and for waitNotify, TCP performs the receive operation.
      // Consequently, we need to progress TCP to ensure it completes the
      // receive operation. Once it does, it will mark tcpDmReq as complete,
      // allowing us to exit the loop.

      if (kernElem == nullptr) {
        CLOGF(ERR, "TCP Device Memory requires kernElem in the notifier");
        return commInternalError;
      }

      const void* rbuff = kernElem->waitNotify.recvbuff;
      size_t len = kernElem->waitNotify.nbytes;

      if (rbuff == nullptr || len == 0) {
        CLOGF(ERR, "TCP Device Memory requires recvbuff in the notifier");
        return commInternalError;
      }

      FB_COMMCHECK(this->ctranTcpDm->irecv(
          peerRank,
          regElem->tcpRegElem,
          (void*)rbuff,
          len,
          notify->tcpDmReq,
          this->context.unpackPool));
    }

    notify->update(peerRank, kernElem, backend, notifyCnt);
    CLOGF_TRACE(
        COLL, "CTRAN-MAPPER: initialized notify {}", notify->toString());
    return commSuccess;
  }

  inline commResult_t initNotifyBatchImplIB(
      const std::vector<int>& peerRanks,
      std::vector<CtranMapperNotify>& notifies) {
    int idx = 0;
    for (auto peer : peerRanks) {
      notifies[idx++].update(peer, nullptr, CtranMapperBackend::IB);
    }
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t checkNotifyImpl(CtranMapperNotify* notify, bool* done) {
    if (notify->backend == CtranMapperBackend::NVL) {
      *done = notify->kernElem->isComplete();
    } else if (notify->backend == CtranMapperBackend::IB) {
      FB_COMMCHECK(this->ctranIb->checkNotify<PerfConfig>(notify->peer, done));
    } else if (notify->backend == CtranMapperBackend::TCPDM) {
      *done = notify->tcpDmReq.isComplete();
    } else {
      CLOGF(ERR, "CTRAN-MAPPER: unexpected backend {}", notify->backend);
      return commInternalError;
    }

    if (*done) {
      CLOGF_TRACE(
          COLL, "CTRAN-MAPPER: check notify({}) completed", notify->toString());
      if (this->mapperTrace) {
        this->mapperTrace->recordMapperEvent(
            ncclx::colltrace::RecvNotified{.peerRank = notify->peer});
      }
    }
    return commSuccess;
  }

  inline commResult_t testSomeRequestsImpl(
      std::vector<std::unique_ptr<CtranMapperRequest>>& reqs,
      std::vector<CtranMapperTimestampPoint>& tps,
      bool recordTime) {
    for (auto it = reqs.begin(); it != reqs.end();) {
      auto& req = *it;
      if (req) {
        // Only icopy request doesn't have peer; expect it is not used in
        // testSomeRequests with timepoints.
        if (req->peer == -1) {
          CLOGF(ERR, "Expect peer is specified in request.");
          return commInternalError;
        }

        bool completed = false;
        FB_COMMCHECK(testRequestImpl(req.get(), &completed));
        if (completed) {
          if (recordTime) {
            tps.emplace_back(req->peer);
          }
          it = reqs.erase(it);
        } else {
          it++;
        }
      } else if (!req) {
        // Remove completed requests
        it = reqs.erase(it);
      }
    }
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t waitAllRequestsImpl(
      std::vector<CtranMapperRequest>& reqs,
      std::vector<CtranMapperTimestampPoint>* tps) {
    for (auto it = reqs.begin(); it != reqs.end(); it++) {
      auto& req = *it;
      if (req.peer == -1) {
        CLOGF(ERR, "Expect peer is specified in request.");
        return commInternalError;
      }
      FB_COMMCHECK(waitRequest<PerfConfig>(&req));
      if (tps) {
        tps->emplace_back(req.peer);
      }
    }
    return commSuccess;
  }

  /* Internal sanity checks to ensure algorithm layers calls ctranMapper
   * correctly. Return commSuccess if yes, otherwise commInternalError.
   */
  inline commResult_t checkValidReq(
      CtranMapperRequest** req,
      const char* fnName) {
    if (req == nullptr) {
      CLOGF(
          ERR,
          "CTRAN-MAPPER: Invalid request pointer. Unexpected {} indicates a COMM internal bug",
          fnName);
      return commInternalError;
    }
    return commSuccess;
  }

  /* Internal sanity checks to ensure algorithm layers calls ctranMapper
   * correctly. Return commSuccess if yes, otherwise commInternalError.
   */
  inline commResult_t checkValidReq(
      CtranMapperRequest* req,
      const char* fnName) {
    if (req == nullptr) {
      CLOGF(
          ERR,
          "CTRAN-MAPPER: Invalid request pointer. Unexpected {} indicates a COMM internal bug",
          fnName);
      return commInternalError;
    }
    return commSuccess;
  }

  commResult_t remReleaseMem(ctran::regcache::RegElem* regElem);

  bool atDestruction{false};

  std::unique_ptr<class CtranIb> ctranIb{nullptr};
  std::unique_ptr<class CtranNvl> ctranNvl{nullptr};
  std::unique_ptr<class CtranSocket> ctranSock{nullptr};
  std::unique_ptr<class ctran::CtranTcpDm> ctranTcpDm{nullptr};
  std::unique_ptr<class CtranCtrlManager> ctrlMgr{nullptr};
  std::unique_ptr<class ctran::IpcRegCache> ipcRegCache_{nullptr};

  // holds enabled backends when the mapper is created.
  // A unified struct for holding all available backends.
  std::vector<bool> enableBackends_{
      std::vector<bool>(CtranMapperBackend::NUM_BACKENDS, false)};

  // List of outstanding internal IPC release requests to be processed
  // when polling progress. We need to temporarily hold the request
  // instance and erase after completion.
  std::deque<std::unique_ptr<ctran::regcache::IpcReqCb>> postedCbCtrlReqs_;

  // Peer IPC server addresses for async socket communication, indexed by rank.
  // Populated via bootstrap allGather during mapper initialization.
  std::vector<sockaddr_storage> peerIpcServerAddrs_;

  // AllGather IPC server addresses from all ranks via bootstrap.
  commResult_t allGatherIpcServerAddrs();

  // Get peer's IPC server address by rank.
  folly::SocketAddress getPeerIpcServerAddr(int rank) const;

  // Record remote ranks that each ipcRegElem has exported to.
  // - For each remote rank, the local rank will send RELEASE_MEM ctrlmsg to
  //   the remote rank at deregMem.
  // - The export cache is maintained per communicator in order to ensure a
  //   valid backend to send RELEASE_MEM ctrlmsg.
  folly::Synchronized<ctran::ExportRegCache> exportRegCache_;

  CtranComm* comm{nullptr};
};

// Convenient RAII class to guard mapper epoch lock.
class CtranMapperEpochRAII {
 public:
  // Allow set mapper to nullptr to skip lock. It can be used when lock is
  // needed only for selected cases.
  explicit CtranMapperEpochRAII(CtranMapper* mapper) : mapper_(mapper) {
    if (mapper_ != nullptr) {
      FB_COMMCHECKTHROW_EX_NOCOMM(mapper_->epochLock());
    }
  }

  ~CtranMapperEpochRAII() {
    if (mapper_ != nullptr) {
      FB_COMMCHECKIGNORE(mapper_->epochUnlock());
    }
  }
  // No copy and move constructor
  CtranMapperEpochRAII(const CtranMapperEpochRAII&) = delete;

 private:
  CtranMapper* mapper_{nullptr};
};

#endif
