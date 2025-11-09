// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/lang/Bits.h>
#include <cstddef>
#include <memory>
#include <vector>

#include "comms/ctran/algos/Broadcast/BroadcastImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/logger/LogUtils.h"

using namespace ctran;

static unsigned int bestThreadBlockSize = 0;
static const auto myAlgo = NCCL_BROADCAST_ALGO::ctbtree;

static inline int getNumGroups(size_t nbytes) {
  // compute needed thread blocks for given bytes
  int nGroups = nbytes / NCCL_CTRAN_NVL_BROADCAST_CHUNK_SIZE;
  return std::min(
      std::max(1, nGroups), // at least 1 thread block
      // not exceed max theshold
      NCCL_CTRAN_NVL_BROADCAST_MAX_NUM_THREAD_BLOCKS);
}

static inline unsigned int getThreadBlockSize() {
  // If first time call, query cuda recommended blockSize
  if (bestThreadBlockSize == 0) {
    int minGridSize = 0;
    FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        (int*)&bestThreadBlockSize,
        reinterpret_cast<const void*>(ncclKernelBroadcast</*UNPACK=*/false>),
        0 /* dynamicSMemSize */,
        0 /* blockSizeLimit */));
  }

  return NCCL_CTRAN_NVL_BROADCAST_THREAD_BLOCK_SIZE == -1
      ? bestThreadBlockSize
      : NCCL_CTRAN_NVL_BROADCAST_THREAD_BLOCK_SIZE;
}

static inline int floorPow2(int x) {
  // Return the greatest power of 2 not-bigger than argument
  // f(x) -> y, where y = 2^i for maximum integer i such that 2^i <= x
  // Example: 001010 -> 001000
  if (x == 0)
    return 0;
  return 1 << (folly::findLastSet(x) - 1);
}

/* Algorithm:
 *   Binomial tree broadcast allows to distribute data in ceil(log(n))
 * communication steps. At the start of the algorithm, only rank 0 has the data.
 *   In the first round, rank 0 sends data to rank 1. In round 2, both rank 0
 * and rank 1 would send the data. This grows exponentially as each round the
 * number of sending nodes is doubled.
 *
 *   Here is an example of data distribution for 8 ranks:
 *   Round 1: only rank_0 has the data; data travels over distance of 1. [0] ->
 * [1] Round 2: ranks 0, 1 have the data; data travels over distance of 2: [0,
 * 1] -> [2, 3] Round 3: ranks 0,1,2,3 have data; data travels over distance of
 * 4: [0,1,2,3] -> [4,5,6,7] There is no explicit synchronization between ranks
 * on each round.
 *
 *   Same routine from POV of each rank:
 *   Given 8 ranks [0, 1, 2, 3, 4, 5, 6, 7] with root rank_0 we would distribute
 * data in the following way: Rank 0 sends data to 1, 2, 4 Rank 1 receives data
 * from 0, sends data to 3, 5 Rank 2 receives data from 1, sends data to 6 Rank
 * 3 receives data from 0, sends data to 7 Ranks 4, 5, 6, 7 receive data from 0,
 * 1, 2, 3.
 *
 *   If root rank is not 0, we virtually renumber the ranks from root rank.
 *   For example, if root rank is 2, in a 8-rank group rank 0 would be 6 ranks
 * far from root.
 */
static inline commResult_t setupPlan(
    CtranComm* comm,
    std::vector<std::unique_ptr<OpElem>>& opGroup,
    KernelConfig& config) {
  const auto statex = comm->statex_.get();
  struct OpElem* op = opGroup.front().get();
  size_t sendSize = op->broadcast.count * commTypeSize(op->broadcast.datatype);
  const int root = op->broadcast.root;
  const int nRanks = statex->nRanks();
  const int myRank = statex->rank();
  int maxNumBlocks = 1;
  // Non-negative distance from the root rank.
  // In a 8-rank group, rank 7 is 5 far from rank 2; rank 2 is 3 far from 7
  // i.e. (7->0->1->2).
  int rootDist = (myRank - root + nRanks) % nRanks;
  int recvOffset = floorPow2(rootDist); // Rank at distance 011011 recv data
                                        // from rank 001011 which is 010000 away
  int sendOffset =
      recvOffset * 2; // Rank at ditance 011011 sends data to rank 111011 which
                      // is 100000 away, then to 1111011 etc.

  auto putNotifyList = CommonList<KernelElem>();
  auto waitNotifyList = CommonList<KernelElem>();

  if (sendSize == 0 || nRanks == 1) {
    return commSuccess;
  }

  KernelElem* elem = nullptr;
  int nGroups = getNumGroups(sendSize);
  // record the max number of thread blocks as final launching grid size
  maxNumBlocks = std::max(maxNumBlocks, nGroups);

  bool hasTcpDmRecv = false;
  if (myRank != root) {
    // Every non-root node receives data from a node up the tree
    int recvFrom = (myRank - recvOffset + nRanks) % nRanks;

    // For TCP Device Memory, if we have peers we are going to receive
    // from, we need to unpack the data from the bounce buffer.
    if (comm->ctran_->mapper->getBackend(recvFrom) ==
        CtranMapperBackend::TCPDM) {
      hasTcpDmRecv = true;
    }

    // Recv config
    if (comm->ctran_->mapper->requiresRecvNotify(op->recv.peerRank)) {
      // only 1 group handles waitNotify elem
      FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, 1, &elem));
      elem->waitNotify.peerLocalRank = statex->localRank(recvFrom);
      elem->waitNotify.recvbuff = op->broadcast.recvbuff;
      elem->waitNotify.nbytes = sendSize;
      // pass the ngroups used by remote put
      elem->waitNotify.ngroups = getNumGroups(sendSize);

      if (comm->ctran_->mapper->requiresPostRecvNotify(op->recv.peerRank)) {
        waitNotifyList.enqueue(elem);
      }
      op->broadcast.waitNotifyMap.insert({recvFrom, elem});
    }

    FB_COMMCHECK(comm->ctran_->mapper->prepareUnpackConsumer(
        &config.args.collective.broadcast.unpack,
        NCCL_CTRAN_UNPACK_NUM_THREAD_BLOCKS));
  } else {
    sendOffset = 1;
  }

  while (rootDist + sendOffset < nRanks) {
    int sendTo = (myRank + sendOffset) % nRanks;
    sendOffset *= 2;

    if (comm->ctran_->mapper->getBackend(sendTo) == CtranMapperBackend::NVL) {
      // Note: we redistributed from recvBuffer on all ranks.
      FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, nGroups, &elem));
      elem->putNotify.sendbuff = op->broadcast.recvbuff;
      elem->putNotify.nbytes = sendSize;
      elem->putNotify.peerLocalRank = statex->localRank(sendTo);
      elem->putNotify.ngroups = nGroups;
      elem->putNotify.notify = true; // each put will be notified to remote peer
      putNotifyList.enqueue(elem);
      op->broadcast.putNotifyMap.insert({sendTo, elem});
    }
  }

  if (putNotifyList.count > 0) {
    // Allow user to increase SM usuage for putNotify involved kernel
    config.numBlocks = maxNumBlocks;
    config.numThreads = getThreadBlockSize();
  }

  if (hasTcpDmRecv) {
    config.numBlocks = NCCL_CTRAN_UNPACK_NUM_THREAD_BLOCKS;
    config.numThreads = NCCL_CTRAN_UNPACK_THREAD_BLOCK_SIZE;
    config.type = KernelConfig::BROADCAST_UNPACK;
  }

  config.args.collective.broadcast.putNotifyList = putNotifyList.head;
  config.args.collective.broadcast.waitNotifyList = waitNotifyList.head;

  return commSuccess;
}

static commResult_t impl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  size_t sendSize = op->broadcast.count * commTypeSize(op->broadcast.datatype);
  CtranComm* comm = op->comm_;
  CtranMapper* mapper = op->ctran->mapper.get();

  const auto statex = comm->statex_.get();
  int root = op->broadcast.root;
  int nRanks = statex->nRanks();
  int myRank = statex->rank();
  void* recvHdl;
  std::vector<void*> remoteRecvBuffs(nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;
  std::vector<std::unique_ptr<CtranMapperRequest>> irecvReq;
  std::vector<std::unique_ptr<CtranMapperRequest>> iputReq;
  bool localReg;
  CtranMapperRequest* req = nullptr;
  CtranMapperRequest* sendReq = nullptr;

  int rootDist = (myRank - root + nRanks) % nRanks;
  int recvOffset =
      floorPow2(rootDist); // Rank at distance 011011 recv data
                           // from rank at 001011 which is 010000 away
  int sendOffsetProto = myRank == root
      ? 1
      : recvOffset * 2; // Rank at distance 011011 sends data to rank 111011
                        // which is 100000 away, then to 1111011 away etc.
  bool isLeaf = (myRank != root) && (rootDist + sendOffsetProto >= nRanks);

  CtranAlgoLogger logger(
      broadcastAlgoName(myAlgo), op->opCount, comm, op->ctran);

  auto& putNotifyMap = op->broadcast.putNotifyMap;
  auto& waitNotifyMap = op->broadcast.waitNotifyMap;

  CtranMapperContext context("CtranBroadcastBinomialTree", sendSize, sendSize);
  mapper->setContext(std::move(context));
  for (int p = 0; p < nRanks; ++p) {
    remoteAccessKeys.emplace_back();
  }

  if (nRanks == 1) {
    return commSuccess;
  }

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(broadcastAlgoName(myAlgo)));

  // If not root, share my handle
  int parent = (myRank - recvOffset + nRanks) % nRanks;
  auto notifyParent = std::make_unique<CtranMapperNotify>();

  // 1. Send handle
  // Non-root ranks share their handle with parent node to receive data.
  // Example: rank 6 is expected to receive data from rank 2 (during the 3rd
  // round of distribution).
  KernelElem* elem = nullptr;

  // register recvbuff for use with both send and recv.
  FB_COMMCHECK(mapper->searchRegHandle(
      op->broadcast.recvbuff, sendSize, &recvHdl, &localReg));

  if (myRank != root) {
    // Some backends (nvl, tcpdm) and elem are set only for device
    // communication; otherwise always use IB backend.
    if (op->isDevice && mapper->requiresRecvNotify(parent)) {
      if (waitNotifyMap.contains(parent)) {
        elem = waitNotifyMap[parent];
      } else {
        CLOGF(
            WARN,
            "Expecting NVLink waitNotify for parent {}. Something bad probably happened.",
            parent);
      }
    }

    FB_COMMCHECK(
        mapper->isendCtrl(op->broadcast.recvbuff, recvHdl, parent, &sendReq));
    // Initialize notify flag to receive from parent
    FB_COMMCHECK(mapper->initNotify(parent, recvHdl, elem, notifyParent.get()));
  }

  // 2. Recv handle(s)
  // Recv handle(s) from all children ranks. For rank 0 ranks [1,2,4,8,...]
  if (!isLeaf) {
    int sendOffset = sendOffsetProto;

    while (rootDist + sendOffset < nRanks) {
      int sendTo = (myRank + sendOffset) % nRanks;
      sendOffset *= 2;

      FB_COMMCHECK(mapper->irecvCtrl(
          &remoteRecvBuffs[sendTo], &remoteAccessKeys[sendTo], sendTo, &req));
      irecvReq.push_back(std::unique_ptr<CtranMapperRequest>(req));
    }
  }

  // 3. Wait for the data to arrive
  // Example: rank 6 waits for data from rank 2.
  if (myRank != root) {
    FB_COMMCHECK(mapper->waitRequest(sendReq));
    // Wait for the put from the sender to complete
    FB_COMMCHECK(mapper->waitNotify(notifyParent.get()));
  }

  // 4. Send data to all children in order
  if (!isLeaf) {
    for (const auto& recvCtrlReq : irecvReq) {
      int peer = recvCtrlReq->peer;
      // We wait for peers in order.
      mapper->waitRequest(recvCtrlReq.get());

      timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(peer));

      KernelElem* elem = nullptr;
      if (op->isDevice && mapper->getBackend(peer) == CtranMapperBackend::NVL) {
        if (putNotifyMap.contains(peer)) {
          elem = putNotifyMap[peer];
        } else {
          CLOGF(
              WARN,
              "Expecting NVLink putNotify for peer {}. Something bad probably happened.",
              peer);
        }
      }

      FB_COMMCHECK(mapper->iput(
          op->broadcast.recvbuff,
          (void*)((uintptr_t)remoteRecvBuffs[peer]),
          sendSize,
          peer,
          CtranMapperConfig{
              .memHdl_ = recvHdl,
              .remoteAccessKey_ = remoteAccessKeys[peer],
              .notify_ = true,
              .kernElem_ = elem},
          &req));
      iputReq.push_back(std::unique_ptr<CtranMapperRequest>(req));
      timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));
    }

    // 4.1 Wait for all remote writes to complete
    while (!iputReq.empty()) {
      FB_COMMCHECK(mapper->testSomeRequests(iputReq, timestamp->putComplete));
    }
  }

  if (localReg == true) {
    FB_COMMCHECK(mapper->deregDynamic(recvHdl));
  }

  mapper->timestamps.emplace_back(std::move(timestamp));
  mapper->reportProfiling();

  return commSuccess;
}

// GPU broadcast with kernel launch
commResult_t ctranBroadcastBinomialTree(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int root,
    CtranComm* comm,
    cudaStream_t stream) {
  auto opCount = comm->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      broadcastAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      count,
      datatype,
      root,
      comm,
      stream);
  const auto statex = comm->statex_.get();

  if (sendbuff != recvbuff && statex->rank() == root) {
    // Copy to recv buffer if it is different from the send buffer
    FB_COMMCHECK(comm->ctran_->mapper->icopy(
        recvbuff, sendbuff, count * commTypeSize(datatype), stream));
  }

  size_t typeSize = commTypeSize(datatype);
  void* sbuf = const_cast<void*>(sendbuff);
  void* dbuf = recvbuff;

  // FIXME: We perform an extra copy here before we submit to the GPE
  // thread.  Ideally we should be doing this copy inside the GPE
  // thread, but that requires two changes first: (1) our
  // searchRegHandle cannot try to dynamically register the buffer (as
  // that will fail); and (2) we need a copy kernel which does not
  // currently exist.
  if (count * typeSize < CTRAN_MIN_REGISTRATION_SIZE) {
    // make sure tmpbuf is allocated and registered
    FB_COMMCHECK(comm->ctran_->algo->initTmpBufs());
    sbuf = comm->ctran_->algo->getTmpBuf(
        CtranAlgo::TmpbufType::MIN_REG_SRC_TMPBUF);
    dbuf = comm->ctran_->algo->getTmpBuf(
        CtranAlgo::TmpbufType::MIN_REG_DST_TMPBUF);
    FB_CUDACHECK(cudaMemcpyAsync(
        dbuf, recvbuff, count * typeSize, cudaMemcpyDefault, stream));
  }

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  std::unique_ptr<struct OpElem> op;

  op = std::make_unique<struct OpElem>(
      // pass in default mapper for now; we will consistently use mapper from
      // Ctran object in future
      OpElem::opType::BROADCAST,
      comm,
      comm->ctran_.get(),
      opCount);
  op->broadcast.sendbuff = reinterpret_cast<const void*>(sbuf);
  op->broadcast.recvbuff = dbuf;
  op->broadcast.count = count;
  op->broadcast.datatype = datatype;
  op->broadcast.root = root;
  opGroup.push_back(std::move(op));

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::BROADCAST,
      stream,
      broadcastAlgoName(myAlgo),
      opCount);
  config.args.devState_d = comm->ctran_->algo->getDevState();
  config.args.collective.broadcast.sendbuff =
      reinterpret_cast<const void*>(sbuf);
  config.args.collective.broadcast.recvbuff = dbuf;
  config.args.collective.broadcast.datatype = datatype;
  config.args.collective.broadcast.count = count;

  FB_COMMCHECK(setupPlan(comm, opGroup, config));
  void* fn = reinterpret_cast<void*>(ncclKernelBroadcast</*UNPACK=*/false>);
  if (config.type == KernelConfig::BROADCAST_UNPACK) {
    fn = reinterpret_cast<void*>(ncclKernelBroadcast</*UNPACK=*/true>);
  }
  FB_COMMCHECK(comm->ctran_->gpe->submit(std::move(opGroup), impl, config, fn));

  if (count * typeSize < CTRAN_MIN_REGISTRATION_SIZE &&
      statex->rank() != root) {
    FB_CUDACHECK(cudaMemcpyAsync(
        recvbuff, dbuf, count * typeSize, cudaMemcpyDefault, stream));
  }

  return commSuccess;
}

// CPU broadcast without kernel launch.
// Rather than using the default Ctran object of comm, it uses the current ctran
// object being called.
commResult_t CtranAlgo::broadcastBinomialTree(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int root,
    std::atomic_flag* cpuFlag) {
  auto opCount = ctran_->getOpCount();
  CTRAN_HOST_COLL_INFO(
      broadcastAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      count,
      datatype,
      root,
      comm_,
      ctran_,
      cpuFlag);
  const auto statex = comm_->statex_.get();

  if (sendbuff != recvbuff && statex->rank() == root) {
    // Copy to recv buffer if it is different from the send buffer
    memcpy(recvbuff, sendbuff, count * commTypeSize(datatype));
  }

  void* sbuf = const_cast<void*>(sendbuff);
  void* dbuf = recvbuff;

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  std::unique_ptr<struct OpElem> op;

  op = std::make_unique<struct OpElem>(
      OpElem::opType::BROADCAST, comm_, ctran_, opCount);
  // Indicate it is host memory communication
  op->isDevice = false;
  op->broadcast.sendbuff = reinterpret_cast<const void*>(sbuf);
  op->broadcast.recvbuff = dbuf;
  op->broadcast.count = count;
  op->broadcast.datatype = datatype;
  op->broadcast.root = root;
  opGroup.push_back(std::move(op));

  // Dummy kernel config for colltrace record, no real kernel will be launched
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::BROADCAST,
      nullptr,
      broadcastAlgoName(myAlgo),
      opCount);
  config.isDevice = false;
  config.args.collective.broadcast.sendbuff =
      reinterpret_cast<const void*>(sbuf);
  config.args.collective.broadcast.recvbuff = dbuf;
  config.args.collective.broadcast.datatype = datatype;
  config.args.collective.broadcast.count = count;

  FB_COMMCHECK(comm_->ctran_->gpe->submitHost(
      std::move(opGroup), impl, config, cpuFlag));

  return commSuccess;
}
