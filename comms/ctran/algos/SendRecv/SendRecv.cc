// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <deque>
#include <optional>

#include "comms/ctran/algos/SendRecv/SendRecvImpl.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/ExtUtils.h"

static thread_local std::deque<OpElem*> CtranOpGroup;

std::unordered_map<KernelConfig::KernelType, void*> kernelFns = {
    {KernelConfig::KernelType::SEND, reinterpret_cast<void*>(ncclKernelSend)},
    {KernelConfig::KernelType::RECV,
     reinterpret_cast<void*>(ncclKernelRecv</*UNPACK=*/false>)},
    {KernelConfig::KernelType::SENDRECV,
     reinterpret_cast<void*>(ncclKernelSendRecv</*UNPACK=*/false>)},
    {KernelConfig::KernelType::SEND_NOTIFY,
     reinterpret_cast<void*>(ncclKernelSendNotifyOnly)},
    {KernelConfig::KernelType::RECV_NOTIFY,
     reinterpret_cast<void*>(ncclKernelRecvNotifyOnly)},
    {KernelConfig::KernelType::SENDRECV_NOTIFY,
     reinterpret_cast<void*>(ncclKernelSendRecvNotifyOnly)},
    {KernelConfig::KernelType::RECV_UNPACK,
     reinterpret_cast<void*>(ncclKernelRecv</*UNPACK=*/true>)},
    {KernelConfig::KernelType::SENDRECV_UNPACK,
     reinterpret_cast<void*>(ncclKernelSendRecv</*UNPACK=*/true>)},
    {KernelConfig::KernelType::SENDRECV_STAGED,
     reinterpret_cast<void*>(ncclKernelSendRecvStaged)},
};

static const auto myAlgo = NCCL_SENDRECV_ALGO::ctran;

bool ctranSendRecvSupport(int peer, CtranComm* comm) {
  const auto statex = comm->statex_.get();

  // Self peer is handled by CE directly, other peers require a valid ctran
  // backend
  if (ctranInitialized(comm) &&
      (peer == statex->rank() ||
       comm->ctran_->mapper->getBackend(peer) != CtranMapperBackend::UNSET)) {
    return true;
  } else {
    return false;
  }
}

commResult_t ctranSend(
    const void* sendbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream) {
  auto opCount = comm->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      "CtranSend", sendbuff, nullptr, count, datatype, peer, comm, stream);

  auto op = new OpElem(OpElem::opType::SEND, stream, comm, opCount);
  op->send.sendbuff = sendbuff;
  op->send.count = count;
  op->send.datatype = datatype;
  op->send.peerRank = peer;
  if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE &&
      comm->ctran_->mapper->getBackend(peer) == CtranMapperBackend::NVL &&
      comm->statex_.get()->rank() != peer) {
    // used for storing recv address (updated by GPE thread), delete after
    // the address is no longer needed
    op->send.recvbuff = new std::atomic<void*>();
  }

  CtranOpGroup.push_back(op);

  return commSuccess;
}

commResult_t ctranRecv(
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream) {
  auto opCount = comm->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      "CtranRecv", nullptr, recvbuff, count, datatype, peer, comm, stream);

  auto op = new OpElem(OpElem::opType::RECV, stream, comm, opCount);
  op->recv.recvbuff = recvbuff;
  op->recv.count = count;
  op->recv.datatype = datatype;
  op->recv.peerRank = peer;

  CtranOpGroup.push_back(op);

  return commSuccess;
}

commResult_t ctranGroupEndHook(
    std::optional<std::chrono::milliseconds> timeout) {
  auto algo = NCCL_SENDRECV_ALGO;
  // By default, use zero-copy kernel for sendrecv.
  if (algo == NCCL_SENDRECV_ALGO::ctran) {
    algo = NCCL_SENDRECV_ALGO::ctzcopy;
  }
  while (!CtranOpGroup.empty()) {
    // TODO: clean up duplicate info in allops, nvlOps and ibOps
    std::vector<OpElem*> allOps;
    std::vector<OpElem*> selfSends, selfRecvs, sendNvlOps, nvlOps, ibOps;
    std::deque<OpElem*> pending;
    bool hasSend = false;
    bool hasRecv = false;
    bool hasTcpDmRecv = false;

    // Submit ops with the same comm and stream in a single batch
    CtranComm* comm = CtranOpGroup.front()->comm_;
    cudaStream_t stream = CtranOpGroup.front()->stream;
    const auto statex = comm->statex_.get();
    auto mapper = comm->ctran_->mapper.get();

    while (!CtranOpGroup.empty()) {
      auto op = dequeFront(CtranOpGroup);

      if (op->comm_ == comm && op->stream == stream) {
        if (op->type == OpElem::opType::SEND) {
          hasSend = true;
          if (op->send.peerRank == statex->rank()) {
            selfSends.push_back(op);
            continue;
          }

          // Async buffer registration for send and recv buffers to hide
          // registration cost.
          // - If the buffer has already been registered, regAsync will return
          //   immediately.
          // - If the buffer is not yet registered at regAsync internal query, a
          //   request will be enqueued to asyncReg thread.
          // - A first-used buffer will be registered either by asyncReg thread
          //   or GPE thread (see MemRegCache::regRange).
          // - regAsync is a no-op if NCCL_CTRAN_REGISTER is not async mode.
          //
          // Expected performance improvement for communication involving
          // first-time registration:
          // - [Improved case] If the buffer is registered by asyncReg thread
          //   ahead of time, it hides registration cost.
          // - If the buffer is registered by GPE thread, e.g., due to too busy
          //   asyncReg thread or not-advanced CPU schedule, the registration
          //   cost has to be exposed similar to lazy registatration mode.
          size_t nbytes = op->send.count * commTypeSize(op->send.datatype);
          FB_COMMCHECK(mapper->regAsync(op->send.sendbuff, nbytes));
          if (comm->ctran_->mapper->getBackend(op->send.peerRank) ==
              CtranMapperBackend::NVL) {
            sendNvlOps.push_back(op);
            nvlOps.push_back(op);
          } else {
            ibOps.push_back(op);
          }
        } else if (op->type == OpElem::opType::RECV) {
          hasRecv = true;
          if (op->recv.peerRank == statex->rank()) {
            selfRecvs.push_back(op);
            continue;
          }

          // For TCP Device Memory, if we have peers we are going to receive
          // from, we need to unpack the data from the bounce buffer.
          if (comm->ctran_->mapper->getBackend(op->recv.peerRank) ==
              CtranMapperBackend::TCPDM) {
            hasTcpDmRecv = true;
          }

          size_t nbytes = op->recv.count * commTypeSize(op->recv.datatype);
          FB_COMMCHECK(mapper->regAsync(op->recv.recvbuff, nbytes));
          if (comm->ctran_->mapper->getBackend(op->recv.peerRank) ==
              CtranMapperBackend::NVL) {
            nvlOps.push_back(op);
          } else {
            ibOps.push_back(op);
          }
        }

        allOps.push_back(op);
      } else {
        // If not belong to this batch, put to pending and handle in next batch
        pending.push_back(op);
      }
    }

    // Handle self sends and recvs via CE-based icopy
    FB_COMMCHECK(selfSendRecvImpl(selfSends, selfRecvs, comm));

    // For non-self sends and recvs: decide the kernel function and submit to
    // GPE
    if (!allOps.empty()) {
      // host side
      std::vector<std::unique_ptr<struct OpElem>> gpeOpGroup;
      FB_COMMCHECK(
          ctran::sendrecv::setupGpeOp(
              comm, allOps, nvlOps, sendNvlOps, ibOps, gpeOpGroup, algo));

      // device side
      KernelConfig::KernelType kernelType =
          ctran::sendrecv::getKernelType(hasSend, hasRecv, hasTcpDmRecv, algo);
      auto config = KernelConfig(
          kernelType,
          stream,
          sendRecvAlgoName(myAlgo, allOps),
          allOps.front()->opCount);

      // Create kernArgs on stack for SENDRECV_STAGED
      ctran::sendrecv::KernArgs kernArgs;
      FB_COMMCHECK(
          ctran::sendrecv::setupKernelConfig(
              comm, allOps, nvlOps, config, kernArgs));

      FB_COMMCHECK(comm->ctran_->gpe->submit(
          std::move(gpeOpGroup),
          sendRecvImpl,
          config,
          kernelFns.at(kernelType),
          timeout));
    }

    // No kernel would be submitted if only self sendrecv is called, update op
    // count here
    if (allOps.empty() && !selfSends.empty()) {
      comm->ctran_->updateOpCount();
    }
    allOps.clear();

    comm->ctran_->numGroupedDefaultOps = 0;

    // handle next batch
    CtranOpGroup = std::move(pending);
  }

  return commSuccess;
}

void ctranGroupTrackDefaultOp(CtranComm* comm) {
  if (ctranInitialized(comm)) {
    comm->ctran_->numGroupedDefaultOps++;
  }
}
