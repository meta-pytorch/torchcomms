// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/SendRecv/SendRecvImpl.h"
#include "comms/ctran/algos/SendRecv/SendRecvCEImpl.h"
#include "comms/ctran/algos/SendRecv/SendRecvP2pImpl.h"
#include "comms/ctran/algos/SendRecv/SendRecvStagedCopyImpl.h"
namespace {

unsigned int bestThreadBlockSize = 0;

inline int getNumGroups(size_t nbytes) {
  if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE) {
    // if copy engine is enabled, we only need 1 group
    return 1;
  }
  // compute needed thread blocks for given bytes
  int nGroups = nbytes / NCCL_CTRAN_NVL_SENDRECV_CHUNK_SIZE;
  return std::min(
      std::max(1, nGroups), // at least 1 thread block
      // not exceed max theshold
      NCCL_CTRAN_NVL_SENDRECV_MAX_NUM_THREAD_BLOCKS);
}

inline unsigned int getThreadBlockSize() {
  // If first time call, query cuda recommended blockSize
  if (bestThreadBlockSize == 0) {
    int minGridSize = 0;
    FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        (int*)&bestThreadBlockSize,
        reinterpret_cast<const void*>(ncclKernelSendRecv</*UNPACK=*/false>),
        0 /* dynamicSMemSize */,
        0 /* blockSizeLimit */));

    // TODO: bestThreadBlockSize may still be 0 after above function, need a
    // check here to avoid causing error in cudaLaunchKernel. Also for other
    // collectives calling getThreadBlockSize().
  }

  return NCCL_CTRAN_NVL_SENDRECV_ZCOPY_THREAD_BLOCK_SIZE == -1
      ? bestThreadBlockSize
      : NCCL_CTRAN_NVL_SENDRECV_ZCOPY_THREAD_BLOCK_SIZE;
}

} // namespace

namespace ctran::sendrecv {
KernelConfig::KernelType getKernelType(
    bool hasSend,
    bool hasRecv,
    bool hasTcpDmRecv,
    enum NCCL_SENDRECV_ALGO algo) {
  KernelConfig::KernelType kernelType = KernelConfig::KernelType::SENDRECV;
  if (algo == NCCL_SENDRECV_ALGO::ctstaged) {
    kernelType = KernelConfig::KernelType::SENDRECV_STAGED;
    return kernelType;
  } else if (algo == NCCL_SENDRECV_ALGO::ctp2p) {
    kernelType = KernelConfig::KernelType::SENDRECV_P2P;
    return kernelType;
  }
  // TODO: send/recv/sendrecv kernels should be combined into one kernel just
  // like SENDRECV_STAGED
  if (hasSend && hasRecv) {
    if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE) {
      kernelType = KernelConfig::KernelType::SENDRECV_NOTIFY;
    } else if (hasTcpDmRecv) {
      kernelType = KernelConfig::KernelType::SENDRECV_UNPACK;
    }
  } else {
    if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE) {
      kernelType = hasSend ? KernelConfig::KernelType::SEND_NOTIFY
                           : KernelConfig::KernelType::RECV_NOTIFY;
    } else if (hasTcpDmRecv) {
      kernelType = hasSend ? KernelConfig::KernelType::SEND
                           : KernelConfig::KernelType::RECV_UNPACK;
    } else {
      kernelType = hasSend ? KernelConfig::KernelType::SEND
                           : KernelConfig::KernelType::RECV;
    }
  }
  return kernelType;
}

commResult_t setupGpeOp(
    CtranComm* comm,
    std::vector<OpElem*>& allOps,
    std::vector<OpElem*>& nvlOps,
    std::vector<OpElem*>& sendNvlOps,
    std::vector<OpElem*>& ibOps,
    std::vector<std::unique_ptr<OpElem>>& gpeOpGroup,
    enum NCCL_SENDRECV_ALGO algo) {
  if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE) {
    if (!nvlOps.empty()) {
      // first, send/recv NVL ops with copy engine
      FB_COMMCHECK(launchSendRecvCopyEngine(nvlOps, sendNvlOps, comm));
    }
  }

  // If kernel is copy-engine or copy-based, GPE deals with IB ops only.
  if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE ||
      algo == NCCL_SENDRECV_ALGO::ctstaged ||
      algo == NCCL_SENDRECV_ALGO::ctp2p) {
    // next, deal with IB ops
    if (!ibOps.empty()) {
      gpeOpGroup.reserve(ibOps.size());
      for (auto x : ibOps) {
        gpeOpGroup.push_back(std::unique_ptr<OpElem>(x));
      }
      ibOps.clear();
    }
  } else {
    // otherwise, submit all ops to GPE
    gpeOpGroup.reserve(allOps.size());
    for (auto x : allOps) {
      gpeOpGroup.push_back(std::unique_ptr<OpElem>(x));
    }
  }
  return commSuccess;
}

commResult_t setupKernelConfig(
    CtranComm* comm,
    const std::vector<OpElem*>& opGroup,
    const std::vector<OpElem*>& nvlOps,
    KernelConfig& config,
    ctran::sendrecv::KernArgs& kernArgs) {
  const auto statex = comm->statex_.get();
  config.args.devState_d = comm->ctran_->algo->getDevState();
  if (config.type == KernelConfig::KernelType::SENDRECV_STAGED) {
    return setupStagedCopyKernelConfig(comm, nvlOps, config, kernArgs);
  } else if (config.type == KernelConfig::KernelType::SENDRECV_P2P) {
    return setupP2pKernelConfig(comm, nvlOps, config, kernArgs);
  }
  auto putNotifyList = CommonList<KernelElem>();
  auto waitNotifyList = CommonList<KernelElem>();
  int maxNumBlocks = 1;

  for (auto op : opGroup) {
    // For each non-zero NVL op, allocate a p2pElem to coordinate with kernel.
    // - For putNotify elem per send op, recvbuff will be assigned and the elem
    // will be posted to kernel once GPE thread imports remote memory.
    // - For waitNotify elem per recv op, the elem will be posted once GPE
    // thread confirmed the local memory registration.
    // - If an elem with a buffer not qualified for NVL backend, the elem will
    // be revoked by GPE thread, thus kernel will skip it.
    if (op->type == OpElem::opType::SEND &&
        comm->ctran_->mapper->getBackend(op->send.peerRank) ==
            CtranMapperBackend::NVL &&
        op->send.count > 0) {
      size_t nbytes = op->send.count * commTypeSize(op->send.datatype);
      int nGroups = getNumGroups(nbytes);
      // record the max number of thread blocks as final launching grid size
      maxNumBlocks = std::max(maxNumBlocks, nGroups);

      KernelElem* elem = nullptr;
      FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, nGroups, &elem));
      elem->putNotify.sendbuff = op->send.sendbuff;
      elem->putNotify.nbytes = nbytes;
      elem->putNotify.peerLocalRank = statex->localRank(op->send.peerRank);
      elem->putNotify.ngroups = nGroups;
      elem->putNotify.notify = true; // each put will be notified to remote peer
      op->send.kElem = elem;
      putNotifyList.enqueue(elem);
    } else if (
        op->type == OpElem::opType::RECV &&
        comm->ctran_->mapper->requiresRecvNotify(op->recv.peerRank) &&
        op->recv.count > 0) {
      KernelElem* elem = nullptr;
      // only 1 group handles waitNotify elem
      FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, 1, &elem));
      elem->waitNotify.peerLocalRank = statex->localRank(op->recv.peerRank);

      // pass the ngroups used by remote put
      size_t nbytes = op->recv.count * commTypeSize(op->recv.datatype);
      elem->waitNotify.recvbuff = op->recv.recvbuff;
      elem->waitNotify.nbytes = nbytes;
      elem->waitNotify.ngroups = getNumGroups(nbytes);

      op->recv.kElem = elem;
      if (comm->ctran_->mapper->requiresPostRecvNotify(op->recv.peerRank)) {
        waitNotifyList.enqueue(elem);
      }
    }
  }

  if (putNotifyList.count > 0) {
    // Allow user to increase SM usuage for putNotify involved kernel
    config.numBlocks = maxNumBlocks;
    config.numThreads = getThreadBlockSize();
  }

  if (config.type == KernelConfig::KernelType::SENDRECV_UNPACK ||
      config.type == KernelConfig::KernelType::RECV_UNPACK) {
    config.numBlocks = NCCL_CTRAN_UNPACK_NUM_THREAD_BLOCKS;
    config.numThreads = NCCL_CTRAN_UNPACK_THREAD_BLOCK_SIZE;
  }

  if (config.type == KernelConfig::KernelType::SENDRECV ||
      config.type == KernelConfig::KernelType::SENDRECV_NOTIFY ||
      config.type == KernelConfig::KernelType::SENDRECV_UNPACK) {
    config.args.collective.sendrecv.putNotifyList = putNotifyList.head;
    config.args.collective.sendrecv.waitNotifyList = waitNotifyList.head;
    FB_COMMCHECK(comm->ctran_->mapper->prepareUnpackConsumer(
        &config.args.collective.sendrecv.unpack,
        NCCL_CTRAN_UNPACK_NUM_THREAD_BLOCKS,
        opGroup,
        config));
  } else if (
      config.type == KernelConfig::KernelType::SEND ||
      config.type == KernelConfig::KernelType::SEND_NOTIFY) {
    config.args.collective.send.putNotifyList = putNotifyList.head;
    config.args.collective.send.sendbuff = nullptr;
    if (opGroup.size() == 1) {
      const auto op = opGroup[0];
      config.args.collective.send.sendbuff = op->send.sendbuff;
      config.args.collective.send.count = op->send.count;
      config.args.collective.send.datatype = op->send.datatype;
    }
  } else if (
      config.type == KernelConfig::KernelType::RECV ||
      config.type == KernelConfig::KernelType::RECV_NOTIFY ||
      config.type == KernelConfig::KernelType::RECV_UNPACK) {
    config.args.collective.recv.waitNotifyList = waitNotifyList.head;
    config.args.collective.recv.recvbuff = nullptr;
    if (opGroup.size() == 1) {
      const auto op = opGroup[0];
      config.args.collective.recv.recvbuff = op->recv.recvbuff;
      config.args.collective.recv.count = op->recv.count;
      config.args.collective.recv.datatype = op->recv.datatype;
    }
    FB_COMMCHECK(comm->ctran_->mapper->prepareUnpackConsumer(
        &config.args.collective.recv.unpack,
        NCCL_CTRAN_UNPACK_NUM_THREAD_BLOCKS,
        opGroup,
        config));
  }

  return commSuccess;
}

} // namespace ctran::sendrecv
