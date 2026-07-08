// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/gpe/CtranGpe.h"

#include <chrono>
#include <iostream>
#include <optional>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpeImpl.h"
#include "comms/ctran/profiler/DefaultGpeProfilerReporter.h"
#include "comms/ctran/profiler/GpeProfiler.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

using namespace ctran;

namespace {
std::string kernelTypeToOpName(KernelConfig::KernelType type) {
  switch (type) {
    case KernelConfig::ALLGATHER:
    case KernelConfig::ALLGATHERP:
    // ALLGATHERP_INIT goes through submitHost(), not submit(), so this
    // case is currently unreachable. Included for completeness.
    case KernelConfig::ALLGATHERP_INIT:
      return "AllGather";
    case KernelConfig::ALLREDUCE:
      return "AllReduce";
    case KernelConfig::SEND:
    case KernelConfig::RECV:
    case KernelConfig::SENDRECV:
    case KernelConfig::SENDRECV_P2P:
    case KernelConfig::RECV_UNPACK:
    case KernelConfig::SENDRECV_UNPACK:
      return "SendRecv";
    case KernelConfig::ALLTOALL:
    case KernelConfig::ALLTOALL_DEDUP:
    case KernelConfig::DEVICE_ALLTOALLV:
    case KernelConfig::ALLTOALLV:
    case KernelConfig::ALLTOALLV_DEDUP:
      return "AllToAll";
    case KernelConfig::BROADCAST:
    case KernelConfig::BROADCAST_UNPACK:
      return "Broadcast";
    case KernelConfig::REDUCESCATTER:
      return "ReduceScatter";
    case KernelConfig::PUTNOTIFY:
    case KernelConfig::WAITNOTIFY:
    case KernelConfig::PUTSIGNAL:
    case KernelConfig::WAITSIGNAL:
    case KernelConfig::SIGNAL:
    case KernelConfig::GET:
      return "RMA";
    default:
      return "Unknown";
  }
}
} // namespace

OpElem::OpElem(enum opType type, CtranComm* comm, uint64_t opCount)
    : OpElem(type, nullptr, comm, nullptr, opCount) {};

OpElem::OpElem(
    enum opType type,
    CtranComm* comm,
    ICtran* ctran,
    uint64_t opCount)
    : OpElem(type, nullptr, comm, ctran, opCount) {};

OpElem::OpElem(
    enum opType type,
    cudaStream_t stream,
    CtranComm* comm,
    uint64_t opCount)
    : OpElem(type, stream, comm, nullptr, opCount) {};

OpElem::OpElem(OpElem* op) {
  this->type = op->type;
  this->stream = op->stream;
  this->comm_ = op->comm_;
  this->opCount = op->opCount;

  if (op->type == ALLTOALL_DEDUP) {
    new (&this->alltoall_dedup.remoteRecvBuffs) std::vector<void*>;
    this->alltoall_dedup.remoteRecvBuffs.resize(comm_->statex_->nRanks());
    for (int i = 0; i < comm_->statex_->nRanks(); i++) {
      this->alltoall_dedup.remoteRecvBuffs[i] =
          op->alltoall_dedup.remoteRecvBuffs[i];
    }
    new (&this->alltoall_dedup.remoteAccessKeys)
        std::vector<struct CtranMapperRemoteAccessKey>;
    this->alltoall_dedup.remoteAccessKeys.resize(comm_->statex_->nRanks());
    for (int i = 0; i < comm_->statex_->nRanks(); i++) {
      this->alltoall_dedup.remoteAccessKeys[i].backend =
          op->alltoall_dedup.remoteAccessKeys[i].backend;
      this->alltoall_dedup.remoteAccessKeys[i].ibKey =
          op->alltoall_dedup.remoteAccessKeys[i].ibKey;
    }
    new (&this->alltoall_dedup.bcastElemMap)
        std::unordered_map<int, KernelElem*>;
    this->alltoall_dedup.bcastElemMap = op->alltoall_dedup.bcastElemMap;
    this->alltoall_dedup.datatype = op->alltoall_dedup.datatype;
    this->alltoall_dedup.sendbuff = op->alltoall_dedup.sendbuff;
    this->alltoall_dedup.recvbuff = op->alltoall_dedup.recvbuff;
    this->alltoall_dedup.sendHdl = op->alltoall_dedup.sendHdl;
    this->alltoall_dedup.recvHdl = op->alltoall_dedup.recvHdl;
    this->alltoall_dedup.sendcounts = op->alltoall_dedup.sendcounts;
    this->alltoall_dedup.sdispls = op->alltoall_dedup.sdispls;
    this->alltoall_dedup.recvcounts = op->alltoall_dedup.recvcounts;
    this->alltoall_dedup.rdispls = op->alltoall_dedup.rdispls;
  } else {
    FB_CHECKABORT(
        false, "This function currently only supports ALLTOALL_DEDUP");
  }
}

OpElem::OpElem(
    enum opType type,
    cudaStream_t stream,
    CtranComm* comm,
    ICtran* ctran,
    uint64_t opCount)
    : type(type), stream(stream), comm_(comm), ctran(ctran), opCount(opCount) {
  if (!ctran && comm_->ctran_) {
    // Set to communicator's default ctran if it is not passed in.
    // NOTE: some UT may use dummyComm without actual Ctran object, thus
    // comm->ctran may be nullptr.
    ctran = comm_->ctran_.get();
  }
  switch (type) {
    case ALLTOALLV:
      new (&this->alltoallv.sendcounts) std::vector<size_t>;
      this->alltoallv.sendcounts.resize(comm_->statex_->nRanks());
      new (&this->alltoallv.sdispls) std::vector<size_t>;
      this->alltoallv.sdispls.resize(comm_->statex_->nRanks());
      new (&this->alltoallv.recvcounts) std::vector<size_t>;
      this->alltoallv.recvcounts.resize(comm_->statex_->nRanks());
      new (&this->alltoallv.rdispls) std::vector<size_t>;
      this->alltoallv.rdispls.resize(comm_->statex_->nRanks());
      break;
    case ALLTOALL_DEDUP:
      new (&this->alltoall_dedup.remoteRecvBuffs) std::vector<void*>;
      this->alltoall_dedup.remoteRecvBuffs.resize(comm_->statex_->nRanks());
      new (&this->alltoall_dedup.remoteAccessKeys)
          std::vector<struct CtranMapperRemoteAccessKey>;
      this->alltoall_dedup.remoteAccessKeys.resize(comm_->statex_->nRanks());
      new (&this->alltoall_dedup.bcastElemMap)
          std::unordered_map<int, KernelElem*>;
      break;
    case ALLGATHER:
      this->allgather.bcastElem = nullptr;
      break;
    case SEND:
      this->send.kElem = nullptr;
      new (&this->send.remoteAccessKey) CtranMapperRemoteAccessKey();
      break;
    case RECV:
      this->recv.kElem = nullptr;
      break;
    case BROADCAST:
      new (&this->broadcast.putNotifyMap) std::unordered_map<int, KernelElem*>;
      new (&this->broadcast.waitNotifyMap) std::unordered_map<int, KernelElem*>;
      break;
    case BROADCASTP:
      new (&this->broadcastP.putNotifyMap) std::unordered_map<int, KernelElem*>;
      new (&this->broadcastP.waitNotifyMap)
          std::unordered_map<int, KernelElem*>;
      break;
    case REDUCESCATTER:
      new (&this->reducescatter.intraReduce) std::vector<KernelElem*>;
      this->reducescatter.intraReduce.resize(comm_->statex_->nNodes(), nullptr);
      this->reducescatter.interReduce = nullptr;
      break;
    case ALLREDUCE:
      new (&this->allreduce.kElemStepMap) std::unordered_map<int, KernelElem*>;
      new (&this->allreduce.remoteRecvBuffs) std::vector<void*>;
      this->allreduce.remoteRecvBuffs.resize(comm_->statex_->nRanks());
      new (&this->allreduce.remoteAccessKeys)
          std::vector<struct CtranMapperRemoteAccessKey>;
      this->allreduce.remoteAccessKeys.resize(comm_->statex_->nRanks());
      new (&this->allreduce.hostArgs) ctran::allreduce::ring::HostArgs();
      new (&this->allreduce.hostResource)
          ctran::allreduce::ring::HostResource();
      break;
    default:
      break;
  }
}

OpElem::~OpElem() {
  switch (type) {
    case ALLTOALLV:
      this->alltoallv.sendcounts.~vector();
      this->alltoallv.sdispls.~vector();
      this->alltoallv.recvcounts.~vector();
      this->alltoallv.rdispls.~vector();
      break;
    case ALLTOALL_DEDUP:
      for (auto& pair : this->alltoall_dedup.bcastElemMap) {
        if (pair.second != nullptr) {
          pair.second->free();
        }
      }
      this->alltoall_dedup.bcastElemMap.~unordered_map();
      break;
    case ALLGATHER:
      if (this->allgather.bcastElem) {
        this->allgather.bcastElem->free();
      }
      break;
    // Free kElem for later reclaim back to KernelElemPool
    case SEND:
      if (this->send.kElem) {
        this->send.kElem->free();
      }
      this->send.remoteAccessKey.~CtranMapperRemoteAccessKey();
      break;
    case RECV:
      if (this->recv.kElem) {
        this->recv.kElem->free();
      }
      break;
    case BROADCAST:
      for (auto& pair : this->broadcast.putNotifyMap) {
        if (pair.second != nullptr) {
          pair.second->free();
        }
      }
      this->broadcast.putNotifyMap.~unordered_map();
      for (auto& pair : this->broadcast.waitNotifyMap) {
        if (pair.second != nullptr) {
          pair.second->free();
        }
      }
      this->broadcast.waitNotifyMap.~unordered_map();
      break;
    case BROADCASTP:
      for (auto& pair : this->broadcastP.putNotifyMap) {
        if (pair.second != nullptr) {
          pair.second->free();
        }
      }
      this->broadcastP.putNotifyMap.~unordered_map();
      for (auto& pair : this->broadcastP.waitNotifyMap) {
        if (pair.second != nullptr) {
          pair.second->free();
        }
      }
      this->broadcastP.waitNotifyMap.~unordered_map();
      break;
    case REDUCESCATTER:
      for (auto elem : this->reducescatter.intraReduce) {
        if (elem != nullptr) {
          elem->free();
        }
      }
      if (this->reducescatter.interReduce) {
        this->reducescatter.interReduce->free();
      }
      break;
    case ALLREDUCE: {
      for (auto& pair : this->allreduce.kElemStepMap) {
        if (pair.second != nullptr) {
          pair.second->free();
        }
      }
      this->allreduce.kElemStepMap.~unordered_map();
      this->allreduce.remoteRecvBuffs.~vector();
      this->allreduce.remoteAccessKeys.~vector();
      this->allreduce.hostArgs.~HostArgs();
      // ~HostResource releases GpeKernelSyncs back to pool
      this->allreduce.hostResource.~HostResource();
      break;
    }
    default:
      break;
  }
}

void OpElem::setStatus(KernelElem::ElemStatus status) {
  switch (type) {
    case ALLGATHER:
      if (this->allgather.bcastElem) {
        this->allgather.bcastElem->setStatus(status);
      }
      break;
    case SEND:
      if (this->send.kElem) {
        this->send.kElem->setStatus(status);
      }
      break;
    case RECV:
      if (this->recv.kElem) {
        this->recv.kElem->setStatus(status);
      }
      break;
    case BROADCAST:
      for (auto& pair : this->broadcast.putNotifyMap) {
        if (pair.second != nullptr) {
          pair.second->setStatus(status);
        }
      }
      for (auto& pair : this->broadcast.waitNotifyMap) {
        if (pair.second != nullptr) {
          pair.second->setStatus(status);
        }
      }
      break;
    case REDUCESCATTER:
      for (auto elem : this->reducescatter.intraReduce) {
        if (elem != nullptr) {
          elem->setStatus(status);
        }
      }
      if (this->reducescatter.interReduce) {
        this->reducescatter.interReduce->setStatus(status);
      }
      break;
    case ALLREDUCE: {
      for (auto& pair : this->allreduce.kElemStepMap) {
        if (pair.second != nullptr) {
          pair.second->setStatus(status);
        }
      }
      break;
    }
    case ALLTOALL_DEDUP: {
      for (auto& pair : this->alltoall_dedup.bcastElemMap) {
        if (pair.second != nullptr) {
          pair.second->setStatus(status);
        }
      }
      break;
    }
    default:
      // FIXME: add a WARN log here
      break;
  }
}

static std::unordered_map<KernelConfig::KernelType, std::string>
    kernelTypeNameMap = {
        {KernelConfig::KernelType::ALLGATHER, "ALLGATHER"},
        {KernelConfig::KernelType::ALLREDUCE, "ALLREDUCE"},
        {KernelConfig::KernelType::ALLTOALL, "ALLTOALL"},
        {KernelConfig::KernelType::DEVICE_ALLTOALLV, "DEVICE_ALLTOALLV"},
        {KernelConfig::KernelType::ALLTOALLV, "ALLTOALLV"},
        {KernelConfig::KernelType::ALLTOALL_DEDUP, "ALLTOALL_DEDUP"},
        {KernelConfig::KernelType::SENDRECV, "SENDRECV"},
        {KernelConfig::KernelType::SEND, "SEND"},
        {KernelConfig::KernelType::RECV, "RECV"},
        {KernelConfig::KernelType::SENDRECV_P2P, "SENDRECV_P2P"},
        {KernelConfig::KernelType::BROADCAST, "BROADCAST"},
        {KernelConfig::KernelType::REDUCESCATTER, "REDUCESCATTER"},
        {KernelConfig::KernelType::PUTNOTIFY, "PUTNOTIFY"},
        {KernelConfig::KernelType::WAITNOTIFY, "WAITNOTIFY"},
        {KernelConfig::KernelType::ALLGATHERP, "ALLGATHERP"},
        {KernelConfig::KernelType::ALLGATHERP_INIT, "ALLGATHERP_INIT"},
};

std::string KernelConfig::toString() {
  std::stringstream ss;
  if (kernelTypeNameMap.find(this->type) != kernelTypeNameMap.end()) {
    ss << kernelTypeNameMap[this->type];
  } else {
    // In case invalid type is assigned
    ss << "UNKNOWN_KERNEL(" << this->type << ")";
  }
  ss << " numBlocks=" << this->numBlocks << " numThreads=" << this->numThreads
     << " stream=" << std::hex << this->stream;
  return ss.str();
}

CtranGpe::CtranGpe(
    int cudaDev,
    CtranComm* comm,
    std::unique_ptr<ctran::IGpeProfilerReporter> reporter) {
  this->pimpl = std::make_unique<Impl>();
  this->pimpl->comm = comm;
  this->pimpl->cudaDev = cudaDev;
  this->pimpl->gpe = this;
  // The cvar is the kill-switch at this integration layer. When false,
  // the reporter is nulled regardless of caller injection — production
  // operators always control whether Scuba rows flow. The profiler still
  // exists and tracks internal state so gpeProfiler_->debugString() keeps
  // populating the abort ERR line on stderr.
  if (!NCCL_CTRAN_GPE_PROFILING_ENABLE) {
    reporter.reset();
  } else if (!reporter) {
    reporter = std::make_unique<ctran::DefaultGpeProfilerReporter>();
  }
  this->pimpl->gpeProfiler_ = std::make_unique<ctran::GpeProfiler>(
      &comm->logMetaData_,
      comm->statex_->rank(),
      comm->statex_->commHash(),
      NCCL_CTRAN_GPE_PROFILING_SAMPLING_WEIGHT,
      std::move(reporter),
      comm->getAbort());
  this->pimpl->start();
}

CtranGpe::~CtranGpe() {
  this->pimpl->terminate();
}

namespace {
inline size_t getMsgSizeFromOpGroup(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  if (opGroup.empty()) {
    return 0;
  }
  const auto& op = *opGroup.front();
  switch (op.type) {
    case OpElem::ALLGATHER:
      return op.allgather.sendcount * commTypeSize(op.allgather.datatype);
    case OpElem::ALLREDUCE:
      return op.allreduce.count * commTypeSize(op.allreduce.datatype);
    case OpElem::REDUCESCATTER:
      return op.reducescatter.recvcount *
          commTypeSize(op.reducescatter.datatype);
    case OpElem::ALLTOALL:
      return op.alltoall.count * commTypeSize(op.alltoall.datatype);
    default:
      return 0;
  }
}
} // namespace

commResult_t CtranGpe::submit(
    std::vector<std::unique_ptr<struct OpElem>> opGroup,
    opFunc func,
    KernelConfig& kernelConfig,
    const void* ncclKernel,
    std::optional<std::chrono::milliseconds> timeout,
    PreLaunchGraphPrepareFn graphPrepareFn) {
  this->pimpl->comm->recordAlgoStat(
      kernelTypeToOpName(kernelConfig.type),
      kernelConfig.algoName,
      getMsgSizeFromOpGroup(opGroup));
  return this->pimpl->submit(
      CtranGpeCmd::TypeEnum::GRAPH_ENQUEUE,
      std::move(opGroup),
      func,
      kernelConfig,
      ncclKernel,
      timeout,
      graphPrepareFn);
}

commResult_t CtranGpe::submitHost(
    std::vector<std::unique_ptr<struct OpElem>> opGroup,
    opFunc func,
    KernelConfig& kernelConfig,
    std::shared_ptr<std::atomic_flag> cpuFlag) {
  return this->pimpl->submitHost(
      CtranGpeCmd::TypeEnum::GRAPH_ENQUEUE,
      std::move(opGroup),
      func,
      kernelConfig,
      std::move(cpuFlag));
}

commResult_t CtranGpe::allocKernelElems(
    size_t numElems,
    int ngroups,
    KernelElem** elemsList) {
  // reclaim from outstanding kernels once if elements are insufficient
  if (numElems > this->pimpl->kernelElemPool->size()) {
    this->pimpl->kernelElemPool->reclaim();

    if (numElems > this->pimpl->kernelElemPool->size()) {
      CLOGF(
          WARN,
          "CTRAN-GPE: Internal KernelElem pool has unexpected high usage (capacity: {}, available: {}, current request: {}). "
          "It is likely that some COMM kernels are not released properly",
          this->pimpl->kernelElemPool->capacity(),
          this->pimpl->kernelElemPool->size(),
          numElems);
      return ErrorStackTraceUtil::log(commInternalError);
    }
  }

  // pop free elements and put into C style list for kernel to use.
  if (numElems > 0) {
    *elemsList = this->pimpl->kernelElemPool->pop(ngroups);
    if (!*elemsList) {
      return ErrorStackTraceUtil::log(commInternalError);
    }
  }
  auto elem = *elemsList;
  for (int i = 1; i < numElems; i++) {
    elem->next = this->pimpl->kernelElemPool->pop(ngroups);
    if (!elem->next) {
      return ErrorStackTraceUtil::log(commInternalError);
    }
    elem = elem->next;
  }

  return commSuccess;
}

size_t CtranGpe::numInUseKernelElems() {
  // Last chance to cleanup
  this->pimpl->kernelElemPool->reclaim();
  // Return the number of inuse elements
  return this->pimpl->kernelElemPool->capacity() -
      this->pimpl->kernelElemPool->size();
}

size_t CtranGpe::numInUseKernelFlags() {
  // Last chance to cleanup
  this->pimpl->kernelFlagPool->reclaim();
  // Return the number of inuse flags
  return this->pimpl->kernelFlagPool->capacity() -
      this->pimpl->kernelFlagPool->size();
}

size_t CtranGpe::numInUseChecksums() {
  this->pimpl->checksumPool->reclaim();
  return this->pimpl->checksumPool->capacity() -
      this->pimpl->checksumPool->size();
}

size_t CtranGpe::numInUseGpeKernelSyncs() {
  // Last chance to cleanup
  this->pimpl->gpeKernelSyncPool->reclaim();
  // Return the number of inuse elements
  return this->pimpl->gpeKernelSyncPool->capacity() -
      this->pimpl->gpeKernelSyncPool->size();
}

commResult_t CtranGpe::allocGpeKernelSyncs(
    size_t count,
    int nworkers,
    std::vector<ctran::algos::GpeKernelSync*>& gpeKernelSyncs) {
  return ::allocGpeKernelSyncs(
      this->pimpl->gpeKernelSyncPool.get(), count, nworkers, gpeKernelSyncs);
}

::GpeKernelSyncPool* CtranGpe::gpeKernelSyncPool() {
  return this->pimpl->gpeKernelSyncPool.get();
}
