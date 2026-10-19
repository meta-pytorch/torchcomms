// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/colltrace/CollTraceWrapper.h"

#include <set>

#include "comms/ctran/gpe/CtranGpeColltrace.h"
#include "comms/ctran/utils/CtranLogger.h"
#include "comms/utils/RankUtils.h"
#include "comms/utils/colltrace/v1/CPUWaitEvent.h"
#include "comms/utils/colltrace/v1/CollMetadataImpl.h"
#include "comms/utils/colltrace/v1/CudaWaitEvent.h"
#include "comms/utils/colltrace/v1/DummyCollTraceHandle.h"
#include "comms/utils/colltrace/v1/GraphCudaWaitEvent.h"
#include "comms/utils/colltrace/v1/plugins/CommDumpPlugin.h"
#include "comms/utils/colltrace/v2/CollTraceV2.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace meta::comms::colltrace {

bool isCapturingStream(cudaStream_t stream) {
  cudaStreamCaptureStatus status;

  // For hipStreamGetCaptureInfo it only takes 3 arguments. Since we don't use
  // the extra arguments anyway, we can just use the same function for both.
  auto res = cudaStreamGetCaptureInfo(stream, &status, nullptr);

  if (res != cudaSuccess) {
    CTRAN_LOG_FIRST_N(
        WARN, 1, "Internal error: cudaStreamGetCaptureInfo failed by {}", res);
    return false;
  }
  return status != cudaStreamCaptureStatusNone;
}

GroupedP2PMetaData getGroupedP2PMetaData(
    const int curRank,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    const uint64_t opCount) {
  int64_t sendOpCount = 0;
  int64_t recvOpCount = 0;
  bool mixedTypes = false;
  commDataType_t dataType = commDataType_t::commNumTypes;
  std::unordered_set<int> ranksInGroupedP2P{};
  ranksInGroupedP2P.insert(curRank);

  std::optional<uintptr_t> sendbuff = std::nullopt;
  std::optional<uintptr_t> recvbuff = std::nullopt;
  uint64_t count = 0;

  for (const auto& opElemPtr : opGroup) {
    const auto& gpeOp = *opElemPtr;
    switch (gpeOp.type) {
      case OpElem::SEND: {
        ranksInGroupedP2P.insert(gpeOp.send.peerRank);
        if (sendbuff == std::nullopt && sendOpCount == 0) {
          sendbuff = reinterpret_cast<uintptr_t>(gpeOp.send.sendbuff);
        } else if (
            sendbuff != reinterpret_cast<uintptr_t>(gpeOp.send.sendbuff)) {
          sendbuff = std::nullopt;
        }
        if (dataType != gpeOp.send.datatype) {
          if (dataType == commNumTypes) {
            dataType = gpeOp.send.datatype;
          } else {
            mixedTypes = true;
          }
        }
        count = count + gpeOp.send.count;
        sendOpCount++;
        break;
      }
      case OpElem::RECV: {
        ranksInGroupedP2P.insert(gpeOp.recv.peerRank);
        ranksInGroupedP2P.insert(gpeOp.recv.peerRank);
        if (recvbuff == std::nullopt && recvOpCount == 0) {
          recvbuff = reinterpret_cast<uintptr_t>(gpeOp.recv.recvbuff);
        } else if (
            recvbuff != reinterpret_cast<uintptr_t>(gpeOp.recv.recvbuff)) {
          recvbuff = std::nullopt;
        }
        if (dataType != gpeOp.recv.datatype) {
          if (dataType == commNumTypes) {
            dataType = gpeOp.recv.datatype;
          } else {
            mixedTypes = true;
          }
        }
        count = count + gpeOp.recv.count;
        recvOpCount++;
        break;
      }
      default: {
        CTRAN_LOG_FIRST_N(
            WARN,
            5,
            "COLLTRACE: Should not encounter anything other than OpElem::SEND/RECV in collTraceRecordCtranCollective. Encountered internal error.");
        return GroupedP2PMetaData{
            .opName = "Unknown",
            .algoName = kernelConfig.algoName,
            .opCount = opCount,
        };
      }
    }
  }

  // Summarizing the result
  std::string opName = "";
  if (sendOpCount > 0) {
    opName = "Send";
  }
  if (recvOpCount > 0) {
    opName += "Recv";
  }
  if (mixedTypes) {
    dataType = commNumTypes;
  }

  std::vector<int> ranksInGroupedP2PVec(
      ranksInGroupedP2P.begin(), ranksInGroupedP2P.end());
  std::ranges::sort(ranksInGroupedP2PVec);

  return GroupedP2PMetaData{
      .opName = std::move(opName),
      .algoName = kernelConfig.algoName,
      .opCount = opCount,
      .ranksInGroupedP2P = ranksInGroupedP2PVec,
      .dataType = dataType,
      .count = count,
  };
}

CollectiveMetadata getCollectiveMetadata(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    const uint64_t opCount) {
  switch (kernelConfig.type) {
    case KernelConfig::KernelType::ALLREDUCE: {
      auto allReduceArgs = kernelConfig.args.collective.allreduce;
      return CollectiveMetadata{
          .opName = "AllReduce",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
          .sendbuff = reinterpret_cast<uintptr_t>(allReduceArgs.sendbuff),
          .recvbuff = reinterpret_cast<uintptr_t>(allReduceArgs.recvbuff),
          .dataType = allReduceArgs.datatype,
          .count = allReduceArgs.count,
      };
    }
    case KernelConfig::KernelType::ALLGATHER: {
      auto allGatherArgs = kernelConfig.args.collective.allgather;
      return CollectiveMetadata{
          .opName = "AllGather",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
          .sendbuff = reinterpret_cast<uintptr_t>(allGatherArgs.sendbuff),
          .recvbuff = reinterpret_cast<uintptr_t>(allGatherArgs.recvbuff),
          .dataType = allGatherArgs.datatype,
          .count = allGatherArgs.count,
      };
    }
    case KernelConfig::KernelType::ALLGATHERP: {
      // TODO: Need to get overall allgatherp information separately
      return CollectiveMetadata{
          .opName = "AllGatherP",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
      };
    }
    case KernelConfig::KernelType::ALLGATHERP_INIT: {
      // TODO: Need to get overall allgatherp init information separately
      return CollectiveMetadata{
          .opName = "AllGatherP_Init",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
      };
    }
    case KernelConfig::KernelType::ALLTOALL: {
      auto allToAllArgs = kernelConfig.args.collective.alltoall;
      auto opName = "AllToAll";
      // Special case for alltoallp
      if (opGroup.size() > 0 && opGroup.front()->type == OpElem::ALLTOALLP) {
        opName = "AllToAllP";
      }
      return CollectiveMetadata{
          .opName = opName,
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
          .sendbuff = reinterpret_cast<uintptr_t>(allToAllArgs.sendbuff),
          .recvbuff = reinterpret_cast<uintptr_t>(allToAllArgs.recvbuff),
          .dataType = allToAllArgs.datatype,
          .count = allToAllArgs.count,
      };
    }
    case KernelConfig::KernelType::ALLTOALLV: {
      auto allToAllvArgs = kernelConfig.args.collective.alltoallv;
      return CollectiveMetadata{
          .opName = "AllToAllv",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
          .sendbuff = reinterpret_cast<uintptr_t>(allToAllvArgs.sendbuff),
          .recvbuff = reinterpret_cast<uintptr_t>(allToAllvArgs.recvbuff),
          .dataType = allToAllvArgs.datatype,
          .count = std::nullopt, // AllToAllv uses variable counts
      };
    }
    case KernelConfig::KernelType::ALLTOALLV_DEDUP: {
      // TODO: Add more info for dedup alltoall
      return CollectiveMetadata{
          .opName = "AllToAllv_Dedup",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
      };
    }
    case KernelConfig::KernelType::BROADCAST: {
      auto broadcastArgs = kernelConfig.args.collective.broadcast;
      return CollectiveMetadata{
          .opName = "Broadcast",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
          .sendbuff = reinterpret_cast<uintptr_t>(broadcastArgs.sendbuff),
          .recvbuff = reinterpret_cast<uintptr_t>(broadcastArgs.recvbuff),
          .dataType = broadcastArgs.datatype,
          .count = broadcastArgs.count,
      };
    }
    case KernelConfig::KernelType::BROADCAST_UNPACK: {
      auto broadcastArgs = kernelConfig.args.collective.broadcast;
      return CollectiveMetadata{
          .opName = "Broadcast_Unpack",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
          .sendbuff = reinterpret_cast<uintptr_t>(broadcastArgs.sendbuff),
          .recvbuff = reinterpret_cast<uintptr_t>(broadcastArgs.recvbuff),
          .dataType = broadcastArgs.datatype,
          .count = broadcastArgs.count,
      };
    }
    case KernelConfig::KernelType::REDUCESCATTER: {
      auto reduceScatterArgs = kernelConfig.args.collective.reducescatter;
      return CollectiveMetadata{
          .opName = "ReduceScatter",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
          .sendbuff = reinterpret_cast<uintptr_t>(reduceScatterArgs.sendbuff),
          .recvbuff = reinterpret_cast<uintptr_t>(reduceScatterArgs.recvbuff),
          .dataType = reduceScatterArgs.datatype,
          .count = reduceScatterArgs.recvcount,
      };
    }
    case KernelConfig::KernelType::PUTNOTIFY: {
      // TODO: Add info for putnotify
      return CollectiveMetadata{
          .opName = "PutNotify",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
      };
    }
    case KernelConfig::KernelType::WAITNOTIFY: {
      // TODO: WaitNotify doesn't have buffer information in KernelConfig
      return CollectiveMetadata{
          .opName = "WaitNotify",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
      };
    }
    case KernelConfig::KernelType::PUTSIGNAL: {
      return CollectiveMetadata{
          .opName = "PutSignal",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
      };
    }
    case KernelConfig::KernelType::WAITSIGNAL: {
      // TODO: WaitSignal doesn't have buffer information in KernelConfig
      return CollectiveMetadata{
          .opName = "WaitSignal",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
      };
    }
    case KernelConfig::KernelType::SIGNAL: {
      // TODO: Signal doesn't have buffer information in KernelConfig
      return CollectiveMetadata{
          .opName = "Signal",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
      };
    }
    case KernelConfig::KernelType::GET: {
      return CollectiveMetadata{
          .opName = "Get",
          .algoName = kernelConfig.algoName,
          .opCount = opCount,
      };
    }
    // Skip P2P kernel types. We intentionally don't use default so that
    // whenever ctran adds a new kernel type, we will get a compile error
    case KernelConfig::KernelType::SEND:
    case KernelConfig::KernelType::RECV:
    case KernelConfig::KernelType::SENDRECV:
    case KernelConfig::KernelType::RECV_UNPACK:
    case KernelConfig::KernelType::SENDRECV_UNPACK:
      CTRAN_LOG_FIRST_N(
          ERR, 3, "P2P kernel types being handled by collective path");
      break;
  }
  return CollectiveMetadata{
      .opName = "Unknown",
      .algoName = kernelConfig.algoName,
      .opCount = opCount,
  };
}

bool isP2PKernel(KernelConfig::KernelType kernelType) {
  static std::set<KernelConfig::KernelType> p2pKernels = {
      KernelConfig::KernelType::SEND,
      KernelConfig::KernelType::RECV,
      KernelConfig::KernelType::SENDRECV,
      KernelConfig::KernelType::RECV_UNPACK,
      KernelConfig::KernelType::SENDRECV_UNPACK,
  };

  return p2pKernels.contains(kernelType);
}

std::unique_ptr<ICollMetadata> getMetadata(
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig) {
  CommLogData commLogData = comm->logMetaData_;
  CtranMetadata metadata{
      .stream = kernelConfig.stream,
  };
  auto opCount = *comm->opCount_;
  if (isP2PKernel(kernelConfig.type)) {
    // opCount is incremented after submit
    return makeCollMetadata(
        commLogData,
        metadata,
        getGroupedP2PMetaData(
            commLogData.rank, opGroup, kernelConfig, opCount));
  } else {
    // opCount is incremented before submit
    return makeCollMetadata(
        commLogData,
        metadata,
        getCollectiveMetadata(opGroup, kernelConfig, opCount - 1));
  }
}

// For Ctran, by default we will use CPU-based tracing via tracing the status of
// GPE thread. However, when opGroup == 0, gpe->submit will not let gpe thread
// trace the kernel, which makes CollTrace endlessly waiting for the CPU event
// from GPE. So we use CUDA event for tracing when opGroup == 0, otherwise
// CollTrace will stuck.
std::unique_ptr<ICollWaitEvent> getWaitEvent(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    cudaStream_t stream) {
  if (isCapturingStream(stream)) {
    return std::make_unique<GraphCudaWaitEvent>(stream);
  }
  if (opGroup.size() > 0) {
    return std::make_unique<CPUWaitEvent>();
  }
  return std::make_unique<CudaWaitEvent>(stream);
}

std::shared_ptr<ICollTraceHandle> getNewCollTraceHandle(
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig) {
  auto colltrace = comm->colltraceNew_;
  if (colltrace == nullptr) {
    // For all the invalid cases, we ruturn a dummy handle just so that we
    // don't need to add extra checks in the baseline NCCL code
    return std::make_unique<DummyCollTraceHandle>();
  }

  auto metadata = getMetadata(comm, opGroup, kernelConfig);
  if (metadata == nullptr) {
    return std::make_unique<DummyCollTraceHandle>();
  }

  auto res = colltrace->recordCollective(
      std::move(metadata), getWaitEvent(opGroup, kernelConfig.stream));

  if (res.hasError()) {
    CTRAN_LOG_FIRST_N(
        ERR,
        5,
        "Failed to get colltrace handle due to: {}",
        res.error().message);
    return std::make_unique<DummyCollTraceHandle>();
  }
  return res.value();
}

std::shared_ptr<ICollTraceHandle> getCollTraceHandleRMA(
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    bool shouldRecord) {
  if (!shouldRecord) {
    return std::make_unique<DummyCollTraceHandle>();
  }

  auto handle = getCollTraceHandle(comm, opGroup, kernelConfig, false);
  if (handle == nullptr) {
    return std::make_unique<DummyCollTraceHandle>();
  }
  return handle;
}

std::shared_ptr<ICollTraceHandle> getCollTraceHandle(
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    const bool ifchecksum) {
  if (NCCL_COLLTRACE.empty()) {
    return nullptr;
  }

  return getNewCollTraceHandle(comm, opGroup, kernelConfig);
}

KernelGroupState::KernelGroupState()
    : v2CollId(colltracev2::COLLTRACE_INVALID_COLLID) {}

std::shared_ptr<ICollTraceHandle> traceKernelSubmit(
    CtranComm* comm,
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    const KernelConfig& kernelConfig,
    bool isCapturing,
    bool ifchecksum,
    KernelGroupState& state,
    ColltraceDeviceHandle& hdr) {
  hdr = {};
  // The ring's 128b atomic device write needs sm_90+ for either colltrace.
  const bool supported = ctran::gpe::inKernelColltraceSupported();

  if (colltracev2::enabled() && supported) {
    auto& v2 = colltracev2::CollTraceV2::instance();
    // The id was minted at the NCCL API entry; a collective's first kernel
    // picks it up and later kernels reuse it. Standalone ctran (no NCCL entry
    // point) mints its own.
    if (kernelConfig.colltraceEmitStart) {
      const auto& commDesc = comm->logMetaData_.commDesc;
      state.v2CollId = v2.getLatestCollId(commDesc);
      if (state.v2CollId == colltracev2::COLLTRACE_INVALID_COLLID) {
        state.v2CollId = v2.newCollId(commDesc);
      }
    }
    hdr = v2.getKernelHandle(state.v2CollId);
    return std::make_unique<DummyCollTraceHandle>();
  }

  // Only graph tracing creates the ring consumed by the kernel.
  const bool inKernel =
      supported && isCapturing && NCCL_COLLTRACE_TRACE_CUDA_GRAPH;
  // Single-kernel collectives emit both boundaries (the KernelConfig defaults);
  // multi-kernel collectives (the AllGatherP pipeline) set these per kernel.
  const bool emitStart = inKernel && kernelConfig.colltraceEmitStart;
  const bool emitEnd = inKernel && kernelConfig.colltraceEmitEnd;
  // Multi-kernel graph collectives create one record at their start boundary.
  const bool createsRecord = !isCapturing || emitStart;
  auto handle = createsRecord
      ? getCollTraceHandle(comm, opGroup, kernelConfig, ifchecksum)
      : nullptr;
  if (inKernel) {
    ctran::gpe::armInKernelColltrace(
        state.pendingGroup, hdr, handle.get(), emitStart, emitEnd);
  }
  return handle;
}

bool testOnlyClearCollTraceRecords(CtranComm* comm) {
  if (comm->colltraceNew_ == nullptr) {
    return false;
  }
  auto commDump = comm->colltraceNew_->getPluginByName(
      std::string{CommDumpPlugin::kCommDumpPluginName});
  if (commDump == nullptr) {
    return false;
  }
  auto commDumpPlugin = dynamic_cast<CommDumpPlugin*>(commDump);
  if (commDumpPlugin == nullptr) {
    return false;
  }

  commDumpPlugin->testOnlyClearColls();
  return true;
}

} // namespace meta::comms::colltrace
