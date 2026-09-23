// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "nccl.h"
#include <cstring>
#include "comm.h"
#include "device.h"
#include "archinfo.h"

#include <folly/DynamicConverter.h>
#include <folly/json.h>

#include "comms/utils/StrUtils.h"

// We turn on the kernel trace by default in rcclx, can be turned off by setting RCCL_KERNEL_COLL_TRACE_ENABLE=0
RCCL_PARAM(KernelCollTraceEnable, "KERNEL_COLL_TRACE_ENABLE", 1);

#ifdef ENABLE_COLLTRACE
std::string ncclCommDumpKernelTrace(const ncclComm_t comm) {
  auto dumpStr = std::string();
  int head[MAXCHANNELS];
  double vega_gpu_rtc_freq;
  vega_gpu_rtc_freq = GetDeviceWallClockRateInKhz(comm->cudaDev) * 1.0E3;
  for (int channel = 0; channel < MAXCHANNELS; channel++) {
    int tail = comm->collTraceTail[channel].tail;
    if (tail < COLLTRACE_NUM_ITEMS)
      head[channel] = 0;
    else
      head[channel] = tail - COLLTRACE_NUM_ITEMS;
  }
  int numActiveChans = MAXCHANNELS;
  INFO(NCCL_COLL, "KL/CL trace fmt: [device_ts] [rank:blockId-chanId:threadId] opCount funcName recvPeer->group_warp_info->sendPeer busId nRanks");
  INFO(NCCL_COLL, "Data trace fmt: [device_ts] [rank:blockId-chanId:threadId] __LINE__ data0 data1");
  for (int channel = 0; channel < MAXCHANNELS; channel++) {
    int tail = comm->collTraceTail[channel].tail;
    int count;
    count = tail - head[channel];
    if (count == 0) {
      numActiveChans--;
      continue;
    }
    for (int i = 0; i < count; i++) {
      volatile struct ncclCollTrace *td = comm->collTrace+COLLTRACE_NUM_ITEMS*channel+head[channel]%COLLTRACE_NUM_ITEMS;
      head[channel] ++;
      uint8_t type = td->type;
      if (type == ncclCollTraceNotReady)
        continue;
      char line[1024];
      int offset = 0;
      uint16_t fIdx = td->funcIndex;
      if (type == ncclCollTraceDataType) {
        sprintf(line, "## [%012.6f] [%02d:%02d-%02d:%02x] L:%04d DT %08x %016lx %016lx",
          (double)(td->timeStamp)/vega_gpu_rtc_freq, comm->rank, td->bid, td->channelId, td->tid,             fIdx, td->data_0, td->opCount, td->data_1);
      } else {
        if (type & ncclCollTraceP2pElemType)
          sprintf(line, "## [%012.6f] [%02d:%02d-%02d:%02x] %06x-%06x", (double)(td->timeStamp)/vega_gpu_rtc_freq, comm->rank, td->bid, td->channelId, td->tid, td->p2pOpCount[0], td->p2pOpCount[1]);
        else
          sprintf(line, "## [%012.6f] [%02d:%02d-%02d:%02x] %06lx", (double)(td->timeStamp)/vega_gpu_rtc_freq, comm->rank, td->bid, td->channelId, td->tid, td->opCount);
        offset = strlen(line);
        if (type == ncclCollTraceCollElemType) {
          sprintf(line+offset, " CE %s nw %d bi %d nc %d root %d busId %lx nRanks %d", funcNames[fIdx], td->coll.nWarps, td->coll.bid, td->coll.nChannels, td->coll.root, comm->busId, comm->nRanks);
        } else if (type == ncclCollTraceP2pElemType) {
             sprintf(line+offset, " Recv %d -> %d/%d/%d/%d ConnIdx/LL/Reg/nc %d/%d/%d/%d -> Send %d cb %d busId %lx nRanks %d",
              td->p2p.recvRank, td->p2p.recvConnIndex, td->p2p.recvProtoLL, td->p2p.recvRegistered, td->p2p.nRecvChannels, td->p2p.sendConnIndex, td->p2p.sendProtoLL, td->p2p.sendRegistered, td->p2p.nSendChannels, td->p2p.sendRank, td->p2p.channelBase,
            comm->busId, comm->nRanks);
        } else {
          switch (type&0xf) {
            case ncclCollTraceKernelLaunchType:
            case ncclCollTraceCollLaunchType:
              if ((type&0xf) == ncclCollTraceKernelLaunchType)
                sprintf(line+offset, " KL HWID %8x %s", td->data_0, funcNames[fIdx]);
              else if ((type&0xf) == ncclCollTraceCollLaunchType)
                  sprintf(line+offset, " CL %d %s", td->batchIx, funcNames[fIdx]);
              offset = strlen(line);
              if ((type&0xf0) == ncclCollTraceCollElemType)
                sprintf(line+offset, " nw %d bi %d nc %d root %d busId %lx nRanks %d", td->coll.nWarps, td->coll.bid, td->coll.nChannels, td->coll.root, comm->busId, comm->nRanks);
              else if ((type&0xf0) == ncclCollTraceP2pElemType)
                  sprintf(line+offset, " Recv %d -> %d/%d/%d/%d ConnIdx/LL/Reg/nc %d/%d/%d/%d -> Send %d cb %d busId %lx nRanks %d",
                    td->p2p.recvRank, td->p2p.recvConnIndex, td->p2p.recvProtoLL, td->p2p.recvRegistered, td->p2p.nRecvChannels, td->p2p.sendConnIndex, td->p2p.sendProtoLL, td->p2p.sendRegistered, td->p2p.nSendChannels, td->p2p.sendRank, td->p2p.channelBase,
                  comm->busId, comm->nRanks);
              break;
            case ncclCollTraceKernelEndType:
              sprintf(line+offset, " KE busId %lx nRanks %d", comm->busId, comm->nRanks);
              break;
            case ncclCollTraceAbortType:
              sprintf(line+offset, " Abort");
              break;
            default:
              sprintf(line+offset, " unknown collective trace data type");
              break;
          }
        }
      }
      INFO(NCCL_COLL, "%s", line);
      td->type = ncclCollTraceNotReady;
    }
  }
  return dumpStr;
}
#else
std::string ncclCommDumpKernelTrace(const ncclComm_t comm) {
  return "Kernel trace is not enabled!";
}
#endif

__attribute__((visibility("default"))) ncclResult_t ncclCommDump(
    const ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  if (comm == nullptr) {
    return ncclSuccess;
  }

  map["commHash"] = toQuotedString(hashToHexStr(comm->commHash));
  map["rank"] = std::to_string(comm->rank);
  map["localRank"] = std::to_string(comm->localRank);
  map["node"] = std::to_string(comm->node);
  map["nRanks"] = std::to_string(comm->nRanks);
  map["localRanks"] = std::to_string(comm->localRanks);
  map["nNodes"] = std::to_string(comm->nNodes);

  if (rcclParamKernelCollTraceEnable()){
    map["KernelTrace"] = ncclCommDumpKernelTrace(comm);
  }

  //
  // Analyzer fields
  //

  if (comm->proxyState->proxyTrace != nullptr) {
    map["ProxyTrace"] = comm->proxyState->proxyTrace->dump();

    auto dump = comm->proxyState->proxyTrace->dumpFormatted(comm->commHash);


    map["PT_activeOps"] = serializeObjects(dump.activeOps);
    map["PT_pastOps"] = serializeObjects(dump.pastOps);
  }

  if (comm->ctrace != nullptr) {
    auto dump = comm->ctrace->dump();

    // Copied from new comm dump implementation. Since we are deprecating
    // old coll trace, we don't care too much about code reuse here.
    map["CT_pastColls"] = folly::toJson(folly::toDynamic(dump.pastColls));
    map["CT_pendingColls"] = folly::toJson(folly::toDynamic(dump.pendingColls));
    if (dump.currentColl != nullptr) {
      map["CT_currentColl"] = folly::toJson(dump.currentColl->toDynamic());
    } else {
      map["CT_currentColl"] = "null";
    }
  }

  return ncclSuccess;
}
