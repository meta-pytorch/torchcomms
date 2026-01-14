/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "nccl.h"
#include "comm.h"
#include "net.h"
#include "register.h"
#include "transport.h"
#include "group.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/wrapper/MetaFactory.h"

// conflict with ctran, disable for now.
NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 0);

ncclResult_t ncclRegLocalIsValid(struct ncclReg *reg, bool *isValid) {
  if (reg && isValid) {
    if (reg->localRefs)
      *isValid = true;
    else
      *isValid = false;
  }
  return ncclSuccess;
}

ncclResult_t ncclRegister(struct ncclComm* comm, void* data, size_t size, bool isGraph, void** handle) {
  NCCLCHECK(CommCheck(comm, "ncclCommRegister", "comm"));
  struct ncclRegCache* cache = &comm->regCache;
  uintptr_t pageSize = cache->pageSize;
  uintptr_t begAddr = (uintptr_t)data & -pageSize;
  uintptr_t endAddr = ((uintptr_t)data + size + pageSize-1) & -pageSize;

  if (comm->checkPointers) NCCLCHECK(CudaPtrCheck(data, comm, "buff", "ncclCommRegister"));
  INFO(NCCL_REG, "register comm %p buffer %p size %zi", comm, data, size);

  for (int slot=0; /*true*/; slot++) {
    if ((slot == cache->population) || (begAddr < cache->slots[slot]->begAddr)) {
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      memmove(cache->slots+slot+1, cache->slots+slot, (cache->population-slot)*sizeof(struct ncclReg*));
      NCCLCHECK(ncclCalloc(cache->slots+slot, 1));
      struct ncclReg* regSlot = cache->slots[slot];
      regSlot->begAddr = begAddr;
      regSlot->endAddr = endAddr;
      if (isGraph) regSlot->graphRefs = 1;
      else regSlot->localRefs = 1;
      cache->population += 1;
      *handle = regSlot;
      goto exit;
    } else if ((cache->slots[slot]->begAddr <= begAddr) &&
               (cache->slots[slot]->endAddr >= endAddr)) {
      if (isGraph) cache->slots[slot]->graphRefs++;
      else cache->slots[slot]->localRefs++;
      *handle = cache->slots[slot];
      goto exit;
    }
  }

exit:
  return ncclSuccess;
}

static ncclResult_t regCleanup(struct ncclComm* comm, struct ncclReg* reg) {
  if (reg->state & NET_REG_COMPLETE) {
    struct ncclRegNetHandles* netHandle = reg->netHandleHead;
    struct ncclRegNetHandles* netHandlePrev;
    while(netHandle) {
      if (ncclNetDeregBuffer(comm, netHandle->proxyConn, netHandle->handle) != ncclSuccess) {
        WARN("rank %d deregister NET buffer handle %p proxy rank %d failed\n", comm->rank, netHandle->handle, netHandle->proxyConn->rank);
      }
      netHandlePrev = netHandle;
      netHandle = netHandle->next;
      free(netHandlePrev);
    }
  }
  if (reg->state & NVLS_REG_COMPLETE) {
    if (ncclNvlsDeregBuffer(comm, &reg->mcHandle, reg->regAddr, reg->dev, reg->regUCSize, reg->regMCSize) != ncclSuccess) {
      WARN("rank %d deregister NVLS buffer %p dev %d ucsize %ld mcsize %ld failed", comm->rank, (void*)reg->regAddr, reg->dev, reg->regUCSize, reg->regMCSize);
    }
    reg->regAddr = (CUdeviceptr)NULL;
  }
  if (reg->state & COLLNET_REG_COMPLETE) {
    if (ncclCollnetDeregBuffer(comm, reg->collnetProxyconn, reg->collnetHandle) != ncclSuccess) {
      WARN("rank %d deregister COLLNET buffer handle %p proxy rank %d failed", comm->rank, reg->collnetHandle, reg->collnetProxyconn->rank);
    }
  }
  if (reg->state & IPC_REG_COMPLETE) {
    for (int i = 0; i < NCCL_MAX_LOCAL_RANKS; ++i)
      if (reg->ipcInfos[i]) {
        if (ncclIpcDeregBuffer(comm, reg->ipcInfos[i]) != ncclSuccess) {
          WARN("rank %d deregister IPC buffer %p peerRank %d failed", comm->rank, reg->ipcInfos[i]->baseAddr, reg->ipcInfos[i]->peerRank);
        }
        free(reg->ipcInfos[i]);
      }
    if (reg->regIpcAddrs.hostPeerRmtAddrs) free(reg->regIpcAddrs.hostPeerRmtAddrs);
    if (reg->regIpcAddrs.devPeerRmtAddrs) NCCLCHECK(ncclCudaFree(reg->regIpcAddrs.devPeerRmtAddrs));
  }
  return ncclSuccess;
}

ncclResult_t ncclRegCleanup(struct ncclComm* comm) {
  struct ncclRegCache* cache = &comm->regCache;
  for (int i = 0; i < cache->population; i++) {
    struct ncclReg* reg = cache->slots[i];
    INFO(NCCL_INIT, "Cleanup buffer %p pages %lx", (void*)reg->begAddr, (reg->endAddr-reg->begAddr)/cache->pageSize);
    NCCLCHECK(regCleanup(comm, reg));
    free(reg);
  }
  free(cache->slots);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  auto timerBegin = std::chrono::steady_clock::now();
  // FIXME: we should eventually support hybrid registration.
  // Disable it for now to avoid undefined behavior due to handle conflict.
  if (NCCL_CTRAN_REGISTER != NCCL_CTRAN_REGISTER::none && ncclParamLocalRegister()) {
    ERR("Invalid usage to turn on NCCL_CTRAN_REGISTER and NCCL_LOCAL_REGISTER at the same time.");
    return metaCommToNccl(ErrorStackTraceUtil::log(commInvalidUsage));
  }

  if (ctranInitialized(comm->ctranComm_.get()) &&
      NCCL_CTRAN_REGISTER != NCCL_CTRAN_REGISTER::none) {
    NCCLCHECK(metaCommToNccl(comm->ctranComm_->ctran_->commRegister(buff, size, handle)));
  } else if (!ncclParamLocalRegister() || ncclP2pUsesMemcpy()) {
    *handle = NULL;
    INFO(NCCL_REG, "Skipping registration for buffer %p size %zi (LocalRegister=%ld, P2pUsesMemcpy=%d)",
         buff, size, ncclParamLocalRegister(), ncclP2pUsesMemcpy());
  } else {
    NCCLCHECK(ncclRegister(comm, buff, size, false, handle));
  }

  INFO(
      NCCL_ALLOC,
      "REGISTER: user registered buffer %p len %ld hdl %p with commHash %lx commDesc %s",
      buff,
      size,
      *handle,
      comm->logMetaData.commHash,
      comm->logMetaData.commDesc.c_str());
  if (NCCL_COMM_REGISTER_LOG_ENABLE) {
      logMemoryEvent(
        comm->logMetaData,
        "",
        "ncclCommRegister",
        reinterpret_cast<uintptr_t>(*handle),
        size,
        0,
        std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - timerBegin)
          .count(),
          true /* isRegMemEvent */);
  }
  return ncclSuccess;
}

ncclResult_t ncclCommGraphRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  if (ncclP2pUsesMemcpy()) {
    *handle = NULL;
    INFO(NCCL_REG, "Skipping graph registration for buffer %p size %zi (P2pUsesMemcpy=%d)",
         buff, size, ncclP2pUsesMemcpy());
  } else {
    NCCLCHECK(ncclRegister(comm, buff, size, true, handle));
  }
  return ncclSuccess;
}

static ncclResult_t commDeregister(struct ncclComm *comm, bool isGraph, struct ncclReg* reg) {
  NCCLCHECK(CommCheck(comm, "ncclCommRegister", "comm"));
  struct ncclRegCache* cache = &comm->regCache;
  int slot;
  int saveDev;
  if (reg == NULL) goto exit;
  CUDACHECK(cudaGetDevice(&saveDev));
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  for (slot = 0; slot < cache->population && cache->slots[slot] != reg; slot++);
  if (slot == cache->population) {
    WARN("Deregister: Could not find handle");
    return ncclInvalidUsage;
  }
  if (isGraph) --reg->graphRefs;
  else --reg->localRefs;
  if (reg->localRefs || reg->graphRefs) return ncclSuccess;
  NCCLCHECK(regCleanup(comm, reg));
  free(reg);
  memmove(cache->slots + slot, cache->slots + slot + 1, (cache->population - slot - 1) * sizeof(struct ncclReg*));
  cache->population -= 1;
  CUDACHECK(cudaSetDevice(saveDev));
exit:
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
  auto timerBegin = std::chrono::steady_clock::now();

  // FIXME: we should eventually support hybrid registration.
  // Disable it for now to avoid undefined behavior due to handle conflict.
  if (NCCL_CTRAN_REGISTER != NCCL_CTRAN_REGISTER::none && ncclParamLocalRegister()) {
    ERR("Invalid usage to turn on NCCL_CTRAN_REGISTER and NCCL_LOCAL_REGISTER at the same time.");
    return metaCommToNccl(ErrorStackTraceUtil::log(commInvalidUsage));
  }

  /* handles are only valid if either ctran registration or baseline
   * registration are set */
  if ((ctranInitialized(comm->ctranComm_.get()) &&
       NCCL_CTRAN_REGISTER != NCCL_CTRAN_REGISTER::none) ||
      ncclParamLocalRegister()) {
    NCCLCHECK(CommCheck(comm, "ncclCommRegister", "comm"));
    NCCLCHECK(PtrCheck(handle, "ncclCommDeregister", "handle"));
  }

  if (ctranInitialized(comm->ctranComm_.get()) &&
      NCCL_CTRAN_REGISTER != NCCL_CTRAN_REGISTER::none) {
    NCCLCHECK(metaCommToNccl(comm->ctranComm_->ctran_->commDeregister(handle)));
  } else if (ncclParamLocalRegister()) {
  NCCLCHECK(commDeregister(comm, false, (struct ncclReg*)handle));
  }

  INFO(
      NCCL_ALLOC,
      "REGISTER: user deregistered hdl %p with commHash %lx commDesc %s",
      handle,
      comm->logMetaData.commHash,
      comm->logMetaData.commDesc.c_str());
  if (NCCL_COMM_REGISTER_LOG_ENABLE) {
    logMemoryEvent(
        comm->logMetaData,
        "",
        "ncclCommDeregister",
        reinterpret_cast<uintptr_t>(handle),
        0,
        0,
        std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - timerBegin)
          .count(),
          true /* isRegMemEvent */);
  }
  return ncclSuccess;
}

ncclResult_t ncclCommGraphDeregister(const ncclComm_t comm, struct ncclReg *handle) {
  NCCLCHECK(commDeregister(comm, true, handle));
  return ncclSuccess;
}
