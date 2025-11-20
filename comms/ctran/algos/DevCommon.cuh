// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevShmState.cuh"
#include "comms/ctran/algos/common/GpeKernelDev.cuh"
#include "comms/ctran/commstate/CommStateXDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/utils/DevUtils.cuh"

#define CTRAN_DEV_LOG(fmt, ...)                                                                    \
  do {                                                                                             \
    printf(                                                                                        \
        " %s [commHash %lx rank %d localRank %d pid %d blockIdx %d gridDim %d opCount %ld]: " fmt, \
        __func__,                                                                                  \
        statex->commHash(),                                                                        \
        statex->rank(),                                                                            \
        statex->localRank(),                                                                       \
        statex->pid(),                                                                             \
        blockIdx.x,                                                                                \
        gridDim.x,                                                                                 \
        shmDevState.opCount,                                                                       \
        ##__VA_ARGS__);                                                                            \
  } while (0);

#define CTRAN_DEV_TRACE(fmt, ...)                  \
  if (shmDevState.enableTraceLog) {                \
    CTRAN_DEV_LOG("[DEVTRACE]" fmt, ##__VA_ARGS__) \
  }

#define CTRAN_DEV_TRACE_IF(cond, fmt, ...)          \
  if ((cond) && shmDevState.enableTraceLog) {       \
    CTRAN_DEV_LOG("[DEVTRACE]" fmt, ##__VA_ARGS__); \
  }

#define CTRAN_DEV_FATAL(fmt, ...)                                              \
  do {                                                                         \
    CTRAN_DEV_LOG("[DEVFATAL] %s:%d " fmt, __FILE__, __LINE__, ##__VA_ARGS__); \
    trap();                                                                    \
  } while (0);

/**
 * KernelElem access functions
 */

// Free an element so that it can be reclaimed by host pool.
// It should be called only by thread 0 and the caller needs to guarantee it.
// NOTE: if host side needs to follow up after a kernel step, the element should
// not be freed but just completed (see elemComplete).
__device__ __forceinline__ void elemFree(KernelElem* elem, int groupIdx) {
  elem->status[groupIdx] = KernelElem::ElemStatus::RESET;
}

// Complete an element for host side to follow-up.
// It should be called only by thread 0 and the caller needs to guarantee it.
__device__ __forceinline__ void elemComplete(KernelElem* elem, int groupIdx) {
  elem->status[groupIdx] = KernelElem::ElemStatus::DONE;
}

// Wait host side post or revoke the element.
// It should be called only by thread 0 and the caller needs to guarantee it.
__device__ __forceinline__ void
elemWaitPostOrRevoke(KernelElem* elem, int groupIdx, bool* revoked) {
  int val;
  bool aborted;
  do {
    val = ctran::utils::loadInt(&elem->status[groupIdx]);
  } while (KernelElem::ElemStatus::POSTED != val &&
           KernelElem::ElemStatus::REVOKED != val &&
           !(aborted = ctran::device::KernelTestHostAbort(kernelFlag)));
  *revoked = (val == KernelElem::ElemStatus::REVOKED || aborted);
}

// Wrapper of elemFree for entire thread block to call.
// It ensures only thread 0 updates the status after all threads have arrived
__device__ __forceinline__ void elemFreeByGroup(
    KernelElem* elem,
    int groupIdx) {
  __syncthreads();
  if (threadIdx.x == 0) {
    elemFree(elem, groupIdx);
  }
}

// Wrapper of elemComplete for entire thread block to call.
// It ensures only thread 0 updates the status after all threads have arrived
__device__ __forceinline__ void elemCompleteByGroup(
    KernelElem* elem,
    int groupIdx) {
  __syncthreads();
  if (threadIdx.x == 0) {
    elemComplete(elem, groupIdx);
  }
}

// Wrapper of elemWaitPostOrRevoke for entire thread block to call.
// It ensures only thread 0 polls the status and the other threads are waiting
__device__ __forceinline__ void
elemWaitPostOrRevokeByGroup(KernelElem* elem, int groupIdx, bool* revoked) {
  __shared__ bool revoked_;
  if (threadIdx.x == 0) {
    elemWaitPostOrRevoke(elem, groupIdx, &revoked_);
  }
  __syncthreads();
  *revoked = revoked_;
}

// Wrapper of elemWaitPostOrRevoke for entire thread block to call.
// It ensures only thread 0 polls the status and the other threads are waiting
// It also sets and returns recvbuffAddr needed for MultiPut since there is
// already a thread sync
__device__ __forceinline__ uint64_t elemWaitPostOrRevokeByGroupForMultiPut(
    KernelElem* elem,
    int groupIdx,
    bool* revoked) {
  __shared__ bool revoked_;
  __shared__ uint64_t recvbuffAddr;
  if (threadIdx.x == 0) {
    elemWaitPostOrRevoke(elem, groupIdx, &revoked_);
    recvbuffAddr = ctran::utils::loadUint64(&elem->putNotify.recvbuff);
  }
  __syncthreads();
  *revoked = revoked_;
  return recvbuffAddr;
}

// Frees elements in the list.
// If FreeAll is true, free all elements; otherwise free only revoked elements
// since host side no longer tracks & frees it. It ensures only thread 0 updates
// the status after all threads have arrived
__device__ __forceinline__ void elemsFreeListByGroup(
    KernelElem* elemsList,
    int groupIdx,
    bool FreeAll = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    KernelElem* elem = elemsList;
    while (elem != nullptr) {
      if (!FreeAll) {
        int status = ctran::utils::loadInt(&elem->status[groupIdx]);
        if (status == KernelElem::ElemStatus::REVOKED) {
          elemFree(elem, groupIdx);
        }
      } else {
        elemFree(elem, groupIdx);
      }
      elem = elem->next;
    }
  }
}

/**
 * Global state management
 */

__device__ __forceinline__ void
copy(uint4* dst, const uint4* src, size_t count);

static inline __device__ void devStateLoadToShm(
    int* flag,
    CtranAlgoDeviceState* devState) {
  const uint4* devStatePtr = reinterpret_cast<const uint4*>(devState);
#if defined(__HIP_PLATFORM_AMD__)
  uint4* dest = reinterpret_cast<uint4*>(&shmDevState);
#else
  // For CUDA, copy to dynamic shared memory
  uint4* dest = reinterpret_cast<uint4*>(dynamicSharedMem);
#endif

  copy(dest, devStatePtr, sizeof(CtranAlgoDeviceState) / sizeof(uint4));
  if (threadIdx.x == 0) {
    kernelFlag = flag;
    kernelDoAbort = false;
  }
  __syncthreads();

#if defined(__HIP_PLATFORM_AMD__)
  statex = &shmDevState.statex;
#else
  statex = &getShmDevState().statex;
#endif
}

static inline __device__ void devStateLoadToShm(
    CtranAlgoDeviceState* devState) {
  devStateLoadToShm(&placeHolderKernelFlag, devState);
}

/**
 * Cross-GPU device sync functions
 */

typedef enum { LOCAL, REMOTE } DevSyncLoc;

template <DevSyncLoc Loc>
__device__ __forceinline__ CtranAlgoDeviceSync* devSyncGetLoc(
    int peerLocalRank) {
  int localRank = statex->localRank();
  if (Loc == DevSyncLoc::LOCAL) {
    // get local sync shared with the REMOTE rank
    return shmDevState.allPeerSyncMap[localRank][peerLocalRank];
  } else {
    // get remote sync shared with the local rank
    return shmDevState.allPeerSyncMap[peerLocalRank][localRank];
  }
}

// Get devSync from global memory directly without a preloaded device state
// from the shared memory space (shmDevState).
template <DevSyncLoc Loc>
__device__ __forceinline__ CtranAlgoDeviceSync* devSyncGetLoc(
    int peerLocalRank,
    CtranAlgoDeviceState* devState) {
  const auto& statex = devState->statex;
  const auto localRank = statex.localRank();
  if (Loc == DevSyncLoc::LOCAL) {
    // get local sync shared with the REMOTE rank
    return devState->allPeerSyncMap[localRank][peerLocalRank];
  } else {
    // get remote sync shared with the local rank
    return devState->allPeerSyncMap[peerLocalRank][localRank];
  }
}

__device__ __forceinline__ void* devBcastBufGetLoc(int localRank) {
  return shmDevState.peerBcastBufsMap[localRank];
}

// Updates the step by thread 0 for each group
__device__ __forceinline__ void
devSyncSetStep(int* sync, int groupIdx, int val) {
  // ensure all threads have finished before setting the step
  __syncthreads();

  if (threadIdx.x == 0) {
    ctran::utils::storeIntRel(sync, val);

    CTRAN_DEV_TRACE("set step %d to groupIdx %d\n", val, groupIdx);
  }
}

__device__ __forceinline__ void
devSyncSetStep(CtranAlgoDeviceSync* sync, int groupIdx, int val) {
  devSyncSetStep(&sync->syncs[groupIdx].stepOnSameBlockIdx, groupIdx, val);
}

// Receiver waits for sender to update the step, indicating that data has been
// copied into internal buffer for receiver to consume. Only thread 0 from each
// group is responsible for updating the step.
__device__ __forceinline__ void
devSyncWaitStep(int* sync, int groupIdx, int val) {
  if (threadIdx.x == 0) {
    int cur;
    do {
      cur = ctran::utils::loadIntAcq(sync);
    } while (cur != val && !ctran::device::KernelTestHostAbort(kernelFlag));

    CTRAN_DEV_TRACE("waited step %d groupIdx %d\n", val, groupIdx);
  }
  // ensure all threads waiting for thread 0 to check step being updated
  __syncthreads();
}

__device__ __forceinline__ void
devSyncWaitStep(CtranAlgoDeviceSync* sync, int groupIdx, int val) {
  devSyncWaitStep(&sync->syncs[groupIdx].stepOnSameBlockIdx, groupIdx, val);
}

// Sender posts a notification to the remote REMOTE. Thread 0 from each thread
// block updates the flag
__device__ __forceinline__ void devSyncSetNotify(
    CtranAlgoDeviceSync* sync,
    int groupIdx) {
  // Ensure all threads have finished before setting the notification
  __syncthreads();

  if (threadIdx.x == 0) {
    // First wait for remote REMOTE to reset, in case the flag was still used by
    // a previous put to the same peer
    int cur;
    do {
      cur = ctran::utils::loadIntAcq(&sync->syncs[groupIdx].stepOnSameBlockIdx);
    } while (cur != CTRAN_ALGO_NOTIFY_RESET &&
             !ctran::device::KernelTestHostAbort(kernelFlag));

    // Update notification
    ctran::utils::storeIntRel(
        &sync->syncs[groupIdx].stepOnSameBlockIdx, CTRAN_ALGO_NOTIFY_SET);
  }
}

// Receiver waits for sender to notify after finished put. Except only thread 0
// from a single thread block calls it and wait for the notification from all
// remote groups
__device__ __forceinline__ void devSyncWaitNotify(
    CtranAlgoDeviceSync* sync,
    int nGroups) {
  for (uint32_t groupIdx = threadIdx.x; groupIdx < nGroups;
       groupIdx += blockDim.x) {
    // Wait notification from each remote group
    int cur;
    do {
      cur = ctran::utils::loadIntAcq(&sync->syncs[groupIdx].stepOnSameBlockIdx);
    } while (cur != CTRAN_ALGO_NOTIFY_SET &&
             !ctran::device::KernelTestHostAbort(kernelFlag));

    // Mark notify has been used, thus peer can use for next notify
    ctran::utils::storeIntRel(
        &sync->syncs[groupIdx].stepOnSameBlockIdx, CTRAN_ALGO_NOTIFY_RESET);
  }

  // Ensure all threads waiting for thread 0 to check step being updated
  __syncthreads();
}

/**
 * Copy functions
 */

__device__ __forceinline__ void
copy(uint4* dst, const uint4* src, size_t count) {
  constexpr int kUnroll = 8;

  uint32_t totalThreads = blockDim.x;
  uint32_t blockSize = totalThreads * kUnroll;
  uint32_t globalId = threadIdx.x;

  const size_t limitCount = ctran::utils::roundDown(count, size_t(blockSize));

  uint4 v[kUnroll];

  // unrolled portion
  for (size_t i = globalId; i < limitCount; i += blockSize) {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      v[j] = src[i + j * totalThreads];
    }

#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      dst[i + j * totalThreads] = v[j];
    }
  }

  if constexpr (kUnroll > 1) {
    // remainder epilogue
    for (size_t i = limitCount + globalId; i < count; i += totalThreads) {
      dst[i] = src[i];
    }
  }
}

__device__ __forceinline__ void
copy(uint4* dst, const volatile uint4* src, size_t count) {
  constexpr int kUnroll = 8;

  uint32_t totalThreads = blockDim.x;
  uint32_t blockSize = totalThreads * kUnroll;
  uint32_t globalId = threadIdx.x;

  const size_t limitCount = ctran::utils::roundDown(count, size_t(blockSize));

  uint4 v[kUnroll];

  // unrolled portion
  for (size_t i = globalId; i < limitCount; i += blockSize) {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      v[j] = const_cast<const uint4&>(src[i + j * totalThreads]);
    }

#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      dst[i + j * totalThreads] = v[j];
    }
  }

  if constexpr (kUnroll > 1) {
    // remainder epilogue
    for (size_t i = limitCount + globalId; i < count; i += totalThreads) {
      dst[i] = const_cast<const uint4&>(src[i]);
    }
  }
}

// Thread blocks with the same groupIdx must own the same area of shared memory
// in order to avoid race conditions. When a send and receive have mismatching
// data types (one will have uint4 and the other will have T), each thread
// must access only the shared memory that it would in the uint4 case.
template <int Unroll16, typename T>
__device__ __forceinline__ void
copyUnroll(T* dst, const T* src, size_t count, int groupIdx, int nGroups) {
  // How many Ts are in a uint4-equivalent
  constexpr int kTPer16Bytes = sizeof(uint4) / sizeof(T);

  // Unroll16 is how many uint4-equivalents we load/store each iteration
  constexpr int kUnroll = kTPer16Bytes * Unroll16;

  auto numPerBlock = blockDim.x * kUnroll;
  auto limitUnroll = ctran::utils::roundDown(count, numPerBlock);

  // Some number of CTAs (groupIdx) will fit into this loop, and we may have
  // some remainder which is guaranteed to lie in a single CTA (groupIdx), as
  // limitUnroll is a multiple of the number of T elements handled per groupIdx
  //
  // As per the comment at the top of the function, we guarantee that each
  // groupIdx handles a contiguous block of blockDim.x * Unroll16 * 16 bytes of
  // data regardless of data type T passed, thus guaranteeing the same
  // writer groupIdx would always touch the same region of memory.

  for (size_t i = groupIdx * numPerBlock + threadIdx.x; i < limitUnroll;
       i += nGroups * numPerBlock) {
    T v[kUnroll];

#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      v[j] = src[i + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      dst[i + j * blockDim.x] = v[j];
    }
  }

  // remainder epilogue

  // To avoid issues of trying to slice up the remaining data among CTAs and
  // having different CTAs access different regions of memory, just give it to
  // the group that is responsible for that part of the temporary
  // buffer.
  // The maximum number of bytes remaining to copy is
  // less than blockDim.x * kUnroll * sizeof(T)
  // e.g., 640 x 4 x 16 = less than 40 kiB, so not huge
  if (count != limitUnroll &&
      groupIdx == ((limitUnroll / numPerBlock) % nGroups)) {
    for (size_t i = limitUnroll + threadIdx.x; i < count; i += blockDim.x) {
      dst[i] = src[i];
    }
  }
}

template <typename T>
__device__ __forceinline__ void
copy(T* dst, const T* src, size_t count, int groupIdx, int nGroups) {
  // Skip if my group is not invovled
  if (groupIdx >= nGroups) {
    return;
  }

  // Each thread handles 4 x uint4 = 64 bytes per loop iteration, regardless
  // of sizeof(T)
  copyUnroll<4, T>(dst, src, count, groupIdx, nGroups);
}

template <typename T>
__device__ __forceinline__ void copyWarp(
    T* dst,
    const T* src,
    const size_t count,
    const int warpId = 0,
    const int numWarps = 1) {
  const auto laneId = warpId * kWarpSize + threadIdx.x & (kWarpSize - 1);
  for (size_t i = laneId; i < count; i += kWarpSize * numWarps) {
    dst[i] = src[i];
  }
}

// Checks whether the buffer and size are aligned to 16 bytes so that 16B
// aligned copy (i.e., copy<uint4>) can be used. It checks only one buffer as
// the other side is NCCL internal buffer which is always 16B aligned.
template <typename T>
__device__ __forceinline__ bool canCopy16(const T* buf, size_t count) {
  bool bufAligned = ((uintptr_t)buf % 16) == 0;
  bool sizeAligned = ((size_t)count * sizeof(T) % 16) == 0;
  return bufAligned && sizeAligned;
}

// Checks whether two-sides buffers and size are aligned to 16 bytes so that 16B
// aligned copy (i.e., copy<uint4>) can be used. It is used in self-copy or NVL
// zero-copy cases where both buffers are provided by user.
template <typename T>
__device__ __forceinline__ bool
canCopy16(const T* sendbuff, T* recvbuff, size_t count) {
  bool sendBuffAligned = ((uintptr_t)sendbuff % 16) == 0;
  bool recvBuffAligned = ((uintptr_t)recvbuff % 16) == 0;
  bool sizeAligned = ((size_t)count * sizeof(T) % 16) == 0;
  return sendBuffAligned && recvBuffAligned && sizeAligned;
}

// Checks whether a single buffer's address is aligned to 16 bytes
template <typename T>
__device__ __forceinline__ bool canCopy16(const T* buf) {
  uintptr_t x = reinterpret_cast<uintptr_t>(buf) & (sizeof(uint4) - 1);
  return x == 0;
}

/**
 * Kernel operator load functions
 */

template <typename T>
static inline __device__ void loadAlgoDevArg(T& dArg, volatile T* arg) {
#if defined(__HIP_PLATFORM_AMD__)
  // TODO: Check if this is a correct and efficient way to do this.
  // Without this, HIP makes error: "initialization is not supported for
  // __shared__ variables".
  __shared__ char tmpBuffer[sizeof(T)];
  T* tmp = reinterpret_cast<T*>(tmpBuffer);
  uint4* tmpPtr = reinterpret_cast<uint4*>(tmp);
#else
  __shared__ T tmp;

  // Parallel H2D load to temporary buffer in shared memory
  uint4* tmpPtr = reinterpret_cast<uint4*>(&tmp);
#endif

  const volatile uint4* argPtr =
      const_cast<const volatile uint4*>(reinterpret_cast<volatile uint4*>(arg));
  copy(tmpPtr, argPtr, sizeof(T) / sizeof(uint4));
  __syncthreads();

  // Each thread D2D copies to final local destination
  uint4* dArgPtr = reinterpret_cast<uint4*>(&dArg);
  for (size_t idx = 0; idx < sizeof(T) / sizeof(uint4); idx++) {
    dArgPtr[idx] = tmpPtr[idx];
  }
}

template <typename T>
static inline __device__ void
loadAlgoDevVecPtrs(const T** dPtrs, const void* volatile* sPtrs, int nvector) {
  __shared__ const T* tmp[CTRAN_MAX_NVL_PEERS];

  // Parallel H2D load to temporary buffer in shared memory
  const auto gtIdx = threadIdx.x;
  for (size_t idx = gtIdx; idx < nvector; idx += blockDim.x) {
    tmp[idx] = reinterpret_cast<const T*>(sPtrs[idx]);
  }
  __syncthreads();

  // Each thread copies to final local destination
  for (int idx = 0; idx < nvector; idx++) {
    dPtrs[idx] = tmp[idx];
  }
}

template <typename T>
static inline __device__ void
loadAlgoDevVecPtr(T** dPtrs, void* volatile* sPtrs, int nvector) {
  __shared__ T* tmp[CTRAN_MAX_NVL_PEERS];

  // Parallel H2D load to temporary buffer in shared memory
  const auto gtIdx = threadIdx.x;
  for (size_t idx = gtIdx; idx < nvector; idx += blockDim.x) {
    tmp[idx] = reinterpret_cast<T*>(sPtrs[idx]);
  }
  __syncthreads();

  // Each thread copies to final local destination
  for (int idx = 0; idx < nvector; idx++) {
    dPtrs[idx] = tmp[idx];
  }
}

__device__ __forceinline__ void trap() {
#if defined(__HIP_PLATFORM_AMD__)
  __builtin_trap();
#else
  asm("trap;");
#endif
}

__device__ __forceinline__ void syncwarp() {
#if defined(__HIP_PLATFORM_AMD__)
  __builtin_amdgcn_wave_barrier();
#else
  __syncwarp();
#endif
}
