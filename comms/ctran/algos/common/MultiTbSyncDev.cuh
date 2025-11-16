#pragma once
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"

/* Define software-based synchronization between multiple thread blocks within
 * the same GPU. Check funtions below for supported sync patterns.
 # How to use:
 * - User creates a shared counter object in global memory.
 * - A given shared counter must be used only by the same set of thread blocks,
 *   and the same sync pattern.
 * - Always reset the counter before use. No thread block should be using the
 *   object before globally reset.
 * - All thread blocks in the sync group must call the sync concurrently and in
 *   the same execution order.
 */
namespace ctran::algos::MultiTbSyncDev {

// Reset the barrier object; should be called by a single thread
template <typename T>
__device__ inline void reset(T* cnt) {
  *cnt = 0;
}

// Full barrier sync: all thread blocks wait for other blocks to arrive
template <typename T>
__device__ inline void
barrier(T* cnt, const int blkIdx, const int numBlks, const int goal) {
  // Do cross-thread-block sync only when there are more than 1 blocks
  if (numBlks > 1) {
    // before thread 0 sign up, ensure all threads in the thread block have
    // arrived
    __syncthreads();

    if (threadIdx.x == 0) {
      atomicAdd(cnt, 1);

      // wait till all workers have added the current step
      while (atomicAdd(cnt, 0) < goal) {
      }
    }
  }

  __syncthreads();
}

// Join sync: all thread blocks sign up and only block 0 waits till all other
// blocks have arrived
template <typename T>
__device__ inline void
join(T* cnt, const int blkIdx, const int numBlks, const int goal) {
  // Do cross-thread-block sync only when there are more than 1 blocks
  if (numBlks > 1) {
    // before thread 0 sign up, ensure all threads in the thread block have
    // arrived
    __syncthreads();

    if (threadIdx.x == 0) {
      // all blocks sign up
      atomicAdd(cnt, 1);

      // block 0 waits till all workers have added the current step;
      if (blkIdx == 0) {
        while (atomicAdd(cnt, 0) < goal) {
        }
      }
    }
  }

  __syncthreads();
}

// Dispatch sync: all thread blocks waits for block 0 to multicast.
template <typename T>
__device__ inline void
dispatch(T* cnt, const int blkIdx, const int numBlks, const int goal) {
  // Do cross-thread-block sync only when there are more than 1 blocks
  if (numBlks > 1 && threadIdx.x == 0) {
    // only block 0 sign up
    if (blkIdx == 0) {
      atomicExch(cnt, goal);
    } else {
      // other blocks wait till block 0 arrived
      while (atomicAdd(cnt, 0) < goal) {
      }
    }
  }

  __syncthreads();
}

// Single thread block signals to the other thread block via a shared counter.
// Optionally skip sync within the thread block by setting withSync
// to false, if callsite has already synced.
// The other thread block should call checkSignal to check if the signal has
// been posted.
// Example usage: producer consumer synchronization
template <typename T, bool withSync>
__device__ inline void signal(T* cnt, const T goal) {
  if (withSync) {
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicExch(cnt, goal);
  }
}

// Single thread block to check a signal from the other thread block via a
// shared counter.
template <typename T>
__device__ inline bool checkSignal(T* cnt, const T goal) {
  __shared__ bool posted;
  if (threadIdx.x == 0) {
    auto cur = utils::atomicRead(cnt);
    posted = cur >= goal;
  }
  __syncthreads();
  return posted;
}

template <typename T>
__device__ inline void waitSignal(T* cnt, const T goal) {
  if (threadIdx.x == 0) {
    T cur;
    do {
      cur = utils::atomicRead(cnt);
    } while (cur < goal);
  }
  __syncthreads();
}

// Broadcast sync: all thread blocks waits for block 0 to multicast, and block 0
// waits for all other blocks to read before returning. It is a combination of
// dispatch and join. Compared to separate calls into dispatch and join, this
// function saves 2 __syncthreads() calls.
template <typename T1, typename T2, typename T3>
__device__ inline T2 bcast(
    T1* dispatchCnt,
    T2* joinCnt,
    T3* valFlag,
    const int blkIdx,
    const int numBlks,
    const T1 dispatchGoal,
    const T2 joinGoal,
    T3& val) {
  __shared__ T3 valShared;
  if (threadIdx.x == 0) {
    // Do cross-thread-block sync only when there are more than 1 blocks
    if (numBlks > 1) {
      // only block 0 sign up and all blocks wait for block 0 to multicast
      if (blkIdx == 0) {
        atomicExch(valFlag, val);
        atomicExch(dispatchCnt, dispatchGoal);
      } else {
        // other blocks wait till block 0 arrived
        while (atomicAdd(dispatchCnt, 0) < dispatchGoal) {
        }
      }
      // load the value
      valShared = atomicAdd(valFlag, 0);

      // block 0 waits till all workers have loaded value
      atomicAdd(joinCnt, 1);
      if (blkIdx == 0) {
        while (atomicAdd(joinCnt, 0) < joinGoal) {
        }
      }
    } else {
      valShared = val;
    }
  }

  __syncthreads();
  val = valShared;
}
} // namespace ctran::algos::MultiTbSyncDev
