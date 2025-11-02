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

} // namespace ctran::algos::MultiTbSyncDev
