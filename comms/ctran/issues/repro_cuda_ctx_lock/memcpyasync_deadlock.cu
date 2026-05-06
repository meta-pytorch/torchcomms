// Copyright (c) Meta Platforms, Inc. and affiliates.
// SPDX-License-Identifier: Apache-2.0

// Demo 2 of 2 (see also memcpyasync_blocked.cu).
//
// Turns the lock contention from memcpyasync_blocked.cu into a true,
// permanent deadlock by replacing the bounded busy kernel with a
// *persistent* kernel that only exits when the host writes shutdown=1
// via cudaMemcpyAsync.
//
// Cycle (LAZY mode):
//   persistentKernel polls *d_shutdown on the GPU
//        ^                                            |
//        | needs shutdown=1 to exit                   |
//        |                                            v
//   GPU is busy ----> cuLibraryLoadData waits for GPU
//                          |
//                          | holds primary context lock
//                          v
//                 cudaMemcpyAsync (delivers shutdown=1) blocked on lock
//                          |
//                          v
//                  shutdown=1 never reaches the GPU
//
// A watchdog thread declares the deadlock and force-exits the process
// after 30 s, since std::_Exit is the only safe way out of a wedged
// CUDA process.
//
// Build:
//   make memcpyasync_deadlock
// Run:
//   CUDA_MODULE_LOADING=LAZY  ./memcpyasync_deadlock   (deadlocks; watchdog
//   fires) CUDA_MODULE_LOADING=EAGER ./memcpyasync_deadlock   (completes in
//   <1s)
//
// Background: P2291856308.

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

__global__ void persistentKernel(volatile int* shutdown);
__global__ void firstTimeKernel();

static double secs(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t)
      .count();
}

int main() {
  cudaFree(nullptr);
  cudaStream_t sA, sB;
  cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking);
  int* d_shutdown;
  cudaMalloc(&d_shutdown, sizeof(int));
  cudaMemset(d_shutdown, 0, sizeof(int));

  // Force line-buffered stdout so prints are visible even if the process
  // is force-exited from the watchdog before stdio buffers get flushed.
  setvbuf(stdout, nullptr, _IOLBF, 0);

  // Watchdog: if no progress in 30s, declare deadlock and force-exit.
  std::thread([] {
    std::this_thread::sleep_for(std::chrono::seconds(30));
    std::printf("\nDEADLOCK DETECTED: process wedged for 30s. Forcing exit.\n");
    std::fflush(stdout);
    std::_Exit(0);
  }).detach();

  // 1. Launch persistent kernel. It will spin until *d_shutdown != 0.
  persistentKernel<<<1, 1, 0, sA>>>(d_shutdown);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // 2. Spawn child thread to deliver the shutdown signal via
  //    cudaMemcpyAsync. In LAZY mode this call blocks on the primary
  //    context lock and never reaches the GPU.
  std::thread b([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    int one = 1;
    auto t = std::chrono::steady_clock::now();
    cudaMemcpyAsync(d_shutdown, &one, sizeof(int), cudaMemcpyHostToDevice, sB);
    std::printf(
        "[B] cudaMemcpyAsync (shutdown=1) returned in %.3fs\n", secs(t));
  });

  // 3. Main thread: first-time launch of firstTimeKernel grabs the
  //    primary context lock and blocks in cuLibraryLoadData waiting for
  //    the GPU to free up. The GPU will only free up if shutdown=1
  //    reaches it -- which can only happen via the cudaMemcpyAsync that
  //    is itself blocked on this lock.
  auto t = std::chrono::steady_clock::now();
  firstTimeKernel<<<1, 1, 0, sA>>>();
  std::printf("[A] cudaLaunchKernel returned in %.3fs\n", secs(t));

  b.join();
  cudaDeviceSynchronize();
  std::printf("Process completed without deadlock.\n");
}
