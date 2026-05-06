// Copyright (c) Meta Platforms, Inc. and affiliates.
// SPDX-License-Identifier: Apache-2.0

// Demo 1 of 2 (see also memcpyasync_deadlock.cu).
//
// Shows that cudaMemcpyAsync can block on the host for arbitrary time while
// another thread's cudaLaunchKernel is stuck inside the lazy
// cuLibraryLoadData path holding the CUDA primary context lock. See
// README.md for full background.
//
// Build:
//   make memcpyasync_blocked
// Run:
//   CUDA_MODULE_LOADING=LAZY ./memcpyasync_blocked
//
// busyKernel and firstTimeKernel live in separate translation units so each
// gets its own lazily-loaded CUDA module. The main thread launches
// busyKernel (loads + runs while the GPU is idle), then launches
// firstTimeKernel for the first time -- its module isn't loaded yet, so the
// runtime calls cuLibraryLoadData under the primary context lock, which
// then waits for the GPU to free up. A child thread's cudaMemcpyAsync on a
// separate stream blocks until the lock is released.

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <thread>

__global__ void busyKernel(long long ticks);
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
  int* d;
  cudaMalloc(&d, sizeof(int));

  // Get busyKernel running on the GPU. The first launch also lazy-loads its
  // module, but the GPU is idle so it's fast.
  busyKernel<<<1, 1, 0, sA>>>(20'000'000'000LL);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Spawn child thread; it sleeps 50ms before timing cudaMemcpyAsync.
  // Meanwhile the main thread (below) immediately enters
  // firstTimeKernel<<<>>>, which takes microseconds to acquire the context
  // lock and start waiting on cuLibraryLoadData. By the time the child's
  // sleep ends, main is firmly inside the locked driver call, so the
  // child's memcpy blocks on the same lock.
  std::thread b([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    int v = 0;
    auto t = std::chrono::steady_clock::now();
    cudaMemcpyAsync(d, &v, sizeof(int), cudaMemcpyHostToDevice, sB);
    std::printf("[B] cudaMemcpyAsync returned in   %.3fs\n", secs(t));
  });

  auto t = std::chrono::steady_clock::now();
  firstTimeKernel<<<1, 1, 0, sA>>>();
  std::printf("[A] cudaLaunchKernel returned in %.3fs\n", secs(t));

  b.join();
  cudaDeviceSynchronize();
}
