// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// Allocate fixed 2 GiB source/destination buffers, then copy the first --bytes
// bytes using either eager cudaMemcpyAsync submissions or a CUDA graph captured
// from the same memcpy sequence. With --p2p, allocate the source on GPU 0 and
// destination on GPU 1. Keeping the allocation fixed makes copy length the main
// variable.
#define CHECK_CUDA(expr)               \
  do {                                 \
    cudaError_t err__ = (expr);        \
    if (err__ != cudaSuccess) {        \
      std::fprintf(                    \
          stderr,                      \
          "CUDA error at %s:%d: %s\n", \
          __FILE__,                    \
          __LINE__,                    \
          cudaGetErrorString(err__));  \
      std::exit(1);                    \
    }                                  \
  } while (0)

struct Options {
  size_t bytes{512ULL * 1024 * 1024};
  bool graph{false};
  bool p2p{false};
};

constexpr size_t kKiB = 1024;
constexpr size_t kMiB = 1024 * kKiB;
constexpr size_t kGiB = 1024 * kMiB;
constexpr size_t kAllocationBytes = 2 * kGiB;
constexpr int kSrcDevice = 0;
constexpr int kDstDevice = 1;
constexpr int kCopiesPerGraph = 100;
constexpr int kLaunches = 100;
constexpr int kWarmups = 100;

size_t parseBytes(const char* value) {
  char* end = nullptr;
  const unsigned long long base = std::strtoull(value, &end, 0);
  if (end == value) {
    std::fprintf(stderr, "invalid byte count: %s\n", value);
    std::exit(1);
  }
  size_t multiplier = 1;
  if (*end != '\0') {
    const char suffix = static_cast<char>(std::toupper(*end));
    if (end[1] != '\0') {
      std::fprintf(stderr, "invalid byte suffix: %s\n", value);
      std::exit(1);
    }
    if (suffix == 'K') {
      multiplier = 1024ULL;
    } else if (suffix == 'M') {
      multiplier = 1024ULL * 1024;
    } else if (suffix == 'G') {
      multiplier = 1024ULL * 1024 * 1024;
    } else {
      std::fprintf(stderr, "invalid byte suffix: %s\n", value);
      std::exit(1);
    }
  }
  return static_cast<size_t>(base) * multiplier;
}

Options parseOptions(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    auto requireValue = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(1);
      }
      return argv[++i];
    };

    if (std::strcmp(argv[i], "--bytes") == 0) {
      opts.bytes = parseBytes(requireValue(argv[i]));
    } else if (std::strcmp(argv[i], "--graph") == 0) {
      opts.graph = true;
    } else if (std::strcmp(argv[i], "--p2p") == 0) {
      opts.p2p = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      std::exit(1);
    }
  }
  if (opts.bytes > kAllocationBytes) {
    std::fprintf(
        stderr,
        "requested %zu bytes but benchmark allocates only %zu bytes\n",
        opts.bytes,
        kAllocationBytes);
    std::exit(1);
  }
  return opts;
}

void enablePeerAccess(int device, int peer) {
  int canAccessPeer = 0;
  CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, device, peer));
  if (!canAccessPeer) {
    std::fprintf(
        stderr, "device %d cannot access peer device %d\n", device, peer);
    std::exit(1);
  }

  CHECK_CUDA(cudaSetDevice(device));
  const cudaError_t err = cudaDeviceEnablePeerAccess(peer, 0);
  if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
    std::fprintf(
        stderr,
        "CUDA error at %s:%d: %s\n",
        __FILE__,
        __LINE__,
        cudaGetErrorString(err));
    std::exit(1);
  }
}

float timeEagerUsPerCopy(
    void* dst,
    const void* src,
    size_t bytes,
    cudaStream_t stream) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int launch = 0; launch < kLaunches; ++launch) {
    for (int copy = 0; copy < kCopiesPerGraph; ++copy) {
      CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
    }
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms * 1000.0f / (kLaunches * kCopiesPerGraph);
}

void warmupEager(
    void* dst,
    const void* src,
    size_t bytes,
    cudaStream_t stream) {
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    for (int copy = 0; copy < kCopiesPerGraph; ++copy) {
      CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
    }
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
}

float timeGraphUsPerCopy(
    void* dst,
    const void* src,
    size_t bytes,
    cudaStream_t stream) {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
  for (int copy = 0; copy < kCopiesPerGraph; ++copy) {
    CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
  }
  CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
  CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int launch = 0; launch < kLaunches; ++launch) {
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaGraphExecDestroy(graphExec));
  CHECK_CUDA(cudaGraphDestroy(graph));
  return ms * 1000.0f / (kLaunches * kCopiesPerGraph);
}

double gbps(size_t bytes, float us) {
  return static_cast<double>(bytes) / (static_cast<double>(us) * 1000.0);
}

} // namespace

int main(int argc, char** argv) {
  const Options opts = parseOptions(argc, argv);

  if (opts.p2p) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= kDstDevice) {
      std::fprintf(
          stderr,
          "--p2p requires at least %d CUDA devices but found %d\n",
          kDstDevice + 1,
          deviceCount);
      std::exit(1);
    }
    enablePeerAccess(kSrcDevice, kDstDevice);
    enablePeerAccess(kDstDevice, kSrcDevice);
  }

  void* src = nullptr;
  void* dst = nullptr;
  CHECK_CUDA(cudaSetDevice(kSrcDevice));
  CHECK_CUDA(cudaMalloc(&src, kAllocationBytes));
  CHECK_CUDA(cudaMemset(src, 0x5a, kAllocationBytes));

  const int dstDevice = opts.p2p ? kDstDevice : kSrcDevice;
  CHECK_CUDA(cudaSetDevice(dstDevice));
  CHECK_CUDA(cudaMalloc(&dst, kAllocationBytes));
  CHECK_CUDA(cudaMemset(dst, 0, kAllocationBytes));

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaSetDevice(dstDevice));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  float us = 0.0f;
  if (opts.graph) {
    us = timeGraphUsPerCopy(dst, src, opts.bytes, stream);
  } else {
    warmupEager(dst, src, opts.bytes, stream);
    us = timeEagerUsPerCopy(dst, src, opts.bytes, stream);
  }
  std::printf(
      "mode=%s src_device=%d dst_device=%d allocation_bytes=%zu bytes=%zu copies_per_graph=%d launches=%d warmups=%d us_per_copy=%.1f gbps=%.1f\n",
      opts.graph ? "graph" : "eager",
      kSrcDevice,
      dstDevice,
      kAllocationBytes,
      opts.bytes,
      kCopiesPerGraph,
      kLaunches,
      kWarmups,
      us,
      gbps(opts.bytes, us));

  CHECK_CUDA(cudaSetDevice(dstDevice));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaSetDevice(dstDevice));
  CHECK_CUDA(cudaFree(dst));
  CHECK_CUDA(cudaSetDevice(kSrcDevice));
  CHECK_CUDA(cudaFree(src));
  return 0;
}
