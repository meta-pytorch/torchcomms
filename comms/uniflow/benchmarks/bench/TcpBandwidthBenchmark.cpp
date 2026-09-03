// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/bench/TcpBandwidthBenchmark.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

#include <fmt/format.h>
#include "comms/uniflow/Segment.h"
#include "comms/uniflow/benchmarks/Rendezvous.h"
#include "comms/uniflow/benchmarks/SegmentHelper.h"
#include "comms/uniflow/benchmarks/Stats.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/tcp/TcpTransport.h"

namespace uniflow::benchmark {

namespace {

// Fixed so a verification failure reproduces exactly; logged with the banner.
constexpr uint64_t kVerifySeed = 0x5EED1234;

/// First global (non-link-local) IPv6 address on `iface`, or "" if none.
std::string getInterfaceIpv6(const std::string& iface) {
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) != 0) {
    return "";
  }
  std::string result;
  for (auto* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET6 ||
        iface != ifa->ifa_name) {
      continue;
    }
    auto* sa = reinterpret_cast<sockaddr_in6*>(ifa->ifa_addr);
    if (sa->sin6_addr.s6_addr[0] == 0xfe &&
        (sa->sin6_addr.s6_addr[1] & 0xc0) == 0x80) {
      continue; // link-local
    }
    char buf[INET6_ADDRSTRLEN] = {};
    if (inet_ntop(AF_INET6, &sa->sin6_addr, buf, sizeof(buf)) != nullptr) {
      result = buf;
      break;
    }
  }
  freeifaddrs(ifaddr);
  return result;
}

// A src/dst buffer pair (DRAM or VRAM), freed on destruction.
struct BufferPair {
  void* src{nullptr};
  void* dst{nullptr};
  bool useGpu{false};
  MemoryType memType{MemoryType::DRAM};
  int device{0};

  BufferPair() = default;
  BufferPair(const BufferPair&) = delete;
  BufferPair& operator=(const BufferPair&) = delete;

  ~BufferPair() {
    if (useGpu) {
      if (src) {
        (void)cudaFree(src);
      }
      if (dst) {
        (void)cudaFree(dst);
      }
    } else {
      std::free(src);
      std::free(dst);
    }
  }
};

bool allocPair(BufferPair& bufs, size_t maxSize, int cudaDevice) {
  bufs.useGpu = cudaDevice >= 0;
  bufs.memType = bufs.useGpu ? MemoryType::VRAM : MemoryType::DRAM;
  bufs.device = bufs.useGpu ? cudaDevice : 0;

  if (bufs.useGpu) {
    if (cudaSetDevice(bufs.device) != cudaSuccess ||
        cudaMalloc(&bufs.src, maxSize) != cudaSuccess ||
        cudaMalloc(&bufs.dst, maxSize) != cudaSuccess) {
      UNIFLOW_LOG_ERROR("TcpBandwidthBenchmark: cudaMalloc failed");
      return false;
    }
    if (cudaMemset(bufs.src, 0xAB, maxSize) != cudaSuccess ||
        cudaMemset(bufs.dst, 0x00, maxSize) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      UNIFLOW_LOG_ERROR("TcpBandwidthBenchmark: cuda buffer init failed");
      return false;
    }
  } else {
    bufs.src = std::malloc(maxSize);
    bufs.dst = std::malloc(maxSize);
    if (bufs.src == nullptr || bufs.dst == nullptr) {
      UNIFLOW_LOG_ERROR("TcpBandwidthBenchmark: malloc failed");
      return false;
    }
    std::memset(bufs.src, 0xAB, maxSize);
    std::memset(bufs.dst, 0x00, maxSize);
  }
  return true;
}

// [uint64_t dstAddr | registration payload]
std::vector<uint8_t> serializeReg(
    uint64_t dstAddr,
    const std::vector<uint8_t>& regPayload) {
  std::vector<uint8_t> buf(sizeof(dstAddr) + regPayload.size());
  std::memcpy(buf.data(), &dstAddr, sizeof(dstAddr));
  std::memcpy(
      buf.data() + sizeof(dstAddr), regPayload.data(), regPayload.size());
  return buf;
}

struct TransferResult {
  bool ok{false};
  double bandwidthGBs{0};
  double messageRateMops{0};
  int totalOps{0};
  std::vector<double> latenciesUs;
};

// Warmup then a pipelined timed loop keeping up to txDepth put/get calls in
// flight over the single TCP connection.
TransferResult runTransfer(
    Transport& transport,
    RegisteredSegment& localReg,
    RemoteRegisteredSegment& remoteReg,
    size_t size,
    const std::string& dir,
    const BenchmarkConfig& config) {
  using Clock = std::chrono::steady_clock;
  const int batchSize = std::max(1, config.batchSize);
  const int txDepth = std::max(1, config.txDepth);

  std::vector<TransferRequest> batch;
  batch.reserve(batchSize);
  for (int i = 0; i < batchSize; ++i) {
    batch.push_back(
        TransferRequest{
            .local = localReg.span(size_t{0}, size),
            .remote = remoteReg.span(size_t{0}, size)});
  }

  const int numBatches =
      std::max(1, (config.iterations + batchSize - 1) / batchSize);
  const int totalOps = numBatches * batchSize;

  auto submit = [&]() -> std::future<Status> {
    return (dir == "put") ? transport.put(batch, {}) : transport.get(batch, {});
  };

  TransferResult out;

  for (int i = 0; i < config.warmupIterations; ++i) {
    if (submit().get().hasError()) {
      UNIFLOW_LOG_ERROR(
          "TcpBandwidthBenchmark: warmup {} failed at size {}", dir, size);
      return out;
    }
  }

  std::deque<std::pair<std::future<Status>, Clock::time_point>> inflight;
  std::vector<double> latenciesUs;
  latenciesUs.reserve(numBatches);

  auto completeOne = [&]() -> bool {
    auto& [fut, submitTime] = inflight.front();
    auto status = fut.get();
    auto done = Clock::now();
    if (status.hasError()) {
      inflight.pop_front();
      UNIFLOW_LOG_ERROR(
          "TcpBandwidthBenchmark: {} failed at size {}: {}",
          dir,
          size,
          status.error().message());
      for (auto& [f, _] : inflight) {
        f.wait();
      }
      return false;
    }
    latenciesUs.push_back(
        std::chrono::duration<double, std::micro>(done - submitTime).count() /
        batchSize);
    inflight.pop_front();
    return true;
  };

  auto start = Clock::now();
  for (int b = 0; b < numBatches; ++b) {
    if (static_cast<int>(inflight.size()) >= txDepth) {
      if (!completeOne()) {
        return out;
      }
    }
    // Timestamp before submit(), not as a sibling argument to emplace_back:
    // argument evaluation order is unspecified, so Clock::now() could be
    // evaluated after submit() returns. That matters now that put() applies
    // backpressure and blocks inside submit() while enqueueing chunks -- the
    // blocked time would fall outside the measured interval and the reported
    // latency would be a fraction of the real one (bandwidth, bracketed by
    // start/end, stayed correct).
    auto submitTime = Clock::now();
    inflight.emplace_back(submit(), submitTime);
  }
  while (!inflight.empty()) {
    if (!completeOne()) {
      return out;
    }
  }
  auto end = Clock::now();

  double sec = std::chrono::duration<double>(end - start).count();
  double bytes = static_cast<double>(size) * static_cast<double>(totalOps);
  out.ok = true;
  out.bandwidthGBs = (sec > 0) ? (bytes / sec) / 1e9 : 0;
  out.messageRateMops = (sec > 0) ? (totalOps / sec) / 1e6 : 0;
  out.totalOps = totalOps;
  out.latenciesUs = std::move(latenciesUs);
  return out;
}

// Byte value at an absolute offset. The period far exceeds any transfer here,
// so a chunk landing at the wrong offset -- or duplicated, dropped, or
// reordered -- changes the bytes. A pattern repeating every 256 bytes (say
// `offset & 0xFF`) is blind to whole-chunk misplacement, because chunk
// boundaries are multiples of 256. `round` is mixed in so each pass differs.
uint8_t patternAt(uint64_t offset, uint64_t round) {
  uint64_t x = offset + 0x9E3779B97F4A7C15ULL * (round + 1);
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDULL;
  x ^= x >> 33;
  x *= 0xC4CEB9FE1A85EC53ULL;
  x ^= x >> 33;
  return static_cast<uint8_t>(x);
}

// Transfer sizes for the correctness sweep, clamped to maxLen. Boundaries are
// enumerated rather than sampled: every failure this transport has had sits on
// one, and a uniform draw over [1, 1 GiB] puts ~99% of its samples above 2 MiB,
// so it would neither reliably hit a boundary nor cover small transfers.
std::vector<size_t> verifySizes(size_t maxLen) {
  constexpr size_t kChunk = 4UL * 1024 * 1024; // kMaxChunkSize in TcpTransport
  constexpr size_t kFrameCap = 64UL * 1024 * 1024; // kMaxFrameSize
  std::vector<size_t> sizes{
      1,
      3,
      4095, // unaligned tails
      4096,
      4097,
      kChunk - 1, // chunk boundary: +1 leaves a 1-byte final chunk
      kChunk,
      kChunk + 1,
      2 * kChunk, // exact multiple: no remainder chunk at all
      2 * kChunk + 1,
      kFrameCap - 1, // wire-frame cap edges
      kFrameCap,
      kFrameCap + 1,
      kFrameCap + kChunk, // just past the outbound queue cap
      2 * kFrameCap, // 128 MiB: a reply-path cap here killed the connection
  };

  // Log-uniform so small sizes actually appear. Fixed seed: a failure must
  // reproduce, and the seed is logged with the phase banner.
  std::mt19937_64 rng(kVerifySeed);
  std::uniform_real_distribution<double> logDist(
      0.0, std::log(static_cast<double>(maxLen)));
  for (int i = 0; i < 8; ++i) {
    sizes.push_back(
        std::max<size_t>(1, static_cast<size_t>(std::exp(logDist(rng)))));
  }

  std::erase_if(sizes, [maxLen](size_t s) { return s > maxLen; });
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  return sizes;
}

// Host<->buffer copies that work for both DRAM and VRAM buffers.
bool writeBuf(
    const BufferPair& bufs,
    size_t offset,
    const uint8_t* src,
    size_t len) {
  auto* dst = static_cast<uint8_t*>(bufs.src) + offset;
  if (!bufs.useGpu) {
    std::copy_n(src, len, dst);
    return true;
  }
  return cudaMemcpy(dst, src, len, cudaMemcpyHostToDevice) == cudaSuccess &&
      cudaDeviceSynchronize() == cudaSuccess;
}

bool readBuf(const BufferPair& bufs, size_t offset, uint8_t* dst, size_t len) {
  const auto* src = static_cast<const uint8_t*>(bufs.src) + offset;
  if (!bufs.useGpu) {
    std::copy_n(src, len, dst);
    return true;
  }
  return cudaMemcpy(dst, src, len, cudaMemcpyDeviceToHost) == cudaSuccess &&
      cudaDeviceSynchronize() == cudaSuccess;
}

// Boundary + randomized size/offset sweep with content verification, run before
// the timed loop. A bandwidth number says nothing about the bytes: a transport
// that moved garbage, or nothing at all, still reports excellent throughput.
//
// Each case is a round trip -- put local src into the peer's dst, zero src,
// then get it back -- so no peer-side code is needed; the passive rank just
// serves. Known blind spot: put and get address the remote with the same
// offset, so a transfer placed at a uniformly shifted remote offset would
// cancel out. Chunk misordering within a transfer does not cancel, because
// put's Write path and get's ReadRequest/ReadReply path reassemble
// independently.
bool verifyTransfers(
    Transport& transport,
    RegisteredSegment& localReg,
    RemoteRegisteredSegment& remoteReg,
    const BufferPair& bufs,
    size_t maxSize,
    int rank) {
  const std::vector<size_t> sizes = verifySizes(maxSize);
  std::mt19937_64 offsetRng(kVerifySeed);
  std::vector<uint8_t> expected;
  std::vector<uint8_t> actual;
  uint64_t round = 0;
  size_t cases = 0;

  UNIFLOW_LOG_WARN(
      "[rank {}] verify: {} sizes x up to 2 offsets, seed {:#x} "
      "(--no-verify to skip)",
      rank,
      sizes.size(),
      kVerifySeed);

  for (size_t len : sizes) {
    // Offset 0 always; plus one unaligned offset when the segment has room, so
    // the responder's `offset <= entry->len - len` bounds check and the
    // `baseOffset + off` chunk arithmetic are exercised away from 0.
    std::vector<size_t> offsets{0};
    if (maxSize > len) {
      offsets.push_back(
          std::uniform_int_distribution<size_t>(1, maxSize - len)(offsetRng));
    }

    for (size_t offset : offsets) {
      ++round;
      ++cases;
      expected.resize(len);
      for (size_t i = 0; i < len; ++i) {
        expected[i] = patternAt(offset + i, round);
      }
      if (!writeBuf(bufs, offset, expected.data(), len)) {
        UNIFLOW_LOG_ERROR("verify: staging src failed (len {})", len);
        return false;
      }

      std::vector<TransferRequest> req{TransferRequest{
          .local = localReg.span(offset, len),
          .remote = remoteReg.span(offset, len)}};
      if (auto st = transport.put(req, {}).get(); st.hasError()) {
        UNIFLOW_LOG_ERROR(
            "verify: put failed at len {} offset {}: {}",
            len,
            offset,
            st.error().message());
        return false;
      }

      // Zero src so the get has to refill it: otherwise a get that moved
      // nothing would pass against the bytes put() just staged there.
      std::vector<uint8_t> zeros(len, 0);
      if (!writeBuf(bufs, offset, zeros.data(), len)) {
        UNIFLOW_LOG_ERROR("verify: zeroing src failed (len {})", len);
        return false;
      }
      if (auto st = transport.get(req, {}).get(); st.hasError()) {
        UNIFLOW_LOG_ERROR(
            "verify: get failed at len {} offset {}: {}",
            len,
            offset,
            st.error().message());
        return false;
      }

      actual.assign(len, 0);
      if (!readBuf(bufs, offset, actual.data(), len)) {
        UNIFLOW_LOG_ERROR("verify: reading src back failed (len {})", len);
        return false;
      }
      if (actual != expected) {
        size_t bad = 0;
        while (bad < len && actual[bad] == expected[bad]) {
          ++bad;
        }
        UNIFLOW_LOG_ERROR(
            "verify: DATA MISMATCH at len {} offset {}: first bad byte {} "
            "(absolute {}), expected {:#x} got {:#x}",
            len,
            offset,
            bad,
            offset + bad,
            expected[bad],
            actual[bad]);
        return false;
      }
    }
  }

  UNIFLOW_LOG_WARN("[rank {}] verify: {} cases passed", rank, cases);
  return true;
}

} // namespace

std::vector<BenchmarkResult> TcpBandwidthBenchmark::run(
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap) {
  if (peers.empty()) {
    UNIFLOW_LOG_WARN("TcpBandwidthBenchmark: no peers, skipping");
    return {};
  }

  const std::string host = getInterfaceIpv6(iface_);
  if (host.empty()) {
    UNIFLOW_LOG_ERROR(
        "TcpBandwidthBenchmark: no global IPv6 on interface '{}'", iface_);
    return {};
  }
  UNIFLOW_LOG_WARN(
      "TcpBandwidthBenchmark: rank {} using {} address {}",
      bootstrap.rank,
      iface_,
      host);

  const int dev = config.cudaDevice;
  BufferPair bufs;
  if (!allocPair(bufs, config.maxSize, dev)) {
    return {};
  }

  ScopedEventBaseThread evbThread("bench-tcp-evb");
  controller::TcpSocketConfig sockConfig;
  sockConfig.socketBufSize = sockBufSize_;
  auto factory = std::make_unique<TcpTransportFactory>(
      dev, evbThread.getEventBase(), sockConfig, host);

  // Handshake over the rendezvous control channel.
  auto localTopo = factory->getTopology();
  auto remoteTopo =
      exchangeMetadata(*peers[0].ctrl, localTopo, bootstrap.isRank0());
  if (!remoteTopo) {
    UNIFLOW_LOG_ERROR("TcpBandwidthBenchmark: topology exchange failed");
    return {};
  }
  auto transportResult =
      factory->createTransport(std::move(remoteTopo).value());
  if (!transportResult) {
    UNIFLOW_LOG_ERROR(
        "TcpBandwidthBenchmark: createTransport failed: {}",
        transportResult.error().message());
    return {};
  }
  auto transport = std::move(transportResult).value();

  auto localInfo = transport->bind();
  auto remoteInfo =
      exchangeMetadata(*peers[0].ctrl, localInfo, bootstrap.isRank0());
  if (!remoteInfo) {
    UNIFLOW_LOG_ERROR("TcpBandwidthBenchmark: transport info exchange failed");
    transport->shutdown();
    return {};
  }
  if (auto st = transport->connect(std::move(remoteInfo).value());
      st.hasError()) {
    UNIFLOW_LOG_ERROR(
        "TcpBandwidthBenchmark: connect failed: {}", st.error().message());
    transport->shutdown();
    return {};
  }

  // Register src/dst, exchange the dst registration so each side can address
  // the peer's destination segment.
  Segment srcSeg(bufs.src, config.maxSize, bufs.memType, bufs.device);
  Segment dstSeg(bufs.dst, config.maxSize, bufs.memType, bufs.device);
  auto srcReg = factory->registerSegment(srcSeg);
  auto dstReg = factory->registerSegment(dstSeg);
  if (!srcReg || !dstReg) {
    UNIFLOW_LOG_ERROR("TcpBandwidthBenchmark: registerSegment failed");
    transport->shutdown();
    return {};
  }
  auto localPayload = serializeReg(
      reinterpret_cast<uint64_t>(bufs.dst), dstReg.value()->serialize());
  auto remotePayload =
      exchangeMetadata(*peers[0].ctrl, localPayload, bootstrap.isRank0());
  if (!remotePayload || remotePayload.value().size() < sizeof(uint64_t)) {
    UNIFLOW_LOG_ERROR("TcpBandwidthBenchmark: registration exchange failed");
    transport->shutdown();
    return {};
  }
  uint64_t remoteDstAddr = 0;
  std::memcpy(
      &remoteDstAddr, remotePayload.value().data(), sizeof(remoteDstAddr));
  std::vector<uint8_t> remoteRegPayload(
      remotePayload.value().begin() + sizeof(uint64_t),
      remotePayload.value().end());
  auto remoteHandle =
      factory->importSegment(config.maxSize, std::move(remoteRegPayload));
  if (!remoteHandle) {
    UNIFLOW_LOG_ERROR(
        "TcpBandwidthBenchmark: importSegment failed: {}",
        remoteHandle.error().message());
    transport->shutdown();
    return {};
  }

  auto localReg =
      SegmentTest::makeRegistered(srcSeg, std::move(srcReg.value()));
  auto remoteReg = SegmentTest::makeRemote(
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      reinterpret_cast<void*>(remoteDstAddr),
      config.maxSize,
      std::move(remoteHandle.value()));
  auto localDstHandle = std::move(dstReg.value()); // keep alive

  // Message-size sweep.
  std::vector<BenchmarkResult> results;
  const bool isActiveRank = config.bidirectional || bootstrap.isRank0();
  auto sizes = generateSizes(config.minSize, config.maxSize);

  // Correctness before performance. Both ranks must reach the same barriers, so
  // the passive rank waits here while the active rank runs the sweep.
  //
  // A rank that gives up on its own work latches a flag and keeps meeting the
  // peer at every rendezvous below, doing no work and recording no results.
  // Returning instead would leave the peer waiting at one it never reaches, and
  // the peer does not fail fast: barrier() retries EAGAIN 300 times against a
  // 30s SO_RCVTIMEO, so a stranded rank looks hung for hours, not seconds.
  bool verifyFailed = false;
  if (config.verify) {
    if (!barrier(peers, bootstrap)) {
      UNIFLOW_LOG_ERROR("TcpBandwidthBenchmark: pre-verify barrier failed");
      transport->shutdown();
      return {};
    }
    if (isActiveRank &&
        !verifyTransfers(
            *transport,
            localReg,
            remoteReg,
            bufs,
            config.maxSize,
            bootstrap.rank)) {
      UNIFLOW_LOG_ERROR(
          "TcpBandwidthBenchmark: correctness sweep FAILED; not reporting "
          "bandwidth for a transport that does not move bytes correctly");
      verifyFailed = true;
    }
    if (!barrier(peers, bootstrap)) {
      UNIFLOW_LOG_ERROR("TcpBandwidthBenchmark: post-verify barrier failed");
      transport->shutdown();
      return {};
    }
  }

  auto runDirection = [&](const std::string& dir) {
    // Seeded from the verify result: a failed correctness sweep still walks the
    // whole size sweep, meeting every barrier and reporting nothing.
    //
    // A failing barrier() is the exception and does return: the rendezvous
    // channel is gone, and every later one would only time out in turn.
    bool aborted = verifyFailed;

    for (auto size : sizes) {
      if (!barrier(peers, bootstrap)) {
        UNIFLOW_LOG_ERROR("TcpBandwidthBenchmark: barrier failed");
        return;
      }
      if (aborted || !isActiveRank) {
        continue;
      }
      // Bracket the timed loop so the phase split covers the same frames the
      // reported bandwidth does, warmup included -- runTransfer does its own
      // warmup internally, so the reset has to sit before the call, not inside.
      auto* tcpTransport =
          dynamic_cast<::uniflow::TcpTransport*>(transport.get());
      if (tcpTransport != nullptr) {
        tcpTransport->logAndResetPhaseStats("reset");
      }
      auto r = runTransfer(*transport, localReg, remoteReg, size, dir, config);
      if (!r.ok) {
        // Latched rather than returned: every remaining size has a barrier the
        // peer is going to execute.
        aborted = true;
        continue;
      }
      if (tcpTransport != nullptr) {
        tcpTransport->logAndResetPhaseStats(
            fmt::format("{} size={}", dir, size));
      }
      auto stats = Stats::compute(std::move(r.latenciesUs));
      results.push_back({
          .benchmarkName = name(),
          .transport = "tcp",
          .direction = dir,
          .messageSize = size,
          .iterations = r.totalOps,
          .batchSize = std::max(1, config.batchSize),
          .txDepth = std::max(1, config.txDepth),
          .chunkSize = config.chunkSize,
          .bandwidthGBs = r.bandwidthGBs,
          .latency = stats,
          .messageRateMops = r.messageRateMops,
      });
      UNIFLOW_LOG_WARN(
          "[rank {}] {} size={:<10} iface={} batch={:<3} txdepth={:<3} "
          "iters={:<6} bw={:.2f} GB/s avg={:.1f} us {}",
          bootstrap.rank,
          dir,
          size,
          iface_,
          std::max(1, config.batchSize),
          std::max(1, config.txDepth),
          r.totalOps,
          r.bandwidthGBs,
          stats.avg,
          config.bidirectional ? "(bidirectional)" : "(unidirectional)");
    }
  };

  if (config.direction == "put" || config.direction == "both") {
    runDirection("put");
  }
  if (config.direction == "get" || config.direction == "both") {
    runDirection("get");
  }

  if (!barrier(peers, bootstrap)) {
    UNIFLOW_LOG_WARN("TcpBandwidthBenchmark: final barrier failed");
  }
  transport->shutdown();
  return results;
}

} // namespace uniflow::benchmark
