/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meta/hierarchical/hierarchical_allreduce.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <utility>

#include "checks.h"
#include "collectives.h"
#include "comm.h"
#include "debug.h"
#include "group.h"
#include "meta/hierarchical/hierarchical_geometry.h"
#include "meta/relay/sharded_relay_allreduce_kernels.h"
#include "meta/relay/sharded_relay_graph_scratch.h"
#include "param.h"

namespace rcclx::hier {

namespace {

// Off unless set to 1: ncclAllReduce stays on the standard path by default.
NCCL_PARAM(HierAllReduce, "HIER_ALLREDUCE", 0);
NCCL_PARAM(HierAllReduceMinBytes, "HIER_ALLREDUCE_MIN_BYTES", 256 * 1024);
NCCL_PARAM(
    HierAllReduceTileBytes,
    "HIER_ALLREDUCE_TILE_BYTES",
    128 * 1024 * 1024);
// High enough that the tile size, not the cap, sets the tile count: a cap that
// binds silently makes tiles larger than the tuned size.
NCCL_PARAM(HierAllReduceMaxTiles, "HIER_ALLREDUCE_MAX_TILES", 256);
// Communicators of at most one full 8-GPU node stay on the intra-node
// allreduce even when they span hosts.
NCCL_PARAM(HierAllReduceMinRanks, "HIER_ALLREDUCE_MIN_RANKS", 9);
// Beyond 2 nodes, hierarchy saves too few network bytes for the Socket
// transport to beat the standard allreduce; raise this to engage on more.
NCCL_PARAM(HierAllReduceMaxNodes, "HIER_ALLREDUCE_MAX_NODES", 2);

std::atomic<uint64_t> engageCount{0};

enum class Decision {
  Engage,
  Disabled,
  SingleNode,
  TooManyNodes,
  FewRanks,
  NonUniform,
  InGroup,
  NonBlocking,
  Op,
  Datatype,
  Size,
};

const char* decisionName(Decision d) {
  switch (d) {
    case Decision::Engage:
      return "engage";
    case Decision::Disabled:
      return "disabled";
    case Decision::SingleNode:
      return "single node";
    case Decision::TooManyNodes:
      return "more nodes than NCCL_HIER_ALLREDUCE_MAX_NODES";
    case Decision::FewRanks:
      return "too few ranks";
    case Decision::NonUniform:
      return "non-uniform ranks per node";
    case Decision::InGroup:
      return "inside a user ncclGroup";
    case Decision::NonBlocking:
      return "non-blocking communicator";
    case Decision::Op:
      return "unsupported reduction op";
    case Decision::Datatype:
      return "unsupported datatype";
    case Decision::Size:
      return "message too small";
  }
  return "unknown";
}

// One INFO line per (communicator, outcome), so a sweep shows both where the
// path declines and that it engages, without logging every call.
bool firstTime(const ncclComm* comm, Decision d) {
  static std::mutex mu;
  static std::set<std::pair<uint64_t, int>> seen;
  std::lock_guard<std::mutex> lock(mu);
  return seen.insert({comm->commHash, static_cast<int>(d)}).second;
}

bool datatypeSupported(ncclDataType_t dt) {
  switch (dt) {
    case ncclInt8:
    case ncclUint8:
    case ncclInt32:
    case ncclUint32:
    case ncclInt64:
    case ncclUint64:
    case ncclFloat16:
    case ncclFloat32:
    case ncclFloat64:
    case ncclBfloat16:
      return true;
    default:
      return false;
  }
}

Decision
decide(size_t count, ncclDataType_t dt, ncclRedOp_t op, const ncclComm* comm) {
  if (ncclParamHierAllReduce() != 1 || comm == nullptr) {
    return Decision::Disabled;
  }
  if (comm->nNodes < 2) {
    return Decision::SingleNode;
  }
  if (comm->nNodes > ncclParamHierAllReduceMaxNodes()) {
    return Decision::TooManyNodes;
  }
  if (comm->nRanks < ncclParamHierAllReduceMinRanks()) {
    return Decision::FewRanks;
  }
  if (comm->minLocalRanks != comm->maxLocalRanks ||
      comm->nRanks != comm->nNodes * comm->localRanks) {
    return Decision::NonUniform;
  }
  // Inside a user group the inner ncclGroupEnd only defers, so the reduce
  // kernels issued between our groups would run before the transfers.
  if (ncclGroupDepth != 0) {
    return Decision::InGroup;
  }
  if (!comm->config.blocking) {
    return Decision::NonBlocking;
  }
  if (op != ncclSum && op != ncclAvg) {
    return Decision::Op;
  }
  if (!datatypeSupported(dt)) {
    return Decision::Datatype;
  }
  const size_t bytes = count * static_cast<size_t>(ncclTypeSize(dt));
  if (bytes < static_cast<size_t>(ncclParamHierAllReduceMinBytes()) ||
      !geometryFeasible(count, comm->nNodes, comm->localRanks)) {
    return Decision::Size;
  }
  return Decision::Engage;
}

/**
 * Grow-only scratch keyed by (device, stream), duplicated from the relay's
 * per-TU ScratchBufferCache. Captures get a graph-scoped buffer instead, since
 * a stream-ordered allocation inside a capture is only valid while that graph
 * runs; see sharded_relay_graph_scratch.h.
 */
class ScratchCache {
 public:
  static ScratchCache& instance() {
    static ScratchCache cache;
    return cache;
  }

  void* get(size_t bytes, cudaStream_t stream, int graphUsageMode) {
    struct ncclCudaGraph graph;
    if (ncclCudaGetCapturingGraph(&graph, stream, graphUsageMode) !=
        ncclSuccess) {
      return nullptr;
    }
    if (ncclCudaGraphValid(graph)) {
      return rcclx::relay::graphScratchGet(this, 0, bytes, stream, graph);
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(mu_);
    Entry& e = buffers_[{device, static_cast<const void*>(stream)}];
    if (e.ptr != nullptr && e.bytes >= bytes) {
      return e.ptr;
    }
    if (e.ptr != nullptr) {
      cudaFreeAsync(e.ptr, stream);
    }
    e = Entry{};
    if (cudaMallocAsync(&e.ptr, bytes, stream) != cudaSuccess) {
      e = Entry{};
      return nullptr;
    }
    e.bytes = bytes;
    return e.ptr;
  }

 private:
  ScratchCache() = default;

  struct Entry {
    void* ptr{nullptr};
    size_t bytes{0};
  };

  std::mutex mu_;
  std::map<std::pair<int, const void*>, Entry> buffers_;
};

template <typename T>
void launchReduce(
    void* dst,
    const void* seed,
    const void* contribs,
    int numContribs,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  if (seed == dst) {
    launchMultiReduceKernel<T>(
        dst, contribs, numContribs, count, divisor, stream);
  } else {
    launchSeededMultiReduceKernel<T>(
        dst, seed, contribs, numContribs, count, divisor, stream);
  }
}

// dst = (seed + sum of numContribs contiguous blocks of count) / divisor.
void reduce(
    ncclDataType_t dt,
    void* dst,
    const void* seed,
    const void* contribs,
    int numContribs,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  switch (dt) {
    case ncclInt8:
      launchReduce<int8_t>(
          dst, seed, contribs, numContribs, count, divisor, stream);
      break;
    case ncclUint8:
      launchReduce<uint8_t>(
          dst, seed, contribs, numContribs, count, divisor, stream);
      break;
    case ncclInt32:
      launchReduce<int32_t>(
          dst, seed, contribs, numContribs, count, divisor, stream);
      break;
    case ncclUint32:
      launchReduce<uint32_t>(
          dst, seed, contribs, numContribs, count, divisor, stream);
      break;
    case ncclInt64:
      launchReduce<int64_t>(
          dst, seed, contribs, numContribs, count, divisor, stream);
      break;
    case ncclUint64:
      launchReduce<uint64_t>(
          dst, seed, contribs, numContribs, count, divisor, stream);
      break;
    case ncclFloat16:
      launchReduce<__half>(
          dst, seed, contribs, numContribs, count, divisor, stream);
      break;
    case ncclFloat32:
      launchReduce<float>(
          dst, seed, contribs, numContribs, count, divisor, stream);
      break;
    case ncclFloat64:
      launchReduce<double>(
          dst, seed, contribs, numContribs, count, divisor, stream);
      break;
    case ncclBfloat16:
      launchReduce<__nv_bfloat16>(
          dst, seed, contribs, numContribs, count, divisor, stream);
      break;
    default:
      break;
  }
}

constexpr size_t kScratchAlignBytes = 256;

size_t alignUp(size_t v, size_t a) {
  return (v + a - 1) / a * a;
}

/**
 * The pipelined schedule. Stage order within every group is fixed on every
 * rank (intra RS, inter RS/exchange, inter AG, intra AG), which is what keeps
 * multiple sends to the same peer inside one group matched.
 *
 * In-place is safe without copies: every region a stage overwrites was sent
 * by an earlier, already completed group.
 */
class Schedule {
 public:
  Schedule(
      const void* sendbuff,
      void* recvbuff,
      ncclDataType_t dt,
      ncclRedOp_t op,
      ncclComm_t comm,
      cudaStream_t stream,
      const Geometry& geo)
      : send_{static_cast<const char*>(sendbuff)},
        recv_{static_cast<char*>(recvbuff)},
        dt_{dt},
        es_{static_cast<size_t>(ncclTypeSize(dt))},
        comm_{comm},
        stream_{stream},
        geo_{geo},
        node_{comm->node},
        local_{comm->localRank},
        divisor_{op == ncclAvg ? comm->nRanks : 1} {}

  size_t scratchABytes() const {
    return alignUp(
        static_cast<size_t>(geo_.nLocal - 1) * geo_.maxShard(local_) * es_,
        kScratchAlignBytes);
  }

  size_t scratchBBytes() const {
    const size_t elems = exchange()
        ? geo_.maxShard(local_)
        : static_cast<size_t>(geo_.nNodes - 1) * geo_.maxSub(local_, node_);
    return elems * es_;
  }

  ncclResult_t run(char* scratch) {
    scratchA_ = scratch;
    scratchB_ = scratch + scratchABytes();
    const int stages = exchange() ? 3 : 4;
    for (int i = 0; i < geo_.nTiles + stages - 1; i++) {
      NCCLCHECK(ncclGroupStart());
      if (valid(i)) {
        NCCLCHECK(postIntraReduceScatter(i));
      }
      if (exchange()) {
        if (valid(i - 1)) {
          NCCLCHECK(postInterExchange(i - 1));
        }
        if (valid(i - 2)) {
          NCCLCHECK(postIntraAllGather(i - 2));
        }
      } else {
        if (valid(i - 1)) {
          NCCLCHECK(postInterReduceScatter(i - 1));
        }
        if (valid(i - 2)) {
          NCCLCHECK(postInterAllGather(i - 2));
        }
        if (valid(i - 3)) {
          NCCLCHECK(postIntraAllGather(i - 3));
        }
      }
      NCCLCHECK(ncclGroupEnd());
      if (valid(i)) {
        reduceIntra(i);
      }
      if (valid(i - 1)) {
        reduceInter(i - 1);
      }
    }
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }

 private:
  bool exchange() const {
    return geo_.inter() == InterAlgo::Exchange;
  }
  bool valid(int t) const {
    return t >= 0 && t < geo_.nTiles;
  }
  bool inPlace() const {
    return send_ == recv_;
  }
  const char* sendAt(size_t elems) const {
    return send_ + elems * es_;
  }
  char* recvAt(size_t elems) const {
    return recv_ + elems * es_;
  }
  int intraPeer(int j) const {
    return comm_->nodeRanks[node_].localRankToRank[j];
  }
  int interPeer(int m) const {
    return comm_->nodeRanks[m].localRankToRank[local_];
  }
  ncclResult_t send(const void* buf, size_t count, int peer) const {
    return ncclSend(buf, count, dt_, peer, comm_, stream_);
  }
  ncclResult_t recv(void* buf, size_t count, int peer) const {
    return ncclRecv(buf, count, dt_, peer, comm_, stream_);
  }

  // Shard k of my input to local rank k; L-1 copies of my shard into scratchA.
  ncclResult_t postIntraReduceScatter(int t) const {
    const size_t mine = geo_.shard(t, local_).count;
    int slot = 0;
    for (int k = 0; k < geo_.nLocal; k++) {
      if (k == local_) {
        continue;
      }
      const Piece p = geo_.shard(t, k);
      NCCLCHECK(send(sendAt(p.offset), p.count, intraPeer(k)));
      NCCLCHECK(recv(scratchA_ + slot * mine * es_, mine, intraPeer(k)));
      slot++;
    }
    return ncclSuccess;
  }

  // Node-partial of my shard, seeded from my own input.
  void reduceIntra(int t) const {
    if (geo_.nLocal == 1 && inPlace()) {
      return;
    }
    const Piece p = geo_.shard(t, local_);
    reduce(
        dt_,
        recvAt(p.offset),
        sendAt(p.offset),
        scratchA_,
        geo_.nLocal - 1,
        p.count,
        1,
        stream_);
  }

  ncclResult_t postInterExchange(int t) const {
    const Piece p = geo_.shard(t, local_);
    const int peer = interPeer(1 - node_);
    NCCLCHECK(send(recvAt(p.offset), p.count, peer));
    NCCLCHECK(recv(scratchB_, p.count, peer));
    return ncclSuccess;
  }

  ncclResult_t postInterReduceScatter(int t) const {
    const size_t mine = geo_.sub(t, local_, node_).count;
    int slot = 0;
    for (int m = 0; m < geo_.nNodes; m++) {
      if (m == node_) {
        continue;
      }
      const Piece p = geo_.sub(t, local_, m);
      NCCLCHECK(send(recvAt(p.offset), p.count, interPeer(m)));
      NCCLCHECK(recv(scratchB_ + slot * mine * es_, mine, interPeer(m)));
      slot++;
    }
    return ncclSuccess;
  }

  // Final value of the piece this rank owns, with the Avg divisor folded in.
  // For the 2-node exchange both nodes compute own + peer; IEEE addition of
  // two terms is commutative, so both get identical bits.
  void reduceInter(int t) const {
    const Piece p =
        exchange() ? geo_.shard(t, local_) : geo_.sub(t, local_, node_);
    reduce(
        dt_,
        recvAt(p.offset),
        recvAt(p.offset),
        scratchB_,
        geo_.nNodes - 1,
        p.count,
        divisor_,
        stream_);
  }

  ncclResult_t postInterAllGather(int t) const {
    const Piece mine = geo_.sub(t, local_, node_);
    for (int m = 0; m < geo_.nNodes; m++) {
      if (m == node_) {
        continue;
      }
      const Piece p = geo_.sub(t, local_, m);
      NCCLCHECK(send(recvAt(mine.offset), mine.count, interPeer(m)));
      NCCLCHECK(recv(recvAt(p.offset), p.count, interPeer(m)));
    }
    return ncclSuccess;
  }

  ncclResult_t postIntraAllGather(int t) const {
    const Piece mine = geo_.shard(t, local_);
    for (int k = 0; k < geo_.nLocal; k++) {
      if (k == local_) {
        continue;
      }
      const Piece p = geo_.shard(t, k);
      NCCLCHECK(send(recvAt(mine.offset), mine.count, intraPeer(k)));
      NCCLCHECK(recv(recvAt(p.offset), p.count, intraPeer(k)));
    }
    return ncclSuccess;
  }

  const char* send_;
  char* recv_;
  ncclDataType_t dt_;
  size_t es_;
  ncclComm_t comm_;
  cudaStream_t stream_;
  Geometry geo_;
  int node_;
  int local_;
  int divisor_;
  char* scratchA_{nullptr};
  char* scratchB_{nullptr};
};

} // namespace

bool hierAllReduceEligible(
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm) {
  const Decision d = decide(count, datatype, op, comm);
  if (d == Decision::Engage) {
    return true;
  }
  if (d != Decision::Disabled && d != Decision::SingleNode &&
      firstTime(comm, d)) {
    INFO(
        NCCL_COLL,
        "Hierarchical allreduce: declined on comm %lx (%s), first seen at %zu bytes",
        static_cast<unsigned long>(comm->commHash),
        decisionName(d),
        count * static_cast<size_t>(ncclTypeSize(datatype)));
  }
  return false;
}

ncclResult_t hierAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream) {
  const Geometry geo{
      count,
      comm->nNodes,
      comm->localRanks,
      chooseTiles(
          count,
          static_cast<size_t>(ncclTypeSize(datatype)),
          comm->nNodes,
          comm->localRanks,
          static_cast<size_t>(ncclParamHierAllReduceTileBytes()),
          static_cast<int>(ncclParamHierAllReduceMaxTiles()))};
  Schedule schedule(sendbuff, recvbuff, datatype, op, comm, stream, geo);

  const size_t scratchBytes =
      schedule.scratchABytes() + schedule.scratchBBytes();
  void* scratch = ScratchCache::instance().get(
      scratchBytes, stream, comm->config.graphUsageMode);
  if (scratch == nullptr) {
    WARN(
        "Hierarchical allreduce: failed to get %zu bytes of scratch",
        scratchBytes);
    return ncclSystemError;
  }

  if (firstTime(comm, Decision::Engage)) {
    INFO(
        NCCL_COLL,
        "Hierarchical allreduce: engaged on comm %lx, %d nodes x %d local ranks, net %s, inter %s, first at %zu bytes with %d tiles",
        static_cast<unsigned long>(comm->commHash),
        geo.nNodes,
        geo.nLocal,
        comm->ncclNet != nullptr ? comm->ncclNet->name : "none",
        geo.inter() == InterAlgo::Exchange ? "exchange"
                                           : "reduce-scatter+all-gather",
        count * static_cast<size_t>(ncclTypeSize(datatype)),
        geo.nTiles);
  }
  NCCLCHECK(schedule.run(static_cast<char*>(scratch)));
  engageCount.fetch_add(1, std::memory_order_relaxed);
  return ncclSuccess;
}

uint64_t hierAllReduceEngageCount() {
  return engageCount.load(std::memory_order_relaxed);
}

} // namespace rcclx::hier
