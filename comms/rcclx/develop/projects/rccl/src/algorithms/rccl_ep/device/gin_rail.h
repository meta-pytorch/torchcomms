/*************************************************************************
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * The hybrid rail over RCCL GIN (RCCL_EP_GIN). Each chunk of a dispatch's records, or of a
 * combine's node partials, crosses to the rail peer as one one-sided put into the peer's
 * receive buffer at this node's slot, then flags its arrival. The receive offsets are fixed
 * slots of T rows per node (device/hybrid.h), so no exchange is needed to place data.
 *
 * One-sided puts lose the rendezvous that RCCL send/recv gave: a sender could overwrite a
 * receive buffer its peer is still reading. So every receiver, once done with a call's data,
 * returns a credit, and a sender's first put of a call waits for the credit of the previous one.
 *
 * Arrival and credits are flagged by plain RDMA writes of a running token (putValue) into a
 * small flag window, not by GIN signals: a signal is an RDMA fetch-and-add into GPU memory, and
 * on AINIC (Pollara) at 16 nodes those fail every few thousand transfers with a transport retry
 * error, after which the GIN proxy stops and the job hangs. RCCL_EP_GIN_SIGNAL=signal keeps
 * the signals for other NICs.
 *
 * A single GIN context is one queue pair per peer, and one queue pair caps well below the
 * NIC's line rate. So each transfer is striped over `lanes` contexts (RCCL_EP_GIN_LANES):
 * lane l moves rows [p0 + rows*l/L, p0 + rows*(l+1)/L) of the chunk and flags on its own
 * context. Dispatch uses contexts [0, L), combine [L, 2L); the credits travel on each
 * direction's lane 0.
 *
 * The same kernels run on the GPU-initiated backend (rocSHMEM GDA, NCCL_GIN_TYPE=5) when
 * librccl_ep is built with it. There each context has a queue pair per peer of its own, on
 * alternating UDMA engines (one engine moves ~28 GB/s on Pollara), and a flush waits only for the
 * destination's queue pair (gin_flush_peer). Data and flag still share a queue pair.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_GIN_RAIL_H_
#define RCCL_EP_GIN_RAIL_H_

#include <nccl_device.h>

#include "device/hybrid.h"

namespace rccl_ep {

constexpr int kGinMaxLanes = 8;
// RCCL_EP_GIN_SIGNAL=signal: GIN signals per context, data from node m and node m's credit for
// what this rank last sent it.
__host__ __device__ constexpr int gin_sig_data(int m) { return m; }
__host__ __device__ constexpr int gin_sig_ack(int m) { return kMaxNodes + m; }
constexpr int kGinSignals = 2 * kMaxNodes;

// Flags (the default): each flag word has one writer (a source node and lane), which writes the
// absolute token, so no atomic is needed. It is written on the same context as the data it
// follows: the proxy posts both on one queue pair in order, and the flag window is registered
// with strict ordering, so the flag cannot become visible before the data. One 64-byte line per
// word. Layout of the flag window, in uint64 words:
//   data[dir][src node][lane] at gin_flag_data(dir, m, l), credits[dir][src node] at gin_flag_ack(dir, m).
constexpr int kGinFlagStride = 8;
__host__ __device__ constexpr size_t gin_flag_data(int dir, int m, int l) {
  return (((size_t)dir * kMaxNodes + m) * kGinMaxLanes + l) * kGinFlagStride;
}
__host__ __device__ constexpr size_t gin_flag_ack(int dir, int m) {
  return (2 * kMaxNodes * kGinMaxLanes + (size_t)dir * kMaxNodes + m) * kGinFlagStride;
}
constexpr size_t kGinFlagWords = (2 * kMaxNodes * kGinMaxLanes + 2 * kMaxNodes) * kGinFlagStride;

// What a GIN wait is for, in timeout reports.
enum GinWaitKind { kGinDispatchData = 0, kGinCombineData = 1, kGinDispatchCredit = 2, kGinCombineCredit = 3 };

// Where a handle's GIN kernels have got to, in host-mapped memory, for the host watchdog
// (RCCL_EP_GIN_WATCH_S). Each stage stores the token of the transfer it last finished: the data
// signal value that transfer completes, (call - 1) * nchunks + chunk + 1. Indexed
// [direction][node * kGinMaxLanes + lane], direction 0 dispatch, 1 combine.
struct GinProgress;
__device__ inline void gin_timeout_report(GinProgress* prog, int rank, int kind, int ctx, int peer_node, int sig,
                                          uint64_t v, uint64_t need);

struct GinProgress {
  static constexpr int kSlots = kMaxNodes * kGinMaxLanes;
  uint64_t put_enter[2][kSlots], put_credit[2][kSlots], put_posted[2][kSlots], put_done[2][kSlots];
  uint64_t wait_enter[2], wait_done[2][kSlots];
  uint64_t ack_done[2];  // the last call whose credits were posted
  // The first wait to time out on this rank (to_set = 1 once the rest is written).
  uint64_t to_set, to_kind, to_node, to_ctx, to_sig, to_seen, to_need;
};

__device__ inline void gin_stamp(uint64_t* p, uint64_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Claimed by the first wait on this device to time out, which alone writes the record (in device
// memory, so the claim needs no atomics on host memory).
__device__ unsigned int gin_timeout_claimed = 0;

// Waits until signal `sig` on `g` reaches `need`. With timeout > 0 (wall-clock ticks), a wait that
// outlives it records what it waited for in `prog`, prints it, and traps a few seconds later (a
// device printf dies with the trap, so the host watchdog reports the record too): a signal that
// never comes then fails the job with a report instead of hanging it until a watchdog kills every rank.
__device__ inline void gin_wait(const ncclGin& g, int sig, uint64_t need, uint64_t timeout, int rank, int kind,
                                int ctx, int peer_node, GinProgress* prog) {
  if (timeout == 0) {
    g.waitSignal(ncclCoopThread(), (ncclGinSignal_t)sig, need);
    return;
  }
  const uint64_t t0 = wall_clock64();
  uint64_t v;
  while ((v = g.readSignal((ncclGinSignal_t)sig)) < need)
    if (wall_clock64() - t0 > timeout) gin_timeout_report(prog, rank, kind, ctx, peer_node, sig, v, need);
}

// As gin_wait, for a flag word in this rank's flag window; `sig` in reports is the word's index.
__device__ inline void gin_wait_flag(const uint64_t* flags, size_t word, uint64_t need, uint64_t timeout, int rank,
                                     int kind, int ctx, int peer_node, GinProgress* prog) {
  const uint64_t* p = flags + word;
  const uint64_t t0 = wall_clock64();
  uint64_t v;
  while ((v = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)) < need)
    if (timeout && wall_clock64() - t0 > timeout) gin_timeout_report(prog, rank, kind, ctx, peer_node, (int)word, v, need);
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
}

// Records a timed-out wait in prog (first one on the device only), prints it, and traps a few
// seconds later, once the host watchdog has had time to read the record.
__device__ inline void gin_timeout_report(GinProgress* prog, int rank, int kind, int ctx, int peer_node, int sig,
                                          uint64_t v, uint64_t need) {
  if (atomicCAS(&gin_timeout_claimed, 0u, 1u) == 0u) {
    gin_stamp(&prog->to_kind, kind);
    gin_stamp(&prog->to_node, peer_node);
    gin_stamp(&prog->to_ctx, ctx);
    gin_stamp(&prog->to_sig, sig);
    gin_stamp(&prog->to_seen, v);
    gin_stamp(&prog->to_need, need);
    __threadfence_system();
    gin_stamp(&prog->to_set, 1);
  }
  static const char* const kinds[] = {"dispatch data", "combine data", "dispatch credit", "combine credit"};
  printf("[rccl_ep] GIN wait timed out on rank %d: %s from node %d, context %d, signal %d at %llu of %llu\n", rank,
         kinds[kind & 3], peer_node, ctx, sig, (unsigned long long)v, (unsigned long long)need);
  const uint64_t t1 = wall_clock64();
  while (wall_clock64() - t1 < 500000000ull) {  // 5 s for the host to read the record
  }
  __builtin_trap();
}

// Waits until everything this rank has posted to rail peer `peer` has completed locally, so the
// source may be rewritten. On GDA, ncclGin::flush would quiet this context's queue pair to every
// peer; quiet only the one this block wrote to.
__device__ inline void gin_flush_peer(const ncclGin& g, ncclTeam team, int peer) {
#if NCCL_GIN_ROCSHMEM_GDA_ENABLE
  if (g._ginBackend == (unsigned)NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA) {
    auto* rs = (ncclGinRocshmemGdaGPUContext*)g._ginHandle + g.contextId;
    const int gp = nccl::gin::internal::teamRankToGinRank(g.comm, team, peer);
    rocshmem::ActiveWFInfo wf(gp, rocshmem::ThreadScope::thread);
    rs->qps[gp]->quiet(wf);
    return;
  }
#endif
  g.flush(ncclCoopThread());
}

// Byte offsets of the hybrid buffers in the rail window. Every rank has the same shape, so
// the same offsets.
struct GinRailOffs {
  size_t so_send, so_recv, comb_x, land_x, comb_w, land_w;
};

// Block (n, l) for each destination node n != D and lane l; thread 0 issues. Puts lane l's
// stripe of chunk hi.chunk of the records for n (dispatch) or of the node partials folded for n
// (combine: w first, then x carrying the signal, which settles both; under hi.tok_split the
// chunk is the owner-token range hyb_comb_range finds in fwd_meta; a dispatch takes each node's
// count from dev_nsend when set), then, with `flush`, flushes,
// so the source may be rewritten once the kernel ends. Without it (RCCL_EP_GIN_STREAM) the
// next chunk's put follows at once and k_gin_rail_flush ends the call. ack_need > 0: first wait
// until n has finished reading the previous call (its credit arrives on lane 0, context ctx0).
// Stamps each stage of transfer `token` in prog. flags: this rank's flag window `fwin`, or null
// with RCCL_EP_GIN_SIGNAL=signal.
template <bool kCombine>
__global__ void k_gin_rail_put(ncclDevComm dc, ncclWindow_t win, HybInfo hi, GinRailOffs o,
                               size_t T, size_t H, size_t K, int with_w, int ctx0, int lanes,
                               uint64_t ack_need, uint64_t timeout, int rank, GinProgress* prog,
                               uint64_t token, ncclWindow_t fwin, const uint64_t* flags,
                               const int32_t* fwd_meta, const int32_t* dev_nsend, int flush, int nput) {
  const int n = blockIdx.x, l = blockIdx.y;
  if (n >= hi.SO || n == hi.D || threadIdx.x != 0) return;
  const int slot = n * kGinMaxLanes + l;
  gin_stamp(&prog->put_enter[kCombine][slot], token);
  const ncclTeam rail = ncclTeamWorld(dc);  // the rail communicator: rank = node index
  if (ack_need) {
    const int kind = kCombine ? kGinCombineCredit : kGinDispatchCredit;
    if (flags) {
      gin_wait_flag(flags, gin_flag_ack(kCombine, n), ack_need, timeout, rank, kind, ctx0, n, prog);
    } else {
      ncclGin g0(dc, ctx0);
      gin_wait(g0, gin_sig_ack(n), ack_need, timeout, rank, kind, ctx0, n, prog);
    }
  }
  gin_stamp(&prog->put_credit[kCombine][slot], token);
  ncclGin g(dc, ctx0 + l);
  const size_t np_dst = hyb_np(hi, n);              // n's slot in this rank's buffers
  const size_t np_me = hi.D < n ? hi.D : hi.D - 1;  // this node's slot in n's buffers
  // nput > 1 (RCCL_EP_GIN_STREAM's dispatch): chunks hi.chunk .. hi.chunk + nput - 1 in one launch,
  // chunk hi.chunk + i flagged with token + i.
  for (int i = 0; i < nput; ++i) {
  HybInfo hc = hi;
  hc.chunk = hi.chunk + i;
  token = token + (i > 0);
  int c0, c1;
  if (kCombine)
    hyb_comb_range(hc, fwd_meta, n, (int)T, (int)K, c0, c1);
  else  // dev_nsend: the node plan's counts on the device, before the host has them (ep_hcounts_post)
    hyb_rec_chunk(hc, dev_nsend ? dev_nsend[n] : hc.nsend[n], hc.chunk, c0, c1);
  int p0, p1;  // this lane's stripe of the chunk
  hyb_chunk(c1 - c0, l, lanes, p0, p1);
  p0 += c0;
  p1 += c0;
  const ncclGin_StrongSignalAdd sig{(ncclGinSignal_t)gin_sig_data(hi.D), 1};
  if (flags) {  // the data without a signal, then the flag on the same context (one queue pair)
    if (p1 > p0) {
      const size_t rows = (size_t)(p1 - p0);
      if (!kCombine) {
        const size_t rb = hi.rec_bytes;
        g.put(rail, n, win, o.so_recv + (np_me * T + p0) * rb, win, o.so_send + (np_dst * T + p0) * rb, rows * rb);
      } else {
        if (with_w)
          g.put(rail, n, win, o.land_w + (np_me * T + p0) * K * sizeof(float), win,
                o.comb_w + (np_dst * T + p0) * K * sizeof(float), rows * K * sizeof(float));
        g.put(rail, n, win, o.land_x + (np_me * T + p0) * H * sizeof(bf16_t), win,
              o.comb_x + (np_dst * T + p0) * H * sizeof(bf16_t), rows * H * sizeof(bf16_t));
      }
    }
    g.putValue(rail, n, fwin, gin_flag_data(kCombine, hi.D, l) * sizeof(uint64_t), (uint64_t)token);
  } else if (p1 > p0) {
    const size_t rows = (size_t)(p1 - p0);
    if (!kCombine) {
      const size_t rb = hi.rec_bytes;
      g.put(rail, n, win, o.so_recv + (np_me * T + p0) * rb, win, o.so_send + (np_dst * T + p0) * rb,
            rows * rb, sig);
    } else {
      if (with_w)
        g.put(rail, n, win, o.land_w + (np_me * T + p0) * K * sizeof(float), win,
              o.comb_w + (np_dst * T + p0) * K * sizeof(float), rows * K * sizeof(float));
      g.put(rail, n, win, o.land_x + (np_me * T + p0) * H * sizeof(bf16_t), win,
            o.comb_x + (np_dst * T + p0) * H * sizeof(bf16_t), rows * H * sizeof(bf16_t), sig);
    }
  } else {
    g.signal(rail, n, sig);  // an empty chunk still counts, so the receiver's tally stays exact
  }
  gin_stamp(&prog->put_posted[kCombine][slot], token);
  }
  if (!flush) return;
  gin_flush_peer(g, rail, n);
  gin_stamp(&prog->put_done[kCombine][slot], token);
}

// RCCL_EP_GIN_STREAM: block (n, l) flushes what lane l posted to node n over the call's unflushed
// puts, so their sources may be rewritten once the kernel ends; `token` is the call's last.
template <bool kCombine>
__global__ void k_gin_rail_flush(ncclDevComm dc, HybInfo hi, int ctx0, GinProgress* prog, uint64_t token) {
  const int n = blockIdx.x, l = blockIdx.y;
  if (n >= hi.SO || n == hi.D || threadIdx.x != 0) return;
  ncclGin g(dc, ctx0 + l);
  gin_flush_peer(g, ncclTeamWorld(dc), n);
  gin_stamp(&prog->put_done[kCombine][n * kGinMaxLanes + l], token);
}

// Thread (m, l) waits until node m's data signal on lane l's context reaches need, the token of
// the transfer it completes.
__global__ void k_gin_rail_wait(ncclDevComm dc, int ctx0, int lanes, int SO, int D, uint64_t need,
                                uint64_t timeout, int rank, int kind, GinProgress* prog,
                                const uint64_t* flags) {
  const int m = threadIdx.x / lanes, l = threadIdx.x % lanes, dir = kind & 1;
  if (threadIdx.x == 0) gin_stamp(&prog->wait_enter[dir], need);
  if (m < SO && m != D) {
    if (flags) {
      gin_wait_flag(flags, gin_flag_data(dir, m, l), need, timeout, rank, kind, ctx0 + l, m, prog);
    } else {
      ncclGin g(dc, ctx0 + l);
      gin_wait(g, gin_sig_data(m), need, timeout, rank, kind, ctx0 + l, m, prog);
    }
    gin_stamp(&prog->wait_done[dir][m * kGinMaxLanes + l], need);
  }
}

// Thread n returns node n the credit for what it last sent this rank on context ctx; then stamps
// call `seq` of direction dir as credited.
__global__ void k_gin_rail_ack(ncclDevComm dc, int ctx, int SO, int D, GinProgress* prog, int dir,
                               uint64_t seq, ncclWindow_t fwin, int write_flags) {
  const int n = threadIdx.x;
  if (n < SO && n != D) {
    ncclGin g(dc, ctx);
    if (write_flags)
      g.putValue(ncclTeamWorld(dc), n, fwin, gin_flag_ack(dir, D) * sizeof(uint64_t), seq);
    else
      g.signal(ncclTeamWorld(dc), n, ncclGin_StrongSignalAdd{(ncclGinSignal_t)gin_sig_ack(D), 1});
  }
  __syncthreads();
  if (threadIdx.x == 0) gin_stamp(&prog->ack_done[dir], seq);
}

}  // namespace rccl_ep

#endif  // RCCL_EP_GIN_RAIL_H_
