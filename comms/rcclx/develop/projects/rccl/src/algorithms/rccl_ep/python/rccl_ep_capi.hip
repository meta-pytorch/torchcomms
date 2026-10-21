/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C ABI for the rccl_ep kernels, consumed from Python via ctypes
 * rather than a torch extension, which keeps the binding independent of the
 * torch version. Owns the communicator and the symmetric window; every
 * per-call structure belongs to the caller, which is what makes a cached
 * replay exact.
 *
 * The stream belongs to the caller too. Every data-path entry point takes one
 * and launches on it: torch creates its streams non-blocking, so the legacy
 * null stream does NOT implicitly order against them, and a null-stream launch
 * would read inputs still being produced on the caller's stream. Setup
 * (ep_create, ep_configure) stays on the null stream -- it runs once, before
 * any caller work exists to race with.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "device/combine.h"
#include "device/dispatch.h"
#include "device/ep_common.h"
#include "device/expand.h"
#include "device/gin_rail.h"
#include "device/hip_prims.h"
#include "device/hybrid.h"
#include "device/internode.h"
#include "include/ep_arch.h"
#include "include/ep_layout.h"

using namespace rccl_ep;

namespace {

struct EpHandle {
  EpConfig cfg{};
  ncclComm_t comm{};
  ncclWindow_t win{};
  void* win_buf{};
  size_t win_bytes{};
  EpWindowLayout lay{};
  ncclDevComm devComm{};
  WindowView* d_views{};
  WindowView self_view{};
  int32_t *d_recvoff{}, *d_total{};
  // Host-mapped [total, counts[epr], recv_pairs[R], max_tokens] that k_ep_plan_notify's
  // reduce writes and ep_wait_counts polls, and its device alias.
  int32_t* host_counts{};
  int32_t* host_counts_dev{};
  // Sequence number of the last uncached handshake. The window memset in ep_configure
  // leaves every flag at 0, so the first handshake uses 1.
  uint32_t notify_seq{0};
  // Sequence number of the last streamed dispatch (RCCL_EP_STREAM_DISPATCH), the high half of the
  // progress words its push publishes. Zeroed with the window like notify_seq; the first is 1.
  uint32_t stream_seq{0};
  // The same for the streamed combine (RCCL_EP_STREAM_COMBINE), whose words have their own region.
  uint32_t cstream_seq{0};
  // k_ep_plan_notify's per-chunk totals, [num_ranks, chunks, kPlanAggStride] ints. Sized
  // and zeroed with the window, whose epr it depends on; flags are notify_seq values.
  int32_t* d_plan_agg{};
  // The HIP device ordinal, kept separately from cfg.rank: the caller is free to
  // map ranks onto devices any way it likes, so querying device properties by
  // rank reads the wrong GPU on every non-identity mapping.
  int device{-1};
  bool devcomm{false};
  bool configured{false};
  // Barrier-A elision (RCCL_EP_ELIDE_A=1). rdv_x / rdv_y: a full EP rendezvous has been
  // enqueued on last_s since this rank's last reader of its dispatch region (x, metadata,
  // counts) / combine region (y, cw). All ranks enqueue the same ops, so every peer has then
  // finished reading the region, a write needs no barrier A, and all ranks elide alike.
  // pending_x: a grouped dispatch's epilogue, a dispatch-region reader, is still to come.
  bool elide_a{false};
  bool rdv_x{false}, rdv_y{false}, pending_x{false};
  hipStream_t last_s{};

  // Internode (ep_create_v2 with node_size < num_ranks). Ranks [node_base, node_base +
  // node_size) form the LSA team: the window, its devComm and every LSA barrier live on
  // node_comm, a split of `comm`. Writes to a remote peer go through d_views_inter (see
  // InterView) into `outbox`, which has the window's layout, and RCCL send/recv on `comm`
  // moves each region; `inbox` receives returned combine rows. ep_inter_bind sets `bound`.
  bool inter{false};
  int node_size{0}, node_base{0};
  ncclComm_t node_comm{};
  ncclComm_t lsa{};  // node_comm internode, comm otherwise
  uint8_t* outbox{};
  uint8_t* inbox{};
  WindowView* d_views_inter{};
  int32_t* d_src_inter{};
  int32_t* d_scratch{};  // count exchange and barrier, kScratchInts
  int32_t* host_scratch{};
  // Hybrid count exchange with per-expert counts (ep_hcounts with topk_idx), grown on demand.
  int32_t* d_xchg{};
  int32_t* host_xchg{};
  size_t xchg_ints{0};
  // RCCL_EP_HPLAN_FUSED=1: ep_hplan already wrote this rank's row of the exchange for this many
  // tokens (-1: not written), so ep_hcounts does not build it again.
  int hplan_row_tokens{-1};
  // RCCL_EP_HCOUNTS_SPIN=1: ep_hcounts polls this event instead of blocking in
  // hipStreamSynchronize, so the thread is still running when the counts land.
  hipEvent_t ev_hc{};
  // ep_hcounts_post: the next dispatch's pack and rail transfer are already queued, for these
  // tokens.
  struct {
    bool on{false};
    bool fwd{false};  // barrier A and the forwards too (RCCL_EP_GIN_STREAM)
    uintptr_t x{};
    int num_tokens{0};
  } rail_posted;
  int32_t* d_nrecv{};  // RCCL_EP_GIN_STREAM: nrecv on the device, from the count exchange
  InterCounts bound{};
  const int32_t* bound_send_list{};
  int bound_num_tokens{0};
  bool is_bound{false};
  // The window's y region holds one row per slot (ep_configure_v2 grouped_only): only a
  // grouped combine fits it.
  bool grouped_only{false};

  // Hybrid (ep_hybrid_enable; the two-level scheme in device/hybrid.h). Records cross
  // nodes only between rail peers (so_send -> so_recv); combine node partials return comb_x/w ->
  // land_x/w. stage_off_x/w: staging past the layout's end for partials bound for a remote owner.
  // hyb_multi: combines reduce within each node first; without it they take the flat path.
  bool hybrid{false}, hyb_multi{false};
  HybInfo hyb{};
  size_t stage_off_x{0}, stage_off_w{0};
  uint8_t *so_send{}, *so_recv{};
  bf16_t *comb_x{}, *land_x{};
  float *comb_w{}, *land_w{};
  uint8_t* sink{};         // count sink for the local push's remote views
  int32_t* d_local_sendc{};
  // The rail: this rank and its peer on every other node (rail rank = node index), carrying all
  // hybrid cross-node traffic. A separate comm because RCCL gives each p2p peer the minimum
  // channel count over all peers, local ones included, which leaves the rail peer one channel.
  ncclComm_t rail_comm{};
  hipStream_t side{};      // the records' transfer, overlapped with the local push
  // Rail transfers go in nchunks per node (RCCL_EP_HYB_CHUNKS, default 1): chunk c moves on
  // `side` while the caller's stream forwards or folds the others. ev_a[c]: chunk c ready to
  // send; ev_b[c]: arrived. Each chunk is another peer-matched round, waiting on the slowest rank.
  static constexpr int kMaxChunks = 8;
  int nchunks{1};
  // RCCL_EP_HYB_COMPACT_REC=1: each dispatch sizes its records to the sections it carries
  // (hyb_call_layout). Off: hybrid_alloc's full layout for every call.
  bool hyb_compact{false};
  hipEvent_t ev_a[kMaxChunks]{}, ev_b[kMaxChunks]{};
  // RCCL_EP_GIN_STREAM: ev_bar, the caller's stream past barrier A, which the side stream's
  // forwards wait for; ev_fwd, the side stream past its last forward; ev_cnt, the caller's stream
  // past d_nrecv (ep_hcounts_post).
  hipEvent_t ev_bar{}, ev_fwd{}, ev_cnt{};
  // RCCL_EP_GIN_STREAM: the side stream past the dispatch's last put launch (every chunk posted);
  // past the dispatch's / combine's flush, which the next call's pack / fold waits for.
  hipEvent_t ev_post{}, ev_dflush{}, ev_cflush{};
  // The rail over RCCL GIN (RCCL_EP_GIN, device/gin_rail.h): the hybrid buffers live in one
  // window registered on rail_comm, and each chunk is a put from the side stream instead of a
  // send/recv. gseq_d / gseq_c count this handle's hybrid dispatches / combines; every rank
  // makes the same calls, so the counts, and the flag tokens they predict, agree.
  bool gin{false}, gdc_made{false};
  int glanes{1};  // contexts each transfer is striped over (RCCL_EP_GIN_LANES)
  // RCCL_EP_GIN_STREAM=1: a call posts every chunk's put before its first arrival wait and flushes
  // once at the end, so the chunks leave back to back instead of one round each; the combine also
  // pushes, folds and sends its remote owners' partials one owner-token range at a time
  // (with RCCL_EP_HYB_COMB_SPLIT). Every rank must set it alike.
  bool gstream{false};
  // Arrival and credit flags are RDMA writes into this window (device/gin_rail.h); false with
  // RCCL_EP_GIN_SIGNAL=signal, which uses GIN signal fetch-and-adds instead.
  bool gwflag{false};
  void* gfbuf{};
  ncclWindow_t gfwin{};
  uint64_t gtimeout{0};  // wall-clock ticks a GIN wait may take before it reports and traps
  ncclDevComm gdc{};
  ncclWindow_t gwin{};
  void* gbuf{};
  GinRailOffs goff{};
  uint64_t gseq_d{0}, gseq_c{0};
  // The GIN kernels' progress (host-mapped) and the host watchdog that reports a stall from it
  // (gin_watch). gtok / gack: the last transfer token and credited call issued per direction.
  GinProgress* gprog{};
  int gwatch_s{0};
  std::thread gwatch;
  std::atomic<bool> gwatch_stop{false};
  std::atomic<uint64_t> gtok[2]{}, gack[2]{};
  const int32_t *bound_slot_of{}, *bound_node_list{}, *bound_node_pos{};
  int32_t* bound_fwd_meta{};
  // A hybrid combine issued with defer (ep_combine_v4): its rail transfer is in flight and the
  // reduce epilogue waits for ep_combine_finish, so the caller can queue other work between.
  struct {
    bool on{false};
    uintptr_t topk_idx{}, bias0{}, bias1{}, out{}, out_w{};
    int num_tokens{0}, cta_reduce{0};
    bool tok_stream{false};  // RCCL_EP_GIN_STREAM's owner-token chunks: reduce each as it lands
  } pend;
  // The same for a hybrid dispatch (ep_dispatch_defer_next): the local push is queued and the
  // rail transfer in flight; forwarding, barrier B and the epilogue wait for ep_dispatch_finish.
  bool defer_dispatch{false};
  // ep_dispatch_payload_later: the next dispatch's epilogue writes the metadata only and leaves
  // the rows in the window for ep_permute_epilogue; `perm` holds the offsets that epilogue used
  // until then. No dispatch may come between, since it would overwrite those rows.
  bool payload_later{false};
  struct {
    bool on{false};
    const int32_t* offsets{};
    int inclusive{0};
  } perm;
  struct {
    bool on{false};
    int payload_later{0};
    int use_fp8{0}, with_meta{1}, cta{0}, hcta{0}, run_epilogue{1}, notified{0}, streamed{0};
    int split_epi{0};  // the local sources' rows were copied right after the local push
    uintptr_t psum_incl{}, expert_counts{}, out_x{}, out_sf{}, out_topk{}, out_tw{}, out_src{};
  } pend_d;
};

// Count exchange scratch: every rank's [sendc (R) | node counts (SO)], plus this rank's own.
constexpr int kScratchInts = kMaxWorldRanks * (kMaxWorldRanks + kMaxNodes) + kMaxWorldRanks + kMaxNodes;

// Row strides differ by dispatch dtype (x) and by combine mode (y), so a remote peer
// needs one outbox view per stride; the local views are the same in all four.
// Hybrid adds two: the local push (local peers' LSA windows, remote peers a count sink) and
// the combine (local owners' windows, remote owners the forwarder's staging).
enum InterView { kViewBf16 = 0, kViewFp8 = 1, kViewCombGrouped = 2, kViewCombFlat = 3,
                 kViewHybLocal = 4, kViewHybComb = 5, kNumInterViews = 6 };

inline bool is_remote(const EpHandle* h, int r) {
  return h->inter && (r < h->node_base || r >= h->node_base + h->node_size);
}

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) {          \
    fprintf(stderr, "[rccl_ep] %s:%d %s\n", __FILE__, __LINE__,          \
            hipGetErrorString(e_)); return -1; } } while (0)

inline int sms_or(int num_sms, int dflt) { return num_sms > 0 ? num_sms : dflt; }

// Rendezvous bookkeeping for barrier-A elision (see EpHandle). Work on another stream is not
// ordered against what came before, so a stream change forgets every rendezvous.
inline void note_stream(EpHandle* h, hipStream_t s) {
  if (s != h->last_s) { h->rdv_x = h->rdv_y = false; h->last_s = s; }
}
inline void note_rdv(EpHandle* h) { h->rdv_x = h->rdv_y = true; }

// `num_sms` is a TOTAL CTA budget, but the dispatch grid is
// (num_ranks, ctas_per_rank), so it has to be divided down. Treating it as a
// per-rank count would quietly take num_ranks times the CUs the caller asked
// for, which defeats the entire point of the knob.
inline int ctas_per_rank(int num_sms, int dflt, int num_ranks) {
  const int total = sms_or(num_sms, dflt);
  const int per = total / num_ranks;
  return per > 0 ? per : 1;
}

// A positive integer environment knob read once, or 0 when unset.
inline int env_int(const char* name) {
  const char* e = getenv(name);
  return e ? atoi(e) : 0;
}

// CTAs per rank for the dispatch copy epilogue. By default it takes the push's grid, which
// num_sms sizes for peer stores; the copy is local and memory-bound, and at one 1024-thread
// CTA per CU the push's budget leaves CUs idle. RCCL_EP_COPY_EPILOGUE_SMS gives it its own
// total budget.
inline int copy_epilogue_cta(int push_cta, int num_ranks) {
  static const int sms = env_int("RCCL_EP_COPY_EPILOGUE_SMS");
  return sms > 0 ? ctas_per_rank(sms, 128, num_ranks) : push_cta;
}

// RCCL_EP_STREAM_DISPATCH=1: an intranode dispatch whose receive offsets are known before the push
// (a notified or cached dispatch) runs the push and the copy out of the window in one grid,
// overlapped, within the same CTA budget (dispatch_stream_impl).
inline bool stream_dispatch_on() {
  static const bool on = env_int("RCCL_EP_STREAM_DISPATCH") == 1;
  return on;
}

// RCCL_EP_STREAM_COMBINE=1: an intranode combine given the plan's positions runs the push and the
// reduce in one grid, overlapped, within the same CTA budget (combine_stream_impl).
inline bool stream_combine_on() {
  static const bool on = env_int("RCCL_EP_STREAM_COMBINE") == 1;
  return on;
}

// A streamed call's push CTAs per peer out of its `cta`: RCCL_EP_STREAM_PUSH_CTA, or half of them,
// at least 3 and at most 6. The rest copy or reduce. Measured on 8x MI350X: the push saturates the
// links at about 6 CTAs per peer and starves below 3. 0 when either side would be empty or the push
// is past kStreamMaxPushCta.
inline int stream_push_cta(int cta) {
  static const int env = env_int("RCCL_EP_STREAM_PUSH_CTA");
  const int p = env > 0 ? env : std::max(3, std::min(cta / 2, 6));
  return p >= 1 && p < cta && p <= kStreamMaxPushCta ? p : 0;
}

// Rows per push wave between two progress words of a streamed call: RCCL_EP_STREAM_CHUNK, or 8.
inline int stream_chunk_rounds() {
  static const int env = env_int("RCCL_EP_STREAM_CHUNK");
  return env > 0 ? env : 8;
}

// Release the device communicator and the symmetric window. ep_configure's
// failure paths run with `configured` still false, so ep_destroy would never
// reach resources they had already taken; they have to hand them back
// themselves. Every step is guarded by its own flag, so a partially built
// handle unwinds through the same path as a fully built one.
//
// The devComm is torn down first: it holds the LSA barrier state that lives in
// the window, so deregistering the window under it would leave it dangling.
inline void release_window(EpHandle* h, bool registered) {
  if (h->devcomm) {
    ncclDevCommDestroy(h->lsa, &h->devComm);
    h->devcomm = false;
  }
  if (registered) ncclCommWindowDeregister(h->lsa, h->win);
  // GIN: the hybrid buffers are carved out of the rail window, not allocated one by one.
  if (h->gbuf) {
    if (h->gwin) ncclCommWindowDeregister(h->rail_comm, h->gwin);
    ncclMemFree(h->gbuf);
    h->gbuf = nullptr;
    h->gwin = nullptr;
    h->so_send = h->so_recv = nullptr;
    h->comb_x = h->land_x = nullptr;
    h->comb_w = h->land_w = nullptr;
  }
  if (h->d_plan_agg) {
    hipFree(h->d_plan_agg);
    h->d_plan_agg = nullptr;
  }
  // The internode buffers are sized from the layout, so they go with the window.
  for (void** p : {(void**)&h->outbox, (void**)&h->inbox, (void**)&h->d_views_inter,
                   (void**)&h->d_src_inter, (void**)&h->so_send, (void**)&h->so_recv,
                   (void**)&h->comb_x, (void**)&h->land_x, (void**)&h->comb_w, (void**)&h->land_w,
                   (void**)&h->sink, (void**)&h->d_local_sendc}) {
    if (*p) {
      hipFree(*p);
      *p = nullptr;
    }
  }
  h->is_bound = false;
  if (h->win_buf) {
    ncclMemFree(h->win_buf);
    h->win_buf = nullptr;
  }
}

// The GIN rail's watchdog, progress record, flag window and device communicator (gin_rail_setup).
inline void release_gin_rail(EpHandle* h) {
  h->gwatch_stop = true;
  if (h->gwatch.joinable()) h->gwatch.join();
  if (h->gprog) hipHostFree(h->gprog);
  h->gprog = nullptr;
  if (h->gfwin) ncclCommWindowDeregister(h->rail_comm, h->gfwin);
  if (h->gfbuf) ncclMemFree(h->gfbuf);
  h->gfwin = nullptr;
  h->gfbuf = nullptr;
  h->gwflag = false;
  if (h->gdc_made) ncclDevCommDestroy(h->rail_comm, &h->gdc);
  h->gdc_made = h->gin = false;
}

// The per-node communicator and the count-exchange scratch, taken in ep_create_v2.
inline void release_inter_comm(EpHandle* h) {
  release_gin_rail(h);
  if (h->rail_comm) ncclCommDestroy(h->rail_comm);
  h->rail_comm = nullptr;
  if (h->side) hipStreamDestroy(h->side);
  for (int c = 0; c < EpHandle::kMaxChunks; ++c) {
    if (h->ev_a[c]) hipEventDestroy(h->ev_a[c]);
    if (h->ev_b[c]) hipEventDestroy(h->ev_b[c]);
    h->ev_a[c] = h->ev_b[c] = nullptr;
  }
  for (hipEvent_t* e : {&h->ev_bar, &h->ev_fwd, &h->ev_cnt, &h->ev_post, &h->ev_dflush, &h->ev_cflush}) {
    if (*e) hipEventDestroy(*e);
    *e = nullptr;
  }
  if (h->ev_hc) hipEventDestroy(h->ev_hc);
  h->ev_hc = nullptr;
  h->side = nullptr;
  if (h->node_comm) ncclCommDestroy(h->node_comm);
  h->node_comm = nullptr;
  hipFree(h->d_scratch);
  h->d_scratch = nullptr;
  if (h->host_scratch) hipHostFree(h->host_scratch);
  h->host_scratch = nullptr;
  hipFree(h->d_xchg);
  h->d_xchg = nullptr;
  if (h->d_nrecv) hipFree(h->d_nrecv);
  h->d_nrecv = nullptr;
  if (h->host_xchg) hipHostFree(h->host_xchg);
  h->host_xchg = nullptr;
  h->xchg_ints = 0;
}

// Every data-path entry point is reachable without the Python layer -- the unit
// tests include this ABI directly -- so an unconfigured handle has to say so
// rather than fault on a null window or an uninitialised devComm.
#define REQUIRE_CONFIGURED(h, fn) do {                                        \
    if (!(h)->configured) {                                                   \
      fprintf(stderr, "[rccl_ep] " fn " before a successful ep_configure\n"); \
      return -1;                                                              \
    }                                                                         \
  } while (0)

// Waves per block for every kernel that moves payload. The parallelism these
// kernels get is CTAs x waves, and only the CTA count follows the caller's
// num_sms budget, so a narrow block starves them at every budget below the
// default. 16 waves is 1024 threads, the gfx950 ceiling.
//
// Widening changes which wave owns which token and which lane owns which
// element, never the order of additions applied to one element, so results
// stay bit-exact.
//
// Caveat: the combine reduce is 1-D over tokens at one wave per token, so a
// wide block over-subscribes when tokens are few -- the decode regime, which
// this path does not serve.
constexpr int kEpWaves = 16;

#define NC(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) {        \
    fprintf(stderr, "[rccl_ep] %s:%d %s\n", __FILE__, __LINE__,          \
            ncclGetErrorString(r_)); return -1; } } while (0)

// Inside ncclGroupStart/End, NC would return with the group still open, and every later NCCL
// call on this thread would then only be queued: a hang instead of an error. NCG keeps the first
// failure in `err` and skips the remaining posts; NCG_END always closes the group, then fails if
// anything did.
#define NCG(err, x) do { if ((err) == ncclSuccess && ((err) = (x)) != ncclSuccess)  \
    fprintf(stderr, "[rccl_ep] %s:%d %s\n", __FILE__, __LINE__,                    \
            ncclGetErrorString(err)); } while (0)
#define NCG_END(err) do { ncclResult_t e_ = ncclGroupEnd();                        \
    if (e_ != ncclSuccess)                                                         \
      fprintf(stderr, "[rccl_ep] %s:%d %s\n", __FILE__, __LINE__,                  \
              ncclGetErrorString(e_));                                             \
    if ((err) != ncclSuccess || e_ != ncclSuccess) return -1; } while (0)

// The layout a remote peer q is seen through: every region offset moves by (q - rank)
// rows of that region, so the kernels' slot(rank, i) = rank * T + i lands on row
// q * T + i, in outbox region q. A negative shift is exact under size_t wrap-around.
EpWindowLayout remote_layout(const EpWindowLayout& l, const EpConfig& c, int q, int kind) {
  const int64_t d = (int64_t)q - c.rank;
  const int64_t T = c.num_max_tokens_per_rank, H = c.hidden, K = c.num_topk;
  const int64_t S = 2 + (int64_t)c.experts_per_rank() + c.num_ranks;
  auto sh = [d](size_t off, int64_t row_bytes) { return off + (size_t)(d * row_bytes); };
  EpWindowLayout r = l;
  r.off_counts = sh(l.off_counts, sizeof(int32_t));
  r.off_flags = sh(l.off_flags, sizeof(uint32_t));
  r.off_notify = sh(l.off_notify, S * (int64_t)sizeof(int32_t));
  r.off_x = sh(l.off_x, T * H * (kind == kViewFp8 ? 1 : 2));
  r.off_sf = sh(l.off_sf, T * c.hidden_sf() * (int64_t)sizeof(float));
  r.off_topk_idx = sh(l.off_topk_idx, T * K * (int64_t)sizeof(int32_t));
  r.off_topk_w = sh(l.off_topk_w, T * K * (int64_t)sizeof(float));
  r.off_src_idx = sh(l.off_src_idx, T * (int64_t)sizeof(int32_t));
  r.off_prog = sh(l.off_prog, kStreamMaxPushCta * (int64_t)sizeof(uint32_t));
  r.off_cprog = sh(l.off_cprog, kStreamMaxPushCta * (int64_t)sizeof(uint32_t));
  r.off_sdone = sh(l.off_sdone, 0);  // local only
  r.off_y = sh(l.off_y, T * H * 2 * (kind == kViewCombFlat ? K : 1));
  r.off_cw = sh(l.off_cw, T * K * (int64_t)sizeof(float));
  r.off_arch = sh(l.off_arch, kArchLen);
  return r;
}

// One region moved per peer: its offset in any buffer with the window's layout, and its
// row size. Region q starts at off + q * T * row in both the outbox and the window.
struct InterRegion {
  size_t off, row;
};

// Send outbox region q and receive region q of `dst` for every remote q, in one group.
// Both sides post ranks and regions in the same order, so every send meets its receive;
// a zero count is skipped on both sides, since one's sendc[q] is the other's recvc[q].
int inter_exchange(EpHandle* h, const InterRegion* regs, int nreg, const int32_t* send_rows,
                   const int32_t* recv_rows, uint8_t* dst, hipStream_t s) {
  const size_t T = h->cfg.num_max_tokens_per_rank;
  NC(ncclGroupStart());
  ncclResult_t gr = ncclSuccess;
  for (int q = 0; q < h->cfg.num_ranks; ++q) {
    if (!is_remote(h, q)) continue;
    for (int g = 0; g < nreg; ++g) {
      const size_t base = regs[g].off + (size_t)q * T * regs[g].row;
      if (send_rows[q])
        NCG(gr, ncclSend(h->outbox + base, (size_t)send_rows[q] * regs[g].row, ncclUint8, q, h->comm, s));
      if (recv_rows[q])
        NCG(gr, ncclRecv(dst + base, (size_t)recv_rows[q] * regs[g].row, ncclUint8, q, h->comm, s));
    }
  }
  NCG_END(gr);
  return 0;
}

// Dispatch's remote leg, after dispatch_impl has filled the outbox: each region goes to
// its peer, and each remote source's rows and count land in this window for the epilogue.
int inter_dispatch(EpHandle* h, int use_fp8, int with_meta, hipStream_t s) {
  const EpConfig& c = h->cfg;
  const EpWindowLayout& l = h->lay;
  const size_t H = c.hidden, K = c.num_topk;
  InterRegion regs[5];
  int n = 0;
  regs[n++] = {l.off_x, H * (use_fp8 ? 1 : sizeof(bf16_t))};
  if (use_fp8) regs[n++] = {l.off_sf, (size_t)c.hidden_sf() * sizeof(float)};
  if (with_meta) {
    regs[n++] = {l.off_topk_idx, K * sizeof(int32_t)};
    regs[n++] = {l.off_topk_w, K * sizeof(float)};
    regs[n++] = {l.off_src_idx, sizeof(int32_t)};
  }
  uint8_t* win = (uint8_t*)h->win_buf;
  if (inter_exchange(h, regs, n, h->bound.sendc, h->bound.recvc, win, s) != 0) return -1;
  hipLaunchKernelGGL(k_inter_set_counts, dim3(1), dim3(kMaxWorldRanks), 0, s, c.num_ranks, h->bound,
                     (int32_t*)(win + l.off_counts));
  return 0;
}

// Combine's remote leg, after combine_impl has filled the outbox: rows held for a remote
// owner go back to it, and rows returned by each remote rank are scattered from the inbox
// into this window's combine region for the reduce epilogue.
int inter_combine(EpHandle* h, int grouped, int with_w, int cta, hipStream_t s) {
  const EpConfig& c = h->cfg;
  const EpWindowLayout& l = h->lay;
  const size_t H = c.hidden, K = c.num_topk;
  const int y_row_elems = (int)(grouped ? H : K * H);
  InterRegion regs[2] = {{l.off_y, (size_t)y_row_elems * sizeof(bf16_t)}, {l.off_cw, K * sizeof(float)}};
  // Owner-side: this rank received recvc[o] rows from o, and gets back sendc[r] from r.
  if (inter_exchange(h, regs, with_w ? 2 : 1, h->bound.recvc, h->bound.sendc, h->inbox, s) != 0)
    return -1;
  hipLaunchKernelGGL(k_inter_unpack_combine, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                     c, h->bound, h->bound_send_list, h->bound_num_tokens,
                     (const bf16_t*)(h->inbox + l.off_y), (const float*)(h->inbox + l.off_cw),
                     y_row_elems, h->self_view, with_w);
  return 0;
}

// CTAs per node for the hybrid record kernels (pack, forward, fold): their grid is (nodes,
// ctas), so the budget divides by nodes, not by ranks as for the per-peer kernels.
inline int hyb_ctas(const EpHandle* h, int num_sms) {
  return std::max(1, sms_or(num_sms, 128) / h->hyb.SO);
}

// One dispatch's record layout under RCCL_EP_HYB_COMPACT_REC: only the sections the call
// carries. A bf16 payload has no sf section, an fp8 payload takes `hidden` bytes, and a payload
// replay (with_meta 0) has no top-k sections. The src section stays, since k_hyb_pack writes it
// on every call. Every rank makes the same call, so both ends of the rail derive the same
// stride. Off, hybrid_alloc's full layout stays. Host-only: the kernels are unchanged.
static void hyb_call_layout(const EpHandle* h, HybInfo& hi, int use_fp8, int with_meta) {
  if (!h->hyb_compact) return;
  const EpConfig& c = h->cfg;
  size_t o = align_up((size_t)c.hidden * (use_fp8 ? 1 : sizeof(bf16_t)), 16);
  if (use_fp8) {
    hi.off_sf = (int)o;
    o = align_up(o + (size_t)c.hidden_sf() * sizeof(float), 16);
  }
  if (with_meta) {
    hi.off_topk = (int)o;
    o = align_up(o + (size_t)c.num_topk * sizeof(int32_t), 16);
    hi.off_w = (int)o;
    o = align_up(o + (size_t)c.num_topk * sizeof(float), 16);
  }
  hi.off_src = (int)o;
  o = align_up(o + sizeof(int32_t), 16);
  hi.off_slots = (int)o;
  hi.rec_bytes = (int)align_up(o + (size_t)hi.SU * sizeof(int32_t), 16);
}

// This rank's flag window for the GIN kernels, or null with RCCL_EP_GIN_SIGNAL=signal.
inline const uint64_t* gin_flags(const EpHandle* h) { return h->gwflag ? (const uint64_t*)h->gfbuf : nullptr; }

static int gin_dispatch_puts(EpHandle* h, HybInfo hi, const int32_t* cnt, int k0, int nk);

// Hybrid dispatch, first half: pack one record per (token, remote node) and hand them to the
// side stream, which moves them to the rail peers while the caller's stream goes on to the
// local push. Every rank posts the same sends and receives in the same node order.
int hybrid_dispatch_begin(EpHandle* h, uintptr_t x, uintptr_t x_sf, uintptr_t topk_idx,
                          uintptr_t topk_w, int num_tokens, int use_fp8, int with_meta, int cta,
                          hipStream_t s) {
  const EpConfig& c = h->cfg;
  HybInfo hi = h->hyb;
  hi.nchunks = h->nchunks;
  hyb_call_layout(h, hi, use_fp8, with_meta);
  const size_t T = c.num_max_tokens_per_rank, rb = hi.rec_bytes;
  if (h->gin) ++h->gseq_d;
  if (h->gin && h->gstream) {
    // Chunk 0 is packed and posted first, so it leaves while the rest is packed; then one pack and
    // one put launch cover every other chunk. The pack waits for the previous dispatch's flush.
    hi.taper = 1;
    HC(hipStreamWaitEvent(s, h->ev_dflush, 0));
    for (int part = 0; part < 2 && part < hi.nchunks; ++part) {
      hi.chunk = part;
      const int span = part == 0 ? 1 : hi.nchunks - 1;
      if (use_fp8)
        hipLaunchKernelGGL(k_hyb_pack<true>, dim3(hi.SO, cta), dim3(kEpWaves * kWarpSize), 0, s, c, hi,
                           (const void*)x, (const float*)x_sf, (const int32_t*)topk_idx,
                           (const float*)topk_w, num_tokens, h->bound_node_list, h->bound_slot_of, h->so_send,
                           with_meta, span);
      else
        hipLaunchKernelGGL(k_hyb_pack<false>, dim3(hi.SO, cta), dim3(kEpWaves * kWarpSize), 0, s, c, hi,
                           (const void*)x, (const float*)nullptr, (const int32_t*)topk_idx,
                           (const float*)topk_w, num_tokens, h->bound_node_list, h->bound_slot_of, h->so_send,
                           with_meta, span);
      HC(hipEventRecord(h->ev_a[part], s));
      HC(hipStreamWaitEvent(h->side, h->ev_a[part], 0));
      if (gin_dispatch_puts(h, hi, nullptr, part, span) != 0) return -1;
    }
    return 0;
  }
  for (int k = 0; k < hi.nchunks; ++k) {
    hi.chunk = k;
    if (use_fp8)
      hipLaunchKernelGGL(k_hyb_pack<true>, dim3(hi.SO, cta), dim3(kEpWaves * kWarpSize), 0, s, c, hi,
                         (const void*)x, (const float*)x_sf, (const int32_t*)topk_idx,
                         (const float*)topk_w, num_tokens, h->bound_node_list, h->bound_slot_of, h->so_send,
                         with_meta, 1);
    else
      hipLaunchKernelGGL(k_hyb_pack<false>, dim3(hi.SO, cta), dim3(kEpWaves * kWarpSize), 0, s, c, hi,
                         (const void*)x, (const float*)nullptr, (const int32_t*)topk_idx,
                         (const float*)topk_w, num_tokens, h->bound_node_list, h->bound_slot_of, h->so_send,
                         with_meta, 1);
    HC(hipEventRecord(h->ev_a[k], s));
    HC(hipStreamWaitEvent(h->side, h->ev_a[k], 0));
    if (h->gin) {
      // Chunk 0 first takes each peer's credit for the previous dispatch's records. The wait for
      // the peers' chunk runs here too, so ev_b[k] means "sent and received" as for send/recv,
      // and the caller's stream only ever waits on events: every spinning kernel stays on the
      // side stream. A spin on the caller's stream can sit in a hardware queue it shares with
      // other streams' collectives (GPU_MAX_HW_QUEUES=2) and deadlock across ranks.
      const uint64_t ack = (k == 0 && h->gseq_d > 1) ? h->gseq_d - 1 : 0;
      const uint64_t tok = (h->gseq_d - 1) * hi.nchunks + k + 1;
      hipLaunchKernelGGL(k_gin_rail_put<false>, dim3(hi.SO, h->glanes), dim3(kWarpSize), 0, h->side, h->gdc,
                         h->gwin, hi, h->goff, T, (size_t)0, (size_t)0, 0, 0, h->glanes, ack, h->gtimeout, c.rank,
                         h->gprog, tok, h->gfwin, gin_flags(h), (const int32_t*)nullptr, (const int32_t*)nullptr, 1, 1);
      hipLaunchKernelGGL(k_gin_rail_wait, dim3(1), dim3(kMaxNodes * h->glanes), 0, h->side, h->gdc, 0, h->glanes,
                         hi.SO, hi.D, tok, h->gtimeout, c.rank, (int)kGinDispatchData, h->gprog, gin_flags(h));
      HC(hipGetLastError());
      HC(hipEventRecord(h->ev_b[k], h->side));
      h->gtok[0] = tok;
      continue;
    }
    NC(ncclGroupStart());
    ncclResult_t gr = ncclSuccess;
    for (int n = 0; n < hi.SO; ++n) {
      if (n == hi.D) continue;
      const size_t np = n < hi.D ? n : n - 1;
      int s0, s1, r0, r1;
      hyb_chunk(hi.nsend[n], k, hi.nchunks, s0, s1);
      hyb_chunk(hi.nrecv[n], k, hi.nchunks, r0, r1);
      if (s1 > s0)
        NCG(gr, ncclSend(h->so_send + (np * T + s0) * rb, (size_t)(s1 - s0) * rb, ncclUint8, n, h->rail_comm, h->side));
      if (r1 > r0)
        NCG(gr, ncclRecv(h->so_recv + (np * T + r0) * rb, (size_t)(r1 - r0) * rb, ncclUint8, n, h->rail_comm, h->side));
    }
    NCG_END(gr);
    HC(hipEventRecord(h->ev_b[k], h->side));
  }
  return 0;
}

// RCCL_EP_GIN_STREAM, the dispatch's posting paths: one launch posts chunks k0 .. k0 + nk - 1
// unflushed and back to back (chunk 0 first takes each peer's credit for the previous dispatch),
// `cnt` (device) or hi.nsend giving each node's records. Few launches rather than one per chunk:
// the side stream ran those one after another, ~20 us each, ahead of the first arrival wait.
static int gin_dispatch_puts(EpHandle* h, HybInfo hi, const int32_t* cnt, int k0, int nk) {
  const EpConfig& c = h->cfg;
  const size_t T = c.num_max_tokens_per_rank;
  hi.chunk = k0;
  const uint64_t ack = (k0 == 0 && h->gseq_d > 1) ? h->gseq_d - 1 : 0;
  const uint64_t tok = (h->gseq_d - 1) * hi.nchunks + k0 + 1;
  hipLaunchKernelGGL(k_gin_rail_put<false>, dim3(hi.SO, h->glanes), dim3(kWarpSize), 0, h->side, h->gdc, h->gwin,
                     hi, h->goff, T, (size_t)0, (size_t)0, 0, 0, h->glanes, ack, h->gtimeout, c.rank, h->gprog, tok,
                     h->gfwin, gin_flags(h), (const int32_t*)nullptr, cnt, 0, nk);
  HC(hipGetLastError());
  h->gtok[0] = tok + nk - 1;
  if (k0 + nk == hi.nchunks) HC(hipEventRecord(h->ev_post, h->side));
  return 0;
}

// RCCL_EP_GIN_STREAM: after the unflushed puts, the side stream forwards each chunk as soon as it
// has arrived (its wait, then k_hyb_forward_pairs), while the caller's stream runs the local push:
// the two xGMI legs overlap instead of queueing one behind the other. The first forward also waits
// for barrier A on the caller's stream (ev_bar), as it writes the local ranks' windows; ev_fwd
// marks the last one done. dev_nrecv: each node's record count on the device (ep_hcounts_post,
// before the host has it), or null for the bound counts.
static int gin_dispatch_forwards(EpHandle* h, int use_fp8, int with_meta, int cta, int32_t* fwd_meta,
                                 const int32_t* dev_nrecv) {
  const EpConfig& c = h->cfg;
  HybInfo hi = h->hyb;
  hi.nchunks = h->nchunks;
  hi.taper = 1;
  hyb_call_layout(h, hi, use_fp8, with_meta);
  const WindowView* local = h->d_views_inter + kViewHybLocal * c.num_ranks + h->node_base;
  // The forward runs beside the local push, which leaves most CUs idle, so it takes 128 CTAs per
  // node rather than the record kernels' budget (64 per node at 2 nodes): 2-4% off a cached fp8
  // dispatch at 2 and 4 nodes.
  const int fcta = std::max(cta, 128);
  HC(hipStreamWaitEvent(h->side, h->ev_bar, 0));
  for (int k = 0; k < hi.nchunks; ++k) {
    hi.chunk = k;
    const uint64_t tok = (h->gseq_d - 1) * hi.nchunks + k + 1;
    hipLaunchKernelGGL(k_gin_rail_wait, dim3(1), dim3(kMaxNodes * h->glanes), 0, h->side, h->gdc, 0, h->glanes,
                       hi.SO, hi.D, tok, h->gtimeout, c.rank, (int)kGinDispatchData, h->gprog, gin_flags(h));
    if (use_fp8)
      hipLaunchKernelGGL(k_hyb_forward_pairs<true>, dim3(hi.SO, fcta), dim3(kEpWaves * kWarpSize), 0, h->side, c,
                         hi, h->so_recv, local, with_meta, fwd_meta, dev_nrecv);
    else
      hipLaunchKernelGGL(k_hyb_forward_pairs<false>, dim3(hi.SO, fcta), dim3(kEpWaves * kWarpSize), 0, h->side, c,
                         hi, h->so_recv, local, with_meta, fwd_meta, dev_nrecv);
    HC(hipGetLastError());
  }
  HC(hipEventRecord(h->ev_fwd, h->side));
  // Off the caller's path: so_recv is read, so each peer gets its credit back from here; and the
  // flush, which the next dispatch's pack waits for (ev_dflush) before it rewrites so_send.
  hipLaunchKernelGGL(k_gin_rail_ack, dim3(1), dim3(kMaxNodes), 0, h->side, h->gdc, 0, hi.SO, hi.D, h->gprog, 0,
                     h->gseq_d, h->gfwin, (int)h->gwflag);
  hipLaunchKernelGGL(k_gin_rail_flush<false>, dim3(hi.SO, h->glanes), dim3(kWarpSize), 0, h->side, h->gdc, hi, 0,
                     h->gprog, (h->gseq_d - 1) * hi.nchunks + hi.nchunks);
  HC(hipGetLastError());
  h->gack[0] = h->gseq_d;
  HC(hipEventRecord(h->ev_dflush, h->side));
  return 0;
}

// Hybrid dispatch, second half: once the records are in, forward each to the local ranks it
// names (k_hyb_forward), and publish the remote sources' counts in this window.
int hybrid_dispatch_end(EpHandle* h, int use_fp8, int with_meta, int cta, hipStream_t s) {
  const EpConfig& c = h->cfg;
  HybInfo hi = h->hyb;
  hi.nchunks = h->nchunks;
  hyb_call_layout(h, hi, use_fp8, with_meta);
  const WindowView* local = h->d_views_inter + kViewHybLocal * c.num_ranks + h->node_base;
  const bool gs = h->gin && h->gstream;
  if (gs) {  // gin_dispatch_forwards forwarded on the side stream and returned the credits
    hipLaunchKernelGGL(k_inter_set_counts, dim3(1), dim3(kMaxWorldRanks), 0, s, c.num_ranks, h->bound,
                       (int32_t*)((uint8_t*)h->win_buf + h->lay.off_counts));
    HC(hipStreamWaitEvent(s, h->ev_fwd, 0));
    return 0;
  }
  for (int k = 0; k < hi.nchunks; ++k) {
    hi.chunk = k;
    HC(hipStreamWaitEvent(s, h->ev_b[k], 0));
    if (use_fp8)
      hipLaunchKernelGGL(k_hyb_forward<true>, dim3(hi.SO, cta), dim3(kEpWaves * kWarpSize), 0, s, c, hi,
                         h->so_recv, local, with_meta, h->bound_fwd_meta);
    else
      hipLaunchKernelGGL(k_hyb_forward<false>, dim3(hi.SO, cta), dim3(kEpWaves * kWarpSize), 0, s, c, hi,
                         h->so_recv, local, with_meta, h->bound_fwd_meta);
  }
  hipLaunchKernelGGL(k_inter_set_counts, dim3(1), dim3(kMaxWorldRanks), 0, s, c.num_ranks, h->bound,
                     (int32_t*)((uint8_t*)h->win_buf + h->lay.off_counts));
  // GIN: the forward has read so_recv, so each peer gets its credit back. Only posts signals.
  if (h->gin) {
    hipLaunchKernelGGL(k_gin_rail_ack, dim3(1), dim3(kMaxNodes), 0, s, h->gdc, 0, hi.SO, hi.D, h->gprog, 0, h->gseq_d,
                       h->gfwin, (int)h->gwflag);
    HC(hipGetLastError());
    h->gack[0] = h->gseq_d;
  }
  return 0;
}

// Waits for the rail and runs the owner's reduce epilogue of the pending combine, on `s`.
int hybrid_combine_finish(EpHandle* h, hipStream_t s) {
  if (!h->pend.on) {
    fprintf(stderr, "[rccl_ep] ep_combine_finish: no deferred combine is pending\n");
    return -1;
  }
  h->pend.on = false;
  note_stream(h, s);
  // The side stream runs the chunks in order, so the last one landing means all have. With
  // owner-token chunks, chunk k from every node holds exactly the node partials of this rank's
  // tokens in range k, so all ranges but the last are reduced once the one before last has landed
  // (under the last chunk's transfer, when the rail is the longer path), and the last range alone
  // after it, its hidden split over 4 waves per token: a launch that small is latency-bound.
  const int C = h->nchunks;
  const bool two = h->pend.tok_stream && C > 1;
  note_rdv(h);
  h->rdv_y = false;
  for (int part = two ? 0 : 1; part < 2; ++part) {
    int t0 = 0, t1 = INT_MAX, d;
    if (two) {
      hyb_tok_chunk((int)h->cfg.num_max_tokens_per_rank, part == 0 ? 0 : C - 1, C, t0, d);
      if (part == 0) hyb_tok_chunk((int)h->cfg.num_max_tokens_per_rank, C - 1, C, t1, d);
    }
    HC(hipStreamWaitEvent(s, h->ev_b[part == 0 ? C - 2 : C - 1], 0));
    hipLaunchKernelGGL(hybrid_combine_reduce_epilogue_impl, dim3(h->pend.cta_reduce), dim3(kEpWaves * kWarpSize),
                       0, s, h->cfg, h->hyb, h->self_view, (const int32_t*)h->pend.topk_idx,
                       h->pend.num_tokens, h->bound_node_pos, h->land_x, h->land_w,
                       (const bf16_t*)h->pend.bias0, (const bf16_t*)h->pend.bias1, (bf16_t*)h->pend.out,
                       (float*)h->pend.out_w, t0, t1, two && part == 1 ? 4 : 1);
  }
  if (h->gin) {  // land_x / land_w are read: the peers may send the next combine's partials
    hipLaunchKernelGGL(k_gin_rail_ack, dim3(1), dim3(kMaxNodes), 0, s, h->gdc, h->glanes, h->hyb.SO, h->hyb.D,
                       h->gprog, 1, h->gseq_c, h->gfwin, (int)h->gwflag);
    HC(hipGetLastError());
    h->gack[1] = h->gseq_c;
  }
  return 0;
}

// Hybrid combine, reducing within each node before crossing it. Each
// expert rank writes its grouped partial to a local owner's window, or for a remote owner to the
// forwarder's staging; after the node barrier each forwarder folds one node partial per token it
// forwarded and returns it on the rail, and the owner reduces its rank and node partials.
int hybrid_combine(EpHandle* h, uintptr_t y, uintptr_t in_w, uintptr_t row_map, uintptr_t expert_rows,
                   uintptr_t scales, int fma_mode, uintptr_t recv_topk, uintptr_t recv_src, int num_recv,
                   uintptr_t topk_idx, int num_tokens, uintptr_t bias0, uintptr_t bias1,
                   uintptr_t out, uintptr_t out_w, int cta, int cta_reduce, int hcta, int defer,
                   hipStream_t s) {
  const EpConfig& c = h->cfg;
  const HybInfo& hi = h->hyb;
  const int with_w = in_w != 0;
  note_stream(h, s);
  // Barrier A holds the partials until the owners' reduce epilogue and the forwarders' fold have
  // read the last ones. A rendezvous since this rank's last epilogue suffices (see EpHandle):
  // every forwarder runs its fold before its epilogue.
  if (!(h->elide_a && h->rdv_y)) {
    hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
    note_rdv(h);
  }
  // RCCL_EP_HYB_COMB_SPLIT=1: only the partials for other nodes' owners feed the rail, so push
  // those first, start the rail, and push this node's owners' partials while it moves; a
  // second barrier then holds the reduce epilogue until they are in. Same rows, same values.
  static const bool split = env_int("RCCL_EP_HYB_COMB_SPLIT") == 1;
  WindowView* comb_views = h->d_views_inter + kViewHybComb * c.num_ranks;
  const int node_hi = h->node_base + h->node_size;
  // RCCL_EP_GIN_STREAM with the split: the remote owners' partials go one owner-token range per
  // chunk -- push, node barrier, fold, put -- so the rail starts after the first range's push,
  // not the whole push. Each range is the same rows, in the same order, as the whole push.
  const bool tok_stream = h->gin && h->gstream && split;
  if (split && !tok_stream)
    hipLaunchKernelGGL(combine_impl<1>, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, (const bf16_t*)y, (const float*)in_w, (const int32_t*)row_map,
                       (const int64_t*)expert_rows, (const float*)scales, fma_mode,
                       (const int32_t*)recv_topk, (const int32_t*)recv_src, num_recv,
                       1, h->self_view, comb_views, h->node_base, node_hi, 0, INT_MAX);
  else if (!split)
    hipLaunchKernelGGL(combine_impl<0>, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, (const bf16_t*)y, (const float*)in_w, (const int32_t*)row_map,
                       (const int64_t*)expert_rows, (const float*)scales, fma_mode,
                       (const int32_t*)recv_topk, (const int32_t*)recv_src, num_recv,
                       1, h->self_view, comb_views, 0, 0, 0, INT_MAX);
  if (!tok_stream) hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
  uint8_t* win = (uint8_t*)h->win_buf;
  const size_t T = c.num_max_tokens_per_rank, H = c.hidden, K = c.num_topk;
  HybInfo hk = hi;
  hk.nchunks = h->nchunks;
  hk.tok_split = tok_stream ? 1 : 0;
  if (h->gin) ++h->gseq_c;
  // RCCL_EP_GIN_STREAM: the previous combine's flush, before the folds rewrite comb_x / comb_w.
  if (h->gin && h->gstream) HC(hipStreamWaitEvent(s, h->ev_cflush, 0));
  for (int k = 0; k < hk.nchunks; ++k) {
    hk.chunk = k;
    if (tok_stream) {
      int t0, t1;
      hyb_tok_chunk((int)T, k, hk.nchunks, t0, t1);
      hipLaunchKernelGGL(combine_impl<1>, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                         c, (const bf16_t*)y, (const float*)in_w, (const int32_t*)row_map,
                         (const int64_t*)expert_rows, (const float*)scales, fma_mode,
                         (const int32_t*)recv_topk, (const int32_t*)recv_src, num_recv,
                         1, h->self_view, comb_views, h->node_base, node_hi, t0, t1);
      hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
    }
    hipLaunchKernelGGL(k_hyb_fold, dim3(hk.SO, hcta), dim3(kEpWaves * kWarpSize), 0, s, c, hk,
                       (const int32_t*)h->bound_fwd_meta, (const bf16_t*)(win + h->stage_off_x),
                       (const float*)(win + h->stage_off_w), with_w, h->comb_x, h->comb_w);
    HC(hipEventRecord(h->ev_a[k], s));
    HC(hipStreamWaitEvent(h->side, h->ev_a[k], 0));
    if (h->gin) {  // as in hybrid_dispatch_begin: put, then wait for the peers' chunk, on `side`
      const uint64_t ack = (k == 0 && h->gseq_c > 1) ? h->gseq_c - 1 : 0;
      const uint64_t tok = (h->gseq_c - 1) * hk.nchunks + k + 1;
      hipLaunchKernelGGL(k_gin_rail_put<true>, dim3(hk.SO, h->glanes), dim3(kWarpSize), 0, h->side, h->gdc,
                         h->gwin, hk, h->goff, T, H, K, with_w, h->glanes, h->glanes, ack, h->gtimeout, c.rank,
                         h->gprog, tok, h->gfwin, gin_flags(h), (const int32_t*)h->bound_fwd_meta,
                         (const int32_t*)nullptr, h->gstream ? 0 : 1, 1);
      h->gtok[1] = tok;
      if (h->gstream) continue;  // one arrival wait and the flush follow the last put, below
      hipLaunchKernelGGL(k_gin_rail_wait, dim3(1), dim3(kMaxNodes * h->glanes), 0, h->side, h->gdc, h->glanes,
                         h->glanes, hk.SO, hk.D, tok, h->gtimeout, c.rank, (int)kGinCombineData, h->gprog, gin_flags(h));
      HC(hipGetLastError());
      HC(hipEventRecord(h->ev_b[k], h->side));
      continue;
    }
    NC(ncclGroupStart());
    ncclResult_t gr = ncclSuccess;
    for (int n = 0; n < hk.SO; ++n) {
      if (n == hk.D) continue;
      const size_t np = n < hk.D ? n : n - 1;
      // Back along the rail (rail rank = node index): this rank folded nrecv[n] records for
      // node n's owner and gets back one row for each of the nsend[n] tokens it sent node n.
      int s0, s1, r0, r1;
      hyb_chunk(hk.nrecv[n], k, hk.nchunks, s0, s1);
      hyb_chunk(hk.nsend[n], k, hk.nchunks, r0, r1);
      if (s1 > s0) {
        NCG(gr, ncclSend(h->comb_x + (np * T + s0) * H, (size_t)(s1 - s0) * H * sizeof(bf16_t), ncclUint8, n, h->rail_comm, h->side));
        if (with_w) NCG(gr, ncclSend(h->comb_w + (np * T + s0) * K, (size_t)(s1 - s0) * K * sizeof(float), ncclUint8, n, h->rail_comm, h->side));
      }
      if (r1 > r0) {
        NCG(gr, ncclRecv(h->land_x + (np * T + r0) * H, (size_t)(r1 - r0) * H * sizeof(bf16_t), ncclUint8, n, h->rail_comm, h->side));
        if (with_w) NCG(gr, ncclRecv(h->land_w + (np * T + r0) * K, (size_t)(r1 - r0) * K * sizeof(float), ncclUint8, n, h->rail_comm, h->side));
      }
    }
    NCG_END(gr);
    HC(hipEventRecord(h->ev_b[k], h->side));
  }
  // RCCL_EP_GIN_STREAM: the chunks went back to back. With owner-token chunks the owner reduces
  // all but the last range once the one before last has landed, so waits for both of those
  // (ev_b[C-2], ev_b[C-1]); otherwise one wait for the last chunk's flags (each lane's flags
  // follow its data in order). Then the flush that lets the next combine's fold rewrite comb_x / comb_w.
  if (h->gin && h->gstream) {
    const uint64_t tok0 = (h->gseq_c - 1) * hk.nchunks;
    for (int k = tok_stream ? std::max(0, hk.nchunks - 2) : hk.nchunks - 1; k < hk.nchunks; ++k) {
      hipLaunchKernelGGL(k_gin_rail_wait, dim3(1), dim3(kMaxNodes * h->glanes), 0, h->side, h->gdc, h->glanes,
                         h->glanes, hk.SO, hk.D, tok0 + k + 1, h->gtimeout, c.rank, (int)kGinCombineData, h->gprog,
                         gin_flags(h));
      HC(hipEventRecord(h->ev_b[k], h->side));
    }
    hipLaunchKernelGGL(k_gin_rail_flush<true>, dim3(hk.SO, h->glanes), dim3(kWarpSize), 0, h->side, h->gdc, hk,
                       h->glanes, h->gprog, tok0 + hk.nchunks);
    HC(hipGetLastError());
    HC(hipEventRecord(h->ev_cflush, h->side));
  }
  if (split) {
    hipLaunchKernelGGL(combine_impl<2>, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, (const bf16_t*)y, (const float*)in_w, (const int32_t*)row_map,
                       (const int64_t*)expert_rows, (const float*)scales, fma_mode,
                       (const int32_t*)recv_topk, (const int32_t*)recv_src, num_recv,
                       1, h->self_view, comb_views, h->node_base, node_hi, 0, INT_MAX);
    hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
  }
  h->pend.on = true;
  h->pend.topk_idx = topk_idx;
  h->pend.bias0 = bias0;
  h->pend.bias1 = bias1;
  h->pend.out = out;
  h->pend.out_w = out_w;
  h->pend.num_tokens = num_tokens;
  h->pend.cta_reduce = cta_reduce;
  h->pend.tok_stream = tok_stream;
  return defer ? 0 : hybrid_combine_finish(h, s);
}

#define REQUIRE_BOUND(h, fn) do {                                                 \
    if ((h)->inter && !(h)->is_bound) {                                           \
      fprintf(stderr, "[rccl_ep] " fn " internode needs ep_inter_bind first\n");  \
      return -1;                                                                  \
    }                                                                             \
  } while (0)

// The arch check over the global communicator: an all-gather of every rank's arch string,
// in place of the LSA window the intranode check travels over, which spans one node here.
int check_uniform_arch_global(EpHandle* h, char* err, size_t err_len) {
  const int R = h->cfg.num_ranks;
  char mine[kArchLen];
  if (local_arch_string(h->cfg, h->device, mine, err, err_len) != 0) return -1;
  char* d = nullptr;
  if (hipMalloc(&d, (size_t)(R + 1) * kArchLen) != hipSuccess) {
    snprintf(err, err_len, "hipMalloc for the arch check failed");
    return -1;
  }
  char* all = (char*)malloc((size_t)R * kArchLen);
  int rc = -1;
  if (all != nullptr &&
      hipMemcpy(d + (size_t)R * kArchLen, mine, kArchLen, hipMemcpyHostToDevice) == hipSuccess &&
      ncclAllGather(d + (size_t)R * kArchLen, d, kArchLen, ncclUint8, h->comm, 0) == ncclSuccess &&
      hipDeviceSynchronize() == hipSuccess &&
      hipMemcpy(all, d, (size_t)R * kArchLen, hipMemcpyDeviceToHost) == hipSuccess) {
    rc = compare_arch_table(all, R, err, err_len);
  } else {
    snprintf(err, err_len, "arch all-gather failed");
  }
  free(all);
  hipFree(d);
  return rc;
}

}  // namespace

__global__ void k_capi_fill_views(ncclWindow_t w, EpWindowLayout l, int n, WindowView* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int p = 0; p < n; ++p) { out[p].base = (uint8_t*)ncclGetLsaPointer(w, 0, p); out[p].l = l; }
  }
}

// Queue probe for the normal-priority rail stream (create_stream_apart): k_queue_probe_wait, on
// the default stream, spins until k_queue_probe_set, on a candidate stream, raises the flag, or
// gives up after `timeout` wall-clock ticks. A candidate on the default stream's in-order hardware
// queue cannot run its kernel before the waiter returns, so the waiter times out.
__global__ void k_queue_probe_wait(int* flag, int* timed_out, long long timeout) {
  const long long t0 = wall_clock64();
  while (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 0) {
    if (wall_clock64() - t0 > timeout) { *timed_out = 1; return; }
    __builtin_amdgcn_s_sleep(1);
  }
}

__global__ void k_queue_probe_set(int* flag) {
  __hip_atomic_store(flag, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// One probe of candidate `c` against the default stream: sets `apart`. Returns -1 on error.
static int probe_queue_apart(hipStream_t c, int* d, hipEvent_t cleared, hipEvent_t waited,
                             long long timeout, bool* apart) {
  // The flag is cleared on the default stream and the candidate waits for that, so its kernel
  // cannot raise the flag before the clear, whichever queue it is on.
  HC(hipMemsetAsync(d, 0, 2 * sizeof(int), nullptr));
  HC(hipEventRecord(cleared, nullptr));
  k_queue_probe_wait<<<1, 1, 0, nullptr>>>(d, d + 1, timeout);
  HC(hipGetLastError());
  HC(hipEventRecord(waited, nullptr));
  HC(hipStreamWaitEvent(c, cleared, 0));
  k_queue_probe_set<<<1, 1, 0, c>>>(d);
  HC(hipGetLastError());
  HC(hipStreamWaitEvent(c, waited, 0));
  int timed_out = 0;
  HC(hipMemcpyAsync(&timed_out, d + 1, sizeof(int), hipMemcpyDeviceToHost, c));
  HC(hipStreamSynchronize(c));
  *apart = timed_out == 0;
  return 0;
}

// A normal-priority stream on a hardware queue that the legacy default stream, where torch runs
// its compute, does not use. HIP gives each new stream one of GPU_MAX_HW_QUEUES queues per
// priority, in an order that differs between HIP versions, so a plain new stream can share the
// default stream's queue and then serializes with it. Candidates are created until one passes the
// queue probe; the others are destroyed. Returns the number of candidates tried when one passed,
// 0 when none did (`out` then keeps the last), -1 on error (`out` untouched, nothing leaked).
static int create_stream_apart(int device, hipStream_t* out) {
  constexpr int kTries = 4;
  int rate_khz = 0;
  HC(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, device));
  const long long timeout = 50LL * rate_khz;  // 50 ms
  int* d = nullptr;
  hipEvent_t cleared = nullptr, waited = nullptr;
  hipStream_t cand[kTries] = {};
  int n = 0, rc = 0;
  bool apart = false;
  if (hipMalloc(&d, 2 * sizeof(int)) != hipSuccess ||
      hipEventCreateWithFlags(&cleared, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&waited, hipEventDisableTiming) != hipSuccess) {
    rc = -1;
  }
  while (rc == 0 && n < kTries && !apart) {
    if (hipStreamCreateWithFlags(&cand[n], hipStreamNonBlocking) != hipSuccess) { rc = -1; break; }
    ++n;
    rc = probe_queue_apart(cand[n - 1], d, cleared, waited, timeout, &apart);
  }
  // Keep the last candidate, the one that passed or the last tried; release the rest.
  for (int i = 0; i < n - (rc == 0 ? 1 : 0); ++i) (void)hipStreamDestroy(cand[i]);
  if (rc == 0) *out = cand[n - 1];
  if (cleared) (void)hipEventDestroy(cleared);
  if (waited) (void)hipEventDestroy(waited);
  if (d) (void)hipFree(d);
  return rc < 0 ? -1 : (apart ? n : 0);
}

// The hybrid scale-out buffers and the record layout (see HybInfo). Sized for a full window
// of tokens per remote node, since any token may go to any node.
static int hybrid_alloc(EpHandle* h) {
  const EpConfig& c = h->cfg;
  HybInfo& hi = h->hyb;
  const size_t T = c.num_max_tokens_per_rank, H = c.hidden, K = c.num_topk, rn = hi.SO - 1;
  size_t o = align_up(H * sizeof(bf16_t), 16);
  hi.off_sf = (int)o;
  o = align_up(o + (size_t)c.hidden_sf() * sizeof(float), 16);
  hi.off_topk = (int)o;
  o = align_up(o + K * sizeof(int32_t), 16);
  hi.off_w = (int)o;
  o = align_up(o + K * sizeof(float), 16);
  hi.off_src = (int)o;
  o = align_up(o + sizeof(int32_t), 16);
  hi.off_slots = (int)o;
  hi.rec_bytes = (int)align_up(o + (size_t)hi.SU * sizeof(int32_t), 16);
  const size_t rec = rn * T * hi.rec_bytes, xb = rn * T * H * sizeof(bf16_t), wb = rn * T * K * sizeof(float);
  if (h->gin) {
    // One rail window holds all six, so a put can address any of them on the peer.
    // Registration is collective over rail_comm, as ep_configure is over every rank.
    size_t off = 0;
    auto take = [&off](size_t b) { const size_t at = off; off = align_up(off + b, 4096); return at; };
    GinRailOffs& g = h->goff;
    g.so_send = take(rec);
    g.so_recv = take(rec);
    g.comb_x = take(xb);
    g.land_x = take(xb);
    g.comb_w = take(wb);
    g.land_w = take(wb);
    if (ncclMemAlloc(&h->gbuf, off) != ncclSuccess ||
        ncclCommWindowRegister(h->rail_comm, h->gbuf, off, &h->gwin, NCCL_WIN_DEFAULT) != ncclSuccess ||
        h->gwin == nullptr) {
      fprintf(stderr, "[rccl_ep] GIN rail window (%zu bytes) failed\n", off);
      return -1;
    }
    uint8_t* b = (uint8_t*)h->gbuf;
    h->so_send = b + g.so_send;
    h->so_recv = b + g.so_recv;
    h->comb_x = (bf16_t*)(b + g.comb_x);
    h->land_x = (bf16_t*)(b + g.land_x);
    h->comb_w = (float*)(b + g.comb_w);
    h->land_w = (float*)(b + g.land_w);
  } else if (hipMalloc(&h->so_send, rec) != hipSuccess || hipMalloc(&h->so_recv, rec) != hipSuccess ||
             hipMalloc(&h->comb_x, xb) != hipSuccess || hipMalloc(&h->land_x, xb) != hipSuccess ||
             hipMalloc(&h->comb_w, wb) != hipSuccess || hipMalloc(&h->land_w, wb) != hipSuccess) {
    fprintf(stderr, "[rccl_ep] hybrid buffers (%zu records of %d bytes per node) failed\n", T, hi.rec_bytes);
    return -1;
  }
  if (hipMalloc(&h->sink, 4096) != hipSuccess ||
      hipMalloc(&h->d_local_sendc, c.num_ranks * sizeof(int32_t)) != hipSuccess) {
    fprintf(stderr, "[rccl_ep] hybrid count buffers failed\n");
    return -1;
  }
  return 0;
}

// RCCL_EP_GIN_WATCH_S (default 60, 0 = off): a host thread that reports, once per stall, GIN
// transfers that have stopped moving for that long: per peer node, the last token each stage of
// this rank's put, wait and credit kernels reached (lowest over lanes), against what the host has
// issued, and the rail communicator's async error (a dead GIN proxy thread shows only there). It
// also prints the record of a wait that timed out on the device, whose own printf dies with the trap.
static void gin_watch(EpHandle* h) {
  const int SO = h->hyb.SO, D = h->hyb.D, L = h->glanes, rank = h->cfg.rank;
  static const char* const kinds[] = {"dispatch data", "combine data", "dispatch credit", "combine credit"};
  GinProgress last{}, now{};
  int idle = 0, tick = 0;
  bool reported = false, to_reported = false, err_reported = false;
  while (!h->gwatch_stop) {
    for (int i = 0; i < 10 && !h->gwatch_stop; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    memcpy(&now, (const void*)h->gprog, sizeof(now));
    if (now.to_set == 1 && !to_reported) {
      fprintf(stderr, "[rccl_ep] GIN wait timed out on rank %d: %s from node %llu, context %llu, signal %llu at %llu of %llu\n",
              rank, kinds[now.to_kind & 3], (unsigned long long)now.to_node, (unsigned long long)now.to_ctx,
              (unsigned long long)now.to_sig, (unsigned long long)now.to_seen, (unsigned long long)now.to_need);
      to_reported = true;
    }
    if (++tick % 10 == 0 && !err_reported) {
      ncclResult_t ar = ncclSuccess;
      if (ncclCommGetAsyncError(h->rail_comm, &ar) == ncclSuccess && ar != ncclSuccess && ar != ncclInProgress) {
        fprintf(stderr, "[rccl_ep] GIN rail communicator async error on rank %d: %s\n", rank, ncclGetErrorString(ar));
        err_reported = true;
        // The GIN progress thread has exited, so nothing posted on this rail will complete: the
        // job would hang until a watchdog. RCCL_EP_GIN_ABORT_ON_ERROR=0 keeps it up for inspection.
        const char* ab = getenv("RCCL_EP_GIN_ABORT_ON_ERROR");
        if (!(ab && ab[0] == '0')) {
          fprintf(stderr, "[rccl_ep] aborting: the GIN rail cannot recover (RCCL_EP_GIN_ABORT_ON_ERROR=0 to wait instead)\n");
          fflush(stderr);
          abort();
        }
      }
    }
    if (memcmp(&now, &last, sizeof(now)) != 0) {
      last = now;
      idle = 0;
      reported = false;
      continue;
    }
    if (++idle < h->gwatch_s || reported) continue;
    const uint64_t tok[2] = {h->gtok[0].load(), h->gtok[1].load()}, ack[2] = {h->gack[0].load(), h->gack[1].load()};
    bool pending = false;
    for (int d = 0; d < 2; ++d) {
      pending |= now.ack_done[d] < ack[d];
      for (int n = 0; n < SO; ++n)
        for (int l = 0; l < L && n != D; ++l)
          pending |= now.put_done[d][n * kGinMaxLanes + l] < tok[d] || now.wait_done[d][n * kGinMaxLanes + l] < tok[d];
    }
    if (!pending) continue;
    char line[2048];
    int o = snprintf(line, sizeof(line), "[rccl_ep] GIN stall on rank %d (node %d of %d), %d s without progress:",
                     rank, D, SO, idle);
    for (int d = 0; d < 2 && o < (int)sizeof(line); ++d) {
      o += snprintf(line + o, sizeof(line) - o, " %s issued %llu wait_enter %llu credits %llu/%llu |",
                    d ? "combine" : "dispatch", (unsigned long long)tok[d], (unsigned long long)now.wait_enter[d],
                    (unsigned long long)now.ack_done[d], (unsigned long long)ack[d]);
      for (int n = 0; n < SO && o < (int)sizeof(line); ++n) {
        if (n == D) continue;
        uint64_t m[5] = {~0ull, ~0ull, ~0ull, ~0ull, ~0ull};
        for (int l = 0; l < L; ++l) {
          const int sl = n * kGinMaxLanes + l;
          m[0] = std::min(m[0], now.put_enter[d][sl]);
          m[1] = std::min(m[1], now.put_credit[d][sl]);
          m[2] = std::min(m[2], now.put_posted[d][sl]);
          m[3] = std::min(m[3], now.put_done[d][sl]);
          m[4] = std::min(m[4], now.wait_done[d][sl]);
        }
        o += snprintf(line + o, sizeof(line) - o, " n%d put %llu/%llu/%llu/%llu recv %llu |", n,
                      (unsigned long long)m[0], (unsigned long long)m[1], (unsigned long long)m[2],
                      (unsigned long long)m[3], (unsigned long long)m[4]);
      }
    }
    fprintf(stderr, "%s (put: entered/credited/posted/flushed)\n", line);
    reported = true;
  }
}

// The rail over GIN (RCCL_EP_GIN, device/gin_rail.h): a device communicator on rail_comm with
// GIN connections between the rail peers. Needs GIN on when the communicators were created
// (NCCL_GIN_ENABLE=1, the default), symmetric windows and cuMem: NCCL_WIN_ENABLE=1 (RCCL also
// takes NCCL_DEVR_INIT=1; RCCLX only WIN_ENABLE) and NCCL_CUMEM_ENABLE=1. Collective over
// rail_comm.
static int gin_rail_setup(EpHandle* h) {
  ncclCommProperties_t p = NCCL_COMM_PROPERTIES_INITIALIZER;
  if (ncclCommQueryProperties(h->rail_comm, &p) != ncclSuccess || !p.deviceApiSupport ||
      p.railedGinType == NCCL_GIN_TYPE_NONE) {
    fprintf(stderr,
            "[rccl_ep] RCCL_EP_GIN: the rail has no GIN (device API %d, GIN type %d; the device "
            "API needs NCCL_WIN_ENABLE=1 and NCCL_CUMEM_ENABLE=1)\n",
            (int)p.deviceApiSupport, (int)p.railedGinType);
    return -1;
  }
  // rocSHMEM GDA (NCCL_GIN_TYPE=5): GPU-initiated, no proxy thread. Its contexts share their
  // connection's queue pairs and signals, so the per-lane, per-direction signal ids of
  // RCCL_EP_GIN_SIGNAL=signal would alias (and they are RDMA atomics); only flags are supported.
  // Every early return here depends only on the GIN type and the environment, which every rank
  // shares, so all ranks take the same branch before the collective ncclDevCommCreate.
  const bool gda = p.railedGinType == NCCL_GIN_TYPE_ROCSHMEM_GDA;
#if !NCCL_GIN_ROCSHMEM_GDA_ENABLE
  if (gda) {
    fprintf(stderr, "[rccl_ep] RCCL_EP_GIN: the rail's GIN is rocSHMEM GDA, but librccl_ep.so was built without it\n");
    return -1;
  }
#endif
  const char* sm = getenv("RCCL_EP_GIN_SIGNAL");
  h->gwflag = !(sm && strcmp(sm, "signal") == 0);
  if (gda && !h->gwflag) {
    fprintf(stderr, "[rccl_ep] RCCL_EP_GIN_SIGNAL=signal is not supported on GDA\n");
    return -1;
  }
  // RCCL_EP_GIN_TIMEOUT_S (default 300, half torch's collective timeout so the report comes first;
  // 0 = wait forever): how long a GIN wait may spin before it reports what it waited for and
  // traps. wall_clock64 runs at 100 MHz.
  const char* to = getenv("RCCL_EP_GIN_TIMEOUT_S");
  h->gtimeout = (uint64_t)std::max(0, to ? atoi(to) : 300) * 100000000ull;
  // One context per lane and direction: dispatch [0, L), combine [L, 2L). On GDA each context
  // has its own queue pairs, on alternating UDMA engines.
  const char* ln = getenv("RCCL_EP_GIN_LANES");
  h->glanes = std::max(1, std::min(ln ? atoi(ln) : 4, kGinMaxLanes));
  h->gstream = env_int("RCCL_EP_GIN_STREAM") == 1;
  ncclDevCommRequirements req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  req.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  req.ginContextCount = 2 * h->glanes;
  req.ginSignalCount = gda ? 0 : kGinSignals;  // GDA runs flags only, which need no signals
  NC(ncclDevCommCreate(h->rail_comm, &req, &h->gdc));
  h->gdc_made = true;
  // The kernels handle PROXY, and GDA when built for it; any other backend would run them as UB.
  const int want = gda ? (int)NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA : (int)NCCL_NET_DEVICE_GIN_PROXY;
  bool ok = h->gdc.ginContextCount >= 2 * h->glanes && h->gdc.ginConnectionCount > 0;
  for (int c = 0; c < h->gdc.ginConnectionCount; ++c)
    ok = ok && (int)h->gdc.ginNetDeviceTypes[c] == want;
  if (!ok) {
    fprintf(stderr, "[rccl_ep] RCCL_EP_GIN: unsupported GIN device communicator (%d contexts, type %d)\n",
            (int)h->gdc.ginContextCount, (int)h->gdc.ginNetDeviceTypes[0]);
    return -1;
  }
  if (h->cfg.rank == 0)
    fprintf(stderr, "[rccl_ep] GIN rail: %s, %d lane(s), %d connection(s)%s\n", gda ? "rocSHMEM GDA" : "proxy",
            h->glanes, (int)h->gdc.ginConnectionCount, h->gstream ? ", streamed chunks" : "");
  // The flag window (unless RCCL_EP_GIN_SIGNAL=signal), collective over rail_comm.
  if (h->gwflag) {
    const size_t fb = kGinFlagWords * sizeof(uint64_t);
    NC(ncclMemAlloc(&h->gfbuf, fb));
    HC(hipMemset(h->gfbuf, 0, fb));
    HC(hipDeviceSynchronize());
    NC(ncclCommWindowRegister(h->rail_comm, h->gfbuf, fb, &h->gfwin, NCCL_WIN_STRICT_ORDERING));
    if (!h->gfwin) {
      fprintf(stderr, "[rccl_ep] RCCL_EP_GIN: flag window registration failed\n");
      return -1;
    }
  }
  HC(hipHostMalloc((void**)&h->gprog, sizeof(GinProgress), hipHostMallocCoherent | hipHostMallocMapped));
  memset(h->gprog, 0, sizeof(GinProgress));
  const char* ws = getenv("RCCL_EP_GIN_WATCH_S");
  h->gwatch_s = std::max(0, ws ? atoi(ws) : 60);
  h->gwatch_stop = false;
  if (h->gwatch_s > 0) h->gwatch = std::thread(gin_watch, h);
  h->gin = true;
  return 0;
}

// Internode views: a local peer's is its LSA window; a remote peer's depends on the table
// (InterView). Flat tables aim at the outbox through remote_layout, the hybrid local push at a
// count sink, and the hybrid combine at the forwarder (the local rank with the owner's local
// index), shifted so this rank's row for owner token t lands at staging row (np * SU + u) * T + t.
// d_views gets the bf16 table. Also allocates the flat-path buffers if needed, and the hybrid ones.
static int inter_fill_views(EpHandle* h) {
  const EpConfig& c = h->cfg;
  const int R = c.num_ranks;
  uint8_t** d_bases = nullptr;
  uint8_t* bases[kMaxWorldRanks] = {};
  if (hipMalloc(&d_bases, h->node_size * sizeof(uint8_t*)) != hipSuccess) return -1;
  hipLaunchKernelGGL(k_inter_lsa_bases, dim3(1), dim3(1), 0, 0, h->win, h->node_size, d_bases);
  const bool ok = hipMemcpy(bases, d_bases, h->node_size * sizeof(uint8_t*), hipMemcpyDeviceToHost) == hipSuccess;
  hipFree(d_bases);
  // The outbox and the inbox each take a full window. Most of each goes unused, but sharing the
  // window's layout is what lets the intranode kernels write the outbox. A hybrid handle that
  // reduces within the node never takes the flat path, so it skips them.
  const bool flat = !(h->hybrid && h->hyb_multi);
  const size_t lay_bytes = h->lay.total_bytes;
  if (!ok ||
      (flat && (hipMalloc(&h->outbox, lay_bytes) != hipSuccess ||
                hipMalloc(&h->inbox, lay_bytes) != hipSuccess ||
                hipMalloc(&h->d_src_inter, (size_t)R * c.num_max_tokens_per_rank * sizeof(int32_t)) != hipSuccess)) ||
      hipMalloc(&h->d_views_inter, kNumInterViews * R * sizeof(WindowView)) != hipSuccess) {
    fprintf(stderr, "[rccl_ep] internode buffers (2 x %zu bytes) failed\n", lay_bytes);
    return -1;
  }
  if (h->hybrid && hybrid_alloc(h) != 0) return -1;
  EpWindowLayout sink_lay{};  // every offset 0: only counts()[rank] is ever stored through it
  const HybInfo& hi = h->hyb;
  WindowView views[kNumInterViews * kMaxWorldRanks];
  for (int kind = 0; kind < kNumInterViews; ++kind) {
    for (int p = 0; p < R; ++p) {
      WindowView& v = views[kind * R + p];
      if (!is_remote(h, p)) {
        v = WindowView{bases[p - h->node_base], h->lay};
      } else if (kind < kViewHybLocal) {
        v = flat ? WindowView{h->outbox, remote_layout(h->lay, c, p, kind)} : WindowView{h->sink, sink_lay};
      } else if (kind == kViewHybLocal || !h->hybrid) {
        v = WindowView{h->sink, sink_lay};
      } else {
        const int n = p / hi.SU, np = n < hi.D ? n : n - 1;
        const int64_t T = c.num_max_tokens_per_rank, H = c.hidden, K = c.num_topk;
        const int64_t row = (int64_t)(np * hi.SU + hi.u) * T - (int64_t)c.rank * T;
        EpWindowLayout l = h->lay;
        l.off_y = h->stage_off_x + (size_t)(row * H * (int64_t)sizeof(bf16_t));
        l.off_cw = h->stage_off_w + (size_t)(row * K * (int64_t)sizeof(float));
        v = WindowView{bases[p % hi.SU], l};
      }
    }
  }
  if (hipMemcpy(h->d_views_inter, views, kNumInterViews * R * sizeof(WindowView), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(h->d_views, views + kViewBf16 * R, R * sizeof(WindowView), hipMemcpyHostToDevice) != hipSuccess)
    return -1;
  return 0;
}

// librccl is built -fvisibility=hidden. Without this the entry points below
// become STB_LOCAL, the library still links, and the ctypes binding fails at
// import with "undefined symbol: ep_create".
#pragma GCC visibility push(default)
extern "C" {

int ep_unique_id_size() { return (int)sizeof(ncclUniqueId); }

int ep_get_unique_id(char* out) {
  ncclUniqueId id;
  if (ncclGetUniqueId(&id) != ncclSuccess) return -1;
  memcpy(out, &id, sizeof(id));
  return 0;
}

void* ep_create_v2(int rank, int num_ranks, const char* unique_id,
                   int max_tokens_per_rank, int hidden, int device, int node_size);

// unique_id must be ncclUniqueId bytes broadcast from rank 0 by the caller.
void* ep_create(int rank, int num_ranks, const char* unique_id,
                int max_tokens_per_rank, int hidden, int device) {
  return ep_create_v2(rank, num_ranks, unique_id, max_tokens_per_rank, hidden, device, num_ranks);
}

// ep_create for num_ranks / node_size nodes, node n holding ranks [n * node_size,
// (n + 1) * node_size); node_size == num_ranks is ep_create. A peer on another node is
// reached by RCCL send/recv (see EpHandle).
void* ep_create_v2(int rank, int num_ranks, const char* unique_id,
                   int max_tokens_per_rank, int hidden, int device, int node_size) {
  // Bind explicitly. The caller's torch context may have selected a different
  // device than this thread's current one.
  hipError_t he = hipSetDevice(device);
  if (he != hipSuccess) {
    fprintf(stderr, "[rccl_ep] hipSetDevice(%d): %s\n", device, hipGetErrorString(he));
    return nullptr;
  }
  // Payload copies move 16B per lane from a 256B-aligned region base, so the
  // vector paths are only aligned when hidden is a multiple of 8. The scalar
  // tails cover a leftover element count, not a misaligned base, so reject an
  // odd hidden here rather than reach a misaligned dwordx4.
  if (hidden <= 0 || hidden % 8 != 0) {
    fprintf(stderr, "[rccl_ep] hidden %d must be a positive multiple of 8 (16B vector alignment)\n", hidden);
    return nullptr;
  }
  // kMaxRanks bounds the scale-up domain the window layout is written for, kMaxWorldRanks the
  // internode group, and ctas_per_rank divides by num_ranks, so zero is a division by zero.
  const int max_ranks = node_size < num_ranks ? kMaxWorldRanks : kMaxRanks;
  if (num_ranks < 1 || num_ranks > max_ranks || node_size > kMaxRanks) {
    fprintf(stderr, "[rccl_ep] num_ranks %d must be in [1, %d], node_size %d at most kMaxRanks=%d\n",
            num_ranks, max_ranks, node_size, kMaxRanks);
    return nullptr;
  }
  // EpWindowLayout::make widens this to size_t, so a negative value sign-extends
  // and wraps every offset in the window; zero reserves empty regions that the
  // slot arithmetic then indexes anyway.
  if (max_tokens_per_rank < 1) {
    fprintf(stderr, "[rccl_ep] max_tokens_per_rank %d must be positive\n", max_tokens_per_rank);
    return nullptr;
  }
  if (node_size < 1 || num_ranks % node_size != 0) {
    fprintf(stderr, "[rccl_ep] node_size %d must divide num_ranks %d\n", node_size, num_ranks);
    return nullptr;
  }
  if (num_ranks / node_size > kMaxNodes) {
    fprintf(stderr, "[rccl_ep] %d nodes exceed kMaxNodes=%d\n", num_ranks / node_size, kMaxNodes);
    return nullptr;
  }

  auto* h = new EpHandle();
  h->cfg.num_ranks = num_ranks;
  h->cfg.rank = rank;
  h->cfg.hidden = hidden;
  h->cfg.num_max_tokens_per_rank = max_tokens_per_rank;
  h->device = device;

  ncclUniqueId id;
  memcpy(&id, unique_id, sizeof(id));
  ncclResult_t nr = ncclCommInitRank(&h->comm, num_ranks, id, rank);
  if (nr != ncclSuccess) {
    fprintf(stderr, "[rccl_ep] ncclCommInitRank: %s\n", ncclGetErrorString(nr));
    delete h;
    return nullptr;
  }
  h->lsa = h->comm;
  h->node_size = node_size;
  if (node_size < num_ranks) {
    h->inter = true;
    h->node_base = rank / node_size * node_size;
    nr = ncclCommSplit(h->comm, rank / node_size, rank, &h->node_comm, nullptr);
    if (nr != ncclSuccess ||
        hipMalloc(&h->d_scratch, kScratchInts * sizeof(int32_t)) != hipSuccess ||
        hipHostMalloc((void**)&h->host_scratch, kScratchInts * sizeof(int32_t)) != hipSuccess) {
      fprintf(stderr, "[rccl_ep] internode setup failed: %s\n",
              nr != ncclSuccess ? ncclGetErrorString(nr) : "scratch allocation");
      release_inter_comm(h);
      ncclCommDestroy(h->comm);
      delete h;
      return nullptr;
    }
    h->lsa = h->node_comm;
  }
  // Unchecked, a failure here returns a handle of null device pointers that
  // faults inside a kernel later, leaking the comm.
  const bool alloced =
      hipMalloc(&h->d_recvoff, num_ranks * sizeof(int32_t)) == hipSuccess &&
      hipMalloc(&h->d_total, sizeof(int32_t)) == hipSuccess &&
      hipMalloc(&h->d_views, num_ranks * sizeof(WindowView)) == hipSuccess &&
      hipHostMalloc((void**)&h->host_counts, (2 + kMaxLocalExperts + kMaxRanks) * sizeof(int32_t),
                    hipHostMallocMapped | hipHostMallocCoherent) == hipSuccess &&
      hipHostGetDevicePointer((void**)&h->host_counts_dev, h->host_counts, 0) == hipSuccess;
  if (!alloced) {
    fprintf(stderr, "[rccl_ep] hipMalloc for scratch failed\n");
    hipFree(h->d_recvoff);
    hipFree(h->d_total);
    hipFree(h->d_views);
    if (h->host_counts) hipHostFree(h->host_counts);
    release_inter_comm(h);
    ncclCommDestroy(h->comm);
    delete h;
    return nullptr;
  }
  return h;
}

// Size and register the symmetric window. Separate from ep_create because the
// window size depends on num_topk and num_experts, which the API only
// reveals at the first dispatch, not at buffer construction.
int ep_configure_v2(void* handle, int num_experts, int num_topk, int grouped_only);

int ep_configure(void* handle, int num_experts, int num_topk) {
  return ep_configure_v2(handle, num_experts, num_topk, 0);
}

// ep_configure with grouped_only: the window's y region holds one row per slot instead of one
// per (slot, k), for a caller that only ever combines grouped (non-expanded handles, or
// allow_multiple_reduction) -- num_topk times smaller, the largest region by far.
int ep_configure_v2(void* handle, int num_experts, int num_topk, int grouped_only) {
  auto* h = static_cast<EpHandle*>(handle);
  if (h->configured && h->cfg.num_experts == num_experts && h->cfg.num_topk == num_topk &&
      h->grouped_only == (grouped_only != 0)) {
    return 0;
  }
  // The hybrid fold and reduce keep up to 8 rank partials and 8 node partials per token.
  if (h->hybrid && h->hyb_multi && num_topk > 8) {
    fprintf(stderr, "[rccl_ep] the hybrid combine supports num_topk <= 8, got %d\n", num_topk);
    return -1;
  }
  // Validated before the teardown below, so a rejected reconfigure leaves the
  // existing window intact rather than freed.
  if (num_topk < 1 || num_topk > kMaxTopk) {
    fprintf(stderr, "[rccl_ep] num_topk %d must be in [1, kMaxTopk=%d]\n", num_topk, kMaxTopk);
    return -1;
  }
  // experts_per_rank() truncates, so a non-multiple leaves the tail experts
  // owned by no rank: k_ep_plan never routes their tokens, and the epilogue's
  // owner = e / experts_per_rank() reaches num_ranks, indexing one whole
  // rank-region past the end of `y`. num_experts <= 0 is the epr == 0 case.
  if (num_experts <= 0 || num_experts % h->cfg.num_ranks != 0) {
    fprintf(stderr, "[rccl_ep] num_experts %d must be a positive multiple of num_ranks %d\n",
            num_experts, h->cfg.num_ranks);
    return -1;
  }
  // The notify payload and the plan's LDS histogram hold one int per local expert.
  if (num_experts / h->cfg.num_ranks > kMaxLocalExperts) {
    fprintf(stderr, "[rccl_ep] %d experts per rank exceeds kMaxLocalExperts=%d\n",
            num_experts / h->cfg.num_ranks, kMaxLocalExperts);
    return -1;
  }
  if (h->configured) {
    release_window(h, true);
    h->configured = false;
  }
  h->cfg.num_experts = num_experts;
  h->cfg.num_topk = num_topk;
  h->grouped_only = grouped_only != 0;
  h->payload_later = false;  // a new window holds no deferred payload
  h->perm.on = false;
  // RCCL_EP_HYB_Y_LOCAL=1: in a hybrid window that reduces within the node, only this node's
  // ranks write y and cw -- each its own slot(rank, t) in a local owner's window, while rows for a
  // remote owner go to the forwarder's staging -- and the reduce epilogue reads only those. So
  // y and cw keep this node's sources alone, (num_ranks - node_size) * T rows smaller.
  // Such a window always gets one y row per slot: the hybrid combine is grouped only (see
  // ep_combine_v4), and the y base moves down by node_base * T layout rows, which with
  // num_topk rows per slot would not line up with the combine's slot rows on any node but 0.
  const bool y_local = h->hybrid && h->hyb_multi && env_int("RCCL_EP_HYB_Y_LOCAL") == 1;
  h->lay = EpWindowLayout::make(h->cfg, sizeof(bf16_t), (h->grouped_only || y_local) ? 1 : 0,
                                y_local ? h->node_size : 0, y_local ? h->node_base : 0);
  h->win_bytes = h->lay.total_bytes;
  if (h->hybrid && h->hyb_multi) {
    // The combine staging goes past the layout's end, inside the registered window, where
    // this node's ranks store their partials for the owners this rank forwards for.
    const size_t rows = (size_t)(h->hyb.SO - 1) * h->hyb.SU * h->cfg.num_max_tokens_per_rank;
    h->stage_off_x = align_up(h->win_bytes);
    h->stage_off_w = align_up(h->stage_off_x + rows * h->cfg.hidden * sizeof(bf16_t));
    h->win_bytes = align_up(h->stage_off_w + rows * num_topk * sizeof(float));
  }

  ncclResult_t nr = ncclMemAlloc(&h->win_buf, h->win_bytes);
  if (nr != ncclSuccess) {
    fprintf(stderr, "[rccl_ep] ncclMemAlloc(%zu): %s\n", h->win_bytes, ncclGetErrorString(nr));
    h->win_buf = nullptr;
    return -1;
  }
  hipMemset(h->win_buf, 0, h->win_bytes);
  h->notify_seq = 0;  // the memset just zeroed every flag
  h->stream_seq = 0;  // and every progress word
  h->cstream_seq = 0;
  {
    const char* e = getenv("RCCL_EP_ELIDE_A");
    // The rendezvous argument (see EpHandle) holds for hybrid too: only local ranks, the
    // forwarders included, write this window. The flat internode path keeps every barrier.
    h->elide_a = e != nullptr && e[0] == '1' && (!h->inter || h->hybrid);
    h->rdv_x = h->rdv_y = h->pending_x = false;
    h->last_s = nullptr;
  }
  {
    const size_t chunks = (h->cfg.num_max_tokens_per_rank + kPlanChunk - 1) / kPlanChunk;
    const size_t agg_bytes = (size_t)h->cfg.num_ranks * chunks *
                             kPlanAggStride(h->cfg.experts_per_rank(), h->cfg.num_ranks) *
                             sizeof(int32_t);
    if (hipMalloc(&h->d_plan_agg, agg_bytes) != hipSuccess ||
        hipMemset(h->d_plan_agg, 0, agg_bytes) != hipSuccess) {
      fprintf(stderr, "[rccl_ep] plan scratch (%zu bytes) failed\n", agg_bytes);
      release_window(h, false);
      return -1;
    }
  }
  nr = ncclCommWindowRegister(h->lsa, h->win_buf, h->win_bytes, &h->win, NCCL_WIN_COLL_SYMMETRIC);
  if (nr != ncclSuccess) {
    fprintf(stderr, "[rccl_ep] WindowRegister: %s (needs NCCL_WIN_ENABLE=1 and NCCL_CUMEM_ENABLE=1)\n", ncclGetErrorString(nr));
    release_window(h, false);
    return -1;
  }

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = kEpBarrierCtas;  // LSA only: no GIN contexts on the intranode path
  nr = ncclDevCommCreate(h->lsa, &reqs, &h->devComm);
  if (nr != ncclSuccess) {
    fprintf(stderr, "[rccl_ep] ncclDevCommCreate: %s\n", ncclGetErrorString(nr));
    release_window(h, true);
    return -1;
  }
  h->devcomm = true;

  // k_capi_fill_views resolves peer windows by LSA index while every data-path
  // kernel indexes d_views by world rank, so the two agree only when the LSA
  // team is this rank's node under the identity mapping. A comm that spans
  // nodes without ep_create_v2, or one whose ranks are reordered within the
  // node, would otherwise push payloads into the wrong peer's window with no error.
  if (h->devComm.lsaSize != h->node_size || h->devComm.lsaRank != h->cfg.rank - h->node_base) {
    fprintf(stderr,
            "[rccl_ep] LSA team is rank %d of %d but this node is ranks [%d, %d) of %d, this rank %d; "
            "rccl_ep needs every node's ranks contiguous, with the identity LSA mapping "
            "(a communicator spanning nodes needs ep_create_v2)\n",
            h->devComm.lsaRank, h->devComm.lsaSize, h->node_base, h->node_base + h->node_size,
            h->cfg.num_ranks, h->cfg.rank);
    release_window(h, true);
    return -1;
  }

  h->self_view = WindowView{(uint8_t*)h->win_buf, h->lay};
  if (h->inter) {
    if (inter_fill_views(h) != 0) {
      release_window(h, true);
      return -1;
    }
  } else {
    hipLaunchKernelGGL(k_capi_fill_views, dim3(1), dim3(1), 0, 0,
                       h->win, h->lay, h->cfg.num_ranks, h->d_views);
  }
  // Spelled out rather than wrapped in HC: the macro returns before the window
  // can be handed back.
  hipError_t he = hipDeviceSynchronize();
  if (he != hipSuccess) {
    fprintf(stderr, "[rccl_ep] view fill: %s\n", hipGetErrorString(he));
    release_window(h, true);
    return -1;
  }

  // Before any payload moves: a mixed-architecture communicator has no correct
  // behaviour, only a slower way to be wrong. This runs here rather than in
  // ep_create because it travels over the LSA window, which does not exist
  // until now.
  char err[256];
  if ((h->inter ? check_uniform_arch_global(h, err, sizeof(err))
                : check_uniform_arch(h->cfg, h->device, h->d_views, h->self_view,
                                     h->devComm, err, sizeof(err))) != 0) {
    fprintf(stderr, "[rccl_ep] %s\n", err);
    release_window(h, true);
    return -1;
  }

  h->configured = true;
  return 0;
}

size_t ep_window_bytes(void* handle) {
  return static_cast<EpHandle*>(handle)->win_bytes;
}

void ep_destroy(void* handle) {
  auto* h = static_cast<EpHandle*>(handle);
  if (!h) return;
  // Unconditional rather than gated on `configured`: a handle whose ep_configure
  // failed part-way has already handed back what it took, and release_window is
  // a no-op on the flags it cleared, so this covers both without double-freeing.
  release_window(h, h->configured);
  hipFree(h->d_recvoff);
  hipFree(h->d_total);
  hipFree(h->d_views);
  if (h->host_counts) hipHostFree(h->host_counts);
  release_inter_comm(h);  // the split goes before its parent
  ncclCommDestroy(h->comm);
  delete h;
}

int ep_barrier(void* handle, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_barrier");
  hipStream_t s = (hipStream_t)stream;
  note_stream(h, s);
  if (h->inter) {
    // The LSA team is one node: rendezvous over the global communicator instead.
    NC(ncclAllReduce(h->d_scratch, h->d_scratch, 1, ncclInt32, ncclSum, h->comm, s));
  } else {
    hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
  }
  note_rdv(h);
  HC(hipStreamSynchronize(s));
  return 0;
}

// Routing plan. Written into caller-owned tensors so a cached dispatch replays
// the identical plan rather than depending on library state.
int ep_plan(void* handle, uintptr_t topk_idx, int num_tokens,
            uintptr_t out_slot, uintptr_t out_list, uintptr_t out_sendc,
            uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_plan");
  hipStream_t s = (hipStream_t)stream;
  // A full block per destination; k_ep_plan sizes its per-wave scan from blockDim.
  static_assert(kEpWaves <= kPlanMaxWaves, "k_ep_plan's wave scan is sized by kPlanMaxWaves");
  hipLaunchKernelGGL(k_ep_plan, dim3(h->cfg.num_ranks), dim3(kEpWaves * kWarpSize), 0, s,
                     h->cfg, (const int32_t*)topk_idx, num_tokens,
                     (int32_t*)out_slot, (int32_t*)out_list, (int32_t*)out_sendc);
  // No sync: the plan is consumed by kernels on the same stream.
  return 0;
}

// The copy epilogue of a pending dispatch, for the sources `sel` picks (epilogue_skips): 0 all,
// 1 the other nodes', 2 this node's.
static int dispatch_copy_epilogue(EpHandle* h, const decltype(EpHandle::pend_d)& t, int sel, hipStream_t s) {
  const EpConfig& c = h->cfg;
  // A launch for one side's sources gets their share of the whole budget: the other blocks exit.
  const int active = sel == 0 ? c.num_ranks : sel == 2 ? h->node_size : c.num_ranks - h->node_size;
  const int epi_cta = std::max(1, copy_epilogue_cta(t.cta, c.num_ranks) * c.num_ranks / std::max(1, active));
  const int lo = h->node_base, hi = h->node_base + h->node_size;
  if (t.use_fp8) {
    hipLaunchKernelGGL(dispatch_copy_epilogue_impl, dim3(c.num_ranks, epi_cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, h->self_view, h->d_recvoff, (fp8_t*)t.out_x, (float*)t.out_sf,
                       (int32_t*)t.out_topk, (float*)t.out_tw, (int32_t*)t.out_src,
                       (int32_t*)t.expert_counts, sel, lo, hi);
  } else {
    hipLaunchKernelGGL(dispatch_copy_epilogue_impl, dim3(c.num_ranks, epi_cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, h->self_view, t.psum_incl ? (const int32_t*)t.psum_incl : h->d_recvoff,
                       (bf16_t*)t.out_x, (int32_t*)t.out_topk, (float*)t.out_tw, (int32_t*)t.out_src,
                       (int32_t*)t.expert_counts, t.psum_incl ? 1 : 0, sel, lo, hi);
  }
  HC(hipGetLastError());
  return 0;
}

// Second half of dispatch_enqueue, from h->pend_d: the remote leg, barrier B and the epilogue.
// Runs right after the local push, or later from ep_dispatch_finish.
static int dispatch_tail(EpHandle* h, hipStream_t s) {
  const EpConfig& c = h->cfg;
  auto& t = h->pend_d;
  t.on = false;
  note_stream(h, s);
  // Same stream, so the remote rows and counts are in this window before barrier B. A remote
  // source writes it only through this rank's own receive (flat) or a local forwarder's store
  // after barrier A (hybrid), so barrier A only has local writers to hold off.
  if (h->hybrid) {
    if (hybrid_dispatch_end(h, t.use_fp8, t.use_fp8 ? 1 : t.with_meta, t.hcta, s) != 0) return -1;

  } else if (h->inter && inter_dispatch(h, t.use_fp8, t.use_fp8 ? 1 : t.with_meta, s) != 0) {
    return -1;
  }
  // A streamed dispatch's grid has already copied every row out of the window, and every source's
  // count is in place (the copy waited for each source's words), so it needs no barrier B. Its copy
  // read the dispatch region since the last rendezvous, so the next dispatch keeps barrier A.
  if (t.streamed) {
    h->rdv_x = false;
    return 0;
  }
  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
  note_rdv(h);
  // The grouped path runs its own epilogue later, from the handle's offsets.
  if (!t.run_epilogue) {
    h->pending_x = true;
    return 0;
  }
  h->rdv_x = false;  // the scan and copy epilogue below read the dispatch region
  // A cached replay's offsets are its plan's: with psum_incl (the handle's inclusive
  // per-source prefix) there is nothing to scan.
  if (!t.notified && !t.psum_incl && !t.split_epi) {
    hipLaunchKernelGGL(k_ep_scan_counts, dim3(1), dim3(1), 0, s,
                       c, h->self_view, h->d_recvoff, h->d_total,
                       (int32_t*)t.expert_counts, c.experts_per_rank());
  }
  if (t.split_epi) return dispatch_copy_epilogue(h, t, 1, s);
  const int epi_cta = copy_epilogue_cta(t.cta, c.num_ranks);
  if (t.payload_later) {
    const int32_t* offs = t.psum_incl ? (const int32_t*)t.psum_incl : h->d_recvoff;
    hipLaunchKernelGGL(dispatch_meta_epilogue_impl, dim3(c.num_ranks, epi_cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, h->self_view, offs, (int32_t*)t.out_topk, (float*)t.out_tw, (int32_t*)t.out_src,
                       (int32_t*)t.expert_counts, t.psum_incl ? 1 : 0);
    h->pending_x = true;  // ep_permute_epilogue still reads the dispatch region
    h->perm.on = true;
    h->perm.offsets = offs;
    h->perm.inclusive = t.psum_incl ? 1 : 0;
    return 0;
  }
  return dispatch_copy_epilogue(h, t, 0, s);
}

// Enqueues a whole dispatch: both barriers, the push, the receive scan and the copy
// out of the window. Never synchronizes: the epilogue reads only device state and
// writes capacity-sized outputs, so the host need not know the receive total first.
// Returns 0, or -1 on a bad argument.
static int dispatch_enqueue(EpHandle* h,
                            uintptr_t x, uintptr_t x_sf,
                            uintptr_t topk_idx, uintptr_t topk_w, int num_tokens,
                            uintptr_t send_list, uintptr_t sendc, int use_fp8, int num_sms,
                            uintptr_t out_x, uintptr_t out_sf,
                            uintptr_t out_topk, uintptr_t out_tw, uintptr_t out_src,
                            uintptr_t expert_counts, int notified, hipStream_t s,
                            int with_meta = 1, uintptr_t rank = 0, int run_epilogue = 1,
                            uintptr_t psum_incl = 0, int zero_counts = 1) {
  const EpConfig& c = h->cfg;
  // Consumed here, so a call that fails below does not leave the next one deferred.
  const bool defer = h->defer_dispatch;
  h->defer_dispatch = false;
  const bool payload_later = h->payload_later;
  h->payload_later = false;
  if (h->perm.on) {
    fprintf(stderr, "[rccl_ep] dispatch: the previous dispatch's payload is still in the window (ep_permute_epilogue)\n");
    return -1;
  }
  if (payload_later && (use_fp8 || out_topk == 0 || !run_epilogue)) {
    fprintf(stderr, "[rccl_ep] ep_dispatch_payload_later needs a bf16 dispatch with metadata and its epilogue\n");
    return -1;
  }
  if (h->pend.on || h->pend_d.on) {
    fprintf(stderr, "[rccl_ep] dispatch: a deferred %s is still pending\n",
            h->pend.on ? "combine (ep_combine_finish)" : "dispatch (ep_dispatch_finish)");
    return -1;
  }
  // fp8 payload is 1 byte/element, so the 16B vector copies need hidden % 16 == 0;
  // ep_create's hidden % 8 guard only covers bf16 (2 bytes/element).
  if (use_fp8 && c.hidden % 16 != 0) {
    fprintf(stderr, "[rccl_ep] hidden %d must be a multiple of 16 for fp8 (16B vector alignment)\n", c.hidden);
    return -1;
  }
  // The fp8 paths read x_sf for every local token and write out_sf unconditionally, so a
  // missing scale tensor is a device-side fault rather than a skipped copy. A rank with no
  // tokens passes a null x_sf (an empty tensor's data pointer).
  if (use_fp8 && ((x_sf == 0 && num_tokens > 0) || out_sf == 0)) {
    fprintf(stderr, "[rccl_ep] fp8 dispatch needs both x_sf and out_sf\n");
    return -1;
  }

  // Slot index is rank * num_max_tokens_per_rank + i, with i bounded only by
  // num_tokens in k_ep_plan. Past capacity the push runs into the next rank's
  // region of the peer's window, and combine's src_tok = g % T then aliases
  // two tokens onto one slot.
  if (num_tokens > c.num_max_tokens_per_rank) {
    fprintf(stderr, "[rccl_ep] num_tokens %d exceeds num_max_tokens_per_rank %d\n",
            num_tokens, c.num_max_tokens_per_rank);
    return -1;
  }

  // 128 total CTAs is the measured bandwidth peak on gfx950; sweeping the CTA
  // budget shows dispatch flat from 32 upward on 8x MI350X.
  const int cta = ctas_per_rank(num_sms, 128, c.num_ranks);

  // Internode: no count handshake (plan_notify_enqueue refuses), and the remote leg below
  // needs this call's counts, bound by the caller.
  if (h->inter && notified) {
    fprintf(stderr, "[rccl_ep] internode dispatch takes the barrier path, not the notify handshake\n");
    return -1;
  }
  REQUIRE_BOUND(h, "dispatch");
  WindowView* views = h->inter ? h->d_views_inter + (use_fp8 ? kViewFp8 : kViewBf16) * c.num_ranks
                               : h->d_views;

  // After ep_plan_notify the count handshake has already done barrier A's job and
  // written d_recvoff and d_total. The counts memset is redundant, since dispatch_impl
  // rewrites every counts[src] (zero included); zero_counts=0 skips it.
  note_stream(h, s);
  // Streamed: every source's count and receive offset must be known before the push (the count
  // handshake's d_recvoff and d_total, or a cached handle's inclusive prefix), and the copy must be
  // the plain one into this call's outputs.
  const int push_cta = stream_dispatch_on() ? stream_push_cta(cta) : 0;
  const bool streamed = push_cta > 0 && !h->inter && !h->hybrid && run_epilogue && !payload_later &&
                        with_meta != 2 && expert_counts == 0 && (notified || psum_incl);
  // Hybrid: pack and the rail transfer touch only this rank's so_send / so_recv, not the window,
  // so they can start before barrier A and move while the node's ranks catch up.
  // RCCL_EP_HYB_EARLY_RAIL=0 starts them after barrier A instead.
  static const bool early_rail = [] {
    const char* e = getenv("RCCL_EP_HYB_EARLY_RAIL");
    return !(e && e[0] == '0');
  }();
  const int hyb_meta = use_fp8 ? 1 : with_meta;
  // ep_hcounts_post already queued this dispatch's pack and rail transfer.
  const bool posted = h->rail_posted.on;
  // ep_hcounts_post under RCCL_EP_GIN_STREAM also ran barrier A and posted the forwards.
  const bool fwd_posted = posted && h->rail_posted.fwd;
  h->rail_posted.on = h->rail_posted.fwd = false;
  if (posted && (!h->hybrid || use_fp8 || hyb_meta != 1 || x != h->rail_posted.x ||
                 num_tokens != h->rail_posted.num_tokens)) {
    fprintf(stderr, "[rccl_ep] dispatch: not the dispatch ep_hcounts_post posted the rail for\n");
    return -1;
  }
  if (h->hybrid && early_rail && !posted &&
      hybrid_dispatch_begin(h, x, x_sf, topk_idx, topk_w, num_tokens, use_fp8, hyb_meta,
                            hyb_ctas(h, num_sms), s) != 0)
    return -1;
  if (!notified && !fwd_posted) {
    // A streamed push writes every count itself, zero included.
    if (zero_counts && !streamed)
      HC(hipMemsetAsync((uint8_t*)h->win_buf + h->lay.off_counts, 0, c.num_ranks * sizeof(int32_t), s));
    // Barrier A keeps this push out of each peer's dispatch region until the peer has read
    // it. A rendezvous since this rank's own last reader of that region already guarantees
    // that (see EpHandle), unless a grouped epilogue is still to come.
    if (!(h->elide_a && !zero_counts && h->rdv_x && !h->pending_x)) {
      hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
      note_rdv(h);
    }
  }
  // Hybrid: the remote-node records go out on the side stream first and travel during the
  // local push, which sees only local destinations.
  const int32_t* push_sendc = (const int32_t*)sendc;
  if (h->hybrid) {
    if (!early_rail && !posted &&
        hybrid_dispatch_begin(h, x, x_sf, topk_idx, topk_w, num_tokens, use_fp8, hyb_meta,
                              hyb_ctas(h, num_sms), s) != 0)
      return -1;
    views = h->d_views_inter + kViewHybLocal * c.num_ranks;
    hipLaunchKernelGGL(k_hyb_local_sendc, dim3(1), dim3(kMaxWorldRanks), 0, s, c.num_ranks,
                       h->node_base, h->node_size, (const int32_t*)sendc, h->d_local_sendc);
    push_sendc = h->d_local_sendc;
    if (h->gin && h->gstream && !fwd_posted) {
      HC(hipEventRecord(h->ev_bar, s));
      if (gin_dispatch_forwards(h, use_fp8, hyb_meta, hyb_ctas(h, num_sms), h->bound_fwd_meta, nullptr) != 0)
        return -1;
    }
    // A GDA put rings its doorbell after a system-scope release, which must write back the L2
    // lines the local push keeps dirtying: posting beside the push took ~40 us a chunk instead of
    // ~20. So the push waits until every chunk is posted; it has slack, as the forwards and the
    // rail are the longer path.
    if (h->gin && h->gstream) HC(hipStreamWaitEvent(s, h->ev_post, 0));
  }
  // The fp8 copy epilogue takes exclusive offsets only, so an fp8 dispatch that does not stream scans.
  if (use_fp8 && !streamed) psum_incl = 0;
  if (streamed) {
    // Only the low 16 bits reach the progress words; 0 is the window's initial value.
    if ((++h->stream_seq & 0xffffu) == 0) ++h->stream_seq;
    const int chunk = stream_chunk_rounds();
    const int32_t* offs = psum_incl ? (const int32_t*)psum_incl : h->d_recvoff;
    const int incl = psum_incl ? 1 : 0;
    if (use_fp8) {
      hipLaunchKernelGGL(dispatch_stream_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                         c, (const fp8_t*)x, (const float*)x_sf, (const int32_t*)topk_idx,
                         (const float*)topk_w, num_tokens, (const int32_t*)send_list, push_sendc, views,
                         h->self_view, offs, incl, h->d_total, (fp8_t*)out_x, (float*)out_sf,
                         (int32_t*)out_topk, (float*)out_tw, (int32_t*)out_src, push_cta, chunk, h->stream_seq);
    } else {
      hipLaunchKernelGGL(dispatch_stream_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                         c, (const bf16_t*)x, (const int32_t*)topk_idx, (const float*)topk_w,
                         num_tokens, (const int32_t*)send_list, push_sendc, views, with_meta,
                         h->self_view, offs, incl, h->d_total, (bf16_t*)out_x, (int32_t*)out_topk,
                         (float*)out_tw, (int32_t*)out_src, push_cta, chunk, h->stream_seq);
    }
  } else if (use_fp8) {
    hipLaunchKernelGGL(dispatch_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, (const fp8_t*)x, (const float*)x_sf, (const int32_t*)topk_idx,
                       (const float*)topk_w, num_tokens, (const int32_t*)send_list,
                       push_sendc, views);
  } else {
    hipLaunchKernelGGL(dispatch_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, (const bf16_t*)x, (const int32_t*)topk_idx, (const float*)topk_w,
                       num_tokens, (const int32_t*)send_list, push_sendc, views,
                       with_meta, (const int32_t*)rank);
  }
  auto& t = h->pend_d;
  t.payload_later = payload_later ? 1 : 0;
  t.use_fp8 = use_fp8;
  t.with_meta = with_meta;
  t.cta = cta;
  t.hcta = h->hybrid ? hyb_ctas(h, num_sms) : 0;  // hyb_ctas divides by the node count
  t.run_epilogue = run_epilogue;
  t.notified = notified;
  t.streamed = streamed ? 1 : 0;
  t.psum_incl = psum_incl;
  t.expert_counts = expert_counts;
  t.out_x = out_x;
  t.out_sf = out_sf;
  t.out_topk = out_topk;
  t.out_tw = out_tw;
  t.out_src = out_src;
  // RCCL_EP_HYB_SPLIT_EPI=1: this node's sources' rows are complete once every local push is,
  // so copy them out now, while the rail moves, and leave only the remote sources' rows for after
  // barrier B. The remote sources' counts are the bound ones, so the scan can run now too.
  static const bool split_epi_env = env_int("RCCL_EP_HYB_SPLIT_EPI") == 1;
  t.split_epi = (split_epi_env && h->hybrid && run_epilogue && !payload_later && !notified) ? 1 : 0;
  if (t.split_epi) {
    hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
    note_rdv(h);
    hipLaunchKernelGGL(k_inter_set_counts, dim3(1), dim3(kMaxWorldRanks), 0, s, c.num_ranks, h->bound,
                       (int32_t*)((uint8_t*)h->win_buf + h->lay.off_counts));
    h->rdv_x = false;  // the scan and copy below read the dispatch region
    if (!psum_incl)
      hipLaunchKernelGGL(k_ep_scan_counts, dim3(1), dim3(1), 0, s, c, h->self_view, h->d_recvoff, h->d_total,
                         (int32_t*)expert_counts, c.experts_per_rank());
    if (dispatch_copy_epilogue(h, t, 2, s) != 0) return -1;
  }
  t.on = true;
  return defer ? 0 : dispatch_tail(h, s);
}

// Returns the number of received tokens, or -1 on error. Synchronizes once, after
// the epilogue, to read back the receive total.
int ep_dispatch(void* handle,
                uintptr_t x, uintptr_t x_sf,
                uintptr_t topk_idx, uintptr_t topk_w, int num_tokens,
                uintptr_t send_list, uintptr_t sendc, int use_fp8, int num_sms,
                uintptr_t out_x, uintptr_t out_sf,
                uintptr_t out_topk, uintptr_t out_tw, uintptr_t out_src,
                uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  hipStream_t s = (hipStream_t)stream;
  REQUIRE_CONFIGURED(h, "ep_dispatch");
  if (dispatch_enqueue(h, x, x_sf, topk_idx, topk_w, num_tokens, send_list, sendc,
                       use_fp8, num_sms, out_x, out_sf, out_topk, out_tw, out_src, 0, 0, s) != 0)
    return -1;
  int total = 0;
  HC(hipMemcpyAsync(&total, h->d_total, sizeof(int32_t), hipMemcpyDeviceToHost, s));
  HC(hipStreamSynchronize(s));
  return total;
}

// Dispatch without host synchronization. Returns 0, or -1 on error.
// `out_counts` (optional) gets the received (token, slot) pairs per local expert,
// counted in the copy epilogue. `host_out` (pinned int32[1 + experts_per_rank], needs
// out_counts) gets async copies of the total and those counts, so one event or stream
// wait covers the host readback. A cached replay needs neither.
int ep_dispatch_v2(void* handle,
                   uintptr_t x, uintptr_t x_sf,
                   uintptr_t topk_idx, uintptr_t topk_w, int num_tokens,
                   uintptr_t send_list, uintptr_t sendc, int use_fp8, int num_sms,
                   uintptr_t out_x, uintptr_t out_sf,
                   uintptr_t out_topk, uintptr_t out_tw, uintptr_t out_src,
                   uintptr_t out_counts, uintptr_t host_out, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  hipStream_t s = (hipStream_t)stream;
  REQUIRE_CONFIGURED(h, "ep_dispatch_v2");
  if (host_out != 0 && out_counts == 0) {
    fprintf(stderr, "[rccl_ep] ep_dispatch_v2: host_out needs out_counts\n");
    return -1;
  }
  if (dispatch_enqueue(h, x, x_sf, topk_idx, topk_w, num_tokens, send_list, sendc,
                       use_fp8, num_sms, out_x, out_sf, out_topk, out_tw, out_src,
                       out_counts, 0, s) != 0)
    return -1;
  const int epr = h->cfg.experts_per_rank();
  if (host_out != 0) {
    int32_t* ho = (int32_t*)host_out;
    HC(hipMemcpyAsync(ho, h->d_total, sizeof(int32_t), hipMemcpyDeviceToHost, s));
    HC(hipMemcpyAsync(ho + 1, (const void*)out_counts, epr * sizeof(int32_t),
                      hipMemcpyDeviceToHost, s));
  }
  return 0;
}

// ep_dispatch_v2 for a cached replay, with its handle's inclusive per-source receive prefix
// (psum_rank): the offsets need no scan after the push, so the copy can stream with it
// (RCCL_EP_STREAM_DISPATCH).
int ep_dispatch_v4(void* handle,
                   uintptr_t x, uintptr_t x_sf,
                   uintptr_t topk_idx, uintptr_t topk_w, int num_tokens,
                   uintptr_t send_list, uintptr_t sendc, int use_fp8, int num_sms,
                   uintptr_t out_x, uintptr_t out_sf,
                   uintptr_t out_topk, uintptr_t out_tw, uintptr_t out_src,
                   uintptr_t psum_rank, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_dispatch_v4");
  if (psum_rank == 0) {
    fprintf(stderr, "[rccl_ep] ep_dispatch_v4 needs the handle's psum_rank\n");
    return -1;
  }
  return dispatch_enqueue(h, x, x_sf, topk_idx, topk_w, num_tokens, send_list, sendc,
                          use_fp8, num_sms, out_x, out_sf, out_topk, out_tw, out_src,
                          0, 0, (hipStream_t)stream, /*with_meta=*/1, /*rank=*/0,
                          /*run_epilogue=*/1, psum_rank);
}

// Payload-only cached replay: moves and compacts the rows without the per-token
// top-k ids, weights and source indices, for a caller that already holds the
// handle's metadata (e.g. a combine backward). bf16 only.
int ep_dispatch_payload(void* handle, uintptr_t x, int num_tokens, uintptr_t send_list,
                        uintptr_t sendc, int num_sms, uintptr_t out_x, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_dispatch_payload");
  return dispatch_enqueue(h, x, 0, 0, 0, num_tokens, send_list, sendc, 0, num_sms, out_x, 0,
                          0, 0, 0, 0, 0, (hipStream_t)stream, /*with_meta=*/0);
}

// ep_dispatch_payload with the plan's own offsets (psum_rank: the handle's inclusive
// per-source prefix): no counts memset and no scan, so barrier A, the push, barrier B and
// the copy are the whole replay.
int ep_dispatch_payload_v2(void* handle, uintptr_t x, int num_tokens, uintptr_t send_list,
                           uintptr_t sendc, uintptr_t psum_rank, int num_sms, uintptr_t out_x,
                           uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_dispatch_payload");
  if (psum_rank == 0) {
    fprintf(stderr, "[rccl_ep] ep_dispatch_payload_v2 needs the handle's psum_rank\n");
    return -1;
  }
  return dispatch_enqueue(h, x, 0, 0, 0, num_tokens, send_list, sendc, 0, num_sms, out_x, 0,
                          0, 0, 0, 0, 0, (hipStream_t)stream, /*with_meta=*/0, /*rank=*/0,
                          /*run_epilogue=*/1, psum_rank, /*zero_counts=*/0);
}

// Plans an uncached dispatch and exchanges its counts in place of barrier A
// (k_ep_plan_notify), writing the receive offsets, `out_expert_counts`, `out_psum_rank`
// and the host-mapped counts ep_wait_counts polls. Follow with ep_dispatch_v3. No sync.
//
// Every EP rank must take this path for the same calls: it launches one LSA barrier
// fewer than the barrier path, so a mismatch hangs rather than corrupts.
// ep_plan_notify_v2 adds the grouped layout's optional outputs, all or none: out_rank
// [num_tokens, num_topk], out_srcpref [R, epr] and out_ebase [epr].
static int plan_notify_enqueue(EpHandle* h, uintptr_t topk_idx, int topk_i64, uintptr_t out_topk_i32,
                               int num_tokens, uintptr_t out_slot, uintptr_t out_list,
                               uintptr_t out_sendc, uintptr_t out_expert_counts,
                               uintptr_t out_psum_rank, uintptr_t out_rank,
                               uintptr_t out_srcpref, uintptr_t out_ebase, uintptr_t stream);

int ep_plan_notify_v2(void* handle, uintptr_t topk_idx, int num_tokens,
                      uintptr_t out_slot, uintptr_t out_list, uintptr_t out_sendc,
                      uintptr_t out_expert_counts, uintptr_t out_psum_rank,
                      uintptr_t out_rank, uintptr_t out_srcpref, uintptr_t out_ebase,
                      uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_plan_notify");
  return plan_notify_enqueue(h, topk_idx, 0, 0, num_tokens, out_slot, out_list, out_sendc,
                             out_expert_counts, out_psum_rank, out_rank, out_srcpref, out_ebase,
                             stream);
}

// ep_plan_notify_v2 for int64 top-k ids; the plan kernel also writes the int32 copy
// the dispatch and combine entry points read into `out_topk_i32`.
int ep_plan_notify_v3(void* handle, uintptr_t topk_idx_i64, uintptr_t out_topk_i32, int num_tokens,
                      uintptr_t out_slot, uintptr_t out_list, uintptr_t out_sendc,
                      uintptr_t out_expert_counts, uintptr_t out_psum_rank,
                      uintptr_t out_rank, uintptr_t out_srcpref, uintptr_t out_ebase,
                      uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_plan_notify");
  // A rank with no tokens has nothing to cast, and an empty tensor's data pointer is null.
  if (out_topk_i32 == 0 && num_tokens > 0) {
    fprintf(stderr, "[rccl_ep] ep_plan_notify_v3 needs out_topk_i32\n");
    return -1;
  }
  return plan_notify_enqueue(h, topk_idx_i64, 1, out_topk_i32, num_tokens, out_slot, out_list,
                             out_sendc, out_expert_counts, out_psum_rank, out_rank, out_srcpref,
                             out_ebase, stream);
}

static int plan_notify_enqueue(EpHandle* h, uintptr_t topk_idx, int topk_i64, uintptr_t out_topk_i32,
                               int num_tokens, uintptr_t out_slot, uintptr_t out_list,
                               uintptr_t out_sendc, uintptr_t out_expert_counts,
                               uintptr_t out_psum_rank, uintptr_t out_rank,
                               uintptr_t out_srcpref, uintptr_t out_ebase, uintptr_t stream) {
  // The handshake stores into every peer's window and waits on its own, inside one kernel;
  // a remote peer's store would land in the outbox and the wait would never end.
  if (h->inter) {
    fprintf(stderr, "[rccl_ep] ep_plan_notify is intranode only; plan with ep_plan and ep_inter_counts\n");
    return -1;
  }
  // Its reduce rewrites the receive offsets a pending ep_permute_epilogue reads.
  if (h->perm.on) {
    fprintf(stderr, "[rccl_ep] ep_plan_notify: the previous dispatch's payload is still in the window (ep_permute_epilogue)\n");
    return -1;
  }
  if ((out_rank != 0) != (out_srcpref != 0) || (out_rank != 0) != (out_ebase != 0)) {
    fprintf(stderr, "[rccl_ep] ep_plan_notify_v2: out_rank, out_srcpref and out_ebase go together\n");
    return -1;
  }
  // The plan scratch holds one row per kPlanChunk tokens of capacity.
  if (num_tokens > h->cfg.num_max_tokens_per_rank) {
    fprintf(stderr, "[rccl_ep] num_tokens %d exceeds num_max_tokens_per_rank %d\n",
            num_tokens, h->cfg.num_max_tokens_per_rank);
    return -1;
  }
  hipStream_t s = (hipStream_t)stream;
  note_stream(h, s);
  // The previous uncached call waited for its counts before returning, so nothing
  // on the device is still writing this buffer.
  __atomic_store_n(&h->host_counts[0], -1, __ATOMIC_RELEASE);
  const uint32_t seq = ++h->notify_seq;
  const int chunks = num_tokens > 0 ? (num_tokens + kPlanChunk - 1) / kPlanChunk : 1;
  const NotifyReduceOut nro{h->d_recvoff, h->d_total, (int32_t*)out_expert_counts,
                            (int32_t*)out_psum_rank, h->host_counts_dev, (int32_t*)out_srcpref,
                            (int32_t*)out_ebase};
  if (topk_i64)
    hipLaunchKernelGGL(k_ep_plan_notify<int64_t>, dim3(h->cfg.num_ranks, chunks), dim3(kPlanChunk), 0, s,
                       h->cfg, (const int64_t*)topk_idx, (int32_t*)out_topk_i32, num_tokens,
                       (int32_t*)out_slot, (int32_t*)out_list, (int32_t*)out_sendc,
                       h->d_views, h->self_view, seq, (int32_t*)out_rank, h->d_plan_agg, nro);
  else
    hipLaunchKernelGGL(k_ep_plan_notify<int32_t>, dim3(h->cfg.num_ranks, chunks), dim3(kPlanChunk), 0, s,
                       h->cfg, (const int32_t*)topk_idx, (int32_t*)nullptr, num_tokens,
                       (int32_t*)out_slot, (int32_t*)out_list, (int32_t*)out_sendc,
                       h->d_views, h->self_view, seq, (int32_t*)out_rank, h->d_plan_agg, nro);
  // Its last CTA waits for every source's publication, which each source makes only after
  // everything before it on its stream: a rendezvous.
  note_rdv(h);
  return 0;
}

int ep_plan_notify(void* handle, uintptr_t topk_idx, int num_tokens,
                   uintptr_t out_slot, uintptr_t out_list, uintptr_t out_sendc,
                   uintptr_t out_expert_counts, uintptr_t out_psum_rank, uintptr_t stream) {
  return ep_plan_notify_v2(handle, topk_idx, num_tokens, out_slot, out_list, out_sendc,
                           out_expert_counts, out_psum_rank, 0, 0, 0, stream);
}

// Grouped dispatch, send half: dispatch_impl with each owned top-k slot sent as
// rank << 8 | local expert, and barrier B -- no epilogue; k_ep_grouped_epilogue compacts
// once the caller knows the row count. notified=1 after ep_plan_notify_v2 (the handshake
// did barrier A's job), 0 for a cached replay. bf16 only.
int ep_dispatch_grouped(void* handle, uintptr_t x, uintptr_t topk_idx, uintptr_t topk_w,
                        int num_tokens, uintptr_t send_list, uintptr_t sendc, uintptr_t rank,
                        int num_sms, int notified, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_dispatch_grouped");
  // The Python layer refuses it too; this keeps a direct caller from a wrong result.
  if (h->inter) {
    fprintf(stderr, "[rccl_ep] ep_dispatch_grouped is intranode only\n");
    return -1;
  }
  // The rank rides above 8 bits of local expert id in a signed 32-bit word.
  if (h->cfg.experts_per_rank() > 256 || h->cfg.num_max_tokens_per_rank >= (1 << 23)) {
    fprintf(stderr, "[rccl_ep] grouped dispatch needs <= 256 local experts and < 2^23 tokens per rank\n");
    return -1;
  }
  return dispatch_enqueue(h, x, 0, topk_idx, topk_w, num_tokens, send_list, sendc, 0, num_sms,
                          0, 0, 0, 0, 0, 0, notified, (hipStream_t)stream, /*with_meta=*/2, rank,
                          /*run_epilogue=*/0, /*psum_incl=*/0, /*zero_counts=*/0);
}

// Grouped dispatch, receive half: see k_ep_grouped_epilogue. psum_rank, srcpref and ebase
// are the plan's own (the handle's), so a cached replay can run it too.
int ep_grouped_epilogue(void* handle, uintptr_t psum_rank, uintptr_t srcpref, uintptr_t ebase,
                        int num_recv, uintptr_t out_rows, uintptr_t out_row_ids,
                        uintptr_t out_row_w, uintptr_t out_map, uintptr_t out_ids32,
                        uintptr_t out_ids64, uintptr_t out_w, uintptr_t out_src, int num_sms,
                        uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_grouped_epilogue");
  const EpConfig& c = h->cfg;
  const int cta = ctas_per_rank(num_sms, 128, c.num_ranks);
  note_stream(h, (hipStream_t)stream);
  h->rdv_x = false;
  h->pending_x = false;
  hipLaunchKernelGGL(k_ep_grouped_epilogue, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0,
                     (hipStream_t)stream, c, h->self_view, (const int32_t*)psum_rank,
                     (const int32_t*)srcpref, (const int32_t*)ebase, num_recv,
                     (bf16_t*)out_rows, (int64_t*)out_row_ids, (float*)out_row_w,
                     (int64_t*)out_map, (int32_t*)out_ids32, (int64_t*)out_ids64,
                     (float*)out_w, (int32_t*)out_src);
  return 0;
}

// The dispatch half of the handshake path: dispatch_impl, barrier B and the copy
// epilogue, with offsets from the plan kernel's reduce. Returns 0, or -1 on error; no sync.
int ep_dispatch_v3(void* handle,
                   uintptr_t x, uintptr_t x_sf,
                   uintptr_t topk_idx, uintptr_t topk_w, int num_tokens,
                   uintptr_t send_list, uintptr_t sendc, int use_fp8, int num_sms,
                   uintptr_t out_x, uintptr_t out_sf,
                   uintptr_t out_topk, uintptr_t out_tw, uintptr_t out_src,
                   uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  hipStream_t s = (hipStream_t)stream;
  REQUIRE_CONFIGURED(h, "ep_dispatch_v3");
  return dispatch_enqueue(h, x, x_sf, topk_idx, topk_w, num_tokens, send_list, sendc,
                          use_fp8, num_sms, out_x, out_sf, out_topk, out_tw, out_src,
                          0, 1, s);
}

// Waits on the host for the counts ep_plan_notify's reduce publishes and copies
// [total, counts per local expert] into `out`. Returns 0, or -1 on a timeout, which
// means some EP rank never reached its plan for this dispatch or took another path.
int ep_wait_counts(void* handle, int32_t* out, int timeout_ms) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_wait_counts");
  const int epr = h->cfg.experts_per_rank();
  const auto t0 = std::chrono::steady_clock::now();
  for (uint64_t spins = 0; __atomic_load_n(&h->host_counts[0], __ATOMIC_ACQUIRE) < 0; ++spins) {
    if ((spins & 1023) == 0) {
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0).count();
      if (timeout_ms > 0 && ms > timeout_ms) {
        fprintf(stderr, "[rccl_ep] rank %d: ep_wait_counts timed out after %d ms at notify seq %u\n",
                h->cfg.rank, timeout_ms, h->notify_seq);
        return -1;
      }
    }
  }
  for (int i = 0; i < 1 + epr; ++i) out[i] = __atomic_load_n(&h->host_counts[i], __ATOMIC_ACQUIRE);
  return 0;
}

// ep_wait_counts, returning the first n of [total, counts[epr], recv_pairs[R], max_tokens]
// (see notify_reduce_wave), n <= 2 + epr + R.
int ep_wait_counts_v2(void* handle, int32_t* out, int n, int timeout_ms) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_wait_counts");
  const int full = 2 + h->cfg.experts_per_rank() + h->cfg.num_ranks;
  if (n < 1 || n > full) {
    fprintf(stderr, "[rccl_ep] ep_wait_counts_v2: n %d must be in [1, %d]\n", n, full);
    return -1;
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (uint64_t spins = 0; __atomic_load_n(&h->host_counts[0], __ATOMIC_ACQUIRE) < 0; ++spins) {
    if ((spins & 1023) == 0) {
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0).count();
      if (timeout_ms > 0 && ms > timeout_ms) {
        fprintf(stderr, "[rccl_ep] rank %d: ep_wait_counts timed out after %d ms at notify seq %u\n",
                h->cfg.rank, timeout_ms, h->notify_seq);
        return -1;
      }
    }
  }
  for (int i = 0; i < n; ++i) out[i] = __atomic_load_n(&h->host_counts[i], __ATOMIC_ACQUIRE);
  return 0;
}

// Per-source received counts, so the caller can report the per-rank prefix sum.
int ep_recv_counts(void* handle, uintptr_t out_counts, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_recv_counts");
  hipStream_t s = (hipStream_t)stream;
  note_stream(h, s);
  h->rdv_x = false;
  // No sync: the destination is a caller tensor read back on the same stream.
  HC(hipMemcpyAsync((void*)out_counts, (uint8_t*)h->win_buf + h->lay.off_counts,
                    h->cfg.num_ranks * sizeof(int32_t), hipMemcpyDeviceToDevice, s));
  return 0;
}

int ep_expert_counts(void* handle, uintptr_t recv_topk, int num_recv, uintptr_t out_counts,
                     uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_expert_counts");
  hipStream_t s = (hipStream_t)stream;
  hipLaunchKernelGGL(k_expand_count, dim3(h->cfg.experts_per_rank()), dim3(256), 0, s,
                     h->cfg, (const int32_t*)recv_topk, num_recv, (int32_t*)out_counts);
  // No sync: a caller that needs the counts on the host reads them back itself.
  return 0;
}

// Build the expanded-layout row map. Returns total rows (including padding).
int ep_expand_build(void* handle, uintptr_t recv_topk, int num_recv, int expert_alignment,
                    uintptr_t counts, uintptr_t offsets, uintptr_t psum,
                    uintptr_t hist, uintptr_t row_map, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_expand_build");
  const EpConfig& c = h->cfg;
  hipStream_t s = (hipStream_t)stream;
  const int epr = c.experts_per_rank();
  const int blocks = num_recv > 0 ? (num_recv + 255) / 256 : 1;

  hipLaunchKernelGGL(k_expand_count, dim3(epr), dim3(256), 0, s,
                     c, (const int32_t*)recv_topk, num_recv, (int32_t*)counts);
  hipLaunchKernelGGL(k_expand_scan, dim3(1), dim3(1), 0, s,
                     c, expert_alignment, (const int32_t*)counts,
                     (int32_t*)offsets, (int32_t*)psum, h->d_total);
  hipLaunchKernelGGL(k_expand_hist, dim3(blocks), dim3(256), 0, s,
                     c, (const int32_t*)recv_topk, num_recv, (int32_t*)hist);
  hipLaunchKernelGGL(k_expand_base, dim3(epr), dim3(kWarpSize), 0, s,
                     c, num_recv, (int32_t*)hist);
  hipLaunchKernelGGL(k_expand_map, dim3(blocks), dim3(256), 0, s,
                     c, (const int32_t*)recv_topk, num_recv,
                     (const int32_t*)offsets, (const int32_t*)hist, (int32_t*)row_map);

  // Row count reaches the host to size the expanded tensors, so this sync is
  // structural. The copy is already ordered behind k_expand_scan on `s`.
  int total = 0;
  HC(hipMemcpyAsync(&total, h->d_total, sizeof(int32_t), hipMemcpyDeviceToHost, s));
  HC(hipStreamSynchronize(s));
  return total;
}

int ep_expand_scatter(void* handle, int use_fp8,
                      uintptr_t src_x, uintptr_t src_sf, uintptr_t src_w,
                      uintptr_t row_map, int num_recv,
                      uintptr_t out_x, uintptr_t out_sf, int sf_row_stride, int sf_col_stride,
                      uintptr_t out_w,
                      int zero_pad, int expert_alignment,
                      uintptr_t counts, uintptr_t offsets, int num_sms, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_expand_scatter");
  const EpConfig& c = h->cfg;
  hipStream_t s = (hipStream_t)stream;
  const int cta = sms_or(num_sms, 64);

  if (use_fp8) {
    hipLaunchKernelGGL(k_expand_scatter_fp8, dim3(cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, (const fp8_t*)src_x, (const float*)src_sf, (const float*)src_w,
                       (const int32_t*)row_map, num_recv, (fp8_t*)out_x, (float*)out_sf,
                       sf_row_stride, sf_col_stride, (float*)out_w);
  } else {
    hipLaunchKernelGGL(k_expand_scatter, dim3(cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, (const bf16_t*)src_x, (const float*)src_w,
                       (const int32_t*)row_map, num_recv, (bf16_t*)out_x, (float*)out_w);
  }
  if (zero_pad) {
    hipLaunchKernelGGL(k_expand_zero_pad, dim3(c.experts_per_rank()), dim3(kEpWaves * kWarpSize), 0, s,
                       c, expert_alignment, (const int32_t*)counts, (const int32_t*)offsets,
                       use_fp8 ? 1 : (int)sizeof(bf16_t), (void*)out_x,
                       (float*)out_sf, sf_row_stride, sf_col_stride, (float*)out_w);
  }
  HC(hipStreamSynchronize(s));
  return 0;
}

// The next dispatch on this handle (bf16, with metadata) writes only the metadata in its
// epilogue and leaves its rows in the window; ep_permute_epilogue moves them, and no other
// dispatch may come between.
int ep_dispatch_payload_later(void* handle) {
  static_cast<EpHandle*>(handle)->payload_later = true;
  return 0;
}

// The rows of the last ep_dispatch_payload_later dispatch, straight from the window into the
// caller's expert-major rows: see dispatch_permute_epilogue_impl. token_ids is int64
// [groups, n], n that dispatch's received tokens. Row r of the result is the bf16 payload at
// out_x + r * x_stride, the fp32 top-k weights at out_w + r * w_stride and the local top-k ids
// (ids_bytes 4 or 8) at out_ids + r * ids_stride; strides are in bytes.
int ep_permute_epilogue(void* handle, uintptr_t token_ids, int groups, int n, uintptr_t out_x,
                        int64_t x_stride, uintptr_t out_w, int64_t w_stride, uintptr_t out_ids,
                        int64_t ids_stride, int ids_bytes, int num_sms, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_permute_epilogue");
  const EpConfig& c = h->cfg;
  if (!h->perm.on) {
    fprintf(stderr, "[rccl_ep] ep_permute_epilogue: no dispatch left its payload in the window\n");
    return -1;
  }
  const int64_t payload = (int64_t)c.hidden * (int64_t)sizeof(bf16_t);
  // The payload goes in 16-byte vectors, the weights and ids in their own width.
  if ((ids_bytes != 4 && ids_bytes != 8) || groups < 0 || n < 0 || x_stride < payload ||
      x_stride % 16 != 0 || out_x % 16 != 0 || w_stride < 4 * c.num_topk || w_stride % 4 != 0 ||
      out_w % 4 != 0 || ids_stride < (int64_t)ids_bytes * c.num_topk || ids_stride % ids_bytes != 0 ||
      out_ids % ids_bytes != 0) {
    fprintf(stderr, "[rccl_ep] ep_permute_epilogue: bad output layout (strides %lld / %lld / %lld, ids_bytes %d)\n",
            (long long)x_stride, (long long)w_stride, (long long)ids_stride, ids_bytes);
    return -1;
  }
  hipStream_t s = (hipStream_t)stream;
  note_stream(h, s);
  h->rdv_x = false;
  h->pending_x = false;
  h->perm.on = false;
  const int cta = copy_epilogue_cta(ctas_per_rank(num_sms, 128, c.num_ranks), c.num_ranks);
  hipLaunchKernelGGL(dispatch_permute_epilogue_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                     c, h->self_view, h->perm.offsets, h->perm.inclusive, (const int64_t*)token_ids, groups, n,
                     (uint8_t*)out_x, x_stride, (uint8_t*)out_w, w_stride, (uint8_t*)out_ids, ids_stride,
                     ids_bytes);
  return 0;
}

int ep_combine_v5(void* handle,
                  uintptr_t y, uintptr_t in_w, uintptr_t row_map, uintptr_t expert_rows,
                  uintptr_t scales, int fma_mode,
                  uintptr_t recv_topk, uintptr_t recv_src, int num_recv,
                  uintptr_t topk_idx, int num_tokens,
                  uintptr_t bias0, uintptr_t bias1, int grouped,
                  uintptr_t out, uintptr_t out_w, int num_sms, int defer, uintptr_t slot_of,
                  uintptr_t stream);

// `grouped` selects what each rank writes into the window: 1 for one reduced
// row per received token, with this rank's owned rows summed here in ascending
// k; 0 for one row per (slot, k). In either mode a non-null `row_map` means the
// caller's input is the expanded layout and is addressed through it.
//
// ep_combine_v2 adds `expert_rows` (int64 [experts_per_rank, num_recv], grouped only,
// row_map null): the input is the caller's own grouped-by-expert layout of a
// non-expanded dispatch, reduced per token in ascending expert order, and in_w is that
// layout's [rows, num_topk] weights. See combine_impl.
//
// ep_combine_v3 adds `scales` (fp32 [num_recv, num_topk], expert_rows only): each row is
// scaled by its token's weight for that row's expert as it is summed. fma_mode 1
// contracts each multiply-add into one fma; 0 rounds the product before the add.
//
// ep_combine_v4 adds `defer` (hybrid only): return once the rail transfer is queued; the output
// is complete only after ep_combine_finish on the caller's stream, and no other call on this
// handle may come between.
int ep_combine_v4(void* handle,
                  uintptr_t y, uintptr_t in_w, uintptr_t row_map, uintptr_t expert_rows,
                  uintptr_t scales, int fma_mode,
                  uintptr_t recv_topk, uintptr_t recv_src, int num_recv,
                  uintptr_t topk_idx, int num_tokens,
                  uintptr_t bias0, uintptr_t bias1, int grouped,
                  uintptr_t out, uintptr_t out_w, int num_sms, int defer, uintptr_t stream) {
  return ep_combine_v5(handle, y, in_w, row_map, expert_rows, scales, fma_mode, recv_topk, recv_src,
                       num_recv, topk_idx, num_tokens, bias0, bias1, grouped, out, out_w, num_sms, defer, 0,
                       stream);
}

// ep_combine_v4 plus slot_of, the dispatch plan's [num_ranks, num_tokens] position of each token in
// its send order to each rank (-1 = not sent), which lets the reduce run while the rows are still
// arriving (RCCL_EP_STREAM_COMBINE). 0 keeps the push, barrier and reduce apart. Every rank must
// pass a non-null slot_of or none does: a rank with no tokens passes any non-null pointer.
int ep_combine_v5(void* handle,
                  uintptr_t y, uintptr_t in_w, uintptr_t row_map, uintptr_t expert_rows,
                  uintptr_t scales, int fma_mode,
                  uintptr_t recv_topk, uintptr_t recv_src, int num_recv,
                  uintptr_t topk_idx, int num_tokens,
                  uintptr_t bias0, uintptr_t bias1, int grouped,
                  uintptr_t out, uintptr_t out_w, int num_sms, int defer, uintptr_t slot_of,
                  uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  const EpConfig& c = h->cfg;
  hipStream_t s = (hipStream_t)stream;
  REQUIRE_CONFIGURED(h, "ep_combine");
  if (h->pend.on || h->pend_d.on || (defer && !(h->hybrid && h->hyb_multi))) {
    fprintf(stderr, "[rccl_ep] ep_combine: %s\n",
            h->pend.on || h->pend_d.on ? "a deferred combine or dispatch is still pending"
                                       : "defer needs a hybrid handle with multiple reduction");
    return -1;
  }
  if (expert_rows && (!grouped || row_map)) {
    fprintf(stderr, "[rccl_ep] expert_rows needs grouped=1 and no row_map\n");
    return -1;
  }
  if (scales && (!expert_rows || in_w)) {
    fprintf(stderr, "[rccl_ep] scales need expert_rows and no in_w\n");
    return -1;
  }
  // Same capacity bound ep_dispatch enforces: combine writes slot
  // rank * num_max_tokens_per_rank + t, and src_tok = g % T aliases two tokens
  // onto one slot once t runs past capacity.
  if (num_tokens > c.num_max_tokens_per_rank) {
    fprintf(stderr, "[rccl_ep] num_tokens %d exceeds num_max_tokens_per_rank %d\n",
            num_tokens, c.num_max_tokens_per_rank);
    return -1;
  }
  // Two different grids. The transport is (num_ranks, per_rank) so a CTA owns
  // one peer; the reduce epilogue is purely local and 1-D over tokens, so it
  // takes the whole budget. Sharing one value silently starved the epilogue of
  // CTAs and cost it 7x.
  const int cta = ctas_per_rank(num_sms, 128, c.num_ranks);
  const int cta_reduce = sms_or(num_sms, 128);

  // Barrier A keeps this push out of each peer's combine region until the peer's last reduce
  // epilogue has read it; a rendezvous since this rank's own last reduce epilogue already
  // guarantees that (see EpHandle).
  REQUIRE_BOUND(h, "ep_combine");
  if (h->grouped_only && !grouped) {
    fprintf(stderr, "[rccl_ep] this window was configured grouped_only; a non-grouped combine does not fit it\n");
    return -1;
  }
  if (h->hybrid && h->hyb_multi) {
    if (!grouped) {
      fprintf(stderr, "[rccl_ep] the hybrid combine is grouped only\n");
      return -1;
    }
    return hybrid_combine(h, y, in_w, row_map, expert_rows, scales, fma_mode, recv_topk, recv_src,
                          num_recv, topk_idx, num_tokens, bias0, bias1, out, out_w, cta, cta_reduce,
                          hyb_ctas(h, num_sms), defer, s);
  }
  note_stream(h, s);
  if (!(h->elide_a && h->rdv_y)) {
    hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
    note_rdv(h);
  }
  h->rdv_x = false;  // combine_impl reads this rank's received counts
  WindowView* views = h->d_views;
  const int32_t* src = (const int32_t*)recv_src;
  if (h->inter) {
    views = h->d_views_inter + (grouped ? kViewCombGrouped : kViewCombFlat) * c.num_ranks;
    hipLaunchKernelGGL(k_inter_remap_src, dim3(num_recv > 0 ? (num_recv + 255) / 256 : 1), dim3(256), 0, s,
                       (const int32_t*)recv_src, num_recv, c.num_max_tokens_per_rank, h->bound,
                       h->d_src_inter);
    src = h->d_src_inter;
  }
  // Streamed: the reduce follows the rows as they land, in the same grid, and needs no barrier
  // after it. The next combine's barrier A is not elided (rdv_y below), so no rank pushes into a
  // combine region its owner is still reducing.
  // The reduce needs at least as many CTAs as the push: at 4 CTAs per peer it would get one, and the
  // streamed combine measured 15% slower than the plain one.
  int cpush = slot_of && !h->inter && stream_combine_on() ? stream_push_cta(cta) : 0;
  if (cta - cpush < cpush) cpush = 0;
  if (cpush) {
    if ((++h->cstream_seq & 0xffffu) == 0) ++h->cstream_seq;
    hipLaunchKernelGGL(combine_stream_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, (const bf16_t*)y, (const float*)in_w, (const int32_t*)row_map,
                       (const int64_t*)expert_rows, (const float*)scales, fma_mode,
                       (const int32_t*)recv_topk, src, num_recv, grouped, h->self_view, views,
                       (const int32_t*)topk_idx, num_tokens, (const int32_t*)slot_of,
                       (const bf16_t*)bias0, (const bf16_t*)bias1, (bf16_t*)out, (float*)out_w, cpush,
                       stream_chunk_rounds(), h->cstream_seq);
    h->rdv_y = false;
    return 0;
  }
  hipLaunchKernelGGL(combine_impl<0>, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                     c, (const bf16_t*)y, (const float*)in_w, (const int32_t*)row_map,
                     (const int64_t*)expert_rows, (const float*)scales, fma_mode,
                     (const int32_t*)recv_topk, src, num_recv,
                     grouped, h->self_view, views, 0, 0, 0, INT_MAX);
  // As in dispatch: the remote rows are in place before barrier B, on the same stream.
  if (h->inter && inter_combine(h, grouped, in_w != 0, cta, s) != 0) return -1;
  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
  note_rdv(h);
  h->rdv_y = false;  // the reduce epilogue reads the combine region
  hipLaunchKernelGGL(combine_reduce_epilogue_impl, dim3(cta_reduce), dim3(kEpWaves * kWarpSize), 0, s,
                     c, h->self_view, (const int32_t*)topk_idx, num_tokens,
                     (const bf16_t*)bias0, (const bf16_t*)bias1, grouped,
                     (bf16_t*)out, (float*)out_w);
  // No sync: nothing is read back, and the caller's temporaries are released to
  // the caching allocator on this same stream, which orders their reuse.
  return 0;
}

// The next dispatch on this (hybrid) handle returns once its local push is queued, with the
// rail transfer in flight; ep_dispatch_finish on the caller's stream completes it, and no other
// call on this handle may come between.
int ep_dispatch_defer_next(void* handle) {
  auto* h = static_cast<EpHandle*>(handle);
  if (!h->hybrid) {
    fprintf(stderr, "[rccl_ep] ep_dispatch_defer_next needs a hybrid handle\n");
    return -1;
  }
  h->defer_dispatch = true;
  return 0;
}

int ep_dispatch_finish(void* handle, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  if (!h->pend_d.on) {
    fprintf(stderr, "[rccl_ep] ep_dispatch_finish: no deferred dispatch is pending\n");
    return -1;
  }
  return dispatch_tail(h, (hipStream_t)stream);
}

int ep_combine_finish(void* handle, uintptr_t stream) {
  return hybrid_combine_finish(static_cast<EpHandle*>(handle), (hipStream_t)stream);
}

int ep_combine_v3(void* handle,
                  uintptr_t y, uintptr_t in_w, uintptr_t row_map, uintptr_t expert_rows,
                  uintptr_t scales, int fma_mode,
                  uintptr_t recv_topk, uintptr_t recv_src, int num_recv,
                  uintptr_t topk_idx, int num_tokens,
                  uintptr_t bias0, uintptr_t bias1, int grouped,
                  uintptr_t out, uintptr_t out_w, int num_sms, uintptr_t stream) {
  return ep_combine_v4(handle, y, in_w, row_map, expert_rows, scales, fma_mode, recv_topk, recv_src,
                       num_recv, topk_idx, num_tokens, bias0, bias1, grouped, out, out_w, num_sms, 0,
                       stream);
}

int ep_combine_v2(void* handle,
                  uintptr_t y, uintptr_t in_w, uintptr_t row_map, uintptr_t expert_rows,
                  uintptr_t recv_topk, uintptr_t recv_src, int num_recv,
                  uintptr_t topk_idx, int num_tokens,
                  uintptr_t bias0, uintptr_t bias1, int grouped,
                  uintptr_t out, uintptr_t out_w, int num_sms, uintptr_t stream) {
  return ep_combine_v3(handle, y, in_w, row_map, expert_rows, 0, 1, recv_topk, recv_src,
                       num_recv, topk_idx, num_tokens, bias0, bias1, grouped, out, out_w,
                       num_sms, stream);
}

int ep_combine(void* handle,
               uintptr_t y, uintptr_t in_w, uintptr_t row_map,
               uintptr_t recv_topk, uintptr_t recv_src, int num_recv,
               uintptr_t topk_idx, int num_tokens,
               uintptr_t bias0, uintptr_t bias1, int grouped,
               uintptr_t out, uintptr_t out_w, int num_sms, uintptr_t stream) {
  return ep_combine_v3(handle, y, in_w, row_map, 0, 0, 1, recv_topk, recv_src, num_recv,
                       topk_idx, num_tokens, bias0, bias1, grouped, out, out_w, num_sms,
                       stream);
}

// Internode count exchange for an ep_plan plan: all-gathers every rank's `sendc` (device
// int32 [num_ranks]) over the global communicator and writes [sendc, recvc] (host int32
// [2 * num_ranks]) to `out`. Synchronizes: each posted receive needs its size on the host.
int ep_inter_counts(void* handle, uintptr_t sendc, int32_t* out, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_inter_counts");
  if (!h->inter) {
    fprintf(stderr, "[rccl_ep] ep_inter_counts needs an internode handle (ep_create_v2)\n");
    return -1;
  }
  const int R = h->cfg.num_ranks, me = h->cfg.rank;
  hipStream_t s = (hipStream_t)stream;
  NC(ncclAllGather((const void*)sendc, h->d_scratch, R, ncclInt32, h->comm, s));
  HC(hipMemcpyAsync(h->host_scratch, h->d_scratch, (size_t)R * R * sizeof(int32_t),
                    hipMemcpyDeviceToHost, s));
  HC(hipStreamSynchronize(s));
  for (int q = 0; q < R; ++q) {
    out[q] = h->host_scratch[me * R + q];
    out[R + q] = h->host_scratch[q * R + me];
  }
  return 0;
}

// Bind a dispatch's `sendc` / `recvc` (host int32 [num_ranks], from ep_inter_counts) and
// send_list (device int32 [num_ranks, num_tokens]) for the internode dispatch or combine
// that follows. Receives are posted at these sizes and the combine scatters by send_list,
// so a cached replay rebinds its handle's. Stays bound until replaced.
int ep_inter_bind(void* handle, const int32_t* sendc, const int32_t* recvc, uintptr_t send_list,
                  int num_tokens) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_inter_bind");
  if (!h->inter) return 0;
  const int R = h->cfg.num_ranks, T = h->cfg.num_max_tokens_per_rank;
  InterCounts ic{};
  int acc = 0;
  for (int r = 0; r < R; ++r) {
    if (sendc[r] < 0 || sendc[r] > T || recvc[r] < 0 || recvc[r] > T) {
      fprintf(stderr, "[rccl_ep] ep_inter_bind: rank %d counts send %d / recv %d outside [0, %d]\n",
              r, sendc[r], recvc[r], T);
      return -1;
    }
    ic.sendc[r] = sendc[r];
    ic.recvc[r] = recvc[r];
    ic.prefix[r] = acc;
    acc += recvc[r];
    ic.remote[r] = is_remote(h, r) ? 1 : 0;
  }
  h->bound = ic;
  h->bound_send_list = (const int32_t*)send_list;
  h->bound_num_tokens = num_tokens;
  h->is_bound = true;
  return 0;
}

// Switch an internode handle (ep_create_v2) to the two-level scheme (device/hybrid.h);
// call before its first ep_configure. multi: combines reduce within each node before crossing
// (allow_multiple_reduction); without it they take the flat path.
int ep_hybrid_enable(void* handle, int multi) {
  auto* h = static_cast<EpHandle*>(handle);
  if (!h->inter || h->configured) {
    fprintf(stderr, "[rccl_ep] ep_hybrid_enable needs an internode handle, before ep_configure\n");
    return -1;
  }
  h->hybrid = true;
  h->hyb_multi = multi != 0;
  h->hyb.SU = h->node_size;
  h->hyb.SO = h->cfg.num_ranks / h->node_size;
  h->hyb.D = h->cfg.rank / h->node_size;
  h->hyb.u = h->cfg.rank % h->node_size;
  // Collective over the global communicator: every rank enables hybrid together.
  ncclConfig_t rail_cfg = NCCL_CONFIG_INITIALIZER;
  const char* rc = getenv("RCCL_EP_RAIL_CHANNELS");
  // More rail channels raise transfer bandwidth, but their CTAs compete with any compute the
  // caller overlaps with the transfer. 32 channels have hung inside RCCL.
  rail_cfg.nChannelsPerNetPeer = (rc && atoi(rc) > 0) ? atoi(rc) : 8;
  NC(ncclCommSplit(h->comm, h->hyb.u, h->hyb.D, &h->rail_comm, &rail_cfg));
  // High priority, so HIP places `side` in a different hardware queue pool from normal-priority
  // compute streams: a spinning rail kernel on a queue shared with the caller's stream would stall
  // its next kernels. RCCL_EP_RAIL_STREAM=normal keeps normal priority instead, on a queue the
  // default stream does not use (create_stream_apart): whether a plain new stream lands on the
  // default stream's queue depends on how many streams the process created before it and on the
  // HIP version's queue assignment order.
  const char* rs = getenv("RCCL_EP_RAIL_STREAM");
  if (rs && strcmp(rs, "normal") == 0) {
    // The probe allocates and launches on the legacy default stream, which invalidates a stream
    // capture in progress in global or thread-local mode, so it is skipped while one is active.
    // Relaxed mode on this thread keeps the probe's own calls, and the fallback stream's
    // creation, from counting against another thread's capture.
    hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
    HC(hipThreadExchangeStreamCaptureMode(&mode));
    hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
    const bool capturing =
        hipStreamIsCapturing(nullptr, &cap) != hipSuccess || cap != hipStreamCaptureStatusNone;
    const int tries = capturing ? -1 : create_stream_apart(h->device, &h->side);
    hipError_t made = hipSuccess;
    if (tries < 0) {
      (void)hipGetLastError();
      made = hipStreamCreateWithFlags(&h->side, hipStreamNonBlocking);
    }
    (void)hipThreadExchangeStreamCaptureMode(&mode);
    HC(made);
    if (tries < 0) {
      fprintf(stderr, "[rccl_ep] rank %d: %s; using a plain normal-priority rail stream\n",
              h->cfg.rank, capturing ? "stream capture active, queue probe skipped"
                                     : "rail stream queue probe failed");
    } else if (tries == 0) {
      fprintf(stderr, "[rccl_ep] rank %d: no rail stream candidate is apart from the default "
                      "stream's queue; keeping the last\n", h->cfg.rank);
    } else if (h->cfg.rank % h->node_size == 0) {
      fprintf(stderr, "[rccl_ep] rank %d: rail stream apart from the default stream's queue "
                      "(candidate %d)\n", h->cfg.rank, tries);
    }
  } else {
    int least = 0, greatest = 0;
    HC(hipDeviceGetStreamPriorityRange(&least, &greatest));
    HC(hipStreamCreateWithPriority(&h->side, hipStreamNonBlocking, greatest));
  }
  for (int k = 0; k < EpHandle::kMaxChunks; ++k) {
    HC(hipEventCreateWithFlags(&h->ev_a[k], hipEventDisableTiming));
    HC(hipEventCreateWithFlags(&h->ev_b[k], hipEventDisableTiming));
  }
  HC(hipEventCreateWithFlags(&h->ev_bar, hipEventDisableTiming));
  HC(hipEventCreateWithFlags(&h->ev_fwd, hipEventDisableTiming));
  HC(hipEventCreateWithFlags(&h->ev_cnt, hipEventDisableTiming));
  HC(hipEventCreateWithFlags(&h->ev_post, hipEventDisableTiming));
  HC(hipEventCreateWithFlags(&h->ev_dflush, hipEventDisableTiming));
  HC(hipEventCreateWithFlags(&h->ev_cflush, hipEventDisableTiming));
  const char* ch = getenv("RCCL_EP_HYB_CHUNKS");
  h->nchunks = (ch && atoi(ch) > 0) ? std::min(atoi(ch), (int)EpHandle::kMaxChunks) : 1;
  // Both ends of the rail derive the record stride from it, so every rank must set it alike.
  const char* cr = getenv("RCCL_EP_HYB_COMPACT_REC");
  h->hyb_compact = cr && cr[0] == '1';
  // RCCL_EP_GIN: 0 (default) RCCL send/recv on the rail; 1 GIN when the rail has it, else
  // send/recv; 2 GIN or fail. GIN support is decided at communicator init and agreed by every
  // rank, so all ranks take the same branch.
  const int gin_mode = env_int("RCCL_EP_GIN");
  if (gin_mode > 0 && h->hyb.SO > 1 && gin_rail_setup(h) != 0) {
    release_gin_rail(h);
    if (gin_mode >= 2) return -1;
    if (h->cfg.rank == 0) fprintf(stderr, "[rccl_ep] RCCL_EP_GIN=1: the rail falls back to RCCL send/recv\n");
  }
  return 0;
}

// 1 when the hybrid rail runs over GIN (ep_hybrid_enable with RCCL_EP_GIN), else 0.
int ep_rail_gin(void* handle) { return static_cast<EpHandle*>(handle)->gin ? 1 : 0; }

// ep_hcounts' exchange buffers for W = R + SO + E ints per rank: [R * W gathered | W this rank's |
// SU * W this node's, for the two-level gather]. Grown on demand; growing synchronizes.
static int hyb_xchg_reserve(EpHandle* h, int E, hipStream_t s) {
  const int R = h->cfg.num_ranks, W = R + h->hyb.SO + E;
  const size_t need = (size_t)R * W + W + (size_t)h->hyb.SU * W;
  if (need <= h->xchg_ints) return 0;
  HC(hipStreamSynchronize(s));
  hipFree(h->d_xchg);
  if (h->host_xchg) hipHostFree(h->host_xchg);
  h->d_xchg = nullptr;
  h->host_xchg = nullptr;
  h->xchg_ints = 0;
  if (hipMalloc(&h->d_xchg, need * sizeof(int32_t)) != hipSuccess ||
      hipHostMalloc((void**)&h->host_xchg, need * sizeof(int32_t)) != hipSuccess) {
    fprintf(stderr, "[rccl_ep] ep_hcounts: exchange buffers (%zu ints) failed\n", need);
    return -1;
  }
  h->xchg_ints = need;
  return 0;
}

// Hybrid routing plan: k_ep_plan over ranks (out_slot / out_list / out_sendc, as ep_plan) and
// again over nodes (out_node_pos [SO, num_tokens]: a token's record index for the node, -1
// if none of its experts is there; out_node_list: record -> token; out_node_cnt [SO]).
int ep_hplan(void* handle, uintptr_t topk_idx, int num_tokens, uintptr_t out_slot, uintptr_t out_list,
             uintptr_t out_sendc, uintptr_t out_node_pos, uintptr_t out_node_list,
             uintptr_t out_node_cnt, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_hplan");
  if (!h->hybrid) {
    fprintf(stderr, "[rccl_ep] ep_hplan needs a hybrid handle\n");
    return -1;
  }
  hipStream_t s = (hipStream_t)stream;
  EpConfig nc = h->cfg;  // the same plan with nodes for ranks: expert_begin(n) = n * E / SO
  nc.num_ranks = h->hyb.SO;
  nc.rank = h->hyb.D;
  // RCCL_EP_HPLAN_FUSED=1: both plans in one launch (k_ep_hplan), which also writes this rank's
  // row of ep_hcounts' exchange, per-expert counts included.
  static const bool fused = env_int("RCCL_EP_HPLAN_FUSED") == 1;
  h->hplan_row_tokens = -1;
  if (fused && h->cfg.experts_per_rank() <= kEpiMaxExperts) {
    const int R = h->cfg.num_ranks, E = h->cfg.num_experts;
    if (hyb_xchg_reserve(h, E, s) != 0) return -1;
    int32_t* row = h->d_xchg + (size_t)R * (R + nc.num_ranks + E);
    hipLaunchKernelGGL(k_ep_hplan, dim3(R + nc.num_ranks), dim3(kEpWaves * kWarpSize), 0, s,
                       h->cfg, nc, (const int32_t*)topk_idx, num_tokens, (int32_t*)out_slot,
                       (int32_t*)out_list, (int32_t*)out_sendc, (int32_t*)out_node_pos,
                       (int32_t*)out_node_list, (int32_t*)out_node_cnt, row);
    h->hplan_row_tokens = num_tokens;
    return 0;
  }
  hipLaunchKernelGGL(k_ep_plan, dim3(h->cfg.num_ranks), dim3(kEpWaves * kWarpSize), 0, s,
                     h->cfg, (const int32_t*)topk_idx, num_tokens,
                     (int32_t*)out_slot, (int32_t*)out_list, (int32_t*)out_sendc);
  hipLaunchKernelGGL(k_ep_plan, dim3(nc.num_ranks), dim3(kEpWaves * kWarpSize), 0, s,
                     nc, (const int32_t*)topk_idx, num_tokens,
                     (int32_t*)out_node_pos, (int32_t*)out_node_list, (int32_t*)out_node_cnt);
  return 0;
}

// ep_hcounts' device half: this rank's row, the all-gather, and the copy to the host (*W: the row
// width). hcounts_finish waits for the copy and folds it.
static int hcounts_enqueue(EpHandle* h, uintptr_t sendc, uintptr_t node_cnt, uintptr_t topk_idx, int num_tokens,
                           hipStream_t s, int* Wout) {
  // With topk_idx, every rank also contributes its per-expert counts (k_expert_hist) and
  // out_expert gets this rank's per-local-expert receive counts, so the host can size the
  // dispatch here and the dispatch never waits.
  const int R = h->cfg.num_ranks, SO = h->hyb.SO;
  const int E = topk_idx ? h->cfg.num_experts : 0, W = R + SO + E;
  *Wout = W;
  // RCCL_EP_HPLAN_FUSED: ep_hplan already wrote this rank's row for these tokens.
  const bool row_ready = E > 0 && h->hplan_row_tokens == num_tokens;
  h->hplan_row_tokens = -1;
  if (hyb_xchg_reserve(h, E, s) != 0) return -1;
  int32_t* mine = h->d_xchg + (size_t)R * W;
  if (!row_ready) {
    HC(hipMemcpyAsync(mine, (const void*)sendc, R * sizeof(int32_t), hipMemcpyDeviceToDevice, s));
    HC(hipMemcpyAsync(mine + R, (const void*)node_cnt, SO * sizeof(int32_t), hipMemcpyDeviceToDevice, s));
  }
  if (E > 0 && !row_ready) {
    HC(hipMemsetAsync(mine + R + SO, 0, E * sizeof(int32_t), s));
    const int n = num_tokens * h->cfg.num_topk;
    if (n > 0)
      hipLaunchKernelGGL(k_expert_hist, dim3(std::min((n + 255) / 256, 256)), dim3(256), 0, s,
                         (const int32_t*)topk_idx, n, E, mine + R + SO);
  }
  // RCCL_EP_HCOUNTS_2LEVEL=1 gathers within the node over xGMI, then the node blocks along the
  // rail: a few rail steps instead of a ring through every rank. Global rank is node * SU + local
  // index, node_comm ranks are local indices and rail ranks node indices, so the result has the
  // single gather's layout.
  static const bool two_level = env_int("RCCL_EP_HCOUNTS_2LEVEL") == 1;
  if (two_level) {
    int32_t* node_block = mine + W;
    NC(ncclAllGather(mine, node_block, W, ncclInt32, h->node_comm, s));
    NC(ncclAllGather(node_block, h->d_xchg, (size_t)h->hyb.SU * W, ncclInt32, h->rail_comm, s));
  } else {
    NC(ncclAllGather(mine, h->d_xchg, W, ncclInt32, h->comm, s));
  }
  HC(hipMemcpyAsync(h->host_xchg, h->d_xchg, (size_t)R * W * sizeof(int32_t), hipMemcpyDeviceToHost, s));
  return 0;
}

static int hcounts_finish(EpHandle* h, int W, int E, int32_t* out, int32_t* out_expert, hipStream_t s) {
  const int R = h->cfg.num_ranks, SO = h->hyb.SO, me = h->cfg.rank;
  static const bool spin = env_int("RCCL_EP_HCOUNTS_SPIN") == 1;
  if (spin) {
    if (!h->ev_hc) HC(hipEventCreateWithFlags(&h->ev_hc, hipEventDisableTiming));
    HC(hipEventRecord(h->ev_hc, s));
    hipError_t q;
    while ((q = hipEventQuery(h->ev_hc)) == hipErrorNotReady) {
    }
    HC(q);
  } else {
    HC(hipStreamSynchronize(s));
  }
  const int32_t* M = h->host_xchg;
  for (int q = 0; q < R; ++q) {
    out[q] = M[me * W + q];
    out[R + q] = M[q * W + me];
  }
  for (int n = 0; n < SO; ++n) {
    out[2 * R + n] = M[me * W + R + n];
    out[2 * R + SO + n] = M[(n * h->hyb.SU + h->hyb.u) * W + R + h->hyb.D];
  }
  if (E > 0 && out_expert) {
    const int epr = h->cfg.experts_per_rank(), e0 = me * epr;
    for (int j = 0; j < epr; ++j) {
      int acc = 0;
      for (int q = 0; q < R; ++q) acc += M[q * W + R + SO + e0 + j];
      out_expert[j] = acc;
    }
  }
  return 0;
}

// Hybrid count exchange: all-gathers every rank's [sendc (R) | node counts (SO)] and writes
// [sendc (R) | recvc (R) | records sent per node (SO) | records received per node (SO)] into
// `out` on the host; the last are the rail peers' node counts for this node. Synchronizes.
int ep_hcounts(void* handle, uintptr_t sendc, uintptr_t node_cnt, uintptr_t topk_idx, int num_tokens,
               int32_t* out, int32_t* out_expert, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_hcounts");
  if (!h->hybrid) {
    fprintf(stderr, "[rccl_ep] ep_hcounts needs a hybrid handle\n");
    return -1;
  }
  hipStream_t s = (hipStream_t)stream;
  int W = 0;
  if (hcounts_enqueue(h, sendc, node_cnt, topk_idx, num_tokens, s, &W) != 0) return -1;
  return hcounts_finish(h, W, topk_idx ? h->cfg.num_experts : 0, out, out_expert, s);
}

// ep_inter_bind plus the hybrid state: per-node record counts (ep_hcounts), slot_of [R,
// num_tokens] and node_list / node_pos [SO, num_tokens] (ep_hplan), and fwd_meta [SO-1, T, 1+K],
// which the dispatch fills and the combine reads.
int ep_hbind(void* handle, const int32_t* sendc, const int32_t* recvc, const int32_t* nsend,
             const int32_t* nrecv, uintptr_t send_list, uintptr_t slot_of, uintptr_t node_list,
             uintptr_t node_pos, uintptr_t fwd_meta, int num_tokens) {
  auto* h = static_cast<EpHandle*>(handle);
  if (ep_inter_bind(handle, sendc, recvc, send_list, num_tokens) != 0) return -1;
  if (!h->hybrid) return 0;
  const int T = h->cfg.num_max_tokens_per_rank;
  for (int n = 0; n < h->hyb.SO; ++n) {
    if (nsend[n] < 0 || nsend[n] > T || nrecv[n] < 0 || nrecv[n] > T) {
      fprintf(stderr, "[rccl_ep] ep_hbind: node %d records send %d / recv %d outside [0, %d]\n",
              n, nsend[n], nrecv[n], T);
      h->is_bound = false;
      return -1;
    }
    h->hyb.nsend[n] = nsend[n];
    h->hyb.nrecv[n] = nrecv[n];
  }
  h->bound_slot_of = (const int32_t*)slot_of;
  h->bound_node_list = (const int32_t*)node_list;
  h->bound_node_pos = (const int32_t*)node_pos;
  h->bound_fwd_meta = (int32_t*)fwd_meta;
  return 0;
}

// ep_hcounts, then ep_hbind of the counts it returns and the next dispatch's pack and rail
// transfer (hybrid_dispatch_begin), all before returning: the rail starts as soon as the counts
// land instead of after the caller sizes its outputs. The dispatch of x that follows (bf16,
// with metadata, same num_tokens) finds them posted and does not post them again; any other
// dispatch fails.
int ep_hcounts_post(void* handle, uintptr_t sendc, uintptr_t node_cnt, uintptr_t topk_idx,
                    int num_tokens, int32_t* out, int32_t* out_expert, uintptr_t send_list,
                    uintptr_t slot_of, uintptr_t node_list, uintptr_t node_pos, uintptr_t fwd_meta,
                    uintptr_t x, uintptr_t topk_w, int num_sms, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_hcounts_post");
  if (!h->hybrid) {
    fprintf(stderr, "[rccl_ep] ep_hcounts_post needs a hybrid handle\n");
    return -1;
  }
  // dispatch_enqueue's guards, before the rail overwrites so_send and posts into so_recv.
  if (h->pend.on || h->pend_d.on || h->perm.on || h->rail_posted.on) {
    fprintf(stderr, "[rccl_ep] ep_hcounts_post: a deferred combine, dispatch or permute is still pending\n");
    return -1;
  }
  hipStream_t s = (hipStream_t)stream;
  note_stream(h, s);
  const EpConfig& c = h->cfg;
  const size_t T = c.num_max_tokens_per_rank;
  // RCCL_EP_HYB_PREFIX=P: pack every record now, each node's count read from the node plan on the
  // device, and send the first P records to every rail peer (and receive P from each) before the
  // count exchange; only the rest waits for the counts. Every rank knows P, so the sizes match.
  // Records past a peer's count are stale bytes the forwarder never reads (it stops at nrecv).
  // The count exchange must stay off rail_comm, which the prefix transfer occupies.
  // Only the environment value is cached: P is clamped to this window's T on every call, since
  // windows can differ in T and a peer's rail region holds T records.
  static const int prefix_env = env_int("RCCL_EP_HYB_PREFIX");
  const int prefix = std::min(prefix_env, (int)T);
  // A GIN rail moves so_send with one-sided puts under its own credits (device/gin_rail.h); the
  // prefix's send/recv round would write so_recv outside them, so it is send/recv-only.
  const bool pre = !h->gin && prefix > 0 && h->nchunks == 1 && env_int("RCCL_EP_HCOUNTS_2LEVEL") != 1;
  // RCCL_EP_GIN_STREAM: the GIN rail needs no counts from the host at all -- each node's record
  // count is the node plan's node_cnt on the device, and arrivals are flags -- so the whole
  // transfer goes out before the count exchange, and the host sync overlaps it.
  const bool gin_early = h->gin && h->gstream;
  HybInfo hi = h->hyb;
  hi.nchunks = 1;
  hi.chunk = 0;
  hyb_call_layout(h, hi, 0, 1);
  const size_t rb = hi.rec_bytes;
  if (gin_early) {
    hi.nchunks = h->nchunks;
    hi.taper = 1;
    ++h->gseq_d;
    HC(hipStreamWaitEvent(s, h->ev_dflush, 0));  // the previous dispatch's flush, before so_send is rewritten
    for (int part = 0; part < 2 && part < hi.nchunks; ++part) {  // chunk 0 first, as hybrid_dispatch_begin
      hi.chunk = part;
      const int span = part == 0 ? 1 : hi.nchunks - 1;
      hipLaunchKernelGGL(k_hyb_pack_dev, dim3(hi.SO, hyb_ctas(h, num_sms)), dim3(kEpWaves * kWarpSize), 0, s,
                         c, hi, (const void*)x, (const int32_t*)topk_idx, (const float*)topk_w, num_tokens,
                         (const int32_t*)node_list, (const int32_t*)slot_of, (const int32_t*)node_cnt,
                         h->so_send, span);
      HC(hipEventRecord(h->ev_a[part], s));
      HC(hipStreamWaitEvent(h->side, h->ev_a[part], 0));
      if (gin_dispatch_puts(h, hi, (const int32_t*)node_cnt, part, span) != 0) return -1;
    }
    // Barrier A now too, as dispatch_enqueue would run it (counts zeroed first), so the forwards
    // need not wait for the host either.
    HC(hipMemsetAsync((uint8_t*)h->win_buf + h->lay.off_counts, 0, c.num_ranks * sizeof(int32_t), s));
    hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
    note_rdv(h);
    HC(hipEventRecord(h->ev_bar, s));
  }
  if (pre) {
    hipLaunchKernelGGL(k_hyb_pack_dev, dim3(hi.SO, hyb_ctas(h, num_sms)), dim3(kEpWaves * kWarpSize), 0, s,
                       c, hi, (const void*)x, (const int32_t*)topk_idx, (const float*)topk_w, num_tokens,
                       (const int32_t*)node_list, (const int32_t*)slot_of, (const int32_t*)node_cnt,
                       h->so_send, 1);
    HC(hipEventRecord(h->ev_a[0], s));
    HC(hipStreamWaitEvent(h->side, h->ev_a[0], 0));
    NC(ncclGroupStart());
    ncclResult_t gr = ncclSuccess;
    for (int n = 0; n < hi.SO; ++n) {
      if (n == hi.D) continue;
      const size_t np = n < hi.D ? n : n - 1;
      NCG(gr, ncclSend(h->so_send + np * T * rb, (size_t)prefix * rb, ncclUint8, n, h->rail_comm, h->side));
      NCG(gr, ncclRecv(h->so_recv + np * T * rb, (size_t)prefix * rb, ncclUint8, n, h->rail_comm, h->side));
    }
    NCG_END(gr);
  }
  int W = 0;
  if (hcounts_enqueue(h, sendc, node_cnt, topk_idx, num_tokens, s, &W) != 0) return -1;
  if (gin_early) {
    // Each node's record count is in the gathered matrix once the all-gather is done: forward
    // from it on the side stream now, while the host waits for its own copy.
    if (!h->d_nrecv) HC(hipMalloc((void**)&h->d_nrecv, kMaxNodes * sizeof(int32_t)));
    hipLaunchKernelGGL(k_hyb_nrecv, dim3(1), dim3(kMaxNodes), 0, s, h->d_xchg, W, c.num_ranks, h->hyb.SU,
                       h->hyb.SO, h->hyb.u, h->hyb.D, h->d_nrecv);
    HC(hipEventRecord(h->ev_cnt, s));
    HC(hipStreamWaitEvent(h->side, h->ev_cnt, 0));
    if (gin_dispatch_forwards(h, 0, 1, hyb_ctas(h, num_sms), (int32_t*)fwd_meta, h->d_nrecv) != 0) return -1;
  }
  if (hcounts_finish(h, W, topk_idx ? c.num_experts : 0, out, out_expert, s) != 0) return -1;
  const int R = c.num_ranks, SO = h->hyb.SO;
  if (ep_hbind(handle, out, out + R, out + 2 * R, out + 2 * R + SO, send_list, slot_of, node_list,
               node_pos, fwd_meta, num_tokens) != 0)
    return -1;
  if (pre) {
    // The rest, behind the prefix on the side stream.
    NC(ncclGroupStart());
    ncclResult_t gr = ncclSuccess;
    for (int n = 0; n < SO; ++n) {
      if (n == h->hyb.D) continue;
      const size_t np = n < h->hyb.D ? n : n - 1;
      const int ns = h->hyb.nsend[n], nr = h->hyb.nrecv[n];
      if (ns > prefix)
        NCG(gr, ncclSend(h->so_send + (np * T + prefix) * rb, (size_t)(ns - prefix) * rb, ncclUint8, n,
                         h->rail_comm, h->side));
      if (nr > prefix)
        NCG(gr, ncclRecv(h->so_recv + (np * T + prefix) * rb, (size_t)(nr - prefix) * rb, ncclUint8, n,
                         h->rail_comm, h->side));
    }
    NCG_END(gr);
    HC(hipEventRecord(h->ev_b[0], h->side));
  } else if (!gin_early &&
             hybrid_dispatch_begin(h, x, 0, topk_idx, topk_w, num_tokens, 0, 1, hyb_ctas(h, num_sms), s) != 0) {
    return -1;
  }
  h->rail_posted.on = true;
  h->rail_posted.fwd = gin_early;
  h->rail_posted.x = x;
  h->rail_posted.num_tokens = num_tokens;
  return 0;
}

}  // extern "C"
#pragma GCC visibility pop
