/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-GPU end-to-end dispatch and combine for rccl_ep.
//
// The single-GPU tests in device/RcclEpTests.cpp cover the wave primitives and
// the window layout arithmetic. Everything below needs peer memory and a
// communicator, so it runs one process per rank under MPI, in the same shape as
// SymmetricWindowMPITests.cpp. Dispatch and combine move data over LSA peer memory,
// so the eight ranks have to share a node.
//
// Build and run:
//   ./install.sh --debug -t --enable-mpi-tests --enable-rccl-ep-tests
//   NCCL_CUMEM_ENABLE=1 mpirun -np 8 --bind-to none \
//     ./build/debug/test/rccl-UnitTestsMPI --gtest_filter=RcclEpDispatchCombineTest.*
//
// What is asserted, and why the expected value is computable on the host: dispatch
// stages a token once per DESTINATION RANK rather than once per expert, and the
// top-k weights ride along as payload without being applied by the reduction.
// So handing combine exactly what dispatch produced -- an identity expert --
// returns each token scaled by the number of distinct ranks that own its top-k
// experts. That single equality exercises the routing plan, the peer stores, the
// receive-side concatenation and the reduction together, and it is computable on
// the host from topk_idx alone.
//
// The tests after the bandwidth one cover the rest of the C ABI the Python layer drives:
// the plan-notify path (counts exchanged with the plan, one host wait), payload-only
// replay, the grouped-by-expert layout and the combine that reads it, barrier-A elision
// under skewed arrival, ranks that receive nothing, and the internode direct and hybrid
// modes with their receive hooks. Internode runs on one host: ep_create_v2's node_size
// splits the eight ranks into nodes by index, and the inter-node legs go over RCCL
// send/recv whatever transport RCCL picks between them.

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;

// rccl_ep is reached through its flat C ABI; it is header-only otherwise and is
// not hipified, so nothing here includes its device headers.
extern "C" {
int ep_unique_id_size();
int ep_get_unique_id(char* out);
void* ep_create(int rank, int num_ranks, const char* unique_id, int max_tokens_per_rank, int hidden,
                int device);
int ep_configure(void* handle, int num_experts, int num_topk);
void ep_destroy(void* handle);
int ep_plan(void* handle, uintptr_t topk_idx, int num_tokens, uintptr_t slot, uintptr_t send_list,
            uintptr_t sendc, uintptr_t stream);
int ep_dispatch(void* handle, uintptr_t x, uintptr_t x_sf, uintptr_t topk_idx, uintptr_t topk_w,
                int num_tokens, uintptr_t send_list, uintptr_t sendc, int use_fp8, int num_sms,
                uintptr_t out_x, uintptr_t out_sf, uintptr_t out_topk, uintptr_t out_tw,
                uintptr_t out_src, uintptr_t stream);
int ep_combine(void* handle, uintptr_t y, uintptr_t in_w, uintptr_t row_map, uintptr_t recv_topk,
               uintptr_t recv_src, int num_recv, uintptr_t topk_idx, int num_tokens,
               uintptr_t bias0, uintptr_t bias1, int grouped, uintptr_t out, uintptr_t out_w,
               int num_sms, uintptr_t stream);
void* ep_create_v2(int rank, int num_ranks, const char* unique_id, int max_tokens_per_rank,
                   int hidden, int device, int node_size);
int ep_configure_v2(void* handle, int num_experts, int num_topk, int grouped_only);
int ep_plan_notify(void* handle, uintptr_t topk_idx, int num_tokens, uintptr_t slot,
                   uintptr_t send_list, uintptr_t sendc, uintptr_t expert_counts,
                   uintptr_t psum_rank, uintptr_t stream);
int ep_plan_notify_v2(void* handle, uintptr_t topk_idx, int num_tokens, uintptr_t slot,
                      uintptr_t send_list, uintptr_t sendc, uintptr_t expert_counts,
                      uintptr_t psum_rank, uintptr_t rank, uintptr_t srcpref, uintptr_t ebase,
                      uintptr_t stream);
int ep_plan_notify_v3(void* handle, uintptr_t topk_idx_i64, uintptr_t out_topk_i32, int num_tokens,
                      uintptr_t slot, uintptr_t send_list, uintptr_t sendc,
                      uintptr_t expert_counts, uintptr_t psum_rank, uintptr_t rank,
                      uintptr_t srcpref, uintptr_t ebase, uintptr_t stream);
int ep_wait_counts_v2(void* handle, int32_t* out, int n, int timeout_ms);
int ep_dispatch_v2(void* handle, uintptr_t x, uintptr_t x_sf, uintptr_t topk_idx, uintptr_t topk_w,
                   int num_tokens, uintptr_t send_list, uintptr_t sendc, int use_fp8, int num_sms,
                   uintptr_t out_x, uintptr_t out_sf, uintptr_t out_topk, uintptr_t out_tw,
                   uintptr_t out_src, uintptr_t out_counts, uintptr_t host_out, uintptr_t stream);
int ep_dispatch_v3(void* handle, uintptr_t x, uintptr_t x_sf, uintptr_t topk_idx, uintptr_t topk_w,
                   int num_tokens, uintptr_t send_list, uintptr_t sendc, int use_fp8, int num_sms,
                   uintptr_t out_x, uintptr_t out_sf, uintptr_t out_topk, uintptr_t out_tw,
                   uintptr_t out_src, uintptr_t stream);
int ep_dispatch_payload_v2(void* handle, uintptr_t x, int num_tokens, uintptr_t send_list,
                           uintptr_t sendc, uintptr_t psum_rank, int num_sms, uintptr_t out_x,
                           uintptr_t stream);
int ep_dispatch_grouped(void* handle, uintptr_t x, uintptr_t topk_idx, uintptr_t topk_w,
                        int num_tokens, uintptr_t send_list, uintptr_t sendc, uintptr_t rank,
                        int num_sms, int notified, uintptr_t stream);
int ep_grouped_epilogue(void* handle, uintptr_t psum_rank, uintptr_t srcpref, uintptr_t ebase,
                        int num_recv, uintptr_t out_rows, uintptr_t out_row_ids,
                        uintptr_t out_row_w, uintptr_t out_map, uintptr_t out_ids32,
                        uintptr_t out_ids64, uintptr_t out_w, uintptr_t out_src, int num_sms,
                        uintptr_t stream);
int ep_combine_v4(void* handle, uintptr_t y, uintptr_t in_w, uintptr_t row_map,
                  uintptr_t expert_rows, uintptr_t scales, int fma_mode, uintptr_t recv_topk,
                  uintptr_t recv_src, int num_recv, uintptr_t topk_idx, int num_tokens,
                  uintptr_t bias0, uintptr_t bias1, int grouped, uintptr_t out, uintptr_t out_w,
                  int num_sms, int defer, uintptr_t stream);
int ep_combine_finish(void* handle, uintptr_t stream);
int ep_dispatch_defer_next(void* handle);
int ep_dispatch_finish(void* handle, uintptr_t stream);
int ep_inter_counts(void* handle, uintptr_t sendc, int32_t* out, uintptr_t stream);
int ep_inter_bind(void* handle, const int32_t* sendc, const int32_t* recvc, uintptr_t send_list,
                  int num_tokens);
int ep_hybrid_enable(void* handle, int multi);
int ep_hplan(void* handle, uintptr_t topk_idx, int num_tokens, uintptr_t out_slot,
             uintptr_t out_list, uintptr_t out_sendc, uintptr_t out_node_pos,
             uintptr_t out_node_list, uintptr_t out_node_cnt, uintptr_t stream);
int ep_hcounts(void* handle, uintptr_t sendc, uintptr_t node_cnt, uintptr_t topk_idx,
               int num_tokens, int32_t* out, int32_t* out_expert, uintptr_t stream);
int ep_hbind(void* handle, const int32_t* sendc, const int32_t* recvc, const int32_t* nsend,
             const int32_t* nrecv, uintptr_t send_list, uintptr_t slot_of, uintptr_t node_list,
             uintptr_t node_pos, uintptr_t fwd_meta, int num_tokens);
}

namespace RcclUnitTesting
{

namespace {

// The MoE shape these kernels are built for. Anything much smaller measures the fixed
// per-call cost (launch plus the device-side barrier) rather than the transfer: at
// 256 x 1024 that floor is ~62 us against ~5 us of payload.
constexpr int kRanks         = 8;     // one node's worth; rccl_ep dispatch is intranode
constexpr int kTokens        = 4096;
constexpr int kHidden        = 7168;  // multiple of 8: ep_create rejects odd hidden
constexpr int kTopk          = 6;
constexpr int kExpertsPerRank = 32;   // 256 experts across eight ranks
constexpr int kNumSms        = 32;    // past saturation; 64 measures slower, not faster
constexpr int kWarmup        = 10;
constexpr int kIters         = 20;   // each timed separately; the minimum is reported

// Values of the form 1 + m/128 land in [1, 2), where bf16 spacing is exactly 1/128, so
// the payload survives the round trip bit-for-bit. m encodes the source rank, so a row
// delivered from the wrong rank is visible rather than matching by coincidence. Tokens
// 16 apart on the same rank do alias.
float TokenValue(int rank, int token)
{
    return 1.0f + static_cast<float>((rank * 16 + (token % 16)) % 128) / 128.0f;
}

// Experts are chosen by OWNER rather than by index, so the number of distinct ranks a
// token reaches actually varies. With a flat index expression that count is constant at
// every interesting rank count -- always 4 at eight ranks -- which would make the
// assertion below indistinguishable from a plain "scaled by num_topk" check and would
// never exercise combine's per-rank grouping.
int ExpertFor(int rank, int token, int k, int numExperts, int nRanks)
{
    const int perRank = numExperts / nRanks;
    // Every third token deliberately puts two of its experts on one rank.
    const int owner = ((token % 3) == 0 && k == 1)
                          ? ((token * 5) + rank) % nRanks
                          : ((token * 5) + (k * 7) + rank) % nRanks;
    return owner * perRank + ((token + k) % perRank);
}

constexpr int kNodeSize = 4;       // internode tests: two nodes of four ranks
constexpr int kWaitMs   = 60000;   // ep_wait_counts timeout

// Every token picks experts 0..K-1, all owned by rank 0: rank 0 receives every token of
// every rank and the other ranks receive nothing.
int HotExpert(int /*rank*/, int /*token*/, int k, int /*numExperts*/, int /*nRanks*/) { return k; }

using ExpertFn = int (*)(int rank, int token, int k, int numExperts, int nRanks);

// Top-k weights of the form (k + 1) / 32 are exact in fp32.
float WeightFor(int k) { return static_cast<float>(k + 1) / 32.0f; }

// A device allocation released on scope exit. Never zero bytes: a rank that receives no
// rows still passes a valid pointer.
struct DevBuf
{
    void* p = nullptr;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { if (p != nullptr) (void)hipFree(p); }
    bool Alloc(size_t bytes)
    {
        if (p != nullptr) return true;
        return hipMalloc(&p, std::max<size_t>(bytes, 256)) == hipSuccess;
    }
    uintptr_t u() const { return reinterpret_cast<uintptr_t>(p); }
};

// Every element of a payload row holds the same value, so three columns -- both ends and
// the middle -- are enough to see a row that is wrong or only partly written.
bool SampleColumns(const DevBuf& rows, int numRows, std::vector<float>& out)
{
    out.assign(static_cast<size_t>(numRows) * 3, 0.0f);
    if (numRows == 0) return true;
    std::vector<__hip_bfloat16> col(static_cast<size_t>(numRows));
    const int cols[3] = {0, kHidden / 2, kHidden - 1};
    for (int c = 0; c < 3; ++c) {
        if (hipMemcpy2D(col.data(), sizeof(__hip_bfloat16),
                        static_cast<const __hip_bfloat16*>(rows.p) + cols[c],
                        kHidden * sizeof(__hip_bfloat16), sizeof(__hip_bfloat16), numRows,
                        hipMemcpyDeviceToHost) != hipSuccess) {
            return false;
        }
        for (int r = 0; r < numRows; ++r) out[static_cast<size_t>(r) * 3 + c] = __bfloat162float(col[r]);
    }
    return true;
}

template <typename T>
bool Download(const DevBuf& b, size_t n, std::vector<T>& out)
{
    out.resize(n);
    return n == 0 || hipMemcpy(out.data(), b.p, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess;
}

// One dispatch's inputs, plan and outputs, kept together so a later combine or replay can
// use the same plan.
struct Dispatch
{
    std::vector<int32_t> topk;          // [T, K]
    DevBuf x, x2, topkIdx, topkW;       // x2: a second payload for the replays
    DevBuf slot, sendList, sendc, counts, psum;
    DevBuf gRank, srcpref, ebase;       // grouped plan
    DevBuf nodePos, nodeList, nodeCnt, fwdMeta;  // hybrid plan
    std::vector<int32_t> sendcH, recvcH, nsendH, nrecvH;
    std::vector<int32_t> expertCounts;  // received (token, local expert) pairs per local expert
    DevBuf hostOut;                     // pinned [1 + experts per rank], direct internode
    int nRecv = 0;
    int cap   = 0;
    DevBuf outX, outTopk, outTw, outSrc;
    ~Dispatch() { if (hostOut.p != nullptr) { (void)hipHostFree(hostOut.p); hostOut.p = nullptr; } }
};

class RcclEpDispatchCombineTest : public MPITestBase
{
protected:
    void* handle_ = nullptr;
    int   rank_ = 0, nRanks_ = 0, device_ = 0, numExperts_ = 0, topk_ = kTopk;
    int   nodeSize_ = 0;   // 0 for an intranode handle
    bool  hybrid_ = false;

    void TearDown() override
    {
        if (handle_ != nullptr) {
            ep_destroy(handle_);
            handle_ = nullptr;
        }
        MPITestBase::TearDown();
    }

    // ENVIRONMENT ONLY. A runtime without symmetric memory, or a run that is not eight
    // ranks on one node, is a reason to skip. rccl_ep itself failing is not, and this
    // used to share one bool with the ep_create/ep_configure calls below -- which turned
    // a null handle or a rejected configure into a green skip, the precise outcome these
    // tests exist to catch.
    bool EnvReady()
    {
        if (!validateTestPrerequisites(kRanks, kNoProcessLimit, kNoPowerOfTwoRequired,
                                       kRequireSingleNode, kRequireSingleNode)) {
            return false;
        }
        const char* cuMemEnv = std::getenv("NCCL_CUMEM_ENABLE");
        if (cuMemEnv == nullptr || std::string(cuMemEnv) != "1") return false;

        rank_   = MPIEnvironment::world_rank;
        nRanks_ = MPIEnvironment::world_size;

        // MPIEnvironment already bound this rank to its local device; adopt that rather
        // than re-deriving from the global rank. Per-rank, so agree before the collectives.
        int localOk = (hipGetDevice(&device_) == hipSuccess) ? 1 : 0;
        int deviceOk = 0;
        if (MPI_Allreduce(&localOk, &deviceOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS) {
            return false;
        }
        return deviceOk != 0;
    }

    // The library calls, all of them red on failure. Every rank calls in unconditionally
    // and the verdict is taken afterwards: ASSERT_MPI_* is itself an allreduce, so
    // asserting on rank 0's unique id BEFORE the Bcast would leave the other seven
    // blocked in it. (A rank that fails *inside* ep_configure still wedges the rest --
    // that is a known library defect, not one a test-side assertion can reach.)
    //
    // `num_experts` must be a multiple of the rank count, so the owner of an expert is
    // simply index / experts_per_rank.
    void CreateEp(int numExperts, int numTopk)
    {
        numExperts_ = numExperts;
        topk_       = numTopk;

        std::vector<char> uid(static_cast<size_t>(ep_unique_id_size()), 0);
        int idOk = (rank_ != 0 || ep_get_unique_id(uid.data()) == 0) ? 1 : 0;
        ASSERT_MPI_EQ(MPI_SUCCESS, MPI_Bcast(uid.data(), static_cast<int>(uid.size()), MPI_BYTE, 0,
                                             MPI_COMM_WORLD));
        ASSERT_MPI_EQ(1, idOk);

        handle_ = ep_create(rank_, nRanks_, uid.data(), kTokens, kHidden, device_);
        ASSERT_MPI_TRUE(handle_ != nullptr);
        ASSERT_MPI_EQ(0, ep_configure(handle_, numExperts_, topk_));
    }

    int DistinctDestinations(const std::vector<int32_t>& topk, int token) const
    {
        const int expertsPerRank = numExperts_ / nRanks_;
        std::set<int> owners;
        for (int k = 0; k < topk_; ++k) {
            owners.insert(topk[static_cast<size_t>(token) * topk_ + k] / expertsPerRank);
        }
        return static_cast<int>(owners.size());
    }

    // Plan, dispatch, combine with an identity expert, and check every token came back
    // scaled by the number of distinct ranks that own its top-k experts -- see the file
    // header for why that value is computable on the host. Reads numExperts_ and topk_,
    // so it can be re-run after a reconfigure at a different shape.
    void RoundTrip();

    // ep_create_v2 for nRanks_ / nodeSize nodes; hybrid selects the two-level scheme
    // (with multiple reduction and a grouped-only combine window).
    void CreateEpNodes(int numExperts, int numTopk, int nodeSize, bool hybrid);
    void DestroyEp();
    int  ExpertsPerRank() const { return numExperts_ / nRanks_; }

    // Routing, weights and payloads for this rank; x2 = 2 * x, also exact in bf16.
    void MakeInputs(Dispatch& d, ExpertFn expertFor);
    void AllocOutputs(Dispatch& d, int cap);
    // Plan and dispatch through the four paths the Python layer uses.
    void DispatchBarrier(Dispatch& d, hipStream_t stream);   // ep_plan + ep_dispatch
    // sync=false returns with the dispatch still running: ASSERT_MPI_* is an allreduce, so a
    // check after a stream sync would line every rank up behind the slowest one's epilogue.
    void DispatchNotify(Dispatch& d, hipStream_t stream, bool sync = true);  // counts exchanged with the plan
    void DispatchInternode(Dispatch& d, hipStream_t stream); // direct or hybrid
    void Bind(Dispatch& d);
    // Grouped combine of `y` (received rows) into `out`; defer runs it through the receive
    // hook, with other work queued between the call and the hook.
    void Combine(Dispatch& d, const DevBuf& y, DevBuf& out, int defer, hipStream_t stream);

    // What this rank receives, from the routing of every senderStride-th rank.
    void ExpectedReceive(ExpertFn expertFor, int& nRecv, std::vector<int32_t>& expertCounts,
                         int senderStride = 1) const;
    // Each token comes back as value x distinct destination ranks (see the file header).
    void CheckScaledByDestinations(const Dispatch& d, const DevBuf& out, hipStream_t stream,
                                   int numTokens = kTokens);
};

// CreateEp and RoundTrip have to be void to use the ASSERT_MPI_* macros, and an
// assertion inside a subroutine only returns from that subroutine -- both the FAIL()
// arm and the GTEST_SKIP() arm. A caller that ignores that runs on with a null handle,
// so every call to one of them goes through this.
#define ASSERT_EP_STEP(call)                          \
    do {                                              \
        call;                                         \
        if (HasFatalFailure() || IsSkipped()) return; \
    } while (0)

// Plain hipMalloc: only rccl_ep's own window has to be symmetric, not the payload.
struct EpBuffers
{
    void* x = nullptr;        // [T, H]      bf16 (or fp8 when dispatching fp8)
    void* xSf = nullptr;      // [T, H/128]  float, the fp8 input scales
    void* topkIdx = nullptr;  // [T, K]      int32
    void* topkW = nullptr;    // [T, K]      float
    void* slot = nullptr;     // [R, T]      int32
    void* sendList = nullptr; // [R, T]      int32
    void* sendc = nullptr;    // [R]         int32
    void* outX = nullptr;     // [cap, H]
    void* outSf = nullptr;    // [cap, H/128] float, fp8 only
    void* outTopk = nullptr;  // [cap, K]    int32
    void* outTw = nullptr;    // [cap, K]    float
    void* outSrc = nullptr;   // [cap]       int32
    void* combined = nullptr; // [T, H]      bf16

    void Free()
    {
        for (void* p : {x, xSf, topkIdx, topkW, slot, sendList, sendc, outX, outSf, outTopk, outTw,
                        outSrc, combined}) {
            if (p != nullptr) (void)hipFree(p);
        }
    }
};

double GigabytesPerSecond(size_t bytes, float milliseconds)
{
    if (milliseconds <= 0.0f) return 0.0;
    return static_cast<double>(bytes) / (static_cast<double>(milliseconds) * 1.0e6);
}

// A received row costs its payload plus the routing metadata that travels with it; both
// cross the link, so both count toward the achieved rate.
size_t DispatchBytesPerRow(bool useFp8, size_t hiddenSf)
{
    const size_t payload =
        useFp8 ? (kHidden + hiddenSf * sizeof(float)) : (kHidden * sizeof(__hip_bfloat16));
    return payload + kTopk * sizeof(int32_t) + kTopk * sizeof(float);
}

// Combine carries the payload back with its weights, but not the expert indices.
size_t CombineBytesPerRow() { return kHidden * sizeof(__hip_bfloat16) + kTopk * sizeof(float); }

void RcclEpDispatchCombineTest::RoundTrip()
{
    const int    cap       = nRanks_ * kTokens;
    const size_t hiddenSf  = (kHidden + 127) / 128;
    hipStream_t  stream    = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    std::vector<__hip_bfloat16> hX(static_cast<size_t>(kTokens) * kHidden);
    std::vector<int32_t>        hTopk(static_cast<size_t>(kTokens) * topk_);
    std::vector<float>          hTopkW(static_cast<size_t>(kTokens) * topk_, 1.0f / topk_);
    for (int t = 0; t < kTokens; ++t) {
        const __hip_bfloat16 v = __float2bfloat16(TokenValue(rank_, t));
        std::fill_n(hX.begin() + static_cast<size_t>(t) * kHidden, kHidden, v);
        for (int k = 0; k < topk_; ++k) {
            hTopk[static_cast<size_t>(t) * topk_ + k] = ExpertFor(rank_, t, k, numExperts_, nRanks_);
        }
    }

    EpBuffers b;
    SCOPE_EXIT(b.Free());
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.x,        hX.size() * sizeof(__hip_bfloat16)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.topkIdx,  hTopk.size() * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.topkW,    hTopkW.size() * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.slot,     static_cast<size_t>(nRanks_) * kTokens * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.sendList, static_cast<size_t>(nRanks_) * kTokens * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.sendc,    static_cast<size_t>(nRanks_) * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outX,     static_cast<size_t>(cap) * kHidden * sizeof(__hip_bfloat16)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outSf,    static_cast<size_t>(cap) * hiddenSf * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outTopk,  static_cast<size_t>(cap) * topk_ * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outTw,    static_cast<size_t>(cap) * topk_ * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outSrc,   static_cast<size_t>(cap) * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.combined, static_cast<size_t>(kTokens) * kHidden * sizeof(__hip_bfloat16)));

    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(b.x, hX.data(), hX.size() * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpy(b.topkIdx, hTopk.data(), hTopk.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(b.topkW, hTopkW.data(), hTopkW.size() * sizeof(float), hipMemcpyHostToDevice));

    ASSERT_MPI_EQ(0, ep_plan(handle_, (uintptr_t)b.topkIdx, kTokens, (uintptr_t)b.slot,
                             (uintptr_t)b.sendList, (uintptr_t)b.sendc, (uintptr_t)stream));

    const int nRecv = ep_dispatch(handle_, (uintptr_t)b.x, 0, (uintptr_t)b.topkIdx,
                                  (uintptr_t)b.topkW, kTokens, (uintptr_t)b.sendList,
                                  (uintptr_t)b.sendc, /*use_fp8=*/0, kNumSms, (uintptr_t)b.outX, 0,
                                  (uintptr_t)b.outTopk, (uintptr_t)b.outTw, (uintptr_t)b.outSrc,
                                  (uintptr_t)stream);
    ASSERT_MPI_TRUE(nRecv >= 0);
    ASSERT_MPI_TRUE(nRecv <= cap);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Poison the destination first: 0xFF bytes read back as NaN in bf16, so a token
    // combine never writes fails the comparison instead of matching whatever the
    // allocator happened to leave there. Zero-filling would not do -- rank 0 token 0
    // legitimately expects 0.
    ASSERT_MPI_EQ(hipSuccess, hipMemset(b.combined, 0xFF,
                                        static_cast<size_t>(kTokens) * kHidden * sizeof(__hip_bfloat16)));

    // Identity expert: combine is handed exactly what dispatch produced.
    ASSERT_MPI_EQ(0, ep_combine(handle_, (uintptr_t)b.outX, 0, /*row_map=*/0, (uintptr_t)b.outTopk,
                                (uintptr_t)b.outSrc, nRecv, (uintptr_t)b.topkIdx, kTokens, 0, 0,
                                /*grouped=*/1, (uintptr_t)b.combined, 0, kNumSms, (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    std::vector<__hip_bfloat16> hOut(static_cast<size_t>(kTokens) * kHidden);
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hOut.data(), b.combined, hOut.size() * sizeof(__hip_bfloat16),
                                        hipMemcpyDeviceToHost));

    // A uniform spread would be satisfied by any implementation that scaled by a constant.
    {
        std::set<int> spread;
        for (int t = 0; t < kTokens; ++t) spread.insert(DistinctDestinations(hTopk, t));
        ASSERT_MPI_GT(spread.size(), 1u);
    }

    int mismatches = 0;
    for (int t = 0; t < kTokens && mismatches < 8; ++t) {
        const float stored = __bfloat162float(__float2bfloat16(TokenValue(rank_, t)));
        const float expected = stored * static_cast<float>(DistinctDestinations(hTopk, t));
        // bf16 keeps ~8 significant bits, so scale the tolerance with the value.
        const float tol = 0.03f * std::max(1.0f, expected);
        for (int j = 0; j < kHidden; j += 97) {  // stride: every element of a row is identical
            const float got = __bfloat162float(hOut[static_cast<size_t>(t) * kHidden + j]);
            // Negated rather than `> tol`: the poison above reads back as NaN, and every
            // comparison against NaN is false, so `> tol` would let an unwritten row pass.
            if (!(std::abs(got - expected) <= tol)) {
                ADD_FAILURE() << "rank " << rank_ << " token " << t << " element " << j
                              << ": got " << got << ", expected " << expected << " (= value "
                              << stored << " x " << DistinctDestinations(hTopk, t)
                              << " destination ranks)";
                ++mismatches;
                break;
            }
        }
    }
}

void RcclEpDispatchCombineTest::CreateEpNodes(int numExperts, int numTopk, int nodeSize, bool hybrid)
{
    numExperts_ = numExperts;
    topk_       = numTopk;
    nodeSize_   = nodeSize;
    hybrid_     = hybrid;
    std::vector<char> uid(static_cast<size_t>(ep_unique_id_size()), 0);
    int idOk = (rank_ != 0 || ep_get_unique_id(uid.data()) == 0) ? 1 : 0;
    ASSERT_MPI_EQ(MPI_SUCCESS, MPI_Bcast(uid.data(), static_cast<int>(uid.size()), MPI_BYTE, 0,
                                         MPI_COMM_WORLD));
    ASSERT_MPI_EQ(1, idOk);
    handle_ = ep_create_v2(rank_, nRanks_, uid.data(), kTokens, kHidden, device_, nodeSize);
    ASSERT_MPI_TRUE(handle_ != nullptr);
    if (hybrid) ASSERT_MPI_EQ(0, ep_hybrid_enable(handle_, 1));
    ASSERT_MPI_EQ(0, ep_configure_v2(handle_, numExperts_, topk_, hybrid ? 1 : 0));
}

void RcclEpDispatchCombineTest::DestroyEp()
{
    if (handle_ != nullptr) ep_destroy(handle_);
    handle_   = nullptr;
    nodeSize_ = 0;
    hybrid_   = false;
}

void RcclEpDispatchCombineTest::MakeInputs(Dispatch& d, ExpertFn expertFor)
{
    const size_t tk = static_cast<size_t>(kTokens) * topk_;
    std::vector<__hip_bfloat16> hX(static_cast<size_t>(kTokens) * kHidden);
    std::vector<__hip_bfloat16> hX2(hX.size());
    std::vector<float>          hW(tk);
    d.topk.assign(tk, 0);
    for (int t = 0; t < kTokens; ++t) {
        const float v = TokenValue(rank_, t);
        std::fill_n(hX.begin() + static_cast<size_t>(t) * kHidden, kHidden, __float2bfloat16(v));
        std::fill_n(hX2.begin() + static_cast<size_t>(t) * kHidden, kHidden, __float2bfloat16(2.0f * v));
        for (int k = 0; k < topk_; ++k) {
            d.topk[static_cast<size_t>(t) * topk_ + k] = expertFor(rank_, t, k, numExperts_, nRanks_);
            hW[static_cast<size_t>(t) * topk_ + k]     = WeightFor(k);
        }
    }
    ASSERT_MPI_TRUE(d.x.Alloc(hX.size() * sizeof(__hip_bfloat16)) && d.x2.Alloc(hX2.size() * sizeof(__hip_bfloat16)) &&
                    d.topkIdx.Alloc(tk * sizeof(int32_t)) && d.topkW.Alloc(tk * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(d.x.p, hX.data(), hX.size() * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(d.x2.p, hX2.data(), hX2.size() * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(d.topkIdx.p, d.topk.data(), tk * sizeof(int32_t), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(d.topkW.p, hW.data(), tk * sizeof(float), hipMemcpyHostToDevice));
}

void RcclEpDispatchCombineTest::AllocOutputs(Dispatch& d, int cap)
{
    d.cap = cap;
    ASSERT_MPI_TRUE(d.outX.Alloc(static_cast<size_t>(cap) * kHidden * sizeof(__hip_bfloat16)) &&
                    d.outTopk.Alloc(static_cast<size_t>(cap) * topk_ * sizeof(int32_t)) &&
                    d.outTw.Alloc(static_cast<size_t>(cap) * topk_ * sizeof(float)) &&
                    d.outSrc.Alloc(static_cast<size_t>(cap) * sizeof(int32_t)));
}

void RcclEpDispatchCombineTest::DispatchBarrier(Dispatch& d, hipStream_t stream)
{
    const size_t rt = static_cast<size_t>(nRanks_) * kTokens;
    ASSERT_MPI_TRUE(d.slot.Alloc(rt * sizeof(int32_t)) && d.sendList.Alloc(rt * sizeof(int32_t)) &&
                    d.sendc.Alloc(nRanks_ * sizeof(int32_t)));
    ASSERT_EP_STEP(AllocOutputs(d, nRanks_ * kTokens));
    ASSERT_MPI_EQ(0, ep_plan(handle_, d.topkIdx.u(), kTokens, d.slot.u(), d.sendList.u(), d.sendc.u(),
                             (uintptr_t)stream));
    d.nRecv = ep_dispatch(handle_, d.x.u(), 0, d.topkIdx.u(), d.topkW.u(), kTokens, d.sendList.u(),
                          d.sendc.u(), 0, kNumSms, d.outX.u(), 0, d.outTopk.u(), d.outTw.u(),
                          d.outSrc.u(), (uintptr_t)stream);
    ASSERT_MPI_TRUE(d.nRecv >= 0 && d.nRecv <= d.cap);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
}

void RcclEpDispatchCombineTest::DispatchNotify(Dispatch& d, hipStream_t stream, bool sync)
{
    const int    epr = ExpertsPerRank();
    const size_t rt  = static_cast<size_t>(nRanks_) * kTokens;
    ASSERT_MPI_TRUE(d.slot.Alloc(rt * sizeof(int32_t)) && d.sendList.Alloc(rt * sizeof(int32_t)) &&
                    d.sendc.Alloc(nRanks_ * sizeof(int32_t)) && d.counts.Alloc(epr * sizeof(int32_t)) &&
                    d.psum.Alloc(nRanks_ * sizeof(int32_t)));
    ASSERT_EP_STEP(AllocOutputs(d, nRanks_ * kTokens));
    ASSERT_MPI_EQ(0, ep_plan_notify(handle_, d.topkIdx.u(), kTokens, d.slot.u(), d.sendList.u(),
                                    d.sendc.u(), d.counts.u(), d.psum.u(), (uintptr_t)stream));
    ASSERT_MPI_EQ(0, ep_dispatch_v3(handle_, d.x.u(), 0, d.topkIdx.u(), d.topkW.u(), kTokens,
                                    d.sendList.u(), d.sendc.u(), 0, kNumSms, d.outX.u(), 0,
                                    d.outTopk.u(), d.outTw.u(), d.outSrc.u(), (uintptr_t)stream));
    // The host learns the receive total from the plan's count exchange, not from dispatch.
    std::vector<int32_t> wait(static_cast<size_t>(2 + epr + nRanks_), -1);
    ASSERT_MPI_EQ(0, ep_wait_counts_v2(handle_, wait.data(), static_cast<int>(wait.size()), kWaitMs));
    d.nRecv = wait[0];
    d.expertCounts.assign(wait.begin() + 1, wait.begin() + 1 + epr);
    if (sync) ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
}

void RcclEpDispatchCombineTest::DispatchInternode(Dispatch& d, hipStream_t stream)
{
    const int    epr = ExpertsPerRank();
    const int    so  = nRanks_ / nodeSize_;
    const size_t rt  = static_cast<size_t>(nRanks_) * kTokens;
    ASSERT_MPI_TRUE(d.slot.Alloc(rt * sizeof(int32_t)) && d.sendList.Alloc(rt * sizeof(int32_t)) &&
                    d.sendc.Alloc(nRanks_ * sizeof(int32_t)) && d.counts.Alloc(epr * sizeof(int32_t)) &&
                    d.psum.Alloc(nRanks_ * sizeof(int32_t)));
    d.expertCounts.assign(epr, 0);
    if (hybrid_) {
        ASSERT_MPI_TRUE(d.nodePos.Alloc(static_cast<size_t>(so) * kTokens * sizeof(int32_t)) &&
                        d.nodeList.Alloc(static_cast<size_t>(so) * kTokens * sizeof(int32_t)) &&
                        d.nodeCnt.Alloc(so * sizeof(int32_t)) &&
                        d.fwdMeta.Alloc(static_cast<size_t>(std::max(so - 1, 1)) * kTokens * (1 + topk_) * sizeof(int32_t)));
        ASSERT_MPI_EQ(0, ep_hplan(handle_, d.topkIdx.u(), kTokens, d.slot.u(), d.sendList.u(), d.sendc.u(),
                                  d.nodePos.u(), d.nodeList.u(), d.nodeCnt.u(), (uintptr_t)stream));
        // One host sync: rank, node and per-expert counts, before the payload moves.
        std::vector<int32_t> out(static_cast<size_t>(2 * nRanks_ + 2 * so), 0);
        ASSERT_MPI_EQ(0, ep_hcounts(handle_, d.sendc.u(), d.nodeCnt.u(), d.topkIdx.u(), kTokens,
                                    out.data(), d.expertCounts.data(), (uintptr_t)stream));
        d.sendcH.assign(out.begin(), out.begin() + nRanks_);
        d.recvcH.assign(out.begin() + nRanks_, out.begin() + 2 * nRanks_);
        d.nsendH.assign(out.begin() + 2 * nRanks_, out.begin() + 2 * nRanks_ + so);
        d.nrecvH.assign(out.begin() + 2 * nRanks_ + so, out.end());
    } else {
        ASSERT_MPI_EQ(0, ep_plan(handle_, d.topkIdx.u(), kTokens, d.slot.u(), d.sendList.u(), d.sendc.u(),
                                 (uintptr_t)stream));
        std::vector<int32_t> out(static_cast<size_t>(2 * nRanks_), 0);
        ASSERT_MPI_EQ(0, ep_inter_counts(handle_, d.sendc.u(), out.data(), (uintptr_t)stream));
        d.sendcH.assign(out.begin(), out.begin() + nRanks_);
        d.recvcH.assign(out.begin() + nRanks_, out.end());
        if (d.hostOut.p == nullptr) {
            ASSERT_MPI_EQ(hipSuccess, hipHostMalloc(&d.hostOut.p, (1 + epr) * sizeof(int32_t)));
        }
    }
    int total = 0;
    std::vector<int32_t> psum(static_cast<size_t>(nRanks_));
    for (int r = 0; r < nRanks_; ++r) psum[r] = (total += d.recvcH[r]);
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(d.psum.p, psum.data(), psum.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    ASSERT_EP_STEP(AllocOutputs(d, std::max(total, 1)));
    ASSERT_EP_STEP(Bind(d));
    ASSERT_MPI_EQ(0, ep_dispatch_v2(handle_, d.x.u(), 0, d.topkIdx.u(), d.topkW.u(), kTokens,
                                    d.sendList.u(), d.sendc.u(), 0, kNumSms, d.outX.u(), 0,
                                    d.outTopk.u(), d.outTw.u(), d.outSrc.u(), d.counts.u(),
                                    hybrid_ ? 0 : d.hostOut.u(), (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    d.nRecv = total;
    if (!hybrid_) {
        const int32_t* h = static_cast<const int32_t*>(d.hostOut.p);
        ASSERT_MPI_EQ(total, h[0]);
        d.expertCounts.assign(h + 1, h + 1 + epr);
    }
}

void RcclEpDispatchCombineTest::Bind(Dispatch& d)
{
    if (nodeSize_ == 0) return;
    if (hybrid_) {
        ASSERT_MPI_EQ(0, ep_hbind(handle_, d.sendcH.data(), d.recvcH.data(), d.nsendH.data(),
                                  d.nrecvH.data(), d.sendList.u(), d.slot.u(), d.nodeList.u(),
                                  d.nodePos.u(), d.fwdMeta.u(), kTokens));
    } else {
        ASSERT_MPI_EQ(0, ep_inter_bind(handle_, d.sendcH.data(), d.recvcH.data(), d.sendList.u(), kTokens));
    }
}

void RcclEpDispatchCombineTest::Combine(Dispatch& d, const DevBuf& y, DevBuf& out, int defer,
                                        hipStream_t stream)
{
    const size_t bytes = static_cast<size_t>(kTokens) * kHidden * sizeof(__hip_bfloat16);
    ASSERT_MPI_TRUE(out.Alloc(bytes));
    // 0xFF reads back as NaN in bf16, so a token that combine never writes cannot match.
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(out.p, 0xFF, bytes, stream));
    ASSERT_EP_STEP(Bind(d));
    ASSERT_MPI_EQ(0, ep_combine_v4(handle_, y.u(), 0, 0, 0, 0, 1, d.outTopk.u(), d.outSrc.u(), d.nRecv,
                                   d.topkIdx.u(), kTokens, 0, 0, /*grouped=*/1, out.u(), 0, kNumSms,
                                   defer, (uintptr_t)stream));
    if (defer) {
        // Work queued between the call and its hook: the transfer is meant to run under it.
        DevBuf scratch;
        ASSERT_MPI_TRUE(scratch.Alloc(64 << 20));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(scratch.p, 0x5A, 64 << 20, stream));
        ASSERT_MPI_EQ(0, ep_combine_finish(handle_, (uintptr_t)stream));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    }
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
}

void RcclEpDispatchCombineTest::ExpectedReceive(ExpertFn expertFor, int& nRecv,
                                                std::vector<int32_t>& expertCounts,
                                                int senderStride) const
{
    const int epr = ExpertsPerRank();
    nRecv = 0;
    expertCounts.assign(epr, 0);
    for (int r = 0; r < nRanks_; r += senderStride) {
        for (int t = 0; t < kTokens; ++t) {
            bool mine = false;
            for (int k = 0; k < topk_; ++k) {
                const int e = expertFor(r, t, k, numExperts_, nRanks_);
                if (e / epr == rank_) {
                    ++expertCounts[e % epr];
                    mine = true;
                }
            }
            nRecv += mine ? 1 : 0;
        }
    }
}

void RcclEpDispatchCombineTest::CheckScaledByDestinations(const Dispatch& d, const DevBuf& out,
                                                          hipStream_t stream, int numTokens)
{
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    std::vector<float> got;
    ASSERT_MPI_TRUE(SampleColumns(out, numTokens, got));
    int mismatches = 0;
    for (int t = 0; t < numTokens && mismatches < 8; ++t) {
        const float stored   = __bfloat162float(__float2bfloat16(TokenValue(rank_, t)));
        const int   dests    = DistinctDestinations(d.topk, t);
        const float expected = stored * static_cast<float>(dests);
        const float tol      = 0.03f * std::max(1.0f, expected);
        for (int c = 0; c < 3; ++c) {
            const float g = got[static_cast<size_t>(t) * 3 + c];
            if (!(std::abs(g - expected) <= tol)) {
                ADD_FAILURE() << "rank " << rank_ << " token " << t << " column " << c << ": got " << g
                              << ", expected " << expected << " (= " << stored << " x " << dests
                              << " destination ranks)";
                ++mismatches;
                break;
            }
        }
    }
}

} // namespace

TEST_F(RcclEpDispatchCombineTest, DispatchCombineRoundTrip)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk));
    ASSERT_EP_STEP(RoundTrip());
}

// ep_configure is re-enterable: called again at a different shape it tears the window
// down, reallocates and re-registers it, rebuilds the devComm and refills the peer
// views. Nothing else covered that path, and it is not hypothetical -- the Python
// layer reconfigures whenever a caller changes num_experts or num_topk between steps.
//
// The second shape changes BOTH: num_experts alone would leave the window the same
// size, so a reconfigure that quietly kept the old allocation would still pass. topk
// scales `y` and `cw`, so the round trip below can only hold if the new layout is the
// one in force. Running the round trip at each shape is what proves the first configure
// was live too, rather than only the last one.
TEST_F(RcclEpDispatchCombineTest, ReconfigureBetweenShapes)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }

    ASSERT_EP_STEP(CreateEp(nRanks_ * (kExpertsPerRank / 2), kTopk / 2));
    ASSERT_EP_STEP(RoundTrip());

    numExperts_ = nRanks_ * kExpertsPerRank;
    topk_       = kTopk;
    ASSERT_MPI_EQ(0, ep_configure(handle_, numExperts_, topk_));
    ASSERT_EP_STEP(RoundTrip());

    // Repeating the shape it already has is the early-return branch, and it must not
    // free and rebuild a window the caller is still using -- the round trip after it
    // reads the same window.
    ASSERT_MPI_EQ(0, ep_configure(handle_, numExperts_, topk_));
    ASSERT_EP_STEP(RoundTrip());
}

TEST_F(RcclEpDispatchCombineTest, DispatchBf16AndFp8Bandwidth)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk));

    const int    cap      = nRanks_ * kTokens;
    const size_t hiddenSf = (kHidden + 127) / 128;
    hipStream_t  stream   = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    std::vector<__hip_bfloat16> hX(static_cast<size_t>(kTokens) * kHidden, __float2bfloat16(1.0f));
    std::vector<int32_t>        hTopk(static_cast<size_t>(kTokens) * kTopk);
    std::vector<float>          hTopkW(static_cast<size_t>(kTokens) * kTopk, 1.0f / kTopk);
    for (int t = 0; t < kTokens; ++t) {
        for (int k = 0; k < kTopk; ++k) {
            hTopk[static_cast<size_t>(t) * kTopk + k] = ExpertFor(rank_, t, k, numExperts_, nRanks_);
        }
    }

    EpBuffers b;
    SCOPE_EXIT(b.Free());
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.x,        hX.size() * sizeof(__hip_bfloat16)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.topkIdx,  hTopk.size() * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.topkW,    hTopkW.size() * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.slot,     static_cast<size_t>(nRanks_) * kTokens * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.sendList, static_cast<size_t>(nRanks_) * kTokens * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.sendc,    static_cast<size_t>(nRanks_) * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outX,     static_cast<size_t>(cap) * kHidden * sizeof(__hip_bfloat16)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outSf,    static_cast<size_t>(cap) * hiddenSf * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outTopk,  static_cast<size_t>(cap) * kTopk * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outTw,    static_cast<size_t>(cap) * kTopk * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outSrc,   static_cast<size_t>(cap) * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.combined, static_cast<size_t>(kTokens) * kHidden * sizeof(__hip_bfloat16)));

    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(b.x, hX.data(), hX.size() * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpy(b.topkIdx, hTopk.data(), hTopk.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(b.topkW, hTopkW.data(), hTopkW.size() * sizeof(float), hipMemcpyHostToDevice));

    // The fp8 phase must be given real scales: ep_dispatch casts x_sf straight through
    // and device/dispatch.h indexes it unconditionally, so a null here is a device-side
    // null dereference rather than a skipped copy.
    const std::vector<float> hXsf(static_cast<size_t>(kTokens) * hiddenSf, 1.0f);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.xSf, hXsf.size() * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(b.xSf, hXsf.data(), hXsf.size() * sizeof(float), hipMemcpyHostToDevice));

    ASSERT_MPI_EQ(0, ep_plan(handle_, (uintptr_t)b.topkIdx, kTokens, (uintptr_t)b.slot,
                             (uintptr_t)b.sendList, (uintptr_t)b.sendc, (uintptr_t)stream));

    hipEvent_t start, stop;
    ASSERT_MPI_EQ(hipSuccess, hipEventCreate(&start));
    ASSERT_MPI_EQ(hipSuccess, hipEventCreate(&stop));
    SCOPE_EXIT((void)hipEventDestroy(start); (void)hipEventDestroy(stop));

    struct Phase { const char* name; int useFp8; };
    for (const Phase& phase : {Phase{"dispatch bf16", 0}, Phase{"dispatch fp8", 1}}) {
        SCOPED_TRACE(phase.name);
        int nRecv = -1;
        for (int i = 0; i < kWarmup; ++i) {
            nRecv = ep_dispatch(handle_, (uintptr_t)b.x, phase.useFp8 ? (uintptr_t)b.xSf : 0,
                                (uintptr_t)b.topkIdx,
                                (uintptr_t)b.topkW, kTokens, (uintptr_t)b.sendList,
                                (uintptr_t)b.sendc, phase.useFp8, kNumSms, (uintptr_t)b.outX,
                                phase.useFp8 ? (uintptr_t)b.outSf : 0, (uintptr_t)b.outTopk,
                                (uintptr_t)b.outTw, (uintptr_t)b.outSrc, (uintptr_t)stream);
            ASSERT_MPI_TRUE(nRecv >= 0);
        }
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

        // Each iteration is timed on its own, behind a barrier. Without the barrier the
        // in-kernel peer wait absorbs the spread in host launch times and the sample
        // measures rank skew rather than the transfer. Errors are accumulated instead of
        // asserted in place: every rank must reach the same number of barriers, so a rank
        // returning early here would hang the rest.
        std::vector<float> samples;
        samples.reserve(kIters);
        bool timingOk = true;
        for (int i = 0; i < kIters; ++i) {
            timingOk &= (hipStreamSynchronize(stream) == hipSuccess);
            MPI_Barrier(MPI_COMM_WORLD);
            timingOk &= (hipEventRecord(start, stream) == hipSuccess);
            (void)ep_dispatch(handle_, (uintptr_t)b.x, phase.useFp8 ? (uintptr_t)b.xSf : 0,
                              (uintptr_t)b.topkIdx, (uintptr_t)b.topkW,
                              kTokens, (uintptr_t)b.sendList, (uintptr_t)b.sendc, phase.useFp8,
                              kNumSms, (uintptr_t)b.outX, phase.useFp8 ? (uintptr_t)b.outSf : 0,
                              (uintptr_t)b.outTopk, (uintptr_t)b.outTw, (uintptr_t)b.outSrc,
                              (uintptr_t)stream);
            timingOk &= (hipEventRecord(stop, stream) == hipSuccess);
            timingOk &= (hipEventSynchronize(stop) == hipSuccess);
            float ms = 0.0f;
            timingOk &= (hipEventElapsedTime(&ms, start, stop) == hipSuccess);
            samples.push_back(ms);
        }
        ASSERT_MPI_TRUE(timingOk);
        std::sort(samples.begin(), samples.end());

        const size_t bytes =
            static_cast<size_t>(nRecv) * DispatchBytesPerRow(phase.useFp8 != 0, hiddenSf);
        if (rank_ == 0) {
            // MPI tests only build under --debug, and this is wall clock for the whole call
            // rather than an isolated kernel time, so treat it as a regression signal and
            // not as achieved bandwidth.
            printf("[ rccl_ep  ] %-14s %6d rows, %7.1f MB | min %8.3f us %6.1f GB/s"
                   " | med %8.3f us %6.1f GB/s (debug build)\n",
                   phase.name, nRecv, static_cast<double>(bytes) / 1.0e6,
                   samples.front() * 1000.0, GigabytesPerSecond(bytes, samples.front()),
                   samples[samples.size() / 2] * 1000.0,
                   GigabytesPerSecond(bytes, samples[samples.size() / 2]));
            fflush(stdout);
        }

        if (phase.useFp8 == 0) {
            ASSERT_MPI_EQ(0, ep_combine(handle_, (uintptr_t)b.outX, 0, 0, (uintptr_t)b.outTopk,
                                        (uintptr_t)b.outSrc, nRecv, (uintptr_t)b.topkIdx, kTokens, 0, 0, 1,
                                        (uintptr_t)b.combined, 0, kNumSms, (uintptr_t)stream));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

            std::vector<float> cSamples;
            cSamples.reserve(kIters);
            bool combineOk = true;
            for (int i = 0; i < kIters; ++i) {
                combineOk &= (hipStreamSynchronize(stream) == hipSuccess);
                MPI_Barrier(MPI_COMM_WORLD);
                combineOk &= (hipEventRecord(start, stream) == hipSuccess);
                (void)ep_combine(handle_, (uintptr_t)b.outX, 0, 0, (uintptr_t)b.outTopk,
                                 (uintptr_t)b.outSrc, nRecv, (uintptr_t)b.topkIdx, kTokens, 0, 0, 1,
                                 (uintptr_t)b.combined, 0, kNumSms, (uintptr_t)stream);
                combineOk &= (hipEventRecord(stop, stream) == hipSuccess);
                combineOk &= (hipEventSynchronize(stop) == hipSuccess);
                float cms = 0.0f;
                combineOk &= (hipEventElapsedTime(&cms, start, stop) == hipSuccess);
                cSamples.push_back(cms);
            }
            ASSERT_MPI_TRUE(combineOk);
            std::sort(cSamples.begin(), cSamples.end());

            const size_t cbytes = static_cast<size_t>(nRecv) * CombineBytesPerRow();
            if (rank_ == 0) {
                printf("[ rccl_ep  ] %-14s %6d rows, %7.1f MB | min %8.3f us %6.1f GB/s"
                       " | med %8.3f us %6.1f GB/s (debug build)\n",
                       "combine bf16", nRecv, static_cast<double>(cbytes) / 1.0e6,
                       cSamples.front() * 1000.0, GigabytesPerSecond(cbytes, cSamples.front()),
                       cSamples[cSamples.size() / 2] * 1000.0,
                       GigabytesPerSecond(cbytes, cSamples[cSamples.size() / 2]));
                fflush(stdout);
            }
        }
    }
}

// The plan-notify path exchanges the counts with the plan and waits on the host once, where
// the barrier path counts inside dispatch. Both must deliver the same rows to the same
// places, report the receive totals the routing implies, and combine to the same result.
TEST_F(RcclEpDispatchCombineTest, NotifyPathMatchesBarrierPath)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    Dispatch a, b;
    ASSERT_EP_STEP(MakeInputs(a, ExpertFor));
    ASSERT_EP_STEP(MakeInputs(b, ExpertFor));
    ASSERT_EP_STEP(DispatchBarrier(a, stream));
    ASSERT_EP_STEP(DispatchNotify(b, stream));

    int                  wantRecv = 0;
    std::vector<int32_t> wantCounts;
    ExpectedReceive(ExpertFor, wantRecv, wantCounts);
    ASSERT_MPI_EQ(wantRecv, a.nRecv);
    ASSERT_MPI_EQ(wantRecv, b.nRecv);
    ASSERT_MPI_TRUE(b.expertCounts == wantCounts);

    // Same rows, keyed by source (rank * max_tokens + token), with the same top-k ids.
    std::vector<int32_t> srcA, srcB, tkA, tkB;
    std::vector<float>   rowA, rowB;
    ASSERT_MPI_TRUE(Download(a.outSrc, a.nRecv, srcA) && Download(b.outSrc, b.nRecv, srcB) &&
                    Download(a.outTopk, static_cast<size_t>(a.nRecv) * topk_, tkA) &&
                    Download(b.outTopk, static_cast<size_t>(b.nRecv) * topk_, tkB) &&
                    SampleColumns(a.outX, a.nRecv, rowA) && SampleColumns(b.outX, b.nRecv, rowB));
    std::map<int32_t, int> rowOfA;
    for (int i = 0; i < a.nRecv; ++i) rowOfA[srcA[i]] = i;
    int bad = 0;
    for (int i = 0; i < b.nRecv && bad < 8; ++i) {
        auto it = rowOfA.find(srcB[i]);
        if (it == rowOfA.end()) {
            ADD_FAILURE() << "rank " << rank_ << ": notify path received source " << srcB[i]
                          << ", which the barrier path did not";
            ++bad;
            continue;
        }
        const int j = it->second;
        const bool same = std::equal(rowA.begin() + 3 * j, rowA.begin() + 3 * j + 3, rowB.begin() + 3 * i) &&
                          std::equal(tkA.begin() + static_cast<size_t>(j) * topk_,
                                     tkA.begin() + static_cast<size_t>(j + 1) * topk_,
                                     tkB.begin() + static_cast<size_t>(i) * topk_);
        if (!same) {
            ADD_FAILURE() << "rank " << rank_ << ": source " << srcB[i] << " differs between the paths";
            ++bad;
        }
    }

    DevBuf outA, outB;
    ASSERT_EP_STEP(Combine(a, a.outX, outA, 0, stream));
    ASSERT_EP_STEP(Combine(b, b.outX, outB, 0, stream));
    std::vector<float> cA, cB;
    ASSERT_MPI_TRUE(SampleColumns(outA, kTokens, cA) && SampleColumns(outB, kTokens, cB));
    ASSERT_MPI_TRUE(std::memcmp(cA.data(), cB.data(), cA.size() * sizeof(float)) == 0);
    ASSERT_EP_STEP(CheckScaledByDestinations(b, outB, stream));
}

// A payload-only replay moves the rows of a new payload through the saved plan and nothing
// else. It must land exactly the rows a full cached replay lands, and must not touch the
// top-k ids and weights that combine reads from the first dispatch.
TEST_F(RcclEpDispatchCombineTest, PayloadReplayMatchesFullReplay)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    ASSERT_EP_STEP(DispatchNotify(d, stream));
    std::vector<int32_t> tkBefore, srcBefore;
    ASSERT_MPI_TRUE(Download(d.outTopk, static_cast<size_t>(d.nRecv) * topk_, tkBefore) &&
                    Download(d.outSrc, d.nRecv, srcBefore));

    DevBuf payload;
    ASSERT_MPI_TRUE(payload.Alloc(static_cast<size_t>(d.cap) * kHidden * sizeof(__hip_bfloat16)));
    ASSERT_MPI_EQ(0, ep_dispatch_payload_v2(handle_, d.x2.u(), kTokens, d.sendList.u(), d.sendc.u(),
                                            d.psum.u(), kNumSms, payload.u(), (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    std::vector<int32_t> tkAfter;
    ASSERT_MPI_TRUE(Download(d.outTopk, static_cast<size_t>(d.nRecv) * topk_, tkAfter));
    ASSERT_MPI_TRUE(tkAfter == tkBefore);

    Dispatch full;
    ASSERT_EP_STEP(AllocOutputs(full, d.cap));
    ASSERT_MPI_EQ(0, ep_dispatch_v2(handle_, d.x2.u(), 0, d.topkIdx.u(), d.topkW.u(), kTokens,
                                    d.sendList.u(), d.sendc.u(), 0, kNumSms, full.outX.u(), 0,
                                    full.outTopk.u(), full.outTw.u(), full.outSrc.u(), 0, 0,
                                    (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    std::vector<float> rp, rf;
    ASSERT_MPI_TRUE(SampleColumns(payload, d.nRecv, rp) && SampleColumns(full.outX, d.nRecv, rf));
    ASSERT_MPI_TRUE(std::memcmp(rp.data(), rf.data(), rp.size() * sizeof(float)) == 0);
    // And each row is the new payload of its source token.
    int bad = 0;
    for (int i = 0; i < d.nRecv && bad < 8; ++i) {
        const int   srcRank = srcBefore[i] / kTokens, srcTok = srcBefore[i] % kTokens;
        const float want    = __bfloat162float(__float2bfloat16(2.0f * TokenValue(srcRank, srcTok)));
        if (rp[static_cast<size_t>(i) * 3] != want) {
            ADD_FAILURE() << "rank " << rank_ << " row " << i << ": got " << rp[static_cast<size_t>(i) * 3]
                          << ", expected " << want;
            ++bad;
        }
    }
}

// Grouped dispatch lands each received (token, expert) pair as its own row, expert-major.
// Check the layout against the routing, then combine straight from it with each row scaled
// by its token's weight for that expert: every token comes back as value x sum of weights.
TEST_F(RcclEpDispatchCombineTest, GroupedDispatchAndExpertRowsCombine)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    const int    epr = ExpertsPerRank();
    const size_t rt  = static_cast<size_t>(nRanks_) * kTokens;
    const size_t tk  = static_cast<size_t>(kTokens) * topk_;
    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    ASSERT_MPI_TRUE(d.slot.Alloc(rt * sizeof(int32_t)) && d.sendList.Alloc(rt * sizeof(int32_t)) &&
                    d.sendc.Alloc(nRanks_ * sizeof(int32_t)) && d.counts.Alloc(epr * sizeof(int32_t)) &&
                    d.psum.Alloc(nRanks_ * sizeof(int32_t)) && d.gRank.Alloc(tk * sizeof(int32_t)) &&
                    d.srcpref.Alloc(static_cast<size_t>(nRanks_) * epr * sizeof(int32_t)) &&
                    d.ebase.Alloc(epr * sizeof(int32_t)));
    ASSERT_MPI_EQ(0, ep_plan_notify_v2(handle_, d.topkIdx.u(), kTokens, d.slot.u(), d.sendList.u(),
                                       d.sendc.u(), d.counts.u(), d.psum.u(), d.gRank.u(), d.srcpref.u(),
                                       d.ebase.u(), (uintptr_t)stream));
    ASSERT_MPI_EQ(0, ep_dispatch_grouped(handle_, d.x.u(), d.topkIdx.u(), d.topkW.u(), kTokens,
                                         d.sendList.u(), d.sendc.u(), d.gRank.u(), kNumSms,
                                         /*notified=*/1, (uintptr_t)stream));
    std::vector<int32_t> wait(static_cast<size_t>(2 + epr + nRanks_), -1);
    ASSERT_MPI_EQ(0, ep_wait_counts_v2(handle_, wait.data(), static_cast<int>(wait.size()), kWaitMs));
    const int n = wait[0];
    const std::vector<int32_t> counts(wait.begin() + 1, wait.begin() + 1 + epr);
    int rows = 0;
    for (int c : counts) rows += c;

    int                  wantRecv = 0;
    std::vector<int32_t> wantCounts;
    ExpectedReceive(ExpertFor, wantRecv, wantCounts);
    ASSERT_MPI_EQ(wantRecv, n);
    ASSERT_MPI_TRUE(counts == wantCounts);

    DevBuf outRows, rowIds, rowW, emap, ids32, ids64, w, src;
    const size_t nk = static_cast<size_t>(n) * topk_;
    ASSERT_MPI_TRUE(outRows.Alloc(static_cast<size_t>(rows) * kHidden * sizeof(__hip_bfloat16)) &&
                    rowIds.Alloc(static_cast<size_t>(rows) * topk_ * sizeof(int64_t)) &&
                    rowW.Alloc(static_cast<size_t>(rows) * topk_ * sizeof(float)) &&
                    emap.Alloc(static_cast<size_t>(epr) * n * sizeof(int64_t)) &&
                    ids32.Alloc(nk * sizeof(int32_t)) && ids64.Alloc(nk * sizeof(int64_t)) &&
                    w.Alloc(nk * sizeof(float)) && src.Alloc(static_cast<size_t>(n) * sizeof(int32_t)));
    ASSERT_MPI_EQ(0, ep_grouped_epilogue(handle_, d.psum.u(), d.srcpref.u(), d.ebase.u(), n, outRows.u(),
                                         rowIds.u(), rowW.u(), emap.u(), ids32.u(), ids64.u(), w.u(),
                                         src.u(), kNumSms, (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Layout: expert e's rows form one block in expert order, every (token, expert) pair
    // has exactly one row, and the row holds that token's payload.
    std::vector<int64_t> hMap;
    std::vector<int32_t> hSrc;
    std::vector<float>   hRows;
    ASSERT_MPI_TRUE(Download(emap, static_cast<size_t>(epr) * n, hMap) && Download(src, n, hSrc) &&
                    SampleColumns(outRows, rows, hRows));
    std::vector<int> blockStart(static_cast<size_t>(epr) + 1, 0);
    for (int e = 0; e < epr; ++e) blockStart[e + 1] = blockStart[e] + counts[e];
    std::vector<int> used(static_cast<size_t>(rows), 0);
    int bad = 0;
    for (int e = 0; e < epr && bad < 8; ++e) {
        for (int i = 0; i < n && bad < 8; ++i) {
            const int64_t row = hMap[static_cast<size_t>(e) * n + i];
            const int srcRank = hSrc[i] / kTokens, srcTok = hSrc[i] % kTokens;
            bool routed = false;
            for (int k = 0; k < topk_; ++k) {
                routed |= ExpertFor(srcRank, srcTok, k, numExperts_, nRanks_) == rank_ * epr + e;
            }
            if (row < 0) {
                if (routed) {
                    ADD_FAILURE() << "rank " << rank_ << ": source " << hSrc[i] << " has no row for expert " << e;
                    ++bad;
                }
                continue;
            }
            const float want = __bfloat162float(__float2bfloat16(TokenValue(srcRank, srcTok)));
            if (!routed || row < blockStart[e] || row >= blockStart[e + 1] || used[row]++ != 0 ||
                hRows[static_cast<size_t>(row) * 3] != want) {
                ADD_FAILURE() << "rank " << rank_ << ": expert " << e << " source " << hSrc[i] << " row " << row
                              << " (block [" << blockStart[e] << ", " << blockStart[e + 1] << "), routed "
                              << routed << ", value " << hRows[static_cast<size_t>(row) * 3] << " vs " << want << ")";
                ++bad;
            }
        }
    }
    ASSERT_MPI_EQ(rows, static_cast<int>(std::count(used.begin(), used.end(), 1)));

    DevBuf out;
    const size_t bytes = static_cast<size_t>(kTokens) * kHidden * sizeof(__hip_bfloat16);
    ASSERT_MPI_TRUE(out.Alloc(bytes));
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(out.p, 0xFF, bytes, stream));
    ASSERT_MPI_EQ(0, ep_combine_v4(handle_, outRows.u(), 0, 0, emap.u(), w.u(), /*fma=*/1, ids32.u(), src.u(),
                                   n, d.topkIdx.u(), kTokens, 0, 0, /*grouped=*/1, out.u(), 0, kNumSms, 0,
                                   (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    std::vector<float> got;
    ASSERT_MPI_TRUE(SampleColumns(out, kTokens, got));
    float wsum = 0.0f;
    for (int k = 0; k < topk_; ++k) wsum += WeightFor(k);
    int mismatches = 0;
    for (int t = 0; t < kTokens && mismatches < 8; ++t) {
        const float want = __bfloat162float(__float2bfloat16(TokenValue(rank_, t))) * wsum;
        for (int c = 0; c < 3; ++c) {
            const float g = got[static_cast<size_t>(t) * 3 + c];
            if (!(std::abs(g - want) <= 0.03f * std::max(1.0f, want))) {
                ADD_FAILURE() << "rank " << rank_ << " token " << t << ": got " << g << ", expected " << want;
                ++mismatches;
                break;
            }
        }
    }
}

// Barrier-A elision skips a dispatch's or combine's pre-push barrier when a rendezvous
// since this rank's last read of the region already guarantees every peer is done with it.
// Ranks arriving at different times are what would expose a wrong skip, so every call
// here is issued after a per-rank, per-call delay, through the call mix a training step
// makes: uncached dispatch, combine, cached replay, combine, payload replay, combine. The
// outputs with the elision on must equal the outputs with it off, bit for bit.
TEST_F(RcclEpDispatchCombineTest, BarrierElisionUnderSkewIsExact)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    constexpr int kIters = 4;
    std::vector<float> result[2];
    for (int elide = 0; elide < 2; ++elide) {
        // Read when the window is allocated, so it applies to this handle only.
        if (elide) {
            setenv("RCCL_EP_ELIDE_A", "1", 1);
        } else {
            unsetenv("RCCL_EP_ELIDE_A");
        }
        // Unset inside the step, so a failed CreateEp cannot leave it on for later tests.
        ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk); unsetenv("RCCL_EP_ELIDE_A"));
        hipStream_t stream = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
        SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

        Dispatch d;
        ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
        int op = 0;
        auto skew = [&]() { usleep(static_cast<useconds_t>(((rank_ * 37 + op++ * 11) % 8) * 250)); };
        DevBuf out, replay;
        for (int it = 0; it < kIters; ++it) {
            skew();
            ASSERT_EP_STEP(DispatchNotify(d, stream));
            skew();
            ASSERT_EP_STEP(Combine(d, d.outX, out, 0, stream));
            std::vector<float> s;
            ASSERT_MPI_TRUE(SampleColumns(out, kTokens, s));
            result[elide].insert(result[elide].end(), s.begin(), s.end());

            skew();
            ASSERT_MPI_EQ(0, ep_dispatch_v2(handle_, d.x2.u(), 0, d.topkIdx.u(), d.topkW.u(), kTokens,
                                            d.sendList.u(), d.sendc.u(), 0, kNumSms, d.outX.u(), 0,
                                            d.outTopk.u(), d.outTw.u(), d.outSrc.u(), 0, 0, (uintptr_t)stream));
            skew();
            ASSERT_EP_STEP(Combine(d, d.outX, out, 0, stream));
            ASSERT_MPI_TRUE(SampleColumns(out, kTokens, s));
            result[elide].insert(result[elide].end(), s.begin(), s.end());

            ASSERT_MPI_TRUE(replay.Alloc(static_cast<size_t>(d.cap) * kHidden * sizeof(__hip_bfloat16)));
            skew();
            ASSERT_MPI_EQ(0, ep_dispatch_payload_v2(handle_, d.x.u(), kTokens, d.sendList.u(), d.sendc.u(),
                                                    d.psum.u(), kNumSms, replay.u(), (uintptr_t)stream));
            skew();
            ASSERT_EP_STEP(Combine(d, replay, out, 0, stream));
            ASSERT_MPI_TRUE(SampleColumns(out, kTokens, s));
            result[elide].insert(result[elide].end(), s.begin(), s.end());
        }
        ASSERT_EP_STEP(CheckScaledByDestinations(d, out, stream));
        DestroyEp();
    }
    ASSERT_MPI_EQ(result[0].size(), result[1].size());
    ASSERT_MPI_TRUE(std::memcmp(result[0].data(), result[1].data(), result[0].size() * sizeof(float)) == 0);
}

// With barrier-A elision on, a payload replay issued straight after a dispatch must still
// take barrier A: that dispatch's copy epilogue reads this rank's dispatch region, and no
// rendezvous has followed it. Every token goes to rank 0, so rank 0's epilogue copies the
// whole window while the other ranks have nothing to copy and push the replay at once; no
// host-side check runs between the two, since each is an allreduce that would wait for rank 0.
// The replay carries x2 = 2 * x, so a row of the first dispatch it overwrites reads back doubled.
TEST_F(RcclEpDispatchCombineTest, BarrierElisionHoldsAfterADispatch)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    constexpr int kRounds = 8;
    setenv("RCCL_EP_ELIDE_A", "1", 1);
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk); unsetenv("RCCL_EP_ELIDE_A"));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, HotExpert));
    DevBuf replay, out;
    ASSERT_MPI_TRUE(replay.Alloc(static_cast<size_t>(nRanks_) * kTokens * kHidden * sizeof(__hip_bfloat16)));
    int bad = 0;
    for (int it = 0; it < kRounds; ++it) {
        ASSERT_EP_STEP(DispatchNotify(d, stream, /*sync=*/false));
        const int rc = ep_dispatch_payload_v2(handle_, d.x2.u(), kTokens, d.sendList.u(), d.sendc.u(),
                                              d.psum.u(), kNumSms, replay.u(), (uintptr_t)stream);
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
        ASSERT_MPI_EQ(0, rc);
        ASSERT_MPI_EQ(rank_ == 0 ? nRanks_ * kTokens : 0, d.nRecv);
        std::vector<float> first, second;
        std::vector<int32_t> src;
        ASSERT_MPI_TRUE(SampleColumns(d.outX, d.nRecv, first) && SampleColumns(replay, d.nRecv, second) &&
                        Download(d.outSrc, d.nRecv, src));
        for (int i = 0; i < d.nRecv; ++i) {
            const float want = __bfloat162float(__float2bfloat16(TokenValue(src[i] / kTokens, src[i] % kTokens)));
            for (int c = 0; c < 3; ++c) {
                const size_t j = static_cast<size_t>(i) * 3 + c;
                bad += (first[j] != want) + (second[j] != 2.0f * want);
            }
        }
        // A combine between rounds, as in a step, so every round starts from the same state.
        ASSERT_EP_STEP(Combine(d, d.outX, out, 0, stream));
    }
    ASSERT_MPI_EQ(0, bad);
    ASSERT_EP_STEP(CheckScaledByDestinations(d, out, stream));
}

// Every token routed to rank 0: rank 0 receives the whole window and the other ranks
// receive nothing. Dispatch and combine must still complete on every rank and return the
// right rows; a rank with no rows used to be able to leave its peers waiting.
TEST_F(RcclEpDispatchCombineTest, RanksThatReceiveNothing)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, HotExpert));
    ASSERT_EP_STEP(DispatchNotify(d, stream));
    ASSERT_MPI_EQ(rank_ == 0 ? nRanks_ * kTokens : 0, d.nRecv);
    DevBuf out;
    ASSERT_EP_STEP(Combine(d, d.outX, out, 0, stream));
    ASSERT_EP_STEP(CheckScaledByDestinations(d, out, stream));
}

// Odd ranks have no tokens, like an idle data-parallel rank, and pass a null pointer for every
// token-sized buffer, as an empty tensor's data pointer is. They still join the plan's count
// exchange and receive their share. Covers the int64 plan, which writes the int32 copy of the
// routing (ep_plan_notify_v3), and both payload types.
TEST_F(RcclEpDispatchCombineTest, RanksThatSendNothing)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    const bool   idle     = (rank_ % 2) == 1;
    const int    nTok     = idle ? 0 : kTokens;
    const int    epr      = ExpertsPerRank();
    const int    cap      = nRanks_ * kTokens;
    const size_t rt       = static_cast<size_t>(nRanks_) * kTokens;
    const size_t hiddenSf = (kHidden + 127) / 128;
    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    const std::vector<int64_t> topk64(d.topk.begin(), d.topk.end());
    DevBuf topkIdx64, xSf, outSf;
    ASSERT_MPI_TRUE(topkIdx64.Alloc(topk64.size() * sizeof(int64_t)) &&
                    xSf.Alloc(static_cast<size_t>(kTokens) * hiddenSf * sizeof(float)) &&
                    outSf.Alloc(static_cast<size_t>(cap) * hiddenSf * sizeof(float)) &&
                    d.slot.Alloc(rt * sizeof(int32_t)) && d.sendList.Alloc(rt * sizeof(int32_t)) &&
                    d.sendc.Alloc(nRanks_ * sizeof(int32_t)) && d.counts.Alloc(epr * sizeof(int32_t)) &&
                    d.psum.Alloc(nRanks_ * sizeof(int32_t)));
    ASSERT_EP_STEP(AllocOutputs(d, cap));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(topkIdx64.p, topk64.data(), topk64.size() * sizeof(int64_t),
                                        hipMemcpyHostToDevice));
    // The fp8 pass reuses the bf16 payload's bytes; nothing below decodes them.
    ASSERT_MPI_EQ(hipSuccess, hipMemset(xSf.p, 0, static_cast<size_t>(kTokens) * hiddenSf * sizeof(float)));
    auto tokenBuf = [idle](const DevBuf& b) { return idle ? uintptr_t{0} : b.u(); };

    int expRecv = 0;
    std::vector<int32_t> expCounts;
    ExpectedReceive(ExpertFor, expRecv, expCounts, /*senderStride=*/2);

    for (int useFp8 = 0; useFp8 <= 1; ++useFp8) {
        // The plan writes the int32 copy of the routing into topkIdx, which dispatch reads.
        ASSERT_MPI_EQ(0, ep_plan_notify_v3(handle_, tokenBuf(topkIdx64), tokenBuf(d.topkIdx), nTok,
                                           tokenBuf(d.slot), tokenBuf(d.sendList), d.sendc.u(),
                                           d.counts.u(), d.psum.u(), 0, 0, 0, (uintptr_t)stream));
        ASSERT_MPI_EQ(0, ep_dispatch_v3(handle_, tokenBuf(d.x), useFp8 ? tokenBuf(xSf) : 0,
                                        tokenBuf(d.topkIdx), tokenBuf(d.topkW), nTok,
                                        tokenBuf(d.sendList), d.sendc.u(), useFp8, kNumSms, d.outX.u(),
                                        useFp8 ? outSf.u() : 0, d.outTopk.u(), d.outTw.u(),
                                        d.outSrc.u(), (uintptr_t)stream));
        std::vector<int32_t> wait(static_cast<size_t>(2 + epr + nRanks_), -1);
        ASSERT_MPI_EQ(0, ep_wait_counts_v2(handle_, wait.data(), static_cast<int>(wait.size()), kWaitMs));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
        ASSERT_MPI_EQ(expRecv, wait[0]);
        ASSERT_MPI_TRUE(std::equal(expCounts.begin(), expCounts.end(), wait.begin() + 1));
        d.nRecv = wait[0];
        if (useFp8 == 0) {
            // Idle ranks combine nothing into a null output; the others get every token back.
            const size_t bytes = static_cast<size_t>(kTokens) * kHidden * sizeof(__hip_bfloat16);
            DevBuf out;
            ASSERT_MPI_TRUE(out.Alloc(bytes));
            ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(out.p, 0xFF, bytes, stream));
            ASSERT_MPI_EQ(0, ep_combine_v4(handle_, d.outX.u(), 0, 0, 0, 0, 1, d.outTopk.u(),
                                           d.outSrc.u(), d.nRecv, tokenBuf(d.topkIdx), nTok, 0, 0,
                                           /*grouped=*/1, tokenBuf(out), 0, kNumSms, 0,
                                           (uintptr_t)stream));
            ASSERT_EP_STEP(CheckScaledByDestinations(d, out, stream, nTok));
        }
    }
}

// Internode, direct mode: one flat domain; peers on the other node get their rows through
// RCCL send/recv, with the counts exchanged by ep_inter_counts.
TEST_F(RcclEpDispatchCombineTest, InternodeDirectRoundTrip)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEpNodes(nRanks_ * kExpertsPerRank, kTopk, kNodeSize, /*hybrid=*/false));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    ASSERT_EP_STEP(DispatchInternode(d, stream));
    int                  wantRecv = 0;
    std::vector<int32_t> wantCounts;
    ExpectedReceive(ExpertFor, wantRecv, wantCounts);
    ASSERT_MPI_EQ(wantRecv, d.nRecv);
    ASSERT_MPI_TRUE(d.expertCounts == wantCounts);
    DevBuf out;
    ASSERT_EP_STEP(Combine(d, d.outX, out, 0, stream));
    ASSERT_EP_STEP(CheckScaledByDestinations(d, out, stream));
}

// Internode, hybrid mode: a token crosses to each destination node once, to its rail peer,
// which forwards it over the node's peer memory; combine folds the partials at the
// forwarder. With an identity expert the result is the same as the flat round trip.
TEST_F(RcclEpDispatchCombineTest, InternodeHybridRoundTrip)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEpNodes(nRanks_ * kExpertsPerRank, kTopk, kNodeSize, /*hybrid=*/true));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    ASSERT_EP_STEP(DispatchInternode(d, stream));
    int                  wantRecv = 0;
    std::vector<int32_t> wantCounts;
    ExpectedReceive(ExpertFor, wantRecv, wantCounts);
    ASSERT_MPI_EQ(wantRecv, d.nRecv);
    ASSERT_MPI_TRUE(d.expertCounts == wantCounts);
    DevBuf out;
    ASSERT_EP_STEP(Combine(d, d.outX, out, 0, stream));
    ASSERT_EP_STEP(CheckScaledByDestinations(d, out, stream));
}

// The receive hooks split a hybrid combine and a hybrid payload replay into "start" and
// "complete", with other work queued in between. The result must be bit-identical to the
// call without the hook.
TEST_F(RcclEpDispatchCombineTest, HybridReceiveHooksAreBitExact)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEpNodes(nRanks_ * kExpertsPerRank, kTopk, kNodeSize, /*hybrid=*/true));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    ASSERT_EP_STEP(DispatchInternode(d, stream));

    DevBuf plain, hooked;
    ASSERT_EP_STEP(Combine(d, d.outX, plain, 0, stream));
    ASSERT_EP_STEP(Combine(d, d.outX, hooked, 1, stream));
    const size_t bytes = static_cast<size_t>(kTokens) * kHidden * sizeof(__hip_bfloat16);
    std::vector<uint8_t> hp, hh;
    ASSERT_MPI_TRUE(Download(plain, bytes, hp) && Download(hooked, bytes, hh));
    ASSERT_MPI_TRUE(hp == hh);
    ASSERT_EP_STEP(CheckScaledByDestinations(d, hooked, stream));

    DevBuf rowsPlain, rowsHooked, scratch;
    const size_t rowBytes = static_cast<size_t>(d.cap) * kHidden * sizeof(__hip_bfloat16);
    ASSERT_MPI_TRUE(rowsPlain.Alloc(rowBytes) && rowsHooked.Alloc(rowBytes) && scratch.Alloc(64 << 20));
    ASSERT_EP_STEP(Bind(d));
    ASSERT_MPI_EQ(0, ep_dispatch_payload_v2(handle_, d.x2.u(), kTokens, d.sendList.u(), d.sendc.u(),
                                            d.psum.u(), kNumSms, rowsPlain.u(), (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    ASSERT_EP_STEP(Bind(d));
    ASSERT_MPI_EQ(0, ep_dispatch_defer_next(handle_));
    ASSERT_MPI_EQ(0, ep_dispatch_payload_v2(handle_, d.x2.u(), kTokens, d.sendList.u(), d.sendc.u(),
                                            d.psum.u(), kNumSms, rowsHooked.u(), (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(scratch.p, 0x5A, 64 << 20, stream));
    ASSERT_MPI_EQ(0, ep_dispatch_finish(handle_, (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    std::vector<float> sp, sh;
    ASSERT_MPI_TRUE(SampleColumns(rowsPlain, d.nRecv, sp) && SampleColumns(rowsHooked, d.nRecv, sh));
    ASSERT_MPI_TRUE(std::memcmp(sp.data(), sh.data(), sp.size() * sizeof(float)) == 0);
}

// The skewed routing again, in hybrid mode: node 1 sends everything across the rail to
// node 0 and receives nothing back but its own combine results.
TEST_F(RcclEpDispatchCombineTest, HybridRanksThatReceiveNothing)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEpNodes(nRanks_ * kExpertsPerRank, kTopk, kNodeSize, /*hybrid=*/true));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, HotExpert));
    ASSERT_EP_STEP(DispatchInternode(d, stream));
    ASSERT_MPI_EQ(rank_ == 0 ? nRanks_ * kTokens : 0, d.nRecv);
    DevBuf out;
    ASSERT_EP_STEP(Combine(d, d.outX, out, 0, stream));
    ASSERT_EP_STEP(CheckScaledByDestinations(d, out, stream));
}

} // namespace RcclUnitTesting

#endif // MPI_TESTS_ENABLED
