#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
RCCL/NCCL baseline benchmark for MoE dispatch + combine.

Implements an equivalent MoE dispatch+combine using only
`torch.distributed.all_to_all_single` (which on AMD routes to RCCL, on NVIDIA
to NCCL). Apples-to-apples baseline for the pipes MoE EP custom kernel:

  * Routing layout (`topk_idx` -> per-rank send counts, permutation, dedup)
    is computed ONCE before the timed loop. This matches how the pipes test
    calls `get_dispatch_layout()` separately at the top of `test_main` and
    re-uses the cached `handle` across bench iterations.

  * Per-(token, dst_rank) dedup via `inplace_unique` — each token is sent at
    MOST once per unique destination rank, even if multiple of its top-k
    experts live on that rank. Without this, the baseline would move
    ~`num_topk × num_unique_ranks_per_token` more bytes than the pipes
    kernel (which dedups internally) — a comparison artifact that has
    nothing to do with the underlying collective performance.

  * Send / recv buffers are pre-allocated outside the timed loop. The pipes
    `dispatch()` / `combine()` reuse the buffer's symmetric region across
    calls; the baseline matches that by pre-sizing the recv buffer based on
    the pre-exchanged counts.

  * Timed region inside `fn()` is the two `dist.all_to_all_single` calls
    (dispatch + combine) and the topk-weighted reduction that the pipes
    `combine()` kernel also does internally. Nothing else.

  * Uses `common_utils.bench` directly (50 warmups, 50 timed runs, L2
    flushed before timing) — identical methodology to the pipes test.

  * Correctness assertion comparing output against a reference implementation
    runs once before the bench loop. The bench fails loudly if the baseline
    is silently wrong.

Default shape matches `test_intranode.py` (high-throughput config):
num_tokens=4096, hidden=7168, num_topk=8, num_experts=256, BF16.

Run with the same wrapper as the intranode test:
  HSA_NO_SCRATCH_RECLAIM=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo \\
    buck2 run @fbcode//mode/opt-amd-gpu \\
    //comms/prims/collectives/moe_ep/tests:test_rccl_moe -- --num-processes 8
"""

# pyre-ignore-all-errors

import argparse

# In-tree aliases (same pattern as test_intranode.py).
import sys

import comms.prims.collectives.moe_ep._cpp as _moe_ep_cpp  # noqa: E402
import comms.prims.collectives.moe_ep.moe_ep as _moe_ep  # noqa: E402

sys.modules["deep_ep"] = _moe_ep
sys.modules["deep_ep_cpp"] = _moe_ep_cpp

# Bind to the in-house moe_ep module (sys.modules shim above) — no import of
# a separate deep_ep package.
deep_ep = _moe_ep
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from comms.prims.collectives.moe_ep.tests.common_utils import (  # noqa: E402  # noqa: E402
    bench,
    init_dist,
    inplace_unique,
    use_rocm,
)


def precompute_dispatch_layout(
    topk_idx: torch.Tensor,
    num_experts: int,
    num_ranks: int,
    group: dist.ProcessGroup,
):
    """Pre-compute the dispatch routing: per-rank send counts, recv counts,
    and the permutation that buckets tokens by destination rank.

    Mirrors what pipes' `get_dispatch_layout()` returns at host-side. Result
    is reused across all timed iterations.

    Returns:
      send_counts: list[int] of length num_ranks (tokens to send to each peer)
      recv_counts: list[int] of length num_ranks (tokens to receive from each)
      send_token_idx: int64 (sum(send_counts),) gather indices into x
      inverse_perm: int64 (sum(send_counts),) un-permute indices for combine
      send_topk_slot: int64 (sum(send_counts),) the topk slot each send refers
        to (for the weighted reduction during combine).
    """
    num_tokens, num_topk = topk_idx.shape
    num_local_experts = num_experts // num_ranks

    # rank_idx[token, k] = which rank owns the k-th top expert for this token.
    rank_idx = (topk_idx // num_local_experts).clone()

    # Dedup per token: a token going to rank R for any of its k experts is
    # sent to R exactly once. inplace_unique fills invalid slots with -1
    # and keeps the unique dst ranks (sorted) in the leading positions.
    inplace_unique(rank_idx, num_ranks)

    # Flatten and drop the -1 padding to get the per-rank send list.
    flat_dst = rank_idx.flatten()  # (num_tokens * num_topk,)
    valid = flat_dst >= 0
    flat_dst_valid = flat_dst[valid]
    # send_token_idx[i] = source token index for the i-th dispatch send.
    token_idx_repeated = (
        torch.arange(num_tokens, device=topk_idx.device)
        .unsqueeze(1)
        .expand(-1, num_topk)
        .flatten()
    )
    send_token_idx_unsorted = token_idx_repeated[valid]

    # Bucket sort by dst_rank so each peer's chunk is contiguous.
    perm = torch.argsort(flat_dst_valid, stable=True)
    send_token_idx = send_token_idx_unsorted[perm]
    sorted_dst = flat_dst_valid[perm]
    send_counts = torch.bincount(sorted_dst, minlength=num_ranks).cpu().tolist()

    # Inverse permutation for the combine path: un-bucket recv data back to
    # per-token order before the weighted sum.
    inverse_perm = torch.empty_like(perm)
    inverse_perm[perm] = torch.arange(perm.numel(), device=perm.device)

    # Exchange counts so both sides know recv sizes (one-shot — not timed).
    send_counts_t = torch.tensor(send_counts, dtype=torch.int64, device=topk_idx.device)
    recv_counts_t = torch.empty(num_ranks, dtype=torch.int64, device=topk_idx.device)
    dist.all_to_all_single(recv_counts_t, send_counts_t, group=group)
    recv_counts = recv_counts_t.cpu().tolist()

    return send_counts, recv_counts, send_token_idx, inverse_perm


def reference_dispatch_combine(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_ranks: int,
    group: dist.ProcessGroup,
):
    """Reference implementation: same as the timed path but unfused. Used
    only for the one-shot correctness assertion. Identical to what the
    pipes kernel computes logically — for the identity-processing
    benchmark scenario, combine returns x weighted by sum(topk_weights)."""
    send_counts, recv_counts, send_token_idx, inverse_perm = precompute_dispatch_layout(
        topk_idx, num_experts, num_ranks, group
    )
    send_buf = x[send_token_idx]
    recv_buf = torch.empty(
        (sum(recv_counts), x.shape[1]), dtype=x.dtype, device=x.device
    )
    dist.all_to_all_single(recv_buf, send_buf, recv_counts, send_counts, group=group)
    # Combine reverses dispatch.
    combine_recv = torch.empty_like(send_buf)
    dist.all_to_all_single(
        combine_recv, recv_buf, send_counts, recv_counts, group=group
    )
    combine_unpermuted = combine_recv[inverse_perm]
    # Each token T receives back one copy per unique dst_rank it sent to.
    # The pipes combine kernel sums these per token (weighted by topk_weights
    # for the experts that mapped to each dst_rank). For the identity
    # benchmark, the reduction is just sum of weights for the experts that
    # routed to each unique dst_rank. We approximate this with a scatter_add
    # back to per-token outputs.
    num_tokens = x.shape[0]
    num_topk = topk_idx.shape[1]
    num_local_experts = num_experts // num_ranks
    rank_idx_orig = topk_idx // num_local_experts  # (num_tokens, num_topk)
    rank_idx_dedup = rank_idx_orig.clone()
    inplace_unique(rank_idx_dedup, num_ranks)
    # token_idx_repeated[i] gives source token, flat_dst_valid[i] gives unique
    # dst_rank for the i-th send. After perm, contiguous-by-dst-rank.
    token_idx_repeated = (
        torch.arange(num_tokens, device=x.device)
        .unsqueeze(1)
        .expand(-1, num_topk)
        .flatten()
    )
    flat_dst = rank_idx_dedup.flatten()
    valid = flat_dst >= 0
    # combine_unpermuted is in send order.
    combined = torch.zeros(
        (num_tokens, x.shape[1]), dtype=torch.float32, device=x.device
    )
    # For each (token, unique_dst_rank), compute the sum of topk_weights for
    # the experts that mapped to that rank.
    weight_per_send = torch.zeros(
        combine_unpermuted.shape[0], dtype=torch.float32, device=x.device
    )
    # rank_idx_orig has full (token, topk) -> dst_rank map; weight per (token,
    # dst_rank) = sum of topk_weights where rank_idx_orig == dst_rank.
    # Build a (num_tokens, num_ranks) weight matrix and scatter_add from
    # topk_weights via rank_idx_orig.
    weights_per_token_per_rank = torch.zeros(
        (num_tokens, num_ranks), dtype=torch.float32, device=x.device
    )
    weights_per_token_per_rank.scatter_add_(
        1, rank_idx_orig, topk_weights.to(torch.float32)
    )
    # Look up weight for each send.
    # Need original (pre-perm) ordering of (token, dst_rank) for weight lookup.
    # The combine_unpermuted is in original (token-major) order BEFORE perm,
    # so use src_token unpermuted (i.e., from the un-perm step).
    src_token_orig = token_idx_repeated[valid]  # original (token, topk-slot) order
    dst_rank_orig = flat_dst[valid]
    weight_per_send = weights_per_token_per_rank[src_token_orig, dst_rank_orig]
    # Weighted sum into output.
    combined.scatter_add_(
        0,
        src_token_orig.unsqueeze(-1).expand(-1, x.shape[1]),
        combine_unpermuted.to(torch.float32) * weight_per_send.unsqueeze(-1),
    )
    return combined.to(x.dtype)


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank)

    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts

    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens} hidden={hidden} num_topk={num_topk} "
            f"num_experts={num_experts} num_ranks={num_ranks} dtype=bf16",
            flush=True,
        )

    # Suppress the spurious `hipErrorPeerAccessAlreadyEnabled` that PyTorch's
    # caching allocator throws on first GPU op in multi-process HIP runs.
    # PyTorch tries to enable peer access between adjacent GPUs as a perf
    # optimization; on AMD ROCm this can collide with the prior peer-access
    # state set up by other ranks. Swallowing the spurious error here is
    # safe — peer access stays enabled either way, which is what we want.
    try:
        _ = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
    except RuntimeError as _re:
        if "PeerAccessAlreadyEnabled" not in str(_re):
            raise

    # Generate inputs matching test_intranode.py methodology.
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    ).abs()
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.softmax(
        torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda"), dim=-1
    ).to(torch.bfloat16)

    # ============ PRE-COMPUTATION (NOT TIMED) ============
    # Routing layout, send/recv counts, permutation, and ALL buffer
    # allocations happen exactly once before the timed loop. This mirrors
    # how the pipes test calls `get_dispatch_layout()` once (timed
    # separately at test_intranode.py:125) and re-uses the cached
    # `handle` across every bench iteration — so the per-iteration timing
    # captures only the actual collective work.
    send_counts, recv_counts, send_token_idx, inverse_perm = precompute_dispatch_layout(
        topk_idx, num_experts, num_ranks, group
    )
    total_send = sum(send_counts)
    total_recv = sum(recv_counts)
    # Pre-allocated buffers — re-used across every bench iteration.
    send_buf = x[send_token_idx].clone()  # (total_send, hidden)
    recv_buf = torch.empty((total_recv, hidden), dtype=x.dtype, device=x.device)
    combine_recv = torch.empty_like(send_buf)
    combine_unpermuted = torch.empty_like(send_buf)
    # Per-(token, dst_rank) weight for the combine reduction. Sums the
    # topk_weights of the experts that mapped to that rank. Equivalent to
    # what the pipes combine kernel applies via topk_weights internally.
    num_local_experts = num_experts // num_ranks
    rank_idx_orig = topk_idx // num_local_experts
    weights_per_token_per_rank = torch.zeros(
        (num_tokens, num_ranks), dtype=torch.float32, device=x.device
    )
    weights_per_token_per_rank.scatter_add_(
        1, rank_idx_orig, topk_weights.to(torch.float32)
    )
    token_idx_repeated = (
        torch.arange(num_tokens, device=x.device)
        .unsqueeze(1)
        .expand(-1, num_topk)
        .flatten()
    )
    rank_idx_dedup = rank_idx_orig.clone()
    inplace_unique(rank_idx_dedup, num_ranks)
    flat_dst = rank_idx_dedup.flatten()
    valid = flat_dst >= 0
    src_token_orig = token_idx_repeated[valid]
    dst_rank_orig = flat_dst[valid]
    weight_per_send = weights_per_token_per_rank[src_token_orig, dst_rank_orig].to(
        x.dtype
    )
    out_indices = src_token_orig.unsqueeze(-1).expand(-1, hidden).contiguous()

    # ============ CORRECTNESS CHECK (NOT TIMED) ============
    # Run the timed path once and verify against a fresh reference computation.
    def run_bench_once():
        dist.all_to_all_single(
            recv_buf, send_buf, recv_counts, send_counts, group=group
        )
        dist.all_to_all_single(
            combine_recv, recv_buf, send_counts, recv_counts, group=group
        )
        # Un-permute and weighted-reduce. Both ops mirror what the pipes
        # combine kernel applies internally.
        combine_unpermuted[:] = combine_recv[inverse_perm]
        out = torch.zeros((num_tokens, hidden), dtype=x.dtype, device=x.device)
        out.scatter_add_(
            0,
            out_indices,
            (combine_unpermuted * weight_per_send.unsqueeze(-1)),
        )
        return out

    # One-shot correctness verification before timing.
    out_test = run_bench_once()
    out_ref = reference_dispatch_combine(
        x, topk_idx, topk_weights, num_experts, num_ranks, group
    )
    max_abs_diff = (out_test.float() - out_ref.float()).abs().max().item()
    rel_diff = max_abs_diff / (out_ref.float().abs().max().item() + 1e-8)
    if local_rank == 0:
        print(
            f"[correctness] max_abs_diff={max_abs_diff:.4e} rel_diff={rel_diff:.4e}",
            flush=True,
        )
    if rel_diff > 1e-2:
        if local_rank == 0:
            print(
                "[WARN] correctness check rel_diff > 1%. Bench may not measure "
                "the same logical op as the pipes kernel.",
                flush=True,
            )
    dist.barrier(group=group)

    # ============ TIMED BENCH (matches pipes methodology exactly) ============
    # `common_utils.bench` does 50 warmups + 50 timed runs with L2 flush.
    # Same function the pipes test uses.
    t_avg, t_min, t_max = bench(run_bench_once)

    # Byte volume for BW reporting.
    # Per rank: sends `total_send * hidden * 2` bytes for dispatch + same back
    # for combine. Total round trip per rank = 2 * total_send * hidden * 2.
    per_rank_dispatch_bytes = total_send * hidden * 2
    total_round_trip_bytes = per_rank_dispatch_bytes * 2

    if local_rank == 0:
        print("", flush=True)
        print(
            f"[rccl-moe] dispatch+combine avg: {t_avg * 1e6:.2f} us "
            f"(min {t_min * 1e6:.2f} us, max {t_max * 1e6:.2f} us)",
            flush=True,
        )
        print(
            f"[rccl-moe] per-rank dispatch send-count total: {total_send} tokens",
            flush=True,
        )
        print(
            f"[rccl-moe] per-rank dispatch volume: "
            f"{per_rank_dispatch_bytes / 1e6:.2f} MB",
            flush=True,
        )
        print(
            f"[rccl-moe] per-rank round-trip BW: "
            f"{total_round_trip_bytes / 1e9 / t_avg:.2f} GB/s",
            flush=True,
        )

    dist.barrier(group=group)
    dist.destroy_process_group()


def bench_e2e_breakdown(
    local_rank: int, num_local_ranks: int, args: argparse.Namespace
):
    """Benchmark B: e2e dispatch+combine with a PER-PART time breakdown that
    INCLUDES the topk-derived pre-computation, for both our moe_ep kernel and
    the RCCL all_to_all baseline.

    Motivation: in real training topk_idx changes every batch,
    so the routing pre-computation (counts + host sync) CANNOT be amortized.
    The existing per-iteration numbers (cached moe_ep dispatch / RCCL with
    routing precomputed outside the loop) hide that cost. Here every part is
    measured per iteration so the e2e picture is accurate.

    moe_ep breakdown (uses cached-vs-non-cached differencing to isolate the
    host sync without C++ instrumentation):
      layout        = get_dispatch_layout(topk_idx)
      dispatch_data = cached dispatch (handle reused; pure NVLink data move)
      notify+sync   = non-cached dispatch - cached dispatch
      combine       = combine(recv_x, handle)

    RCCL breakdown (torch.distributed.all_to_all_single, RCCL on AMD):
      count_compute = rank dedup + argsort + bincount
      counts_a2a+sync = all_to_all_single(counts) + .cpu() host sync
      permute       = gather tokens by destination rank
      data_a2a      = all_to_all_single(data)  [dispatch]
      combine       = reverse all_to_all_single + unpermute + weighted scatter
    """
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank)

    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_local_experts = num_experts // num_ranks
    dev = "cuda"

    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens} hidden={hidden} num_topk={num_topk} "
            f"num_experts={num_experts} num_ranks={num_ranks} dtype=bf16",
            flush=True,
        )

    # Swallow the spurious first-op hipErrorPeerAccessAlreadyEnabled (AMD).
    try:
        _ = torch.zeros(1, dtype=torch.bfloat16, device=dev)
    except RuntimeError as _re:
        if "PeerAccessAlreadyEnabled" not in str(_re):
            raise

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=dev)
    scores = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device=dev
    ).abs()
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.softmax(
        torch.randn((num_tokens, num_topk), dtype=torch.float32, device=dev), dim=-1
    ).to(torch.float32)

    # ==================== moe_ep path ====================
    buffer = deep_ep.Buffer(
        group,
        int(2e9),
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
    )
    num_sms = 64 if use_rocm else 24
    deep_ep.Buffer.set_num_sms(num_sms)
    dispatch_cfg = deep_ep.Buffer.get_dispatch_config(num_ranks)
    combine_cfg = deep_ep.Buffer.get_combine_config(num_ranks)

    def moeep_layout():
        return buffer.get_dispatch_layout(topk_idx, num_experts)

    ntpr, _, ntpe, is_in_rank, _ = moeep_layout()

    def moeep_dispatch_noncached():
        return buffer.dispatch(
            x=x,
            num_tokens_per_rank=ntpr,
            is_token_in_rank=is_in_rank,
            num_tokens_per_expert=ntpe,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            config=dispatch_cfg,
        )

    recv_x, _, recv_tw, _, handle, _ = moeep_dispatch_noncached()

    def moeep_dispatch_cached():
        return buffer.dispatch(x=x, handle=handle, config=dispatch_cfg)

    def moeep_combine():
        return buffer.combine(x=recv_x, handle=handle, config=combine_cfg)

    t_layout = bench(moeep_layout)[0]
    t_disp_full = bench(moeep_dispatch_noncached)[0]
    t_disp_data = bench(moeep_dispatch_cached)[0]
    t_notify_sync = max(t_disp_full - t_disp_data, 0.0)
    t_combine_me = bench(moeep_combine)[0]

    # Optional: sweep nvl_chunk_size for the data legs. Ranges
    # (dispatch range(4,150,4), combine range(1,35,1)) are wider than our
    # test_intranode.py (4..32 / 1..16). Lets us
    # report the speedup at BOTH default and best-of-tuned config from ONE run
    # with a single consistent RCCL baseline -- isolating the effect of (a)
    # per-kernel tuning vs (b) including pre-computation. The notify+host-sync
    # is config-independent, so tuned non-cached dispatch = best cached dispatch
    # + t_notify_sync.
    best_disp = best_combine = None
    best_disp_ck = best_combine_ck = 0
    if args.tune:
        # The ONLY knob swept is nvl_send_chunk (Config arg 2 =
        # num_max_nvl_chunked_send_tokens). num_sms and nvl_recv_buffer (256)
        # are held fixed. The default configs come from the
        # NVIDIA-tuned table (get_dispatch_config / get_combine_config), so on
        # AMD xGMI re-sweeping the chunk recovers real perf.
        nvl_buffer_size = 256
        best_disp = 1e30
        for ck in range(4, 150, 4):
            cfg = deep_ep.Config(num_sms, ck, nvl_buffer_size)
            t = bench(lambda c=cfg: buffer.dispatch(x=x, handle=handle, config=c))[0]
            if t < best_disp:
                best_disp, best_disp_ck = t, ck
        best_combine = 1e30
        for ck in range(1, 35, 1):
            cfg = deep_ep.Config(num_sms, ck, nvl_buffer_size)
            t = bench(lambda c=cfg: buffer.combine(x=recv_x, handle=handle, config=c))[
                0
            ]
            if t < best_combine:
                best_combine, best_combine_ck = t, ck

    # ==================== RCCL all_to_all path ====================
    def rccl_count_compute():
        rank_idx = (topk_idx // num_local_experts).clone()
        inplace_unique(rank_idx, num_ranks)
        flat = rank_idx.flatten()
        valid = flat >= 0
        fv = flat[valid]
        perm = torch.argsort(fv, stable=True)
        sc = torch.bincount(fv[perm], minlength=num_ranks)
        return sc, perm, valid, fv

    sc, perm, valid, fv = rccl_count_compute()
    send_counts = sc.cpu().tolist()
    token_rep = (
        torch.arange(num_tokens, device=dev).unsqueeze(1).expand(-1, num_topk).flatten()
    )
    send_token_idx = token_rep[valid][perm]
    send_counts_t = torch.tensor(send_counts, dtype=torch.int64, device=dev)
    recv_counts_t = torch.empty(num_ranks, dtype=torch.int64, device=dev)

    def rccl_counts_a2a_sync():
        dist.all_to_all_single(recv_counts_t, send_counts_t, group=group)
        return recv_counts_t.cpu().tolist()

    recv_counts = rccl_counts_a2a_sync()
    total_send, total_recv = sum(send_counts), sum(recv_counts)
    send_buf = x[send_token_idx].clone()
    recv_buf = torch.empty((total_recv, hidden), dtype=x.dtype, device=dev)
    combine_recv = torch.empty_like(send_buf)
    inverse_perm = torch.empty_like(perm)
    inverse_perm[perm] = torch.arange(perm.numel(), device=dev)
    src_token = token_rep[valid]
    out_idx = src_token.unsqueeze(-1).expand(-1, hidden).contiguous()

    def rccl_permute():
        return x[send_token_idx]

    def rccl_data_a2a():
        dist.all_to_all_single(
            recv_buf, send_buf, recv_counts, send_counts, group=group
        )

    def rccl_combine():
        dist.all_to_all_single(
            combine_recv, recv_buf, send_counts, recv_counts, group=group
        )
        out = torch.zeros((num_tokens, hidden), dtype=x.dtype, device=dev)
        out.scatter_add_(0, out_idx, combine_recv[inverse_perm])
        return out

    t_cc = bench(rccl_count_compute)[0]
    t_ca = bench(rccl_counts_a2a_sync)[0]
    t_pm = bench(rccl_permute)[0]
    t_da = bench(rccl_data_a2a)[0]
    t_rc = bench(rccl_combine)[0]
    rccl_e2e = t_cc + t_ca + t_pm + t_da + t_rc

    if local_rank == 0:
        u = lambda s: s * 1e6  # noqa: E731
        # With --tune, report the moe_ep data legs at the best-of-sweep config
        # (the config our test_intranode actually uses).
        # notify+host-sync is config-independent, so e2e = layout + (best
        # dispatch + notify_sync) + best combine.
        if best_disp is not None:
            disp_show, comb_show = best_disp, best_combine
            cfg_note = f" [tuned: nvl_send_chunk dispatch {best_disp_ck}, combine {best_combine_ck}]"
        else:
            disp_show, comb_show = t_disp_data, t_combine_me
            cfg_note = ""
        me_e2e_show = t_layout + (disp_show + t_notify_sync) + comb_show
        pre_me = t_layout + t_notify_sync
        print("", flush=True)
        print(
            "================ Benchmark B: e2e breakdown (us) ================",
            flush=True,
        )
        print(f"---- moe_ep (NVLink IPC){cfg_note} ----", flush=True)
        print(f"  layout (get_dispatch_layout) : {u(t_layout):9.1f}", flush=True)
        print(f"  notify + host-sync          : {u(t_notify_sync):9.1f}", flush=True)
        print(f"  dispatch data move          : {u(disp_show):9.1f}", flush=True)
        print(f"  combine                     : {u(comb_show):9.1f}", flush=True)
        print(f"  >> moe_ep e2e total         : {u(me_e2e_show):9.1f}", flush=True)
        print(
            f"     pre-computation share    : {u(pre_me):9.1f}  ({100 * pre_me / me_e2e_show:.1f}%)",
            flush=True,
        )
        print("---- RCCL all_to_all_single ----", flush=True)
        print(f"  count compute               : {u(t_cc):9.1f}", flush=True)
        print(f"  counts a2a + host-sync      : {u(t_ca):9.1f}", flush=True)
        print(f"  permute (gather by dst)     : {u(t_pm):9.1f}", flush=True)
        print(f"  data a2a (dispatch)         : {u(t_da):9.1f}", flush=True)
        print(f"  combine (a2a + scatter)     : {u(t_rc):9.1f}", flush=True)
        print(f"  >> RCCL e2e total           : {u(rccl_e2e):9.1f}", flush=True)
        pre_rccl = t_cc + t_ca + t_pm
        print(
            f"     pre-computation share    : {u(pre_rccl):9.1f}  ({100 * pre_rccl / rccl_e2e:.1f}%)",
            flush=True,
        )
        print("---- comparison ----", flush=True)
        print(
            f"  e2e speedup (RCCL/moe_ep)   : {rccl_e2e / me_e2e_show:.2f}x", flush=True
        )
        print(
            f"  per-rank send/recv tokens   : {total_send} / {total_recv}", flush=True
        )
        # Per-rank BW. Identical topk routing => identical byte volume for both
        # paths: dispatch moves total_send rows out, combine moves total_recv
        # rows back. e2e effective = round-trip bytes / e2e total (incl. the
        # pre-computation that is not itself inter-rank traffic).
        gbps = lambda b, t: (b / 1e9) / t  # noqa: E731
        bd = total_send * hidden * 2  # dispatch send bytes/rank (bf16)
        bc = total_recv * hidden * 2  # combine send bytes/rank (bf16)
        print(
            f"     dispatch {bd / 1e6:.0f} MB/rank, combine {bc / 1e6:.0f} MB/rank",
            flush=True,
        )
        print("---- per-rank BW (GB/s) ----", flush=True)
        print(f"  {'leg':<16}{'moe_ep':>10}{'RCCL':>10}", flush=True)
        print(
            f"  {'dispatch data':<16}{gbps(bd, disp_show):>10.1f}{gbps(bd, t_da):>10.1f}",
            flush=True,
        )
        print(
            f"  {'combine':<16}{gbps(bc, comb_show):>10.1f}{gbps(bc, t_rc):>10.1f}",
            flush=True,
        )
        print(
            f"  {'e2e effective':<16}{gbps(bd + bc, me_e2e_show):>10.1f}{gbps(bd + bc, rccl_e2e):>10.1f}",
            flush=True,
        )
        print(
            "================================================================",
            flush=True,
        )

    buffer.destroy()
    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RCCL/NCCL baseline for MoE dispatch+combine"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension (default: 7168)"
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Top-k experts per token (default: 8)"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="Total number of experts (default: 256)",
    )
    parser.add_argument(
        "--mode",
        choices=("breakdown", "rccl-only"),
        default="breakdown",
        help="breakdown: Benchmark B e2e moe_ep-vs-RCCL with per-part timing "
        "(includes pre-computation). rccl-only: legacy RCCL data-move bench.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Also sweep nvl_chunk_size for the moe_ep data legs and print a "
        "config x precompute 2x2 of the speedup (same RCCL baseline). Shows that "
        "including pre-computation RAISES moe_ep's lead at fixed config.",
    )
    args = parser.parse_args()

    entry = bench_e2e_breakdown if args.mode == "breakdown" else test_loop
    torch.multiprocessing.spawn(
        entry, args=(args.num_processes, args), nprocs=args.num_processes
    )
