# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Signal: the comms/dsl framework reproduces genai ``all_to_all_single_non_contig``.

The M1 "signal" — an apple-to-apple comparison of the SAME non-contiguous all_to_all on 8x
GB300, run through BOTH implementations with IDENTICAL functionality (the layout transform IS
applied on both sides), so the perf numbers are directly comparable:

* The operation: an equal-split all_to_all where each received ``[rows, cols]`` chunk arrives
  **transposed** to ``[cols, rows]`` (the non-contiguous headline). Torch gold =
  ``dist.all_to_all_single`` then a per-chunk ``[rows, cols] -> [cols, rows]`` transpose.
* **comms/dsl framework** expresses it as ONE call: ``all_to_all(t, out, inp, rows=rows)``, where
  ``rows > 0`` sugar-selects the block-tile ``transpose_tile`` HOOK. The framework coalesced-loads
  each ``[tile, tile]`` chunk into padded SMEM and barriers; the hook coalesced-stores it
  transposed, so the transform is fused into the transfer leg (a single SMEM-staging kernel; the
  raw-byte direct/ce/tma paths do not apply a hook).
* **genai reference** (``all_to_all_single_non_contig``) expresses it on permuted views: input
  ``[ws, rows, cols]`` contiguous, output ``y[ws, cols, rows]`` passed as the strided view
  ``y.transpose(1, 2)`` so the kernel writes each received ``[rows, cols]`` block transposed.

Correctness gate (checked before timing at every shape): framework output, genai output, and
the torch gold are **all three bit-exact** (``torch.equal``) — same functionality, verified.
Then both are timed **runtime SM-matched** (framework grid ``world_size * num_blocks`` = NCCL's
probed active-SM count) and **CUDA-graph** (capture once, median over barriered replay windows).
Every shape's full raw row (framework + genai + NCCL, latency us AND busbw GB/s) is emitted to
the UniBench/AnyBench sink; NCCL plain a2a is a bandwidth reference (it does no transform).

Dual execution: single-node ``mp.spawn`` locally, one-process-per-rank under the conda / fbpkg
MAST launcher (``--module comms.dsl.tests.benchmark_a2a_non_contig``) for GB300 2x4. The genai
kernel's compiled deps require fbpkg delivery (not the conda source overlay).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ._bench_common import (
    _bw_gbps,
    _capture_one_call,
    _DTYPE,
    _find_free_port,
    _fmt_size,
    _framework_num_blocks,
    _iters_for_size,
    _max_num_blocks,
    _max_rank_latency,
    _time_replays,
    emit_result_rows,
)


# (rows, cols) per-peer chunk shapes. rows/cols are multiples of 32 (the SMEM transpose tile),
# so 32x32 = 2KB/peer is the practical floor (a transpose below one tile is degenerate). The
# default ladder spans ~2KB -> ~2GB per peer in ~2x steps (per-peer bytes = rows*cols*2, bf16),
# incl the required 48MB & 96MB. Override with env A2A_SHAPES="RxC,RxC,..." (no rebuild needed).
# Each [rows, cols] block arrives transposed to [cols, rows]; cols is the contiguous inner dim.
def _default_shapes() -> tuple[tuple[int, int], ...]:
    return (
        (32, 32),
        (32, 64),
        (64, 64),
        (64, 128),
        (128, 128),
        (128, 256),  # 2KB..64KB
        (256, 256),
        (256, 512),
        (512, 512),
        (512, 1024),
        (1024, 1024),  # 128KB..2MB
        (1024, 2048),
        (2048, 2048),
        (2048, 4096),
        (4096, 4096),
        (4096, 6144),  # 4MB..48MB
        (6144, 8192),
        (8192, 8192),
        (8192, 16384),
        (16384, 16384),  # 96MB..512MB
        (16384, 32768),
        (32768, 32768),  # 1GB, 2GB
    )


def _parse_shapes(s: str) -> tuple[tuple[int, int], ...]:
    shapes: list[tuple[int, int]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split("x")
        try:
            if len(parts) != 2:
                raise ValueError
            shapes.append((int(parts[0]), int(parts[1])))
        except ValueError:
            # A2A_SHAPES is a dev-facing override; surface a typo clearly instead of a bare
            # IndexError/ValueError from deep in the parse.
            raise ValueError(
                f"bad A2A_SHAPES token {tok!r}, expected RxC (e.g. 512x1024)"
            )
    return tuple(shapes)


def _get_shapes() -> tuple[tuple[int, int], ...]:
    """Chunk shapes, overridable via A2A_SHAPES=RxC,... Read lazily (not at import) per python.md."""
    env = os.environ.get("A2A_SHAPES")
    return _parse_shapes(env) if env else _default_shapes()


def _get_cap() -> int:
    """SM cap, overridable via A2A_CAPS. Read lazily (not at import) per python.md."""
    return int(os.environ.get("A2A_CAPS", "32"))


def _gold_transpose(group, inp, ws: int, rows: int, cols: int):
    """Torch gold: standard all_to_all, then transpose each received [rows, cols] -> [cols, rows]."""
    a2a = torch.empty_like(inp)
    dist.all_to_all_single(a2a, inp, group=group)
    return a2a.view(ws, rows, cols).transpose(1, 2).reshape(-1).contiguous()


def _genai_num_blocks(msg_bytes: int, ws: int, device: torch.device) -> int:
    """Framework ``num_blocks`` that SM-matches GENAI's grid (for the fair framework-vs-genai
    column -- match the baseline you compare against, not always NCCL).

    genai ``all_to_all_single_non_contig`` launches ``grid=(ws, 2*num_send_blocks)`` =
    ``ws*2*num_send_blocks`` CTAs, with ``num_send_blocks = clamp(ceil(per_peer_numel/tile_numel),
    1, blocks_per_peer)`` and ``blocks_per_peer = _compute_max_blocks_per_peer(ws, device)``
    (= ``effective_sms // (2*ws)``). The framework grid is ``ws*num_blocks``, so matching genai's
    SM count => ``num_blocks = 2*num_send_blocks``. Uses genai's OWN importable blocks-per-peer
    helper so the two stay in lockstep (incl the ``A2A_NON_CONTIG_MAX_SMS`` env override, which
    must be set identically for both when capping SMs). Falls back to the full-SM budget if genai
    is not importable (conda delivery)."""
    elem = torch.tensor([], dtype=_DTYPE).element_size()
    per_peer_numel = msg_bytes // elem
    try:
        from ops.kernels.triton.comm.all_to_all_non_contig_common import (
            _compute_max_blocks_per_peer,
        )

        blocks_per_peer = _compute_max_blocks_per_peer(ws, device)
    except Exception:  # noqa: BLE001 -- genai unavailable (conda): fall back to full-SM budget
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        blocks_per_peer = max(sm // (2 * ws), 1)
    tile_numel = 32 * (
        2048 // elem
    )  # genai regular-path 64KB tile (small-msg clamp only)
    num_tiles = max((per_peer_numel + tile_numel - 1) // tile_numel, 1)
    num_send_blocks = max(min(num_tiles, blocks_per_peer), 1)
    return 2 * num_send_blocks


def _framework_transform(t, inp, rows, num_blocks):
    """Returns a fn that runs the non-contig transpose a2a and RETURNS its output tensor.

    Uses :func:`all_to_all_transpose` -- the block-tile ``transpose_tile`` HOOK composed into the
    zero-copy transfer (auto-selects orchestrated for the mid band / fused for the large band). This
    is the shipped path a user gets from ``all_to_all(rows=R)``. The copy-staged variant was
    evaluated and removed (mid-band send-leg-bound; rationale + measurements are archived off-tree
    with the removal diff)."""
    from comms.dsl.cute.a2a.host import all_to_all_transpose
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    cfg = CuteA2AConfig(num_blocks=num_blocks)

    def fn():
        return all_to_all_transpose(t, inp, rows, config=cfg)

    return fn


def _genai_transform(inp, y_buf, ws, rows, cols, group):
    """genai all_to_all_single_non_contig doing the SAME per-chunk transpose via permuted views."""
    from ops.kernels.triton.comm.all_to_all_single_non_contig import (
        all_to_all_single_non_contig,
    )

    x = inp.view(
        ws, rows, cols
    )  # contiguous input, dim0 = ws (one [rows, cols] chunk/peer)
    out_view = y_buf.view(ws, cols, rows).transpose(
        1, 2
    )  # [ws, rows, cols] STRIDED view

    def fn():
        all_to_all_single_non_contig(out_view, x, group=group)

    return fn


def _time(fn, msg_bytes, ws, device, group):
    g = _capture_one_call(fn)
    dist.barrier(group)
    iters = _iters_for_size(msg_bytes)
    lat = _max_rank_latency(_time_replays(g, iters, device, group), device, group)
    numel = msg_bytes * ws // torch.tensor([], dtype=_DTYPE).element_size()
    return lat, _bw_gbps(numel, ws, lat)


def _run(rank, ws, group, device) -> bool:  # noqa: C901
    from comms.dsl import nvl_rendezvous

    elem = torch.tensor([], dtype=_DTYPE).element_size()
    mbp = 32
    rows_out = []
    ok = True
    for rows, cols in _get_shapes():
        per_peer = rows * cols * elem
        total = ws * rows * cols
        inp = (
            (rank + 1) * 977 + torch.arange(total, device=device, dtype=torch.float32)
        ).to(_DTYPE)
        gold = _gold_transpose(group, inp, ws, rows, cols)
        # Per-baseline SM-match: the framework grid is ws*num_blocks; match it to the SM count of
        # WHICHEVER baseline the column compares against -- genai's grid for the genai columns,
        # NCCL's grid for the NCCL column. (Matching genai to NCCL's SMs, as before, handicapped
        # the fair comparison.)
        nb_genai = _genai_num_blocks(per_peer, ws, device)
        nb_nccl, nccl_grid = _framework_num_blocks(
            per_peer, _get_cap(), ws, device, mbp
        )
        nb_nccl = min(nb_nccl, _max_num_blocks(ws, device, mbp))

        def _time_fw(nb, _pp=per_peer, _rows=rows, _inp=inp, _gold=gold):
            # Fresh transport per num_blocks geometry -- the reuse guard forbids switching
            # num_blocks / primitive on one transport without a drain.
            t = nvl_rendezvous(group, device, per_peer_bytes=_pp)
            fn = _framework_transform(t, _inp, _rows, nb)
            out = fn()
            torch.cuda.synchronize(device)
            okk = torch.equal(out, _gold)
            # `out` is a VIEW into `t`'s symmetric-memory output buffer (zero-copy path); it is
            # freed when `t` goes out of scope on return, and `_time`'s graph replays overwrite it.
            # Clone to an owned tensor so callers (the fw-vs-genai bit-exact check) stay valid.
            out_owned = out.clone()
            lat, bw = _time(fn, _pp, ws, device, group)
            return okk, out_owned, lat, bw

        # HEADLINE: framework transpose HOOK (zero-copy transfer), SM-matched to GENAI.
        fw_ok, fw_out, fw_lat, fw_bw = False, None, None, None
        try:
            fw_ok, fw_out, fw_lat, fw_bw = _time_fw(nb_genai)
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                print(
                    f"  # fw {rows}x{cols} FAILED: {type(e).__name__}: {e}", flush=True
                )
        dist.barrier(group)

        # vs NCCL: framework, SM-matched to NCCL.
        fw_nccl_bw = None
        try:
            _n_ok, _n_out, _n_lat, fw_nccl_bw = _time_fw(nb_nccl)
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                print(
                    f"  # fw@nccl {rows}x{cols} FAILED: {type(e).__name__}: {e}",
                    flush=True,
                )
        dist.barrier(group)

        # --- genai reference (permuted-view transpose, SAME op; genai's own grid) ---
        genai_lat = genai_bw = None
        genai_out = None
        try:
            y_buf = torch.empty(ws * cols * rows, device=device, dtype=_DTYPE)
            gn_fn = _genai_transform(inp, y_buf, ws, rows, cols, group)
            gn_fn()
            torch.cuda.synchronize(device)
            genai_out = y_buf
            genai_lat, genai_bw = _time(gn_fn, per_peer, ws, device, group)
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                print(
                    f"  # genai {rows}x{cols} FAILED: {type(e).__name__}: {e}",
                    flush=True,
                )
        dist.barrier(group)

        # --- NCCL plain a2a (bandwidth reference; no transform) ---
        nccl_lat = nccl_bw = None
        try:
            ref = torch.empty_like(inp)

            def nccl_fn(_r=ref, _i=inp):
                dist.all_to_all_single(_r, _i, group=group)

            nccl_lat, nccl_bw = _time(nccl_fn, per_peer, ws, device, group)
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                print(f"  # nccl {rows}x{cols} FAILED: {e}", flush=True)
        dist.barrier(group)

        # Correctness gate = framework vs torch GOLD (non-negotiable); + framework == genai when
        # genai is present (its absence -- conda without its deps -- must NOT fail the framework
        # check, but is recorded so the summary does not silently claim genai coverage it lacks).
        be_gold = bool(fw_ok)
        # genai COVERAGE (genai_ok) = genai produced output AND was TIMED -- purely a genai-side
        # fact, INDEPENDENT of framework success. `genai_out = y_buf` is assigned before
        # `_time(gn_fn)`, so a genai timing failure leaves genai_out set but genai_bw=None; gating on
        # genai_bw makes that a genai coverage gap (parser's n_genai_failed). A FRAMEWORK failure
        # (fw_out None) must NOT be counted as a genai gap -- it is captured by be_gold instead.
        genai_ok = genai_out is not None and genai_bw is not None
        # Equality only when BOTH sides produced comparable output. None = not compared (distinct
        # from a real mismatch); the emitted row pairs it with genai_ok so downstream aggregation
        # never counts "not verified" as "verified equal".
        be_genai = (
            torch.equal(fw_out, genai_out)
            if (fw_out is not None and genai_ok)
            else None
        )
        ok = ok and be_gold and (be_genai is not False)
        if rank == 0:
            rows_out.append(
                (
                    rows,
                    cols,
                    per_peer,
                    nb_genai,
                    nb_nccl,
                    be_gold,
                    fw_lat,
                    fw_bw,
                    fw_nccl_bw,
                    genai_lat,
                    genai_bw,
                    nccl_lat,
                    nccl_bw,
                    genai_ok,
                    be_genai,
                )
            )

    if rank == 0:
        lines = [
            "=== comms/dsl transpose a2a: framework HOOK (zero-copy, SM-matched to genai) vs "
            "genai vs NCCL ===",
            f"{'shape':>12} {'per-peer':>9} {'bitex':>6} {'fw_GB/s':>8} "
            f"{'gn_GB/s':>8} {'nccl_GB/s':>9} {'fw/gn':>6} {'fw/nccl':>8}",
        ]
        for (
            rows,
            cols,
            pp,
            _ng,
            _nn,
            be_gold,
            _fl,
            fb,
            fnb,
            _gl,
            gb,
            _nl,
            nbw,
            _ge_ok,
            be_gn,
        ) in rows_out:
            fg = (fb / gb) if (fb and gb) else 0.0
            fn_nccl = (fnb / nbw) if (fnb and nbw) else 0.0
            # bitex column: gold verdict, then genai verdict (Y/N/na when genai absent).
            bitex = f"{'Y' if be_gold else 'N'}/{'na' if be_gn is None else ('Y' if be_gn else 'N')}"
            lines.append(
                f"{f'{rows}x{cols}':>12} {_fmt_size(pp):>9} {bitex:>6} "
                f"{(fb or 0):>8.1f} {(gb or 0):>8.1f} {(nbw or 0):>9.1f} "
                f"{fg:>5.2f}x {fn_nccl:>7.2f}x"
            )
        table = "\n".join(lines)
        print("\n" + table, flush=True)
        print(
            "\n" + table, file=sys.stderr, flush=True
        )  # torchx keeps stderr, drops stdout
        emit_result_rows(
            [
                {
                    "backend": "cute",
                    "variant": "non_contig_transpose",
                    "fw_variant": "zc",  # shipped path: transpose HOOK + zero-copy transfer
                    "world_size": ws,
                    "shape": f"{rows}x{cols}",
                    "size_bytes": int(pp),
                    "fw_genai_ctas": int(
                        (ng or 0) * ws
                    ),  # framework grid SM-matched to genai
                    "fw_nccl_ctas": int(
                        (nn or 0) * ws
                    ),  # framework grid SM-matched to NCCL
                    # framework vs torch gold (non-negotiable, always meaningful).
                    "bit_exact_fw_eq_gold": bool(be_gold),
                    # whether genai ran & was compared on this shape (False = conda without
                    # genai's deps, or genai raised); the parser counts these as coverage gaps.
                    "genai_ok": bool(ge_ok),
                    # framework vs genai; False when genai was not compared, but genai_ok=False
                    # then marks it not-verified so it is never aggregated as a genai-equal shape.
                    "bit_exact_fw_eq_genai": bool(be_gn)
                    if be_gn is not None
                    else False,
                    "fw_busbw_gbps": float(fb or 0.0),  # HEADLINE = framework (zc)
                    "genai_busbw_gbps": float(gb or 0.0),
                    "nccl_busbw_gbps": float(nbw or 0.0),
                    # HEADLINE ratio: framework transpose HOOK vs genai, framework SM-matched to genai.
                    "ratio_fw_vs_genai": float(fb / gb) if (fb and gb) else 0.0,
                    "ratio_fw_vs_nccl": (
                        float(fnb / nbw) if (fnb and nbw) else 0.0
                    ),  # framework, SM-matched to NCCL
                }
                for (
                    rows,
                    cols,
                    pp,
                    ng,
                    nn,
                    be_gold,
                    _fl,
                    fb,
                    fnb,
                    _gl,
                    gb,
                    _nl,
                    nbw,
                    ge_ok,
                    be_gn,
                ) in rows_out
            ]
        )
    return ok


def _worker(rank: int, ws: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    device = torch.device(f"cuda:{rank}")
    # try/finally so a raising _run still tears the process group down -- otherwise NCCL /
    # rendezvous state leaks and cross-rank cleanup stalls.
    try:
        ok = _run(rank, ws, dist.group.WORLD, device)
        dist.barrier(dist.group.WORLD)
    finally:
        dist.destroy_process_group()
    if not ok:
        sys.exit(1)


def main() -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        ws = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=ws)
        device = torch.device(f"cuda:{local_rank}")
        # try/finally so a raising _run still tears the process group down (see _worker).
        try:
            ok = _run(rank, ws, dist.group.WORLD, device)
            dist.barrier(dist.group.WORLD)
        finally:
            dist.destroy_process_group()
        if not ok:
            sys.exit(1)
        return
    ws = min(torch.cuda.device_count(), 8)
    if ws < 2:
        print("needs >=2 GPUs")
        return
    mp.spawn(_worker, args=(ws, _find_free_port()), nprocs=ws, join=True)


if __name__ == "__main__":
    main()
