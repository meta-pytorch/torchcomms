# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""AnyBench/UniBench parser for the non-contig transpose a2a benchmark (results -> Scuba).

UniBench runs ``benchmark_a2a_non_contig`` on MAST, then invokes this parser via ``bash -c``
with ``$ANYBENCH_LOGS_DIR`` pointing at ``<dir>/rank_<n>/stdout.log`` per rank. The benchmark's
rank 0 prints one ``A2A_RESULT_JSON {<json>}`` line per shape (framework transpose vs genai
``all_to_all_single_non_contig`` vs NCCL, bit-exact-gated) via ``_bench_common.emit_result_rows``.
This scrapes those lines and prints a SINGLE JSON object to stdout; UniBench pushes it to the
``anybench_parser_output`` Scuba dataset (top-level keys become columns). Submit with
``unibench --uni_bench_config comms/dsl/tests/anybench_a2a_non_contig.json``.

Contract: read logs, print exactly one JSON object, exit 0. A parse miss is reported in the
object (never a nonzero exit) so it does not fail the benchmark run.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any

_TAG = "A2A_RESULT_JSON "


def _collect_rows(logs_dir: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for log in sorted(glob.glob(os.path.join(logs_dir, "rank_*", "stdout.log"))):
        try:
            with open(log, "r", errors="replace") as f:
                for line in f:
                    idx = line.find(_TAG)
                    if idx == -1:
                        continue
                    try:
                        obj = json.loads(line[idx + len(_TAG) :].strip())
                    except json.JSONDecodeError:
                        continue
                    # json.loads can return a list/number/string; keep only dict rows so
                    # _summarize's r.get(...) never hits a non-dict (never-fail contract).
                    if isinstance(obj, dict):
                        rows.append(obj)
        except OSError:
            continue
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-shape transpose rows into one AnyBench result object.

    Verdict: the HEADLINE is the framework transpose HOOK (zero-copy transfer) vs genai, framework
    SM-matched to genai (``ratio_fw_vs_genai``). Reports the min ratio, whether EVERY size is
    on-par-or-exceeds genai (>= 0.99), bit-exactness (non-negotiable), per-size arrays + raw rows.
    (The copy-staged variant was removed -- mid-band send-leg-bound; archived off-tree.)

    Genai coverage is tracked explicitly (``n_genai_failed``): a shape where genai did not run
    (conda without its deps, or genai raised) emits ``genai_ok=False`` + ``ratio_fw_vs_genai=0.0``.
    Such shapes are EXCLUDED from the ratio/vs-genai bit-exact aggregates (so a 0.0 sentinel never
    poses as a real datapoint) but COUNTED as coverage gaps: ``all_on_par_or_exceed_genai`` and the
    three-way ``all_bit_exact`` are False whenever any genai shape is missing, so a partial-genai run
    can never report full parity."""
    tr = sorted(
        (r for r in rows if r.get("variant") == "non_contig_transpose"),
        key=lambda r: int(r.get("size_bytes", 0)),
    )
    genai_rows = [r for r in tr if r.get("genai_ok")]
    n_genai_failed = sum(1 for r in tr if not r.get("genai_ok"))
    # `genai_ok` is genai-side coverage (genai ran & was timed), so genai_rows can still include a
    # shape where the FRAMEWORK failed -- that emits ratio_fw_vs_genai==0.0. The truthy filter drops
    # those sentinel zeros (a framework failure is already flagged by all_fw_bit_exact_vs_gold), so
    # the ratio aggregates only see shapes where both sides produced a real measured ratio.
    gn = [
        float(r["ratio_fw_vs_genai"]) for r in genai_rows if r.get("ratio_fw_vs_genai")
    ]
    bit_gold = [bool(r.get("bit_exact_fw_eq_gold")) for r in tr]
    bit_genai = [bool(r.get("bit_exact_fw_eq_genai")) for r in genai_rows]
    all_fw_bit_exact_vs_gold = bool(bit_gold) and all(bit_gold)
    all_fw_bit_exact_vs_genai = bool(bit_genai) and all(bit_genai)
    out: dict[str, Any] = {
        "n_result_rows": len(tr),
        "world_size": int(tr[0]["world_size"]) if tr else 0,
        "fw_variant": str(tr[0].get("fw_variant", "")) if tr else "",
        # framework vs torch gold, non-negotiable, over ALL shapes.
        "all_fw_bit_exact_vs_gold": all_fw_bit_exact_vs_gold,
        # framework vs genai, over shapes where genai actually ran.
        "all_fw_bit_exact_vs_genai": all_fw_bit_exact_vs_genai,
        # genai coverage gap: shapes where genai did not run / was not compared.
        "n_genai_failed": n_genai_failed,
        # strict three-way verdict: fw==gold everywhere AND genai ran & matched on every shape.
        "all_bit_exact": all_fw_bit_exact_vs_gold
        and all_fw_bit_exact_vs_genai
        and n_genai_failed == 0,
        # HEADLINE: framework transpose hook (zero-copy) vs genai, SM-matched to genai. Requires
        # full genai coverage -- a missing-genai shape makes "all on par" unprovable, so it is False.
        "min_ratio_fw_vs_genai": min(gn) if gn else 0.0,
        "all_on_par_or_exceed_genai": n_genai_failed == 0
        and bool(gn)
        and all(x >= 0.99 for x in gn),
        "size_bytes": [int(r.get("size_bytes", 0)) for r in tr],
        "fw_busbw_gbps": [float(r.get("fw_busbw_gbps", 0.0)) for r in tr],
        "genai_busbw_gbps": [float(r.get("genai_busbw_gbps", 0.0)) for r in tr],
        "ratio_fw_vs_genai": [float(r.get("ratio_fw_vs_genai", 0.0)) for r in tr],
        "ratio_fw_vs_nccl": [float(r.get("ratio_fw_vs_nccl", 0.0)) for r in tr],
        # Full per-shape matrix for ad-hoc querying (JSON-encoded column).
        "rows_json": json.dumps(tr, sort_keys=True),
    }
    return out


def main() -> None:
    # Honor the never-fail contract end-to-end: a malformed row value (wrong type / non-numeric)
    # could still raise inside _summarize, so wrap it and always emit ONE JSON object + exit 0.
    logs_dir = os.environ.get("ANYBENCH_LOGS_DIR", "")
    try:
        rows = _collect_rows(logs_dir) if logs_dir else []
        result = _summarize(rows)
    except Exception as e:  # noqa: BLE001 -- parser must never fail the benchmark run
        result = {"parse_note": f"summarize error: {type(e).__name__}: {e}"[:900]}
    if not logs_dir:
        result["parse_note"] = "ANYBENCH_LOGS_DIR unset"
    print(json.dumps(result, sort_keys=True))
    sys.exit(0)


if __name__ == "__main__":
    main()
