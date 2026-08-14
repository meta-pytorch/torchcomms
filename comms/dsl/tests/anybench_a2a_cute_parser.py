# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""AnyBench parser for the CuTe a2a (copy) benchmark -- the UniBench result sink.

AnyBench runs ``benchmark_a2a_cute`` on MAST, then invokes this parser via ``bash -c`` with
``$ANYBENCH_LOGS_DIR`` pointing at ``<dir>/rank_<n>/stdout.log`` per rank. The benchmark's
rank 0 prints one ``A2A_RESULT_JSON {<json>}`` line per size via
``comms.dsl.tests._bench_common.emit_result_rows``; this parser scrapes those lines and
prints a SINGLE JSON object to stdout, which AnyBench pushes to the ``anybench_parser_output``
Scuba dataset (top-level keys become columns).

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
    """Every ``A2A_RESULT_JSON`` row across all per-rank stdout logs (rank 0 emits them)."""
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
    """Aggregate per-size copy rows into one AnyBench result object.

    Emits the perf verdict (min busbw ratio vs NCCL, whether every size cleared NCCL) plus
    per-size arrays and the raw rows (JSON-encoded)."""
    perf = [r for r in rows if r.get("backend") == "cute" and "ratio" in r]
    errors = [r for r in rows if r.get("backend") == "cute_error"]
    copy_rows = sorted(
        (r for r in perf if r.get("variant") == "copy"),
        key=lambda r: int(r.get("size_bytes", 0)),
    )
    # Include ALL copy rows (not `if r.get("ratio")`): a ratio of 0.0 is the sentinel the
    # benchmark emits when NCCL timing failed/was unavailable for that size, so a truthy
    # filter would silently drop the failed sizes and let min_ratio / all_clear_nccl
    # over-report. Keep the 0.0s (they sink min_ratio and fail the >=1.0 check) and surface
    # the count as a coverage gap.
    ratios = [float(r["ratio"]) for r in copy_rows]
    n_nccl_failed = sum(1 for x in ratios if x <= 0.0)
    out: dict[str, Any] = {
        "n_result_rows": len(copy_rows),
        "n_errors": len(errors),
        "n_nccl_failed": n_nccl_failed,
        # world_size from the first copy row: one AnyBench job == one benchmark invocation at
        # a single world_size, so every row agrees.
        "world_size": int(copy_rows[0].get("world_size", 0)) if copy_rows else 0,
        "min_ratio_busbw": min(ratios) if ratios else 0.0,
        "all_clear_nccl": bool(ratios) and all(x >= 1.0 for x in ratios),
        "size_bytes": [int(r.get("size_bytes", 0)) for r in copy_rows],
        "cute_us": [float(r.get("cute_us", 0.0)) for r in copy_rows],
        "nccl_us": [float(r.get("nccl_us", 0.0)) for r in copy_rows],
        "cute_busbw_gbps": [float(r.get("cute_busbw_gbps", 0.0)) for r in copy_rows],
        "nccl_busbw_gbps": [float(r.get("nccl_busbw_gbps", 0.0)) for r in copy_rows],
        "ratio": [float(r.get("ratio", 0.0)) for r in copy_rows],
        # Full per-size matrix for ad-hoc querying (JSON-encoded column).
        "rows_json": json.dumps(perf, sort_keys=True),
    }
    if errors:
        out["first_error"] = str(errors[0].get("error", ""))[:900]
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
