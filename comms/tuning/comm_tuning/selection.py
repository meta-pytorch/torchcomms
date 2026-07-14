# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Select best Triton comm tuning configs from raw JSONL results."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from comm_tuning.adapter import CommKernelTuningAdapter


def _ws_cat(path: str) -> str:
    return subprocess.check_output(["ws", "cat", path], text=True)


def _ws_ls(path: str) -> list[str]:
    output = subprocess.check_output(["ws", "ls", path], text=True)
    entries = []
    for line in output.splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        entries.extend(field for field in fields if field.startswith("ws://"))
    return entries


def _join_uri(base: str, child: str) -> str:
    return base.rstrip("/") + "/" + child.lstrip("/")


def _iter_result_lines(input_dir: str) -> Iterable[str]:
    if input_dir.startswith("ws://"):
        results_dir = _join_uri(input_dir, "results")
        for entry in _ws_ls(results_dir):
            name = entry.rstrip("/").split("/")[-1]
            if name.startswith("rank_") and name.endswith(".jsonl"):
                yield from _ws_cat(_join_uri(results_dir, name)).splitlines()
        return

    results_dir = Path(input_dir) / "results"
    for path in sorted(results_dir.glob("rank_*.jsonl")):
        with path.open() as f:
            yield from f


def load_result_records(input_dir: str) -> list[dict[str, Any]]:
    records = []
    for line in _iter_result_lines(input_dir):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _key_group(record: dict[str, Any]) -> tuple[str, str]:
    return record["key_type"], json.dumps(record["key"], sort_keys=True)


def _is_valid_candidate(record: dict[str, Any]) -> bool:
    correctness = record.get("correctness") or {}
    perf = record.get("perf") or {}
    return (
        record.get("kind") == "candidate"
        and record.get("status") == "ok"
        and correctness.get("status") == "pass"
        and perf.get("latency_us") is not None
    )


def _candidate_sort_key(record: dict[str, Any]) -> tuple[float, str, int]:
    config_json = json.dumps(record.get("config"), sort_keys=True)
    return (
        float(record["perf"]["latency_us"]),
        config_json,
        int(record["work_item_id"]),
    )


def select_best_configs(
    *,
    adapter: CommKernelTuningAdapter,
    records: list[dict[str, Any]],
    source_rank: int = 0,
) -> dict[str, Any]:
    rank_records = [
        record
        for record in records
        if record.get("rank") == source_rank
        and record.get("record_type") == "work_item_result"
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in rank_records:
        groups[_key_group(record)].append(record)

    results = []
    for _group_key, group_records in sorted(groups.items()):
        candidate_records = [
            record for record in group_records if record.get("kind") == "candidate"
        ]
        valid_candidates = [
            record for record in candidate_records if _is_valid_candidate(record)
        ]
        if not valid_candidates:
            continue

        best_overall = min(valid_candidates, key=_candidate_sort_key)
        artifact_candidates = [
            record
            for record in valid_candidates
            if adapter.result_is_artifact_eligible(record)
        ]
        if not artifact_candidates:
            continue

        best = min(artifact_candidates, key=_candidate_sort_key)
        baselines = {}
        for record in group_records:
            if record.get("kind") != "baseline":
                continue
            perf = record.get("perf") or {}
            baselines[str(record.get("baseline"))] = {
                "work_item_id": record.get("work_item_id"),
                "latency_us": perf.get("latency_us"),
                "status": record.get("status"),
            }

        failed_count = sum(
            1
            for record in candidate_records
            if record.get("status") != "ok"
            or (record.get("correctness") or {}).get("status") != "pass"
        )
        results.append(
            {
                "spec_type": best["spec_type"],
                "spec": best["spec"],
                "key_type": best["key_type"],
                "key": best["key"],
                "best_work_item_id": best["work_item_id"],
                "best_config_type": best["config_type"],
                "best_config": best["config"],
                "best_latency_us": best["perf"]["latency_us"],
                "best_overall_work_item_id": best_overall["work_item_id"],
                "best_overall_config_type": best_overall["config_type"],
                "best_overall_config": best_overall["config"],
                "best_overall_latency_us": best_overall["perf"]["latency_us"],
                "best_overall_artifact_eligible": adapter.result_is_artifact_eligible(
                    best_overall
                ),
                "baselines": baselines,
                "candidate_count": len(candidate_records),
                "valid_candidate_count": len(valid_candidates),
                "artifact_eligible_candidate_count": len(artifact_candidates),
                "failed_candidate_count": failed_count,
            }
        )

    return {
        "schema_version": 1,
        "adapter": adapter.name,
        "selection_policy": {
            "name": "lowest_latency_us_artifact_eligible",
            "source_rank": source_rank,
            "requires_correctness": True,
            "records_best_overall_candidate": True,
        },
        "results": results,
    }
