# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Artifact writers for Triton comm tuning results."""

from __future__ import annotations

import json
from pathlib import Path

from comm_tuning.adapter import CommKernelTuningAdapter


def write_best_configs_json(output_dir: Path, best_configs: dict) -> Path:
    path = output_dir / "best_configs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(best_configs, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_generated_python(
    *,
    adapter: CommKernelTuningAdapter,
    output_dir: Path,
    best_configs: dict,
) -> Path:
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    path = generated_dir / adapter.generated_filename()

    lines = [
        "# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.",
        "",
        '"""Generated tuned configs for Triton communication kernels.',
        "",
        "Generated from `best_configs.json` by the comm tuning framework.",
        "Do not edit entries manually; regenerate from autotune results instead.",
        '"""',
        "",
        adapter.generated_imports().rstrip(),
        "",
        "",
        f"{adapter.generated_table_name()} = {{",
    ]
    for result in best_configs["results"]:
        if not adapter.result_is_artifact_eligible(result):
            continue
        comment = adapter.render_result_comment(result)
        if comment:
            lines.append(f"    # {comment}")
        key_expr = adapter.render_key(result["key_type"], result["key"])
        config_expr = adapter.render_config(
            result["best_config_type"],
            result["best_config"],
        )
        lines.append(f"    {key_expr}: {config_expr},")
    lines.extend(["}", ""])

    with path.open("w") as f:
        f.write("\n".join(lines))
    return path
