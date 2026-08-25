#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Drop build matrix coordinates whose torch wheel has not been published yet.

The Nova build matrix comes from pytorch/test-infra@main, so a new CUDA version
lands here as soon as it is added upstream -- which can be days before the
matching torch nightly reaches download.pytorch.org. Building against a torch
that does not exist fails every affected coordinate, so this drops them until
the wheels show up.

Coordinates are only dropped when the index is reachable AND lists no matching
wheel. Any error keeps the coordinate so the build fails loudly rather than the
matrix silently shrinking.

Reads the matrix from the MAT environment variable, writes the filtered matrix
to stdout, and logs every dropped coordinate to stderr and the job summary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

INDEX_HOST = "https://download.pytorch.org/whl"
FETCH_TIMEOUT_SECONDS = 30
FETCH_ATTEMPTS = 3
WHEEL_RE = re.compile(r"torch-[^\"#<>\s]*\.whl")


def cpython_tag(python_version: str) -> str:
    """Map a matrix python_version onto its wheel tag: 3.14t -> cp314-cp314t."""
    free_threaded = python_version.endswith("t")
    base = python_version[:-1] if free_threaded else python_version
    nodots = base.replace(".", "")
    suffix = f"cp{nodots}t" if free_threaded else f"cp{nodots}"
    return f"cp{nodots}-{suffix}"


def index_url(channel: str, desired_cuda: str) -> str:
    # The release channel lives at the index root; every other channel is nested.
    prefix = INDEX_HOST if channel == "release" else f"{INDEX_HOST}/{channel}"
    return f"{prefix}/{desired_cuda}/torch/"


def fetch_index(url: str) -> str | None:
    """Return the listing, or None when the index holds nothing for this CUDA.

    The index is S3-backed, so a prefix that was never written answers 403
    rather than 404 -- both mean "no wheels published". Anything else is
    inconclusive and raises, so the caller can keep the coordinate.
    """
    last_error: Exception | None = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_SECONDS) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                return None
            last_error = exc
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
        if attempt + 1 < FETCH_ATTEMPTS:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last_error}")


def published_wheels(page: str) -> list[str]:
    # The listing carries both percent-encoded hrefs and decoded link text.
    return WHEEL_RE.findall(page.replace("%2B", "+"))


def has_wheel(wheels: list[str], python_version: str, architecture: str) -> bool:
    tag = f"-{cpython_tag(python_version)}-"
    suffix = f"_{architecture}.whl"
    return any(tag in wheel and wheel.endswith(suffix) for wheel in wheels)


def emit_summary(dropped: list[str], warnings: list[str]) -> None:
    for line in warnings:
        print(f"WARNING: {line}", file=sys.stderr)
    for line in dropped:
        print(f"DROPPED: {line}", file=sys.stderr)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = ["## Torch wheel availability filter", ""]
    if dropped:
        lines.append("Dropped (no published torch wheel):")
        lines.extend(f"- `{item}`" for item in dropped)
    else:
        lines.append("No coordinates dropped.")
    if warnings:
        lines.extend(["", "Kept despite check failure:"])
        lines.extend(f"- `{item}`" for item in warnings)
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def filter_matrix(matrix: dict, architecture: str) -> dict:
    cache: dict[str, list[str] | None] = {}
    kept, dropped, warnings = [], [], []

    for coordinate in matrix.get("include", []):
        gpu_arch_type = coordinate.get("gpu_arch_type", "")
        desired_cuda = coordinate.get("desired_cuda")
        python_version = coordinate.get("python_version")

        if not gpu_arch_type.startswith("cuda") or not desired_cuda:
            kept.append(coordinate)
            continue
        if not python_version:
            warnings.append(f"{desired_cuda} (no python_version in coordinate)")
            kept.append(coordinate)
            continue

        url = index_url(coordinate.get("channel", "nightly"), desired_cuda)
        if url not in cache:
            try:
                page = fetch_index(url)
                cache[url] = [] if page is None else published_wheels(page)
                if page is None:
                    print(f"NOTE: {url} holds no wheels yet", file=sys.stderr)
            except RuntimeError as exc:
                cache[url] = None
                warnings.append(str(exc))

        wheels = cache[url]
        if wheels is None:
            kept.append(coordinate)
        elif has_wheel(wheels, python_version, architecture):
            kept.append(coordinate)
        else:
            dropped.append(f"{desired_cuda} {python_version} {architecture}")

    emit_summary(dropped, warnings)
    return {"include": kept}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        default="x86_64",
        choices=["x86_64", "aarch64"],
        help="Wheel platform architecture the matrix is being built for.",
    )
    args = parser.parse_args()

    matrix = json.loads(os.environ["MAT"])
    print(json.dumps(filter_matrix(matrix, args.architecture)))


if __name__ == "__main__":
    main()
