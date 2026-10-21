# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Prepare one farm comms wheel bump from a successful build manifest."""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

PACKAGES = "genai/msl_infra/dev/wheels/packages.py"
PROJECT = "genai/msl/farm/pyproject.toml"
LOCK = "genai/msl/farm/uv.lock"
FILES = (PACKAGES, PROJECT, LOCK)
AUTOMATION = "Comms wheel bump automation: hpc_comms/ncclx_v2_30_latest"
VARIANTS = {"gbunified", "h100"}
OWNERS = {
    "meta-ncclx-cu13": "ncclx",
    "torchcomms": "torchcomms",
    "torchcomms-mccl": "mccl",
    "commsutils": "commsutils",
    "cca4py": "torchcomms",
    "nccl4py": "ncclx",
}
VERSION = re.compile(
    r"(?P<base>\d+\.\d+\.\d+)\.(?P<stamp>\d{4}\.\d+\.\d+\.\d+)\+(?P<hash>[0-9a-f]{10})"
)


def version_parts(version: str) -> re.Match[str]:
    match = VERSION.fullmatch(version)
    if not match:
        raise ValueError(f"Invalid comms wheel version: {version}")
    return match


def source_hashes(source: str) -> dict[str, str]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_COMMS_SRC_HASH":
                if not isinstance(node.value, ast.Dict):
                    raise ValueError("_COMMS_SRC_HASH must be a literal dictionary")
                hashes = ast.literal_eval(node.value)
                if set(hashes) != VARIANTS or not all(
                    isinstance(h, str) and re.fullmatch(r"[0-9a-f]{40}", h)
                    for h in hashes.values()
                ):
                    raise ValueError("Unexpected _COMMS_SRC_HASH entries")
                return hashes
    raise ValueError("Missing _COMMS_SRC_HASH")


def comms_pins(project: dict) -> dict[str, str]:
    members = project["tool"]["env-checks"]["stamp-groups"]["comms"]
    if set(members) != set(OWNERS):
        raise ValueError("Comms stamp group changed; update wheel ownership mapping")
    if project["tool"]["env-checks"].get("stamp-group-exceptions"):
        raise ValueError("Resolve stamp-group exceptions before an automated bump")
    pins = {}
    for requirement in project["project"]["dependencies"]:
        name, separator, version = requirement.partition("==")
        if name in members and separator:
            if name in pins or not VERSION.fullmatch(version):
                raise ValueError(f"Expected one exact dated source pin for {name}")
            pins[name] = version
    if set(pins) != set(members):
        raise ValueError("Every comms stamp-group member must have one exact pin")
    return pins


def derive_pins(
    manifest: dict, revision: str, current: dict[str, str]
) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("fbsource hash must be 40 lowercase hexadecimal characters")
    if manifest.get("state") != "passed":
        raise ValueError("Only a passed wheel build can produce a bump")
    if manifest.get("registry") != "msl-shared":
        raise ValueError("Farm wheel pins require the msl-shared registry")
    if not re.fullmatch(r"\d{4}\.\d+\.\d+\.\d+", manifest.get("segment", "")):
        raise ValueError("Missing or invalid manifest segment")
    number = manifest.get("number")
    if (
        type(number) is not int
        or number < 1
        or manifest.get("web_url")
        != (f"https://buildkite.com/msl/msl-wheel-adhoc/builds/{number}")
    ):
        raise ValueError("Missing or inconsistent Buildkite provenance")
    pins = {}
    for name, owner in OWNERS.items():
        versions = manifest.get("wheels", {}).get(
            name
            if name in {"cca4py", "nccl4py"} and name in manifest.get("wheels", {})
            else owner,
            {},
        )
        if set(versions) != VARIANTS or len(set(versions.values())) != 1:
            raise ValueError(
                f"Missing or inconsistent architecture versions for {name}"
            )
        version = versions["h100"]
        match = VERSION.fullmatch(version)
        if (
            not match
            or match["stamp"] != manifest["segment"]
            or match["hash"] != revision[:10]
        ):
            raise ValueError(f"{name} does not match the build stamp and source hash")
        if name in {"cca4py", "nccl4py"} and name not in manifest["wheels"]:
            # Companion distributions share their owner's stamp and source hash.
            # The resolver and env checker verify the derived version exists.
            base = version_parts(current[name])["base"]
            version = f"{base}.{match['stamp']}+{match['hash']}"
        pins[name] = version
    return pins


def update_sources(
    source: str, project: str, revision: str, pins: dict[str, str]
) -> tuple[str, str]:
    lines = source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_COMMS_SRC_HASH"
        ):
            if not isinstance(node.value, ast.Dict):
                raise ValueError("_COMMS_SRC_HASH must be a literal dictionary")
            for value in node.value.values:
                line = lines[value.lineno - 1]
                lines[value.lineno - 1] = (
                    line[: value.col_offset]
                    + json.dumps(revision)
                    + line[value.end_col_offset :]
                )
            break
    current = comms_pins(tomllib.loads(project))
    for name, version in pins.items():
        old = f'"{name}=={current[name]}"'
        if project.count(old) != 1:
            raise ValueError(f"Cannot uniquely replace {name}")
        project = project.replace(old, f'"{name}=={version}"')
    return "".join(lines), project


def normalize_lock(value: object, pins: dict[str, str]) -> object:
    if isinstance(value, list):
        return [normalize_lock(item, pins) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: normalize_lock(item, pins) for key, item in value.items()}
    if value.get("name") in pins:
        # A package's artifacts and references change with its version. Its
        # dependency names, markers, and source registry must remain stable.
        version = pins[value["name"]]
        if result.get("version") == version:
            result["version"] = "<comms bump>"
        if result.get("specifier") == f"=={version}":
            result["specifier"] = "<comms bump>"
        for key in ("wheels", "sdist"):
            result.pop(key, None)
    return result


def audit_lock(before: str, after: str, pins: dict[str, str]) -> None:
    old, new = tomllib.loads(before), tomllib.loads(after)
    old_pins = {p["name"]: p["version"] for p in old["package"] if p["name"] in pins}
    if normalize_lock(old, old_pins) != normalize_lock(new, pins):
        raise ValueError(
            "Lock changed versions, edges, sources, or metadata beyond the comms bump"
        )
    for name, version in pins.items():
        entries = [p for p in new["package"] if p["name"] == name]
        if len(entries) != 1 or entries[0]["version"] != version:
            raise ValueError(f"Lock does not resolve {name} to {version}")
        urls = [wheel["url"] for wheel in entries[0].get("wheels", [])]
        if not all(any(arch in url for url in urls) for arch in ("aarch64", "x86_64")):
            raise ValueError(f"Lock is missing an architecture for {name}")


def sl(root: Path, command: str, *args: str) -> str:
    subprocess.run(
        ["sl", "help", command], cwd=root, check=True, stdout=subprocess.DEVNULL
    )
    return subprocess.check_output(
        [
            "sl",
            command,
            *args,
            "--reason",
            f"prepare comms wheel bump - sl help {command}",
        ],
        cwd=root,
        text=True,
    ).strip()


def check_revision(
    root: Path, revision: str, hashes: dict[str, str], current: dict[str, str]
) -> None:
    previous = set(hashes.values())
    for prefix in {version_parts(pin)["hash"] for pin in current.values()}:
        previous.add(sl(root, "log", "-r", prefix, "-T", "{node}"))
    for old in previous:
        if sl(root, "log", "-r", f"ancestor({old}, {revision})", "-T", "{node}") != old:
            raise ValueError(
                f"Refusing source downgrade or unrelated revision: {old} -> {revision}"
            )


def check_farm_lock(root: Path) -> None:
    path = root / "genai/msl/.buildkite/uv_lock_check.py"
    spec = importlib.util.spec_from_file_location("uv_lock_check", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load farm uv_lock_check")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.check_projects(["farm"]) != 0:
        raise ValueError("Farm uv_lock_check failed")


def env_warnings(root: Path) -> set[str]:
    result = subprocess.run(
        [sys.executable, "tools/env_checks/check_env_pins.py"],
        cwd=root / "genai/msl/farm",
        text=True,
        capture_output=True,
    )
    logger.info("%s%s", result.stdout, result.stderr)
    result.check_returncode()
    return {
        VERSION.sub("<comms-version>", line)
        for line in result.stderr.splitlines()
        if line.startswith("WARN: ")
    }


def prepare(
    root: Path, manifest: dict, revision: str, output: Path, dry_run: bool
) -> None:
    subprocess.run(
        ["sl", "help", "agent"], cwd=root, check=True, stdout=subprocess.DEVNULL
    )
    if sl(root, "status", "-mardu"):
        raise ValueError("Wheel bump requires a clean checkout")
    if sl(root, "log", "-r", ".", "-T", "{phase}") != "public":
        raise ValueError(
            "Wheel bump must start on a landed commit to avoid a stacked diff"
        )
    source = (root / PACKAGES).read_text()
    project = (root / PROJECT).read_text()
    current = comms_pins(tomllib.loads(project))
    pins = derive_pins(manifest, revision, current)
    hashes = source_hashes(source)
    check_revision(root, revision, hashes, current)
    source_new, project_new = update_sources(source, project, revision, pins)
    if source == source_new and project == project_new:
        raise ValueError("This wheel cohort is already pinned")
    output.mkdir(parents=True, exist_ok=True)
    originals = {path: (root / path).read_text() for path in FILES}
    complete = False
    try:
        apply_and_validate(root, source_new, project_new, originals[LOCK], pins)
        write_message(root, hashes, revision, manifest, output)
        write_patch(root, originals, output)
        complete = True
    finally:
        if dry_run or not complete:
            for path, content in originals.items():
                (root / path).write_text(content)


def apply_and_validate(
    root: Path, source: str, project: str, lock: str, pins: dict[str, str]
) -> None:
    warnings_before = env_warnings(root)
    (root / PACKAGES).write_text(source)
    (root / PROJECT).write_text(project)
    env = dict(os.environ, UV_FROZEN="0")
    args = ["uv", "lock"]
    for name in pins:
        args.extend(["--upgrade-package", name])
    subprocess.run(args, cwd=root / "genai/msl/farm", env=env, check=True)
    audit_lock(lock, (root / LOCK).read_text(), pins)
    run_checks(root)
    if env_warnings(root) != warnings_before:
        raise ValueError(
            "Environment warning set changed; review the new pairing before posting"
        )
    changed = set(sl(root, "status", "-n", "-mardu").splitlines())
    if changed - set(FILES):
        raise ValueError(f"Unexpected files changed: {sorted(changed - set(FILES))}")


def write_patch(root: Path, originals: dict[str, str], output: Path) -> None:
    patch = "".join(
        "".join(
            difflib.unified_diff(
                old.splitlines(True),
                (root / path).read_text().splitlines(True),
                fromfile=path,
                tofile=path,
            )
        )
        for path, old in originals.items()
    )
    (output / "bump.patch").write_text(patch)
    logger.info("Combined patch written to %s", output / "bump.patch")


def run_checks(root: Path) -> None:
    wheels = root / "genai/msl_infra/dev/wheels"
    for extra in (["-k", "comms or ncclx or mccl"], []):
        subprocess.run(
            [
                "uv",
                "run",
                "--group",
                "dev",
                "python",
                "-m",
                "pytest",
                "test_build.py",
                *extra,
            ],
            cwd=wheels,
            check=True,
        )
    subprocess.run(["arc", "lint", "-a", *FILES], cwd=root, check=True)
    check_farm_lock(root)


def write_message(
    root: Path, hashes: dict[str, str], revision: str, manifest: dict, output: Path
) -> None:
    previous = sorted(set(hashes.values()))
    changes = sl(
        root,
        "log",
        "-r",
        f"{previous[0]}::{revision} - {previous[0]}",
        "-l",
        "50",
        "-T",
        "{phabdiff} {desc|firstline}\\n",
        "fbcode/comms/mccl",
        "fbcode/comms/ncclx",
    )
    message = f"""[farm][wheels] Bump comms wheels to {revision[:10]}

Summary:

Context: Promote the passed comms wheel cohort from {manifest["web_url"]} (build {manifest["number"]}, stamp `{manifest["segment"]}`, registry `{manifest["registry"]}`).

{AUTOMATION}

Motivation: Keep farm wheel pins and wheel source defaults at the same revision.

This diff: Update both `_COMMS_SRC_HASH` variants to `{revision}`, all six farm comms pins, and only farm's `uv.lock`. Previous source revisions: {", ".join(previous)}.

MCCL/NCCLX changes included (at most 50):
{changes or "(no MCCL/NCCLX changes)"}

Test Plan:

Passed `test_build.py` comms subset and full file; `arc lint -a`; `check_env_pins.py` before and after (same warning set); farm-only `uv_lock_check.py`; lock audit (no unrelated version, edge, source, or metadata movement).

Deterministic and nightly validation are performed separately.
"""
    (output / "commit_message.txt").write_text(message)
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "automation": AUTOMATION,
                "fbsource_hash": revision,
                "previous_hashes": previous,
                "manifest": manifest,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fbsource-root", type=Path, required=True)
    parser.add_argument("--fbsource-hash", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepare(
        args.fbsource_root.resolve(),
        json.loads(args.manifest.read_text()),
        args.fbsource_hash,
        args.output_dir.resolve(),
        args.dry_run,
    )


if __name__ == "__main__":
    main()
