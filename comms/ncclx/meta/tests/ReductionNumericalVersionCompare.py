#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Compare reduction numerical outputs produced by two NCCLX versions.

from __future__ import annotations

import argparse
import re
import shlex
import struct
import subprocess
import sys
from pathlib import Path

OUTPUT_PREFIX = "REDUCTION_NUMERICAL_ACTUAL "
MAX_FAILURE_LINES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run versioned reduction numerical binaries and compare "
        "their actual NCCL outputs bitwise."
    )
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--old-version", required=True)
    parser.add_argument("--new-version", required=True)
    parser.add_argument("--ppn", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--gtest-filter")
    parser.add_argument(
        "--allow-numerical-test-failures",
        action="store_true",
        help=(
            "Compare structured NCCL outputs even when the underlying "
            "FP64-reference numerical test fails. The binary must still produce "
            "structured outputs and show gtest failures; launcher failures and "
            "timeouts remain hard failures."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("COLLECTIVE", "OLD_BINARY", "NEW_BINARY"),
        required=True,
        help="Collective name and the two versioned binaries to compare.",
    )
    return parser.parse_args()


def run_binary(
    launcher: str,
    binary: str,
    ppn: int,
    timeout_seconds: int,
    gtest_filter: str | None,
    allow_numerical_test_failures: bool,
) -> dict[str, str]:
    launcher_path = str(Path(launcher).resolve())
    binary_path = str(Path(binary).resolve())
    binary_command = [binary_path]
    if gtest_filter:
        binary_command.append(f"--gtest_filter={gtest_filter}")

    cmd = [
        launcher_path,
        "--ppn",
        str(ppn),
        "--env",
        "REDUCTION_NUMERICAL_PRINT_ACTUAL=1",
        "--env",
        f"COMMS_TEST_TCPSTORE_TIMEOUT_SECONDS={timeout_seconds}",
        "--",
        "timeout",
        str(timeout_seconds),
        "bash",
        "-c",
        shlex.join(binary_command),
    ]
    print(f"Running {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    outputs = parse_actual_outputs(result.stdout)
    if result.returncode != 0:
        if allow_numerical_test_failures and is_numerical_test_failure(
            result.returncode, result.stdout, outputs
        ):
            print(
                "FP64_REFERENCE_STATUS "
                f"binary={binary_path} status=FAIL "
                f"returncode={result.returncode} structured_outputs={len(outputs)}"
            )
            print_numerical_failure_summary(result.stdout)
            return outputs
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        result.check_returncode()
    if not outputs:
        raise RuntimeError(f"{binary_path} produced no structured numerical outputs")
    print(
        "FP64_REFERENCE_STATUS "
        f"binary={binary_path} status=PASS structured_outputs={len(outputs)}"
    )
    return outputs


REQUIRED_OUTPUT_FIELDS = ("collective", "case", "rank", "bytes")


def parse_actual_outputs(stdout: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for line in stdout.splitlines():
        if not line.startswith(OUTPUT_PREFIX):
            continue
        fields: dict[str, str] = {}
        for token in line[len(OUTPUT_PREFIX) :].split():
            if "=" not in token:
                raise ValueError(
                    f"malformed structured output token {token!r} in line: {line}"
                )
            name, value = token.split("=", 1)
            fields[name] = value
        missing = [name for name in REQUIRED_OUTPUT_FIELDS if name not in fields]
        if missing:
            raise ValueError(f"structured output line missing fields {missing}: {line}")
        key = f"{fields['collective']}__{fields['case']}__rank_{fields['rank']}"
        if key in outputs:
            raise RuntimeError(f"duplicate numerical output record: {key}")
        outputs[key] = fields["bytes"]
    return outputs


def is_numerical_test_failure(
    returncode: int, stdout: str, outputs: dict[str, str]
) -> bool:
    timeout_markers = (
        "exit code: 124",
        "timed out",
        "Timed out",
    )
    if returncode == 124 or any(marker in stdout for marker in timeout_markers):
        return False
    # A numerical failure means the binary ran the reduction path (it emitted
    # structured outputs) and gtest reported a failing assertion. Match the gtest
    # marker with a whitespace-tolerant regex so the classification does not
    # silently break if gtest changes the marker spacing.
    gtest_failed = re.search(r"\[\s*FAILED\s*\]", stdout) is not None
    return bool(outputs) and gtest_failed


def print_numerical_failure_summary(stdout: str) -> None:
    printed = 0
    for line in stdout.splitlines():
        if " reference=" not in line or " actual=" not in line:
            continue
        print(f"FP64_REFERENCE_FAILURE {line}")
        printed += 1
        if printed >= MAX_FAILURE_LINES:
            print("FP64_REFERENCE_FAILURE ... truncated")
            return


def bfloat16_to_float(raw: bytes) -> float:
    bits = struct.unpack("<H", raw)[0] << 16
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def element_size(output_name: str) -> int:
    if "_Float32_" in output_name:
        return 4
    if "_Bfloat16_" in output_name:
        return 2
    return 1


def format_value(output_name: str, data: bytes, offset: int) -> str:
    # offset must be aligned to the element size so the formatted value
    # corresponds to a single element rather than bytes spanning two elements.
    if "_Float32_" in output_name and offset + 4 <= len(data):
        value = struct.unpack("<f", data[offset : offset + 4])[0]
        return f"{value:.9g}"
    if "_Bfloat16_" in output_name and offset + 2 <= len(data):
        raw = data[offset : offset + 2]
        return f"{bfloat16_to_float(raw):.9g} (bf16=0x{raw.hex()})"
    return f"0x{data[offset : offset + 8].hex()}"


def element_index(output_name: str, offset: int) -> int:
    return offset // element_size(output_name)


def compare_output(
    old_version: str, new_version: str, output_name: str, old_hex: str, new_hex: str
) -> bool:
    old_data = bytes.fromhex(old_hex)
    new_data = bytes.fromhex(new_hex)
    if old_data == new_data:
        return True

    if len(old_data) != len(new_data):
        print(
            f"BITWISE MISMATCH {output_name}: length differs "
            f"{old_version}={len(old_data)} bytes {new_version}={len(new_data)} bytes"
        )
        return False

    mismatch_offset = next(
        i
        for i, (old_byte, new_byte) in enumerate(zip(old_data, new_data))
        if old_byte != new_byte
    )
    # Align to the element boundary so the formatted values correspond to the
    # reported element_index; byte_offset still points at the differing byte.
    index = element_index(output_name, mismatch_offset)
    aligned_offset = index * element_size(output_name)
    print(
        "BITWISE MISMATCH "
        f"{output_name}: byte_offset={mismatch_offset} "
        f"element_index={index} "
        f"{old_version}={format_value(output_name, old_data, aligned_offset)} "
        f"{new_version}={format_value(output_name, new_data, aligned_offset)}"
    )
    return False


def compare_outputs(
    old_version: str,
    new_version: str,
    old_outputs: dict[str, str],
    new_outputs: dict[str, str],
) -> bool:
    passed = True

    missing = sorted(old_outputs.keys() - new_outputs.keys())
    extra = sorted(new_outputs.keys() - old_outputs.keys())
    for name in missing:
        print(f"MISSING {new_version} output: {name}")
        passed = False
    for name in extra:
        print(f"EXTRA {new_version} output: {name}")
        passed = False

    for name in sorted(old_outputs.keys() & new_outputs.keys()):
        passed = (
            compare_output(
                old_version, new_version, name, old_outputs[name], new_outputs[name]
            )
            and passed
        )
    return passed


def main() -> int:
    args = parse_args()
    all_passed = True
    for collective, old_binary, new_binary in args.case:
        print(
            f"Comparing {collective}: {args.old_version} vs {args.new_version}",
            flush=True,
        )
        old_outputs = run_binary(
            args.launcher,
            old_binary,
            args.ppn,
            args.timeout_seconds,
            args.gtest_filter,
            args.allow_numerical_test_failures,
        )
        new_outputs = run_binary(
            args.launcher,
            new_binary,
            args.ppn,
            args.timeout_seconds,
            args.gtest_filter,
            args.allow_numerical_test_failures,
        )
        comparison_passed = compare_outputs(
            args.old_version, args.new_version, old_outputs, new_outputs
        )
        print(
            "RELEASE_COMPARE_STATUS "
            f"collective={collective} "
            f"old_version={args.old_version} "
            f"new_version={args.new_version} "
            f"status={'PASS' if comparison_passed else 'FAIL'}"
        )
        all_passed = comparison_passed and all_passed
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
