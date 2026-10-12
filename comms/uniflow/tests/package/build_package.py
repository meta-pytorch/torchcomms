# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

import argparse
import os
from pathlib import Path

from scikit_build_core.build import build_sdist, build_wheel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a UniFlow package without pip.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=("sdist", "wheel"), default="wheel")
    parser.add_argument("-C", "--setting", action="append", default=[])
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings: dict[str, str | list[str]] = {}
    for setting in args.setting:
        key, separator, value = setting.partition("=")
        if not separator:
            parser.error("Each setting must have the form key=value.")
        current = settings.get(key)
        if current is None:
            settings[key] = value
        elif isinstance(current, list):
            current.append(value)
        else:
            settings[key] = [current, value]

    builders = {"sdist": build_sdist, "wheel": build_wheel}
    previous_directory = Path.cwd()
    try:
        os.chdir(source)
        builders[args.format](str(output), config_settings=settings)
    finally:
        os.chdir(previous_directory)
