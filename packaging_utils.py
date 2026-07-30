# Copyright (c) Meta Platforms, Inc. and affiliates.

import re


def parse_readelf_dynamic(output: str) -> dict[str, tuple[str, ...]]:
    entries: dict[str, list[str]] = {}
    for line in output.splitlines():
        match = re.search(r"\(([^)]+)\).*?\[([^\n]*)\]\s*$", line)
        if match is not None:
            entries.setdefault(match.group(1), []).append(match.group(2))
    for name in ("RPATH", "RUNPATH"):
        observed = len(
            re.findall(rf"^\s*0x[0-9a-fA-F]+\s+\({name}\)", output, re.MULTILINE)
        )
        if observed != len(entries.get(name, ())):
            raise ValueError(f"readelf returned malformed {name} metadata")
    return {name: tuple(values) for name, values in entries.items()}


def validate_core_dynamic_search_paths(output: str, *, strict: bool) -> None:
    entries = parse_readelf_dynamic(output)
    rpaths = entries.get("RPATH", ())
    runpaths = entries.get("RUNPATH", ())
    if len(rpaths) > 1 or len(runpaths) > 1 or (rpaths and runpaths):
        raise ValueError("core native artifact has ambiguous dynamic search paths")
    values = rpaths + runpaths
    if not values:
        return
    if strict:
        raise ValueError("core native artifact has a dynamic search path")
    search_paths = [entry for value in values for entry in value.split(":")]
    unsafe = [
        entry
        for entry in search_paths
        if entry not in {"$ORIGIN", "${ORIGIN}"}
        and not entry.startswith(("$ORIGIN/", "${ORIGIN}/"))
    ]
    if unsafe:
        raise ValueError(
            f"core native artifact has unsafe dynamic search paths {unsafe}"
        )
