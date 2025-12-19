# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe
import os
import re

# This script extracts all the NCCL environment variables and parameters from source code files in a given directory.
# Specifically, it searches for calls to ncclGetEnv() and NCCL_PARAM() macros, and extracts the environment variable
# or parameter name from the arguments.
#
# Usage:
#   python extract_nccl.py <directory>
#
# Example Usage:
#   python3 "/data/users/<username>/fbsource/third-party/nccl/v2.27-1"
#
# Example Output:
# CUDA_LAUNCH_BLOCKING
# NCCL_ALGO
# NCCL_ALLOC_P2P_NET_LL_BUFFERS
# NCCL_BUFFSIZE
# NCCL_CGA_CLUSTER_SIZE
# NCCL_CHECK_POINTERS
# NCCL_CHUNK_SIZE
# ...
# NCCL_WIN_STRIDE
# NCCL_WORK_ARGS_BYTES
# NCCL_WORK_FIFO_BYTES


def extract_nccl_params_and_envs(root_dir: str):
    """
    Parses all NCCL environment variables and parameters from source code files in a given directory.

    Specifically, it searches for calls to ncclGetEnv() and NCCL_PARAM() macros, and extracts the
    environment variable or parameter name from the arguments.
    """
    # Regex for ncclGetEnv("...") or ncclGetEnv('...').
    nccl_getenv_regex = re.compile(r'ncclGetEnv\s*\(\s*([\'"])(.+?)\1')
    # Regex for NCCL_PARAM(name, "env", deftVal) or NCCL_PARAM(name, 'env', deftVal).
    nccl_param_regex = re.compile(r'NCCL_PARAM\s*\(\s*[^,]+,\s*([\'"])(.+?)\1\s*,')
    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for _, line in enumerate(f, 1):
                        getenv_matches = nccl_getenv_regex.findall(line)
                        param_match = nccl_param_regex.search(line)
                        if getenv_matches:
                            for _, arg in getenv_matches:
                                results.append(arg)
                        if param_match:
                            results.append("NCCL_" + param_match.group(2))
            except Exception:
                continue
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extract_nccl.py <directory>")
        sys.exit(1)
    root_directory = sys.argv[1]
    results = extract_nccl_params_and_envs(root_directory)
    results = set(results)

    # Check if "nccl_cvars.txt" exists and create a backup if it does
    nccl_cvars_path = "nccl_cvars.txt"
    if os.path.exists(nccl_cvars_path):
        import shutil

        idx: int = 1
        while os.path.exists(nccl_cvars_path + ".bak." + str(idx)):
            idx += 1

        backup_path = f"{nccl_cvars_path}.bak.{idx}"
        shutil.copy2(nccl_cvars_path, backup_path)
        print(f"Backup of existing '{nccl_cvars_path}' created at '{backup_path}'")

    with open("nccl_cvars.txt", "w") as f:
        for match in sorted(results):
            f.write(match + "\n")
