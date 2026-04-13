#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
#
# Generate bandwidth-vs-size plots from benchmark results.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Results from 8-rank CPU benchmark (latency in microseconds)
SIZES = [64, 4096, 262144, 4194304, 67108864, 104857600]

# 8 ranks, CPU, single node
DATA = {
    "all_reduce": {
        "UCC (shm)": [4.38, 14.48, 384.73, 5323.21, 65721.39, 98008.67],
        "UCC (tcp)": [37.49, 66.79, 792.22, 7140.41, 102138.42, 155671.75],
        "MPI (shm)": [4.38, 43.47, 524.28, 7029.73, 73083.98, 106005.20],
        "MPI (tcp)": [34.06, 42.74, 684.04, 9463.10, 150737.24, 236261.68],
        "Gloo": [1162.57, 1186.42, 1767.53, 10328.96, 136087.37, 205611.60],
    },
    "all_gather": {
        "UCC (shm)": [16.74, 55.73, 1489.68, 28126.84, 351086.83, 498710.01],
        "UCC (tcp)": [43.85, 210.58, 2083.45, 44077.65, 540097.46, 825883.94],
        "MPI (shm)": [16.36, 65.45, 1652.15, 37458.29, 527391.27, 790732.49],
        "MPI (tcp)": [45.59, 68.29, 2080.78, 44719.21, 780076.00, 1269653.75],
        "Gloo": [630.90, 698.33, 3571.05, 43499.36, 578841.58, 938607.62],
    },
    "broadcast": {
        "UCC (shm)": [2.04, 11.29, 215.35, 2351.52, 29628.26, 46591.16],
        "UCC (tcp)": [17.95, 31.15, 487.86, 4628.59, 60145.72, 92161.93],
        "MPI (shm)": [2.16, 14.25, 157.80, 1254.38, 21044.99, 37325.11],
        "MPI (tcp)": [14.99, 11.35, 1247.91, 13812.64, 286546.08, 326405.83],
        "Gloo": [80.84, 99.59, 708.84, 8190.41, 78704.75, 169123.91],
    },
}

COLORS = {
    "UCC (shm)": "#2ca02c",
    "UCC (tcp)": "#1f77b4",
    "MPI (shm)": "#ff7f0e",
    "MPI (tcp)": "#9467bd",
    "Gloo": "#d62728",
}
MARKERS = {
    "UCC (shm)": "o",
    "UCC (tcp)": "^",
    "MPI (shm)": "D",
    "MPI (tcp)": "v",
    "Gloo": "s",
}
BACKENDS = ["UCC (shm)", "MPI (shm)", "UCC (tcp)", "MPI (tcp)", "Gloo"]


def size_label(b):
    if b >= 1048576:
        return f"{b // 1048576}MB"
    if b >= 1024:
        return f"{b // 1024}KB"
    return f"{b}B"


def bandwidth_gbps(size_bytes, latency_us):
    return (size_bytes / latency_us) * 1e6 / 1e9


fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for ax, (coll_name, backends_data) in zip(axes, DATA.items()):
    size_labels = [size_label(s) for s in SIZES]
    x_positions = list(range(len(SIZES)))

    for backend in BACKENDS:
        if backend not in backends_data:
            continue
        latencies = backends_data[backend]
        bw = [bandwidth_gbps(s, l) for s, l in zip(SIZES, latencies)]
        ax.plot(
            x_positions,
            bw,
            marker=MARKERS[backend],
            color=COLORS[backend],
            label=backend,
            linewidth=2.5,
            markersize=9,
        )

    ax.set_title(coll_name, fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel("Message Size", fontsize=14)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(size_labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.legend(fontsize=13, loc="upper left")
    ax.set_xlim(-0.3, len(SIZES) - 0.7)

fig.suptitle(
    "CPU Collective Bandwidth \u2014 Gloo vs UCC vs MPI (single node)",
    fontsize=20,
    fontweight="bold",
    y=0.98,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out = "comms/torchcomms/tests/perf/py/gloo_vs_ucc_bandwidth.png"
fig.savefig(out, dpi=150)
print(f"Saved to {out}")
