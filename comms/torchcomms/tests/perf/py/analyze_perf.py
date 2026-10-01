#!/usr/bin/env python3
"""
Analyze collective performance results.

For every CSV found across the provided result directories a figure is produced
with two separate figures: Latency (us) and Bus Bandwidth (GB/s) vs message
size.  When --baseline is provided, relative plots and comparison reports are
also generated.

Usage:
    # Single adapter results (saves PNGs into ./perf_results/plots/)
    python analyze_perf.py  --dir torchcomms:./torchcomms

    # Compare multiple adapters
    python analyze_perf.py  --dir torchcomms:./torchcomms --dir c10d:./c10d

    # Three-way comparison with baseline for relative plots
    python analyze_perf.py  --dir torchcomms:./torchcomms --dir c10d:./c10d \
                             --dir c10d_torchcomms:./c10d_torchcomms --baseline c10d

    # Save all figures into a custom directory
    python analyze_perf.py  --dir torchcomms:./torchcomms --outdir my_plots
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_COLS = {
    "Collective",
    "SendMsgSize(B)",
    "LatAvg(us)",
    "LatMin(us)",
    "LatMax(us)",
    "BusBw(GB/s)",
}


def human_bytes(n):
    """Human-readable byte count using powers of 1024."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:g} {unit}"
        n /= 1024
    return f"{n:g} TB"


def fmt_xaxis(ax, sizes):
    """Log-2 x-axis with human-readable tick labels."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels(
        [human_bytes(s) for s in sizes], rotation=45, ha="right", fontsize=8
    )
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def load_csv(path: Path) -> pd.DataFrame | None:
    """Load and validate a CSV file. Returns None on failure."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Warning: could not read {path}: {e}", file=sys.stderr)
        return None
    if not REQUIRED_COLS.issubset(df.columns):
        missing = REQUIRED_COLS - set(df.columns)
        print(f"Warning: {path} missing columns {missing}, skipping.", file=sys.stderr)
        return None
    if df.empty:
        print(f"Warning: {path} has no data rows, skipping.", file=sys.stderr)
        return None
    return df.sort_values("SendMsgSize(B)").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def add_latency_series(ax, df: pd.DataFrame, label: str, color: str, marker: str):
    """Draw one latency curve (+ optional min/max band) onto ax."""
    sizes = df["SendMsgSize(B)"].values
    avg = df["LatAvg(us)"].values
    mn = df["LatMin(us)"].values
    mx = df["LatMax(us)"].values

    ax.plot(
        sizes, avg, marker=marker, ms=4, linewidth=1.5, color=color, label=f"{label}"
    )
    if not ((mn == avg).all() and (mx == avg).all()):
        ax.fill_between(
            sizes, mn, mx, alpha=0.18, color=color, label=f"{label} Min-Max"
        )


def add_bw_series(ax, df: pd.DataFrame, label: str, color: str, marker: str):
    """Draw one bus-bandwidth curve onto ax."""
    sizes = df["SendMsgSize(B)"].values
    bw = df["BusBw(GB/s)"].values
    ax.plot(sizes, bw, marker=marker, ms=4, linewidth=1.5, color=color, label=label)


def build_latency_figure(
    collective: str, series: list[tuple[str, pd.DataFrame]]
) -> plt.Figure:
    """Return a standalone Latency figure for *collective*."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=(8, 5))

    all_sizes = sorted({s for _, df in series for s in df["SendMsgSize(B)"].values})

    for idx, (label, df) in enumerate(series):
        add_latency_series(
            ax,
            df,
            label,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
        )

    ax.set_title(collective)
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Latency (µs)")
    ax.set_yscale("log")
    fmt_xaxis(ax, all_sizes)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def build_bw_figure(
    collective: str, series: list[tuple[str, pd.DataFrame]]
) -> plt.Figure:
    """Return a standalone Bus Bandwidth figure for *collective*."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=(8, 5))

    all_sizes = sorted({s for _, df in series for s in df["SendMsgSize(B)"].values})

    for idx, (label, df) in enumerate(series):
        add_bw_series(
            ax,
            df,
            label,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
        )

    ax.set_title(collective)
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Bus Bandwidth (GB/s)")
    fmt_xaxis(ax, all_sizes)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _get_relative_data(
    series: list[tuple[str, pd.DataFrame]],
    baseline_label: str = "c10d",
) -> tuple[pd.DataFrame | None, list[tuple[str, pd.DataFrame]]]:
    """Find the baseline and the non-baseline series. Returns (baseline_df, others)."""
    baseline_df = None
    for label, df in series:
        if label == baseline_label:
            baseline_df = df
            break
    others = [(label, df) for label, df in series if label != baseline_label]
    return baseline_df, others


def build_relative_latency_figure(
    collective: str,
    series: list[tuple[str, pd.DataFrame]],
    baseline_label: str = "c10d",
) -> plt.Figure | None:
    """Return a figure showing % latency change relative to *baseline_label*.

    Latency change = (baseline - other) / baseline * 100
    (positive ⟹ lower latency ⟹ better)

    Returns None when the baseline is not present in *series*.
    """
    baseline_df, others = _get_relative_data(series, baseline_label)
    if baseline_df is None or not others:
        return None

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, other_df) in enumerate(others):
        merged = pd.merge(
            baseline_df[["SendMsgSize(B)", "LatAvg(us)"]],
            other_df[["SendMsgSize(B)", "LatAvg(us)"]],
            on="SendMsgSize(B)",
            suffixes=("_base", "_other"),
        )
        if merged.empty:
            continue

        sizes = merged["SendMsgSize(B)"].values
        lat_pct = (
            (merged["LatAvg(us)_base"] - merged["LatAvg(us)_other"])
            / merged["LatAvg(us)_base"]
            * 100
        ).values

        ax.plot(
            sizes,
            lat_pct,
            marker=markers[idx % len(markers)],
            ms=4,
            linewidth=1.5,
            color=colors[idx % len(colors)],
            label=label,
        )

    all_sizes = sorted({s for _, df in series for s in df["SendMsgSize(B)"].values})
    ax.set_title(f"{collective} — Latency Change vs {baseline_label}")
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Latency Change (%)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    fmt_xaxis(ax, all_sizes)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def build_relative_bw_figure(
    collective: str,
    series: list[tuple[str, pd.DataFrame]],
    baseline_label: str = "c10d",
) -> plt.Figure | None:
    """Return a figure showing % bandwidth change relative to *baseline_label*.

    Bandwidth change = (other - baseline) / baseline * 100
    (positive ⟹ higher bandwidth ⟹ better)

    Returns None when the baseline is not present in *series*.
    """
    baseline_df, others = _get_relative_data(series, baseline_label)
    if baseline_df is None or not others:
        return None

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, other_df) in enumerate(others):
        merged = pd.merge(
            baseline_df[["SendMsgSize(B)", "BusBw(GB/s)"]],
            other_df[["SendMsgSize(B)", "BusBw(GB/s)"]],
            on="SendMsgSize(B)",
            suffixes=("_base", "_other"),
        )
        if merged.empty:
            continue

        sizes = merged["SendMsgSize(B)"].values
        bw_pct = (
            (merged["BusBw(GB/s)_other"] - merged["BusBw(GB/s)_base"])
            / merged["BusBw(GB/s)_base"]
            * 100
        ).values

        ax.plot(
            sizes,
            bw_pct,
            marker=markers[idx % len(markers)],
            ms=4,
            linewidth=1.5,
            color=colors[idx % len(colors)],
            label=label,
        )

    all_sizes = sorted({s for _, df in series for s in df["SendMsgSize(B)"].values})
    ax.set_title(f"{collective} — Bandwidth Change vs {baseline_label}")
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Bus Bandwidth Change (%)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    fmt_xaxis(ax, all_sizes)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def build_adapter_summary_df(
    all_series: dict[str, list[tuple[str, pd.DataFrame]]],
    adapter_label: str,
) -> pd.DataFrame:
    """Build a summary DataFrame for a single adapter (no baseline comparison).

    Returns an empty DataFrame when there is nothing to summarise.
    """
    rows: list[dict] = []

    for file_base, series in sorted(all_series.items()):
        for label, df in series:
            if label != adapter_label:
                continue
            for _, r in df.iterrows():
                msg_size = int(r["SendMsgSize(B)"])
                rows.append(
                    {
                        "Collective": file_base,
                        "NumRanks": int(r["Ranks"]) if "Ranks" in r.index else None,
                        "DeviceName": (
                            r["DeviceName"] if "DeviceName" in r.index else None
                        ),
                        "MsgSize": human_bytes(msg_size),
                        "SendMsgSize(B)": msg_size,
                        "LatAvg(us)": round(r["LatAvg(us)"], 2),
                        "LatMin(us)": round(r["LatMin(us)"], 2),
                        "LatMax(us)": round(r["LatMax(us)"], 2),
                        "BusBw(GB/s)": round(r["BusBw(GB/s)"], 2),
                    }
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(["Collective", "SendMsgSize(B)"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_comparison_summary_df(
    all_series: dict[str, list[tuple[str, pd.DataFrame]]],
    baseline_label: str,
) -> pd.DataFrame:
    """Build a summary DataFrame with per-collective, per-message-size stats.

    *all_series* maps ``file_base`` (e.g. ``"all_reduce_sum"``) to the list of
    ``(adapter_label, df)`` pairs collected during the per-CSV loop.

    For every adapter and message size the report contains:
      - ``LatAvg(us)``, ``LatMin(us)``, ``LatMax(us)``
      - ``BusBw(GB/s)``
      - ``LatChange(%)``  = (baseline - adapter) / baseline * 100
      - ``BwChange(%)``   = (adapter  - baseline) / baseline * 100
    (positive values = improvement over baseline)

    Returns an empty DataFrame when there is nothing to summarise.
    """
    rows: list[dict] = []

    for file_base, series in sorted(all_series.items()):
        # Pre-compute baseline keyed by message size
        baseline_by_size: dict[int, pd.Series] = {}
        for label, df in series:
            if label == baseline_label:
                for _, r in df.iterrows():
                    baseline_by_size[int(r["SendMsgSize(B)"])] = r
                break

        for label, df in series:
            for _, r in df.iterrows():
                msg_size = int(r["SendMsgSize(B)"])
                lat_avg = r["LatAvg(us)"]
                lat_min = r["LatMin(us)"]
                lat_max = r["LatMax(us)"]
                bw = r["BusBw(GB/s)"]

                ranks = int(r["Ranks"]) if "Ranks" in r.index else None
                device = r["DeviceName"] if "DeviceName" in r.index else None

                row = {
                    "Collective": file_base,
                    "Adapter": label,
                    "NumRanks": ranks,
                    "DeviceName": device,
                    "MsgSize": human_bytes(msg_size),
                    "SendMsgSize(B)": msg_size,
                    "LatAvg(us)": round(lat_avg, 2),
                    "LatMin(us)": round(lat_min, 2),
                    "LatMax(us)": round(lat_max, 2),
                    "BusBw(GB/s)": round(bw, 2),
                    "LatChange(%)": None,
                    "BwChange(%)": None,
                }

                if msg_size in baseline_by_size and label != baseline_label:
                    base = baseline_by_size[msg_size]
                    base_lat = base["LatAvg(us)"]
                    base_bw = base["BusBw(GB/s)"]
                    if base_lat > 0:
                        row["LatChange(%)"] = round(
                            (base_lat - lat_avg) / base_lat * 100, 1
                        )
                    if base_bw > 0:
                        row["BwChange(%)"] = round((bw - base_bw) / base_bw * 100, 1)

                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(["Collective", "SendMsgSize(B)", "Adapter"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _format_table(df: pd.DataFrame) -> str:
    """Format a DataFrame as a table string with separators between collectives."""
    table = df.to_string(index=False)
    table_lines = table.splitlines()
    header = table_lines[0]
    separator = "-" * len(header)

    lines: list[str] = [header, separator]
    prev_collective = None
    for line in table_lines[1:]:
        collective = line.split()[0]
        if prev_collective is not None and collective != prev_collective:
            lines.append(separator)
        lines.append(line)
        prev_collective = collective
    lines.append("")
    return "\n".join(lines)


def _filter_regressions(
    summary_df: pd.DataFrame,
    baseline_label: str,
    threshold: float,
) -> pd.DataFrame:
    """Return non-baseline rows where latency regressed beyond *threshold*.

    A regression is a row where ``LatChange(%) < -threshold`` (higher latency)
    compared to the baseline.
    """
    non_baseline = summary_df[summary_df["Adapter"] != baseline_label]
    return non_baseline[
        non_baseline["LatChange(%)"].notna()
        & (non_baseline["LatChange(%)"] < -threshold)
    ]


def _filter_improvements(
    summary_df: pd.DataFrame,
    baseline_label: str,
    threshold: float,
) -> pd.DataFrame:
    """Return non-baseline rows where latency improved beyond *threshold*.

    An improvement is a row where ``LatChange(%) > threshold`` (lower latency)
    compared to the baseline.
    """
    non_baseline = summary_df[summary_df["Adapter"] != baseline_label]
    return non_baseline[
        non_baseline["LatChange(%)"].notna()
        & (non_baseline["LatChange(%)"] > threshold)
    ]


def save_adapter_summary_txt(
    summary_df: pd.DataFrame, adapter_label: str, path: Path
) -> None:
    """Save a per-adapter summary report as a formatted text file."""
    if summary_df.empty:
        return

    display_cols = [
        "Collective",
        "NumRanks",
        "MsgSize",
        "LatAvg(us)",
        "LatMin(us)",
        "LatMax(us)",
        "BusBw(GB/s)",
        "DeviceName",
    ]
    df = summary_df[display_cols].copy()

    lines = [
        "=" * 80,
        f"PERFORMANCE SUMMARY — {adapter_label}",
        "=" * 80,
        _format_table(df),
    ]
    path.write_text("\n".join(lines) + "\n")


def save_adapter_summary_csv(summary_df: pd.DataFrame, path: Path) -> None:
    """Save a per-adapter summary DataFrame to a CSV file."""
    if summary_df.empty:
        return
    csv_cols = [
        "Collective",
        "NumRanks",
        "MsgSize",
        "SendMsgSize(B)",
        "LatAvg(us)",
        "LatMin(us)",
        "LatMax(us)",
        "BusBw(GB/s)",
        "DeviceName",
    ]
    summary_df[csv_cols].to_csv(path, index=False)


def save_comparison_summary_txt(
    summary_df: pd.DataFrame, baseline_label: str, path: Path
) -> None:
    """Save the comparison summary report as a formatted text file."""
    if summary_df.empty:
        return

    display_cols = [
        "Collective",
        "Adapter",
        "NumRanks",
        "MsgSize",
        "LatAvg(us)",
        "LatMin(us)",
        "LatMax(us)",
        "BusBw(GB/s)",
        "LatChange(%)",
        "BwChange(%)",
        "DeviceName",
    ]
    df = summary_df[display_cols].copy()
    is_baseline = df["Adapter"] == baseline_label
    df["LatChange(%)"] = df["LatChange(%)"].apply(
        lambda v: f"{v:+.1f}" if pd.notna(v) else "n/a"
    )
    df["BwChange(%)"] = df["BwChange(%)"].apply(
        lambda v: f"{v:+.1f}" if pd.notna(v) else "n/a"
    )
    df.loc[is_baseline, "LatChange(%)"] = "baseline"
    df.loc[is_baseline, "BwChange(%)"] = "baseline"

    lines = ["=" * 80, "COMPARISON SUMMARY", "=" * 80, _format_table(df)]
    path.write_text("\n".join(lines) + "\n")


def save_comparison_summary_csv(summary_df: pd.DataFrame, path: Path) -> None:
    """Save the comparison summary DataFrame to a CSV file."""
    if summary_df.empty:
        return
    csv_cols = [
        "Collective",
        "Adapter",
        "NumRanks",
        "MsgSize",
        "SendMsgSize(B)",
        "LatAvg(us)",
        "LatMin(us)",
        "LatMax(us)",
        "BusBw(GB/s)",
        "LatChange(%)",
        "BwChange(%)",
        "DeviceName",
    ]
    summary_df[csv_cols].to_csv(path, index=False)


_REPORT_DISPLAY_COLS = [
    "Collective",
    "Adapter",
    "NumRanks",
    "MsgSize",
    "LatAvg(us)",
    "LatMin(us)",
    "LatMax(us)",
    "BusBw(GB/s)",
    "LatChange(%)",
    "BwChange(%)",
    "DeviceName",
]

_REPORT_CSV_COLS = [
    "Collective",
    "Adapter",
    "NumRanks",
    "MsgSize",
    "SendMsgSize(B)",
    "LatAvg(us)",
    "LatMin(us)",
    "LatMax(us)",
    "BusBw(GB/s)",
    "LatChange(%)",
    "BwChange(%)",
    "DeviceName",
]


def _write_filtered_report(
    filtered_df: pd.DataFrame,
    title: str,
    threshold: float,
    empty_msg: str,
    txt_path: Path,
    csv_path: Path,
) -> None:
    """Write txt + csv for an already-filtered DataFrame."""
    if filtered_df.empty:
        txt_path.write_text(f"{empty_msg} {threshold:.1f}% detected.\n")
        pd.DataFrame(columns=_REPORT_CSV_COLS).to_csv(csv_path, index=False)
        return

    df = filtered_df[_REPORT_DISPLAY_COLS].copy()
    df["LatChange(%)"] = df["LatChange(%)"].apply(
        lambda v: f"{v:+.1f}" if pd.notna(v) else "n/a"
    )
    df["BwChange(%)"] = df["BwChange(%)"].apply(
        lambda v: f"{v:+.1f}" if pd.notna(v) else "n/a"
    )

    lines = ["=" * 80, title, "=" * 80, _format_table(df)]
    txt_path.write_text("\n".join(lines) + "\n")

    filtered_df[_REPORT_CSV_COLS].to_csv(csv_path, index=False)


def save_regression_reports(
    summary_df: pd.DataFrame,
    baseline_label: str,
    threshold: float,
    outdir: Path,
) -> list[Path]:
    """Save per-adapter regression reports (txt + csv).

    Files are written to ``{adapter}_vs_{baseline}/perf_regress.{ext}``.
    Returns the list of saved paths.
    """
    if summary_df.empty:
        return []

    saved: list[Path] = []
    adapters = [a for a in summary_df["Adapter"].unique() if a != baseline_label]

    for adapter in adapters:
        adapter_df = summary_df[summary_df["Adapter"] == adapter]
        regressed = _filter_regressions(adapter_df, baseline_label, threshold)
        tag = f"{adapter}_vs_{baseline_label}"
        pair_dir = outdir / tag
        pair_dir.mkdir(parents=True, exist_ok=True)

        txt_path = pair_dir / "perf_regress.txt"
        csv_path = pair_dir / "perf_regress.csv"
        _write_filtered_report(
            regressed,
            title=f"PERFORMANCE REGRESSIONS — {adapter} vs {baseline_label} (threshold: {threshold:.1f}%)",
            threshold=threshold,
            empty_msg=f"No performance regressions for {adapter} vs {baseline_label} exceeding",
            txt_path=txt_path,
            csv_path=csv_path,
        )
        saved.extend([txt_path, csv_path])

    return saved


def save_improvement_reports(
    summary_df: pd.DataFrame,
    baseline_label: str,
    threshold: float,
    outdir: Path,
) -> list[Path]:
    """Save per-adapter improvement reports (txt + csv).

    Files are written to ``{adapter}_vs_{baseline}/perf_improve.{ext}``.
    Returns the list of saved paths.
    """
    if summary_df.empty:
        return []

    saved: list[Path] = []
    adapters = [a for a in summary_df["Adapter"].unique() if a != baseline_label]

    for adapter in adapters:
        adapter_df = summary_df[summary_df["Adapter"] == adapter]
        improved = _filter_improvements(adapter_df, baseline_label, threshold)
        tag = f"{adapter}_vs_{baseline_label}"
        pair_dir = outdir / tag
        pair_dir.mkdir(parents=True, exist_ok=True)

        txt_path = pair_dir / "perf_improve.txt"
        csv_path = pair_dir / "perf_improve.csv"
        _write_filtered_report(
            improved,
            title=f"PERFORMANCE IMPROVEMENTS — {adapter} vs {baseline_label} (threshold: {threshold:.1f}%)",
            threshold=threshold,
            empty_msg=f"No performance improvements for {adapter} vs {baseline_label} exceeding",
            txt_path=txt_path,
            csv_path=csv_path,
        )
        saved.extend([txt_path, csv_path])

    return saved


def _extract_bench_info(
    all_series: dict[str, list[tuple[str, pd.DataFrame]]],
    adapter_label: str | None = None,
) -> dict[str, str]:
    """Extract high-level benchmark metadata from the raw CSV DataFrames.

    Scans DataFrames for common metadata columns and returns a dict of
    field-name to display-value (using the first non-empty value found, or
    aggregating where appropriate like message-size range).

    When *adapter_label* is given, only DataFrames whose label matches are
    considered — this prevents mixing metadata from different adapters
    (e.g. baseline backend version leaking into an adapter-specific report).
    """
    info: dict[str, str] = {}
    all_dfs: list[pd.DataFrame] = []
    for series in all_series.values():
        for label, df in series:
            if adapter_label is None or label == adapter_label:
                all_dfs.append(df)

    if not all_dfs:
        return info

    sample = all_dfs[0]

    # DeviceName
    if "DeviceName" in sample.columns:
        info["Device"] = str(sample["DeviceName"].iloc[0])

    # DeviceType
    if "DeviceType" in sample.columns:
        info["Device type"] = str(sample["DeviceType"].iloc[0])

    # Ranks
    if "Ranks" in sample.columns:
        all_ranks = sorted(
            {
                int(r)
                for df in all_dfs
                if "Ranks" in df.columns
                for r in df["Ranks"].unique()
            }
        )
        info["Number of ranks"] = ", ".join(str(r) for r in all_ranks)

    # Message size range
    all_sizes = sorted(
        {int(s) for df in all_dfs for s in df["SendMsgSize(B)"].unique()}
    )
    if all_sizes:
        info["Message size range"] = (
            f"{human_bytes(all_sizes[0])} – {human_bytes(all_sizes[-1])}"
        )
        info["Message sizes"] = str(len(all_sizes))

    # DispatchMode
    if "DispatchMode" in sample.columns:
        modes = sorted(
            {
                str(m)
                for df in all_dfs
                if "DispatchMode" in df.columns
                for m in df["DispatchMode"].unique()
            }
        )
        info["Dispatch mode"] = ", ".join(modes)

    # DType
    if "DType" in sample.columns:
        dtypes = sorted(
            {
                str(d)
                for df in all_dfs
                if "DType" in df.columns
                for d in df["DType"].unique()
            }
        )
        info["DType"] = ", ".join(dtypes)

    # WarmupIters / MeasureIters / SyncInterval
    for col, label in [
        ("WarmupIters", "Warmup iterations"),
        ("MeasureIters", "Measure iterations"),
        ("SyncInterval", "Sync interval"),
    ]:
        if col in sample.columns:
            vals = sorted(
                {
                    int(v)
                    for df in all_dfs
                    if col in df.columns
                    for v in df[col].unique()
                }
            )
            info[label] = ", ".join(str(v) for v in vals)

    # CommBackend
    if "CommBackend" in sample.columns:
        backends = sorted(
            {
                str(b)
                for df in all_dfs
                if "CommBackend" in df.columns
                for b in df["CommBackend"].unique()
            }
        )
        info["Backend"] = ", ".join(backends)

    # CommBackendVersion
    if "CommBackendVersion" in sample.columns:
        bversions = sorted(
            {
                str(v)
                for df in all_dfs
                if "CommBackendVersion" in df.columns
                for v in df["CommBackendVersion"].dropna().unique()
                if str(v) not in ("n/a", "nan", "")
            }
        )
        if bversions:
            info["Backend version"] = ", ".join(bversions)

    # TorchVersion
    if "TorchVersion" in sample.columns:
        versions = sorted(
            {
                str(v)
                for df in all_dfs
                if "TorchVersion" in df.columns
                for v in df["TorchVersion"].unique()
            }
        )
        info["Torch version"] = ", ".join(versions)

    return info


def save_highlights_report(
    summary_df: pd.DataFrame,
    baseline_label: str,
    regression_threshold: float,
    improvement_threshold: float,
    outdir: Path,
    all_series: dict[str, list[tuple[str, pd.DataFrame]]] | None = None,
) -> list[Path]:
    """Save per-adapter highlights reports (txt + csv).

    Each report contains:
      - High-level benchmark information (device, ranks, message sizes, etc.).
      - Total collectives and data points examined.
      - Number of collectives and data points classified as improvements,
        regressions, or neutral (within both thresholds).

    Files are written to ``{adapter}_vs_{baseline}/perf_highlights.{ext}``.
    Returns the list of saved paths.
    """
    if summary_df.empty:
        return []

    saved: list[Path] = []
    adapters = [a for a in summary_df["Adapter"].unique() if a != baseline_label]

    for adapter in adapters:
        bench_info = (
            _extract_bench_info(all_series, adapter_label=adapter) if all_series else {}
        )
        adapter_df = summary_df[summary_df["Adapter"] == adapter]
        # Only rows with a valid LatChange comparison
        compared = adapter_df[adapter_df["LatChange(%)"].notna()]

        total_collectives = compared["Collective"].nunique()
        total_points = len(compared)

        improved = _filter_improvements(
            adapter_df, baseline_label, improvement_threshold
        )
        regressed = _filter_regressions(
            adapter_df, baseline_label, regression_threshold
        )

        improve_collectives = (
            improved["Collective"].nunique() if not improved.empty else 0
        )
        improve_points = len(improved)
        regress_collectives = (
            regressed["Collective"].nunique() if not regressed.empty else 0
        )
        regress_points = len(regressed)

        # Neutral: compared points that are neither improved nor regressed
        neutral_points = total_points - improve_points - regress_points
        # Collectives that appear only in neutral (not in either filtered set)
        all_colls = set(compared["Collective"].unique())
        improve_colls = (
            set(improved["Collective"].unique()) if not improved.empty else set()
        )
        regress_colls = (
            set(regressed["Collective"].unique()) if not regressed.empty else set()
        )
        neutral_only_colls = all_colls - improve_colls - regress_colls
        neutral_collectives = len(neutral_only_colls)

        # Build neutral DataFrame for stats
        imp_idx = improved.index if not improved.empty else pd.Index([])
        reg_idx = regressed.index if not regressed.empty else pd.Index([])
        neutral_df = compared.loc[~compared.index.isin(imp_idx.union(reg_idx))]

        def _avg(df: pd.DataFrame, col: str = "LatChange(%)") -> float | None:
            vals = df[col].dropna()
            return round(vals.mean(), 1) if len(vals) > 0 else None

        def _med(df: pd.DataFrame, col: str = "LatChange(%)") -> float | None:
            vals = df[col].dropna()
            return round(vals.median(), 1) if len(vals) > 0 else None

        # Global stats
        imp_avg = _avg(improved)
        imp_med = _med(improved)
        reg_avg = _avg(regressed)
        reg_med = _med(regressed)
        neu_avg = _avg(neutral_df)
        neu_med = _med(neutral_df)
        total_avg = _avg(compared)
        total_med = _med(compared)

        tag = f"{adapter}_vs_{baseline_label}"
        pair_dir = outdir / tag
        pair_dir.mkdir(parents=True, exist_ok=True)
        txt_path = pair_dir / "perf_highlights.txt"
        csv_path = pair_dir / "perf_highlights.csv"

        # --- Build text report ---
        w = 89

        def _pct(n: int, total: int) -> str:
            return f"{n / total * 100:.1f}%" if total > 0 else "n/a"

        def _fmt(v: float | None) -> str:
            return f"{v:+.1f}%" if v is not None else "n/a"

        lines = [
            "=" * w,
            f"PERFORMANCE HIGHLIGHTS — {adapter} vs {baseline_label}",
            "=" * w,
        ]

        # Benchmark info section
        if bench_info:
            lines.append("")
            lines.append("Benchmark Info:")
            label_w = max(len(k) for k in bench_info) + 2
            for key, val in bench_info.items():
                lines.append(f"  {key:<{label_w}}: {val}")

        lines += [
            "",
            f"  Regression threshold : {regression_threshold:.1f}%",
            f"  Improvement threshold: {improvement_threshold:.1f}%",
            "",
            "-" * w,
            f"  {'Category':<20} {'Collectives':>12} {'Points':>10} {'%':>8} {'AvgLatChange(%)':>16} {'MedLatChange(%)':>16}",
            "-" * w,
            f"  {'Improved':<20} {improve_collectives:>12} {improve_points:>10} {_pct(improve_points, total_points):>8} {_fmt(imp_avg):>16} {_fmt(imp_med):>16}",
            f"  {'Regressions':<20} {regress_collectives:>12} {regress_points:>10} {_pct(regress_points, total_points):>8} {_fmt(reg_avg):>16} {_fmt(reg_med):>16}",
            f"  {'Neutral':<20} {neutral_collectives:>12} {neutral_points:>10} {_pct(neutral_points, total_points):>8} {_fmt(neu_avg):>16} {_fmt(neu_med):>16}",
            "-" * w,
            f"  {'Total examined':<20} {total_collectives:>12} {total_points:>10} {'100.0%':>8}",
            "",
        ]

        # Per-collective breakdown
        lines.append("Per-collective breakdown:")
        lines.append("")
        lines.append(
            f"  {'Collective':<30} {'Imp':>6} {'Reg':>6} {'Neu':>6} {'Tot':>6}"
            f" {'Imp(%)':>7} {'Reg(%)':>7} {'Neu(%)':>7}"
            f" {'AvgLatImp(%)':>13} {'AvgLatReg(%)':>13} {'AvgLatNeu(%)':>13}"
            f" {'MedLatImp(%)':>13} {'MedLatReg(%)':>13} {'MedLatNeu(%)':>13}"
        )
        lines.append("  " + "-" * 166)

        for coll in sorted(all_colls):
            coll_compared = compared[compared["Collective"] == coll]
            coll_total = len(coll_compared)
            coll_imp_df = (
                improved[improved["Collective"] == coll]
                if not improved.empty
                else improved
            )
            coll_reg_df = (
                regressed[regressed["Collective"] == coll]
                if not regressed.empty
                else regressed
            )
            coll_imp = len(coll_imp_df)
            coll_reg = len(coll_reg_df)
            coll_neu = coll_total - coll_imp - coll_reg
            coll_neu_df = coll_compared.loc[
                ~coll_compared.index.isin(coll_imp_df.index.union(coll_reg_df.index))
            ]
            lines.append(
                f"  {coll:<30} {coll_imp:>6} {coll_reg:>6} {coll_neu:>6} {coll_total:>6}"
                f" {_pct(coll_imp, coll_total):>7} {_pct(coll_reg, coll_total):>7} {_pct(coll_neu, coll_total):>7}"
                f" {_fmt(_avg(coll_imp_df)):>13} {_fmt(_avg(coll_reg_df)):>13} {_fmt(_avg(coll_neu_df)):>13}"
                f" {_fmt(_med(coll_imp_df)):>13} {_fmt(_med(coll_reg_df)):>13} {_fmt(_med(coll_neu_df)):>13}"
            )

        lines.append("  " + "-" * 166)
        lines.append(
            f"  {'Total':<30} {improve_points:>6} {regress_points:>6} {neutral_points:>6} {total_points:>6}"
        )

        lines.append("")
        txt_path.write_text("\n".join(lines) + "\n")

        # --- Build CSV report ---
        def _pct_val(n: int, total: int) -> float | None:
            return round(n / total * 100, 1) if total > 0 else None

        csv_rows = []
        for coll in sorted(all_colls):
            coll_compared = compared[compared["Collective"] == coll]
            coll_total = len(coll_compared)
            coll_imp_df = (
                improved[improved["Collective"] == coll]
                if not improved.empty
                else improved
            )
            coll_reg_df = (
                regressed[regressed["Collective"] == coll]
                if not regressed.empty
                else regressed
            )
            coll_imp = len(coll_imp_df)
            coll_reg = len(coll_reg_df)
            coll_neu = coll_total - coll_imp - coll_reg
            coll_neu_df = coll_compared.loc[
                ~coll_compared.index.isin(coll_imp_df.index.union(coll_reg_df.index))
            ]
            csv_rows.append(
                {
                    "Collective": coll,
                    "Adapter": adapter,
                    "Baseline": baseline_label,
                    "ImprovedPoints": coll_imp,
                    "RegressedPoints": coll_reg,
                    "NeutralPoints": coll_neu,
                    "TotalPoints": coll_total,
                    "Improved(%)": _pct_val(coll_imp, coll_total),
                    "Regressed(%)": _pct_val(coll_reg, coll_total),
                    "Neutral(%)": _pct_val(coll_neu, coll_total),
                    "AvgLatImp(%)": _avg(coll_imp_df),
                    "AvgLatReg(%)": _avg(coll_reg_df),
                    "AvgLatNeu(%)": _avg(coll_neu_df),
                    "MedLatImp(%)": _med(coll_imp_df),
                    "MedLatReg(%)": _med(coll_reg_df),
                    "MedLatNeu(%)": _med(coll_neu_df),
                }
            )
        # Summary row
        csv_rows.append(
            {
                "Collective": "ALL",
                "Adapter": adapter,
                "Baseline": baseline_label,
                "ImprovedPoints": improve_points,
                "RegressedPoints": regress_points,
                "NeutralPoints": neutral_points,
                "TotalPoints": total_points,
                "Improved(%)": _pct_val(improve_points, total_points),
                "Regressed(%)": _pct_val(regress_points, total_points),
                "Neutral(%)": _pct_val(neutral_points, total_points),
                "AvgLatImp(%)": imp_avg,
                "AvgLatReg(%)": reg_avg,
                "AvgLatNeu(%)": neu_avg,
                "MedLatImp(%)": imp_med,
                "MedLatReg(%)": reg_med,
                "MedLatNeu(%)": neu_med,
            }
        )
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        saved.extend([txt_path, csv_path])

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_dir_arg(value: str) -> tuple[str, Path]:
    """Parse a 'label:path' string into (label, Path)."""
    if ":" in value:
        label, path_str = value.split(":", 1)
        return label.strip(), Path(path_str.strip())
    # No label provided — derive from directory name
    p = Path(value)
    return p.name, p


def parse_args():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Analyze collective performance results.",
        epilog=(
            "Examples:\n"
            "  %(prog)s --dir torchcomms:./torchcomms\n"
            "  %(prog)s --dir torchcomms:./torchcomms --dir c10d:./c10d\n"
            "  %(prog)s --dir c10d:./c10d --dir c10d_torchcomms:./c10d_torchcomms --baseline c10d\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        dest="dirs",
        action="append",
        metavar="LABEL:PATH",
        help=(
            "Result directory in 'label:path' format. May be repeated. "
            "CSVs are discovered across all directories. "
            "If omitted, defaults to ./torchcomms/."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help=(
            "Adapter label to use as the baseline for relative plots "
            "and comparison reports. Must match a --dir label. "
            "When omitted, only absolute plots are generated."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=script_dir / "perf_results",
        help="Directory to write performance analysis results (default: ./perf_results/).",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=5.0,
        metavar="PCT",
        help=(
            "Minimum percentage regression in latency to include in the performance "
            "regression report (default: 5.0)."
        ),
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=5.0,
        metavar="PCT",
        help=(
            "Minimum percentage improvement in latency to include in the performance "
            "improvement report (default: 5.0)."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        default=False,
        help="Generate only the summary reports (CSV + terminal), skip generating figures.",
    )

    args = parser.parse_args()

    # Default: single torchcomms directory
    if not args.dirs:
        args.dirs = [f"torchcomms:{script_dir / 'torchcomms'}"]

    # Parse label:path pairs
    args.parsed_dirs = [_parse_dir_arg(d) for d in args.dirs]

    return args


def main():
    args = parse_args()

    # Validate directories — warn and skip any that don't exist.
    validated_dirs: list[tuple[str, Path]] = []
    for label, dir_path in args.parsed_dirs:
        if not dir_path.is_dir():
            print(
                f"Warning: directory not found for '{label}' ({dir_path}), skipping.",
                file=sys.stderr,
            )
            continue
        validated_dirs.append((label, dir_path))

    if not validated_dirs:
        print("Error: no valid result directories found.", file=sys.stderr)
        sys.exit(1)

    # Validate --baseline label matches a directory if provided
    dir_labels = {label for label, _ in validated_dirs}
    if args.baseline is not None and args.baseline not in dir_labels:
        print(
            f"Error: --baseline '{args.baseline}' does not match any --dir label "
            f"({', '.join(sorted(dir_labels))})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Discover CSV filenames across all directories (union)
    csv_names: dict[str, None] = {}
    for _, dir_path in validated_dirs:
        for csv_path in sorted(dir_path.glob("*.csv")):
            csv_names[csv_path.name] = None

    if not csv_names:
        print(
            f"No CSV files found in any result directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120})

    # Collect all series for the summary report (keyed by file_base)
    all_series: dict[str, list[tuple[str, pd.DataFrame]]] = {}

    for csv_name in csv_names:
        # Load from each directory that has this CSV
        series: list[tuple[str, pd.DataFrame]] = []
        for label, dir_path in validated_dirs:
            csv_path = dir_path / csv_name
            if csv_path.exists():
                df = load_csv(csv_path)
                if df is not None:
                    series.append((label, df))

        if not series:
            continue

        collective = series[0][1]["Collective"].iloc[0]

        # Detect reduce op from the CSV (non-"n/a" ReduceOp column)
        reduce_op = None
        first_df = series[0][1]
        if "ReduceOp" in first_df.columns:
            op_val = first_df["ReduceOp"].iloc[0]
            if isinstance(op_val, str) and op_val != "n/a":
                reduce_op = op_val

        # Use "collective (op)" for titles when a reduce op is present
        title = f"{collective} ({reduce_op})" if reduce_op else collective
        # Append op to filenames for reduce collectives
        file_base = f"{collective}_{reduce_op}" if reduce_op else collective

        all_series[file_base] = series

        if not args.summary_only:
            fig_lat = build_latency_figure(title, series)
            fig_bw = build_bw_figure(title, series)

            collective_dir = args.outdir / "plots" / collective
            collective_dir.mkdir(parents=True, exist_ok=True)

            adapters = "_vs_".join(label for label, _ in series)
            suffix = f"_{adapters}" if len(series) > 1 else f"_{series[0][0]}"
            lat_path = collective_dir / f"{file_base}{suffix}_latency.png"
            bw_path = collective_dir / f"{file_base}{suffix}_busbw.png"
            fig_lat.savefig(lat_path, bbox_inches="tight")
            fig_bw.savefig(bw_path, bbox_inches="tight")
            print(f"Saved: {lat_path}")
            print(f"Saved: {bw_path}")
            plt.close(fig_lat)
            plt.close(fig_bw)

            if args.baseline is not None:
                fig_rel_lat = build_relative_latency_figure(
                    title, series, baseline_label=args.baseline
                )
                fig_rel_bw = build_relative_bw_figure(
                    title, series, baseline_label=args.baseline
                )
                if fig_rel_lat is not None:
                    rel_lat_path = (
                        collective_dir / f"{file_base}{suffix}_relative_latency.png"
                    )
                    fig_rel_lat.savefig(rel_lat_path, bbox_inches="tight")
                    print(f"Saved: {rel_lat_path}")
                    plt.close(fig_rel_lat)
                if fig_rel_bw is not None:
                    rel_bw_path = (
                        collective_dir / f"{file_base}{suffix}_relative_busbw.png"
                    )
                    fig_rel_bw.savefig(rel_bw_path, bbox_inches="tight")
                    print(f"Saved: {rel_bw_path}")
                    plt.close(fig_rel_bw)

    # --- Per-adapter summary reports ---
    for label, _ in validated_dirs:
        adapter_df = build_adapter_summary_df(all_series, label)
        if adapter_df.empty:
            continue

        csv_path = args.outdir / f"perf_summary_{label}.csv"
        save_adapter_summary_csv(adapter_df, csv_path)
        print(f"Saved: {csv_path}")

        txt_path = args.outdir / f"perf_summary_{label}.txt"
        save_adapter_summary_txt(adapter_df, label, txt_path)
        print(f"Saved: {txt_path}")

    # --- Baseline comparison reports ---
    if args.baseline is not None:
        summary_df = build_comparison_summary_df(all_series, args.baseline)
        if not summary_df.empty:
            summary_csv_path = args.outdir / "perf_comparison.csv"
            save_comparison_summary_csv(summary_df, summary_csv_path)
            print(f"Saved: {summary_csv_path}")

            summary_txt_path = args.outdir / "perf_comparison.txt"
            save_comparison_summary_txt(summary_df, args.baseline, summary_txt_path)
            print(f"Saved: {summary_txt_path}")

            for p in save_regression_reports(
                summary_df,
                args.baseline,
                args.regression_threshold,
                args.outdir,
            ):
                print(f"Saved: {p}")

            for p in save_improvement_reports(
                summary_df,
                args.baseline,
                args.improvement_threshold,
                args.outdir,
            ):
                print(f"Saved: {p}")

            for p in save_highlights_report(
                summary_df,
                args.baseline,
                args.regression_threshold,
                args.improvement_threshold,
                args.outdir,
                all_series=all_series,
            ):
                print(f"Saved: {p}")


if __name__ == "__main__":
    main()
