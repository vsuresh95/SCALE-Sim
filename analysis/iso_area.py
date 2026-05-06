#!/usr/bin/env python3
"""Build an iso-area service-time summary from XRBench ScaleSim outputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


DEFAULT_MODELS = ["hand_pose", "eyecod", "key_res15", "tcn", "d2go", "emformer", "deit_small"]
DEFAULT_SA_SIZES = [16, 32, 64, 128]


@dataclass
class ReportData:
    total_cycles: List[int]
    overall_util_pct: List[float]


@dataclass
class IsoAreaRow:
    model: str
    lat_128: int
    lat_64: int
    lat_32: int
    lat_16: int
    slowdown_64: float
    slowdown_32: float
    slowdown_16: float
    throughput_gain_4x64: float
    throughput_gain_16x32: float
    throughput_gain_64x16: float
    util_128: float
    util_64: float
    util_32: float
    util_16: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/xrbench"),
        help="Root directory containing XRBench ScaleSim outputs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/iso_area_summary.csv"),
        help="Path to write the iso-area summary CSV.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to include in the iso-area summary.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress missing-file warnings.",
    )
    return parser.parse_args()


def print_warning(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"WARNING: {message}")


def report_path(results_root: Path, model: str, sa_size: int) -> Path:
    run_name = f"scale_{sa_size}x{sa_size}_ws"
    return results_root / model / f"sa{sa_size}" / run_name / "COMPUTE_REPORT.csv"


def load_compute_report(path: Path) -> ReportData:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        reader.fieldnames = [field.strip() for field in reader.fieldnames]

        total_cycles: List[int] = []
        overall_util_pct: List[float] = []
        for row in reader:
            normalized = {key.strip(): value for key, value in row.items() if key is not None}
            total_cycles.append(int(float(normalized["Total Cycles"])))
            overall_util_pct.append(float(normalized["Overall Util %"]))

    if not total_cycles:
        raise ValueError(f"{path} contains no layer rows")

    return ReportData(total_cycles=total_cycles, overall_util_pct=overall_util_pct)


def maybe_load_reports(
    results_root: Path,
    model: str,
    sa_sizes: Iterable[int],
    quiet: bool = False,
) -> Optional[Dict[int, ReportData]]:
    reports: Dict[int, ReportData] = {}
    for sa_size in sa_sizes:
        path = report_path(results_root, model, sa_size)
        if not path.exists():
            print_warning(f"missing compute report for {model} sa{sa_size}: {path}", quiet)
            return None
        try:
            reports[sa_size] = load_compute_report(path)
        except (KeyError, ValueError) as exc:
            print_warning(f"failed to parse {path}: {exc}", quiet)
            return None
    return reports


def build_iso_area_row(model: str, reports: Dict[int, ReportData]) -> IsoAreaRow:
    lat_128 = sum(reports[128].total_cycles)
    lat_64 = sum(reports[64].total_cycles)
    lat_32 = sum(reports[32].total_cycles)
    lat_16 = sum(reports[16].total_cycles)

    return IsoAreaRow(
        model=model,
        lat_128=lat_128,
        lat_64=lat_64,
        lat_32=lat_32,
        lat_16=lat_16,
        slowdown_64=lat_64 / lat_128,
        slowdown_32=lat_32 / lat_128,
        slowdown_16=lat_16 / lat_128,
        throughput_gain_4x64=(4.0 * lat_128) / lat_64,
        throughput_gain_16x32=(16.0 * lat_128) / lat_32,
        throughput_gain_64x16=(64.0 * lat_128) / lat_16,
        util_128=mean(reports[128].overall_util_pct) / 100.0,
        util_64=mean(reports[64].overall_util_pct) / 100.0,
        util_32=mean(reports[32].overall_util_pct) / 100.0,
        util_16=mean(reports[16].overall_util_pct) / 100.0,
    )


def build_iso_area_summary(
    results_root: Path,
    models: Iterable[str],
    quiet: bool = False,
) -> List[IsoAreaRow]:
    rows: List[IsoAreaRow] = []
    for model in models:
        reports = maybe_load_reports(results_root, model, DEFAULT_SA_SIZES, quiet=quiet)
        if reports is None:
            print_warning(f"skipping incomplete model: {model}", quiet)
            continue
        rows.append(build_iso_area_row(model, reports))
    return rows


def write_summary_csv(rows: Iterable[IsoAreaRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "lat_128",
        "lat_64",
        "lat_32",
        "lat_16",
        "slowdown_64",
        "slowdown_32",
        "slowdown_16",
        "throughput_gain_4x64",
        "throughput_gain_16x32",
        "throughput_gain_64x16",
        "util_128",
        "util_64",
        "util_32",
        "util_16",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row.model,
                    "lat_128": row.lat_128,
                    "lat_64": row.lat_64,
                    "lat_32": row.lat_32,
                    "lat_16": row.lat_16,
                    "slowdown_64": f"{row.slowdown_64:.6f}",
                    "slowdown_32": f"{row.slowdown_32:.6f}",
                    "slowdown_16": f"{row.slowdown_16:.6f}",
                    "throughput_gain_4x64": f"{row.throughput_gain_4x64:.6f}",
                    "throughput_gain_16x32": f"{row.throughput_gain_16x32:.6f}",
                    "throughput_gain_64x16": f"{row.throughput_gain_64x16:.6f}",
                    "util_128": f"{row.util_128:.6f}",
                    "util_64": f"{row.util_64:.6f}",
                    "util_32": f"{row.util_32:.6f}",
                    "util_16": f"{row.util_16:.6f}",
                }
            )


def print_summary_table(rows: List[IsoAreaRow]) -> None:
    if not rows:
        print("No complete models found; summary not generated.")
        return

    headers = [
        "model",
        "lat_128",
        "lat_64",
        "lat_32",
        "lat_16",
        "slowdown_64",
        "slowdown_32",
        "slowdown_16",
        "throughput_gain_4x64",
        "throughput_gain_16x32",
        "throughput_gain_64x16",
        "util_128",
        "util_64",
        "util_32",
        "util_16",
    ]
    table_rows = [
        [
            row.model,
            str(row.lat_128),
            str(row.lat_64),
            str(row.lat_32),
            str(row.lat_16),
            f"{row.slowdown_64:.3f}",
            f"{row.slowdown_32:.3f}",
            f"{row.slowdown_16:.3f}",
            f"{row.throughput_gain_4x64:.3f}",
            f"{row.throughput_gain_16x32:.3f}",
            f"{row.throughput_gain_64x16:.3f}",
            f"{row.util_128:.3f}",
            f"{row.util_64:.3f}",
            f"{row.util_32:.3f}",
            f"{row.util_16:.3f}",
        ]
        for row in rows
    ]

    widths = []
    for col_idx, header in enumerate(headers):
        width = max(len(header), *(len(data_row[col_idx]) for data_row in table_rows))
        widths.append(width)

    def fmt(cells: List[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    print(fmt(headers))
    print(fmt(["-" * width for width in widths]))
    for table_row in table_rows:
        print(fmt(table_row))


def main() -> int:
    args = parse_args()
    rows = build_iso_area_summary(args.results_root, args.models, quiet=args.quiet)
    write_summary_csv(rows, args.output_csv)
    print_summary_table(rows)
    print(f"\nWrote {len(rows)} model summaries to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
