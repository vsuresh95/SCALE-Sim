#!/usr/bin/env python3
"""Run dynamic multi-tenant scheduling experiments for XRBench-inspired workloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from iso_area import DEFAULT_MODELS, build_iso_area_summary


HARDWARE_CONFIGS = {
    "mono128": {"num_workers": 1, "latency_field": "lat_128"},
    "comp4x64": {"num_workers": 4, "latency_field": "lat_64"},
    "comp16x32": {"num_workers": 16, "latency_field": "lat_32"},
    "comp64x16": {"num_workers": 64, "latency_field": "lat_16"},
}

SUPPORTED_POLICIES = {"fifo", "lpt", "deadline_task_aware"}


@dataclass
class Request:
    request_id: int
    model: str
    release_cycle: int
    deadline_cycle: int
    service_cycles: int
    task_priority: int
    fps: float
    deadline_scale: float


@dataclass
class ScheduledRequest:
    request: Request
    worker_id: int
    start_cycle: int
    finish_cycle: int
    slack_at_dispatch: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("analysis/scenarios/xrbench_subset_5model.json"),
        help="Path to the scenario JSON file.",
    )
    parser.add_argument(
        "--policy",
        choices=sorted(SUPPORTED_POLICIES),
        default="fifo",
        help="Scheduler policy to evaluate.",
    )
    parser.add_argument(
        "--hardware",
        choices=sorted(HARDWARE_CONFIGS),
        default="mono128",
        help="Hardware pool to simulate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/scheduler_outputs"),
        help="Directory where trace and summary CSVs are written.",
    )
    parser.add_argument(
        "--service-csv",
        type=Path,
        default=Path("analysis/iso_area_summary.csv"),
        help="Stage 1 summary CSV to use as the service-time table.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/xrbench"),
        help="XRBench results root used if the service summary CSV must be rebuilt.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational messages.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def ensure_service_table(service_csv: Path, results_root: Path, quiet: bool) -> Dict[str, Dict[str, float]]:
    if service_csv.exists():
        return load_service_table_csv(service_csv)

    rows = build_iso_area_summary(results_root, DEFAULT_MODELS, quiet=quiet)
    if not rows:
        raise FileNotFoundError(
            f"service summary {service_csv} is missing and could not be rebuilt from {results_root}"
        )

    if not quiet:
        print(f"INFO: rebuilding missing service summary from {results_root}")
    write_iso_area_csv(rows, service_csv)
    return load_service_table_csv(service_csv)


def write_iso_area_csv(rows, output_csv: Path) -> None:
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


def load_service_table_csv(path: Path) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row["model"]
            table[model] = {
                "lat_128": float(row["lat_128"]),
                "lat_64": float(row["lat_64"]),
                "lat_32": float(row["lat_32"]),
                "lat_16": float(row["lat_16"]),
                "util_128": float(row["util_128"]),
                "util_64": float(row["util_64"]),
                "util_32": float(row["util_32"]),
                "util_16": float(row["util_16"]),
            }
    return table


def build_requests(
    scenario: dict,
    service_table: Dict[str, Dict[str, float]],
    hardware: str,
) -> Tuple[List[Request], int]:
    hardware_cfg = HARDWARE_CONFIGS[hardware]
    latency_field = hardware_cfg["latency_field"]
    cycles_per_second = int(scenario["cycles_per_second"])
    duration_s = float(scenario["duration_s"])
    duration_cycles = seconds_to_cycles(duration_s, cycles_per_second)
    jitter_fraction = float(scenario.get("jitter_fraction", 0.0))
    default_deadline_scale = float(scenario.get("deadline_scale", 1.0))
    random_seed = int(scenario.get("random_seed", 0))
    rng = random.Random(random_seed)

    requests: List[Request] = []
    request_id = 0
    tasks = scenario["tasks"]
    for model, task_cfg in tasks.items():
        if model not in service_table:
            raise KeyError(f"model {model} missing from service table")
        fps = float(task_cfg["fps"])
        if fps <= 0:
            continue
        priority = int(task_cfg.get("task_priority", 1))
        period_s = 1.0 / fps
        period_cycles = seconds_to_cycles(period_s, cycles_per_second)
        deadline_scale = float(task_cfg.get("deadline_scale", default_deadline_scale))
        deadline_cycles = max(1, seconds_to_cycles(period_s * deadline_scale, cycles_per_second))
        service_cycles = int(round(service_table[model][latency_field]))
        arrivals = generate_arrival_cycles(
            duration_s=duration_s,
            period_s=period_s,
            cycles_per_second=cycles_per_second,
            jitter_fraction=jitter_fraction,
            rng=rng,
        )
        for release_cycle in arrivals:
            deadline_cycle = release_cycle + deadline_cycles
            requests.append(
                Request(
                    request_id=request_id,
                    model=model,
                    release_cycle=release_cycle,
                    deadline_cycle=deadline_cycle,
                    service_cycles=service_cycles,
                    task_priority=priority,
                    fps=fps,
                    deadline_scale=deadline_scale,
                )
            )
            request_id += 1

    requests.sort(key=lambda req: (req.release_cycle, req.request_id))
    return requests, duration_cycles


def generate_arrival_cycles(
    duration_s: float,
    period_s: float,
    cycles_per_second: int,
    jitter_fraction: float,
    rng: random.Random,
) -> List[int]:
    arrivals: List[int] = []
    count = int(math.ceil(duration_s / period_s - 1e-12))
    jitter_span_s = jitter_fraction * period_s
    for idx in range(count):
        nominal_s = idx * period_s
        jitter_s = 0.0
        if jitter_span_s > 0:
            jitter_s = rng.uniform(-jitter_span_s, jitter_span_s)
        actual_s = nominal_s + jitter_s
        actual_s = min(max(actual_s, 0.0), max(duration_s - 1e-12, 0.0))
        arrivals.append(seconds_to_cycles(actual_s, cycles_per_second))
    arrivals.sort()
    return arrivals


def seconds_to_cycles(seconds: float, cycles_per_second: int) -> int:
    return int(round(seconds * cycles_per_second))


def cycles_to_seconds(cycles: int, cycles_per_second: int) -> float:
    return cycles / float(cycles_per_second)


def select_request(policy: str, waiting: List[Request], current_cycle: int) -> Request:
    if policy == "fifo":
        return min(waiting, key=lambda req: (req.release_cycle, req.request_id))
    if policy == "lpt":
        return min(waiting, key=lambda req: (-req.service_cycles, req.release_cycle, req.request_id))
    if policy == "deadline_task_aware":
        return min(
            waiting,
            key=lambda req: (
                req.deadline_cycle - current_cycle - req.service_cycles,
                req.deadline_cycle,
                -req.task_priority,
                -req.service_cycles,
                req.release_cycle,
                req.request_id,
            ),
        )
    raise ValueError(f"unsupported policy: {policy}")


def simulate_scheduler(
    requests: List[Request],
    num_workers: int,
    policy: str,
) -> Tuple[List[ScheduledRequest], int]:
    requests_by_release = list(requests)
    waiting: List[Request] = []
    scheduled: List[ScheduledRequest] = []
    worker_available = [0 for _ in range(num_workers)]
    current_cycle = 0
    arrival_idx = 0

    while arrival_idx < len(requests_by_release) or waiting or any(
        available > current_cycle for available in worker_available
    ):
        while arrival_idx < len(requests_by_release) and requests_by_release[arrival_idx].release_cycle <= current_cycle:
            waiting.append(requests_by_release[arrival_idx])
            arrival_idx += 1

        free_workers = [wid for wid, available in enumerate(worker_available) if available <= current_cycle]
        while waiting and free_workers:
            worker_id = free_workers.pop(0)
            request = select_request(policy, waiting, current_cycle)
            waiting.remove(request)
            start_cycle = current_cycle
            finish_cycle = start_cycle + request.service_cycles
            worker_available[worker_id] = finish_cycle
            slack_at_dispatch = request.deadline_cycle - start_cycle - request.service_cycles
            scheduled.append(
                ScheduledRequest(
                    request=request,
                    worker_id=worker_id,
                    start_cycle=start_cycle,
                    finish_cycle=finish_cycle,
                    slack_at_dispatch=slack_at_dispatch,
                )
            )

        pending_times = []
        if arrival_idx < len(requests_by_release):
            pending_times.append(requests_by_release[arrival_idx].release_cycle)
        for available in worker_available:
            if available > current_cycle:
                pending_times.append(available)

        if pending_times:
            current_cycle = min(pending_times)
        else:
            break

    final_cycle = max((item.finish_cycle for item in scheduled), default=0)
    scheduled.sort(key=lambda item: item.request.request_id)
    return scheduled, final_cycle


def percentile(sorted_values: List[int], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(sorted_values[lower])
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (rank - lower)


def summarize_run(
    scheduled: List[ScheduledRequest],
    duration_cycles: int,
    num_workers: int,
) -> Dict[str, float]:
    response_cycles = [item.finish_cycle - item.request.release_cycle for item in scheduled]
    wait_cycles = [item.start_cycle - item.request.release_cycle for item in scheduled]
    deadline_misses = sum(1 for item in scheduled if item.finish_cycle > item.request.deadline_cycle)
    total_busy_cycles = sum(item.request.service_cycles for item in scheduled)
    last_finish = max((item.finish_cycle for item in scheduled), default=0)
    horizon_cycles = max(duration_cycles, last_finish)

    response_cycles_sorted = sorted(response_cycles)
    wait_cycles_sorted = sorted(wait_cycles)
    summary = {
        "arrivals": len(scheduled),
        "completions": len(scheduled),
        "deadline_misses": deadline_misses,
        "miss_rate": (deadline_misses / len(scheduled)) if scheduled else 0.0,
        "mean_response_cycles": (sum(response_cycles) / len(response_cycles)) if response_cycles else 0.0,
        "p95_response_cycles": percentile(response_cycles_sorted, 0.95),
        "max_response_cycles": max(response_cycles, default=0),
        "mean_wait_cycles": (sum(wait_cycles) / len(wait_cycles)) if wait_cycles else 0.0,
        "p95_wait_cycles": percentile(wait_cycles_sorted, 0.95),
        "worker_utilization": (
            total_busy_cycles / float(num_workers * horizon_cycles) if horizon_cycles > 0 else 0.0
        ),
        "last_finish_cycle": last_finish,
        "horizon_cycles": horizon_cycles,
    }
    return summary


def write_trace_csv(
    scheduled: List[ScheduledRequest],
    output_path: Path,
    cycles_per_second: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "request_id",
        "model",
        "fps",
        "deadline_scale",
        "task_priority",
        "worker_id",
        "release_cycle",
        "start_cycle",
        "finish_cycle",
        "deadline_cycle",
        "service_cycles",
        "wait_cycles",
        "response_cycles",
        "slack_at_dispatch_cycles",
        "deadline_miss",
        "release_s",
        "start_s",
        "finish_s",
        "deadline_s",
        "wait_s",
        "response_s",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in scheduled:
            wait_cycles = item.start_cycle - item.request.release_cycle
            response_cycles = item.finish_cycle - item.request.release_cycle
            deadline_miss = int(item.finish_cycle > item.request.deadline_cycle)
            writer.writerow(
                {
                    "request_id": item.request.request_id,
                    "model": item.request.model,
                    "fps": item.request.fps,
                    "deadline_scale": item.request.deadline_scale,
                    "task_priority": item.request.task_priority,
                    "worker_id": item.worker_id,
                    "release_cycle": item.request.release_cycle,
                    "start_cycle": item.start_cycle,
                    "finish_cycle": item.finish_cycle,
                    "deadline_cycle": item.request.deadline_cycle,
                    "service_cycles": item.request.service_cycles,
                    "wait_cycles": wait_cycles,
                    "response_cycles": response_cycles,
                    "slack_at_dispatch_cycles": item.slack_at_dispatch,
                    "deadline_miss": deadline_miss,
                    "release_s": f"{cycles_to_seconds(item.request.release_cycle, cycles_per_second):.9f}",
                    "start_s": f"{cycles_to_seconds(item.start_cycle, cycles_per_second):.9f}",
                    "finish_s": f"{cycles_to_seconds(item.finish_cycle, cycles_per_second):.9f}",
                    "deadline_s": f"{cycles_to_seconds(item.request.deadline_cycle, cycles_per_second):.9f}",
                    "wait_s": f"{cycles_to_seconds(wait_cycles, cycles_per_second):.9f}",
                    "response_s": f"{cycles_to_seconds(response_cycles, cycles_per_second):.9f}",
                }
            )


def write_summary_csv(
    summary: Dict[str, float],
    output_path: Path,
    scenario_name: str,
    policy: str,
    hardware: str,
    cycles_per_second: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "policy",
        "hardware",
        "cycles_per_second",
        "arrivals",
        "completions",
        "deadline_misses",
        "miss_rate",
        "mean_response_cycles",
        "p95_response_cycles",
        "max_response_cycles",
        "mean_wait_cycles",
        "p95_wait_cycles",
        "worker_utilization",
        "last_finish_cycle",
        "horizon_cycles",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "scenario": scenario_name,
                "policy": policy,
                "hardware": hardware,
                "cycles_per_second": cycles_per_second,
                **summary,
            }
        )


def print_run_summary(
    summary: Dict[str, float],
    scenario_name: str,
    policy: str,
    hardware: str,
    trace_path: Path,
    summary_path: Path,
) -> None:
    print(f"scenario={scenario_name} policy={policy} hardware={hardware}")
    print(
        "arrivals={arrivals} completions={completions} misses={deadline_misses} miss_rate={miss_rate:.3f} "
        "mean_resp={mean_response_cycles:.1f}cy p95_resp={p95_response_cycles:.1f}cy "
        "mean_wait={mean_wait_cycles:.1f}cy util={worker_utilization:.3f}".format(**summary)
    )
    print(f"trace_csv={trace_path}")
    print(f"summary_csv={summary_path}")


def main() -> int:
    args = parse_args()
    scenario = load_json(args.scenario)
    service_table = ensure_service_table(args.service_csv, args.results_root, args.quiet)
    requests, duration_cycles = build_requests(scenario, service_table, args.hardware)
    num_workers = HARDWARE_CONFIGS[args.hardware]["num_workers"]
    scheduled, _ = simulate_scheduler(requests, num_workers, args.policy)
    summary = summarize_run(scheduled, duration_cycles, num_workers)

    scenario_name = str(scenario.get("name", args.scenario.stem))
    output_prefix = f"{scenario_name}_{args.hardware}_{args.policy}"
    trace_path = args.output_dir / f"{output_prefix}_trace.csv"
    summary_path = args.output_dir / f"{output_prefix}_summary.csv"
    cycles_per_second = int(scenario["cycles_per_second"])

    write_trace_csv(scheduled, trace_path, cycles_per_second)
    write_summary_csv(summary, summary_path, scenario_name, args.policy, args.hardware, cycles_per_second)
    print_run_summary(summary, scenario_name, args.policy, args.hardware, trace_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
