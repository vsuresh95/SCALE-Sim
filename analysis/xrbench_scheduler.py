#!/usr/bin/env python3
"""Run dynamic multi-tenant scheduling experiments for XRBench-inspired workloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, field
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
SUPPORTED_ALLOC = {"greedy", "fair", "deadline_fold"}
SA_SIZE_FROM_LATENCY = {"lat_128": 128, "lat_64": 64, "lat_32": 32, "lat_16": 16}

# Sentinel deadline for throughput-mode tasks: effectively no deadline.
# Value is large enough that finish_cycle never exceeds it, but small enough
# to avoid any overflow issues in arithmetic comparisons.
THROUGHPUT_NO_DEADLINE = 10**15


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
    mode: str = "latency"   # "latency" | "throughput"
    burst_remaining: int = 0  # requests left to chain after this one (autoregressive)


@dataclass
class ScheduledRequest:
    request: Request
    worker_id: int
    start_cycle: int
    finish_cycle: int
    slack_at_dispatch: int   # deadline - finish_cycle  (actual slack remaining)
    array_cycles: int = 0    # Σ n_arrays × layer_duration across all layers
    dropped: bool = False    # True if evicted mid-inference once deadline is unachievable


@dataclass
class LayerSpec:
    fold_count: int
    fold_cycles: float  # mean cycles per fold


@dataclass
class ActiveJob:
    request: Request
    layer_specs: List[LayerSpec]
    current_layer_idx: int = 0
    job_start_cycle: int = 0
    layer_start_cycle: int = 0
    n_arrays: int = 0
    array_cycles: int = 0    # running Σ n_arrays × layer_duration
    _rem_cycles: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rem_cycles = sum(s.fold_count * s.fold_cycles for s in self.layer_specs)


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
        default=Path("outputs/scheduler_outputs"),
        help="Directory where trace and summary CSVs are written.",
    )
    parser.add_argument(
        "--service-csv",
        type=Path,
        default=Path("outputs/scheduler_outputs/iso_area_summary.csv"),
        help="Stage 1 summary CSV to use as the service-time table.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/xrbench_5_06"),
        help="XRBench results root containing FOLD_REPORT.csv outputs.",
    )
    parser.add_argument(
        "--alloc",
        choices=sorted(SUPPORTED_ALLOC),
        default="greedy",
        help="Array allocation policy for layer-granularity scheduler.",
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
        mode = task_cfg.get("mode", "latency")
        priority = int(task_cfg.get("task_priority", 1))
        service_cycles = int(round(service_table[model][latency_field]))

        if mode == "throughput":
            # Bursts arrive as a Poisson process; size is geometrically distributed.
            # Within each burst requests are autoregressive: the scheduler re-injects
            # the next step only after the previous one finishes (burst_remaining > 0).
            mean_burst_interval_s = float(task_cfg.get("mean_burst_interval_s", 5.0))
            mean_burst_size = int(task_cfg.get("mean_burst_size", 30))
            t = rng.expovariate(1.0 / mean_burst_interval_s)
            while t < duration_s:
                arrival_cycle = seconds_to_cycles(t, cycles_per_second)
                burst_size = max(1, int(rng.expovariate(1.0 / mean_burst_size)))
                requests.append(
                    Request(
                        request_id=request_id,
                        model=model,
                        release_cycle=arrival_cycle,
                        deadline_cycle=THROUGHPUT_NO_DEADLINE,
                        service_cycles=service_cycles,
                        task_priority=priority,
                        fps=0.0,
                        deadline_scale=0.0,
                        mode="throughput",
                        burst_remaining=burst_size - 1,
                    )
                )
                request_id += 1
                t += rng.expovariate(1.0 / mean_burst_interval_s)
            continue

        fps = float(task_cfg.get("fps", 0))
        if fps <= 0:
            continue
        period_s = 1.0 / fps
        period_cycles = seconds_to_cycles(period_s, cycles_per_second)
        deadline_scale = float(task_cfg.get("deadline_scale", default_deadline_scale))
        deadline_cycles = max(1, seconds_to_cycles(period_s * deadline_scale, cycles_per_second))
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
                    mode="latency",
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


def load_fold_specs(results_root: Path, model: str, sa_size: int) -> Optional[List[LayerSpec]]:
    """Load per-layer LayerSpec list from FOLD_REPORT.csv for a given model and SA size."""
    path = results_root / model / f"sa{sa_size}" / f"scale_{sa_size}x{sa_size}_os" / "FOLD_REPORT.csv"
    if not path.exists():
        return None
    layer_data: Dict[int, List[float]] = {}
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            reader.fieldnames = [f.strip() for f in reader.fieldnames]
            for row in reader:
                lid = int(row["LayerID"].strip())
                cyc = float(row["Cycles"].strip())
                layer_data.setdefault(lid, []).append(cyc)
    except (KeyError, ValueError):
        return None
    if not layer_data:
        return None
    specs = []
    for lid in sorted(layer_data.keys()):
        folds = layer_data[lid]
        specs.append(LayerSpec(fold_count=len(folds), fold_cycles=sum(folds) / len(folds)))
    return specs


def layer_time(spec: LayerSpec, n_arrays: int) -> float:
    """Cycles to execute one layer on n_arrays arrays in parallel."""
    return math.ceil(spec.fold_count / n_arrays) * spec.fold_cycles


def remaining_cycles(job: "ActiveJob") -> float:
    """Single-array execution time for all layers not yet started (O(1) cached)."""
    return job._rem_cycles


def min_remaining_cycles(job: "ActiveJob", max_arrays: int) -> float:
    """Optimistic lower bound on remaining cycles assuming max_arrays arrays are available.

    Each layer is parallelised up to min(max_arrays, fold_count) arrays, so this gives
    the fastest the job could possibly finish from the current layer onward.  If this
    value exceeds (deadline - now), the deadline is surely unachievable.
    """
    return sum(
        math.ceil(spec.fold_count / min(max_arrays, spec.fold_count)) * spec.fold_cycles
        for spec in job.layer_specs[job.current_layer_idx:]
    )


def _is_tp(req: Request) -> int:
    return 0 if req.mode == "latency" else 1


def select_request(policy: str, waiting: List[Request], current_cycle: int) -> Request:
    if policy == "fifo":
        return min(waiting, key=lambda req: (_is_tp(req), req.release_cycle, req.request_id))
    if policy == "lpt":
        return min(waiting, key=lambda req: (_is_tp(req), -req.service_cycles, req.release_cycle, req.request_id))
    if policy == "deadline_task_aware":
        return min(
            waiting,
            key=lambda req: (
                _is_tp(req),
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
    duration_cycles: int = 0,
) -> Tuple[List[ScheduledRequest], int]:
    requests_by_release = list(requests)
    waiting: List[Request] = []
    scheduled: List[ScheduledRequest] = []
    worker_available = [0 for _ in range(num_workers)]
    current_cycle = 0
    arrival_idx = 0
    next_req_id = (max(r.request_id for r in requests) + 1) if requests else 0

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
            # Chain next autoregressive step if burst is not exhausted.
            if (request.mode == "throughput"
                    and request.burst_remaining > 0
                    and duration_cycles > 0
                    and finish_cycle < duration_cycles):
                new_req = Request(
                    request_id=next_req_id,
                    model=request.model,
                    release_cycle=finish_cycle,
                    deadline_cycle=THROUGHPUT_NO_DEADLINE,
                    service_cycles=request.service_cycles,
                    task_priority=request.task_priority,
                    fps=0.0,
                    deadline_scale=0.0,
                    mode="throughput",
                    burst_remaining=request.burst_remaining - 1,
                )
                next_req_id += 1
                # Insert in sorted order; re-injection is always >= current_cycle
                # so the insertion point is always at or after arrival_idx.
                lo, hi = arrival_idx, len(requests_by_release)
                while lo < hi:
                    mid = (lo + hi) // 2
                    r = requests_by_release[mid]
                    if (r.release_cycle, r.request_id) < (new_req.release_cycle, new_req.request_id):
                        lo = mid + 1
                    else:
                        hi = mid
                requests_by_release.insert(lo, new_req)

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


def _is_tp_job(job: "ActiveJob") -> int:
    return 0 if job.request.mode == "latency" else 1


def select_active_job(policy: str, ready: List[ActiveJob], current_cycle: int) -> ActiveJob:
    """Select next job from the layer-scheduler ready queue using the given policy.

    Latency-mode jobs always take priority over throughput-mode jobs.
    Within each class, ordering criteria use remaining_cycles (sum of
    single-array layer times for all unfinished layers) so priority
    correctly reflects how much work is left as layers complete.
    """
    if policy == "fifo":
        return min(ready, key=lambda j: (_is_tp_job(j), j.request.release_cycle, j.request.request_id))
    if policy == "lpt":
        return min(ready, key=lambda j: (_is_tp_job(j), -remaining_cycles(j), j.request.release_cycle, j.request.request_id))
    if policy == "deadline_task_aware":
        return min(
            ready,
            key=lambda j: (
                _is_tp_job(j),
                j.request.deadline_cycle - current_cycle - remaining_cycles(j),
                j.request.deadline_cycle,
                -j.request.task_priority,
                -remaining_cycles(j),
                j.request.release_cycle,
                j.request.request_id,
            ),
        )
    raise ValueError(f"unsupported policy: {policy}")


def simulate_scheduler_layer(
    requests: List[Request],
    fold_specs_table: Dict[str, List[LayerSpec]],
    num_workers: int,
    policy: str,
    alloc: str,
    duration_cycles: int = 0,
) -> Tuple[List[ScheduledRequest], int]:
    """Layer-granularity scheduler: allocate arrays dynamically at each layer boundary."""
    requests_by_release = sorted(requests, key=lambda r: (r.release_cycle, r.request_id))
    arrival_idx = 0
    next_req_id = (max(r.request_id for r in requests) + 1) if requests else 0

    free_arrays = num_workers
    ready_queue: List[ActiveJob] = []
    active_jobs: List[Tuple[int, ActiveJob]] = []  # (finish_cycle, job)
    scheduled: List[ScheduledRequest] = []
    current_cycle = 0

    def _start_layer(job: ActiveJob, n: int) -> None:
        nonlocal free_arrays
        spec = job.layer_specs[job.current_layer_idx]
        n = min(n, spec.fold_count)  # cap at useful amount
        free_arrays -= n
        if job.current_layer_idx == 0:
            job.job_start_cycle = current_cycle
        job.layer_start_cycle = current_cycle
        job.n_arrays = n
        duration = int(math.ceil(layer_time(spec, n)))
        job.array_cycles += n * duration
        active_jobs.append((current_cycle + duration, job))

    def _dispatch() -> None:
        if not ready_queue or free_arrays == 0:
            return
        if alloc == "greedy":
            tmp = list(ready_queue)
            while tmp and free_arrays > 0:
                job = select_active_job(policy, tmp, current_cycle)
                tmp.remove(job)
                ready_queue.remove(job)
                n = free_arrays  # give all remaining (capped inside _start_layer)
                _start_layer(job, n)
        elif alloc == "fair":
            n_dispatch = min(len(ready_queue), free_arrays)
            if n_dispatch == 0:
                return
            tmp = list(ready_queue)
            jobs_to_start = []
            for _ in range(n_dispatch):
                job = select_active_job(policy, tmp, current_cycle)
                tmp.remove(job)
                jobs_to_start.append(job)
            base = free_arrays // len(jobs_to_start)
            extra = free_arrays % len(jobs_to_start)
            for i, job in enumerate(jobs_to_start):
                ready_queue.remove(job)
                n = base + (1 if i < extra else 0)
                _start_layer(job, n)
        else:  # deadline_fold
            n_dispatch = min(len(ready_queue), free_arrays)
            if n_dispatch == 0:
                return
            tmp = list(ready_queue)
            jobs_to_start = []
            for _ in range(n_dispatch):
                job = select_active_job(policy, tmp, current_cycle)
                tmp.remove(job)
                jobs_to_start.append(job)

            # Per-job urgency (inverse slack) and per-layer fold cap.
            # Throughput jobs get near-zero urgency so latency jobs absorb
            # arrays first; leftover goes to throughput.
            job_urgency = []
            job_fold_cap = []
            for job in jobs_to_start:
                spec = job.layer_specs[job.current_layer_idx]
                job_fold_cap.append(spec.fold_count)
                if job.request.mode == "throughput":
                    job_urgency.append(0.0)
                else:
                    slack = job.request.deadline_cycle - current_cycle - remaining_cycles(job)
                    job_urgency.append(1.0 / max(float(slack), 1.0))

            # Guarantee 1 array to every selected job; distribute the rest
            # proportionally to urgency, capped at each job's fold_count.
            alloc_n = [1] * len(jobs_to_start)
            budget = free_arrays - len(jobs_to_start)

            # Only jobs whose fold_count > 1 can absorb extra arrays.
            active = [i for i in range(len(jobs_to_start)) if job_fold_cap[i] > 1]
            while budget > 0 and active:
                total_urg = sum(job_urgency[i] for i in active)
                if total_urg == 0.0:
                    # All throughput (or zero-urgency): distribute uniformly.
                    per = budget // len(active)
                    for i in active:
                        alloc_n[i] += min(per, job_fold_cap[i] - alloc_n[i])
                    break

                # Find jobs whose proportional share hits the fold cap;
                # absorb those, then redistribute surplus in the next pass.
                newly_capped = []
                cap_absorbed = 0
                for i in active:
                    share = budget * job_urgency[i] / total_urg
                    max_extra = job_fold_cap[i] - alloc_n[i]
                    if share >= max_extra:
                        alloc_n[i] = job_fold_cap[i]
                        cap_absorbed += max_extra
                        newly_capped.append(i)

                if newly_capped:
                    budget -= cap_absorbed
                    active = [i for i in active if i not in newly_capped]
                else:
                    # No caps hit: floor-allocate, give remainder to highest urgency.
                    distributed = 0
                    for i in active:
                        extra = int(budget * job_urgency[i] / total_urg)
                        alloc_n[i] += extra
                        distributed += extra
                    budget -= distributed
                    for i in sorted(active, key=lambda x: -job_urgency[x]):
                        if budget <= 0:
                            break
                        if alloc_n[i] < job_fold_cap[i]:
                            alloc_n[i] += 1
                            budget -= 1
                    break

            for job, n in zip(jobs_to_start, alloc_n):
                ready_queue.remove(job)
                _start_layer(job, n)

    while arrival_idx < len(requests_by_release) or ready_queue or active_jobs:
        # Admit arrivals at current cycle
        while (
            arrival_idx < len(requests_by_release)
            and requests_by_release[arrival_idx].release_cycle <= current_cycle
        ):
            req = requests_by_release[arrival_idx]
            arrival_idx += 1
            specs = fold_specs_table.get(req.model)
            if specs is None:
                continue
            ready_queue.append(ActiveJob(request=req, layer_specs=specs))

        # Process layer completions
        still_active: List[Tuple[int, ActiveJob]] = []
        for finish_cycle, job in active_jobs:
            if finish_cycle <= current_cycle:
                free_arrays += job.n_arrays
                # Update cached remaining cycles before advancing the layer index
                completed_spec = job.layer_specs[job.current_layer_idx]
                job._rem_cycles -= completed_spec.fold_count * completed_spec.fold_cycles
                job.current_layer_idx += 1
                if job.current_layer_idx >= len(job.layer_specs):
                    scheduled.append(
                        ScheduledRequest(
                            request=job.request,
                            worker_id=job.n_arrays,
                            start_cycle=job.job_start_cycle,
                            finish_cycle=finish_cycle,
                            slack_at_dispatch=job.request.deadline_cycle - finish_cycle,
                            array_cycles=job.array_cycles,
                        )
                    )
                    # Chain the next autoregressive step if the burst is not done.
                    if (job.request.mode == "throughput"
                            and job.request.burst_remaining > 0
                            and duration_cycles > 0
                            and finish_cycle < duration_cycles):
                        specs = fold_specs_table.get(job.request.model)
                        if specs is not None:
                            new_req = Request(
                                request_id=next_req_id,
                                model=job.request.model,
                                release_cycle=finish_cycle,
                                deadline_cycle=THROUGHPUT_NO_DEADLINE,
                                service_cycles=job.request.service_cycles,
                                task_priority=job.request.task_priority,
                                fps=0.0,
                                deadline_scale=0.0,
                                mode="throughput",
                                burst_remaining=job.request.burst_remaining - 1,
                            )
                            next_req_id += 1
                            ready_queue.append(ActiveJob(request=new_req, layer_specs=specs))
                else:
                    # Drop if the deadline is surely unachievable: even with every
                    # available array the remaining layers cannot finish in time.
                    time_left = job.request.deadline_cycle - current_cycle
                    if time_left < min_remaining_cycles(job, num_workers):
                        scheduled.append(
                            ScheduledRequest(
                                request=job.request,
                                worker_id=job.n_arrays,
                                start_cycle=job.job_start_cycle,
                                finish_cycle=current_cycle,
                                slack_at_dispatch=time_left,
                                array_cycles=job.array_cycles,
                                dropped=True,
                            )
                        )
                    else:
                        ready_queue.append(job)
            else:
                still_active.append((finish_cycle, job))
        active_jobs = still_active

        _dispatch()

        # Advance to next event
        pending: List[int] = []
        if arrival_idx < len(requests_by_release):
            pending.append(requests_by_release[arrival_idx].release_cycle)
        for fc, _ in active_jobs:
            pending.append(fc)
        if pending:
            current_cycle = min(pending)
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
    completed = [item for item in scheduled if not item.dropped]

    response_cycles = [item.finish_cycle - item.request.release_cycle for item in completed]
    wait_cycles = [item.start_cycle - item.request.release_cycle for item in completed]
    deadline_misses = sum(
        1 for item in scheduled
        if item.dropped or item.finish_cycle > item.request.deadline_cycle
    )
    # Use actual array-cycles consumed if tracked (layer scheduler), else fall back to
    # service_cycles × 1 array (model scheduler).
    total_busy_cycles = sum(
        item.array_cycles if item.array_cycles > 0 else item.request.service_cycles
        for item in scheduled
    )
    last_finish = max((item.finish_cycle for item in scheduled), default=0)
    horizon_cycles = max(duration_cycles, last_finish)

    n_total = len(scheduled)
    n_completed = len(completed)
    response_cycles_sorted = sorted(response_cycles)
    wait_cycles_sorted = sorted(wait_cycles)
    summary = {
        "arrivals": n_total,
        "completions": n_completed,
        "deadline_misses": deadline_misses,
        "miss_rate": (deadline_misses / n_total) if n_total else 0.0,
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
        "array_cycles",
        "deadline_miss",
        "mode",
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
            deadline_miss = int(item.finish_cycle > item.request.deadline_cycle or item.dropped)
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
                    "array_cycles": item.array_cycles,
                    "deadline_miss": deadline_miss,
                    "mode": item.request.mode,
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


def write_model_logs(
    scheduled: List[ScheduledRequest],
    requests: List[Request],
    output_dir: Path,
    output_prefix: str,
    scenario_name: str,
    policy: str,
    hardware: str,
    cycles_per_second: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Index simulation results by request_id for O(1) lookup
    result_by_id: Dict[int, ScheduledRequest] = {item.request.request_id: item for item in scheduled}

    # Group requests by model
    models: Dict[str, List[Request]] = {}
    for req in requests:
        models.setdefault(req.model, []).append(req)

    def _cy(cycles: int) -> str:
        return f"{cycles_to_seconds(cycles, cycles_per_second)*1000:.3f}ms"

    for model, reqs in sorted(models.items()):
        log_path = output_dir / f"{output_prefix}_{model}_log.txt"
        violations = sum(
            1 for r in reqs
            if result_by_id[r.request_id].dropped
            or result_by_id[r.request_id].finish_cycle > r.deadline_cycle
        )
        hit_reqs = [r for r in reqs if not result_by_id[r.request_id].dropped
                    and result_by_id[r.request_id].finish_cycle <= r.deadline_cycle]

        with log_path.open("w") as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write(f"  MODEL        : {model}\n")
            f.write(f"  SCENARIO     : {scenario_name}\n")
            f.write(f"  HARDWARE     : {hardware}\n")
            f.write(f"  POLICY       : {policy}\n")
            f.write(f"  CLOCK        : {cycles_per_second/1e6:.0f} MHz\n")
            f.write(f"  TOTAL REQS   : {len(reqs)}\n")
            f.write(f"  DEADLINE MISS: {violations} / {len(reqs)} ({violations/len(reqs)*100:.1f}%)\n")
            first = reqs[0]
            f.write(f"  MODE         : {first.mode}\n")
            f.write(f"  SERVICE TIME : {_cy(first.service_cycles)}\n")
            if first.mode == "throughput":
                f.write(f"  DEADLINE WIN : n/a (throughput task)\n")
            else:
                f.write(f"  DEADLINE WIN : {_cy(int(first.deadline_scale * cycles_per_second / first.fps))}\n")
            f.write("=" * 80 + "\n\n")

            # Request trace
            f.write("--- REQUEST TRACE ---\n")
            max_finish = max((result_by_id[r.request_id].finish_cycle for r in reqs), default=0)
            cy_w = max(14, len(_cy(max_finish)))
            hdr = (f"{'req_id':>7}  {'release':>{cy_w}}  {'deadline':>{cy_w}}  "
                   f"{'worker':>6}  {'start':>{cy_w}}  {'finish':>{cy_w}}  "
                   f"{'wait':>{cy_w}}  {'slack@dispatch(ms)':>19}  {'miss':>4}\n")
            f.write(hdr)
            f.write("-" * len(hdr.rstrip()) + "\n")

            for req in sorted(reqs, key=lambda r: r.release_cycle):
                item = result_by_id[req.request_id]
                wait = item.start_cycle - req.release_cycle
                miss = item.dropped or item.finish_cycle > req.deadline_cycle
                f.write(
                    f"{req.request_id:>7}  "
                    f"{_cy(req.release_cycle):>{cy_w}}  "
                    f"{_cy(req.deadline_cycle):>{cy_w}}  "
                    f"{item.worker_id:>6}  "
                    f"{_cy(item.start_cycle):>{cy_w}}  "
                    f"{_cy(item.finish_cycle):>{cy_w}}  "
                    f"{_cy(wait):>{cy_w}}  "
                    f"{cycles_to_seconds(item.slack_at_dispatch, cycles_per_second)*1000:>+16.3f}ms  "
                    f"{'MISS' if miss else 'ok':>4}\n"
                )

            # Per-model summary stats (on-time completions only)
            waits = [result_by_id[r.request_id].start_cycle - r.release_cycle for r in hit_reqs]
            responses = [result_by_id[r.request_id].finish_cycle - r.release_cycle for r in hit_reqs]
            f.write("\n--- MODEL SUMMARY ---\n")
            if waits:
                f.write(f"  mean wait    : {_cy(int(sum(waits)/len(waits)))}\n")
                f.write(f"  max wait     : {_cy(max(waits))}\n")
            else:
                f.write(f"  mean wait    : n/a (all missed)\n")
                f.write(f"  max wait     : n/a\n")
            if responses:
                f.write(f"  mean response: {_cy(int(sum(responses)/len(responses)))}\n")
                f.write(f"  max response : {_cy(max(responses))}\n")
            else:
                f.write(f"  mean response: n/a (all missed)\n")
                f.write(f"  max response : n/a\n")


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
    cycles_per_second = int(scenario["cycles_per_second"])

    hardware_cfg = HARDWARE_CONFIGS[args.hardware]
    sa_size = SA_SIZE_FROM_LATENCY[hardware_cfg["latency_field"]]

    # Load fold specs for layer-granularity scheduling
    models_needed = set(r.model for r in requests)
    fold_specs_table: Dict[str, List[LayerSpec]] = {}
    for model in models_needed:
        specs = load_fold_specs(args.results_root, model, sa_size)
        if specs is not None:
            fold_specs_table[model] = specs
        elif not args.quiet:
            print(f"INFO: fold specs missing for {model} sa{sa_size}, will fall back to model scheduler")

    if fold_specs_table.keys() >= models_needed:
        scheduled, _ = simulate_scheduler_layer(
            requests, fold_specs_table, num_workers, args.policy, args.alloc,
            duration_cycles=duration_cycles,
        )
        scheduler_mode = f"layer/{args.alloc}"
    else:
        scheduled, _ = simulate_scheduler(requests, num_workers, args.policy, duration_cycles)
        scheduler_mode = "model"

    if not args.quiet:
        print(f"INFO: scheduler_mode={scheduler_mode}")

    summary = summarize_run(scheduled, duration_cycles, num_workers)

    scenario_name = str(scenario.get("name", args.scenario.stem))
    output_prefix = f"{scenario_name}_{args.hardware}_{args.policy}_{args.alloc}"
    trace_path = args.output_dir / f"{output_prefix}_trace.csv"
    summary_path = args.output_dir / f"{output_prefix}_summary.csv"

    write_trace_csv(scheduled, trace_path, cycles_per_second)
    write_summary_csv(summary, summary_path, scenario_name, args.policy, args.hardware, cycles_per_second)
    write_model_logs(scheduled, requests, args.output_dir, output_prefix,
                     scenario_name, args.policy, args.hardware, cycles_per_second)
    print_run_summary(summary, scenario_name, args.policy, args.hardware, trace_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
