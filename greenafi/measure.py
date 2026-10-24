"""Timing, FLOP counting and energy for one workload."""
from __future__ import annotations

import gc
import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode


def count_flops(fn) -> int:
    """Exact FLOPs of one call (matmuls, convolutions, attention), forward and
    backward if fn calls backward. Hardware-independent."""
    with FlopCounterMode(display=False) as fc:
        fn()
    return int(fc.get_total_flops())


def measure(fn, meter, idle_w, warmup=1, min_seconds=3.0, min_reps=3, max_reps=200) -> dict:
    """Run fn repeatedly; return median seconds per call and net energy per call."""
    for _ in range(warmup):
        fn()
    gc.collect()
    times = []
    meter.start()
    t_start = time.perf_counter()
    while len(times) < min_reps or (time.perf_counter() - t_start < min_seconds and len(times) < max_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    r = meter.stop()
    out = {
        "reps": len(times),
        "median_s": statistics.median(times),
        "mean_s": statistics.fmean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "joules_per_call": None,
        "energy_source": meter.name,
    }
    if r["joules"] is not None:
        net = r["joules"] - (idle_w or 0.0) * r["seconds"]
        out["joules_per_call"] = max(net, 0.0) / len(times)
        out["avg_power_w"] = r["joules"] / r["seconds"]
        out["idle_power_w"] = idle_w
    return out


def setup_threads(n: int | None):
    if n:
        torch.set_num_threads(n)
    return torch.get_num_threads()
