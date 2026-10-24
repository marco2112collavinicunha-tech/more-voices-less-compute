"""Summarise the prototype's replay logs (demo/live_demo.py, torgo mode).

Latency is wall-clock time measured on the author's laptop (Apple M5, CPU) for
embedding + classification of one test recording; it is a measurement, unlike
the energy figures, which are estimates. Speaker labels come from the run log
(results/RUNLOG_mac14.txt), recorded here explicitly.
"""
import json, statistics
from pathlib import Path

RUNS = {  # log file -> (speaker, int8), from RUNLOG_mac14.txt
    "demo_torgo_20261021_083027.jsonl": ("M04", False),
    "demo_torgo_20261021_083036.jsonl": ("F03", False),
    "demo_torgo_20261021_083045.jsonl": ("M04", True),
}

def uar(rows):
    by = {}
    for r in rows:
        by.setdefault(r["truth"], []).append(r["pred"] == r["truth"])
    return 100 * statistics.mean(statistics.mean(v) for v in by.values())

def main():
    d = Path("results/demo"); out = []
    for f, (spk, q) in RUNS.items():
        rows = [json.loads(l) for l in (d / f).read_text().splitlines() if l.strip()]
        lat = [1000 * r["latency_s"] for r in rows]
        out.append(dict(file=f, speaker=spk, int8=q, n=len(rows), uar=uar(rows),
                        latency_ms_median=statistics.median(lat),
                        latency_ms_max=max(lat),
                        gflop_ratio_whisper_medium=statistics.mean(r["gflop_whisper_medium"] for r in rows)
                        / statistics.mean(r["gflop"] for r in rows)))
    (d / "demo_summary.json").write_text(json.dumps(out, indent=1))
    for o in out: print(o)

if __name__ == "__main__":
    main()
