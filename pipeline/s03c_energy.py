"""Stage 3c — energy from computation, anchored on published measurements.

  python -m pipeline.s03c_energy [--cost-dir results/vm_random_weights]

Until a hardware meter is used, energy is estimated as
    E = FLOPs x (joules per GFLOP of a device class),
with joules per GFLOP taken from published measurements (config/energy_anchors.json):
  * low-cost ARM board: Gondi (2022) measured 11 mWh for wav2vec 2.0 Base on a 10 s
    utterance on a Raspberry Pi 4; we count that workload's FLOPs with the same counter
    used for every system here, which gives J/GFLOP for that board.
  * Apple laptop CPU: measured XLSR-53 throughput on the author's MacBook (GFLOP/s,
    results/s03c/apple_throughput.json, if present) x published package power
    (13.2-21.2 W, Notebookcheck 2022).
FLOP-based energy assumes every operation costs the same. Backward passes and
autoregressive decoding run fewer FLOP per second than a forward pass, so this
estimate is conservative toward fine-tuning and Whisper (it understates their
energy); the time-based ratio is reported next to it for every system.

Writes results/s03c/energy.json and results/s03c/lifecycle_flop_energy.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import Wav2Vec2Config, Wav2Vec2Model

from greenafi.archs import bake_weight_norm

ap = argparse.ArgumentParser()
ap.add_argument("--cost-dir", default="results/vm_random_weights")
ap.add_argument("--out", default="results/s03c")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
anchors = json.load(open("config/energy_anchors.json"))["anchors"]

# ---- anchor 1: Raspberry Pi 4, wav2vec 2.0 Base, 10 s
torch.manual_seed(0)
base = bake_weight_norm(Wav2Vec2Model(Wav2Vec2Config())).eval()  # default config = wav2vec 2.0 Base
with torch.inference_mode(), FlopCounterMode(display=False) as fc:
    base(torch.randn(1, 160_000))
gflop_base10 = fc.get_total_flops() / 1e9
pi = anchors["low_cost_arm_board"]["reported"]
pi_j = pi["energy_mWh_per_10s_utterance_unquantised"] * 3.6  # mWh -> J
j_per_gflop = {"low_cost_arm_board": pi_j / gflop_base10}
derivation = {"low_cost_arm_board": {"reported_J": pi_j, "workload_GFLOP": gflop_base10,
                                     "J_per_GFLOP": pi_j / gflop_base10}}

# ---- anchor 2: Apple laptop CPU (needs a measured throughput on Apple cores)
apt = f"{args.out}/apple_throughput.json"
if os.path.exists(apt):
    t = json.load(open(apt))
    gflops_per_s = t["gflop_per_s"]
    lo = anchors["apple_laptop_cpu"]["reported"]["package_power_W_multi_core_low"]
    hi = anchors["apple_laptop_cpu"]["reported"]["package_power_W_multi_core_high"]
    j_per_gflop["apple_laptop_low"] = lo / gflops_per_s
    j_per_gflop["apple_laptop_high"] = hi / gflops_per_s
    derivation["apple_laptop"] = {"measured_GFLOP_per_s": gflops_per_s, "power_W": [lo, hi],
                                  "J_per_GFLOP": [lo / gflops_per_s, hi / gflops_per_s], "threads": t.get("threads")}

# ---- per-system FLOPs (exact) and measured times
C = args.cost_dir
xl = {(r["system"], r["audio_s"]): r for r in json.load(open(f"{C}/compute_xlsr.json"))["rows"]}
wh = {r["system"]: r for r in json.load(open(f"{C}/compute_whisper.json"))["rows"]}
ft = {r["system"]: r for r in json.load(open(f"{C}/compute_finetune.json"))["rows"]}
N = json.load(open(f"{C}/compute_finetune.json"))["n_train_clips"]

rec = {  # one recognition, 2.2 s: (GFLOP, seconds)
    "whisper_free": (wh["whisper_free"]["flops"] / 1e9, wh["whisper_free"]["median_s"]),
    "whisper_forced12": (wh["whisper_forced12"]["flops"] / 1e9, wh["whisper_forced12"]["median_s"]),
    "xlsr_L24": (xl[("xlsr_L24", 2.2)]["flops"] / 1e9, xl[("xlsr_L24", 2.2)]["median_s"]),
    "xlsr_L12": (xl[("xlsr_L12", 2.2)]["flops"] / 1e9, xl[("xlsr_L12", 2.2)]["median_s"]),
}
pers = {  # personalising one user: (GFLOP, seconds)
    "frozen_L12": (rec["xlsr_L12"][0] * N, rec["xlsr_L12"][1] * N),
}
for k in ("finetune_full", "finetune_top", "finetune_lora", "finetune_full_L12"):
    if k in ft:
        pers[f"{k}_10ep"] = (10 * ft[k]["flops_per_epoch"] / 1e9, 10 * ft[k]["s_per_epoch"])

systems = {  # name: (personalisation key or None, recognition key)
    "Whisper-medium, free transcription": (None, "whisper_free"),
    "Fine-tuning, all layers (10 ep)": ("finetune_full_10ep", "xlsr_L24"),
    "LoRA (10 ep)": ("finetune_lora_10ep", "xlsr_L24"),
    "Fine-tuning, model cut at layer 12 (10 ep)": ("finetune_full_L12_10ep", "xlsr_L12"),
    "Frozen, layer 12, nearest neighbour": ("frozen_L12", "xlsr_L12"),
}
rows = []
for name, (pk, rk) in systems.items():
    if pk is not None and pk not in pers:
        continue
    pg, ps_ = pers[pk] if pk else (0.0, 0.0)
    rg, rs_ = rec[rk]
    for u in (20, 100):
        year_g = pg + 365 * u * rg
        year_s = ps_ + 365 * u * rs_
        row = {"system": name, "uses_per_day": u, "personalisation_GFLOP": pg, "recognition_GFLOP": rg,
               "year_GFLOP": year_g, "year_seconds_benchmark": year_s}
        for a, j in j_per_gflop.items():
            row[f"recognition_J_{a}"] = rg * j
            row[f"personalisation_J_{a}"] = pg * j
            row[f"year_Wh_{a}"] = year_g * j / 3600
        rows.append(row)

with open(f"{args.out}/lifecycle_flop_energy.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

ref = {r["system"]: r for r in rows if r["uses_per_day"] == 100}
fz = ref["Frozen, layer 12, nearest neighbour"]
ratios = {n: {"flop_ratio": r["year_GFLOP"] / fz["year_GFLOP"],
              "time_ratio": r["year_seconds_benchmark"] / fz["year_seconds_benchmark"]} for n, r in ref.items()}
json.dump({"J_per_GFLOP": j_per_gflop, "derivation": derivation, "year_ratio_vs_frozen_L12_at_100_per_day": ratios},
          open(f"{args.out}/energy.json", "w"), indent=2)
print(json.dumps({"J_per_GFLOP": j_per_gflop, "derivation": derivation}, indent=2))
for n, r in ratios.items():
    print(f"{n:45s} FLOP ratio {r['flop_ratio']:6.2f}   time ratio {r['time_ratio']:6.2f}")
