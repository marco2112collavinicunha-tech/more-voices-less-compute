"""Per-user and per-population cost of each way of building the system.

Reads results/compute_*.json and writes results/lifecycle.csv and
results/summary.json.

Energy: if the benchmark measured joules (powermetrics / RAPL), those are used.
Otherwise energy is ESTIMATED as time x --power-w and every output row carries
energy_measured=False. FLOP counts are exact and hardware-independent; the
FLOP ratios between systems are the primary result.

Scenarios are explicit assumptions, not observations:
  --uses-per-day   recognitions per user per day
  --epochs         epochs of per-speaker fine-tuning (for the fine-tuning baselines)
  --clip-s         audio length of one recognition (seconds)
"""
from __future__ import annotations

import argparse
import csv
import json

N_TRAIN = 96

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/s03")
ap.add_argument("--clip-s", type=float, default=2.2)
ap.add_argument("--uses-per-day", type=float, nargs="+", default=[20, 100])
ap.add_argument("--epochs", type=int, nargs="+", default=[10, 30])
ap.add_argument("--power-w", type=float, default=10.0,
                help="assumed net CPU power when no energy meter was available")
ap.add_argument("--users", type=int, nargs="+", default=[1_000, 10_000, 100_000])
ap.add_argument("--j-per-gflop", type=float, default=None,
                help="FLOP-based energy: joules per GFLOP from a published measurement (stage s03c); overrides --power-w")
ap.add_argument("--out", default=None, help="output directory (default: --results)")
args = ap.parse_args()

R = args.results
xlsr = json.load(open(f"{R}/compute_xlsr.json"))["rows"]
heads = json.load(open(f"{R}/compute_heads.json"))
whisper = json.load(open(f"{R}/compute_whisper.json"))["rows"]
ft = json.load(open(f"{R}/compute_finetune.json"))["rows"]
grids = json.load(open("config/grids.json"))
grid_all = {**grids["founding_members"], **grids["other_members"]}


def energy(row, seconds_key="median_s", joules_key="joules_per_call", flops_key="flops"):
    j = row.get(joules_key)
    if j is not None:
        return j, True
    if args.j_per_gflop is not None and row.get(flops_key):
        return row[flops_key] / 1e9 * args.j_per_gflop, False
    return row[seconds_key] * args.power_w, False


def pick(rows, system, audio_s=None):
    cands = [r for r in rows if r["system"] == system]
    if audio_s is not None:
        cands = [r for r in cands if abs(r.get("audio_s", audio_s) - audio_s) < 1e-6]
    if not cands:
        raise KeyError((system, audio_s))
    return cands[0]


h = {r["system"]: r for r in heads["rows"]}
cost = {}  # name -> dict(flops, joules, measured)
for name in ("xlsr_L24", "xlsr_L18", "xlsr_L12", "xlsr_L12_int8"):
    try:
        r = pick(xlsr, name, args.clip_s)
    except KeyError:
        continue
    j, meas = energy(r)
    cost[name] = {"flops": r["flops"], "joules": j, "measured": meas, "seconds": r["median_s"]}
for name in ("whisper_free", "whisper_forced12", "whisper_encoder_only"):
    r = pick(whisper, name)
    j, meas = energy(r)
    cost[name] = {"flops": r["flops"], "joules": j, "measured": meas, "seconds": r["median_s"]}
for name in ("predict_1nn", "predict_logreg", "fit_1nn", "fit_logreg", "fit_protocol_selection"):
    j, meas = energy(h[name]) if h[name].get("flops") or args.j_per_gflop is None else (h[name]["median_s"] * args.power_w, False)
    cost[name] = {"flops": h[name].get("flops"), "joules": j, "measured": meas, "seconds": h[name]["median_s"]}
for r in ft:
    j = r.get("joules_per_epoch")
    meas = j is not None
    if j is None:
        j = (r["flops_per_epoch"] / 1e9 * args.j_per_gflop) if args.j_per_gflop is not None else r["s_per_epoch"] * args.power_w
    cost[r["system"] + "_epoch"] = {"flops": r["flops_per_epoch"], "joules": j, "measured": meas,
                                    "seconds": r["s_per_epoch"], "per_user_bytes": r["per_user_model_bytes_fp32"]}


def add(*parts, k=1):
    out = {"flops": 0, "joules": 0.0, "seconds": 0.0, "measured": True}
    for p, mult in parts:
        out["flops"] += (p["flops"] or 0) * mult
        out["joules"] += p["joules"] * mult
        out["seconds"] += p["seconds"] * mult
        out["measured"] &= p["measured"]
    return out


S = {}
emb12 = (cost["xlsr_L12"], N_TRAIN)
emb18 = (cost["xlsr_L18"], N_TRAIN)
emb24 = (cost["xlsr_L24"], N_TRAIN)
# --- systems: (personalisation cost per user, inference cost per recognition, per-user storage bytes)
S["Generic: Whisper-medium, free transcription"] = (add(), cost["whisper_free"], 0)
S["Generic: Whisper-medium, forced choice (12)"] = (add(), cost["whisper_forced12"], 0)
FT = [("full", "all layers", "xlsr_L24"), ("top", "top 4 layers", "xlsr_L24"),
      ("lora", "LoRA", "xlsr_L24"), ("full_L12", "all layers of a model cut at layer 12", "xlsr_L12")]
for e in args.epochs:
    for key, label, inf in FT:
        if f"finetune_{key}_epoch" not in cost:
            continue
        S[f"Per-speaker fine-tuning, {label} ({e} ep)"] = (
            add((cost[f"finetune_{key}_epoch"], e)), add((cost[inf], 1), (cost["predict_logreg"], 1)),
            cost[f"finetune_{key}_epoch"]["per_user_bytes"])
S["Personalized (protocol), full 24-layer pass"] = (
    add(emb24, (cost["fit_protocol_selection"], 1)), add((cost["xlsr_L24"], 1), (cost["predict_logreg"], 1)),
    heads["storage_bytes_per_user"]["logreg"])
S["Personalized, truncated at layer 18"] = (
    add(emb18, (cost["fit_protocol_selection"], 1)), add((cost["xlsr_L18"], 1), (cost["predict_logreg"], 1)),
    heads["storage_bytes_per_user"]["logreg"])
S["Personalized, truncated at layer 12 (1-NN)"] = (
    add(emb12, (cost["fit_1nn"], 1)), add((cost["xlsr_L12"], 1), (cost["predict_1nn"], 1)),
    heads["storage_bytes_per_user"]["1nn"])
if "xlsr_L12_int8" in cost:
    cost["xlsr_L12_int8"]["flops"] = cost["xlsr_L12"]["flops"]  # same operation count, in 8-bit integers
    if args.j_per_gflop is not None and not cost["xlsr_L12_int8"]["measured"]:
        # FLOP-based energy cannot see the cheaper integer arithmetic: apply the energy reduction that the
        # anchor's own source measured for 8-bit dynamic quantisation on the same board (27% less;
        # Gondi 2022), not a run-time ratio from a different machine
        cost["xlsr_L12_int8"]["joules"] = cost["xlsr_L12"]["joules"] * 0.73
    S["Personalized, layer 12, int8 (fidelity pending)"] = (
        add((cost["xlsr_L12_int8"], N_TRAIN), (cost["fit_1nn"], 1)),
        add((cost["xlsr_L12_int8"], 1), (cost["predict_1nn"], 1)),
        heads["storage_bytes_per_user"]["1nn"])

rows = []
for name, (pers, inf, store) in S.items():
    for u in args.uses_per_day:
        year_inf_j = inf["joules"] * u * 365
        year_j = pers["joules"] + year_inf_j
        row = {
            "system": name,
            "uses_per_day": u,
            "personalisation_flops": pers["flops"],
            "personalisation_J": pers["joules"],
            "inference_flops": inf["flops"],
            "inference_J": inf["joules"],
            "inference_s": inf["seconds"],
            "user_year_J": year_j,
            "user_year_Wh": year_j / 3600,
            "personalisation_share_of_year": pers["joules"] / year_j if year_j else 0.0,
            "breakeven_days_personalisation_vs_inference": (pers["joules"] / (inf["joules"] * u)) if inf["joules"] else None,
            "per_user_storage_bytes": store,
            "energy_measured": pers["measured"] and inf["measured"],
        }
        for country, g in grid_all.items():
            row[f"gCO2e_user_year_{country}"] = year_j / 3.6e6 * g
        for n in args.users:
            row[f"kWh_year_{n}_users"] = year_j * n / 3.6e6
            row[f"storage_GB_{n}_users"] = store * n / 1e9
        rows.append(row)

OUT = args.out or R
import os as _os
_os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/lifecycle.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

ref_inf = cost["whisper_free"]
summary = {
    "clip_s": args.clip_s,
    "energy_mode": "measured" if all(c["measured"] for c in cost.values()) else (
        f"estimated from computation at {args.j_per_gflop:.3f} J/GFLOP (published measurement, low-cost ARM board)"
        if args.j_per_gflop is not None else f"estimated at {args.power_w} W"),
    "inference_flops": {k: v["flops"] for k, v in cost.items() if not k.endswith("_epoch") and v["flops"]},
    "inference_flops_ratio_vs_whisper_free": {
        k: v["flops"] / ref_inf["flops"] for k, v in cost.items()
        if k.startswith(("xlsr", "whisper")) and v["flops"]},
    "layer12_vs_layer24_flops": cost["xlsr_L12"]["flops"] / cost["xlsr_L24"]["flops"],
    "layer18_vs_layer24_flops": cost["xlsr_L18"]["flops"] / cost["xlsr_L24"]["flops"],
    "personalisation_flops": {n: p["flops"] for n, (p, _, _) in S.items()},
    "per_user_storage_bytes": {n: s for n, (_, _, s) in S.items()},
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
