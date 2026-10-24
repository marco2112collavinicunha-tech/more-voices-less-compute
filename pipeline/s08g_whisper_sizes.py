"""s08g — Whisper at three sizes (tiny, base, medium) against the shared classifier, on the same
draw-0 test recordings (analysis C), with each size's computation per recognition.

Accuracy: forced choice (mean and summed log-probability; the better per group is reported, as for
medium) and free transcription, from results/s06/torgo_whisper_testsplit[_tiny|_base].jsonl.
Computation: counted from each architecture with the same conventions as s03d (matrix products,
convolutional front end; attention products not counted), forced choice over the speaker's K words
with 5 decoder tokens per candidate. The formula is checked against medium's measured encoder count.
Added 16 Oct 2026 (exploratory, after the second-round review asked for an equal-computation point).

Run: python -m pipeline.s08g_whisper_sizes torgo
"""
from __future__ import annotations

import difflib
import json
import os
import sys

import numpy as np
import pandas as pd

from greenafi.data import by_speaker, load_config, make_split, read_manifest, uar
from greenafi.stats import tost_paired
from pipeline.s08_stats import norm

ARCH = {  # d_model, encoder layers, decoder layers (Radford et al. 2023, Table 1)
    "tiny": (384, 4, 4), "base": (512, 6, 6), "medium": (1024, 24, 24)}
V, T, MEL, TOK = 51865, 1500, 80, 5


def costs(size: str, K: int) -> dict:
    d, ne, nd = ARCH[size]
    conv = 2 * MEL * d * 3 * 2 * T + 2 * d * d * 3 * T          # conv1 (3000 frames) + conv2 (stride 2)
    enc = conv + ne * 2 * 12 * d * d * T                          # attention projections 4d^2 + MLP 8d^2
    per_token = 2 * (nd * 14 * d * d + d * V)
    oneoff = 2 * nd * 2 * d * d * T + per_token
    return {"encoder": enc, "forced": enc + oneoff + K * TOK * per_token}


def main():
    corpus = sys.argv[1]
    cfg = load_config(); p = cfg["personalisation"]
    counts = json.load(open(f"results/s01/{corpus}_counts.json"))
    k = counts["primary_k"]; keep = set(counts["speakers_included"])
    rows = [r for r in read_manifest(f"data/manifests/{corpus}.csv") if r["speaker"] in keep]
    C = json.load(open(f"results/s03d/{corpus}_costs.json"))
    med_enc = C["whisper"]["encoder_flops"]
    check = costs("medium", 1)["encoder"] / med_enc
    files = {"medium": f"results/s06/{corpus}_whisper_testsplit.jsonl",
             "tiny": f"results/s06/{corpus}_whisper_testsplit_tiny.jsonl",
             "base": f"results/s06/{corpus}_whisper_testsplit_base.jsonl"}
    W = {s: {json.loads(l)["utt_id"]: json.loads(l) for l in open(f)} for s, f in files.items() if os.path.exists(f)}
    ps = pd.read_csv(f"results/s08b/{corpus}_matched_per_speaker.csv")
    shared = ps[(ps.analysis_group == "C_full_split_whisper") & (ps.system == "shared_L12_logreg")].set_index("speaker").uar
    per = []
    for spk, rs in sorted(by_speaker(rows).items()):
        sp = make_split(rs, spk, k, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
        if sp is None:
            continue
        item_of = {r["utt_id"]: r["item"] for r in rs}
        yte = [item_of[u] for u in sp.test]
        K = len(sp.items); l12 = C["speakers"][spk]["per_recognition_flops"]["L12"]
        for size, w in W.items():
            if not all(u in w for u in sp.test):
                continue
            pr = {"forced_mean": [max(sp.items, key=lambda it: w[u]["scores"][it][1]) for u in sp.test],
                  "forced_sum": [max(sp.items, key=lambda it: w[u]["scores"][it][0]) for u in sp.test],
                  "free": [max(sp.items, key=lambda it: difflib.SequenceMatcher(None, norm(w[u]["transcript"]), norm(it)).ratio())
                           for u in sp.test]}
            c = costs(size, K)
            for m, y in pr.items():
                per.append({"speaker": spk, "group": rs[0]["group"], "size": size, "mode": m, "uar": uar(yte, y),
                            "shared_L12": float(shared.get(spk, np.nan)), "gflop_forced": c["forced"] / 1e9,
                            "gflop_encoder": c["encoder"] / 1e9, "ratio_forced_to_L12": c["forced"] / l12})
    df = pd.DataFrame(per)
    out = {"encoder_formula_check_vs_measured_medium": check, "groups": {}}
    for (g, size), sub in df.groupby(["group", "size"]):
        best = max(("forced_mean", "forced_sum"), key=lambda m: sub[sub["mode"] == m].uar.mean())
        b = sub[sub["mode"] == best]
        dd = ((b.shared_L12 - b.uar) * 100).to_numpy()
        out["groups"].setdefault(g, {})[size] = {
            "n_speakers": int(len(b)), "forced_scoring": best, "uar_forced": float(b.uar.mean()),
            "uar_free": float(sub[sub["mode"] == "free"].uar.mean()),
            "gflop_forced_mean": float(b.gflop_forced.mean()), "gflop_encoder": float(b.gflop_encoder.mean()),
            "ratio_forced_to_L12_mean": float(b.ratio_forced_to_L12.mean()),
            "shared_minus_whisper": tost_paired(dd, 5.0, 0.05), "n_shared_better": int((dd > 0).sum()),
            "per_speaker_diff": dict(zip(b.speaker, dd.round(2).tolist()))}
    os.makedirs("results/s08g", exist_ok=True)
    df.to_csv(f"results/s08g/{corpus}_whisper_sizes_per_speaker.csv", index=False)
    json.dump(out, open(f"results/s08g/{corpus}_whisper_sizes.json", "w"), indent=2, default=float)
    print(json.dumps({g: {s: (round(v["uar_forced"] * 100, 1), round(v["gflop_forced_mean"], 1), v["n_speakers"])
                          for s, v in d.items()} for g, d in out["groups"].items()}, indent=1), "check", round(check, 4))


if __name__ == "__main__":
    main()
