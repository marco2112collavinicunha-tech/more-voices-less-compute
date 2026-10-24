"""Stage 8 — pre-registered analysis.

  python -m pipeline.s08_stats torgo

Reads results/s05 (frozen and shared classifiers), s06 (Whisper) and s07
(fine-tuning); every system is scored on the SAME test recordings per
speaker, k and draw (splits are recomputed from the seed). Writes
results/s08/<corpus>_per_speaker.csv and results/s08/<corpus>_summary.json.
"""
from __future__ import annotations

import difflib
import json
import os
import sys
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

from greenafi.data import by_speaker, load_config, make_split, read_manifest, uar
from greenafi.stats import bootstrap_ci, tost_paired, wilcoxon_or_none

PRIMARY = "frozen_L12_1nn"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower().replace("|", " "))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in s).split())


def whisper_rows(corpus, cfg, rows, k):
    path = f"results/s06/{corpus}_whisper.jsonl"
    if not os.path.exists(path):
        return []
    W = {}
    for line in open(path):
        r = json.loads(line)
        W[r["utt_id"]] = r
    p = cfg["personalisation"]
    out = []
    for spk, rs in by_speaker(rows).items():
        for d in range(p["n_draws"]):
            sp = make_split(rs, spk, k, d, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
            if sp is None:
                continue
            test = [u for u in sp.test if u in W]
            if not test:
                continue
            yt = [W[u]["item"] for u in test]
            free = [max(sp.items, key=lambda it: difflib.SequenceMatcher(None, norm(W[u]["transcript"]), norm(it)).ratio())
                    for u in test]
            fsum = [max(sp.items, key=lambda it: W[u]["scores"][it][0]) for u in test]
            fmean = [max(sp.items, key=lambda it: W[u]["scores"][it][1]) for u in test]
            for name, pred in (("whisper_free", free), ("whisper_forced_sum", fsum), ("whisper_forced_mean", fmean)):
                out.append({"speaker": spk, "group": rs[0]["group"], "k": k, "draw": d, "system": name,
                            "uar": uar(yt, pred), "n_test": len(test)})
    return out


def main():
    corpus = sys.argv[1]
    cfg = load_config()
    h = cfg["hypotheses"]
    counts = json.load(open(f"results/s01/{corpus}_counts.json"))
    k = counts["primary_k"]
    keep = set(counts["speakers_included"])
    rows = [r for r in read_manifest(f"data/manifests/{corpus}.csv") if r["speaker"] in keep]

    frames = []
    s5 = pd.read_csv(f"results/s05/{corpus}_uar.csv")
    frames.append(s5[["speaker", "group", "k", "draw", "system", "uar"]])
    frames.append(pd.DataFrame(whisper_rows(corpus, cfg, rows, k)))
    if os.path.exists(f"results/s07/{corpus}_finetune.csv"):
        s7 = pd.read_csv(f"results/s07/{corpus}_finetune.csv").drop_duplicates(
            subset=["speaker", "k", "draw", "system", "epoch"], keep="last")
        s7 = s7[~s7["random_weights"].astype(str).str.lower().eq("true")] if "random_weights" in s7 else s7
        s7["system"] = s7["system"] + "_ep" + s7["epoch"].astype(str)
        frames.append(s7[["speaker", "group", "k", "draw", "system", "uar"]])
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)

    # per speaker: mean over draws, at the primary k
    ps = (df[df["k"] == k].groupby(["speaker", "group", "system"], as_index=False)["uar"].mean())
    os.makedirs("results/s08", exist_ok=True)
    ps.pivot_table(index=["speaker", "group"], columns="system", values="uar").to_csv(f"results/s08/{corpus}_per_speaker.csv")

    summary = {"corpus": corpus, "primary_k": k, "primary_system": PRIMARY,
               "speakers_included": sorted(keep), "speakers_excluded": counts["speakers_excluded"],
               "systems": {}, "H4": {}, "layer_sweep": {}, "few_shot": {}}
    for (g, s), sub in ps.groupby(["group", "system"]):
        v = sub["uar"].to_numpy()
        lo, hi = bootstrap_ci(v) if len(v) > 1 else (float("nan"), float("nan"))
        summary["systems"].setdefault(g, {})[s] = {"n_speakers": len(v), "mean_uar": float(v.mean()),
                                                  "ci95": [lo, hi], "min": float(v.min()), "max": float(v.max())}
    # H4: primary vs fine-tuning (all layers, first evaluation epoch), per group
    ft = f"finetune_full_ep{min(cfg['finetune']['eval_at_epochs'])}"
    wide = ps.pivot_table(index=["speaker", "group"], columns="system", values="uar").reset_index()
    for g, sub in wide.groupby("group"):
        for other in [c for c in wide.columns if c.startswith(("finetune_", "whisper_", "shared_"))]:
            pair = sub[[PRIMARY, other]].dropna()
            if len(pair) == 0:
                continue
            d = (pair[PRIMARY] - pair[other]).to_numpy() * 100
            res = tost_paired(d, h["H4_equivalence_margin_uar_points"], h["H4_alpha"])
            res["wilcoxon_p"] = wilcoxon_or_none(d)
            res["per_speaker_diff_points"] = dict(zip(sub.loc[pair.index, "speaker"], d.round(2).tolist()))
            res["is_preregistered_primary"] = other == ft
            summary["H4"].setdefault(g, {})[f"{PRIMARY} - {other}"] = res
    # layer sweep and few-shot curve (descriptive)
    for (g, clf), sub in s5[s5["system"].str.match(r"frozen_L\d+_(1nn|logreg)$")].groupby(["group", "classifier"]):
        cur = sub[sub["k"] == k].groupby(["speaker", "layer"])["uar"].mean().groupby("layer").mean()
        summary["layer_sweep"].setdefault(g, {})[clf] = {int(L): float(u) for L, u in cur.items()}
    for g, sub in s5[s5["system"] == PRIMARY].groupby("group"):
        cur = sub.groupby(["speaker", "k"])["uar"].mean().groupby("k").mean()
        summary["few_shot"][g] = {int(kk): float(u) for kk, u in cur.items()}
    json.dump(summary, open(f"results/s08/{corpus}_summary.json", "w"), indent=2, default=float)
    print(json.dumps({g: {s: round(v["mean_uar"], 3) for s, v in d.items()} for g, d in summary["systems"].items()}, indent=2))


if __name__ == "__main__":
    main()
