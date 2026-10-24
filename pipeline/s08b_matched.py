"""Stage 8b — matched comparisons on identical recordings (correction to s08).

  python -m pipeline.s08b_matched torgo

s08 averaged the frozen system over its 5 draws while fine-tuning ran on draw 0
only, and scored Whisper on the subset of test recordings it processed (at most
60 per speaker, sampled from all of a speaker's recordings). Neither pairing used
identical recordings, which the pre-registration requires. This stage recomputes
the embedding-based systems per recording from data/embeddings and pairs:

  A. fine-tuning comparisons: draw 0, the full test split (same as s07);
  B. Whisper comparisons: every draw, the test recordings Whisper processed,
     averaged over draws per speaker, every system on the same recordings.

It also checks that the recomputed frozen and shared UARs reproduce s05 exactly.
Writes results/s08b/<corpus>_matched.json and <corpus>_matched_per_speaker.csv.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from greenafi.data import by_speaker, load_config, make_split, read_manifest, uar
from greenafi.stats import bootstrap_ci, tost_paired, wilcoxon_or_none
from pipeline.s05_classifiers import fit_predict
from pipeline.s08_stats import norm

import difflib

PRIMARY = "frozen_L12_1nn"


def verdict(res: dict, margin: float) -> str:
    """Plain-language reading of the 90% CI against the +/- margin band."""
    if "ci_90_low" not in res:
        return res.get("verdict", "")
    lo, hi = res["ci_90_low"], res["ci_90_high"]
    if lo > -margin and hi < margin:
        return "equivalent within the margin"
    if lo > margin:
        return "primary better by more than the margin"
    if hi < -margin:
        return "primary worse by more than the margin"
    if lo > -margin:
        return "primary not worse by more than the margin (non-inferior); equivalence not shown"
    if hi < margin:
        return "primary not better by more than the margin; equivalence not shown"
    return "inconclusive"


def main():
    corpus = sys.argv[1]
    cfg = load_config()
    p, h = cfg["personalisation"], cfg["hypotheses"]
    margin, alpha = h["H4_equivalence_margin_uar_points"], h["H4_alpha"]
    counts = json.load(open(f"results/s01/{corpus}_counts.json"))
    k = counts["primary_k"]
    keep = set(counts["speakers_included"])
    rows = [r for r in read_manifest(f"data/manifests/{corpus}.csv") if r["speaker"] in keep]
    E = np.load(f"data/embeddings/{corpus}/emb.npy", mmap_mode="r")
    idx = {r["utt_id"]: int(r["row"]) for r in csv.DictReader(open(f"data/embeddings/{corpus}/index.csv"))}
    item_of = {r["utt_id"]: r["item"] for r in rows}
    W = {}
    wp = f"results/s06/{corpus}_whisper.jsonl"
    if os.path.exists(wp):
        for line in open(wp):
            r = json.loads(line)
            W[r["utt_id"]] = r
    W2 = {}
    wp2 = f"results/s06/{corpus}_whisper_testsplit.jsonl"
    if os.path.exists(wp2):
        for line in open(wp2):
            r = json.loads(line)
            W2[r["utt_id"]] = r
    s5 = pd.read_csv(f"results/s05/{corpus}_uar.csv")
    s7 = pd.read_csv(f"results/s07/{corpus}_finetune.csv") if os.path.exists(f"results/s07/{corpus}_finetune.csv") else None
    if s7 is not None:  # a run re-done to more epochs rewrites its earlier evaluations: keep the last row
        s7 = s7.drop_duplicates(subset=["speaker", "k", "draw", "system", "epoch"], keep="last")

    def emb(ids, L):
        return np.asarray(E[[idx[u] for u in ids], L - 1], dtype=np.float32)

    per = []          # rows: speaker, group, analysis, system, uar, n_test
    repro = []        # (speaker, draw, system, s05, recomputed)
    chance = []       # per speaker: items, chance level, split sizes
    for spk, rs in sorted(by_speaker(rows).items()):
        g = rs[0]["group"]
        for d in range(p["n_draws"]):
            sp = make_split(rs, spk, k, d, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
            if sp is None:
                continue
            ytr = [item_of[u] for u in sp.train]
            yte = np.array([item_of[u] for u in sp.test])
            Ls = (12, 24) if d == 0 else (12,)  # layer-24 systems only on draw 0 (analysis A), for time
            preds = {f"frozen_L{L}_1nn": np.array(fit_predict(emb(sp.train, L), ytr, emb(sp.test, L), "1nn"))
                     for L in Ls}
            other = [r for r in rows if r["speaker"] != spk and r["item"] in set(sp.items)]
            oid, oy = [r["utt_id"] for r in other], [r["item"] for r in other]
            for L in Ls:
                preds[f"shared_L{L}_logreg"] = np.array(fit_predict(emb(oid, L), oy, emb(sp.test, L), "logreg"))
            # EXPLORATORY (not pre-registered): other speakers' recordings plus the speaker's own k
            preds["xpl_shared_own_L12_logreg"] = np.array(fit_predict(emb(oid + sp.train, 12), oy + ytr,
                                                                      emb(sp.test, 12), "logreg"))
            for s in preds:
                ref = s5[(s5.speaker == spk) & (s5.k == k) & (s5.draw == d) & (s5.system == s)]
                if len(ref):
                    repro.append({"speaker": spk, "draw": d, "system": s, "s05": float(ref.uar.iloc[0]),
                                  "recomputed": round(uar(list(yte), list(preds[s])), 4)})
            if d == 0:  # A: the fine-tuning test split
                for s, pr in preds.items():
                    per.append({"speaker": spk, "group": g, "analysis": "A_finetune_split", "system": s,
                                "uar": uar(list(yte), list(pr)), "n_test": len(yte)})
                if s7 is not None:
                    for _, r in s7[(s7.speaker == spk) & (s7.k == k) & (s7.draw == 0)].iterrows():
                        per.append({"speaker": spk, "group": g, "analysis": "A_finetune_split",
                                    "system": f"{r.system}_ep{r.epoch}", "uar": float(r.uar), "n_test": len(yte)})
                # C: Whisper on the whole draw-0 test split (s06 --test-split), when complete
                if W2 and all(u in W2 for u in sp.test):
                    sysc = {s: list(pr) for s, pr in preds.items() if "L24" not in s}
                    sysc["whisper_free"] = [max(sp.items, key=lambda it: difflib.SequenceMatcher(
                        None, norm(W2[u]["transcript"]), norm(it)).ratio()) for u in sp.test]
                    sysc["whisper_forced_mean"] = [max(sp.items, key=lambda it: W2[u]["scores"][it][1]) for u in sp.test]
                    sysc["whisper_forced_sum"] = [max(sp.items, key=lambda it: W2[u]["scores"][it][0]) for u in sp.test]
                    for s_, pr in sysc.items():
                        per.append({"speaker": spk, "group": g, "analysis": "C_full_split_whisper", "system": s_,
                                    "uar": uar(list(yte), pr), "n_test": len(yte)})
                    if s7 is not None:
                        for _, r in s7[(s7.speaker == spk) & (s7.k == k) & (s7.draw == 0)].iterrows():
                            per.append({"speaker": spk, "group": g, "analysis": "C_full_split_whisper",
                                        "system": f"{r.system}_ep{r.epoch}", "uar": float(r.uar), "n_test": len(yte)})
            # B: Whisper subset of this draw's test split
            m = np.array([u in W for u in sp.test])
            if m.sum() == 0:
                continue
            tu = [u for u in sp.test if u in W]
            yt = list(yte[m])
            sysp = {s: list(pr[m]) for s, pr in preds.items() if "L24" not in s}
            sysp["whisper_free"] = [max(sp.items, key=lambda it: difflib.SequenceMatcher(
                None, norm(W[u]["transcript"]), norm(it)).ratio()) for u in tu]
            sysp["whisper_forced_mean"] = [max(sp.items, key=lambda it: W[u]["scores"][it][1]) for u in tu]
            sysp["whisper_forced_sum"] = [max(sp.items, key=lambda it: W[u]["scores"][it][0]) for u in tu]
            for s, pr in sysp.items():
                per.append({"speaker": spk, "group": g, "analysis": f"B_whisper_subset_d{d}", "system": s,
                            "uar": uar(yt, pr), "n_test": len(yt)})
            if d == 0:
                chance.append({"speaker": spk, "group": g, "n_items": len(sp.items), "chance_uar": 1 / len(sp.items),
                               "n_train": len(sp.train), "n_test": len(sp.test), "n_test_whisper": int(m.sum()),
                               "n_other_train": len(oid)})
        print(spk, "done", flush=True)

    df = pd.DataFrame(per)
    df["analysis_group"] = df["analysis"].str.replace(r"_d\d+$", "", regex=True)
    ps = df.groupby(["speaker", "group", "analysis_group", "system"], as_index=False).agg(uar=("uar", "mean"), n_test=("n_test", "mean"))
    os.makedirs("results/s08b", exist_ok=True)
    ps.to_csv(f"results/s08b/{corpus}_matched_per_speaker.csv", index=False)

    out = {"corpus": corpus, "primary_k": k, "margin_points": margin, "alpha": alpha,
           "reproduces_s05": all(abs(r["s05"] - r["recomputed"]) < 1e-4 for r in repro),
           "n_reproduction_checks": len(repro),
           "reproduction_mismatches": [r for r in repro if abs(r["s05"] - r["recomputed"]) >= 1e-4],
           "speakers": chance, "systems": {}, "paired": {}}
    for (ag, g, s), sub in ps.groupby(["analysis_group", "group", "system"]):
        v = sub.uar.to_numpy()
        if len(v) > 1:  # t-interval over speakers (a bootstrap under-covers with six speakers)
            from scipy import stats as _st
            h_ = _st.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            lo, hi = float(v.mean() - h_), float(v.mean() + h_)
        else:
            lo, hi = float("nan"), float("nan")
        out["systems"].setdefault(ag, {}).setdefault(g, {})[s] = {
            "n_speakers": len(v), "mean_uar": float(v.mean()), "ci95": [lo, hi],
            "mean_n_test": float(sub.n_test.mean()), "speakers": sorted(sub.speaker)}
    for ag, sub_a in ps.groupby("analysis_group"):
        wide = sub_a.pivot_table(index=["speaker", "group"], columns="system", values="uar").reset_index()
        for g, sub in wide.groupby("group"):
            for other in [c for c in wide.columns if c not in ("speaker", "group", PRIMARY)]:
                pair = sub[["speaker", PRIMARY, other]].dropna()
                if not len(pair):
                    continue
                dd = (pair[PRIMARY] - pair[other]).to_numpy() * 100
                res = tost_paired(dd, margin, alpha)
                res["reading"] = verdict(res, margin)
                res["wilcoxon_p"] = wilcoxon_or_none(dd)
                res["per_speaker_diff_points"] = dict(zip(pair.speaker, dd.round(2).tolist()))
                res["is_preregistered_primary"] = (ag == "A_finetune_split" and other == "finetune_full_ep10")
                out["paired"].setdefault(ag, {}).setdefault(g, {})[f"{PRIMARY} - {other}"] = res
    PAIRS = [("shared_L12_logreg", "whisper_forced_mean"), ("shared_L12_logreg", "whisper_forced_sum"),
             ("shared_L12_logreg", "whisper_free"), ("shared_L12_logreg", "frozen_L12_1nn"),
             ("xpl_shared_own_L12_logreg", "shared_L12_logreg"), ("shared_L12_logreg", "shared_L24_logreg"),
             ("frozen_L12_1nn", "frozen_L24_1nn"), ("shared_L12_logreg", "finetune_full_L12_ep10"),
             ("shared_L12_logreg", "finetune_full_ep10"),
             ("xpl_shared_own_L12_logreg", "whisper_forced_mean"), ("xpl_shared_own_L12_logreg", "whisper_forced_sum"),
             ("xpl_shared_own_L12_logreg", "frozen_L12_1nn")]
    out["paired_extra"] = {}
    for ag, sub_a in ps.groupby("analysis_group"):
        wide = sub_a.pivot_table(index=["speaker", "group"], columns="system", values="uar").reset_index()
        for g, sub in wide.groupby("group"):
            for a_, b_ in PAIRS:
                if a_ not in sub or b_ not in sub:
                    continue
                pair = sub[["speaker", a_, b_]].dropna()
                if not len(pair):
                    continue
                dd = (pair[a_] - pair[b_]).to_numpy() * 100
                res = tost_paired(dd, margin, alpha)
                res["reading"] = verdict(res, margin).replace("primary", a_)
                res["wilcoxon_p"] = wilcoxon_or_none(dd)
                res["n_better"] = int((dd > 0).sum())
                res["per_speaker_diff_points"] = dict(zip(pair.speaker, dd.round(2).tolist()))
                out["paired_extra"].setdefault(ag, {}).setdefault(g, {})[f"{a_} - {b_}"] = res
    json.dump(out, open(f"results/s08b/{corpus}_matched.json", "w"), indent=2, default=float)
    print("reproduces s05:", out["reproduces_s05"], "checks:", out["n_reproduction_checks"])
    for ag, d in out["systems"].items():
        for g, ss in d.items():
            print(ag, g, {s: (round(v["mean_uar"], 3), v["n_speakers"]) for s, v in ss.items()})


if __name__ == "__main__":
    main()
