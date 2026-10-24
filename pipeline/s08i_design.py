"""s08i — design analyses for deployment (exploratory, added 20 Oct 2026; no pre-registered result changes).

All on analysis C (draw-0 test split, the recordings on which Whisper was scored), shared classifier
(logistic regression on other included speakers' recordings of the speaker's words), TORGO.

1. Layer-wise accuracy-computation frontier: the shared classifier at every layer 1..24, with the
   operations needed to compute up to that layer for the speaker's test recordings (linear in
   duration, from the s03d fits), against Whisper-medium forced choice. Finds the cheapest layer
   whose mean UAR is not below Whisper's.
2. 8-bit weights: the shared classifier on the layer-12 embeddings computed with 8-bit weights.
3. Rejection: accuracy against coverage when the classifier abstains below a confidence threshold
   (maximum class probability), pooled over speakers per group; a descriptive curve (thresholds set on the test recordings themselves, not tuned on held-out data).
4. Number of other speakers (speakers with dysarthria only, for time): the shared classifier trained on recordings from m randomly chosen
   other included speakers (m in 1,2,3,4,6,8,n-1; 5 random draws each), to show how many contributors a corpus needs.
5. Energy-constant sensitivity: the per-message energy under the range of J/GFLOP values in
   config/energy_anchors.json (if present) or 0.5x-2x the anchor.

  python -m pipeline.s08i_design torgo
Writes results/s08i/<corpus>_design.json and CSVs.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from greenafi.data import by_speaker, load_config, make_split, read_manifest, uar

warnings.filterwarnings("ignore")


def fit(Xtr, ytr):
    s = StandardScaler().fit(Xtr)
    c = LogisticRegression(max_iter=1000).fit(s.transform(Xtr), ytr)
    return s, c


def main():
    corpus = sys.argv[1]
    cfg = load_config(); p = cfg["personalisation"]
    counts = json.load(open(f"results/s01/{corpus}_counts.json"))
    k = counts["primary_k"]; keep = set(counts["speakers_included"])
    rows = [r for r in read_manifest(f"data/manifests/{corpus}.csv") if r["speaker"] in keep]
    E = np.load(f"data/embeddings/{corpus}/emb.npy", mmap_mode="r")
    E8 = np.load(f"data/embeddings/{corpus}/emb_int8_l12.npy", mmap_mode="r")
    idx = {r["utt_id"]: int(r["row"]) for r in csv.DictReader(open(f"data/embeddings/{corpus}/index.csv"))}
    item_of = {r["utt_id"]: r["item"] for r in rows}
    W = {json.loads(l)["utt_id"]: json.loads(l) for l in open(f"results/s06/{corpus}_whisper_testsplit.jsonl")}
    C = json.load(open(f"results/s03d/{corpus}_costs.json"))
    fits = C["xlsr_linear_fit"]
    # operations up to layer L for a recording of t seconds: (a + b L) t + (c + d L), from the 12/24 fits
    b = (fits["24"]["flops_per_second"] - fits["12"]["flops_per_second"]) / 12
    a = fits["12"]["flops_per_second"] - 12 * b
    d_ = (fits["24"]["intercept_flops"] - fits["12"]["intercept_flops"]) / 12
    c_ = fits["12"]["intercept_flops"] - 12 * d_
    chk = (a + 18 * b) / fits["18"]["flops_per_second"]
    rng = np.random.default_rng(cfg["seed"])
    secs = json.load(open(f"results/s01/{corpus}_trimmed_seconds.json")) if os.path.exists(f"results/s01/{corpus}_trimmed_seconds.json") else {}

    def emb(ids, L):
        return np.asarray(E[[idx[u] for u in ids], L - 1], dtype=np.float32)

    layer_rows, rej_rows, spk_rows_, int8_rows, whisper = [], [], [], [], {}
    forced_choice = {}
    os.makedirs("results/s08i/cache", exist_ok=True)
    for spk, rs in sorted(by_speaker(rows).items(), reverse=bool(os.environ.get("REVERSE"))):
        g = rs[0]["group"]
        cpath = f"results/s08i/cache/{corpus}_{spk}.json"
        if os.path.exists(cpath):
            cc = json.load(open(cpath))
            layer_rows += cc["layer"]; rej_rows += cc["rej"]; spk_rows_ += cc["spk"]; int8_rows += cc["int8"]
            whisper[spk] = cc["whisper"]; print(spk, "cached", flush=True); continue
        n0 = (len(layer_rows), len(rej_rows), len(spk_rows_), len(int8_rows))
        sp = make_split(rs, spk, k, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
        if sp is None or not all(u in W for u in sp.test):
            continue
        yte = [item_of[u] for u in sp.test]
        other = [r for r in rows if r["speaker"] != spk and r["item"] in set(sp.items)]
        oid, oy = [r["utt_id"] for r in other], [r["item"] for r in other]
        t = C["speakers"][spk]["mean_s_test"]
        wm = [max(sp.items, key=lambda it: W[u]["scores"][it][1]) for u in sp.test]
        ws = [max(sp.items, key=lambda it: W[u]["scores"][it][0]) for u in sp.test]
        whisper[spk] = {"group": g, "mean": uar(yte, wm), "sum": uar(yte, ws),
                        "gflop": C["speakers"][spk]["per_recognition_flops"]["whisper_forced"] / 1e9}
        for L in range(1, 25):
            s, c = fit(emb(oid, L), oy)
            pr = c.predict(s.transform(emb(sp.test, L)))
            layer_rows.append({"speaker": spk, "group": g, "layer": L, "uar": uar(yte, list(pr)),
                               "gflop": ((a + b * L) * t + (c_ + d_ * L)) / 1e9})
            if L == 12:
                P = c.predict_proba(s.transform(emb(sp.test, L)))
                conf = P.max(1); pred = c.classes_[P.argmax(1)]
                for u, y, pp, cf in zip(sp.test, yte, pred, conf):
                    rej_rows.append({"speaker": spk, "group": g, "correct": bool(pp == y), "conf": float(cf), "item": y})
                # 8-bit weights: same classifier fitted on 8-bit embeddings of the other speakers
                X8tr = np.asarray(E8[[idx[u] for u in oid]], np.float32); X8te = np.asarray(E8[[idx[u] for u in sp.test]], np.float32)
                s8, c8 = fit(X8tr, oy)
                int8_rows.append({"speaker": spk, "group": g, "uar_fp": uar(yte, list(pr)),
                                  "uar_int8": uar(yte, list(c8.predict(s8.transform(X8te))))})
        # number of other speakers
        others = sorted({r["speaker"] for r in other})
        X12 = {r["utt_id"]: None for r in other}
        Xo = emb(oid, 12); spk_of = np.array([r["speaker"] for r in other]); oy_a = np.array(oy)
        Xte = emb(sp.test, 12)
        for m in (sorted({1, 2, 3, 4, 6, 8, len(others)} & set(range(1, len(others) + 1))) if g == "dysarthric" else []):
            for rep in range(5 if m < len(others) else 1):
                pick = set(rng.choice(others, m, replace=False))
                msk = np.isin(spk_of, list(pick))
                if len(set(oy_a[msk])) < 2:
                    continue
                s, c = fit(Xo[msk], list(oy_a[msk]))
                pr = c.predict(s.transform(Xte))
                # words never seen from the chosen speakers cannot be recognised: counted as errors (UAR over all words)
                spk_rows_.append({"speaker": spk, "group": g, "m": m, "rep": rep, "uar": uar(yte, list(pr)),
                                  "n_recordings": int(msk.sum())})
        json.dump({"layer": layer_rows[n0[0]:], "rej": [{**r, "correct": bool(r["correct"])} for r in rej_rows[n0[1]:]],
                   "spk": spk_rows_[n0[2]:], "int8": int8_rows[n0[3]:], "whisper": whisper[spk]},
                  open(cpath, "w"), default=float)
        print(spk, "done", flush=True)

    Ld = pd.DataFrame(layer_rows); R = pd.DataFrame(rej_rows); S = pd.DataFrame(spk_rows_); I8 = pd.DataFrame(int8_rows)
    Wd = pd.DataFrame(whisper).T
    out = {"note": "exploratory design analyses, analysis C", "flop_model_check_L18": chk, "groups": {}}
    for g in sorted(Ld.group.unique()):
        Wg = Wd[Wd.group == g]
        wcol = "mean" if Wg["mean"].astype(float).mean() >= Wg["sum"].astype(float).mean() else "sum"
        wu = float(Wg[wcol].astype(float).mean() * 100)
        wg = float(Wg["gflop"].astype(float).mean())
        lay = Ld[Ld.group == g].groupby("layer").agg(uar=("uar", "mean"), gflop=("gflop", "mean"))
        lay["uar"] *= 100
        ok = lay[lay.uar >= wu]
        cheapest = int(ok.index.min()) if len(ok) else None
        best = int(lay.uar.idxmax())
        # per-speaker check at the cheapest layer
        nb = None
        if cheapest:
            pl = Ld[(Ld.group == g) & (Ld.layer == cheapest)].set_index("speaker").uar
            nb = int(sum(pl[s_] * 100 >= Wd.loc[s_, wcol] * 100 for s_ in pl.index))
        # rejection curve: coverage vs accuracy at confidence thresholds (pooled recordings)
        Rg = R[R.group == g].sort_values("conf", ascending=False)
        curve = []
        for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            n = int(round(cov * len(Rg))); top = Rg.iloc[:n]
            curve.append({"coverage": cov, "accuracy": float(top.correct.mean() * 100), "threshold": float(top.conf.min())})
        # speakers curve
        Sg = S[S.group == g].groupby("m").uar.mean() * 100 if len(S) else pd.Series(dtype=float)
        I8g = I8[I8.group == g]
        out["groups"][g] = {
            "whisper_forced_uar": wu, "whisper_scoring": wcol, "whisper_gflop": wg,
            "layers": {int(L): {"uar": round(float(r.uar), 2), "gflop": round(float(r.gflop), 2)} for L, r in lay.iterrows()},
            "best_layer": best, "best_layer_uar": float(lay.uar.max()),
            "cheapest_layer_matching_whisper": cheapest,
            "cheapest_layer_uar": float(lay.loc[cheapest, "uar"]) if cheapest else None,
            "cheapest_layer_gflop": float(lay.loc[cheapest, "gflop"]) if cheapest else None,
            "cheapest_layer_ratio_whisper": wg / float(lay.loc[cheapest, "gflop"]) if cheapest else None,
            "cheapest_layer_n_speakers_not_below_whisper": nb, "n_speakers": int(len(Wg)),
            "accuracy_coverage": curve,
            "other_speakers_curve": {int(m): round(float(v), 2) for m, v in Sg.items()},
            "int8_shared_uar": float(I8g.uar_int8.mean() * 100), "fp_shared_uar": float(I8g.uar_fp.mean() * 100),
            "int8_max_abs_diff_points": float((I8g.uar_int8 - I8g.uar_fp).abs().max() * 100)}
    # energy-constant sensitivity (J per message for L12 and Whisper forced, 2.2-s scenario)
    anchors = json.load(open("config/energy_anchors.json")) if os.path.exists("config/energy_anchors.json") else {}
    out["energy_anchors_config"] = anchors
    os.makedirs("results/s08i", exist_ok=True)
    Ld.to_csv(f"results/s08i/{corpus}_layers.csv", index=False)
    S.to_csv(f"results/s08i/{corpus}_other_speakers.csv", index=False)
    R.to_csv(f"results/s08i/{corpus}_rejection.csv", index=False)
    json.dump(out, open(f"results/s08i/{corpus}_design.json", "w"), indent=2, default=float)
    for g, o in out["groups"].items():
        print(g, "whisper", round(o["whisper_forced_uar"], 1), "best L", o["best_layer"], round(o["best_layer_uar"], 1),
              "cheapest>=whisper", o["cheapest_layer_matching_whisper"], o["cheapest_layer_uar"], o["cheapest_layer_gflop"],
              "ratio", o["cheapest_layer_ratio_whisper"], "nb", o["cheapest_layer_n_speakers_not_below_whisper"])
        print("  int8", round(o["int8_shared_uar"], 2), "fp", round(o["fp_shared_uar"], 2), "maxdiff", round(o["int8_max_abs_diff_points"], 2))
        print("  coverage", [(c["coverage"], round(c["accuracy"], 1)) for c in o["accuracy_coverage"]])
        print("  speakers", o["other_speakers_curve"])


if __name__ == "__main__":
    main()
