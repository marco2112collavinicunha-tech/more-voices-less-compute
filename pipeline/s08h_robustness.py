"""s08h — robustness of the central comparison (shared classifier vs Whisper-medium forced choice).

Exploratory, added 20 Oct 2026 after the analyses were final; nothing here changes a pre-registered
result. Everything is computed on analysis C (draw-0 test split, identical recordings for all systems).

1. Exact sign-flip permutation test over speakers (all 2^n sign patterns) for the paired
   per-speaker UAR difference, per group.
2. Hierarchical bootstrap (resample speakers, then recordings within speaker; 2,000 replicates)
   for the mean per-speaker UAR difference.
3. Recording-level logistic regression with speaker-clustered (GEE, exchangeable) errors:
   correct ~ system, per group, as a model-based check that does not average within speaker.
4. Leave-one-speaker-out influence: the mean difference with each speaker removed.
5. Analysis flexibility ("forking paths"): the shared classifier re-fitted with 9 other classifier
   settings and 4 other layers; how many of these 13 variants stay above Whisper's mean UAR.

  python -m pipeline.s08h_robustness torgo
Writes results/s08h/<corpus>_robustness.json and _variants.csv.
"""
from __future__ import annotations

import csv
import difflib
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from greenafi.data import by_speaker, load_config, make_split, read_manifest, uar
from pipeline.s08_stats import norm

warnings.filterwarnings("ignore")
B = 2_000


def clf_factory(name):
    return {
        "logreg_C1 (pre-specified)": lambda: LogisticRegression(max_iter=1000),
        "logreg_C0.01": lambda: LogisticRegression(C=0.01, max_iter=2000),
        "logreg_C0.1": lambda: LogisticRegression(C=0.1, max_iter=2000),
        "logreg_C10": lambda: LogisticRegression(C=10, max_iter=2000),
        "lda_shrinkage": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "linear_svm": lambda: LinearSVC(C=0.1, max_iter=1000, dual=False),
        "knn_1": lambda: KNeighborsClassifier(n_neighbors=1),
        "knn_5": lambda: KNeighborsClassifier(n_neighbors=5),
        "nearest_centroid": lambda: NearestCentroid(),
        "knn_5_cosine": lambda: KNeighborsClassifier(n_neighbors=5, metric="cosine"),
    }[name]()


CLFS = ["logreg_C1 (pre-specified)", "logreg_C0.01", "logreg_C0.1", "logreg_C10", "lda_shrinkage",
        "linear_svm", "knn_1", "knn_5", "nearest_centroid", "knn_5_cosine"]
LAYERS = [10, 11, 13, 14]


def fit_predict(Xtr, ytr, Xte, name):
    s = StandardScaler().fit(Xtr)
    return clf_factory(name).fit(s.transform(Xtr), ytr).predict(s.transform(Xte)).tolist()


def signflip(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    allm = [abs((d * np.array(s)).mean()) for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean([m >= obs - 1e-12 for m in allm]))


def main():
    corpus = sys.argv[1]
    cfg = load_config(); p = cfg["personalisation"]
    counts = json.load(open(f"results/s01/{corpus}_counts.json"))
    k = counts["primary_k"]; keep = set(counts["speakers_included"])
    rows = [r for r in read_manifest(f"data/manifests/{corpus}.csv") if r["speaker"] in keep]
    E = np.load(f"data/embeddings/{corpus}/emb.npy", mmap_mode="r")
    idx = {r["utt_id"]: int(r["row"]) for r in csv.DictReader(open(f"data/embeddings/{corpus}/index.csv"))}
    item_of = {r["utt_id"]: r["item"] for r in rows}
    W = {json.loads(l)["utt_id"]: json.loads(l) for l in open(f"results/s06/{corpus}_whisper_testsplit.jsonl")}
    M = json.load(open(f"results/s08b/{corpus}_matched.json"))
    rng = np.random.default_rng(cfg["seed"])

    def emb(ids, L):
        return np.asarray(E[[idx[u] for u in ids], L - 1], dtype=np.float32)

    recs, variants = [], []
    os.makedirs("results/s08h/cache", exist_ok=True)
    for spk, rs in sorted(by_speaker(rows).items()):
        g = rs[0]["group"]
        cpath = f"results/s08h/cache/{corpus}_{spk}.json"
        if os.path.exists(cpath):
            cc = json.load(open(cpath)); recs += cc["recs"]; variants += cc["variants"]; print(spk, "cached"); continue
        n_r, n_v = len(recs), len(variants)
        sp = make_split(rs, spk, k, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
        if sp is None or not all(u in W for u in sp.test):
            continue
        yte = [item_of[u] for u in sp.test]
        other = [r for r in rows if r["speaker"] != spk and r["item"] in set(sp.items)]
        oid, oy = [r["utt_id"] for r in other], [r["item"] for r in other]
        shared = fit_predict(emb(oid, 12), oy, emb(sp.test, 12), CLFS[0])
        fm = [max(sp.items, key=lambda it: W[u]["scores"][it][1]) for u in sp.test]
        fs = [max(sp.items, key=lambda it: W[u]["scores"][it][0]) for u in sp.test]
        for u, y, a, b, c in zip(sp.test, yte, shared, fm, fs):
            recs.append({"speaker": spk, "group": g, "utt": u, "item": y, "shared": a == y,
                         "whisper_forced_mean": b == y, "whisper_forced_sum": c == y})
        for name in CLFS:
            pr = shared if name == CLFS[0] else fit_predict(emb(oid, 12), oy, emb(sp.test, 12), name)
            variants.append({"speaker": spk, "group": g, "variant": f"L12 {name}", "uar": uar(yte, pr)})
        for L in LAYERS:
            pr = fit_predict(emb(oid, L), oy, emb(sp.test, L), CLFS[0])
            variants.append({"speaker": spk, "group": g, "variant": f"L{L} logreg_C1", "uar": uar(yte, pr)})
        json.dump({"recs": [{**r, "shared": bool(r["shared"]), "whisper_forced_mean": bool(r["whisper_forced_mean"]),
                             "whisper_forced_sum": bool(r["whisper_forced_sum"])} for r in recs[n_r:]],
                   "variants": variants[n_v:]}, open(cpath, "w"))
        print(spk, "done", flush=True)

    R = pd.DataFrame(recs); V = pd.DataFrame(variants)
    out = {"note": "exploratory robustness checks, analysis C (draw-0 split)", "groups": {}}
    for g, sub in R.groupby("group"):
        # forced-choice scoring as in the report: the better of mean / summed log-probability for the group
        wcol = max(("whisper_forced_mean", "whisper_forced_sum"),
                   key=lambda c: np.mean([uar(list(s.item), list(np.where(s[c], s.item, "__x__")))
                                          for _, s in sub.groupby("speaker")]))

        def spk_uar(s, col):
            return uar(list(s.item), list(np.where(s[col], s.item, "__wrong__")))
        per = sub.groupby("speaker").apply(lambda s: pd.Series({"shared": spk_uar(s, "shared"), "whisper": spk_uar(s, wcol)}))
        d = (per.shared - per.whisper).to_numpy() * 100
        # hierarchical bootstrap (vectorised per speaker)
        spks = list(per.index); bs = []
        arr = {}
        for s_ in spks:
            t = sub[sub.speaker == s_]
            codes = pd.factorize(t.item)[0]
            arr[s_] = (codes, t.shared.to_numpy(bool), t[wcol].to_numpy(bool), codes.max() + 1)

        def uar_np(codes, ok, n):
            cnt = np.bincount(codes, minlength=n); hit = np.bincount(codes, weights=ok, minlength=n)
            m = cnt > 0
            return (hit[m] / cnt[m]).mean()
        for _ in range(B):
            pick = rng.choice(spks, len(spks), replace=True); vals = []
            for s_ in pick:
                codes, a1, a2, n = arr[s_]; ii = rng.integers(0, len(codes), len(codes))
                vals.append((uar_np(codes[ii], a1[ii], n) - uar_np(codes[ii], a2[ii], n)) * 100)
            bs.append(np.mean(vals))
        bs = np.array(bs)
        # GEE logistic on recordings
        gee = {}
        try:
            import statsmodels.api as sm
            long = pd.concat([pd.DataFrame({"speaker": sub.speaker, "correct": sub.shared.astype(int), "shared": 1}),
                              pd.DataFrame({"speaker": sub.speaker, "correct": sub[wcol].astype(int), "shared": 0})])
            X = sm.add_constant(long[["shared"]].astype(float))
            res = sm.GEE(long.correct, X, groups=long.speaker, family=sm.families.Binomial(),
                         cov_struct=sm.cov_struct.Exchangeable()).fit()
            b, se = float(res.params["shared"]), float(res.bse["shared"])
            gee = {"log_odds_ratio": b, "odds_ratio": float(np.exp(b)), "or_ci95": [float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))],
                   "p": float(res.pvalues["shared"]), "n_recordings": int(len(sub)), "accuracy_shared": float(sub.shared.mean()),
                   "accuracy_whisper": float(sub[wcol].mean())}
        except Exception as e:  # noqa: BLE001
            gee = {"error": str(e)}
        loo = {s: float(np.delete(d, i).mean()) for i, s in enumerate(spks)}
        vg = V[V.group == g].groupby("variant").uar.mean() * 100
        wmean = float(per.whisper.mean() * 100)
        out["groups"][g] = {
            "whisper_scoring": wcol, "n_speakers": len(d), "mean_diff_points": float(d.mean()),
            "per_speaker_diff_points": dict(zip(spks, np.round(d, 2).tolist())), "n_shared_better": int((d > 0).sum()),
            "signflip_exact_p_two_sided": signflip(d), "n_sign_patterns": 2 ** len(d),
            "hier_bootstrap_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "hier_bootstrap_ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
            "hier_bootstrap_p_le0": float(np.mean(bs <= 0)), "gee": gee,
            "loo_mean_diff_points": loo, "loo_min": float(min(loo.values())), "loo_max": float(max(loo.values())),
            "whisper_mean_uar": wmean, "variants_mean_uar": vg.round(2).to_dict(),
            "n_variants": int(len(vg)), "n_variants_above_whisper": int((vg > wmean).sum()),
            "variants_min": float(vg.min()), "variants_max": float(vg.max()),
            "prespecified_check_vs_s08b": M["systems"]["C_full_split_whisper"][g]["shared_L12_logreg"]["mean_uar"] * 100}
    os.makedirs("results/s08h", exist_ok=True)
    V.to_csv(f"results/s08h/{corpus}_variants.csv", index=False)
    json.dump(out, open(f"results/s08h/{corpus}_robustness.json", "w"), indent=2, default=float)
    for g, o in out["groups"].items():
        print(g, round(o["mean_diff_points"], 2), "signflip p", round(o["signflip_exact_p_two_sided"], 3),
              "hier95", np.round(o["hier_bootstrap_ci95"], 2), "GEE OR", o["gee"].get("odds_ratio"), o["gee"].get("p"),
              "LOO", round(o["loo_min"], 2), round(o["loo_max"], 2),
              "variants above Whisper", o["n_variants_above_whisper"], "/", o["n_variants"], round(o["variants_min"], 1), round(o["variants_max"], 1))


if __name__ == "__main__":
    main()
