"""Stage 8d — number of messages and speaker consistency (exploratory, descriptive).

  python -m pipeline.s08d_vocab_consistency

1. Vocabulary size. For each TORGO speaker (k = 1, draw 0, the split every
   system was run on), random subsets of K of the speaker's words are drawn
   (20 per K, seeded); the test recordings of those words are recognised by
   choosing among the K words only. The shared classifier and the own-recording
   classifier are fitted once on all of the speaker's words and decide within
   the subset (argmax of their scores over the K words); Whisper's forced choice
   compares the K words' scores. Accuracy is averaged over subsets.
2. Consistency. For each speaker, at layer 12: the mean cosine distance between
   recordings of the same word, divided by the mean distance between recordings
   of different words (lower = more consistent repetitions). Compared with each
   system's accuracy across speakers (Spearman correlation, descriptive).
Writes results/s08d/vocab_consistency.json.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from greenafi.data import by_speaker, load_config, make_split, read_manifest, uar

KS = (5, 10, 20, 40, 80)
N_SUB = 20


def main():
    cfg = load_config()
    p = cfg["personalisation"]
    counts = json.load(open("results/s01/torgo_counts.json"))
    rows = [r for r in read_manifest("data/manifests/torgo.csv") if r["speaker"] in set(counts["speakers_included"])]
    E = np.load("data/embeddings/torgo/emb.npy", mmap_mode="r")
    idx = {r["utt_id"]: int(r["row"]) for r in csv.DictReader(open("data/embeddings/torgo/index.csv"))}
    item_of = {r["utt_id"]: r["item"] for r in rows}
    emb = lambda ids: np.asarray(E[[idx[u] for u in ids], 11], np.float32)
    W = {}
    if os.path.exists("results/s06/torgo_whisper_testsplit.jsonl"):
        for line in open("results/s06/torgo_whisper_testsplit.jsonl"):
            r = json.loads(line)
            W[r["utt_id"]] = r
    out = {"K": list(KS), "speakers": {}}
    for spk, rs in sorted(by_speaker(rows).items()):
        g = rs[0]["group"]
        sp = make_split(rs, spk, 1, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
        items = list(sp.items)
        yte = np.array([item_of[u] for u in sp.test])
        # shared classifier scores (others' recordings), own-recording 1-NN distances
        other = [r for r in rows if r["speaker"] != spk and r["item"] in set(items)]
        Xo = emb([r["utt_id"] for r in other]); yo = [r["item"] for r in other]
        sc = StandardScaler().fit(Xo)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xo), yo)
        Xte = emb(sp.test)
        P = clf.decision_function(sc.transform(Xte))  # n_test x n_items, columns = clf.classes_
        cls = list(clf.classes_)
        Xtr = emb(sp.train); ytr = [item_of[u] for u in sp.train]
        s2 = StandardScaler().fit(Xtr)
        D = ((s2.transform(Xte)[:, None, :] - s2.transform(Xtr)[None, :, :]) ** 2).sum(-1)  # n_test x n_train
        has_w = all(u in W for u in sp.test)
        rng = np.random.default_rng(20261014 + sum(map(ord, spk)))
        res = {}
        for K in [k for k in KS if k < len(items)] + [len(items)]:
            acc = {"shared": [], "own_1nn": [], "whisper_forced": []}
            subs = [items] if K == len(items) else [list(rng.choice(items, K, replace=False)) for _ in range(N_SUB)]
            for sub in subs:
                m = np.isin(yte, sub)
                yt = list(yte[m])
                ci = [cls.index(w) for w in sub]
                acc["shared"].append(uar(yt, [sub[j] for j in P[m][:, ci].argmax(1)]))
                ti = [i for i, w in enumerate(ytr) if w in sub]
                acc["own_1nn"].append(uar(yt, [ytr[ti[j]] for j in D[m][:, ti].argmin(1)]))
                if has_w:
                    tu = [u for u, mm in zip(sp.test, m) if mm]
                    acc["whisper_forced"].append(uar(yt, [max(sub, key=lambda w: W[u]["scores"][w][1]) for u in tu]))
            res[str(K)] = {k_: float(np.mean(v)) for k_, v in acc.items() if v}
        # consistency at layer 12, all of the speaker's recordings of eligible words
        ids = [r["utt_id"] for r in rs if r["item"] in set(items)]
        X = emb(ids); X /= np.linalg.norm(X, axis=1, keepdims=True)
        lab = np.array([item_of[u] for u in ids])
        C = 1 - X @ X.T
        same = (lab[:, None] == lab[None, :]) & ~np.eye(len(ids), dtype=bool)
        diff = lab[:, None] != lab[None, :]
        out["speakers"][spk] = {"group": g, "n_items": len(items), "vocab": res, "has_whisper_split": has_w,
                                "within_word": float(C[same].mean()), "between_word": float(C[diff].mean()),
                                "inconsistency": float(C[same].mean() / C[diff].mean())}
        print(spk, g, {K: {k_: round(v, 3) for k_, v in d.items()} for K, d in res.items()},
              round(out["speakers"][spk]["inconsistency"], 3), flush=True)
    # correlation of inconsistency with accuracy (from s08b, analysis A)
    import pandas as pd
    ps = pd.read_csv("results/s08b/torgo_matched_per_speaker.csv")
    A = ps[ps.analysis_group == "A_finetune_split"]
    out["correlation"] = {}
    for system in ("shared_L12_logreg", "frozen_L12_1nn", "xpl_shared_own_L12_logreg"):
        v = A[A.system == system].set_index("speaker").uar
        sp_ = [s for s in out["speakers"] if s in v.index]
        r, pv = stats.spearmanr([out["speakers"][s]["inconsistency"] for s in sp_], [v[s] for s in sp_])
        out["correlation"][system] = {"spearman_r": float(r), "p": float(pv), "n": len(sp_)}
    print(out["correlation"])
    os.makedirs("results/s08d", exist_ok=True)
    json.dump(out, open("results/s08d/vocab_consistency.json", "w"), indent=2)


if __name__ == "__main__":
    main()
