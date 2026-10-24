"""s08f — sensitivity: test recordings from the same session as the learning recording.

The split takes the learning recording from the speaker's earliest session and puts every
other recording in the test set, so part of the test set shares the learning recording's
session. This stage counts that overlap on the draw-0 split (the split of analyses A and C)
and recomputes UAR on the test recordings from OTHER sessions only, for the frozen,
shared and Whisper systems. Descriptive; added after the final audit (15 Oct 2026).

Run: python -m pipeline.s08f_session torgo
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

from greenafi.data import by_speaker, load_config, make_split, read_manifest, uar
from pipeline.s05_classifiers import fit_predict


def main():
    corpus = sys.argv[1]
    cfg = load_config()
    p = cfg["personalisation"]
    counts = json.load(open(f"results/s01/{corpus}_counts.json"))
    k = counts["primary_k"]
    keep = set(counts["speakers_included"])
    rows = [r for r in read_manifest(f"data/manifests/{corpus}.csv") if r["speaker"] in keep]
    E = np.load(f"data/embeddings/{corpus}/emb.npy", mmap_mode="r")
    idx = {r["utt_id"]: int(r["row"]) for r in csv.DictReader(open(f"data/embeddings/{corpus}/index.csv"))}
    item_of = {r["utt_id"]: r["item"] for r in rows}
    sess = {r["utt_id"]: (r.get("session") or "") for r in rows}
    W2 = {}
    wp2 = f"results/s06/{corpus}_whisper_testsplit.jsonl"
    if os.path.exists(wp2):
        for line in open(wp2):
            r = json.loads(line)
            W2[r["utt_id"]] = r

    def emb(ids, L=12):
        return np.asarray(E[[idx[u] for u in ids], L - 1], dtype=np.float32)

    out = {"corpus": corpus, "k": k, "speakers": {}}
    for spk, rs in sorted(by_speaker(rows).items()):
        sp = make_split(rs, spk, k, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
        if sp is None:
            continue
        train_sess = {item_of[u]: sess[u] for u in sp.train}
        same = np.array([sess[u] == train_sess[item_of[u]] for u in sp.test])
        yte = np.array([item_of[u] for u in sp.test])
        ytr = [item_of[u] for u in sp.train]
        other = [r for r in rows if r["speaker"] != spk and r["item"] in set(sp.items)]
        oid, oy = [r["utt_id"] for r in other], [r["item"] for r in other]
        preds = {
            "frozen_L12_1nn": np.array(fit_predict(emb(sp.train), ytr, emb(sp.test), "1nn")),
            "shared_L12_logreg": np.array(fit_predict(emb(oid), oy, emb(sp.test), "logreg")),
            "xpl_shared_own_L12_logreg": np.array(fit_predict(emb(oid + sp.train), oy + ytr, emb(sp.test), "logreg")),
        }
        if all(u in W2 for u in sp.test):
            preds["whisper_forced_mean"] = np.array([max(sp.items, key=lambda it: W2[u]["scores"][it][1]) for u in sp.test])
        rec = {"group": rs[0]["group"], "n_test": int(len(yte)), "n_same_session": int(same.sum()),
               "all": {s: uar(list(yte), list(pr)) for s, pr in preds.items()}}
        cross = ~same
        if cross.sum() > 0:
            rec["n_items_cross"] = int(len(set(yte[cross])))
            rec["cross_session"] = {s: uar(list(yte[cross]), list(pr[cross])) for s, pr in preds.items()}
        if same.sum() > 0:
            rec["same_session"] = {s: uar(list(yte[same]), list(pr[same])) for s, pr in preds.items()}
        out["speakers"][spk] = rec
        print(spk, rec["n_same_session"], "/", rec["n_test"], {s: round(v, 3) for s, v in rec.get("cross_session", {}).items()}, flush=True)
    os.makedirs("results/s08f", exist_ok=True)
    json.dump(out, open(f"results/s08f/{corpus}_session.json", "w"), indent=2)


if __name__ == "__main__":
    main()
