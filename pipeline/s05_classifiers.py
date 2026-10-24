"""Stage 5 — frozen-encoder personalisation: every speaker, k, draw, layer, classifier.

  python -m pipeline.s05_classifiers torgo

Systems written to results/s05/<corpus>_uar.csv (one row per speaker x k x draw x system):
  frozen_L{layer}_{1nn|logreg}   personal classifier on the speaker's own k recordings per item
  frozen_L12_int8_1nn            same with the int8 encoder
  shared_L{12,18,24}_logreg      classifier trained on the OTHER speakers' recordings of the
                                 same items (no per-user personalisation cost)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from greenafi.data import by_speaker, load_config, make_split, read_manifest, uar

warnings.filterwarnings("ignore")


def fit_predict(Xtr, ytr, Xte, kind):
    s = StandardScaler().fit(Xtr)
    if kind == "1nn":
        clf = KNeighborsClassifier(n_neighbors=1)
    else:
        clf = LogisticRegression(max_iter=1000)
    return clf.fit(s.transform(Xtr), ytr).predict(s.transform(Xte)).tolist()


def main():
    corpus = sys.argv[1]
    cfg = load_config()
    p = cfg["personalisation"]
    counts = json.load(open(f"results/s01/{corpus}_counts.json"))
    keep = set(counts["speakers_included"])
    rows = [r for r in read_manifest(f"data/manifests/{corpus}.csv") if r["speaker"] in keep]

    E = np.load(f"data/embeddings/{corpus}/emb.npy", mmap_mode="r")
    E8 = np.load(f"data/embeddings/{corpus}/emb_int8_l12.npy")
    index = list(csv.DictReader(open(f"data/embeddings/{corpus}/index.csv")))
    row_of = {r["utt_id"]: int(r["row"]) for r in index}
    item_of = {r["utt_id"]: r["item"] for r in rows}
    spk_rows = by_speaker(rows)

    os.makedirs("results/s05", exist_ok=True)
    out = open(f"results/s05/{corpus}_uar.csv", "w", newline="")
    w = csv.writer(out)
    w.writerow(["corpus", "speaker", "group", "k", "draw", "system", "layer", "classifier", "uar", "n_items", "n_train", "n_test"])
    for spk, rs in sorted(spk_rows.items()):
        group = rs[0]["group"]
        for k in p["k_values"]:
            for d in range(p["n_draws"]):
                sp = make_split(rs, spk, k, d, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
                if sp is None:
                    continue
                tr = [row_of[u] for u in sp.train]
                te = [row_of[u] for u in sp.test]
                ytr = [item_of[u] for u in sp.train]
                yte = [item_of[u] for u in sp.test]
                base = [corpus, spk, group, k, d]
                tail = [len(sp.items), len(tr), len(te)]
                for L in cfg["encoder"]["layers_swept"]:
                    Xtr = np.asarray(E[tr, L - 1], dtype=np.float32)
                    Xte = np.asarray(E[te, L - 1], dtype=np.float32)
                    for kind in p["classifiers"]:
                        if kind == "logreg" and k == 1:
                            continue  # one example per class: logistic regression is not meaningful
                        u = uar(yte, fit_predict(Xtr, ytr, Xte, kind))
                        w.writerow(base + [f"frozen_L{L}_{kind}", L, kind, f"{u:.4f}"] + tail)
                u = uar(yte, fit_predict(E8[tr], ytr, E8[te], "1nn"))
                w.writerow(base + ["frozen_L12_int8_1nn", 12, "1nn", f"{u:.4f}"] + tail)
                # shared classifier: other speakers' recordings of the same items
                if d == 0:
                    other = [r for r in rows if r["speaker"] != spk and r["item"] in set(sp.items)]
                    if len({r["item"] for r in other}) == len(sp.items):
                        ot = [row_of[r["utt_id"]] for r in other]
                        oy = [r["item"] for r in other]
                        for L in (12, 18, 24):
                            pred = fit_predict(np.asarray(E[ot, L - 1], np.float32), oy,
                                               np.asarray(E[te, L - 1], np.float32), "logreg")
                            w.writerow(base + [f"shared_L{L}_logreg", L, "logreg", f"{uar(yte, pred):.4f}"] + tail)
            out.flush()
        print(spk, "done", flush=True)
    out.close()


if __name__ == "__main__":
    main()
