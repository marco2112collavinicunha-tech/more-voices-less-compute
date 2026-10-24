"""Stage 8c — where the accuracy comes from: examples (exploratory, descriptive).

  python -m pipeline.s08c_examples

Three analyses on the stored embeddings, all at layer 12 unless stated, every
system scored on the same test recordings of each split:

1. k-curve. Speakers that qualify at every k of a corpus (Speech Commands:
   k = 1, 2, 4, 8; TORGO: k = 1, 2), so the curve is over a fixed set of
   speakers. Systems: own recordings (1-NN; logistic regression for k >= 2),
   shared classifier (other speakers only), shared + own. Five draws.
2. Source of the shared examples (TORGO, k = 1, draw 0): the shared classifier
   trained on all other speakers, on control speakers only, or on speakers
   with dysarthria only.
3. All layers vs layer 12 (TORGO, k = 1, draw 0): the same classifiers on the
   average of all 24 layers' embeddings, a stand-in for the learned weighted
   sum of layers commonly used with these models (it needs all 24 layers).
The k-curve was planned (k values pre-registered); 2 and 3 were added after the
first results. Writes results/s08c/examples.json.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np

from greenafi.data import by_speaker, eligible_items, load_config, make_split, read_manifest, uar
from pipeline.s05_classifiers import fit_predict


def setup(corpus):
    counts = json.load(open(f"results/s01/{corpus}_counts.json"))
    rows = [r for r in read_manifest(f"data/manifests/{corpus}.csv") if r["speaker"] in set(counts["speakers_included"])]
    E = np.load(f"data/embeddings/{corpus}/emb.npy", mmap_mode="r")
    idx = {r["utt_id"]: int(r["row"]) for r in csv.DictReader(open(f"data/embeddings/{corpus}/index.csv"))}
    return counts, rows, E, idx


def main():
    cfg = load_config()
    p = cfg["personalisation"]
    out = {"k_curve": {}, "source": {}, "all_layers": {}}
    for corpus, ks in (("speech_commands", (1, 2, 4, 8)), ("torgo", (1, 2))):
        counts, rows, E, idx = setup(corpus)
        item_of = {r["utt_id"]: r["item"] for r in rows}
        spk_rows = by_speaker(rows)
        emb = lambda ids, L=12: np.asarray(E[[idx[u] for u in ids], L - 1], np.float32)
        keep = [s for s, rs in spk_rows.items()
                if all(len(eligible_items(rs, k, p["min_test_per_item"])) >= p["min_items_per_speaker"] for k in ks)]
        res = defaultdict(lambda: defaultdict(list))  # (k, system) -> speaker -> [uar per draw]
        for s in sorted(keep):
            rs = spk_rows[s]
            for k in ks:
                for d in range(p["n_draws"]):
                    sp = make_split(rs, s, k, d, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
                    ytr = [item_of[u] for u in sp.train]
                    yte = [item_of[u] for u in sp.test]
                    Xtr, Xte = emb(sp.train), emb(sp.test)
                    other = [r for r in rows if r["speaker"] != s and r["item"] in set(sp.items)]
                    if len({r["item"] for r in other}) < len(sp.items):
                        continue
                    oid, oy = [r["utt_id"] for r in other], [r["item"] for r in other]
                    Xo = emb(oid)
                    sysp = {"own_1nn": fit_predict(Xtr, ytr, Xte, "1nn")}
                    if k >= 2:
                        sysp["own_logreg"] = fit_predict(Xtr, ytr, Xte, "logreg")
                    sysp["shared"] = fit_predict(Xo, oy, Xte, "logreg")
                    sysp["shared_own"] = fit_predict(np.vstack([Xo, Xtr]), oy + ytr, Xte, "logreg")
                    for name, pr in sysp.items():
                        res[(k, name)][s].append(uar(yte, pr))
            print(corpus, s, "k-curve done", flush=True)
        cur = {}
        for (k, name), d in res.items():
            per = {s: float(np.mean(v)) for s, v in d.items()}
            cur.setdefault(str(k), {})[name] = {"mean_uar": float(np.mean(list(per.values()))), "n_speakers": len(per),
                                                "per_speaker": per}
        groups = {s: spk_rows[s][0]["group"] for s in keep}
        out["k_curve"][corpus] = {"speakers": groups, "curve": cur}

        if corpus != "torgo":
            continue
        # 2. source of the shared examples, and 3. all layers vs layer 12
        A = np.zeros((E.shape[0], E.shape[2]), np.float32)  # mean over the 24 layers, computed once in chunks
        for i in range(0, E.shape[0], 256):
            A[i:i + 256] = np.asarray(E[i:i + 256], np.float32).mean(axis=1)
        json.dump(out, open("results/s08c/partial.json", "w"))
        for s, rs in sorted(spk_rows.items()):
            g = rs[0]["group"]
            sp = make_split(rs, s, 1, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
            if sp is None:
                continue
            yte = [item_of[u] for u in sp.test]
            ytr = [item_of[u] for u in sp.train]
            src = {"all": lambda r: True, "controls_only": lambda r: r["group"] == "control",
                   "dysarthric_only": lambda r: r["group"] == "dysarthric"}
            for name, f in src.items():
                other = [r for r in rows if r["speaker"] != s and r["item"] in set(sp.items) and f(r)]
                if len({r["item"] for r in other}) < len(sp.items):
                    continue
                pr = fit_predict(emb([r["utt_id"] for r in other]), [r["item"] for r in other], emb(sp.test), "logreg")
                out["source"].setdefault(g, {}).setdefault(name, {})[s] = {"uar": uar(yte, pr), "n_train": len(other)}
            # all layers averaged
            def avg(ids):
                return A[[idx[u] for u in ids]]
            other = [r for r in rows if r["speaker"] != s and r["item"] in set(sp.items)]
            Xo, Xte, Xtr = avg([r["utt_id"] for r in other]), avg(sp.test), avg(sp.train)
            out["all_layers"].setdefault(g, {}).setdefault("own_1nn_avg24", {})[s] = uar(yte, fit_predict(Xtr, ytr, Xte, "1nn"))
            out["all_layers"].setdefault(g, {}).setdefault("shared_avg24", {})[s] = uar(
                yte, fit_predict(Xo, [r["item"] for r in other], Xte, "logreg"))
            print("torgo", s, "source/all-layers done", flush=True)
    os.makedirs("results/s08c", exist_ok=True)
    json.dump(out, open("results/s08c/examples.json", "w"), indent=2)
    for c, v in out["k_curve"].items():
        print(c, len(v["speakers"]), {k: {n: round(x["mean_uar"], 3) for n, x in d.items()} for k, d in v["curve"].items()})
    for g, d in out["source"].items():
        print(g, {n: round(float(np.mean([x["uar"] for x in v.values()])), 3) for n, v in d.items()},
              {n: len(v) for n, v in d.items()})
    for g, d in out["all_layers"].items():
        print(g, {n: round(float(np.mean(list(v.values()))), 3) for n, v in d.items()})


if __name__ == "__main__":
    main()
