"""Stage 8e — the exchange rate between examples and computation (exploratory).

  python -m pipeline.s08e_exchange

All on TORGO, k = 1, draw 0 (the split every system was run on), layer 12 unless
stated, logistic regression as in s05 (standardised features).

1. Exchange-rate curve: the shared classifier trained on n recordings per word
   from the other speakers (n = 1, 2, 4, 8, 16, 24, all; 10 random draws per n),
   to find how many other-speaker examples match Whisper-medium.
2. Equal-size sources: for each word, the same number of recordings from control
   speakers and from speakers with dysarthria (the smaller of the two counts),
   10 draws, so that the comparison of sources is not a comparison of sizes.
3. Linear probe on the speaker's own single recording per word (logistic
   regression at layers 12 and 24): a floor that costs seconds of CPU, for the
   retraining baselines.
4. Other recording conditions: a classifier trained on Speech Commands (other
   speakers, other microphones, crowd-sourced) for the TORGO words that Speech
   Commands also contains, against the TORGO-trained shared classifier on the same
   words and test recordings.
Writes results/s08e/exchange.json.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np

from greenafi.data import by_speaker, load_config, make_split, read_manifest, uar
from pipeline.s05_classifiers import fit_predict

NS = (1, 2, 4, 8, 16, 24)
REPS = 10


def load(corpus):
    counts = json.load(open(f"results/s01/{corpus}_counts.json"))
    rows = [r for r in read_manifest(f"data/manifests/{corpus}.csv") if r["speaker"] in set(counts["speakers_included"])]
    E = np.load(f"data/embeddings/{corpus}/emb.npy", mmap_mode="r")
    idx = {r["utt_id"]: int(r["row"]) for r in csv.DictReader(open(f"data/embeddings/{corpus}/index.csv"))}
    return counts, rows, E, idx


def main():
    cfg = load_config()
    p = cfg["personalisation"]
    _, rows, E, idx = load("torgo")
    emb = lambda ids, L=12: np.asarray(E[[idx[u] for u in ids], L - 1], np.float32)
    # Speech Commands: all included speakers' recordings, all words
    _, sc_rows, Esc, idx_sc = load("speech_commands")
    sc_by_word = defaultdict(list)
    for r in sc_rows:
        sc_by_word[r["item"]].append(r["utt_id"])
    out = {"n_values": list(NS), "speakers": {}}
    for spk, rs in sorted(by_speaker(rows).items()):
        g = rs[0]["group"]
        sp = make_split(rs, spk, 1, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
        items = list(sp.items)
        yte = [r_ for r_ in (dict((r["utt_id"], r["item"]) for r in rs)[u] for u in sp.test)]
        Xte = emb(sp.test)
        other = [r for r in rows if r["speaker"] != spk and r["item"] in set(items)]
        per_word = defaultdict(list)
        for r in other:
            per_word[r["item"]].append(r)
        rng = np.random.default_rng(20261015 + sum(map(ord, spk)))
        rec = {"group": g, "n_items": len(items), "curve": {}}
        # 1. exchange-rate curve
        for n in NS:
            us = []
            for _ in range(REPS):
                pick = [r for w in items for r in rng.permutation(per_word[w])[:n]]
                us.append(uar(yte, fit_predict(emb([r["utt_id"] for r in pick]), [r["item"] for r in pick], Xte, "logreg")))
            rec["curve"][str(n)] = float(np.mean(us))
        rec["curve"]["all"] = uar(yte, fit_predict(emb([r["utt_id"] for r in other]), [r["item"] for r in other], Xte, "logreg"))
        rec["mean_other_per_word"] = float(np.mean([len(per_word[w]) for w in items]))
        # 2. equal-size sources
        ctl = {w: [r for r in per_word[w] if r["group"] == "control"] for w in items}
        dys = {w: [r for r in per_word[w] if r["group"] == "dysarthric"] for w in items}
        if all(ctl[w] and dys[w] for w in items):
            ue = {"controls": [], "dysarthric": []}
            for _ in range(REPS):
                for name, src in (("controls", ctl), ("dysarthric", dys)):
                    pick = [r for w in items for r in rng.permutation(src[w])[:min(len(ctl[w]), len(dys[w]))]]
                    ue[name].append(uar(yte, fit_predict(emb([r["utt_id"] for r in pick]), [r["item"] for r in pick], Xte, "logreg")))
            rec["equal_sources"] = {k: float(np.mean(v)) for k, v in ue.items()}
            rec["equal_sources"]["per_word"] = float(np.mean([min(len(ctl[w]), len(dys[w])) for w in items]))
        # 3. linear probe on own single recording per word
        ytr = [dict((r["utt_id"], r["item"]) for r in rs)[u] for u in sp.train]
        for L in (12, 24):
            rec[f"own_logreg_L{L}"] = uar(yte, fit_predict(emb(sp.train, L), ytr, emb(sp.test, L), "logreg"))
        # 4. other recording conditions (Speech Commands vocabulary overlap)
        common = [w for w in items if w in sc_by_word]
        if len(common) >= 5:
            m = [i for i, y in enumerate(yte) if y in common]
            yt = [yte[i] for i in m]
            Xs = np.asarray(Esc[[idx_sc[u] for w in common for u in sc_by_word[w]], 11], np.float32)
            ys = [w for w in common for _ in sc_by_word[w]]
            pick = [r for w in common for r in per_word[w]]
            rec["cross_corpus"] = {
                "n_words": len(common), "n_test": len(yt),
                "speech_commands_trained": uar(yt, fit_predict(Xs, ys, Xte[m], "logreg")),
                "torgo_trained": uar(yt, fit_predict(emb([r["utt_id"] for r in pick]), [r["item"] for r in pick], Xte[m], "logreg")),
                "n_train_sc_per_word": float(len(ys) / len(common))}
        out["speakers"][spk] = rec
        print(spk, g, {k: round(v, 3) for k, v in rec["curve"].items()}, rec.get("equal_sources"),
              round(rec["own_logreg_L12"], 3), round(rec["own_logreg_L24"], 3), rec.get("cross_corpus"), flush=True)
    os.makedirs("results/s08e", exist_ok=True)
    json.dump(out, open("results/s08e/exchange.json", "w"), indent=2)


if __name__ == "__main__":
    main()
