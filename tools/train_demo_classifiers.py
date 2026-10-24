"""Fit the classifiers used by demo/live_demo.py and save them as plain arrays (no pickle).

- commands: logistic regression on layer-12 XLSR-53 embeddings of 10 Speech Commands words,
  all speakers in the study's Speech Commands subset (for a live demo with any English speaker).
- torgo_<SPK>: for each included TORGO speaker with dysarthria, the shared classifier of the
  report (other included speakers' recordings of that speaker's words), plus the speaker's
  draw-0 test recordings, so the demo replays exactly the report's analysis-C recognitions.

  python -m tools.train_demo_classifiers
"""
import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from greenafi.data import by_speaker, load_config, make_split, read_manifest

WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def save(path, X, y, extra=None):
    s = StandardScaler().fit(X)
    c = LogisticRegression(max_iter=2000).fit(s.transform(X), y)
    np.savez_compressed(path, mean=s.mean_, scale=s.scale_, coef=c.coef_, intercept=c.intercept_,
                        classes=np.array(c.classes_), **(extra or {}))
    return c


def emb_loader(corpus):
    E = np.load(f"data/embeddings/{corpus}/emb.npy", mmap_mode="r")
    idx = {r["utt_id"]: int(r["row"]) for r in csv.DictReader(open(f"data/embeddings/{corpus}/index.csv"))}
    return lambda ids: np.asarray(E[[idx[u] for u in ids], 11], dtype=np.float32)


def main():
    cfg = load_config(); p = cfg["personalisation"]
    sc = [r for r in read_manifest("data/manifests/speech_commands.csv") if r["item"] in WORDS]
    e = emb_loader("speech_commands")
    have = {r["utt_id"] for r in csv.DictReader(open("data/embeddings/speech_commands/index.csv"))}
    sc = [r for r in sc if r["utt_id"] in have]
    save("demo/classifier_commands.npz", e([r["utt_id"] for r in sc]), [r["item"] for r in sc])
    print("commands:", len(sc), "recordings,", len({r['speaker'] for r in sc}), "speakers")
    counts = json.load(open("results/s01/torgo_counts.json"))
    keep = set(counts["speakers_included"]); k = counts["primary_k"]
    rows = [r for r in read_manifest("data/manifests/torgo.csv") if r["speaker"] in keep]
    e = emb_loader("torgo")
    for spk, rs in sorted(by_speaker(rows).items()):
        if rs[0]["group"] != "dysarthric":
            continue
        sp = make_split(rs, spk, k, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
        other = [r for r in rows if r["speaker"] != spk and r["item"] in set(sp.items)]
        path_of = {r["utt_id"]: r["path"] for r in rs}; item_of = {r["utt_id"]: r["item"] for r in rs}
        save(f"demo/classifier_torgo_{spk}.npz", e([r["utt_id"] for r in other]), [r["item"] for r in other],
             {"test_paths": np.array([path_of[u] for u in sp.test]), "test_items": np.array([item_of[u] for u in sp.test])})
        print(spk, len(other), "training recordings from other speakers;", len(sp.test), "test recordings")


if __name__ == "__main__":
    main()
