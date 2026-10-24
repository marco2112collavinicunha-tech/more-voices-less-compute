"""s08k — the same shared classifier on Whisper's encoder instead of XLSR-53 layer 12.

Exploratory, added 21 Oct 2026; reading (3) declared on 20 Oct in results/DEVIATIONS.md:
if the shared classifier on Whisper-encoder embeddings matches the XLSR-53 one, the gain comes
from the examples and the classifier rather than from XLSR-53; if lower, the encoder matters.

Same draw-0 split, same other-speaker training set and same logistic regression as
`shared_L12_logreg` in s08b. Embeddings from s04b (30-s padded encoder, mean over the frames
covering the audio). Computation per recognition = Whisper encoder (s08g cost formula).
  python -m pipeline.s08k_whisper_encoder_shared torgo
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pandas as pd

from greenafi.data import by_speaker, load_config, make_split, read_manifest, uar
from greenafi.stats import tost_paired
from pipeline.s05_classifiers import fit_predict
from pipeline.s08g_whisper_sizes import costs


def main():
    corpus = sys.argv[1]
    cfg = load_config(); p = cfg["personalisation"]
    counts = json.load(open(f"results/s01/{corpus}_counts.json"))
    k = counts["primary_k"]; keep = set(counts["speakers_included"])
    rows = [r for r in read_manifest(f"data/manifests/{corpus}.csv") if r["speaker"] in keep]
    item_of = {r["utt_id"]: r["item"] for r in rows}
    C = json.load(open(f"results/s03d/{corpus}_costs.json"))
    ps = pd.read_csv(f"results/s08b/{corpus}_matched_per_speaker.csv")
    shared = ps[(ps.analysis_group == "C_full_split_whisper") & (ps.system == "shared_L12_logreg")].set_index("speaker").uar
    out, per = {}, []
    for tag in ("tiny", "base", "medium"):
        d = f"data/embeddings/{corpus}_whisper_{tag}"
        if not os.path.exists(f"{d}/emb.npy"):
            continue
        E = np.load(f"{d}/emb.npy").astype(np.float32)
        idx = {r["utt_id"]: int(r["row"]) for r in csv.DictReader(open(f"{d}/index.csv"))}
        for spk, rs in sorted(by_speaker(rows).items()):
            sp = make_split(rs, spk, k, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
            if sp is None:
                continue
            other = [r for r in rows if r["speaker"] != spk and r["item"] in set(sp.items)]
            oid, oy = [r["utt_id"] for r in other], [r["item"] for r in other]
            pred = fit_predict(E[[idx[u] for u in oid]], oy, E[[idx[u] for u in sp.test]], "logreg")
            yte = [item_of[u] for u in sp.test]
            l12 = C["speakers"][spk]["per_recognition_flops"]["L12"]
            per.append({"speaker": spk, "group": rs[0]["group"], "encoder": tag, "uar": uar(yte, list(pred)),
                        "shared_L12": float(shared.get(spk, np.nan)),
                        "gflop_encoder": costs(tag, len(sp.items))["encoder"] / 1e9,
                        "ratio_encoder_to_L12": costs(tag, len(sp.items))["encoder"] / l12})
    df = pd.DataFrame(per)
    for (g, tag), sub in df.groupby(["group", "encoder"]):
        dd = ((sub.shared_L12 - sub.uar) * 100).to_numpy()
        out.setdefault(g, {})[tag] = {"n_speakers": int(len(sub)), "uar": float(sub.uar.mean()),
                                      "gflop_encoder": float(sub.gflop_encoder.mean()),
                                      "ratio_encoder_to_L12_mean": float(sub.ratio_encoder_to_L12.mean()),
                                      "xlsr_minus_whisper_encoder": tost_paired(dd, 5.0, 0.05),
                                      "n_xlsr_better": int((dd > 0).sum()),
                                      "per_speaker_diff": dict(zip(sub.speaker, dd.round(2).tolist()))}
    os.makedirs("results/s08k", exist_ok=True)
    df.to_csv(f"results/s08k/{corpus}_whisper_encoder_shared_per_speaker.csv", index=False)
    json.dump(out, open(f"results/s08k/{corpus}_whisper_encoder_shared.json", "w"), indent=2, default=float)
    for g, d_ in out.items():
        for tag, v in d_.items():
            print(g, tag, round(v["uar"] * 100, 1), "GFLOP", round(v["gflop_encoder"], 1),
                  "diff", round(v["xlsr_minus_whisper_encoder"]["mean_diff"], 2),
                  (round(v["xlsr_minus_whisper_encoder"]["ci_90_low"], 1), round(v["xlsr_minus_whisper_encoder"]["ci_90_high"], 1)),
                  "xlsr better", v["n_xlsr_better"], "/", v["n_speakers"])


if __name__ == "__main__":
    main()
