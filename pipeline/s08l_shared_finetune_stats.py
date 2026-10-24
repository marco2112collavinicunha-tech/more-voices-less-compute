"""s08l — summary of s07b (XLSR-53 cut at layer 12, retrained once on other speakers' recordings).

Exploratory. Fixed endpoint: epoch 5, the last epoch of the s07b plan; epochs 1 and 3 reported too.
Paired against the shared classifier (shared_L12_logreg, no retraining) and Whisper-medium forced choice
(analysis C of s08b) on the same draw-0 test split, with an exact sign-flip test.
  python -m pipeline.s08l_shared_finetune_stats torgo
"""
from __future__ import annotations
import itertools, json, os, sys
import numpy as np, pandas as pd


def signflip(d):
    d = np.asarray(d, float); d = np.where(np.abs(d) <= 0.05, 0.0, d); s = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float((np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-9).mean())


def main():
    corpus = sys.argv[1]
    ft = pd.read_csv(f"results/s07b/{corpus}_shared_finetune.csv")
    ps = pd.read_csv(f"results/s08b/{corpus}_matched_per_speaker.csv")
    C = ps[ps.analysis_group == "C_full_split_whisper"]
    out = {}
    for g in ("dysarthric", "control"):
        sub = ft[ft.group == g]
        res = {"epochs": {int(e): float(x.uar.mean()) for e, x in sub.groupby("epoch")},
               "train_s_min": float(sub[sub.epoch == 5].train_s.min()), "train_s_max": float(sub[sub.epoch == 5].train_s.max())}
        e5 = sub[sub.epoch == 5].set_index("speaker").uar
        res["n"] = int(len(e5)); res["steps_min"] = int((sub[sub.epoch == 5].n_train / 8).apply(np.ceil).min() * 5); res["steps_max"] = int((sub[sub.epoch == 5].n_train / 8).apply(np.ceil).max() * 5)
        for ref, key in (("shared_L12_logreg", "vs_shared"), ("whisper_forced_mean", "vs_whisper")):
            r = C[C.system == ref].set_index("speaker").uar.reindex(e5.index)
            d = (100 * (e5 - r)).to_numpy()
            res[key] = {"mean_diff": float(d.mean()), "n_better": int((d > 0.05).sum()), "n_tied": int((abs(d) <= 0.05).sum()),
                        "sign_p": signflip(d), "per_speaker": dict(zip(e5.index, d.round(2).tolist()))}
        out[g] = res
    os.makedirs("results/s08l", exist_ok=True)
    json.dump(out, open(f"results/s08l/{corpus}_shared_finetune_stats.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
