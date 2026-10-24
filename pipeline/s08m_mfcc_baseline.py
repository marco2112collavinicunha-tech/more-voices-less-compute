"""s08m — classical baseline: log-mel / MFCC summary features + the same linear classifier.

Exploratory, added 22 Oct 2026 after a premise review ("does the pre-trained encoder matter at all,
or do the shared same-setting recordings carry the result?"). No pre-trained model: 40 log-mel bands
(25-ms window, 10-ms hop) and 13 MFCCs, summarised by mean and standard deviation over frames
(plus delta means), i.e. a 146-dimensional hand-crafted feature. Same draw-0 split, same
other-speaker training set and same logistic regression as `shared_L12_logreg`; also the
own-recording nearest neighbour. Paired against the XLSR-53 and Whisper-encoder shared classifiers.
  python -m pipeline.s08m_mfcc_baseline torgo
"""
from __future__ import annotations
import itertools, json, os, sys
import numpy as np, pandas as pd
from scipy.fft import dct
from scipy.signal import get_window

from greenafi.data import SR, by_speaker, load_audio, load_config, make_split, read_manifest, uar
from pipeline.s05_classifiers import fit_predict

N_MEL, N_MFCC, WIN, HOP, NFFT = 40, 13, 400, 160, 512


def mel_filterbank(sr=SR, nfft=NFFT, n_mel=N_MEL, fmin=20.0, fmax=None):
    fmax = fmax or sr / 2
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700.0)
    mel2hz = lambda m: 700 * (10 ** (m / 2595.0) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mel + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mel, nfft // 2 + 1))
    for m in range(1, n_mel + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(1, c - l)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(1, r - c)
    return fb


FB = mel_filterbank()
WINDOW = get_window("hann", WIN, fftbins=True)


def features(x: np.ndarray) -> np.ndarray:
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n = 1 + (len(x) - WIN) // HOP
    frames = np.stack([x[i * HOP:i * HOP + WIN] * WINDOW for i in range(n)])
    spec = np.abs(np.fft.rfft(frames, NFFT)) ** 2
    logmel = np.log(spec @ FB.T + 1e-8)                       # (n, 40)
    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :N_MFCC]  # (n, 13)
    d = np.diff(logmel, axis=0) if n > 1 else np.zeros((1, N_MEL))
    return np.concatenate([logmel.mean(0), logmel.std(0), mfcc.mean(0), mfcc.std(0), np.abs(d).mean(0)])


def signflip(d):
    d = np.asarray(d, float); s = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float((np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-9).mean())


def main():
    corpus = sys.argv[1]
    cfg = load_config(); p = cfg["personalisation"]
    counts = json.load(open(f"results/s01/{corpus}_counts.json"))
    k = counts["primary_k"]; keep = set(counts["speakers_included"])
    rows = [r for r in read_manifest(f"data/manifests/{corpus}.csv") if r["speaker"] in keep]
    item_of = {r["utt_id"]: r["item"] for r in rows}
    feats = {}
    for i, r in enumerate(rows):
        feats[r["utt_id"]] = features(load_audio(r["path"]))
        if i % 500 == 0:
            print(f"features {i}/{len(rows)}", flush=True)
    F = lambda ids: np.stack([feats[u] for u in ids])
    ps = pd.read_csv(f"results/s08b/{corpus}_matched_per_speaker.csv")
    C = ps[ps.analysis_group == "C_full_split_whisper"]
    shared = C[C.system == "shared_L12_logreg"].set_index("speaker").uar
    per = []
    for spk, rs in sorted(by_speaker(rows).items()):
        sp = make_split(rs, spk, k, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
        if sp is None:
            continue
        yte = [item_of[u] for u in sp.test]
        other = [r for r in rows if r["speaker"] != spk and r["item"] in set(sp.items)]
        oid, oy = [r["utt_id"] for r in other], [r["item"] for r in other]
        pr_shared = fit_predict(F(oid), oy, F(sp.test), "logreg")
        pr_own = fit_predict(F(sp.train), [item_of[u] for u in sp.train], F(sp.test), "1nn")
        per.append({"speaker": spk, "group": rs[0]["group"], "uar_mfcc_shared": uar(yte, list(pr_shared)),
                    "uar_mfcc_own_1nn": uar(yte, list(pr_own)), "uar_xlsr_shared": float(shared.get(spk, np.nan)),
                    "n_train_other": len(oid), "n_test": len(yte)})
        print(per[-1], flush=True)
    df = pd.DataFrame(per); out = {}
    for g, sub in df.groupby("group"):
        d = (100 * (sub.uar_xlsr_shared - sub.uar_mfcc_shared)).to_numpy()
        out[g] = {"n": int(len(sub)), "uar_mfcc_shared": float(sub.uar_mfcc_shared.mean()),
                  "uar_mfcc_own_1nn": float(sub.uar_mfcc_own_1nn.mean()), "uar_xlsr_shared": float(sub.uar_xlsr_shared.mean()),
                  "xlsr_minus_mfcc": {"mean_diff": float(d.mean()), "n_xlsr_better": int((d > 0.05).sum()),
                                      "sign_p": signflip(d), "per_speaker": dict(zip(sub.speaker, d.round(2).tolist()))}}
    os.makedirs("results/s08m", exist_ok=True)
    df.to_csv(f"results/s08m/{corpus}_mfcc_baseline_per_speaker.csv", index=False)
    json.dump(out, open(f"results/s08m/{corpus}_mfcc_baseline.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
