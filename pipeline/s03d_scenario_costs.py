"""Stage 3d — costs in the scenario the accuracy was measured in (TORGO, primary k).

  python -m pipeline.s03d_scenario_costs torgo

The cost benchmark (s03) used 2.2-second recordings and a deployment scenario's 96
enrolment recordings. Accuracy on TORGO was measured with each speaker's own
recording lengths, k = 1 recording per word and that speaker's number of words.
This stage re-expresses the measured FLOP counts for that scenario:

* XLSR-53 to layer L: FLOPs as a linear function of recording length, fitted to
  the four measured lengths (1.5, 2.2, 3, 5 s); residuals reported.
* Whisper-medium: always 30 s of input, so the encoder cost is fixed; free
  transcription as measured (14 decoder tokens); forced choice = encoder +
  K x (per-candidate decoder cost measured with 12 candidates), with K = the
  speaker's number of words (the closed set the decision uses).
* Per-user personalisation: frozen = embed the user's k recordings per word;
  retraining = n_train x epochs x (training FLOPs per 2.2-s clip, scaled
  linearly with length); shared classifier = nothing per user (it uses other
  people's recordings, embedded once); shared + own = the user's recordings
  embedded, plus a refit of the shared classifier (time only, not FLOPs).
Energy uses the same J/GFLOP anchor as s03c. Writes results/s03d/<corpus>_costs.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from greenafi.data import by_speaker, load_config, make_split, read_manifest

COST = "results/vm_random_weights"


def main():
    corpus = sys.argv[1]
    cfg = load_config()
    p = cfg["personalisation"]
    x = json.load(open(f"{COST}/compute_xlsr.json"))["rows"]
    wsp = {r["system"]: r for r in json.load(open(f"{COST}/compute_whisper.json"))["rows"]}
    ft = {r["system"]: r for r in json.load(open(f"{COST}/compute_finetune.json"))["rows"]}
    jpg = json.load(open("results/s03c/energy.json")).get("j_per_gflop") or 0.285
    fits = {}
    for L in (12, 18, 24):
        rr = [r for r in x if r["system"] == f"xlsr_L{L}"]
        s = np.array([r["audio_s"] for r in rr]); f = np.array([r["flops"] for r in rr], float)
        b, a = np.polyfit(s, f, 1)
        fits[L] = {"intercept_flops": float(a), "flops_per_second": float(b),
                   "max_rel_residual": float(np.max(np.abs((a + b * s) - f) / f))}
    flop = lambda L, sec: fits[L]["intercept_flops"] + fits[L]["flops_per_second"] * sec
    enc = wsp["whisper_encoder_only"]["flops"]
    # Forced choice, from Whisper-medium's architecture (d = 1024, 24 decoder layers, 51,865-token
    # vocabulary, 1,500 encoder frames), counted the same way as the benchmark (weight products only):
    #   one-off: cross-attention keys and values of the 1,500 frames, 2 x 24 x 2d^2 x 1500, plus one token;
    #   per decoder token: self-attention 4d^2 + cross-attention queries/outputs 2d^2 + MLP 8d^2 per layer,
    #   plus the output projection d x 51,865.
    d, nl, V, T = 1024, 24, 51865, 1500
    per_token = 2 * (nl * 14 * d * d + d * V)
    oneoff = 2 * nl * 2 * d * d * T + per_token
    check12 = enc + oneoff + 12 * 14 * per_token  # the benchmark scored 12 candidates of 14 tokens
    assert abs(check12 - wsp["whisper_forced12"]["flops"]) / wsp["whisper_forced12"]["flops"] < 0.005, check12
    TOK = 5  # tokens fed per candidate: 4 prefix tokens + at least 1 for the word (lower bound, favours Whisper)
    per_cand = TOK * per_token
    free = wsp["whisper_free"]["flops"]
    clip = 2.2
    ft_clip = {"full": ft["finetune_full"]["flops_per_clip"], "top4": ft["finetune_top"]["flops_per_clip"],
               "lora": ft["finetune_lora"]["flops_per_clip"]}
    l12 = [r for r in ft.values() if "L12" in r["system"] or "l12" in r["system"].lower()]
    if l12:
        ft_clip["full_L12"] = l12[0]["flops_per_clip"]
    secs = json.load(open(f"results/s01/{corpus}_trimmed_seconds.json"))
    counts = json.load(open(f"results/s01/{corpus}_counts.json"))
    k = counts["primary_k"]
    rows = [r for r in read_manifest(f"data/manifests/{corpus}.csv") if r["speaker"] in set(counts["speakers_included"])]
    out = {"corpus": corpus, "k": k, "j_per_gflop": jpg, "xlsr_linear_fit": fits,
           "whisper": {"encoder_flops": enc, "free_flops": free, "per_candidate_flops": per_cand, "oneoff_flops": oneoff,
                       "per_token_flops": per_token, "tokens_per_candidate": TOK,
                       "check_forced12_model": check12, "check_forced12_measured": wsp["whisper_forced12"]["flops"]},
           "finetune_flops_per_2p2s_clip": ft_clip, "speakers": {}}
    for spk, rs in sorted(by_speaker(rows).items()):
        sp = make_split(rs, spk, k, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
        if sp is None:
            continue
        dtr = float(np.mean([secs[u] for u in sp.train])); dte = float(np.mean([secs[u] for u in sp.test]))
        n_tr, K = len(sp.train), len(sp.items)
        others = [r["utt_id"] for r in rows if r["speaker"] != spk and r["item"] in set(sp.items)]
        rec = {"L12": flop(12, dte), "L18": flop(18, dte), "L24": flop(24, dte), "whisper_free": free,
               "whisper_forced": enc + oneoff + K * per_cand}
        pers = {"frozen_L12": n_tr * flop(12, dtr), "frozen_L24": n_tr * flop(24, dtr), "shared_L12": 0.0, "xpl_shared_own_L12": n_tr * flop(12, dtr)}
        for m, fc in ft_clip.items():
            for ep in (10, 30):
                pers[f"finetune_{m}_ep{ep}"] = n_tr * ep * fc * dtr / clip
        out["speakers"][spk] = {"group": rs[0]["group"], "n_items": K, "n_train": n_tr, "n_test": len(sp.test),
                                "mean_s_train": dtr, "mean_s_test": dte, "n_other_recordings": len(others),
                                "shared_oneoff_embed_flops": len(others) * flop(12, float(np.mean([secs[u] for u in others]))),
                                "per_recognition_flops": rec, "per_user_flops": pers}
    # group means
    agg = {}
    for g in sorted({v["group"] for v in out["speakers"].values()}):
        S = [v for v in out["speakers"].values() if v["group"] == g]
        agg[g] = {"n_speakers": len(S),
                  "mean_s_test": float(np.mean([v["mean_s_test"] for v in S])),
                  "mean_n_items": float(np.mean([v["n_items"] for v in S])),
                  "per_recognition_flops": {s: float(np.mean([v["per_recognition_flops"][s] for v in S])) for s in S[0]["per_recognition_flops"]},
                  "per_user_flops": {s: float(np.mean([v["per_user_flops"][s] for v in S])) for s in S[0]["per_user_flops"]}}
    out["group_means"] = agg
    # time to fit the shared classifier once (logistic regression), measured here
    try:
        from pipeline.s05_classifiers import fit_predict
        E = np.load(f"data/embeddings/{corpus}/emb.npy", mmap_mode="r")
        import csv
        idx = {r["utt_id"]: int(r["row"]) for r in csv.DictReader(open(f"data/embeddings/{corpus}/index.csv"))}
        spk = max(out["speakers"], key=lambda s: out["speakers"][s]["n_other_recordings"])
        rs = [r for r in rows if r["speaker"] != spk]
        sp_items = set(make_split(by_speaker(rows)[spk], spk, k, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"]).items)
        oth = [r for r in rs if r["item"] in sp_items]
        X = np.asarray(E[[idx[r["utt_id"]] for r in oth], 11], np.float32)
        t0 = time.perf_counter(); fit_predict(X, [r["item"] for r in oth], X[:1], "logreg")
        out["shared_fit_seconds_largest"] = {"speaker": spk, "n_recordings": len(oth), "n_items": len(sp_items),
                                            "seconds": time.perf_counter() - t0, "machine": "cloud VM, 2 CPU threads"}
    except Exception as e:  # noqa
        out["shared_fit_seconds_largest"] = {"error": repr(e)[:200]}
    os.makedirs("results/s03d", exist_ok=True)
    json.dump(out, open(f"results/s03d/{corpus}_costs.json", "w"), indent=2)
    for g, a in agg.items():
        print(g, {k_: f"{v / 1e9:.1f} GFLOP" for k_, v in a["per_recognition_flops"].items()})
        print(g, {k_: f"{v / 1e12:.2f} TFLOP" for k_, v in a["per_user_flops"].items()})
    print(out["shared_fit_seconds_largest"]); print(fits)


if __name__ == "__main__":
    main()
