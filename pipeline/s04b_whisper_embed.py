"""s04b — Whisper encoder embeddings, so the SAME shared classifier can be put on Whisper's encoder.

Separates "examples vs computation" from "XLSR-53 vs Whisper": for each recording, the Whisper
encoder (as released, 30-s padded input) is run once and its last hidden state is mean-pooled over
the frames that cover the actual audio (20 ms per frame). Writes data/embeddings/<corpus>_whisper_<tag>/
emb.npy (float16, n x d) and index.csv in the same row order as the XLSR embeddings' index.

Added 20 Oct 2026, exploratory (requested by the round-6 review).
  python -m pipeline.s04b_whisper_embed torgo --checkpoint openai/whisper-tiny --tag tiny
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

from greenafi.data import load_audio, load_config, read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = load_config()
    idx = list(csv.DictReader(open(f"data/embeddings/{args.corpus}/index.csv")))
    man = {r["utt_id"]: r for r in read_manifest(f"data/manifests/{args.corpus}.csv")}
    fe = WhisperFeatureExtractor.from_pretrained(args.checkpoint)
    enc = WhisperModel.from_pretrained(args.checkpoint).encoder.eval().to(args.device)
    out_dir = f"data/embeddings/{args.corpus}_whisper_{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    E = np.zeros((len(idx), enc.config.d_model), dtype=np.float16)
    done_path = f"{out_dir}/partial.npy"
    start = 0
    if os.path.exists(done_path):
        E[:] = np.load(done_path); start = int(open(f"{out_dir}/partial_n.txt").read())
    for i in range(start, len(idx)):
        r = man.get(idx[i]["utt_id"])
        if r is None:
            continue
        x = load_audio(r["path"], cfg["audio"]["trim_db_below_peak"], cfg["audio"]["peak_normalise"])
        feats = fe(x, sampling_rate=16000, return_tensors="pt").input_features.to(args.device)
        with torch.inference_mode():
            h = enc(feats).last_hidden_state[0]
        n = max(1, min(h.shape[0], int(np.ceil(len(x) / 16000 / 0.02))))
        E[i] = h[:n].mean(0).float().cpu().numpy().astype(np.float16)
        if i % 200 == 0:
            np.save(done_path, E); open(f"{out_dir}/partial_n.txt", "w").write(str(i + 1))
            print(args.tag, i, "/", len(idx), flush=True)
    np.save(f"{out_dir}/emb.npy", E)
    with open(f"{out_dir}/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(idx[0].keys())); w.writeheader(); w.writerows(idx)
    print("done", out_dir, E.shape)


if __name__ == "__main__":
    main()
