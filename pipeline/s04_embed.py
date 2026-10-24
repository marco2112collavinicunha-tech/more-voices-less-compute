"""Stage 4 — embed every recording once, at all 24 layers (fp32 model, stored
as float16) and at layer 12 with the int8 model.

  python -m pipeline.s04_embed torgo [--random-weights]

Writes data/embeddings/<corpus>/{emb.npy, emb_int8_l12.npy, index.csv}.
emb.npy has shape [n_recordings, 24, 1024]: mean-pooled hidden_states[1..24].
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time

import numpy as np
import torch
from torch import nn

from greenafi.archs import build_xlsr, truncate_xlsr
from greenafi.data import load_audio, load_config, read_manifest


def xlsr_input(x: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(x)[None]
    return (t - t.mean()) / (t.std() + 1e-7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--random-weights", action="store_true", help="plumbing test only")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = load_config()
    counts = json.load(open(f"results/s01/{args.corpus}_counts.json"))
    keep = set(counts["speakers_included"])
    rows = [r for r in read_manifest(f"data/manifests/{args.corpus}.csv") if r["speaker"] in keep]

    model = build_xlsr(pretrained=not args.random_weights)
    torch.backends.quantized.engine = "qnnpack" if platform.machine() in ("arm64", "aarch64") else "x86"
    q12 = torch.ao.quantization.quantize_dynamic(truncate_xlsr(model, 12), {nn.Linear},
                                                 dtype=torch.qint8, inplace=True)
    out = f"data/embeddings/{args.corpus}"
    os.makedirs(out, exist_ok=True)
    emb = np.lib.format.open_memmap(f"{out}/emb.npy", mode="w+", dtype=np.float16, shape=(len(rows), 24, model.config.hidden_size))
    emb8 = np.zeros((len(rows), model.config.hidden_size), dtype=np.float32)
    a = cfg["audio"]
    t0 = time.time()
    with open(f"{out}/index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "utt_id", "speaker", "item", "duration_s"])
        for i, r in enumerate(rows):
            x = load_audio(r["path"], a["trim_db_below_peak"], a["peak_normalise"])
            with torch.inference_mode():
                hs = model(xlsr_input(x), output_hidden_states=True).hidden_states
                emb[i] = torch.stack([h.mean(1)[0] for h in hs[1:25]]).numpy().astype(np.float16)
                emb8[i] = q12(xlsr_input(x)).last_hidden_state.mean(1)[0].numpy()
            w.writerow([i, r["utt_id"], r["speaker"], r["item"], f"{len(x) / 16000:.3f}"])
            if i % 200 == 0:
                print(f"{i}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    emb.flush()
    np.save(f"{out}/emb_int8_l12.npy", emb8)
    json.dump({"n": len(rows), "random_weights": args.random_weights, "seconds": time.time() - t0},
              open(f"{out}/meta.json", "w"), indent=2)
    print("done", len(rows))


if __name__ == "__main__":
    main()
