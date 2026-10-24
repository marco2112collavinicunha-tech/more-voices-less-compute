"""Correctness checks that the cost comparisons rely on.

1. Truncating XLSR-53 after layer L gives *exactly* the representation the
   protocol uses (hidden_states[L] of the full model), so the cheaper model is
   the same method, not an approximation.
2. Whisper's encoder input is always 3000 mel frames (30 s), whatever the
   length of the phrase.

Run:  python check_equivalence.py [--pretrained]
"""
import argparse
import json
import sys

import torch

from greenafi.archs import build_xlsr, truncate_xlsr, whisper_features

ap = argparse.ArgumentParser()
ap.add_argument("--pretrained", action="store_true")
args = ap.parse_args()

torch.manual_seed(1)
x = torch.randn(1, int(16000 * 2.2))
full = build_xlsr(args.pretrained)
report = {"pretrained": args.pretrained, "truncation": {}}
with torch.inference_mode():
    hs = full(x, output_hidden_states=True).hidden_states
    for L in (12, 18):
        ref = hs[L]
        out = truncate_xlsr(full, L)(x).last_hidden_state
        diff = (out - ref).abs().max().item()
        pooled_cos = torch.nn.functional.cosine_similarity(out.mean(1), ref.mean(1)).item()
        report["truncation"][f"layer_{L}"] = {
            "max_abs_diff": diff,
            "pooled_cosine": pooled_cos,
            "identical": diff < 1e-5,
        }

frames = {}
for sec in (1.5, 2.2, 5.0, 29.0):
    frames[str(sec)] = list(whisper_features(torch.zeros(int(16000 * sec))).shape)
report["whisper_input_shape_by_audio_seconds"] = frames
report["whisper_always_3000_frames"] = all(v[-1] == 3000 for v in frames.values())

print(json.dumps(report, indent=2))
ok = all(v["identical"] for v in report["truncation"].values()) and report["whisper_always_3000_frames"]
import os
os.makedirs("results/s02", exist_ok=True)
json.dump(report, open("results/s02/equivalence.json", "w"), indent=2)
sys.exit(0 if ok else 1)
