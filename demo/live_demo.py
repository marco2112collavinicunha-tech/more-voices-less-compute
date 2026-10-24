"""Working demonstration of the low-computation recogniser (XLSR-53 truncated at layer 12 + shared classifier).

Three modes:
  torgo  replays the report's analysis-C test recordings of one TORGO speaker with dysarthria
         (classifier trained only on OTHER speakers' recordings of the same words) and prints,
         for each recording, the true word, the recognised word, latency and computation;
  mic    live: say one of ten English command words (yes no up down left right on off stop go);
         the classifier was trained on other people's recordings (Speech Commands);
  file   recognise a given .wav file against the command words.

For every recognition it prints the operations actually needed (counted, same method as the report),
the estimated energy on a low-cost ARM board (0.285 J/GFLOP, Gondi 2022), and the operations
Whisper-medium would need for the same decision. A log is written to results/demo/.

  python -m demo.live_demo torgo --speaker M04
  python -m demo.live_demo mic
  python -m demo.live_demo file path/to/word.wav
Add --int8 for 8-bit weights. Needs the XLSR-53 weights in the local Hugging Face cache.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time

import numpy as np
import torch
from torch import nn

from greenafi.archs import build_xlsr, truncate_xlsr
from greenafi.data import load_audio, load_config

J_PER_GFLOP = 0.285          # Raspberry Pi 4, Gondi (2022); see config/energy_anchors.json


def load_classifier(path):
    z = np.load(path, allow_pickle=False)
    return z


def predict(z, e):
    x = (e - z["mean"]) / z["scale"]
    logits = z["coef"] @ x + z["intercept"]
    p = np.exp(logits - logits.max()); p /= p.sum()
    order = np.argsort(-p)
    return [(str(z["classes"][i]), float(p[i])) for i in order[:3]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["torgo", "mic", "file"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--speaker", default="M04")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5, help="below this confidence the system asks to repeat")
    ap.add_argument("--seconds", type=float, default=1.5)
    args = ap.parse_args()
    cfg = load_config(); a = cfg["audio"]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    t0 = time.perf_counter()
    model = truncate_xlsr(build_xlsr(pretrained=True), 12)
    if args.int8:
        torch.backends.quantized.engine = "qnnpack" if platform.machine() in ("arm64", "aarch64") else "x86"
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: XLSR-53 truncated at layer 12{' (8-bit)' if args.int8 else ''}, "
          f"{n_params / 1e6:.0f} M parameters, loaded in {time.perf_counter() - t0:.1f} s")
    C = json.load(open("results/s03d/torgo_costs.json"))
    fit = C["xlsr_linear_fit"]["12"]; wh = C["whisper"]

    def gflop_l12(sec):
        return (fit["flops_per_second"] * sec + fit["intercept_flops"]) / 1e9

    def gflop_whisper(K):
        return (wh["encoder_flops"] + wh["oneoff_flops"] + K * wh["per_candidate_flops"]) / 1e9

    def embed(x):
        t = torch.from_numpy(x)[None]; t = (t - t.mean()) / (t.std() + 1e-7)
        with torch.inference_mode():
            t1 = time.perf_counter()
            h = model(t).last_hidden_state.mean(1)[0].float().numpy()
            return h, time.perf_counter() - t1

    os.makedirs("results/demo", exist_ok=True)
    log = open(f"results/demo/demo_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl", "w")

    def recognise(x, z, truth=None):
        e, lat = embed(x)
        top = predict(z, e)
        sec = len(x) / 16000; K = len(z["classes"])
        g, gw = gflop_l12(sec), gflop_whisper(K)
        word, conf = top[0]
        shown = word if conf >= args.threshold else f"(not sure: '{word}'? please repeat)"
        mark = "" if truth is None else ("  OK" if word == truth else f"  wrong (said '{truth}')")
        print(f"  heard: {shown:<28} conf {conf:4.2f} | {lat * 1000:5.0f} ms | {g:5.1f} GFLOP ≈ {g * J_PER_GFLOP:4.1f} J"
              f" (Whisper-medium: {gw:,.0f} GFLOP ≈ {gw * J_PER_GFLOP:,.0f} J; J estimated for a Raspberry Pi 4){mark}")
        rec = {"truth": truth, "pred": word, "conf": conf, "top3": top, "seconds": sec, "latency_s": lat,
               "gflop": g, "gflop_whisper_medium": gw, "j_est": g * J_PER_GFLOP, "int8": args.int8}
        log.write(json.dumps(rec) + "\n"); log.flush()
        return rec

    if args.mode == "torgo":
        z = load_classifier(f"demo/classifier_torgo_{args.speaker}.npz")
        print(f"TORGO speaker {args.speaker} (dysarthria). Classifier trained only on other speakers' recordings "
              f"of the same {len(z['classes'])} words. Replaying {len(z['test_paths'])} test recordings.\n")
        recs = []
        for pth, it in zip(z["test_paths"], z["test_items"]):
            print(f"[{os.path.basename(os.path.dirname(os.path.dirname(pth)))}/{os.path.basename(pth)}]", end="")
            recs.append(recognise(load_audio(str(pth), a["trim_db_below_peak"], a["peak_normalise"]), z, str(it)))
        acc = np.mean([r["pred"] == r["truth"] for r in recs])
        items = sorted({r["truth"] for r in recs})
        u = np.mean([np.mean([r["pred"] == r["truth"] for r in recs if r["truth"] == i]) for i in items])
        print(f"\nAccuracy {acc * 100:.1f}% (UAR {u * 100:.1f}%), median latency {np.median([r['latency_s'] for r in recs]) * 1000:.0f} ms, "
              f"estimated energy {sum(r['j_est'] for r in recs):.0f} J in total vs "
              f"{sum(r['gflop_whisper_medium'] for r in recs) * J_PER_GFLOP:,.0f} J for Whisper-medium.")
    elif args.mode == "file":
        z = load_classifier("demo/classifier_commands.npz")
        recognise(load_audio(args.path, a["trim_db_below_peak"], a["peak_normalise"]), z)
    else:
        import sounddevice as sd
        import soundfile as sf
        z = load_classifier("demo/classifier_commands.npz")
        print("Words:", " ".join(map(str, z["classes"])), "\nPress Enter, then say one word (Ctrl-C to stop).")
        while True:
            input()
            print("  recording…", end="", flush=True)
            x = sd.rec(int(args.seconds * 16000), samplerate=16000, channels=1, dtype="float32"); sd.wait()
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                sf.write(f.name, x[:, 0], 16000)
                recognise(load_audio(f.name, a["trim_db_below_peak"], a["peak_normalise"]), z)


if __name__ == "__main__":
    main()
