"""Compute/energy benchmark for this study.

Stages (run any subset with --stages):
  xlsr     cost of one recognition with XLSR-53 run to layer 24 / 18 / 12 (+ int8)
  heads    cost of the per-user classifiers (1-NN, logistic regression, the
           protocol's 4-configuration selection)
  whisper  cost of one recognition with Whisper-medium, free transcription
           (greedy) and forced choice among the 12 phrases
  finetune cost of one epoch of per-speaker fine-tuning (full / top-4 layers)

Results go to results/compute_<stage>.json. Use --pretrained on a machine with
HuggingFace access; without it, models have the exact architecture and random
weights (same cost, meaningless outputs).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time

import numpy as np
import torch
from torch import nn
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput

from greenafi.archs import (
    FinetuneHead, apply_lora, build_whisper, build_xlsr, n_params, set_trainable,
    truncate_xlsr, whisper_features,
)
from greenafi.energy import idle_power, make_meter
from greenafi.measure import count_flops, measure, setup_threads

SR = 16000
N_PHRASES = 12
N_TRAIN = 96            # 12 phrases x 4 repetitions x 2 sessions (protocol, Quadro 2)
PROMPT_TOKENS = 4       # <|startoftranscript|><|pt|><|transcribe|><|notimestamps|>


def env_info(args, meter):
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "pretrained": args.pretrained,
        "energy_meter": meter.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


OUT = "results/s03"


def save(stage, payload):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/compute_{stage}.json"
    json.dump(payload, open(path, "w"), indent=2)
    print(f"[saved] {path}")


def audio(seconds):
    g = torch.Generator().manual_seed(int(seconds * 1000))
    return (0.1 * torch.randn(1, int(SR * seconds), generator=g))


def xlsr_pre(x):  # zero-mean unit-variance normalisation used by the XLSR feature extractor
    return (x - x.mean()) / (x.std() + 1e-7)


# ----------------------------------------------------------------- stages
def stage_xlsr(args, meter, idle_w):
    full = build_xlsr(args.pretrained)
    variants = {
        "xlsr_L24": full,
        "xlsr_L18": truncate_xlsr(full, 18),
        "xlsr_L12": truncate_xlsr(full, 12),
    }
    try:
        torch.backends.quantized.engine = "qnnpack" if platform.machine() == "arm64" else "x86"
        variants["xlsr_L12_int8"] = torch.ao.quantization.quantize_dynamic(
            truncate_xlsr(full, 12), {nn.Linear}, dtype=torch.qint8, inplace=True)
    except Exception as e:  # pragma: no cover
        print("int8 quantisation unavailable:", e)
    rows = []
    for name, model in variants.items():
        for sec in args.durations:
            x = audio(sec)
            def fn(model=model, x=x):
                with torch.inference_mode():
                    return model(xlsr_pre(x)).last_hidden_state.mean(1)
            flops = count_flops(fn) if not name.endswith("int8") else None
            m = measure(fn, meter, idle_w, min_seconds=args.min_seconds)
            row = {"system": name, "audio_s": sec, "flops": flops, "params": n_params(model), **m}
            print(json.dumps({k: row[k] for k in ("system", "audio_s", "flops", "median_s", "joules_per_call")}))
            rows.append(row)
    save("xlsr", {"env": env_info(args, meter), "rows": rows})


def stage_heads(args, meter, idle_w):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(N_PHRASES, 1024))
    y = np.repeat(np.arange(N_PHRASES), N_TRAIN // N_PHRASES)
    X = centers[y] + 2.0 * rng.normal(size=(N_TRAIN, 1024))
    q = centers[[3]] + 2.0 * rng.normal(size=(1, 1024))
    sess = np.tile(np.repeat([1, 2], 4), N_PHRASES)  # session of each training clip

    def fit_1nn():
        s = StandardScaler().fit(X)
        return s, KNeighborsClassifier(n_neighbors=1).fit(s.transform(X), y)

    def fit_lr():
        s = StandardScaler().fit(X)
        return s, LogisticRegression().fit(s.transform(X), y)

    def protocol_selection():
        # 2 layers x 2 classifiers x 2 rounds (train on one session, predict the other) + final fit
        for _layer in (12, 18):
            for make in (lambda: KNeighborsClassifier(n_neighbors=1), lambda: LogisticRegression()):
                for tr, te in ((1, 2), (2, 1)):
                    s = StandardScaler().fit(X[sess == tr])
                    make().fit(s.transform(X[sess == tr]), y[sess == tr]).predict(s.transform(X[sess == te]))
        return fit_lr()

    s1, knn = fit_1nn()
    s2, lr = fit_lr()
    work = {
        "fit_1nn": fit_1nn,
        "fit_logreg": fit_lr,
        "fit_protocol_selection": protocol_selection,
        "predict_1nn": lambda: knn.predict(s1.transform(q)),
        "predict_logreg": lambda: lr.predict(s2.transform(q)),
    }
    analytic_flops = {  # dominant terms
        "predict_1nn": 3 * N_TRAIN * 1024,          # squared distances to 96 stored vectors
        "predict_logreg": 2 * N_PHRASES * 1024,     # one 1024x12 matrix-vector product
    }
    storage_bytes = {
        "1nn": N_TRAIN * 1024 * 4 + 2 * 1024 * 4,          # stored vectors + scaler, float32
        "logreg": (N_PHRASES * 1025 + 2 * 1024) * 4,
    }
    rows = []
    for name, fn in work.items():
        m = measure(fn, meter, idle_w, min_seconds=args.min_seconds / 2)
        rows.append({"system": name, "flops": analytic_flops.get(name), **m})
        print(name, f"{m['median_s']*1e3:.3f} ms")
    save("heads", {"env": env_info(args, meter), "rows": rows, "storage_bytes_per_user": storage_bytes})


def stage_whisper(args, meter, idle_w):
    model = build_whisper(args.pretrained)
    dec_tokens = PROMPT_TOKENS + args.text_tokens
    rows = []
    for sec in args.whisper_durations:
        x = audio(sec)[0]

        def encode():
            feats = whisper_features(x)
            return model.model.encoder(feats).last_hidden_state

        def free_transcription():
            with torch.inference_mode():
                enc = BaseModelOutput(last_hidden_state=encode())
                ids = torch.full((1, PROMPT_TOKENS), 50258)
                out = model(encoder_outputs=enc, decoder_input_ids=ids, use_cache=True)
                for _ in range(args.text_tokens - 1):  # greedy; WhisperX default is beam 5 (costlier)
                    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                    out = model(encoder_outputs=enc, decoder_input_ids=nxt,
                                past_key_values=out.past_key_values, use_cache=True)
                return out.logits

        def forced_choice():
            # Efficient implementation, favourable to Whisper: the cross-attention
            # keys/values of the audio are computed once and shared by the 12
            # candidate phrases (verified to give identical logits to the naive
            # batched pass). The silence-conditioned score of the protocol uses
            # an encoding of silence that is computed once and cached, so it is
            # not charged per recognition.
            with torch.inference_mode():
                enc = BaseModelOutput(last_hidden_state=encode())
                ids = torch.full((1, dec_tokens), 50258)
                prime = model(encoder_outputs=enc, decoder_input_ids=ids[:, :1], use_cache=True)
                cross = prime.past_key_values.cross_attention_cache
                scores = []
                for _ in range(N_PHRASES):
                    cache = EncoderDecoderCache(DynamicCache(), cross)
                    logp = model(encoder_outputs=enc, decoder_input_ids=ids,
                                 past_key_values=cache, use_cache=True).logits.log_softmax(-1)
                    scores.append(logp.gather(-1, ids[..., None]).sum())
                return torch.stack(scores)

        def encoder_only():
            with torch.inference_mode():
                return encode()

        for name, fn in (("whisper_encoder_only", encoder_only),
                         ("whisper_free", free_transcription),
                         ("whisper_forced12", forced_choice)):
            flops = count_flops(fn)
            m = measure(fn, meter, idle_w, min_seconds=args.min_seconds, min_reps=2)
            row = {"system": name, "audio_s": sec, "flops": flops, "params": n_params(model),
                   "decoder_tokens": dec_tokens, **m}
            print(json.dumps({k: row[k] for k in ("system", "audio_s", "flops", "median_s", "joules_per_call")}))
            rows.append(row)
    save("whisper", {"env": env_info(args, meter), "rows": rows})


def stage_finetune(args, meter, idle_w):
    rows = []
    x = audio(args.train_clip_s)
    y = torch.tensor([3])
    for mode in args.ft_modes:
        gc.collect()
        base = build_xlsr(args.pretrained)
        if mode == "full_L12":
            base = truncate_xlsr(base, 12)
        elif mode == "lora":
            base = apply_lora(base)
        model = FinetuneHead(base, N_PHRASES)
        model.train()
        trainable = set_trainable(model, {"top": "top", "full": "full", "full_L12": "full", "lora": "lora"}[mode], top_k=4)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, foreach=False)
        lossf = nn.CrossEntropyLoss()

        def step():
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xlsr_pre(x)), y)
            loss.backward()
            opt.step()

        def fwd_bwd():
            opt.zero_grad(set_to_none=True)
            lossf(model(xlsr_pre(x)), y).backward()

        flops = count_flops(fwd_bwd)
        m = measure(step, meter, idle_w, warmup=2, min_seconds=args.min_seconds, min_reps=3, max_reps=20)
        row = {"system": f"finetune_{mode}", "clip_s": args.train_clip_s, "trainable_params": trainable,
               "flops_per_clip": flops, "flops_per_epoch": flops * N_TRAIN,
               "s_per_clip": m["median_s"], "s_per_epoch": m["median_s"] * N_TRAIN,
               "joules_per_epoch": None if m["joules_per_call"] is None else m["joules_per_call"] * N_TRAIN,
               # what must be stored per user: the trained part only (top layers, LoRA matrices + head),
               # or the whole (possibly truncated) model when every layer was trained
               "per_user_model_bytes_fp32": trainable * 4 if mode in ("top", "lora") else n_params(model) * 4,
               **m}
        print(json.dumps({k: row[k] for k in ("system", "trainable_params", "flops_per_epoch", "s_per_epoch")}))
        rows.append(row)
        del model, opt
    save("finetune", {"env": env_info(args, meter), "rows": rows, "n_train_clips": N_TRAIN})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="xlsr,heads,whisper,finetune")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--durations", type=float, nargs="+", default=[1.5, 2.2, 3.0, 5.0])
    ap.add_argument("--whisper-durations", type=float, nargs="+", default=[2.2])
    ap.add_argument("--text-tokens", type=int, default=10, help="tokens per phrase incl. end-of-text")
    ap.add_argument("--train-clip-s", type=float, default=2.2)
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--out", default="results/s03")
    ap.add_argument("--ft-modes", nargs="+", default=["top", "full", "lora", "full_L12"])
    args = ap.parse_args()
    global OUT
    OUT = args.out

    setup_threads(args.threads)
    meter = make_meter()
    idle_w = idle_power(meter) if meter.name != "none" else None
    print(f"meter={meter.name} idle_w={idle_w} threads={torch.get_num_threads()}")
    for st in args.stages.split(","):
        {"xlsr": stage_xlsr, "heads": stage_heads, "whisper": stage_whisper,
         "finetune": stage_finetune}[st](args, meter, idle_w)


if __name__ == "__main__":
    main()
