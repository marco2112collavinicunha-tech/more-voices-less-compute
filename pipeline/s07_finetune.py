"""Stage 7 — per-speaker fine-tuning baselines, with accuracy AND measured cost.

  python -m pipeline.s07_finetune torgo --device mps [--speakers F01,MC01] [--random-weights]

For each included speaker, the primary k (results/s01) and each draw in
finetune.n_draws: XLSR-53 + mean pooling + linear head, trained on the same
training split as stage 5, mode 'full' (all transformer layers) and 'top4'.
Accuracy (UAR) on the same test split is recorded at the pre-registered
epochs (finetune.eval_at_epochs) together with the cumulative training time
and energy. Writes results/s07/<corpus>_finetune.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch import nn

from greenafi.archs import FinetuneHead, apply_lora, build_xlsr, set_trainable, truncate_xlsr
from greenafi.data import by_speaker, load_audio, load_config, make_split, read_manifest, uar
from greenafi.energy import idle_power, make_meter


def batchify(xs: list[np.ndarray], device):
    n = max(len(x) for x in xs)
    wav = torch.zeros(len(xs), n)
    mask = torch.zeros(len(xs), n, dtype=torch.long)
    for i, x in enumerate(xs):
        t = torch.from_numpy(x)
        wav[i, : len(t)] = (t - t.mean()) / (t.std() + 1e-7)
        mask[i, : len(t)] = 1
    return wav.to(device), mask.to(device)


def forward(model: FinetuneHead, wav, mask):
    enc = model.encoder
    h = enc(wav, attention_mask=mask).last_hidden_state
    fmask = enc._get_feature_vector_attention_mask(h.shape[1], mask).unsqueeze(-1).to(h.dtype)
    pooled = (h * fmask).sum(1) / fmask.sum(1).clamp(min=1)
    return model.head(pooled)


def evaluate(model, audio, ids, items, device, bs=8):
    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(ids), bs):
            wav, mask = batchify([audio[u] for u in ids[i:i + bs]], device)
            preds += forward(model, wav, mask).argmax(-1).tolist()
    model.train()
    return [items[p] for p in preds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--speakers", default=None)
    ap.add_argument("--random-weights", action="store_true", help="plumbing test only")
    ap.add_argument("--modes", default=None, help="comma list, subset of finetune.modes (run order only)")
    ap.add_argument("--max-epochs", type=int, default=None, help="override for plumbing tests")
    args = ap.parse_args()

    cfg = load_config()
    fc, p = cfg["finetune"], cfg["personalisation"]
    counts = json.load(open(f"results/s01/{args.corpus}_counts.json"))
    k = counts["primary_k"]
    keep = set(counts["speakers_included"])
    if args.speakers:
        keep &= set(args.speakers.split(","))
    rows = [r for r in read_manifest(f"data/manifests/{args.corpus}.csv") if r["speaker"] in keep]
    eval_at = sorted(fc["eval_at_epochs"])
    max_ep = args.max_epochs or max(eval_at)
    if args.max_epochs:
        eval_at = sorted({min(e, max_ep) for e in eval_at})

    meter = make_meter()
    idle_w = idle_power(meter) if meter.name != "none" else None
    os.makedirs("results/s07", exist_ok=True)
    path = f"results/s07/{args.corpus}_finetune.csv"
    new = not os.path.exists(path)
    done = set()  # resumable: (speaker, draw, system) already evaluated at the last epoch
    if not new:
        for r in csv.DictReader(open(path)):
            if int(r["epoch"]) == max(eval_at):
                done.add((r["speaker"], int(r["draw"]), r["system"]))
    fail_path = f"results/s07/{args.corpus}_finetune_failures.csv"
    out = open(path, "a", newline="")
    w = csv.writer(out)
    if new:
        w.writerow(["corpus", "speaker", "group", "k", "draw", "system", "epoch", "uar", "train_s", "train_j",
                    "energy_meter", "trainable_params", "device", "random_weights"])

    for spk, rs in sorted(by_speaker(rows).items()):
        audio = {r["utt_id"]: load_audio(r["path"], cfg["audio"]["trim_db_below_peak"], cfg["audio"]["peak_normalise"])
                 for r in rs}
        item_of = {r["utt_id"]: r["item"] for r in rs}
        for d in range(fc["n_draws"]):
            sp = make_split(rs, spk, k, d, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
            if sp is None:
                continue
            items = sp.items
            label = {it: i for i, it in enumerate(items)}
            yte = [item_of[u] for u in sp.test]
            modes = [m for m in fc["modes"] if not args.modes or m in args.modes.split(",")]
            for mode in modes:
                if (spk, d, f"finetune_{mode}") in done:
                    continue
                try:
                    run_one(args, cfg, fc, spk, rs, d, k, mode, sp, items, label, item_of, yte, audio,
                            max_ep, eval_at, meter, idle_w, w, out)
                except RuntimeError as e:  # e.g. out of memory: log, free, go on with the next run
                    print("FAILED", spk, d, mode, repr(e)[:300], flush=True)
                    with open(fail_path, "a") as fl:
                        fl.write(f"{spk},{d},{mode},{repr(e)[:300]!r}\n")
                if args.device == "mps":
                    torch.mps.empty_cache()
    out.close()


def run_one(args, cfg, fc, spk, rs, d, k, mode, sp, items, label, item_of, yte, audio,
            max_ep, eval_at, meter, idle_w, w, out):
    torch.manual_seed(cfg["seed"] + d)
    base = build_xlsr(pretrained=not args.random_weights)
    if mode == "full_L12":
        base = truncate_xlsr(base, 12)
    elif mode == "lora":
        base = apply_lora(base, fc["lora"]["r"], fc["lora"]["alpha"])
    model = FinetuneHead(base, len(items))
    model.encoder.config.layerdrop = 0.0
    trainable = set_trainable(model, {"full": "full", "full_L12": "full", "top4": "top", "lora": "lora"}[mode], top_k=4)
    head = list(model.head.parameters())
    enc_params = [q for n, q in model.named_parameters() if q.requires_grad and not n.startswith("head")]
    lr_enc = fc["lora"]["lr"] if mode == "lora" else fc["lr_encoder"]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": lr_enc},
                             {"params": head, "lr": fc["lr_head"]}], weight_decay=fc["weight_decay"])
    model.to(args.device).train()
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(cfg["seed"] + d)
    train_s, train_j = 0.0, 0.0
    for ep in range(1, max_ep + 1):
        order = rng.permutation(len(sp.train))
        meter.start()
        t0 = time.perf_counter()
        for b in range(0, len(order), fc["batch_size"]):
            ids = [sp.train[j] for j in order[b:b + fc["batch_size"]]]
            wav, mask = batchify([audio[u] for u in ids], args.device)
            y = torch.tensor([label[item_of[u]] for u in ids], device=args.device)
            opt.zero_grad(set_to_none=True)
            lossf(forward(model, wav, mask), y).backward()
            opt.step()
        if args.device == "mps":
            torch.mps.synchronize()
        r = meter.stop()
        ep_s = time.perf_counter() - t0
        train_s += ep_s
        mem = f" mps={torch.mps.driver_allocated_memory() / 2**30:.1f}GiB" if args.device == "mps" else ""
        print(f"  {spk} {mode} epoch {ep}/{max_ep} {ep_s:.1f}s{mem}", flush=True)  # heartbeat (no accuracy)
        if r["joules"] is not None:
            train_j += max(r["joules"] - (idle_w or 0) * r["seconds"], 0)
        if ep in eval_at:
            u = uar(yte, evaluate(model, audio, sp.test, items, args.device))
            w.writerow([args.corpus, spk, rs[0]["group"], k, d, f"finetune_{mode}", ep, f"{u:.4f}",
                        f"{train_s:.2f}", f"{train_j:.1f}" if meter.name != "none" else "",
                        meter.name, trainable, args.device, args.random_weights])
            out.flush()
            print(spk, mode, ep, f"uar={u:.3f}", f"{train_s:.0f}s", flush=True)
    del model, opt


if __name__ == "__main__":
    main()
