"""s07b — retraining ONCE on other people's recordings (the strongest untested alternative).

For each included speaker (leave-one-speaker-out): XLSR-53 cut at layer 12, all layers trained, with
mean pooling and a linear head over the speaker's own closed set, trained on the OTHER included
speakers' recordings of those words — exactly the data of the shared classifier — and evaluated on the
speaker's draw-0 test split (the split of analyses A and C). No recording of the target speaker is
used. Same optimiser settings as s07 (AdamW, 3e-5 encoder, 1e-3 head, batch 8, weight decay 0.01).
Evaluated after 1, 3 and 5 epochs. Per-recording predictions are kept so that the result can be
paired with every other system and split by session (s08f).

Added 19 Oct 2026, exploratory (requested by the reviews: "spend computation once, not per user").

  python -m pipeline.s07b_shared_finetune torgo --device cpu [--speakers M02,F03]
Writes results/s07b/<corpus>_shared_finetune.csv and ..._preds.jsonl (resumable).
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

from greenafi.archs import FinetuneHead, build_xlsr, set_trainable, truncate_xlsr
from greenafi.data import by_speaker, load_audio, load_config, make_split, read_manifest, uar
from pipeline.s07_finetune import batchify, evaluate, forward

EVAL_AT = (1, 3, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--speakers", default=None)
    ap.add_argument("--max-epochs", type=int, default=max(EVAL_AT))
    args = ap.parse_args()
    cfg = load_config()
    fc, p = cfg["finetune"], cfg["personalisation"]
    counts = json.load(open(f"results/s01/{args.corpus}_counts.json"))
    k = counts["primary_k"]
    keep = set(counts["speakers_included"])
    rows = [r for r in read_manifest(f"data/manifests/{args.corpus}.csv") if r["speaker"] in keep]
    targets = sorted(set(args.speakers.split(",")) & keep) if args.speakers else sorted(keep)
    os.makedirs("results/s07b", exist_ok=True)
    path = f"results/s07b/{args.corpus}_shared_finetune.csv"
    ppath = f"results/s07b/{args.corpus}_shared_finetune_preds.jsonl"
    done = set()
    if os.path.exists(path):
        done = {r["speaker"] for r in csv.DictReader(open(path)) if int(r["epoch"]) == args.max_epochs}
    new = not os.path.exists(path)
    out = open(path, "a", newline=""); w = csv.writer(out)
    if new:
        w.writerow(["corpus", "speaker", "group", "epoch", "uar", "n_train", "n_test", "train_s", "device"])
    pout = open(ppath, "a")
    audio_cache: dict[str, np.ndarray] = {}

    def aud(r):
        if r["utt_id"] not in audio_cache:
            audio_cache[r["utt_id"]] = load_audio(r["path"], cfg["audio"]["trim_db_below_peak"], cfg["audio"]["peak_normalise"])
        return audio_cache[r["utt_id"]]

    spk_rows = by_speaker(rows)
    for spk in targets:
        if spk in done:
            continue
        rs = spk_rows[spk]
        sp = make_split(rs, spk, k, 0, cfg["seed"], p["min_test_per_item"], p["min_items_per_speaker"])
        if sp is None:
            continue
        items = list(sp.items); label = {it: i for i, it in enumerate(items)}
        other = [r for r in rows if r["speaker"] != spk and r["item"] in label]
        test = [r for r in rs if r["utt_id"] in set(sp.test)]
        audio = {r["utt_id"]: aud(r) for r in other + test}
        item_of = {r["utt_id"]: r["item"] for r in other + test}
        train_ids = [r["utt_id"] for r in other]
        test_ids = [r["utt_id"] for r in test]
        yte = [item_of[u] for u in test_ids]
        torch.manual_seed(cfg["seed"])
        model = FinetuneHead(truncate_xlsr(build_xlsr(pretrained=True), 12), len(items))
        model.encoder.config.layerdrop = 0.0
        set_trainable(model, "full", top_k=4)
        head = list(model.head.parameters())
        enc_params = [q for n, q in model.named_parameters() if q.requires_grad and not n.startswith("head")]
        opt = torch.optim.AdamW([{"params": enc_params, "lr": fc["lr_encoder"]}, {"params": head, "lr": fc["lr_head"]}],
                                weight_decay=fc["weight_decay"])
        model.to(args.device).train()
        lossf = nn.CrossEntropyLoss(); rng = np.random.default_rng(cfg["seed"]); train_s = 0.0
        print(f"### {spk}: {len(train_ids)} training recordings from other speakers, {len(items)} words, {len(test_ids)} test", flush=True)
        for ep in range(1, args.max_epochs + 1):
            order = rng.permutation(len(train_ids)); t0 = time.perf_counter()
            for b in range(0, len(order), fc["batch_size"]):
                ids = [train_ids[j] for j in order[b:b + fc["batch_size"]]]
                wav, mask = batchify([audio[u] for u in ids], args.device)
                y = torch.tensor([label[item_of[u]] for u in ids], device=args.device)
                opt.zero_grad(set_to_none=True)
                lossf(forward(model, wav, mask), y).backward()
                opt.step()
            train_s += time.perf_counter() - t0
            print(f"  {spk} shared-FT epoch {ep}/{args.max_epochs} {train_s:.0f}s", flush=True)
            if ep in EVAL_AT or ep == args.max_epochs:
                pr = evaluate(model, audio, test_ids, items, args.device)
                u = uar(yte, pr)
                w.writerow([args.corpus, spk, rs[0]["group"], ep, f"{u:.4f}", len(train_ids), len(test_ids),
                            f"{train_s:.1f}", args.device]); out.flush()
                pout.write(json.dumps({"speaker": spk, "epoch": ep, "preds": dict(zip(test_ids, pr))}) + "\n"); pout.flush()
                print(spk, "shared-FT", ep, f"uar={u:.3f}", flush=True)
        del model, opt
        audio_cache.clear()
    out.close(); pout.close()


if __name__ == "__main__":
    main()
