"""Stage 6 — general-purpose baseline: Whisper-medium, no personalisation.

  python -m pipeline.s06_whisper torgo --language en [--device mps]

For each evaluated recording it stores (results/s06/<corpus>_whisper.jsonl):
  transcript       free transcription, greedy decoding
  scores           summed and mean log-probability of EVERY item of that speaker
                   (forced choice; the closed set of a given split is chosen in s08)
Up to whisper.max_utts_per_speaker recordings per speaker (deterministic sample).

With --checkpoint/--tag (added 16 Oct): another Whisper size on the same recordings, e.g.
  --checkpoint openai/whisper-tiny --tag tiny  ->  <corpus>_whisper_testsplit_tiny.jsonl

With --test-split (added 15 Oct, see DEVIATIONS): every test recording of draw 0 at the
primary k, scored against the speaker's eligible items only (the closed set the
decision uses), written to results/s06/<corpus>_whisper_testsplit.jsonl. Optional
--speakers restricts the speakers.
"""
from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput

from greenafi.data import by_speaker, load_audio, load_config, make_split, read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--language", default="en")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--test-split", action="store_true")
    ap.add_argument("--speakers", default=None)
    ap.add_argument("--checkpoint", default=None, help="other Whisper size, e.g. openai/whisper-tiny (added 16 Oct)")
    ap.add_argument("--tag", default="", help="suffix of the output file for another checkpoint, e.g. tiny")
    args = ap.parse_args()
    cfg = load_config()
    wc = cfg["whisper"]
    counts = json.load(open(f"results/s01/{args.corpus}_counts.json"))
    keep = set(counts["speakers_included"])
    rows = [r for r in read_manifest(f"data/manifests/{args.corpus}.csv") if r["speaker"] in keep]

    ckpt = args.checkpoint or wc["checkpoint"]
    proc = WhisperProcessor.from_pretrained(ckpt)
    model = WhisperForConditionalGeneration.from_pretrained(ckpt).eval().to(args.device)
    tok = proc.tokenizer
    tok.set_prefix_tokens(language=args.language, task="transcribe", predict_timestamps=False)
    prefix_len = len(tok("").input_ids) - 1  # prefix tokens before the text (last id = end of text)

    os.makedirs("results/s06", exist_ok=True)
    if args.speakers:
        rows = [r for r in rows if r["speaker"] in set(args.speakers.split(","))]
    path = f"results/s06/{args.corpus}_whisper{'_testsplit' if args.test_split else ''}{'_' + args.tag if args.tag else ''}.jsonl"
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["utt_id"] for l in open(path)}
    out = open(path, "a")
    for spk, rs in sorted(by_speaker(rows).items()):
        items = sorted({r["item"] for r in rs})
        if args.test_split:
            p = cfg["personalisation"]
            sp = make_split(rs, spk, counts["primary_k"], 0, cfg["seed"], p["min_test_per_item"],
                            p["min_items_per_speaker"])
            if sp is None:
                continue
            items = list(sp.items)
            test = set(sp.test)
        cand = {it: torch.tensor([tok(" " + it.replace("|", " ")).input_ids], device=args.device) for it in items}
        rng = np.random.default_rng(zlib.crc32(f"{cfg['seed']}|whisper|{spk}".encode()))
        order = rng.permutation(len(rs))[: wc["max_utts_per_speaker"]]
        if args.test_split:
            order = [i for i, r in enumerate(rs) if r["utt_id"] in test]
        for i in order:
            r = rs[i]
            if r["utt_id"] in done:
                continue
            x = load_audio(r["path"], cfg["audio"]["trim_db_below_peak"], cfg["audio"]["peak_normalise"])
            feats = proc(x, sampling_rate=16000, return_tensors="pt").input_features.to(args.device)
            with torch.inference_mode():
                ids = model.generate(feats, language=args.language, task="transcribe",
                                     max_new_tokens=wc["max_new_tokens"], num_beams=1, do_sample=False)
                text = tok.batch_decode(ids, skip_special_tokens=True)[0].strip()
                enc = BaseModelOutput(last_hidden_state=model.model.encoder(feats).last_hidden_state)
                first = next(iter(cand.values()))[:, :1]
                cross = model(encoder_outputs=enc, decoder_input_ids=first, use_cache=True).past_key_values.cross_attention_cache
                scores = {}
                for it, seq in cand.items():
                    cache = EncoderDecoderCache(DynamicCache(), cross)
                    lp = model(encoder_outputs=enc, decoder_input_ids=seq[:, :-1], past_key_values=cache,
                               use_cache=True).logits.log_softmax(-1)
                    tgt = seq[:, prefix_len:]
                    tl = lp[:, prefix_len - 1:].gather(-1, tgt[..., None])[0, :, 0]
                    scores[it] = [float(tl.sum()), float(tl.mean())]
            out.write(json.dumps({"utt_id": r["utt_id"], "speaker": spk, "item": r["item"],
                                  "transcript": text, "scores": scores}) + "\n")
            out.flush()
        print(spk, "done", flush=True)


if __name__ == "__main__":
    main()
