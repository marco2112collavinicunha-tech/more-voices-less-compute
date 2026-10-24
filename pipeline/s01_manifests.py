"""Stage 1 — build per-speaker manifests and fix the primary k from counts only.

  python -m pipeline.s01_manifests torgo
  python -m pipeline.s01_manifests speech_commands
  python -m pipeline.s01_manifests fsc
  python -m pipeline.s01_manifests synthetic      # tiny fake corpus for pipeline tests

Writes data/manifests/<corpus>.csv and results/s01/<corpus>_counts.json.
No model is loaded here: the primary-k decision cannot see any accuracy.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from greenafi.data import (by_speaker, eligible_items, load_config, primary_k,
                           write_manifest)

WORD = re.compile(r"^[a-z']+$")


def torgo(cfg):
    c = cfg["corpora"]["torgo"]
    rows = []
    for group, speakers in (("dysarthric", c["dysarthric"]), ("control", c["control"])):
        for spk in speakers:
            for sess in sorted(glob.glob(os.path.join(c["root"], "**", spk, "Session*"), recursive=True)):
                prompts = os.path.join(sess, "prompts")
                if not os.path.isdir(prompts):
                    continue
                for ptxt in sorted(glob.glob(os.path.join(prompts, "*.txt"))):
                    uid = os.path.splitext(os.path.basename(ptxt))[0]
                    text = open(ptxt, errors="ignore").read().strip().lower()
                    if not WORD.match(text) or text == "xxx":  # single-word prompts only; "xxx" is TORGO's placeholder, not a word
                        continue
                    wav = None
                    for mic in c["mic_preference"]:
                        cand = os.path.join(sess, mic, uid + ".wav")
                        if os.path.exists(cand):
                            wav = cand
                            break
                    if wav is None:
                        continue
                    rows.append({"corpus": "torgo", "speaker": spk, "group": group, "item": text,
                                 "utt_id": f"{spk}/{os.path.basename(sess)}/{uid}", "path": wav,
                                 "session": os.path.basename(sess)})
    return rows


def speech_commands(cfg):
    c = cfg["corpora"]["speech_commands"]
    rows = []
    for wav in glob.glob(os.path.join(c["root"], "*", "*.wav")):
        word = os.path.basename(os.path.dirname(wav))
        if word.startswith("_"):
            continue
        spk = os.path.basename(wav).split("_nohash_")[0]
        rows.append({"corpus": "speech_commands", "speaker": spk, "group": "typical", "item": word,
                     "utt_id": f"{word}/{os.path.basename(wav)}", "path": wav})
    # keep the speakers with the most eligible items at the largest k, deterministic order
    k_max = max(cfg["personalisation"]["k_values"])
    mt = cfg["personalisation"]["min_test_per_item"]
    spk = by_speaker(rows)
    score = sorted(spk, key=lambda s: (-len(eligible_items(spk[s], k_max, mt)), s))
    keep = set(score[: c["max_speakers"]])
    return [r for r in rows if r["speaker"] in keep]


def fsc(cfg):
    c = cfg["corpora"]["fsc"]
    rows = []
    for split in ("train", "valid", "test"):
        p = os.path.join(c["root"], "data", f"{split}_data.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            intent = f"{r['action']}|{r['object']}|{r['location']}"
            rows.append({"corpus": "fsc", "speaker": r["speakerId"], "group": "typical", "item": intent,
                         "utt_id": r["path"], "path": os.path.join(c["root"], r["path"])})
    return rows


def synthetic(cfg):
    """3 'speakers' x 12 'items' x 12 tones each: only for testing the plumbing."""
    import soundfile as sf
    root = "data/synthetic"
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for s, spk in enumerate(["S1", "S2", "S3"]):
        for i in range(12):
            for rep in range(12):
                t = np.arange(int(16000 * (1.2 + 0.1 * rng.random()))) / 16000
                f0 = 110 + 25 * i + 7 * s
                x = 0.3 * np.sin(2 * np.pi * f0 * t) * np.hanning(len(t)) + 0.01 * rng.normal(size=len(t))
                p = f"{root}/{spk}_{i:02d}_{rep:02d}.wav"
                sf.write(p, x.astype(np.float32), 16000)
                rows.append({"corpus": "synthetic", "speaker": spk, "group": "control" if s else "dysarthric",
                             "item": f"item{i:02d}", "utt_id": f"{spk}/{i:02d}/{rep:02d}", "path": p,
                             "session": f"s{rep // 6}"})
    return rows


def main():
    corpus = sys.argv[1]
    cfg = load_config()
    rows = {"torgo": torgo, "speech_commands": speech_commands, "fsc": fsc, "synthetic": synthetic}[corpus](cfg)
    if not rows:
        sys.exit(f"no recordings found for {corpus}; check config/study.json corpora.{corpus}.root")
    os.makedirs("data/manifests", exist_ok=True)
    os.makedirs("results/s01", exist_ok=True)
    write_manifest(f"data/manifests/{corpus}.csv", rows)

    p = cfg["personalisation"]
    spk = by_speaker(rows)
    report = {
        "corpus": corpus,
        "n_recordings": len(rows),
        "n_speakers": len(spk),
        "groups": dict(Counter(r["group"] for r in rows)),
        "per_speaker": {
            s: {"group": rs[0]["group"], "n_recordings": len(rs), "n_items": len({r["item"] for r in rs}),
                "eligible_items_by_k": {k: len(eligible_items(rs, k, p["min_test_per_item"])) for k in p["k_values"]}}
            for s, rs in sorted(spk.items())
        },
        **primary_k(rows, p["k_values"], p["min_test_per_item"], p["min_items_per_speaker"]),
        "rule": p["primary_k_rule"],
    }
    json.dump(report, open(f"results/s01/{corpus}_counts.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "per_speaker"}, indent=2))


if __name__ == "__main__":
    main()
