"""Audio loading, manifests and deterministic per-speaker splits.

Manifest = CSV with columns:
  corpus, speaker, group, item, utt_id, path
  group: 'dysarthric' | 'control' | 'typical'
  item:  the label the system must recover (a word, a phrase or an intent)
Splits are a pure function of (seed, speaker, k, draw), so every stage that
needs the same test set (classifiers, Whisper, fine-tuning, statistics)
recomputes it identically instead of passing files around.
"""
from __future__ import annotations

import csv
import json
import zlib
from collections import defaultdict
from dataclasses import dataclass
from math import gcd

import numpy as np

CONFIG_PATH = "config/study.json"
SR = 16000


def load_config(path: str = CONFIG_PATH) -> dict:
    return json.load(open(path))


def load_audio(path: str, trim_db: float = 30.0, peak_normalise: bool = True) -> np.ndarray:
    """Mono, 16 kHz, edges more than `trim_db` below the peak trimmed, peak-normalised
    (the preparation fixed in config/study.json)."""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    peak = float(np.abs(x).max()) + 1e-9
    keep = np.where(np.abs(x) > peak * 10 ** (-trim_db / 20))[0]
    if len(keep):
        x = x[keep[0]: keep[-1] + 1]
    if peak_normalise:
        x = x / (np.abs(x).max() + 1e-9)
    return x.astype(np.float32)


def read_manifest(path: str) -> list[dict]:
    return list(csv.DictReader(open(path, newline="")))


def write_manifest(path: str, rows: list[dict]) -> None:
    cols = ["corpus", "speaker", "group", "item", "utt_id", "path", "session"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def by_speaker(rows: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[r["speaker"]].append(r)
    return dict(out)


def eligible_items(rows: list[dict], k: int, min_test: int) -> list[str]:
    counts = defaultdict(int)
    for r in rows:
        counts[r["item"]] += 1
    return sorted(i for i, c in counts.items() if c >= k + min_test)


@dataclass
class Split:
    speaker: str
    k: int
    draw: int
    items: list[str]
    train: list[str]   # utt_ids
    test: list[str]    # utt_ids


def make_split(rows: list[dict], speaker: str, k: int, draw: int, seed: int,
               min_test: int, min_items: int) -> Split | None:
    """k training recordings per eligible item, all remaining recordings of
    that item for test. None when the speaker has too few eligible items."""
    items = eligible_items(rows, k, min_test)
    if len(items) < min_items:
        return None
    rng = np.random.default_rng(zlib.crc32(f"{seed}|{speaker}|{k}|{draw}".encode()))
    per_item = defaultdict(list)
    sess = {}
    for r in rows:
        if r["item"] in items:
            per_item[r["item"]].append(r["utt_id"])
            sess[r["utt_id"]] = r.get("session") or ""
    train, test = [], []
    for it in items:
        ids = sorted(per_item[it])
        rng.shuffle(ids)
        ids.sort(key=lambda u: sess[u])  # stable: earliest sessions train, later sessions test
        train += ids[:k]
        test += ids[k:]
    return Split(speaker, k, draw, items, train, test)


def primary_k(rows: list[dict], k_values: list[int], min_test: int, min_items: int) -> dict:
    """Pre-registered rule, counts only (never accuracy).

    1. Speakers with fewer than `min_items` eligible items even at the
       smallest k are excluded from every analysis (and reported).
    2. Primary k = the largest k at which every remaining speaker still has
       `min_items` eligible items.
    """
    spk = by_speaker(rows)
    k0 = min(k_values)
    base = sorted(s for s, r in spk.items() if len(eligible_items(r, k0, min_test)) >= min_items)
    excluded = sorted(set(spk) - set(base))
    ok = [k for k in k_values
          if base and all(len(eligible_items(spk[s], k, min_test)) >= min_items for s in base)]
    return {"primary_k": max(ok) if ok else None, "speakers_included": base, "speakers_excluded": excluded}


def uar(y_true: list[str], y_pred: list[str]) -> float:
    """Unweighted average recall: mean over items of the share of that item's
    test recordings recognised correctly."""
    per = defaultdict(list)
    for t, p in zip(y_true, y_pred):
        per[t].append(t == p)
    return float(np.mean([np.mean(v) for v in per.values()]))
