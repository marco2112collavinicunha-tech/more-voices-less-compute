"""Stage 1b — data-quality filter, counts only (no model).

  python -m pipeline.s01b_quality torgo
  python -m pipeline.s01b_quality torgo --max-seconds 30


Drops recordings that are shorter than 0.1 s after the protocol's edge trimming
(30 dB below peak): these are empty or clipped files, too short for XLSR-53's
convolutional front end. Rewrites the manifest, keeps the dropped list, and
recomputes the counts and the primary k with the same pre-registered rule.

With --max-seconds S, also drops recordings longer than S seconds after trimming
(a single-word prompt cannot last 30 s; one 194 s TORGO file made per-speaker
fine-tuning impossible). That pass writes its own dropped list (suffix _long).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter

from greenafi.data import (by_speaker, eligible_items, load_audio, load_config, primary_k,
                           read_manifest, write_manifest)

MIN_SAMPLES = 1600  # 0.1 s at 16 kHz

corpus = sys.argv[1]
MAX_SAMPLES = None
if "--max-seconds" in sys.argv:
    MAX_SAMPLES = int(float(sys.argv[sys.argv.index("--max-seconds") + 1]) * 16000)
cfg = load_config()
a, p = cfg["audio"], cfg["personalisation"]
rows = read_manifest(f"data/manifests/{corpus}.csv")
keep, drop = [], []
for r in rows:
    try:
        n = len(load_audio(r["path"], a["trim_db_below_peak"], a["peak_normalise"]))
    except Exception as e:  # unreadable file
        n = -1
    ok = n >= MIN_SAMPLES and (MAX_SAMPLES is None or n <= MAX_SAMPLES)
    (keep if ok else drop).append({**r, "samples_after_trim": n})
write_manifest(f"data/manifests/{corpus}.csv", keep)
suffix = "_long" if MAX_SAMPLES else ""
with open(f"results/s01/{corpus}_dropped_quality{suffix}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(drop[0].keys()) if drop else ["utt_id"])
    w.writeheader(); w.writerows(drop)
counts = json.load(open(f"results/s01/{corpus}_counts.json"))
spk = by_speaker(keep)
counts.update({
    "n_recordings": len(keep),
    "groups": dict(Counter(r["group"] for r in keep)),
    ("dropped_quality_long" if MAX_SAMPLES else "dropped_quality"): len(drop),
    "per_speaker": {s: {"group": rs[0]["group"], "n_recordings": len(rs), "n_items": len({r["item"] for r in rs}),
                        "eligible_items_by_k": {k: len(eligible_items(rs, k, p["min_test_per_item"])) for k in p["k_values"]}}
                    for s, rs in sorted(spk.items())},
    **primary_k(keep, p["k_values"], p["min_test_per_item"], p["min_items_per_speaker"]),
})
json.dump(counts, open(f"results/s01/{corpus}_counts.json", "w"), indent=2)
print(json.dumps({"kept": len(keep), "dropped": len(drop), "primary_k": counts["primary_k"],
                  "speakers_included": counts["speakers_included"]}, indent=2))
