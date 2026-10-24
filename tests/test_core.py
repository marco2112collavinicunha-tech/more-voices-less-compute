"""Fast checks that run without audio or model weights (pytest -q)."""
import json
import re
from pathlib import Path

import numpy as np

from greenafi.data import uar

ROOT = Path(__file__).resolve().parents[1]


def test_uar_is_mean_of_per_item_recall():
    y = ["a", "a", "b", "b", "b", "b"]
    p = ["a", "b", "b", "b", "b", "b"]
    assert abs(uar(y, p) - (0.5 + 1.0) / 2) < 1e-12


def test_truncation_is_exact():
    s02 = json.load(open(ROOT / "results/s02/checks.json")) if (ROOT / "results/s02/checks.json").exists() else \
        json.load(open(next((ROOT / "results/s02").glob("*.json"))))
    assert s02["truncation"]["layer_12"]["max_abs_diff"] == 0.0


def test_whisper_cost_formula_matches_measured_medium_encoder():
    from pipeline.s08g_whisper_sizes import costs
    c = json.load(open(ROOT / "results/s03d/torgo_costs.json"))
    ratio = costs("medium", 1)["encoder"] / c["whisper"]["encoder_flops"]
    assert abs(ratio - 1) < 0.01


def test_every_number_in_the_report_is_generated():
    defined = set(re.findall(r"\\newcommand\{\\(num[A-Za-z]+)\}", (ROOT / "report/generated/numbers.tex").read_text()))
    used = set()
    for f in (ROOT / "report/sections").glob("*.tex"):
        used |= set(re.findall(r"\\(num[A-Za-z]+)", f.read_text()))
    assert not (used - defined), sorted(used - defined)


def test_matched_analysis_reproduces_first_analysis_closely():
    m = json.load(open(ROOT / "results/s08b/torgo_matched.json"))
    diffs = [abs(r["s05"] - r["recomputed"]) for r in m["reproduction_mismatches"]]
    assert max(diffs, default=0) < 0.02  # disclosed in Appendix A: within about one UAR point


def test_layer12_beats_layer24_for_every_speaker():
    import pandas as pd
    ps = pd.read_csv(ROOT / "results/s08b/torgo_matched_per_speaker.csv")
    a = ps[ps.analysis_group == "A_finetune_split"].pivot_table(index="speaker", columns="system", values="uar")
    assert (a["frozen_L12_1nn"] > a["frozen_L24_1nn"]).all()
    assert (a["shared_L12_logreg"] > a["shared_L24_logreg"]).all()
