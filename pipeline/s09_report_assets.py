"""Stage 9 — every number and figure in the report, generated from result files.

  python -m pipeline.s09_report_assets

Writes report/generated/numbers.tex (one LaTeX macro per number) and
report/figures/*.pdf. A number whose result file does not exist yet becomes
\\pending{...}, printed in red in the PDF, so a missing result can never be
mistaken for a real one. Nothing in the report is typed by hand.
"""
from __future__ import annotations

import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Palatino-like figure font to match the report's mathpazo, when TeX Gyre Pagella is installed
for _f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-*.otf") + \
        glob.glob(os.path.expanduser("~/Library/Fonts/texgyrepagella-*.otf")):
    font_manager.fontManager.addfont(_f)
_SERIF = ["TeX Gyre Pagella", "Palatino", "Book Antiqua", "Bitstream Charter", "DejaVu Serif"]

OUT_TEX = "report/generated/numbers.tex"
FIG = "report/figures"
COST_DIRS = ["results/s03", "results/vm_random_weights"]  # measured run first, VM fallback

# Palette (journal style): the proposed design in near-black, the general-purpose recogniser in dark brick,
# retraining in mid grey, the speaker's own-recording systems in dark grey; grid in light warm grey.
BLUE, GREY, INK, INK2, GRID = "#1a1a1a", "#a8a8a8", "#0b0b0b", "#5a5a5a", "#e6e5e0"
RUST = "#8c2f1e"  # general-purpose recogniser (Whisper) in every figure
_DARK = os.environ.get("FIG_DARK") == "1"  # slide variant: light ink on a dark background, sans-serif
_WARM = os.environ.get("FIG_WARM") == "1"  # slide variant: cream background, teal for the proposed design, terracotta for Whisper
_BG = "white"
if _DARK:
    FIG = os.environ.get("FIG_DIR", "report/figures_dark"); OUT_TEX = os.environ.get("FIG_TEX", "/tmp/numbers_dark.tex")
    BLUE, GREY, INK, INK2, GRID = "#F5F3EE", "#8A8F99", "#F5F3EE", "#C9CCD2", "#2A2E36"
    RUST = "#E9B44C"; _BG = "#0F1115"
elif _WARM:
    FIG = os.environ.get("FIG_DIR", "report/figures_warm"); OUT_TEX = os.environ.get("FIG_TEX", "/tmp/numbers_warm.tex")
    BLUE, GREY, INK, INK2, GRID = "#0F6E63", "#A9A49A", "#15181D", "#4A4F57", "#E3DCCF"
    RUST = "#B5472B"; _BG = os.environ.get("FIG_BG", "#F6F1E7")
plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif" if (_DARK or _WARM) else "serif", "mathtext.fontset": "stix", "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.bbox": "tight", "font.serif": _SERIF, "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "text.color": INK, "legend.labelcolor": INK, "figure.facecolor": _BG, "axes.facecolor": _BG, "savefig.facecolor": _BG,
})

macros: dict[str, str] = {}


def put(name: str, value, fmt: str = "{:,.1f}"):
    assert name.isalpha(), name
    if value is None:
        macros[name] = r"\pending{" + name + "}"
    elif not isinstance(value, str) and fmt.format(value).startswith("-"):
        macros[name] = "$-$" + fmt.format(value)[1:]
    elif isinstance(value, str):
        macros[name] = value
    else:
        macros[name] = fmt.format(value)


def first_existing(fname):
    for d in COST_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p, d
    return None, None


def load(fname):
    p, d = first_existing(fname)
    return (json.load(open(p)), d) if p else (None, None)


# ------------------------------------------------------------------ cost
xl, src = load("compute_xlsr.json")
wh, _ = load("compute_whisper.json")
ft, _ = load("compute_finetune.json")
hd, _ = load("compute_heads.json")
eq, _ = load("equivalence.json")
summ = json.load(open("results/s03c/summary.json")) if os.path.exists("results/s03c/summary.json") else load("summary.json")[0]
EN = json.load(open("results/s03c/energy.json")) if os.path.exists("results/s03c/energy.json") else None
grids = json.load(open("config/grids.json"))

put("CostSource", "measured run with released weights" if src == "results/s03" else
    "the exact model architectures with random weights, on a 2-core Intel Xeon server processor (virtual machine)")
put("EnergyMode", (summ or {}).get("energy_mode", None) if summ else None, "{}")

X = {(r["system"], r["audio_s"]): r for r in xl["rows"]}
W = {r["system"]: r for r in wh["rows"]}
F = {r["system"]: r for r in ft["rows"]}
H = {r["system"]: r for r in hd["rows"]}
g = lambda s, a=2.2: X[(s, a)]["flops"] / 1e9

put("GflopLtwentyfour", g("xlsr_L24")); put("GflopLeighteen", g("xlsr_L18")); put("GflopLtwelve", g("xlsr_L12"))
put("GflopWhisperFree", W["whisper_free"]["flops"] / 1e9, "{:,.0f}")
put("GflopWhisperForced", W["whisper_forced12"]["flops"] / 1e9, "{:,.0f}")
put("GflopWhisperEnc", W["whisper_encoder_only"]["flops"] / 1e9, "{:,.0f}")
put("RatioWhisperFreeLtwelve", W["whisper_free"]["flops"] / 1e9 / g("xlsr_L12"))
put("RatioWhisperForcedLtwelve", W["whisper_forced12"]["flops"] / 1e9 / g("xlsr_L12"))
put("RatioWhisperFreeLtwentyfour", W["whisper_free"]["flops"] / 1e9 / g("xlsr_L24"))
put("ShareLtwelve", 100 * g("xlsr_L12") / g("xlsr_L24"), "{:.0f}")
put("SaveLtwelve", 100 * (1 - g("xlsr_L12") / g("xlsr_L24")), "{:.0f}")
put("SaveLeighteen", 100 * (1 - g("xlsr_L18") / g("xlsr_L24")), "{:.0f}")
per_layer = (g("xlsr_L24") - g("xlsr_L12")) / 12
put("GflopPerLayer", per_layer, "{:.2f}")
put("GflopFrontEnd", g("xlsr_L12") - 12 * per_layer)
put("GflopLtwelveFive", g("xlsr_L12", 5.0), "{:,.0f}")
put("RatioWhisperLtwelveFive", W["whisper_free"]["flops"] / 1e9 / g("xlsr_L12", 5.0))
put("SecLtwentyfour", X[("xlsr_L24", 2.2)]["median_s"], "{:.2f}")
put("SecLtwelve", X[("xlsr_L12", 2.2)]["median_s"], "{:.2f}")
put("SecLtwelveInt", X[("xlsr_L12_int8", 2.2)]["median_s"], "{:.2f}")
put("SecWhisperFree", W["whisper_free"]["median_s"], "{:.1f}")
put("SecWhisperForced", W["whisper_forced12"]["median_s"], "{:.1f}")
put("RatioIntSpeed", X[("xlsr_L12", 2.2)]["median_s"] / X[("xlsr_L12_int8", 2.2)]["median_s"], "{:.1f}")
put("SecFtFullClip", F["finetune_full"]["s_per_clip"], "{:.1f}")
put("SecFtTopClip", F["finetune_top"]["s_per_clip"], "{:.1f}")
put("ParamsFtFull", F["finetune_full"]["trainable_params"] / 1e6, "{:.0f}")
put("ParamsFtTop", F["finetune_top"]["trainable_params"] / 1e6, "{:.0f}")

N = ft["n_train_clips"]
put("NTrain", N, "{:d}")
tf_full = F["finetune_full"]["flops_per_epoch"] / 1e12
tf_top = F["finetune_top"]["flops_per_epoch"] / 1e12
tf_l12 = g("xlsr_L12") * N / 1e3
tf_l18 = g("xlsr_L18") * N / 1e3
put("TflopFtFullEpoch", tf_full); put("TflopFtTopEpoch", tf_top)
put("TflopFtFullTen", 10 * tf_full); put("TflopFtFullThirty", 30 * tf_full); put("TflopFtTopTen", 10 * tf_top)
put("TflopFrozenLtwelve", tf_l12); put("TflopFrozenLeighteen", tf_l18)
put("RatioFtTenFrozen", 10 * tf_full / tf_l12, "{:.0f}"); put("RatioFtThirtyFrozen", 30 * tf_full / tf_l12, "{:.0f}")
put("RecogEquivFtTen", 10 * tf_full * 1e3 / g("xlsr_L24"), "{:,.0f}")
put("RecogEquivFtThirty", 30 * tf_full * 1e3 / g("xlsr_L24"), "{:,.0f}")
tf_lora = F["finetune_lora"]["flops_per_epoch"] / 1e12 if "finetune_lora" in F else None
tf_ftl12 = F["finetune_full_L12"]["flops_per_epoch"] / 1e12 if "finetune_full_L12" in F else None
put("TflopFtLoraEpoch", tf_lora); put("TflopFtLtwelveEpoch", tf_ftl12)
put("TflopFtLoraTen", 10 * tf_lora if tf_lora else None); put("TflopFtLtwelveTen", 10 * tf_ftl12 if tf_ftl12 else None)
put("RatioFtLoraTenFrozen", 10 * tf_lora / tf_l12 if tf_lora else None, "{:.0f}")
put("RatioFtLtwelveTenFrozen", 10 * tf_ftl12 / tf_l12 if tf_ftl12 else None, "{:.0f}")
put("ParamsFtLora", F["finetune_lora"]["trainable_params"] / 1e3 if "finetune_lora" in F else None, "{:,.0f}")
put("StoreFtLora", F["finetune_lora"]["per_user_model_bytes_fp32"] / 1e6 if "finetune_lora" in F else None, "{:.1f}")
put("StoreFtLtwelve", F["finetune_full_L12"]["per_user_model_bytes_fp32"] / 1e6 if "finetune_full_L12" in F else None, "{:,.0f}")
put("GflopFtClip", F["finetune_full"]["flops_per_clip"] / 1e9, "{:.0f}")
pL12 = X[("xlsr_L12", 2.2)]["params"]
put("ParamsLtwelve", pL12 / 1e6, "{:.0f}")
put("MbLtwelve", pL12 * 4 / 1e6, "{:,.0f}")
put("MbLtwelveInt", pL12 * 1 / 1e6, "{:,.0f}")
from scipy import stats as _st
put("NeedSdEight", 5 * 8 ** 0.5 / _st.t.ppf(0.95, 7), "{:.1f}")
put("RatioFtClipInference", F["finetune_full"]["flops_per_clip"] / 1e9 / g("xlsr_L24"))

st = hd["storage_bytes_per_user"]
put("StoreFtFull", F["finetune_full"]["per_user_model_bytes_fp32"] / 1e9, "{:.2f}")
put("StoreFtTop", F["finetune_top"]["per_user_model_bytes_fp32"] / 1e6, "{:.0f}")
put("StoreLogreg", st["logreg"] / 1e3, "{:.0f}")
put("StoreKnn", st["1nn"] / 1e3, "{:.0f}")
put("StoreRatioKnn", F["finetune_full"]["per_user_model_bytes_fp32"] / st["1nn"], "{:,.0f}")
put("StoreFtFullHundredK", F["finetune_full"]["per_user_model_bytes_fp32"] * 1e5 / 1e12, "{:.0f}")
put("StoreKnnHundredK", st["1nn"] * 1e5 / 1e9, "{:.0f}")
put("MsPredictKnn", H["predict_1nn"]["median_s"] * 1e3, "{:.2f}")
put("MsFitLogreg", H["fit_logreg"]["median_s"] * 1e3, "{:.0f}")

if eq:
    d = max(v["max_abs_diff"] for v in eq["truncation"].values())
    put("TruncMaxDiff", d, "{:.1f}")

# lifecycle
lc_path = "results/s03c/lifecycle.csv" if os.path.exists("results/s03c/lifecycle.csv") else first_existing("lifecycle.csv")[0]
LC = {(r["system"], float(r["uses_per_day"])): r for r in csv.DictReader(open(lc_path))}
def lc(sys_name, u, col, fmt="{:,.0f}", scale=1.0):
    return float(LC[(sys_name, u)][col]) * scale
S_WH = "Generic: Whisper-medium, free transcription"
S_WF = "Generic: Whisper-medium, forced choice (12)"
S_FT10 = "Per-speaker fine-tuning, all layers (10 ep)"
S_FT30 = "Per-speaker fine-tuning, all layers (30 ep)"
S_P24 = "Personalized (protocol), full 24-layer pass"
S_L12 = "Personalized, truncated at layer 12 (1-NN)"
S_I8 = "Personalized, layer 12, int8 (fidelity pending)"
S_LORA10 = "Per-speaker fine-tuning, LoRA (10 ep)"
S_FTL12 = "Per-speaker fine-tuning, all layers of a model cut at layer 12 (10 ep)"
S_FTL1230 = "Per-speaker fine-tuning, all layers of a model cut at layer 12 (30 ep)"
for key, s in (("Whisper", S_WH), ("WhisperForced", S_WF), ("FtTen", S_FT10), ("FtThirty", S_FT30), ("Protocol", S_P24), ("Ltwelve", S_L12),
               ("Int", S_I8), ("Lora", S_LORA10), ("FtLtwelve", S_FTL12)):
    put(f"WhYear{key}", lc(s, 100.0, "user_year_Wh"), "{:,.0f}")
    put(f"WhYearLow{key}", lc(s, 20.0, "user_year_Wh"), "{:,.1f}")
    put(f"GcoBrazil{key}", lc(s, 100.0, "gCO2e_user_year_Brazil"), "{:,.1f}")
    put(f"GcoIndia{key}", lc(s, 100.0, "gCO2e_user_year_India"), "{:,.0f}")
    put(f"KwhHundredK{key}", lc(s, 100.0, "kWh_year_100000_users"), "{:,.0f}")
put("ShareFtTenLow", 100 * lc(S_FT10, 20.0, "personalisation_share_of_year"), "{:.0f}")
put("ShareFtThirtyLow", 100 * lc(S_FT30, 20.0, "personalisation_share_of_year"), "{:.0f}")
put("ShareFtTenHigh", 100 * lc(S_FT10, 100.0, "personalisation_share_of_year"), "{:.0f}")
put("ShareLtwelveHigh", 100 * lc(S_L12, 100.0, "personalisation_share_of_year"), "{:.1f}")
put("BreakevenFtTen", lc(S_FT10, 100.0, "breakeven_days_personalisation_vs_inference"), "{:.0f}")
put("BreakevenLtwelve", lc(S_L12, 100.0, "breakeven_days_personalisation_vs_inference"), "{:.1f}")
put("ShareFtTenHighPct", 100 * lc(S_FT10, 100.0, "personalisation_share_of_year"), "{:.0f}")
put("ShareFtThirtyHigh", 100 * lc(S_FT30, 100.0, "personalisation_share_of_year"), "{:.0f}")
put("ShareFtLtwelveHigh", 100 * lc(S_FTL12, 100.0, "personalisation_share_of_year"), "{:.0f}")
put("ShareFtLtwelveLow", 100 * lc(S_FTL12, 20.0, "personalisation_share_of_year"), "{:.0f}")
put("RatioYearFtTenLtwelve", lc(S_FT10, 100.0, "user_year_Wh") / lc(S_L12, 100.0, "user_year_Wh"), "{:.1f}")
put("RatioYearFtLtwelveLtwelve", lc(S_FTL12, 100.0, "user_year_Wh") / lc(S_L12, 100.0, "user_year_Wh"), "{:.2f}")
put("RatioYearFtLtwelveLtwelveLow", lc(S_FTL12, 20.0, "user_year_Wh") / lc(S_L12, 20.0, "user_year_Wh"), "{:.1f}")
put("RatioYearWhisperLtwelve", lc(S_WH, 100.0, "user_year_Wh") / lc(S_L12, 100.0, "user_year_Wh"), "{:.0f}")
gf = grids["founding_members"]
for name, key in (("Brazil", "Brazil"), ("India", "India"), ("China", "China"), ("Russia", "Russia"), ("SouthAfrica", "South Africa")):
    put(f"Grid{name}", gf[key], "{:,.0f}")
put("GridRatioIndiaBrazil", gf["India"] / gf["Brazil"], "{:.1f}")
put("PowerAssumed", 10, "{:d}")
if EN:
    d = EN["derivation"]["low_cost_arm_board"]
    put("JperGflopPi", d["J_per_GFLOP"], "{:.3f}")
    put("GflopPiWorkload", d["workload_GFLOP"], "{:.0f}")
    put("JPiWorkload", d["reported_J"], "{:.1f}")
    put("GflopPerJPi", 1 / d["J_per_GFLOP"], "{:.1f}")
    rr = EN["year_ratio_vs_frozen_L12_at_100_per_day"]
    put("XcheckFtFlop", rr["Fine-tuning, all layers (10 ep)"]["flop_ratio"], "{:.2f}")
    put("XcheckFtTime", rr["Fine-tuning, all layers (10 ep)"]["time_ratio"], "{:.2f}")
    put("XcheckWhFlop", rr["Whisper-medium, free transcription"]["flop_ratio"], "{:.1f}")
    put("XcheckWhTime", rr["Whisper-medium, free transcription"]["time_ratio"], "{:.1f}")
    put("XcheckFtLtwelveFlop", rr["Fine-tuning, model cut at layer 12 (10 ep)"]["flop_ratio"], "{:.2f}")
    put("XcheckFtLtwelveTime", rr["Fine-tuning, model cut at layer 12 (10 ep)"]["time_ratio"], "{:.2f}")
    put("JRecLtwelvePi", g("xlsr_L12") * d["J_per_GFLOP"], "{:.0f}")
    put("JRecWhisperPi", W["whisper_free"]["flops"] / 1e9 * d["J_per_GFLOP"], "{:.0f}")
    put("HasAppleAnchor", "yes" if "apple_laptop" in EN["derivation"] else "no")
put("GridEthiopia", grids["other_members"]["Ethiopia"], "{:,.0f}")
put("StudyKwh", None)  # energy used by this study's own experiments: filled from the Mac run log

os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
with open(OUT_TEX, "w") as f:
    f.write("% Generated by pipeline/s09_report_assets.py. Do not edit by hand.\n")
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
print(f"{len(macros)} macros, {sum('pending' in v for v in macros.values())} pending -> {OUT_TEX}")

# ------------------------------------------------------------------ figures
os.makedirs(FIG, exist_ok=True)


def bars(path, labels, vals, colors, xlabel, log=False, fmt="{:,.0f}"):
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(vals) + 0.6))
    y = range(len(vals))
    if log:
        ax.scatter(vals, y, color=colors, s=36, zorder=3)
        for yi, v, c in zip(y, vals, colors):
            ax.plot([min(vals) / 3, v], [yi, yi], color=c, lw=1.5, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlim(min(vals) / 3, max(vals) * 4)
    else:
        ax.barh(list(y), vals, color=colors, height=0.6)
        ax.set_xlim(0, max(vals) * 1.15)
    for yi, v in zip(y, vals):
        ax.text(v * (1.15 if log else 1) + (0 if log else max(vals) * 0.01), yi, fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    fig.savefig(path); plt.close(fig)


bars(f"{FIG}/flops_per_recognition.pdf",
     ["Whisper-medium, forced choice", "Whisper-medium, free transcription", "XLSR-53 to layer 24",
      "XLSR-53 to layer 18", "XLSR-53 to layer 12"],
     [W["whisper_forced12"]["flops"] / 1e9, W["whisper_free"]["flops"] / 1e9, g("xlsr_L24"), g("xlsr_L18"), g("xlsr_L12")],
     [GREY, GREY, GREY, GREY, BLUE], "GFLOP per recognition (2.2 s phrase)")

bars(f"{FIG}/personalisation_cost.pdf",
     ["Fine-tune all layers, 30 epochs", "Fine-tune all layers, 10 epochs", "Fine-tune top 4 layers, 10 epochs",
      "LoRA, 10 epochs", "Retrain layer-12 model, 10 epochs",
      "Frozen encoder, 96 embeddings to layer 18", "Frozen encoder, 96 embeddings to layer 12"],
     [30 * tf_full, 10 * tf_full, 10 * tf_top, 10 * tf_lora, 10 * tf_ftl12, tf_l18, tf_l12],
     [GREY, GREY, GREY, GREY, GREY, GREY, BLUE], "TFLOP to personalise for one new user (log scale)", log=True, fmt="{:,.1f}")

fig, ax = plt.subplots(figsize=(6.0, 3.0))
for name, col, lab in (("xlsr_L24", GREY, "layer 24"), ("xlsr_L18", "#7a7a7a", "layer 18"), ("xlsr_L12", BLUE, "layer 12")):
    rs = sorted([r for r in xl["rows"] if r["system"] == name], key=lambda r: r["audio_s"])
    xs, ys = [r["audio_s"] for r in rs], [r["flops"] / 1e9 for r in rs]
    ax.plot(xs, ys, color=col, lw=1.8, marker="o", ms=4)
    ax.text(xs[-1] + 0.1, ys[-1], f"XLSR-53, {lab}", va="center", fontsize=8)
wf = W["whisper_free"]["flops"] / 1e9
ax.axhline(wf, color=GREY, ls="--", lw=1.5)
ax.text(1.5, wf * 0.88, "Whisper-medium: same cost for any length up to 30 s", va="top", fontsize=8)
ax.set_yscale("log"); ax.set_xlim(1.3, 6.7)
ax.set_xlabel("phrase length (s)"); ax.set_ylabel("GFLOP per recognition (log)")
fig.savefig(f"{FIG}/flops_vs_length.pdf"); plt.close(fig)

# carbon per user-year on each of the ten grids (horizontal dot plot, sorted by grid intensity)
allg = {**grids["founding_members"], **grids["other_members"]}
order = sorted(allg, key=allg.get)
fig, ax = plt.subplots(figsize=(6.0, 3.3))
for s_, col, lab, mk in ((S_WF, GREY, "Whisper-medium, forced choice (12)", "s"), (S_FT10, "#7a7a7a", "retraining, all layers (10 epochs)", "^"),
                         (S_L12, BLUE, "frozen, stopped at layer 12", "o")):
    xs = [lc(s_, 100.0, f"gCO2e_user_year_{c}") for c in order]
    ax.scatter(xs, range(len(order)), color=col, marker=mk, s=22, label=lab, zorder=3)
ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{c} ({allg[c]:.0f} g/kWh)" for c in order], fontsize=7.5)
ax.set_xscale("log"); ax.set_xlabel("estimated g CO$_2$e per user-year, 100 recognitions a day (log)")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.savefig(f"{FIG}/carbon_by_grid.pdf"); plt.close(fig)

# accuracy: numbers, main table, per-speaker table, layer and accuracy-vs-compute figures
from pipeline.s09_accuracy import run as accuracy_assets
accuracy_assets(put, FIG, plt, BLUE, GREY, INK, INK2)
with open(OUT_TEX, "w") as f:  # rewrite with the accuracy macros included
    f.write("% Generated by pipeline/s09_report_assets.py. Do not edit by hand.\n")
    for k_, v_ in macros.items():
        f.write(f"\\newcommand{{\\num{k_}}}{{{v_}}}\n")
print("accuracy done;", len(macros), "macros,", sum("pending" in v for v in macros.values()), "pending")
