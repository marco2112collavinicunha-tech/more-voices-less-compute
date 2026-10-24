# Run from the repository root. Corpus: make embed C=torgo
C ?= torgo
PY = python -m

.PHONY: checks cost lifecycle manifests embed classifiers whisper finetune stats assets report pages all

checks:      ; $(PY) pipeline.s02_checks --pretrained
cost:        ; $(PY) pipeline.s03_cost --pretrained --stages xlsr,heads,whisper,finetune
lifecycle:   ; $(PY) pipeline.s03b_lifecycle
manifests:   ; $(PY) pipeline.s01_manifests $(C)
embed:       ; $(PY) pipeline.s04_embed $(C)
classifiers: ; $(PY) pipeline.s05_classifiers $(C)
whisper:     ; $(PY) pipeline.s06_whisper $(C) --language en --device $(DEVICE)
finetune:    ; $(PY) pipeline.s07_finetune $(C) --device $(DEVICE)
stats:       ; $(PY) pipeline.s08_stats $(C)
assets:      ; $(PY) pipeline.s09_report_assets
report: assets
	cd report && pdflatex -interaction=nonstopmode main.tex >/dev/null; bibtex main >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex | tail -2
pages:       ; pdfinfo report/main.pdf | grep Pages
all: manifests embed classifiers whisper finetune stats report

DEVICE ?= mps
