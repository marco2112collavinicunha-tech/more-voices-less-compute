#!/bin/bash
# Energy metered with the Mac's own power meter (powermetrics: CPU + GPU + Neural Engine, idle subtracted).
# Asks for the Mac password once (sudo). Run with the laptop on mains power and other programs closed.
cd "$(dirname "$0")/.."
LOG=results/RUNLOG_energy15.txt
exec > >(tee -a "$LOG") 2>&1
echo "### energy measurement (run 15) started $(date)"
echo "Type the Mac password (not shown while typing):"
sudo -v || { echo "no sudo, stopping"; exit 1; }
( while true; do sudo -n true; sleep 50; done ) &
KEEP=$!
source .venv/bin/activate
pmset -g batt | head -2
python -m pipeline.s03_cost --pretrained --stages xlsr,heads,whisper --durations 1.6 2.2 --whisper-durations 1.6 2.2 --text-tokens 6 --min-seconds 30 --out results/s03_mac_measured
kill $KEEP
echo "### energy measurement finished $(date)"
