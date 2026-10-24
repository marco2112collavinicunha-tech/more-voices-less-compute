#!/bin/bash
# Live demo with the microphone: say one of ten English words (yes no up down left right on off stop go).
# First use installs the small 'sounddevice' package (~1 MB). macOS will ask for microphone permission for Terminal.
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -c "import sounddevice" 2>/dev/null || pip install sounddevice
python -m demo.live_demo mic
