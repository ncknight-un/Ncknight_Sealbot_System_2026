#!/usr/bin/env bash
# run_windows.sh - run_checks.sh at several window lengths, then plot per-class accuracy against them.
#   ./run_windows.sh DATA_FOLDER [WINDOW ...]        e.g.  ./run_windows.sh data_11_17 4 8 16 32
# Models already trained at a window length are skipped, so re-running only fills in what is missing.
set -euo pipefail
DATA="${1:?usage: ./run_windows.sh DATA_FOLDER [WINDOW ...]}"; DATA="${DATA%/}"; shift
WINDOWS=("${@:-4 8 16 32}"); [ $# -eq 0 ] && WINDOWS=(4 8 16 32)
for W in "${WINDOWS[@]}"; do
    echo "=== window $W"
    ./run_checks.sh "$DATA" "$W"          # models already trained at this window are skipped
done
python plot_windows.py "$DATA" --windows "${WINDOWS[@]}" --confusion 16
python plot_windows.py "$DATA" --windows "${WINDOWS[@]}" --split val test --confusion 16
