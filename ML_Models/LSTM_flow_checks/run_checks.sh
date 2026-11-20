#!/usr/bin/env bash
# run_checks.sh - shortcut checks for one folder of tag_imu_to_flow_class angle-test recordings.
#
#   ./run_checks.sh DATA_FOLDER [SEG_LEN]        e.g.  ./run_checks.sh data_11_17 16
#
# Everything goes into results_<DATA_FOLDER>_<SEG_LEN>/ : the offset checks, one training log and one
# evaluation log per model, the models themselves, and summary.txt with the final scores side by side.
# The original data folder is never changed (transformed copies are written next to it).
#
# Tags left out everywhere: DROP_TAGS below (0 and 20: unusable reference tags; 7: missed ~40% of the time,
# its long runs of "no change" would leak when it dropped out). Edit to suit the data set.
set -euo pipefail

DATA="${1:?usage: ./run_checks.sh DATA_FOLDER [SEG_LEN]}"
DATA="${DATA%/}"
SEG="${2:-16}"
DROP_TAGS="0,7,20"
EXCLUDE="0,7,20"          # tags left out of the array / panel averages in cmdiff / cmonly
PANEL_GROUPS="1-9:10-18"        # the two whisker panels: right = 1-9, left = 10-18 (for the *_panel models)
OUT="results_$(basename "$DATA")_${SEG}"
COMMON="--test angle --target angle_to_head --inputs tags --hidden 128 --layers 2 --dropout 0.2 \
        --in_dropout 0.1 --epochs 80 --patience 15 --seg_len $SEG"

[ -d "$DATA" ] || { echo "no folder $DATA"; exit 1; }
N=$(find "$DATA" -name '*.csv' | wc -l)
[ "$N" -gt 0 ] || { echo "no CSV files in $DATA"; exit 1; }
mkdir -p "$OUT"
echo "== $N recordings in $DATA, ${SEG}-frame windows, results in $OUT/"

echo "== 1. offset checks (resting position and jitter per tag)"
python offset_check.py "$DATA" --ref-tags "$EXCLUDE" --top 20               > "$OUT/offset_mean.txt"
python offset_check.py "$DATA" --ref-tags "$EXCLUDE" --top 20 --stat jitter > "$OUT/offset_jitter.txt"

echo "== 2. transformed copies"
make_copy () {   # suffix, mode, extra args
    local DST="${DATA}_$1" MODE="$2"; shift 2
    if [ -d "$DST" ]; then
        echo "   $DST exists, reusing it (delete it to rebuild)"
    else
        python transform_tags.py "$DATA" "$DST" --mode "$MODE" --no-seen --rotvec --exclude "$EXCLUDE" "$@"
    fi
}
make_copy diff         diff
make_copy cmdiff       cmdiff
make_copy cmonly       cmonly
make_copy cmdiff_panel cmdiff --groups "$PANEL_GROUPS"
make_copy cmonly_panel cmonly --groups "$PANEL_GROUPS"

echo "== 3. training + evaluation"
train_eval () {   # name, data folder, extra args
    local NAME="$1" DIR="$2"; shift 2
    if [ -s "$OUT/${NAME}_eval.txt" ] && grep -q "=== test" "$OUT/${NAME}_eval.txt"; then
        echo "   $NAME: already done, skipping (delete $OUT/${NAME}_eval.txt to redo)"
        return
    fi
    echo "   $NAME"
    python train_lstm_flow.py $COMMON --data_dir "$DIR" "$@" --out "$OUT/$NAME.pt" > "$OUT/${NAME}_train.txt" 2>&1
    python evaluate.py --checkpoint "$OUT/$NAME.pt" --split val test         > "$OUT/${NAME}_eval.txt" 2>&1
}
train_eval diff_whiskers "${DATA}_diff"   --drop_tags "$DROP_TAGS"   # all whisker motion
train_eval cmdiff        "${DATA}_cmdiff" --drop_tags "$DROP_TAGS"   # motion relative to the rest of the array
train_eval cmonly        "${DATA}_cmonly"                            # only the motion the whole array shares
train_eval cmdiff_panel  "${DATA}_cmdiff_panel" --drop_tags "$DROP_TAGS"   # relative to its own panel
train_eval cmonly_panel  "${DATA}_cmonly_panel"                      # each panel's own motion (2 made-up tags)

echo "== 4. summary"
{
    echo "Data: $DATA ($N recordings), ${SEG}-frame windows, tags dropped: $DROP_TAGS"
    echo
    printf "%-15s %-28s %-28s\n" model "val (acc / balanced)" "test (acc / balanced)"
    for NAME in diff_whiskers cmdiff cmdiff_panel cmonly_panel cmonly; do
        V=$(awk '/=== val/{f=1} f && /angle_to_head: accuracy/{print $3" / "$5; exit}' "$OUT/${NAME}_eval.txt" | tr -d ',')
        T=$(awk '/=== test/{f=1} f && /angle_to_head: accuracy/{print $3" / "$5; exit}' "$OUT/${NAME}_eval.txt" | tr -d ',')
        printf "%-15s %-28s %-28s\n" "$NAME" "${V:-?}" "${T:-?}"
    done
    echo
    grep -h "^Whisker tag/axis pairs\|^Reference tag max F" "$OUT/offset_mean.txt" | sed 's/^/resting position: /'
    grep -h "^Whisker tag/axis pairs\|^Reference tag max F" "$OUT/offset_jitter.txt" | sed 's/^/jitter:           /'
} | tee "$OUT/summary.txt"
