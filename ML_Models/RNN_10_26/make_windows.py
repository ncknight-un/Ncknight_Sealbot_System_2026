"""
Cut a CSV into overlapping sub-CSVs with a running window.

Each output file holds --length consecutive rows (default 65, ~1 s at 65 fps),
and the window moves down --window_size rows (default 16) between files.
Every output keeps the original header and columns.

Usage:
    python make_windows.py recording.csv
    python make_windows.py recording.csv --window_size 16 --length 65 --out_dir windows/
"""

import argparse
import os

import pandas as pd


def make_windows(csv_path, out_dir=None, window_size=16, length=65):
    df = pd.read_csv(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), f"{base}_windows")
    os.makedirs(out_dir, exist_ok=True)

    if len(df) < length:
        print(f"{csv_path} has only {len(df)} rows (< {length}); no windows written.")
        return 0

    starts = range(0, len(df) - length + 1, window_size)
    for i, start in enumerate(starts):
        name = f"{base}_w{i:04d}_rows{start}-{start + length - 1}.csv"
        df.iloc[start:start + length].to_csv(os.path.join(out_dir, name), index=False)

    last_end = starts[-1] + length
    print(f"{len(df)} rows -> {len(starts)} files of {length} rows "
          f"(step {window_size}) in {out_dir}")
    if last_end < len(df):
        print(f"Last {len(df) - last_end} row(s) not covered by a full window.")
    return len(starts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="input CSV file")
    p.add_argument("--window_size", type=int, default=16, help="rows to move down between windows (default 16)")
    p.add_argument("--length", type=int, default=65, help="rows per output file (default 65)")
    p.add_argument("--out_dir", help="output folder (default: <csv name>_windows next to the input)")
    args = p.parse_args()
    if args.window_size < 1 or args.length < 1:
        p.error("--window_size and --length must be at least 1")
    make_windows(args.csv, args.out_dir, args.window_size, args.length)


if __name__ == "__main__":
    main()
