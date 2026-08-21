"""Builds data/warehouse_sample.jsonl, a small stratified sample of the
real dataset used as the default fast/low-cost input for the rest of
the pipeline.

Takes a fixed number of rows per (warehouse_level x inventory_issue
value) combination instead of a blind random sample, so the small
sample still covers all real issue types across all warehouse levels -
a plain random sample of the same size could miss some of the rarer
issue codes entirely, which would silently weaken the
no_invented_reason rubric check (gen_pairs.py) for anything it never
saw an example of.

Usage: python build_sample.py [--data-dir data] [--rows-per-issue 6] [--out data/warehouse_sample.jsonl]
"""

import argparse
import csv
import json
import os

import data_utils as du


def build_sample(data_dir="data", rows_per_issue=6):
    """Reads the 3 real source CSVs and returns rows_per_issue rows for
    every (warehouse_level, inventory_issue) combination found."""
    rows = []
    for level, fname in du.CSV_FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"expected {path} - this needs the real source CSVs, not the "
                f"sample file itself. place them in {data_dir}/ first."
            )

        by_issue = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                by_issue.setdefault(row["inventory_issue"], []).append(row)

        for issue_val, issue_rows in by_issue.items():
            for row in issue_rows[:rows_per_issue]:
                rows.append(du._cast_row(row))

    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="folder with the 3 real CSVs")
    ap.add_argument("--rows-per-issue", type=int, default=6,
                     help="rows to take per (warehouse level, issue type) combination")
    ap.add_argument("--out", default="data/warehouse_sample.jsonl")
    args = ap.parse_args()

    sample = build_sample(args.data_dir, args.rows_per_issue)

    levels = {r["warehouse_level"] for r in sample}
    issues = {r["inventory_issue"] for r in sample}
    print(f"{len(sample)} rows, {len(levels)} warehouse level(s), {len(issues)} issue type(s) covered")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")
