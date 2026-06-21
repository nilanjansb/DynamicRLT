#!/usr/bin/env python3
"""
Evaluate one or more TREC run files against standard qrels.

Usage examples
--------------
# Auto-detect dataset from filename, print all metrics:
  python evaluateTrec.py path/to/run.txt

# Multiple runs side-by-side:
  python evaluateTrec.py runs/*.txt

# Override dataset (when filename doesn't contain a known name):
  python evaluateTrec.py run.txt --dataset scifact

# Custom qrels file (any dataset not in the 7 defaults):
  python evaluateTrec.py run.txt --qrels /path/to/qrels.txt --qrels-format trec

# Show only a subset of metrics:
  python evaluateTrec.py run.txt --metrics ndcg_cut_10 map recip_rank

# Save results to JSON:
  python evaluateTrec.py runs/*.txt --json results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytrec_eval

BASE     = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parents[1]))
DATA_DIR = BASE / "1_Download_dataset/data"

# ─────────────────────────────────────────────────────────────────────────────
# Known datasets
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_DATASETS: Dict[str, Tuple[Path, str]] = {
    "msmarco-dl19":     (DATA_DIR / "msmarco/qrels.dl19-passage.txt",      "trec"),
    "msmarco-dl20":     (DATA_DIR / "msmarco/qrels.dl20-passage.txt",      "trec"),
    "scifact":          (DATA_DIR / "beir/scifact/qrels/test.tsv",          "beir_tsv"),
    "trec-covid":       (DATA_DIR / "beir/trec-covid/qrels/test.tsv",       "beir_tsv"),
    "webis-touche2020": (DATA_DIR / "beir/webis-touche2020/qrels/test.tsv", "beir_tsv"),
    "fever":            (DATA_DIR / "beir/fever/qrels/test.tsv",            "beir_tsv"),
    "dbpedia-entity":   (DATA_DIR / "beir/dbpedia-entity/qrels/test.tsv",   "beir_tsv"),
}

# ─────────────────────────────────────────────────────────────────────────────
# All supported metrics (grouped for display)
# ─────────────────────────────────────────────────────────────────────────────

ALL_METRICS: List[str] = [
    # nDCG
    "ndcg_cut_5",
    "ndcg_cut_10",
    "ndcg_cut_20",
    "ndcg_cut_100",
    # MAP
    "map",
    "map_cut_10",
    "map_cut_100",
    # MRR
    "recip_rank",
    # Recall
    "recall_5",
    "recall_10",
    "recall_20",
    "recall_100",
    "recall_1000",
    # Precision
    "P_5",
    "P_10",
    "P_20",
    "P_100",
    # R-precision
    "Rprec",
    # Binary preference
    "bpref",
    # Set-based
    "set_recall",
    "set_P",
    "set_F",
    # Geometric MAP
    "gm_map",
    # Counts (diagnostic)
    "num_rel",
    "num_ret",
    "num_rel_ret",
]

METRIC_DISPLAY: Dict[str, str] = {
    "ndcg_cut_5":   "nDCG@5",
    "ndcg_cut_10":  "nDCG@10",
    "ndcg_cut_20":  "nDCG@20",
    "ndcg_cut_100": "nDCG@100",
    "map":          "MAP",
    "map_cut_10":   "MAP@10",
    "map_cut_100":  "MAP@100",
    "recip_rank":   "MRR",
    "recall_5":     "Recall@5",
    "recall_10":    "Recall@10",
    "recall_20":    "Recall@20",
    "recall_100":   "Recall@100",
    "recall_1000":  "Recall@1000",
    "P_5":          "P@5",
    "P_10":         "P@10",
    "P_20":         "P@20",
    "P_100":        "P@100",
    "Rprec":        "R-Prec",
    "bpref":        "bpref",
    "set_recall":   "Set-Recall",
    "set_P":        "Set-P",
    "set_F":        "Set-F1",
    "gm_map":       "GM-MAP",
    "num_rel":      "#Relevant",
    "num_ret":      "#Retrieved",
    "num_rel_ret":  "#Rel-Ret",
}

COUNT_METRICS = {"num_rel", "num_ret", "num_rel_ret"}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_qrels(qrels_file: Path, fmt: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if fmt == "beir_tsv" and i == 0:
                continue  # skip header
            if fmt == "trec":
                parts = line.split()
                qid, docid, rel = parts[0], parts[2], int(parts[3])
            else:  # beir_tsv
                parts = line.split("\t")
                qid, docid, rel = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[docid] = rel
    return qrels


def load_run(run_file: Path) -> Dict[str, Dict[str, float]]:
    run: Dict[str, Dict[str, float]] = {}
    with open(run_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, _, score = parts[0], parts[1], parts[2], parts[3], parts[4]
            run.setdefault(qid, {})[docid] = float(score)
    return run


def detect_dataset(run_file: Path) -> Optional[str]:
    name = run_file.name
    # Longest match first to avoid 'trec' matching 'trec-covid' partially
    for ds in sorted(KNOWN_DATASETS, key=len, reverse=True):
        if ds in name:
            return ds
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    run_file: Path,
    qrels: Dict[str, Dict[str, int]],
    metrics: List[str],
) -> Dict[str, float]:
    run = load_run(run_file)
    # Restrict to queries present in qrels
    run = {qid: docs for qid, docs in run.items() if qid in qrels}

    if not run:
        return {}

    ev      = pytrec_eval.RelevanceEvaluator(qrels, set(metrics))
    results = ev.evaluate(run)

    n = len(results)
    aggregated: Dict[str, float] = {}
    for m in metrics:
        values = [results[qid][m] for qid in results if m in results[qid]]
        aggregated[m] = sum(values) / len(values) if values else 0.0

    aggregated["__n_queries__"] = float(n)
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def fmt_value(metric: str, value: float) -> str:
    if metric in COUNT_METRICS:
        return f"{value:>10.1f}"
    return f"{value:>10.4f}"


def print_single(run_file: Path, scores: Dict[str, float], metrics: List[str]) -> None:
    n = int(scores.get("__n_queries__", 0))
    print(f"\n{'─' * 55}")
    print(f"  Run    : {run_file.name}")
    print(f"  Queries: {n:,}")
    print(f"{'─' * 55}")
    print(f"  {'Metric':<22} {'Value':>10}")
    print(f"  {'─'*22} {'─'*10}")
    for m in metrics:
        label = METRIC_DISPLAY.get(m, m)
        val   = scores.get(m, float("nan"))
        print(f"  {label:<22} {fmt_value(m, val)}")
    print()


def print_table(run_files: List[Path], all_scores: List[Dict[str, float]], metrics: List[str]) -> None:
    col_w = max(18, max(len(f.name) for f in run_files) + 2)
    header = f"{'Metric':<22}" + "".join(f"{f.name:>{col_w}}" for f in run_files)
    print(f"\n{'─' * len(header)}")
    print(header)
    print("─" * len(header))
    for m in metrics:
        label = METRIC_DISPLAY.get(m, m)
        row   = f"{label:<22}"
        for scores in all_scores:
            val = scores.get(m, float("nan"))
            row += fmt_value(m, val).rjust(col_w)
        print(row)
    # Query counts
    print("─" * len(header))
    row = f"{'#Queries':<22}"
    for scores in all_scores:
        row += f"{int(scores.get('__n_queries__', 0)):>{col_w},}"
    print(row)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate TREC run file(s) with pytrec_eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "runs", nargs="+", type=Path,
        help="One or more TREC run files to evaluate.",
    )
    p.add_argument(
        "--dataset", "-d",
        choices=list(KNOWN_DATASETS.keys()),
        help="Dataset name. Auto-detected from filename if not given.",
    )
    p.add_argument(
        "--qrels", type=Path,
        help="Path to a custom qrels file (overrides --dataset lookup).",
    )
    p.add_argument(
        "--qrels-format", choices=["trec", "beir_tsv"], default="trec",
        help="Format of the custom qrels file (default: trec).",
    )
    p.add_argument(
        "--metrics", nargs="+", default=None,
        metavar="METRIC",
        help=(
            "Metrics to compute. Defaults to all supported metrics. "
            f"Available: {', '.join(ALL_METRICS)}"
        ),
    )
    p.add_argument(
        "--per-query", action="store_true",
        help="Also print per-query breakdown for each run.",
    )
    p.add_argument(
        "--json", type=Path, default=None, metavar="FILE",
        help="Save all results to a JSON file.",
    )
    p.add_argument(
        "--relevance-level", type=int, default=1,
        help="Minimum relevance level counted as relevant (default: 1).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    metrics = args.metrics if args.metrics else ALL_METRICS

    # Validate custom metric names
    invalid = [m for m in metrics if m not in ALL_METRICS]
    if invalid:
        print(f"[ERROR] Unknown metric(s): {invalid}", file=sys.stderr)
        print(f"        Valid: {ALL_METRICS}", file=sys.stderr)
        sys.exit(1)

    all_scores: List[Dict[str, float]] = []
    json_out:   List[dict]             = []

    for run_file in args.runs:
        if not run_file.exists():
            print(f"[ERROR] Run file not found: {run_file}", file=sys.stderr)
            continue

        # Resolve qrels
        if args.qrels:
            qrels_path = args.qrels
            qrels_fmt  = args.qrels_format
            dataset    = args.qrels.stem
        else:
            ds = args.dataset or detect_dataset(run_file)
            if ds is None:
                print(
                    f"[ERROR] Cannot detect dataset for '{run_file.name}'. "
                    f"Pass --dataset or --qrels.",
                    file=sys.stderr,
                )
                continue
            if ds not in KNOWN_DATASETS:
                print(f"[ERROR] Unknown dataset '{ds}'.", file=sys.stderr)
                continue
            qrels_path, qrels_fmt = KNOWN_DATASETS[ds]
            dataset = ds

        if not qrels_path.exists():
            print(f"[ERROR] Qrels file not found: {qrels_path}", file=sys.stderr)
            continue

        qrels  = load_qrels(qrels_path, qrels_fmt)
        scores = evaluate(run_file, qrels, metrics)

        if not scores:
            print(f"[WARN] No overlapping queries for '{run_file.name}' — skipping.")
            continue

        all_scores.append(scores)

        if args.per_query:
            run     = load_run(run_file)
            run     = {qid: docs for qid, docs in run.items() if qid in qrels}
            ev      = pytrec_eval.RelevanceEvaluator(
                qrels, set(metrics), relevance_level=args.relevance_level
            )
            per_q   = ev.evaluate(run)
            print(f"\nPer-query results for: {run_file.name}")
            pq_metrics = [m for m in metrics if m not in COUNT_METRICS]
            header = f"{'QueryID':<20}" + "".join(
                f"{METRIC_DISPLAY.get(m, m):>12}" for m in pq_metrics
            )
            print(header)
            print("─" * len(header))
            for qid in sorted(per_q):
                row = f"{qid:<20}"
                for m in pq_metrics:
                    row += f"{per_q[qid].get(m, 0.0):>12.4f}"
                print(row)

        if args.json is not None:
            entry = {
                "run_file": str(run_file),
                "dataset":  dataset,
                "n_queries": int(scores.get("__n_queries__", 0)),
                "scores":   {m: round(scores[m], 6) for m in metrics if m in scores},
            }
            json_out.append(entry)

    if not all_scores:
        print("[ERROR] No results to display.", file=sys.stderr)
        sys.exit(1)

    if len(args.runs) == 1:
        print_single(args.runs[0], all_scores[0], metrics)
    else:
        valid_files  = [f for f, s in zip(args.runs, all_scores) if s]
        print_table(valid_files, all_scores, metrics)

    if args.json and json_out:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(json_out, f, indent=2)
        print(f"Results saved to: {args.json}")


if __name__ == "__main__":
    main()
