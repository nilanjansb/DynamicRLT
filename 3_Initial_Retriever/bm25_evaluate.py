#!/usr/bin/env python3
"""Compute nDCG@10 for BM25 top-100 runs using pytrec_eval."""

import os
from pathlib import Path
import pytrec_eval

BASE = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parents[1]))
DATA_DIR = BASE / "1_Download_dataset/data"
BM25_DIR = BASE / "3_Initial_Retriever/bm25"

QRELS_FILES = {
    "msmarco-dl19":    (DATA_DIR / "msmarco/qrels.dl19-passage.txt",     "trec"),
    "msmarco-dl20":    (DATA_DIR / "msmarco/qrels.dl20-passage.txt",     "trec"),
    "scifact":         (DATA_DIR / "beir/scifact/qrels/test.tsv",         "beir_tsv"),
    "trec-covid":      (DATA_DIR / "beir/trec-covid/qrels/test.tsv",      "beir_tsv"),
    "webis-touche2020":(DATA_DIR / "beir/webis-touche2020/qrels/test.tsv","beir_tsv"),
    "fever":           (DATA_DIR / "beir/fever/qrels/test.tsv",           "beir_tsv"),
    "dbpedia-entity":  (DATA_DIR / "beir/dbpedia-entity/qrels/test.tsv",  "beir_tsv"),
}


def load_qrels(qrels_file, fmt):
    """Load qrels as {qid: {docid: rel}}."""
    qrels = {}
    with open(qrels_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if fmt == "beir_tsv" and i == 0:
                continue
            if fmt == "trec":
                parts = line.split()
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            elif fmt == "beir_tsv":
                parts = line.split("\t")
                qid, docid, rel = parts[0], parts[1], int(parts[2])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = rel
    return qrels


def load_run(run_file):
    """Load TREC run as {qid: {docid: score}}."""
    run = {}
    with open(run_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, _, score, *_ = parts[:6]
            if qid not in run:
                run[qid] = {}
            run[qid][docid] = float(score)
    return run


def evaluate_run(run_file, qrels_file, fmt, metric="ndcg_cut_10"):
    """Evaluate a TREC run file against qrels; return (mean_score, num_queries)."""
    qrels = load_qrels(qrels_file, fmt)
    run = load_run(run_file)

    # Filter run to only queries in qrels
    run = {q: docs for q, docs in run.items() if q in qrels}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric})
    results = evaluator.evaluate(run)

    scores = [results[q][metric] for q in results]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, len(scores)


def main():
    print(f"\n{'Dataset':<25} {'nDCG@10 (top-100)':>20}  {'#Queries':>10}")
    print("-" * 60)

    for name, (qrels_file, fmt) in QRELS_FILES.items():
        run_file = BM25_DIR / f"{name}.top100.txt"
        if not run_file.exists():
            print(f"{name:<25} {'MISSING':>20}")
            continue
        score, n_q = evaluate_run(run_file, qrels_file, fmt)
        print(f"{name:<25} {score:>20.4f}  {n_q:>10}")

    print()


if __name__ == "__main__":
    main()
