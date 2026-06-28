#!/usr/bin/env python3
"""
BM25-score generated pivot documents — Step 2 of Gen-PART.

Gen-PART (Generative PARTitioning) needs a query-specific relevance threshold:
the BM25 score the first-stage retriever assigns to a synthetically generated,
"somewhat relevant" pivot document D* for each query Q.  This score, θ(Q, D*),
is later inserted into the BM25 top-100 list to find the cutoff rank p — the
point below which documents are assumed irrelevant and bypassed by the
expensive monoT5/duoT5 reranker (see genpart.py).

This script computes θ(Q, D*) for every generated pivot, on the *exact same
scale* as the BM25 run files in 3_Initial_Retriever/bm25/, and writes them to
9_GenPART/pivotbm25scores/.

Pivot types (all supported; mirrors SNOW / GenTDPart layout)
------------------------------------------------------------
  separate     (default, a.k.a. "normal") — pivot_docs/{dataset}.json
                 one pivot per tau level (0..3); scores written for every tau.
                 → pivotbm25scores/{dataset}.json
  tautogether  — pivot_docs/tautogether/{dataset}.json
                 jointly-generated tau levels (0..3); scores for every tau.
                 → pivotbm25scores/tautogether/{dataset}.json
  single       — pivot_docs/single/{dataset}.json
                 one pivot per query, no tau levels.
                 → pivotbm25scores/single/{dataset}.json

All tau levels found in a file are scored — nothing is dropped — so the saved
JSON can be reused by genpart.py for any tau (default downstream tau is 2).

Exact Lucene-10 BM25 reproduction
---------------------------------
The pivot is not in the Lucene index, so its score is computed manually from
index statistics, validated to reproduce castorini's run-file scores to within
~1e-3 % per document:

    score(Q, D*) = Σ_t  qtf(t) · idf(t) · tf(t,D*)
                        ─────────────────────────────────────────
                         tf(t,D*) + k1·(1 − b + b·|D*|/avgdl)

    idf(t) = ln( 1 + (N − df(t) + 0.5) / (df(t) + 0.5) )      (Lucene BM25 idf)
    k1 = 0.9,  b = 0.4                                        (BEIR defaults)

Three details are required to match Lucene exactly:
  • NO (k1+1) numerator factor — the Lucene104 codec is Lucene 10, which
    dropped it (LUCENE-8563) as a rank-invariant constant.
  • |D*| (the pivot field length) is quantized with Lucene's SmallFloat byte4
    norm encoding, exactly as a real indexed document would be.
  • Each query term is weighted by qtf(t), its multiplicity in the analyzed
    query (Lucene scores repeated query terms once per occurrence).

df(t), N and avgdl all come from the real corpus index, so the pivot score
lands on the same scale as the BM25 scores of the real documents.

Output naming:
  pivotbm25scores/{dataset}.json    (covers all tau levels of the pivot)

Datasets: every dataset with a pivot file *except FEVER* (its index is large
and not needed for the Gen-PART experiments).

Usage (JAVA_HOME must point at the bundled JDK so pyserini can boot the JVM):
    export JAVA_HOME=/DATA/cs26int00020/tools/jdk-21.0.5+11
    python3 9_GenPART/bm25pivots.py                          # separate, all datasets
    python3 9_GenPART/bm25pivots.py --dataset scifact trec-covid
    python3 9_GenPART/bm25pivots.py --pivot-type tautogether
    python3 9_GenPART/bm25pivots.py --pivot-type single
    python3 9_GenPART/bm25pivots.py --pivot-type all --force  # every type, overwrite
"""

import os
import sys
import json
import time
import math
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

BASE      = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parents[1]))
INDEX_DIR = BASE / "BM25_indexes"
PIVOT_DIR = BASE / "2_Pivot_generation/pivot_docs"
OUT_DIR   = BASE / "9_GenPART/pivotbm25scores"

BM25_K1 = 0.9
BM25_B  = 0.4

# dataset -> Lucene index subdir.  FEVER is intentionally excluded.
# msmarco-dl19 and msmarco-dl20 share the single msmarco passage index.
DATASET_INDEX = {
    "msmarco-dl19":     INDEX_DIR / "msmarco",
    "msmarco-dl20":     INDEX_DIR / "msmarco",
    "scifact":          INDEX_DIR / "scifact",
    "trec-covid":       INDEX_DIR / "trec-covid",
    "webis-touche2020": INDEX_DIR / "webis-touche2020",
    "dbpedia-entity":   INDEX_DIR / "dbpedia-entity",
}


# ─────────────────────────────────────────────────────────────────────────────
# Pivot types — directory, loader, and output routing for each mode
# ─────────────────────────────────────────────────────────────────────────────

PIVOT_TYPES = ("separate", "tautogether", "single")


def pivot_input_dir(pivot_type: str) -> Path:
    if pivot_type == "separate":
        return PIVOT_DIR
    return PIVOT_DIR / pivot_type            # tautogether/ or single/


def pivot_output_dir(pivot_type: str) -> Path:
    if pivot_type == "separate":
        return OUT_DIR
    return OUT_DIR / pivot_type


def load_pivots(pivot_file: Path, pivot_type: str):
    """
    Parse a pivot file for the given type.

    Returns (tau_levels, entries) where:
      • tau_levels is the list of tau-key strings present (["single"] for the
        single-pivot mode, which has no tau dimension).
      • entries is a list of {"qid", "query", "pivots": {tau_key: text}}.
        For the single mode the one pivot is stored under the key "single".
    """
    data = json.loads(pivot_file.read_text())

    if pivot_type == "single":
        tau_levels = ["single"]
        entries = [{
            "qid":    str(e["qid"]),
            "query":  e.get("query", ""),
            "pivots": {"single": e.get("pivot", "")},
        } for e in data.get("queries", [])]
        return tau_levels, entries

    # separate / tautogether share the tau-keyed structure
    tau_levels = [str(t) for t in data.get("tau_levels", [0, 1, 2, 3])]
    entries = [{
        "qid":    str(e["qid"]),
        "query":  e.get("query", ""),
        "pivots": {str(k): v for k, v in e.get("pivots", {}).items()},
    } for e in data.get("queries", [])]
    return tau_levels, entries


# ─────────────────────────────────────────────────────────────────────────────
# BM25 pivot scorer
# ─────────────────────────────────────────────────────────────────────────────

class PivotBM25Scorer:
    """Reproduces the Lucene-10 BM25 score of arbitrary text against a query,
    on the same scale as the dataset's BM25 run files."""

    def __init__(self, index_path: Path, k1: float = BM25_K1, b: float = BM25_B):
        # Import order matters: importing pyserini boots the JVM with the
        # anserini fatjar on the classpath, which is what makes SmallFloat
        # resolvable via jnius below.
        from pyserini.index.lucene import LuceneIndexReader
        from pyserini.analysis import Analyzer, get_lucene_analyzer
        from jnius import autoclass

        self.ir = LuceneIndexReader(str(index_path))
        self.an = Analyzer(get_lucene_analyzer())
        self._SmallFloat = autoclass("org.apache.lucene.util.SmallFloat")

        stats       = self.ir.stats()
        self.N      = stats["documents"]
        self.avgdl  = stats["total_terms"] / stats["documents"]
        self.k1     = k1
        self.b      = b
        self._df_cache: Dict[str, int] = {}

    # — helpers ————————————————————————————————————————————————————————————————
    def analyze(self, text: str) -> List[str]:
        return self.an.analyze(text)

    def _df(self, term: str) -> int:
        """Document frequency of an already-analyzed term (cached)."""
        df = self._df_cache.get(term)
        if df is None:
            # analyzer=None: the term is fed verbatim (it is already stemmed).
            df = self.ir.get_term_counts(term, analyzer=None)[0]
            self._df_cache[term] = df
        return df

    def _idf(self, df: int) -> float:
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def _quantized_len(self, raw_len: int) -> int:
        """Lucene's lossy SmallFloat byte4 norm encode→decode of a field length."""
        if raw_len <= 0:
            return 0
        return self._SmallFloat.byte4ToInt(self._SmallFloat.intToByte4(raw_len))

    # — scoring ————————————————————————————————————————————————————————————————
    def score(self, query_terms: Counter, doc_text: str):
        """
        BM25 score of doc_text against an already-analyzed query (Counter of
        {term: qtf}).  Returns (score, raw_doc_len, quantized_doc_len).
        """
        doc_terms = self.an.analyze(doc_text)
        raw_len   = len(doc_terms)
        dl        = self._quantized_len(raw_len)
        if dl == 0:
            return 0.0, raw_len, dl

        tf = Counter(doc_terms)
        denom_const = self.k1 * (1 - self.b + self.b * dl / self.avgdl)

        score = 0.0
        for term, qtf in query_terms.items():
            f = tf.get(term, 0)
            if not f:
                continue
            score += qtf * self._idf(self._df(term)) * f / (f + denom_const)
        return score, raw_len, dl


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset driver
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(dataset_name: str, pivot_type: str, force: bool) -> None:
    index_path = DATASET_INDEX[dataset_name]
    in_dir     = pivot_input_dir(pivot_type)
    out_dir    = pivot_output_dir(pivot_type)
    pivot_file = in_dir / f"{dataset_name}.json"
    out_file   = out_dir / f"{dataset_name}.json"

    log.info("")
    log.info("─" * 65)
    log.info(f"Dataset : {dataset_name}   Pivot type : {pivot_type}")
    log.info("─" * 65)

    if not index_path.exists():
        log.warning(f"  Index not found: {index_path} — skipping.")
        return
    if not pivot_file.exists():
        log.warning(f"  Pivot file not found: {pivot_file} — skipping.")
        return
    if out_file.exists() and not force:
        log.info(f"  Output exists (--force to overwrite): {out_file}")
        return

    tau_levels, entries = load_pivots(pivot_file, pivot_type)
    log.info(f"  Pivots: {len(entries):,} queries, tau levels {tau_levels}")

    t_load = time.time()
    scorer = PivotBM25Scorer(index_path)
    log.info(
        f"  Index loaded in {time.time() - t_load:.1f}s  "
        f"(N={scorer.N:,}  avgdl={scorer.avgdl:.2f})"
    )

    t0          = time.time()
    out_queries = []
    n_scored    = 0

    for e in entries:
        qid         = e["qid"]
        query_terms = Counter(scorer.analyze(e["query"]))

        scores: Dict[str, float] = {}
        lengths: Dict[str, int]  = {}
        for tau in tau_levels:
            text = e["pivots"].get(tau, "").strip()
            if not text:
                continue
            s, _raw, dl = scorer.score(query_terms, text)
            scores[tau]  = round(s, 6)
            lengths[tau] = dl

        out_queries.append({
            "qid":            qid,
            "query":          e["query"],
            "scores":         scores,       # {tau: BM25 score of that pivot}
            "pivot_doc_len":  lengths,      # {tau: quantized token length}
        })
        n_scored += 1
        if n_scored % 500 == 0:
            log.info(f"    … {n_scored:,}/{len(entries):,} queries scored")

    elapsed = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset":    dataset_name,
        "pivot_type": pivot_type,
        "index":      str(index_path),
        "retriever":  "bm25",
        "bm25":       {"k1": scorer.k1, "b": scorer.b},
        "N":          scorer.N,
        "avgdl":      round(scorer.avgdl, 6),
        # int tau levels for tau-keyed modes; ["single"] for the single mode
        "tau_levels": ([int(t) for t in tau_levels]
                       if pivot_type != "single" else ["single"]),
        "n_queries":  len(out_queries),
        "queries":    out_queries,
    }
    out_file.write_text(json.dumps(payload, indent=2))

    log.info(
        f"  Scored {n_scored:,} queries × {len(tau_levels)} tau levels "
        f"in {elapsed:.2f}s ({elapsed / max(n_scored,1) * 1000:.2f} ms/query)"
    )
    log.info(f"  Wrote → {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BM25-score Gen-PART pivot documents (all datasets except FEVER).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", nargs="+", default=["all"],
        help=(
            "Dataset(s) to process. Pass one name, several names, or 'all'. "
            f"Choices: {list(DATASET_INDEX.keys()) + ['all']}"
        ),
    )
    p.add_argument(
        "--pivot-type", default="separate",
        choices=list(PIVOT_TYPES) + ["all"],
        help=(
            "Pivot source: 'separate' (default, the normal per-tau pivots), "
            "'tautogether' (jointly-generated tau levels), 'single' (one pivot "
            "per query, no tau), or 'all' to score every type."
        ),
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    raw = args.dataset
    if raw == ["all"] or raw == "all":
        datasets_to_run = list(DATASET_INDEX.keys())
    else:
        datasets_to_run = raw if isinstance(raw, list) else [raw]
        unknown = [d for d in datasets_to_run if d not in DATASET_INDEX]
        if unknown:
            log.error(
                f"Unknown/unsupported dataset(s): {unknown}. "
                f"Valid: {list(DATASET_INDEX.keys())} (FEVER is intentionally excluded)."
            )
            sys.exit(1)

    if not os.environ.get("JAVA_HOME"):
        log.warning(
            "JAVA_HOME is not set — pyserini may fail to boot the JVM. "
            "export JAVA_HOME=/DATA/cs26int00020/tools/jdk-21.0.5+11"
        )

    types_to_run = list(PIVOT_TYPES) if args.pivot_type == "all" else [args.pivot_type]

    log.info("=" * 65)
    log.info("Gen-PART — BM25 pivot scoring")
    log.info(f"BM25 params : k1={BM25_K1}  b={BM25_B}")
    log.info(f"Datasets    : {datasets_to_run}")
    log.info(f"Pivot types : {types_to_run}")
    log.info(f"Output root : {OUT_DIR}")
    log.info("=" * 65)

    for pivot_type in types_to_run:
        for dataset_name in datasets_to_run:
            process_dataset(dataset_name, pivot_type, args.force)

    log.info("")
    log.info("All done.")


if __name__ == "__main__":
    main()
