#!/usr/bin/env python3
"""
Gen-PART — Generative PARTitioning for Ranked List Truncation.
  Pointwise reranker : castorini/monot5-3b-msmarco-10k
  Pairwise  reranker : castorini/duot5-3b-msmarco   (optional, --monot5duot5)

Gen-PART is the cheap-cutoff counterpart of the Fixed-100 baseline in
fixedRLT.py.  Instead of sending all 100 BM25 candidates to the (expensive)
monoT5/duoT5 rerankers, it uses a per-query relevance threshold — the BM25
score of a synthetically generated "somewhat relevant" pivot document D* — to
decide how many candidates are worth reranking.

Algorithm  (per query Q, on the BM25 top-100 list)
---------------------------------------------------
  1. θ(Q, D*)  : the pivot's BM25 score, precomputed by bm25pivots.py and read
                 from 9_GenPART/pivotbm25scores/.
  2. Cutoff p  : insert D* into the BM25-sorted list by score; p = number of
                 documents scoring ≥ θ(Q, D*).
                     Score(D_p) ≥ θ(Q, D*) ≥ Score(D_{p+1})
  3. Partition : D+ = D_1..D_p  (promising)   D- = D_{p+1}..D_m  (remainder)
  4. Rerank D+ : monoT5 scores ONLY the p promising docs and reorders them.
  5. Assemble  : reranked(D+) ++ D-  (D- kept in original BM25 order).

  Optional duoT5 stage (--monot5duot5 K): pairwise-rerank only the top
  min(K, p) of the monoT5-reranked promising list — so duoT5, too, sees far
  fewer docs than the Fixed-100 baseline.  Assembly:
      duoT5(top min(K,p) of D+) ++ rest of D+ ++ D-

Because only D+ reaches the rerankers, the number of (query, doc) monoT5
inferences and duoT5 pair inferences drops with the cutoff — the RLT speedup.
Effectiveness is preserved when the bypassed D- docs are truly irrelevant.

All heavy lifting (monoT5/duoT5 prompt building + GPU scoring + duoT5
aggregation) is reused verbatim from fixedRLT.py — this file only adds the
pivot-based partitioning and the stitch-back assembly.

Output naming (mirrors SNOW's pivot-type tags):
  separate    : {ds}.{retriever}.monot5_3b.genpart_tau{tau}.top{K}.txt
  tautogether : {ds}.{retriever}.monot5_3b.genpart_tautogether_tau{tau}.top{K}.txt
  single      : {ds}.{retriever}.monot5_3b.genpart_single.top{K}.txt
  + duoT5      : …monot5_3b_duot5_3b.genpart_<tag>_duo{duoK}.top{K}.txt

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 9_GenPART/genpart.py \\
        --dataset scifact --retriever bm25 --pivot-tau 2

    CUDA_VISIBLE_DEVICES=0 python3 9_GenPART/genpart.py \\
        --dataset all --pivot-type separate --pivot-tau 2 --monot5duot5 50
"""

import os
import sys
import json
import time
import datetime
import argparse
import logging
import warnings
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values`")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

BASE              = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parents[1]))
FIXEDRLT_PATH     = BASE / "9_GenPART/fixedRLT.py"
RETRIEVER_DIR     = BASE / "3_Initial_Retriever"
PIVOT_SCORES_DIR  = BASE / "9_GenPART/pivotbm25scores"
OUT_DIR           = BASE / "9_GenPART/runs"

PIVOT_TAU = 2   # default tau level (matches SNOW / GenTDPart)


# ─────────────────────────────────────────────────────────────────────────────
# Reuse the Fixed-100 pipeline (monoT5/duoT5 prompt + scoring helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _load_fixedrlt():
    spec = importlib.util.spec_from_file_location("fixedrlt_base", FIXEDRLT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load fixedRLT module from {FIXEDRLT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx = _load_fixedrlt()
td = fx.td                       # shared data-loading helpers

MODEL_NAME        = fx.MODEL_NAME
DUOT5_MODEL_NAME  = fx.DUOT5_MODEL_NAME
TOP_K             = fx.TOP_K            # 100
DUOT5_TOP_K       = fx.DUOT5_TOP_K      # 50
MAX_INPUT_LEN     = fx.MAX_INPUT_LEN    # 512
BATCH_SIZE        = fx.BATCH_SIZE       # 64
DATASETS          = fx.DATASETS

# bm25pivots.py scored every dataset except FEVER
GENPART_DATASETS  = [d for d in DATASETS if d != "fever"]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders: BM25 run with scores, and precomputed pivot scores
# ─────────────────────────────────────────────────────────────────────────────

def load_run_with_scores(run_file: Path) -> Dict[str, List[Tuple[str, float]]]:
    """Parse a TREC run file → {qid: [(docid, score), …]} sorted by score desc."""
    run: Dict[str, List[Tuple[str, float]]] = {}
    with open(run_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            qid, _q0, docid, _rank, score = parts[0], parts[1], parts[2], parts[3], parts[4]
            run.setdefault(qid, []).append((docid, float(score)))
    for qid in run:
        run[qid].sort(key=lambda x: x[1], reverse=True)
    return run


def pivot_scores_path(pivot_type: str, dataset: str) -> Path:
    if pivot_type == "separate":
        return PIVOT_SCORES_DIR / f"{dataset}.json"
    return PIVOT_SCORES_DIR / pivot_type / f"{dataset}.json"


def load_pivot_scores(pivot_type: str, dataset: str, tau: int) -> Dict[str, float]:
    """Return {qid: θ(Q, D*)} for the requested tau (key 'single' for single mode)."""
    path = pivot_scores_path(pivot_type, dataset)
    data = json.loads(path.read_text())
    tau_key = "single" if pivot_type == "single" else str(tau)
    scores: Dict[str, float] = {}
    for e in data.get("queries", []):
        s = e.get("scores", {}).get(tau_key)
        if s is not None:
            scores[str(e["qid"])] = float(s)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Shared batched scorer (same OOM-halving loop as fixedRLT, reusing score_batch)
# ─────────────────────────────────────────────────────────────────────────────

def _score_all(
    model, tokenizer, true_id, false_id, device,
    texts: List[str], batch_size: int, max_input_len: int, label: str,
) -> Tuple[List[float], int, float]:
    """Score every prompt string; auto-halves batch size on OOM. Reuses
    fixedRLT.score_batch for the actual forward pass."""
    n               = len(texts)
    effective_batch = batch_size
    all_scores: List[float] = []
    t0 = time.time()
    i  = 0
    while i < n:
        batch = texts[i : i + effective_batch]
        try:
            scores = fx.score_batch(
                model, tokenizer, batch, true_id, false_id, device, max_len=max_input_len
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            effective_batch = max(1, effective_batch // 2)
            log.warning(f"  OOM — halving batch size to {effective_batch} and retrying …")
            continue
        all_scores.extend(scores)
        i += len(batch)
        batch_num = len(all_scores) // max(effective_batch, 1)
        if batch_num % 50 == 0 or i >= n:
            log.info(
                f"  [{label}] … {i:,}/{n:,} ({i / n * 100:.1f}%)  "
                f"{time.time() - t0:.1f}s  batch_size={effective_batch}"
            )
    return all_scores, effective_batch, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Partitioning
# ─────────────────────────────────────────────────────────────────────────────

def compute_cutoff(docs_scores: List[Tuple[str, float]], pivot_score: Optional[float],
                   top_k: int) -> int:
    """Cutoff p = #docs scoring ≥ pivot among the top_k.  Missing pivot → rerank all."""
    capped = docs_scores[:top_k]
    if pivot_score is None:
        return len(capped)
    return sum(1 for _docid, s in capped if s >= pivot_score)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — monoT5 over the promising set only
# ─────────────────────────────────────────────────────────────────────────────

def genpart_monot5_rerank(
    model, tokenizer, true_id, false_id, device,
    queries       : Dict[str, str],
    run_scores    : Dict[str, List[Tuple[str, float]]],
    pivot_scores  : Dict[str, float],
    corpus        : Dict[str, str],
    top_k         : int = TOP_K,
    batch_size    : int = BATCH_SIZE,
    max_input_len : int = MAX_INPUT_LEN,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]], Dict[str, int], dict]:
    """
    Returns (ranked, promising_ranked, remainder, cutoffs, timing):
      ranked          : {qid: full top_k list = reranked(D+) ++ D-}
      promising_ranked: {qid: monoT5-reordered D+}   (handed to the duoT5 stage)
      remainder       : {qid: D- in original BM25 order}
      cutoffs         : {qid: p}
    """
    qids_ordered = [qid for qid in run_scores if qid in queries]
    n_queries    = len(qids_ordered)

    wall_start     = time.time()
    wall_start_iso = datetime.datetime.now().isoformat(timespec="seconds")

    # ── Partition every query by the pivot cutoff ─────────────────────────────
    cutoffs   : Dict[str, int]       = {}
    promising : Dict[str, List[str]] = {}
    remainder : Dict[str, List[str]] = {}
    n_missing_pivot = 0

    for qid in qids_ordered:
        ds  = run_scores[qid][:top_k]
        piv = pivot_scores.get(qid)
        if piv is None:
            n_missing_pivot += 1
        p = compute_cutoff(ds, piv, top_k)
        cutoffs[qid]   = p
        promising[qid] = [d for d, _ in ds[:p]]
        remainder[qid] = [d for d, _ in ds[p:]]

    cut_vals = list(cutoffs.values())
    log.info(
        f"  Partition: avg cutoff p={sum(cut_vals)/n_queries:.1f}  "
        f"min={min(cut_vals)}  max={max(cut_vals)}  "
        f"(p=0 for {sum(1 for v in cut_vals if v == 0)} queries)"
    )
    if n_missing_pivot:
        log.warning(f"  {n_missing_pivot} queries had no pivot score — reranked in full.")

    # ── Build monoT5 prompts for promising docs only ──────────────────────────
    all_texts  : List[str]             = []
    pair_index : List[Tuple[str, str]] = []
    for qid in qids_ordered:
        query = queries[qid]
        for docid in promising[qid]:
            all_texts.append(fx.build_monot5_input(query, corpus.get(docid, ""), tokenizer, max_input_len))
            pair_index.append((qid, docid))

    n_pairs = len(all_texts)
    n_fixed = sum(min(len(run_scores[qid]), top_k) for qid in qids_ordered)
    log.info(
        f"  monoT5 pairs to score: {n_pairs:,}  "
        f"(Fixed-100 would score {n_fixed:,} → "
        f"{(1 - n_pairs / max(n_fixed, 1)) * 100:.1f}% fewer inferences)"
    )

    all_scores, eff_batch, t_infer = _score_all(
        model, tokenizer, true_id, false_id, device,
        all_texts, batch_size, max_input_len, label="monoT5",
    )

    # ── Sort D+ by monoT5 score; stitch reranked(D+) ++ D- ────────────────────
    qscores: Dict[str, Dict[str, float]] = {qid: {} for qid in qids_ordered}
    for (qid, docid), sc in zip(pair_index, all_scores):
        qscores[qid][docid] = sc

    ranked           : Dict[str, List[str]] = {}
    promising_ranked : Dict[str, List[str]] = {}
    for qid in qids_ordered:
        dplus_sorted = [
            d for d, _ in sorted(qscores[qid].items(), key=lambda x: x[1], reverse=True)
        ]
        promising_ranked[qid] = dplus_sorted
        ranked[qid]           = (dplus_sorted + remainder[qid])[:top_k]

    total_s         = time.time() - wall_start
    avg_s_per_query = total_s / n_queries if n_queries else 0
    log.info(
        f"\n  ── Gen-PART monoT5 summary ─────────────────────────────\n"
        f"  Queries          : {n_queries:,}\n"
        f"  monoT5 pairs      : {n_pairs:,}  (vs Fixed-100 {n_fixed:,})\n"
        f"  Avg cutoff p      : {sum(cut_vals)/n_queries:.1f}\n"
        f"  Total wall time   : {total_s:.2f}s\n"
        f"  Infer time        : {t_infer:.2f}s ({t_infer/total_s*100:.1f}% of wall)\n"
        f"  ─────────────────────────────────────────────────────────"
    )

    timing = {
        "wall_clock_start"    : wall_start_iso,
        "wall_clock_end"      : datetime.datetime.now().isoformat(timespec="seconds"),
        "total_rerank_s"      : round(total_s,                 4),
        "avg_s_per_query"     : round(avg_s_per_query,         4),
        "avg_ms_per_query"    : round(avg_s_per_query * 1000,  2),
        "total_infer_s"       : round(t_infer,                 4),
        "pct_time_in_infer"   : round(t_infer / total_s * 100, 2) if total_s else 0,
        "n_queries"           : n_queries,
        "n_pairs_scored"      : n_pairs,
        "n_pairs_fixed100"    : n_fixed,
        "inference_savings_pct": round((1 - n_pairs / max(n_fixed, 1)) * 100, 2),
        "avg_cutoff_p"        : round(sum(cut_vals) / n_queries, 3) if n_queries else 0,
        "max_cutoff_p"        : max(cut_vals) if cut_vals else 0,
        "n_queries_cutoff0"   : sum(1 for v in cut_vals if v == 0),
        "n_missing_pivot"     : n_missing_pivot,
        "effective_batch_size": eff_batch,
    }
    return ranked, promising_ranked, remainder, cutoffs, timing


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — duoT5 over the top of the promising set only
# ─────────────────────────────────────────────────────────────────────────────

def genpart_duot5_rerank(
    model, tokenizer, true_id, false_id, device,
    queries          : Dict[str, str],
    promising_ranked : Dict[str, List[str]],
    remainder        : Dict[str, List[str]],
    corpus           : Dict[str, str],
    duot5_k          : int = DUOT5_TOP_K,
    top_k            : int = TOP_K,
    batch_size       : int = BATCH_SIZE,
    max_input_len    : int = MAX_INPUT_LEN,
) -> Tuple[Dict[str, List[str]], dict]:
    """
    Pairwise duoT5 over the top min(duot5_k, |D+|) of each query's monoT5-reranked
    promising list.  Aggregation uses SYM-SUM (matching fixedRLT): each pair
    contributes P(doc_i ≻ doc_j) to doc_i AND (1 - P) to doc_j.
    Assembly: duoT5(top) ++ rest of D+ ++ D-.
    """
    qids_ordered = [qid for qid in promising_ranked if qid in queries]
    n_queries    = len(qids_ordered)

    wall_start     = time.time()
    wall_start_iso = datetime.datetime.now().isoformat(timespec="seconds")

    # candidate set per query (the only docs duoT5 sees)
    candidates: Dict[str, List[str]] = {
        qid: promising_ranked[qid][:duot5_k] for qid in qids_ordered
    }

    all_texts  : List[str]                  = []
    pair_index : List[Tuple[str, str, str]] = []
    for qid in qids_ordered:
        query = queries[qid]
        cand  = candidates[qid]
        for i, doc_i in enumerate(cand):
            text_i = corpus.get(doc_i, "")
            for j, doc_j in enumerate(cand):
                if i == j:
                    continue
                all_texts.append(fx.build_duot5_input(query, text_i, corpus.get(doc_j, ""), tokenizer, max_input_len))
                pair_index.append((qid, doc_i, doc_j))

    n_pairs = len(all_texts)
    # Fixed-100 duoT5 scores duot5_k*(duot5_k-1) pairs per query
    kk      = min(duot5_k, top_k)
    n_fixed = kk * (kk - 1) * n_queries
    log.info(
        f"  duoT5 pairs to score: {n_pairs:,}  "
        f"(Fixed-100 duoT5 would score {n_fixed:,} → "
        f"{(1 - n_pairs / max(n_fixed, 1)) * 100:.1f}% fewer)"
    )

    all_scores, eff_batch, t_infer = _score_all(
        model, tokenizer, true_id, false_id, device,
        all_texts, batch_size, max_input_len, label="duoT5",
    )

    agg: Dict[str, Dict[str, float]] = {
        qid: {docid: 0.0 for docid in candidates[qid]} for qid in qids_ordered
    }
    for (qid, doc_i, doc_j), sc in zip(pair_index, all_scores):
        agg[qid][doc_i] += sc
        agg[qid][doc_j] += 1.0 - sc

    ranked: Dict[str, List[str]] = {}
    for qid in qids_ordered:
        duo_sorted = [d for d, _ in sorted(agg[qid].items(), key=lambda x: x[1], reverse=True)]
        tail       = promising_ranked[qid][duot5_k:] + remainder[qid]
        ranked[qid] = (duo_sorted + tail)[:top_k]

    total_s         = time.time() - wall_start
    avg_s_per_query = total_s / n_queries if n_queries else 0
    log.info(
        f"\n  ── Gen-PART duoT5 summary ──────────────────────────────\n"
        f"  Queries          : {n_queries:,}\n"
        f"  duoT5 pairs       : {n_pairs:,}  (vs Fixed-100 {n_fixed:,})\n"
        f"  Total wall time   : {total_s:.2f}s\n"
        f"  Infer time        : {t_infer:.2f}s ({t_infer/total_s*100:.1f}% of wall)\n"
        f"  ─────────────────────────────────────────────────────────"
    )

    timing = {
        "wall_clock_start"     : wall_start_iso,
        "wall_clock_end"       : datetime.datetime.now().isoformat(timespec="seconds"),
        "total_rerank_s"       : round(total_s,                 4),
        "avg_s_per_query"      : round(avg_s_per_query,         4),
        "avg_ms_per_query"     : round(avg_s_per_query * 1000,  2),
        "total_infer_s"        : round(t_infer,                 4),
        "pct_time_in_infer"    : round(t_infer / total_s * 100, 2) if total_s else 0,
        "n_queries"            : n_queries,
        "n_pairs_scored"       : n_pairs,
        "n_pairs_fixed100"     : n_fixed,
        "inference_savings_pct": round((1 - n_pairs / max(n_fixed, 1)) * 100, 2),
        "effective_batch_size" : eff_batch,
    }
    return ranked, timing


# ─────────────────────────────────────────────────────────────────────────────
# Naming helpers
# ─────────────────────────────────────────────────────────────────────────────

def pivot_tag(pivot_type: str, tau: int) -> str:
    if pivot_type == "single":
        return "genpart_single"
    if pivot_type == "tautogether":
        return f"genpart_tautogether_tau{tau}"
    return f"genpart_tau{tau}"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gen-PART — pivot-cutoff RLT for monoT5/duoT5 reranking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", nargs="+", default=["all"],
        help=f"Dataset(s). Choices: {GENPART_DATASETS + ['all']} (FEVER unsupported).",
    )
    p.add_argument("--retriever", default="bm25", choices=["bm25", "splade"],
                   help="First-stage retriever whose run files / pivots are used.")
    p.add_argument("--pivot-type", default="separate",
                   choices=["separate", "tautogether", "single"],
                   help="Which precomputed pivot scores to threshold on.")
    p.add_argument("--pivot-tau", type=int, default=PIVOT_TAU,
                   help="Tau level of the pivot (ignored for --pivot-type single).")
    p.add_argument("--top-k", type=int, default=TOP_K,
                   help="Candidate pool size per query.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="Initial pairs per forward pass. Auto-halved on OOM.")
    p.add_argument("--max-input-len", type=int, default=MAX_INPUT_LEN,
                   help="Max encoder input tokens (monoT5/duoT5 trained at 512).")
    p.add_argument("--monot5duot5", nargs="?", const=DUOT5_TOP_K, default=None, type=int,
                   metavar="K",
                   help=("Enable the duoT5 stage on the top min(K, p) of the monoT5-reranked "
                         f"promising set. K defaults to {DUOT5_TOP_K}. Reuses an existing "
                         "Gen-PART monoT5 run if present."))
    p.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    raw = args.dataset
    if raw == ["all"] or raw == "all":
        datasets_to_run = list(GENPART_DATASETS)
    else:
        datasets_to_run = raw if isinstance(raw, list) else [raw]
        unknown = [d for d in datasets_to_run if d not in DATASETS]
        if unknown:
            log.error(f"Unknown dataset(s): {unknown}. Valid: {GENPART_DATASETS}")
            sys.exit(1)

    retriever = args.retriever
    duot5_k   = args.monot5duot5
    tau       = args.pivot_tau
    ptag      = pivot_tag(args.pivot_type, tau)

    log.info("=" * 65)
    log.info(f"monoT5 model : {MODEL_NAME}")
    if duot5_k:
        log.info(f"duoT5  model : {DUOT5_MODEL_NAME}")
        log.info(f"Mode         : Gen-PART monoT5(D+) → duoT5(top-{duot5_k} of D+)")
    else:
        log.info(f"Mode         : Gen-PART monoT5(D+) only")
    log.info(f"Pivot        : type={args.pivot_type}  tau={tau}  ({ptag})")
    log.info(f"Batch size   : {args.batch_size} (auto-halved on OOM)  |  max_input_len={args.max_input_len}")
    log.info("=" * 65)

    if not torch.cuda.is_available():
        log.warning("No CUDA device found — running on CPU (very slow).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")
    if device.type == "cuda":
        log.info(f"GPU    : {torch.cuda.get_device_name(0)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def monot5_run_path(dn: str) -> Path:
        return OUT_DIR / f"{dn}.{retriever}.monot5_3b.{ptag}.top{args.top_k}.txt"

    # ─── Phase 1 : monoT5 ───────────────────────────────────────────────────
    if duot5_k:
        datasets_for_monot5 = [
            dn for dn in datasets_to_run
            if not monot5_run_path(dn).exists() or args.force
        ]
        for dn in datasets_to_run:
            if monot5_run_path(dn).exists() and not args.force:
                log.info(f"  monoT5 run found, will reuse for duoT5: {monot5_run_path(dn).name}")
    else:
        datasets_for_monot5 = datasets_to_run

    t_model_load        = 0.0
    monot5_model_loaded = False

    if datasets_for_monot5:
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        log.info("Loading monoT5 tokenizer …")
        t_load    = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        log.info("Loading monoT5-3B model (bfloat16) …")
        model = T5ForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()
        t_model_load = time.time() - t_load
        log.info(f"monoT5 loaded in {t_model_load:.1f}s.")

        vocab    = tokenizer.get_vocab()
        true_id  = vocab["▁true"]
        false_id = vocab["▁false"]
        log.info(f"Token IDs — ▁true: {true_id}  ▁false: {false_id}")
        monot5_model_loaded = True
        run_tag_m5 = f"monoT5_3B_{retriever}_{ptag}"

        for dataset_name in datasets_for_monot5:
            cfg = DATASETS[dataset_name]
            log.info("")
            log.info("─" * 65)
            log.info(f"[monoT5] Dataset : {dataset_name}   Retriever : {retriever}")
            log.info("─" * 65)

            bm25_run_file = RETRIEVER_DIR / retriever / f"{dataset_name}.top{args.top_k}.txt"
            if not bm25_run_file.exists():
                log.warning(f"  Input run not found: {bm25_run_file} — skipping.")
                continue
            piv_file = pivot_scores_path(args.pivot_type, dataset_name)
            if not piv_file.exists():
                log.warning(f"  Pivot scores not found: {piv_file} — skipping.")
                continue

            out_file        = monot5_run_path(dataset_name)
            out_timing_file = OUT_DIR / f"{dataset_name}.{retriever}.monot5_3b.{ptag}.top{args.top_k}.timing.json"
            if out_file.exists() and not args.force:
                log.info(f"  Output exists (--force to overwrite): {out_file.name}")
                continue

            valid_qids = td.load_qrel_qids(cfg["qrels"], cfg["qrels_format"])
            queries    = td.load_queries(cfg["queries"], cfg["query_format"], valid_qids)
            log.info(f"  Queries with qrels: {len(queries):,}")

            run_scores = load_run_with_scores(bm25_run_file)
            run_scores = {qid: ds for qid, ds in run_scores.items() if qid in queries}
            log.info(f"  Run queries (overlap with qrels): {len(run_scores):,}")
            if not run_scores:
                log.warning("  No queries in run — skipping.")
                continue

            pivot_scores = load_pivot_scores(args.pivot_type, dataset_name, tau)
            log.info(f"  Pivot scores loaded: {len(pivot_scores):,} (tau={tau}, {args.pivot_type})")

            t_corpus = time.time()
            needed   = {docid for ds in run_scores.values() for docid, _ in ds[:args.top_k]}
            corpus   = td.load_corpus_selective(cfg["corpus"], cfg["corpus_format"], needed)
            t_corpus = time.time() - t_corpus

            ranked, _promising, _remainder, _cutoffs, timing = genpart_monot5_rerank(
                model=model, tokenizer=tokenizer, true_id=true_id, false_id=false_id,
                device=device, queries=queries, run_scores=run_scores,
                pivot_scores=pivot_scores, corpus=corpus, top_k=args.top_k,
                batch_size=args.batch_size, max_input_len=args.max_input_len,
            )

            timing.update({
                "dataset": dataset_name, "retriever": retriever, "model": MODEL_NAME,
                "corpus_load_s": round(t_corpus, 4), "model_load_s": round(t_model_load, 4),
                "pivot_type": args.pivot_type, "pivot_tau": tau,
                "config": {
                    "top_k": args.top_k, "pivot_type": args.pivot_type,
                    "pivot_tau": tau if args.pivot_type != "single" else None,
                    "batch_size": args.batch_size, "max_input_len": args.max_input_len,
                },
            })
            td.write_trec_run(ranked, out_file, tag=run_tag_m5)
            td.write_timing(timing, out_timing_file)

    if not duot5_k:
        log.info("")
        log.info("All done.")
        return

    # ─── Phase 2 : duoT5 ────────────────────────────────────────────────────
    if monot5_model_loaded:
        log.info("")
        log.info("Unloading monoT5 to free VRAM before loading duoT5 …")
        del model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    from transformers import AutoTokenizer, T5ForConditionalGeneration

    log.info("Loading duoT5 tokenizer …")
    t_load    = time.time()
    tokenizer = AutoTokenizer.from_pretrained(DUOT5_MODEL_NAME)
    log.info("Loading duoT5-3B model (bfloat16) …")
    model = T5ForConditionalGeneration.from_pretrained(
        DUOT5_MODEL_NAME, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    t_duot5_load = time.time() - t_load
    log.info(f"duoT5 loaded in {t_duot5_load:.1f}s.")

    vocab    = tokenizer.get_vocab()
    true_id  = vocab["▁true"]
    false_id = vocab["▁false"]
    run_tag_d5 = f"monot5_3b_duot5_3b_{retriever}_{ptag}"

    for dataset_name in datasets_to_run:
        cfg = DATASETS[dataset_name]
        log.info("")
        log.info("─" * 65)
        log.info(f"[duoT5] Dataset : {dataset_name}   Retriever : {retriever}")
        log.info("─" * 65)

        m5_run_file = monot5_run_path(dataset_name)
        if not m5_run_file.exists():
            log.warning(f"  Gen-PART monoT5 run not found: {m5_run_file} — skipping duoT5.")
            continue
        piv_file = pivot_scores_path(args.pivot_type, dataset_name)
        if not piv_file.exists():
            log.warning(f"  Pivot scores not found: {piv_file} — skipping.")
            continue

        out_stem        = f"{dataset_name}.{retriever}.monot5_3b_duot5_3b.{ptag}_duo{duot5_k}.top{args.top_k}"
        out_file        = OUT_DIR / f"{out_stem}.txt"
        out_timing_file = OUT_DIR / f"{out_stem}.timing.json"
        if out_file.exists() and not args.force:
            log.info(f"  duoT5 output exists (--force to overwrite): {out_file.name}")
            continue

        valid_qids = td.load_qrel_qids(cfg["qrels"], cfg["qrels_format"])
        queries    = td.load_queries(cfg["queries"], cfg["query_format"], valid_qids)

        # Reconstruct D+ / D- from the saved Gen-PART monoT5 run + the original
        # BM25 cutoff: the first p docs (those ≥ pivot) are the monoT5-reranked
        # promising set; the rest are the bypassed remainder.
        m5_ranked = td.load_run(m5_run_file)
        m5_ranked = {qid: docs for qid, docs in m5_ranked.items() if qid in queries}
        if not m5_ranked:
            log.warning("  No queries in monoT5 run — skipping.")
            continue

        bm25_run_file = RETRIEVER_DIR / retriever / f"{dataset_name}.top{args.top_k}.txt"
        run_scores    = load_run_with_scores(bm25_run_file)
        pivot_scores  = load_pivot_scores(args.pivot_type, dataset_name, tau)

        promising_ranked: Dict[str, List[str]] = {}
        remainder       : Dict[str, List[str]] = {}
        for qid, docs in m5_ranked.items():
            p = compute_cutoff(run_scores.get(qid, []), pivot_scores.get(qid), args.top_k)
            promising_ranked[qid] = docs[:p]
            remainder[qid]        = docs[p:]
        log.info(f"  monoT5 run queries: {len(m5_ranked):,}")

        needed   = {d for qid in promising_ranked for d in promising_ranked[qid][:duot5_k]}
        t_corpus = time.time()
        corpus   = td.load_corpus_selective(cfg["corpus"], cfg["corpus_format"], needed)
        t_corpus = time.time() - t_corpus

        ranked, timing = genpart_duot5_rerank(
            model=model, tokenizer=tokenizer, true_id=true_id, false_id=false_id,
            device=device, queries=queries, promising_ranked=promising_ranked,
            remainder=remainder, corpus=corpus, duot5_k=duot5_k, top_k=args.top_k,
            batch_size=args.batch_size, max_input_len=args.max_input_len,
        )

        timing.update({
            "dataset": dataset_name, "retriever": retriever,
            "monot5_model": MODEL_NAME, "duot5_model": DUOT5_MODEL_NAME,
            "corpus_load_s": round(t_corpus, 4), "duot5_load_s": round(t_duot5_load, 4),
            "pivot_type": args.pivot_type, "pivot_tau": tau,
            "config": {
                "top_k": args.top_k, "duot5_top_k": duot5_k,
                "pivot_type": args.pivot_type,
                "pivot_tau": tau if args.pivot_type != "single" else None,
                "batch_size": args.batch_size, "max_input_len": args.max_input_len,
            },
        })
        td.write_trec_run(ranked, out_file, tag=run_tag_d5)
        td.write_timing(timing, out_timing_file)

    log.info("")
    log.info("All done.")


if __name__ == "__main__":
    main()
