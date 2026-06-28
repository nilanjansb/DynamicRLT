#!/usr/bin/env python3
"""
Fixed-100 monoT5 Reranking — No-RLT upper-bound effectiveness baseline.
  Model: castorini/monot5-3b-msmarco-10k

This baseline scores ALL top-100 BM25-retrieved documents with monoT5 and
ranks them by relevance score.  No document is skipped or truncated from
the candidate set, making it the upper-bound for effectiveness among all
RLT variants (at the cost of scoring every doc).

Scoring
-------
  For each (query, document) pair, the model is given:
      "Query: {query} Document: {document} Relevant:"
  The relevance score is the normalised probability of the token "true" at
  the first decoder output position (softmax over {true, false}):
      score = P(true | query, document)  via softmax([logit_true, logit_false])

  All 100 docs per query are scored independently (pointwise), then sorted
  in descending score order.

Output naming:
  {dataset}.{retriever}.monot5_3b.fixed100.top{K}.txt
  {dataset}.{retriever}.monot5_3b.fixed100.top{K}.timing.json

   e.g.: msmarco-dl19.bm25.monot5_3b.fixed100.top100.txt

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 9_GenPART/fixedRLT.py \\
        --dataset msmarco-dl19 --retriever bm25

    CUDA_VISIBLE_DEVICES=0 python3 9_GenPART/fixedRLT.py \\
        --dataset all --retriever bm25 --batch-size 64
"""

import os
import re
import sys
import json
import time
import datetime
import argparse
import logging
import warnings
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Reduce allocator fragmentation — helps when document lengths vary widely across datasets
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Suppress T5 internal past_key_values tuple deprecation (not triggered by our code)
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

BASE          = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parents[1]))
TDPART_PATH   = BASE / "5_TDPart/tdpart.py"
RETRIEVER_DIR = BASE / "3_Initial_Retriever"
OUT_DIR       = BASE / "9_GenPART/runs"

MODEL_NAME       = "castorini/monot5-3b-msmarco-10k"
DUOT5_MODEL_NAME = "castorini/duot5-3b-msmarco"
TOP_K            = 100
DUOT5_TOP_K      = 50
MAX_INPUT_LEN    = 512   # encoder token budget (both mono and duo T5)
BATCH_SIZE       = 64    # conservative default; auto-halved on OOM
RESERVED_TOKENS  = 16    # safety margin so the "Relevant:" scaffold + EOS always
                         # survive after document truncation (rank_llm uses 64;
                         # we only emit 1 output token, so a small margin suffices)


# ─────────────────────────────────────────────────────────────────────────────
# Load TDPart module for shared data-loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tdpart():
    spec = importlib.util.spec_from_file_location("tdpart_base", TDPART_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load TDPart module from {TDPART_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


td       = _load_tdpart()
DATASETS = td.DATASETS


# ─────────────────────────────────────────────────────────────────────────────
# monoT5 scoring
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_doc_to_budget(tokenizer, doc_text: str, budget: int) -> str:
    """Truncate doc_text to at most `budget` tokens (no special tokens added)."""
    if budget <= 0:
        return ""
    ids = tokenizer.encode(doc_text, add_special_tokens=False)
    if len(ids) <= budget:
        return doc_text
    return tokenizer.decode(ids[:budget], skip_special_tokens=True)


def build_monot5_input(
    query: str,
    doc_text: str,
    tokenizer=None,
    max_input_len: int = MAX_INPUT_LEN,
    reserved: int = RESERVED_TOKENS,
) -> str:
    """
    Build the monoT5 prompt 'Query: {q} Document: {d} Relevant: '.

    When a tokenizer is supplied, the DOCUMENT is truncated to fit the encoder
    budget so the trailing 'Relevant:' relevance cue (and EOS) always survive —
    matching rank_llm/pygaggle.  Without a tokenizer the old un-truncated string
    is returned (relying on the encoder's own right-truncation, which can drop
    the cue on long docs).
    """
    if tokenizer is not None:
        scaffold = f"Query: {query} Document:  Relevant: "          # doc slot empty
        budget   = max_input_len - reserved - len(tokenizer.encode(scaffold))
        doc_text = _truncate_doc_to_budget(tokenizer, doc_text, budget)
    return f"Query: {query} Document: {doc_text} Relevant: "


def score_batch(
    model,
    tokenizer,
    texts     : List[str],
    true_id   : int,
    false_id  : int,
    device    : torch.device,
    max_len   : int = MAX_INPUT_LEN,
) -> List[float]:
    """
    Score a batch of monoT5 prompt strings.

    Runs a single encoder-decoder forward pass with the decoder primed with
    the decoder-start token, then reads the logits for the first output
    position.  Returns P(true) = softmax over [true, false] for each item.
    """
    enc = tokenizer(
        texts,
        padding        = True,
        truncation     = True,
        max_length     = max_len,
        return_tensors = "pt",
    ).to(device)

    batch_size    = enc["input_ids"].shape[0]
    decoder_input = torch.full(
        (batch_size, 1),
        model.config.decoder_start_token_id,
        dtype  = torch.long,
        device = device,
    )

    with torch.no_grad():
        outputs = model(
            input_ids         = enc["input_ids"],
            attention_mask    = enc["attention_mask"],
            decoder_input_ids = decoder_input,
        )

    # outputs.logits: (batch, 1, vocab_size) — first decoder output position
    logits    = outputs.logits[:, 0, :]                     # (batch, vocab)
    tf_logits = logits[:, [true_id, false_id]]              # (batch, 2)
    scores    = torch.nn.functional.softmax(tf_logits, dim=-1)[:, 0]  # P(true)
    return scores.cpu().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-100 reranking
# ─────────────────────────────────────────────────────────────────────────────

def fixed100_rerank(
    model,
    tokenizer,
    true_id       : int,
    false_id      : int,
    device        : torch.device,
    queries       : Dict[str, str],
    run           : Dict[str, List[str]],
    corpus        : Dict[str, str],
    top_k         : int = TOP_K,
    batch_size    : int = BATCH_SIZE,
    max_input_len : int = MAX_INPUT_LEN,
) -> Tuple[Dict[str, List[str]], dict]:
    """
    Score every top_k document for every query and return ranked lists.

    Batch size is halved automatically on OOM, so the run survives datasets
    with long documents (e.g. scifact) without manual tuning.

    Returns
    -------
    (ranked, timing)
      ranked : {qid: [docid_rank1, …, docid_rank_top_k]}
      timing : structured dict with aggregate timing info
    """
    qids_ordered = [qid for qid in run if qid in queries]
    n_queries    = len(qids_ordered)

    wall_start     = time.time()
    wall_start_iso = datetime.datetime.now().isoformat(timespec="seconds")

    # ── Build all (query, doc) prompt strings ────────────────────────────────
    log.info(
        f"  Building {n_queries:,} × {top_k} = {n_queries * top_k:,} "
        f"(query, doc) prompts …"
    )
    all_texts  : List[str]             = []
    pair_index : List[Tuple[str, str]] = []   # (qid, docid)

    for qid in qids_ordered:
        query = queries[qid]
        for docid in run[qid][:top_k]:
            doc_text = corpus.get(docid, "")
            all_texts.append(build_monot5_input(query, doc_text, tokenizer, max_input_len))
            pair_index.append((qid, docid))

    n_pairs          = len(all_texts)
    effective_batch  = batch_size

    log.info(
        f"  Scoring {n_pairs:,} pairs "
        f"(batch_size={effective_batch}, max_input_len={max_input_len}) …"
    )

    # ── Batched inference with automatic OOM recovery ────────────────────────
    all_scores    : List[float] = []
    t_infer_start = time.time()
    i             = 0

    while i < n_pairs:
        batch = all_texts[i : i + effective_batch]
        try:
            scores = score_batch(
                model, tokenizer, batch, true_id, false_id, device, max_len=max_input_len
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            effective_batch = max(1, effective_batch // 2)
            log.warning(f"  OOM — halving batch size to {effective_batch} and retrying …")
            continue

        all_scores.extend(scores)
        batch_num = len(all_scores) // effective_batch
        i        += len(batch)

        if batch_num % 50 == 0 or i >= n_pairs:
            elapsed = time.time() - t_infer_start
            log.info(
                f"  … {i:,}/{n_pairs:,} pairs ({i / n_pairs * 100:.1f}%)  "
                f"{elapsed:.1f}s elapsed  batch_size={effective_batch}"
            )

    t_infer = time.time() - t_infer_start
    log.info(f"  Inference done in {t_infer:.2f}s.")

    # ── Collect scores per query and sort ────────────────────────────────────
    query_scores: Dict[str, Dict[str, float]] = {qid: {} for qid in qids_ordered}
    for (qid, docid), score in zip(pair_index, all_scores):
        query_scores[qid][docid] = score

    ranked: Dict[str, List[str]] = {}
    for qid in qids_ordered:
        ranked[qid] = [
            docid
            for docid, _ in sorted(
                query_scores[qid].items(), key=lambda x: x[1], reverse=True
            )
        ]

    # ── Aggregate timing ─────────────────────────────────────────────────────
    total_s         = time.time() - wall_start
    wall_end_iso    = datetime.datetime.now().isoformat(timespec="seconds")
    avg_s_per_query = total_s / n_queries if n_queries else 0

    log.info(
        f"\n  ── Fixed-100 monoT5 summary ────────────────────────────\n"
        f"  Queries         : {n_queries:,}\n"
        f"  Docs/query      : {top_k}\n"
        f"  Total pairs     : {n_pairs:,}\n"
        f"  Total wall time : {total_s:.2f}s\n"
        f"  Avg per query   : {avg_s_per_query * 1000:.1f}ms\n"
        f"  Infer time      : {t_infer:.2f}s  "
        f"({t_infer / total_s * 100:.1f}% of wall time)\n"
        f"  ─────────────────────────────────────────────────────────"
    )

    timing = {
        "wall_clock_start"   : wall_start_iso,
        "wall_clock_end"     : wall_end_iso,
        "total_rerank_s"     : round(total_s,                 4),
        "avg_s_per_query"    : round(avg_s_per_query,         4),
        "avg_ms_per_query"   : round(avg_s_per_query * 1000,  2),
        "total_infer_s"      : round(t_infer,                 4),
        "pct_time_in_infer"  : round(t_infer / total_s * 100, 2) if total_s else 0,
        "n_queries"          : n_queries,
        "n_pairs_scored"     : n_pairs,
        "effective_batch_size": effective_batch,
    }

    return ranked, timing


# ─────────────────────────────────────────────────────────────────────────────
# duoT5 scoring
# ─────────────────────────────────────────────────────────────────────────────

def build_duot5_input(
    query: str,
    doc0_text: str,
    doc1_text: str,
    tokenizer=None,
    max_input_len: int = MAX_INPUT_LEN,
    reserved: int = RESERVED_TOKENS,
) -> str:
    """
    Build the duoT5 prompt
    'Query: {q} Document0: {d0} Document1: {d1} Relevant: '.

    When a tokenizer is supplied, the remaining budget after the scaffold is
    split evenly between the two documents and each is truncated to fit, so the
    'Relevant:' cue and both document slots always survive (rank_llm/pygaggle
    behaviour).  Bracketed rank markers in the query (e.g. '[2]') are rewritten
    to parentheses, matching rank_llm's DuoT5.create_prompt.
    """
    query = re.sub(r"\[(\d+)\]", r"(\1)", query)
    if tokenizer is not None:
        scaffold = f"Query: {query} Document0:  Document1:  Relevant: "
        budget   = max_input_len - reserved - len(tokenizer.encode(scaffold))
        per_doc  = max(0, budget // 2)
        doc0_text = _truncate_doc_to_budget(tokenizer, doc0_text, per_doc)
        doc1_text = _truncate_doc_to_budget(tokenizer, doc1_text, per_doc)
    return f"Query: {query} Document0: {doc0_text} Document1: {doc1_text} Relevant: "


def duot5_rerank(
    model,
    tokenizer,
    true_id       : int,
    false_id      : int,
    device        : torch.device,
    queries       : Dict[str, str],
    monot5_ranked : Dict[str, List[str]],
    corpus        : Dict[str, str],
    top_k         : int = DUOT5_TOP_K,
    batch_size    : int = BATCH_SIZE,
    max_input_len : int = MAX_INPUT_LEN,
) -> Tuple[Dict[str, List[str]], dict]:
    """
    Pairwise duoT5 reranking over the top_k candidates from a monoT5 run.

    For each query, all K*(K-1) ordered (doc_i, doc_j) pairs are scored.
    Aggregation follows pygaggle's default SYM-SUM: each pair contributes
    P(doc_i ≻ doc_j) to doc_i AND its complement (1 - P) to doc_j, so every
    document accrues evidence from both pair orderings.  (Summing only the
    doc_i term — plain SUM — discards half the pairwise signal and yields a
    different, weaker ranking.)

    Returns
    -------
    (ranked, timing)
      ranked : {qid: [docid_rank1, …, docid_rank_top_k]}
      timing : structured dict with aggregate timing info
    """
    qids_ordered = [qid for qid in monot5_ranked if qid in queries]
    n_queries    = len(qids_ordered)

    wall_start     = time.time()
    wall_start_iso = datetime.datetime.now().isoformat(timespec="seconds")

    log.info(
        f"  Building duoT5 pairs for {n_queries:,} queries "
        f"(top_k={top_k}, {top_k * (top_k - 1):,} pairs/query) …"
    )

    all_texts  : List[str]                        = []
    pair_index : List[Tuple[str, str, str]]       = []   # (qid, doc_i_id, doc_j_id)

    for qid in qids_ordered:
        query      = queries[qid]
        candidates = monot5_ranked[qid][:top_k]
        for i, doc_i in enumerate(candidates):
            text_i = corpus.get(doc_i, "")
            for j, doc_j in enumerate(candidates):
                if i == j:
                    continue
                text_j = corpus.get(doc_j, "")
                all_texts.append(build_duot5_input(query, text_i, text_j, tokenizer, max_input_len))
                pair_index.append((qid, doc_i, doc_j))

    n_pairs         = len(all_texts)
    effective_batch = batch_size

    log.info(
        f"  Scoring {n_pairs:,} duoT5 pairs "
        f"(batch_size={effective_batch}, max_input_len={max_input_len}) …"
    )

    all_scores    : List[float] = []
    t_infer_start = time.time()
    i             = 0

    while i < n_pairs:
        batch = all_texts[i : i + effective_batch]
        try:
            scores = score_batch(
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
        if batch_num % 50 == 0 or i >= n_pairs:
            elapsed = time.time() - t_infer_start
            log.info(
                f"  … {i:,}/{n_pairs:,} pairs ({i / n_pairs * 100:.1f}%)  "
                f"{elapsed:.1f}s elapsed  batch_size={effective_batch}"
            )

    t_infer = time.time() - t_infer_start
    log.info(f"  duoT5 inference done in {t_infer:.2f}s.")

    # Aggregate (pygaggle SYM-SUM): each ordered pair adds P(doc_i ≻ doc_j) to
    # doc_i and its complement (1 - P) to doc_j.
    agg_scores: Dict[str, Dict[str, float]] = {
        qid: {docid: 0.0 for docid in monot5_ranked[qid][:top_k]}
        for qid in qids_ordered
    }
    for (qid, doc_i, doc_j), score in zip(pair_index, all_scores):
        agg_scores[qid][doc_i] += score
        agg_scores[qid][doc_j] += 1.0 - score

    # duoT5 reorders the top_k candidates; the remaining monoT5 docs
    # (ranks top_k+1 … end) are appended unchanged to preserve full depth
    # for recall/MAP-style metrics (expando-mono-duo design).
    ranked: Dict[str, List[str]] = {}
    for qid in qids_ordered:
        reranked = [
            docid
            for docid, _ in sorted(
                agg_scores[qid].items(), key=lambda x: x[1], reverse=True
            )
        ]
        ranked[qid] = reranked + monot5_ranked[qid][top_k:]

    total_s         = time.time() - wall_start
    wall_end_iso    = datetime.datetime.now().isoformat(timespec="seconds")
    avg_s_per_query = total_s / n_queries if n_queries else 0

    log.info(
        f"\n  ── DuoT5 summary ───────────────────────────────────────\n"
        f"  Queries         : {n_queries:,}\n"
        f"  Candidates/query: {top_k}\n"
        f"  Pairs scored    : {n_pairs:,}\n"
        f"  Total wall time : {total_s:.2f}s\n"
        f"  Avg per query   : {avg_s_per_query * 1000:.1f}ms\n"
        f"  Infer time      : {t_infer:.2f}s  "
        f"({t_infer / total_s * 100:.1f}% of wall time)\n"
        f"  ─────────────────────────────────────────────────────────"
    )

    timing = {
        "wall_clock_start"    : wall_start_iso,
        "wall_clock_end"      : wall_end_iso,
        "total_rerank_s"      : round(total_s,                 4),
        "avg_s_per_query"     : round(avg_s_per_query,         4),
        "avg_ms_per_query"    : round(avg_s_per_query * 1000,  2),
        "total_infer_s"       : round(t_infer,                 4),
        "pct_time_in_infer"   : round(t_infer / total_s * 100, 2) if total_s else 0,
        "n_queries"           : n_queries,
        "n_pairs_scored"      : n_pairs,
        "effective_batch_size": effective_batch,
    }

    return ranked, timing


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-100 monoT5-3B Reranking (No-RLT baseline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", nargs="+", default=["all"],
        help=(
            "Dataset(s) to process. Pass one name, several names, or 'all'. "
            f"Choices: {list(DATASETS.keys()) + ['all']}"
        ),
    )
    p.add_argument(
        "--retriever", default="bm25", choices=["bm25", "splade"],
        help="First-stage retriever whose TREC files are used as input.",
    )
    p.add_argument(
        "--top-k", type=int, default=TOP_K,
        help="Number of retrieved docs to score per query.",
    )
    p.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Initial (query, doc) pairs per forward pass. Auto-halved on OOM.",
    )
    p.add_argument(
        "--max-input-len", type=int, default=MAX_INPUT_LEN,
        help="Max encoder input tokens. monoT5 was trained at 512.",
    )
    p.add_argument(
        "--monot5duot5", nargs="?", const=DUOT5_TOP_K, default=None, type=int,
        metavar="K",
        help=(
            "Enable the monoT5 → duoT5 pipeline. monoT5 scores all --top-k docs "
            "first; duoT5 then pairwise-reranks the top K of those results. "
            f"K defaults to {DUOT5_TOP_K} when the flag is given without a value. "
            "If a monoT5 run already exists in runs/ it is loaded directly, "
            "skipping monoT5 inference."
        ),
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    raw = args.dataset
    if raw == ["all"] or raw == "all":
        datasets_to_run = list(DATASETS.keys())
    else:
        datasets_to_run = raw if isinstance(raw, list) else [raw]
        unknown = [d for d in datasets_to_run if d not in DATASETS]
        if unknown:
            log.error(f"Unknown dataset(s): {unknown}. Valid: {list(DATASETS.keys())}")
            sys.exit(1)

    retriever = args.retriever
    duot5_k   = args.monot5duot5   # None → normal monoT5-only mode

    log.info("=" * 65)
    log.info(f"monoT5 model : {MODEL_NAME}")
    if duot5_k:
        log.info(f"duoT5  model : {DUOT5_MODEL_NAME}")
        log.info(f"Mode         : monoT5(top-{args.top_k}) → duoT5(top-{duot5_k}) pipeline")
    else:
        log.info(f"Mode         : Fixed-100 — score all top-{args.top_k} docs (No-RLT baseline)")
    log.info(f"Batch size   : {args.batch_size} (auto-halved on OOM)  |  max_input_len={args.max_input_len}")
    log.info("=" * 65)

    if not torch.cuda.is_available():
        log.warning("No CUDA device found — running on CPU (will be very slow).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")
    if device.type == "cuda":
        log.info(f"GPU    : {torch.cuda.get_device_name(0)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Helper: canonical path for a monoT5 fixed-100 run file
    def monot5_run_path(dn: str) -> Path:
        return OUT_DIR / f"{dn}.{retriever}.monot5_3b.fixed100.top{args.top_k}.txt"

    # ─── Phase 1 : monoT5 ───────────────────────────────────────────────────
    # Normal mode  → process every dataset (skip existing unless --force).
    # Pipeline mode → only run monoT5 for datasets whose run file is missing.

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

    monot5_model_loaded = False
    t_model_load        = 0.0

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

        # Resolve ▁true / ▁false token IDs directly from the vocab to avoid
        # any SentencePiece prefix ambiguity (mirrors pygaggle's approach).
        vocab    = tokenizer.get_vocab()
        true_id  = vocab["▁true"]
        false_id = vocab["▁false"]
        log.info(f"Token IDs — ▁true: {true_id}  ▁false: {false_id}")

        monot5_model_loaded = True
        run_tag_m5 = f"monoT5_3B_{retriever}_fixed100"

        for dataset_name in datasets_for_monot5:
            cfg = DATASETS[dataset_name]

            log.info("")
            log.info("─" * 65)
            log.info(f"[monoT5] Dataset : {dataset_name}   Retriever : {retriever}")
            log.info("─" * 65)

            bm25_run_file = RETRIEVER_DIR / retriever / f"{dataset_name}.top{args.top_k}.txt"
            if not bm25_run_file.exists():
                log.warning(f"  Input BM25 run not found: {bm25_run_file} — skipping.")
                continue

            out_file        = monot5_run_path(dataset_name)
            out_timing_file = OUT_DIR / f"{dataset_name}.{retriever}.monot5_3b.fixed100.top{args.top_k}.timing.json"

            if out_file.exists() and not args.force:
                log.info(f"  Output exists (--force to overwrite): {out_file.name}")
                continue

            valid_qids = td.load_qrel_qids(cfg["qrels"], cfg["qrels_format"])
            queries    = td.load_queries(cfg["queries"], cfg["query_format"], valid_qids)
            log.info(f"  Queries with qrels: {len(queries):,}")

            run = td.load_run(bm25_run_file)
            run = {qid: docs for qid, docs in run.items() if qid in queries}
            log.info(f"  Run queries (overlap with qrels): {len(run):,}")

            if not run:
                log.warning("  No queries in run — skipping.")
                continue

            t_corpus = time.time()
            needed   = td.get_needed_docids(run)
            corpus   = td.load_corpus_selective(cfg["corpus"], cfg["corpus_format"], needed)
            t_corpus = time.time() - t_corpus

            ranked, timing = fixed100_rerank(
                model         = model,
                tokenizer     = tokenizer,
                true_id       = true_id,
                false_id      = false_id,
                device        = device,
                queries       = queries,
                run           = run,
                corpus        = corpus,
                top_k         = args.top_k,
                batch_size    = args.batch_size,
                max_input_len = args.max_input_len,
            )

            timing["dataset"]       = dataset_name
            timing["retriever"]     = retriever
            timing["model"]         = MODEL_NAME
            timing["corpus_load_s"] = round(t_corpus,     4)
            timing["model_load_s"]  = round(t_model_load, 4)
            timing["config"] = {
                "top_k"        : args.top_k,
                "batch_size"   : args.batch_size,
                "max_input_len": args.max_input_len,
            }

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
    log.info(f"Token IDs — ▁true: {true_id}  ▁false: {false_id}")

    run_tag_d5 = f"monot5_3b_duot5_3b_{retriever}"

    for dataset_name in datasets_to_run:
        cfg = DATASETS[dataset_name]

        log.info("")
        log.info("─" * 65)
        log.info(f"[duoT5] Dataset : {dataset_name}   Retriever : {retriever}")
        log.info("─" * 65)

        m5_run_file = monot5_run_path(dataset_name)
        if not m5_run_file.exists():
            log.warning(f"  monoT5 run not found: {m5_run_file} — skipping duoT5.")
            continue

        out_stem        = f"{dataset_name}.{retriever}.monot5_3b_duot5_3b.top{duot5_k}"
        out_file        = OUT_DIR / f"{out_stem}.txt"
        out_timing_file = OUT_DIR / f"{out_stem}.timing.json"

        if out_file.exists() and not args.force:
            log.info(f"  duoT5 output exists (--force to overwrite): {out_file.name}")
            continue

        valid_qids = td.load_qrel_qids(cfg["qrels"], cfg["qrels_format"])
        queries    = td.load_queries(cfg["queries"], cfg["query_format"], valid_qids)
        log.info(f"  Queries with qrels: {len(queries):,}")

        monot5_ranked = td.load_run(m5_run_file)
        monot5_ranked = {qid: docs for qid, docs in monot5_ranked.items() if qid in queries}
        log.info(f"  monoT5 run queries (overlap with qrels): {len(monot5_ranked):,}")

        if not monot5_ranked:
            log.warning("  No queries in monoT5 run — skipping.")
            continue

        # Load only the corpus docs needed for the duoT5 candidate set
        needed   = {docid for docs in monot5_ranked.values() for docid in docs[:duot5_k]}
        t_corpus = time.time()
        corpus   = td.load_corpus_selective(cfg["corpus"], cfg["corpus_format"], needed)
        t_corpus = time.time() - t_corpus

        ranked, timing = duot5_rerank(
            model         = model,
            tokenizer     = tokenizer,
            true_id       = true_id,
            false_id      = false_id,
            device        = device,
            queries       = queries,
            monot5_ranked = monot5_ranked,
            corpus        = corpus,
            top_k         = duot5_k,
            batch_size    = args.batch_size,
            max_input_len = args.max_input_len,
        )

        timing["dataset"]       = dataset_name
        timing["retriever"]     = retriever
        timing["monot5_model"]  = MODEL_NAME
        timing["duot5_model"]   = DUOT5_MODEL_NAME
        timing["corpus_load_s"] = round(t_corpus,     4)
        timing["duot5_load_s"]  = round(t_duot5_load, 4)
        timing["config"] = {
            "monot5_top_k" : args.top_k,
            "duot5_top_k"  : duot5_k,
            "batch_size"   : args.batch_size,
            "max_input_len": args.max_input_len,
        }

        td.write_trec_run(ranked, out_file, tag=run_tag_d5)
        td.write_timing(timing, out_timing_file)

    log.info("")
    log.info("All done.")


if __name__ == "__main__":
    main()
