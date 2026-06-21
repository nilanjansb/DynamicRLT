#!/usr/bin/env python3
"""
TDPart (Top-Down Partitioning) Reranking with RankZephyr or RankVicuna.
  Models: castorini/rank_zephyr_7b_v1_full  (default)
          castorini/rank_vicuna_7b_v1        (--model rankvicuna)

Algorithm  : Top-down pivot-based partitioning (Algorithm 1 from the paper).
Inference  : vLLM offline mode, TP=1 per process (or TP=N via --gpus N).
Input      : TREC top-100 run files from BM25 or SPLADE.
Output     : Reranked TREC run files in TDPart/runs/.
Resumeable : Checkpoint saved after each phase; safe to kill and restart.

File naming: {dataset}.{retriever}.{model_tag}.tdpart_w{W}k{K}.top{N}.txt
   e.g.    : dbpedia-entity.bm25.rankzephyr_7b.tdpart_w20k10.top100.txt
             dbpedia-entity.bm25.rankvicuna_7b.tdpart_w20k10.top100.txt

Phase structure:
  Phase 1 – First window  : 1 vLLM call   — rank first W docs; extract pivot at rank k
  Phase 2 – Parallel cmp  : 1 vLLM call   — rank ALL (pivot + W-1 docs) batches
  Phase 3 – Final sort    : ≥1 vLLM call  — sort candidate set A_i
                                             (1 call if |A_i| ≤ W; else a
                                              sliding window pass over A_i)

Budget note:
  The budget b caps |A_i| to keep Phase 3 tractable.  When budget is set,
  above-pivot docs that would exceed the cap are sent to B instead of A_i.
  All docs in R are still ranked in Phase 2 (they are queued before results
  are known); over-budget docs are just routed to B in post-processing.
  Set --budget equal to --window-size (default) to guarantee Phase 3 fits
  in a single window call.

Usage (prefer launch_tdpart.sh for multi-GPU runs):
    CUDA_VISIBLE_DEVICES=0 python3 tdpart.py \\
        --dataset dbpedia-entity --retriever bm25 --gpus 1
"""

import os
import re
import sys
import json
import time
import random
import datetime
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# This environment is pinned to the vLLM V0 engine because V1 pulls in a
# flashinfer/tvm_ffi stack that is ABI-incompatible on this machine.
os.environ.setdefault("VLLM_USE_V1", "0")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Paths & global constants
# ─────────────────────────────────────────────────────────────────────────────

BASE          = Path("/DATA/cs26int00020/Cultural_ablation")
DATA_DIR      = BASE / "1_Download_dataset/data"
RETRIEVER_DIR = BASE / "3_Initial_Retriever"
OUT_DIR       = BASE / "5_TDPart/runs"

MODELS = {
    "rankzephyr": "castorini/rank_zephyr_7b_v1_full",
    "rankvicuna":  "castorini/rank_vicuna_7b_v1",
}
WINDOW_SIZE   = 20    # W  — size of first window and each comparison window
PIVOT_K       = 10    # k  — 1-indexed rank of pivot within the first window (= W/2)
BUDGET        = 20    # b  — max |A_i| before Phase 3; set to W so A_i fits in one window
TOP_K         = 100   # docs per query to rerank
STRIDE        = 10    # stride for Phase 3 sliding window over A_i (only when |A_i| > W)
MAX_DOC_WORDS = 100   # per-document word budget inside the prompt


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry  (identical to Sliding_Window.py)
# ─────────────────────────────────────────────────────────────────────────────

DATASETS: Dict[str, dict] = {
    "msmarco-dl19": {
        "corpus":        DATA_DIR / "msmarco/collection.tsv",
        "corpus_format": "msmarco_tsv",
        "queries":       DATA_DIR / "msmarco/queries.dl19.tsv",
        "qrels":         DATA_DIR / "msmarco/qrels.dl19-passage.txt",
        "query_format":  "tsv",
        "qrels_format":  "trec",
    },
    "msmarco-dl20": {
        "corpus":        DATA_DIR / "msmarco/collection.tsv",
        "corpus_format": "msmarco_tsv",
        "queries":       DATA_DIR / "msmarco/queries.dl20.tsv",
        "qrels":         DATA_DIR / "msmarco/qrels.dl20-passage.txt",
        "query_format":  "tsv",
        "qrels_format":  "trec",
    },
    "scifact": {
        "corpus":        DATA_DIR / "beir/scifact/corpus.jsonl",
        "corpus_format": "beir_jsonl",
        "queries":       DATA_DIR / "beir/scifact/queries.jsonl",
        "qrels":         DATA_DIR / "beir/scifact/qrels/test.tsv",
        "query_format":  "jsonl",
        "qrels_format":  "beir_tsv",
    },
    "trec-covid": {
        "corpus":        DATA_DIR / "beir/trec-covid/corpus.jsonl",
        "corpus_format": "beir_jsonl",
        "queries":       DATA_DIR / "beir/trec-covid/queries.jsonl",
        "qrels":         DATA_DIR / "beir/trec-covid/qrels/test.tsv",
        "query_format":  "jsonl",
        "qrels_format":  "beir_tsv",
    },
    "webis-touche2020": {
        "corpus":        DATA_DIR / "beir/webis-touche2020/corpus.jsonl",
        "corpus_format": "beir_jsonl",
        "queries":       DATA_DIR / "beir/webis-touche2020/queries.jsonl",
        "qrels":         DATA_DIR / "beir/webis-touche2020/qrels/test.tsv",
        "query_format":  "jsonl",
        "qrels_format":  "beir_tsv",
    },
    "fever": {
        "corpus":        DATA_DIR / "beir/fever/corpus.jsonl",
        "corpus_format": "beir_jsonl",
        "queries":       DATA_DIR / "beir/fever/queries.jsonl",
        "qrels":         DATA_DIR / "beir/fever/qrels/test.tsv",
        "query_format":  "jsonl",
        "qrels_format":  "beir_tsv",
    },
    "dbpedia-entity": {
        "corpus":        DATA_DIR / "beir/dbpedia-entity/corpus.jsonl",
        "corpus_format": "beir_jsonl",
        "queries":       DATA_DIR / "beir/dbpedia-entity/queries.jsonl",
        "qrels":         DATA_DIR / "beir/dbpedia-entity/qrels/test.tsv",
        "query_format":  "jsonl",
        "qrels_format":  "beir_tsv",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers  (identical to Sliding_Window.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_qrel_qids(qrels_file: Path, fmt: str) -> set:
    qids: set = set()
    with open(qrels_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if fmt == "beir_tsv" and i == 0:
                continue
            parts = line.split("\t") if fmt == "beir_tsv" else line.split()
            qids.add(parts[0])
    return qids


def load_queries(
    query_file: Path,
    fmt: str,
    valid_qids: Optional[set] = None,
) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    if fmt == "tsv":
        with open(query_file) as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    queries[parts[0]] = parts[1]
    elif fmt == "jsonl":
        with open(query_file) as f:
            for line in f:
                obj = json.loads(line.strip())
                queries[str(obj["_id"])] = obj["text"]
    if valid_qids is not None:
        queries = {q: t for q, t in queries.items() if q in valid_qids}
    return queries


def load_run(run_file: Path) -> Dict[str, List[str]]:
    run: Dict[str, Dict[int, str]] = {}
    with open(run_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, docid, rank = parts[0], parts[2], int(parts[3])
            run.setdefault(qid, {})[rank] = docid
    return {qid: [run[qid][r] for r in sorted(run[qid])] for qid in run}


def get_needed_docids(run: Dict[str, List[str]]) -> set:
    return {docid for docs in run.values() for docid in docs}


def load_corpus_selective(
    corpus_file: Path,
    fmt: str,
    needed_ids: set,
) -> Dict[str, str]:
    corpus: Dict[str, str] = {}
    total_needed = len(needed_ids)
    log.info(f"  Loading {total_needed:,} docs from {corpus_file.name} …")
    t0 = time.time()

    if fmt == "msmarco_tsv":
        with open(corpus_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sep = line.find("\t")
                if sep == -1:
                    continue
                docid = line[:sep]
                if docid in needed_ids:
                    corpus[docid] = line[sep + 1:]
                    if len(corpus) == total_needed:
                        break
    elif fmt == "beir_jsonl":
        with open(corpus_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj   = json.loads(line)
                docid = str(obj["_id"])
                if docid in needed_ids:
                    title = obj.get("title", "").strip()
                    text  = obj.get("text",  "").strip()
                    corpus[docid] = f"{title} {text}".strip() if title else text
                    if len(corpus) == total_needed:
                        break

    missing = total_needed - len(corpus)
    elapsed = time.time() - t0
    if missing:
        log.warning(f"  {missing:,} docids not found in corpus.")
    log.info(f"  Loaded {len(corpus):,} docs in {elapsed:.1f}s.")
    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction  (identical to Sliding_Window.py)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_MSG = (
    "You are RankLLM, an intelligent assistant that can rank passages "
    "based on their relevancy to the query"
)

# Vicuna v1 chat template — rank_vicuna tokenizer has no built-in template
VICUNA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] }} "
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "USER: {{ message['content'] }} "
    "{% elif message['role'] == 'assistant' %}"
    "ASSISTANT: {{ message['content'] }}</s>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}ASSISTANT:{% endif %}"
)


def truncate_text(text: str, max_words: int = MAX_DOC_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def build_ranking_prompt(
    query: str,
    window_docs: List[Tuple[str, str]],
) -> str:
    num = len(window_docs)
    prompt = (
        f"I will provide you with {num} passages, each indicated by a numerical "
        f"identifier []. Rank the passages based on their relevance to the search "
        f"query: {query}.\n\n"
    )
    for rank_i, (docid, text) in enumerate(window_docs, 1):
        prompt += f"[{rank_i}] {truncate_text(text, MAX_DOC_WORDS)}\n"
    prompt += (
        f"\nSearch Query: {query}.\n"
        f"Rank the {num} passages above based on their relevance to the search query. "
        f"All the passages should be included and listed using identifiers, in "
        f"descending order of relevance. The output format should be [] > [], e.g., "
        f"[2] > [1], Answer concisely and directly and only respond with the ranking "
        f"results, do not say any word or explain."
    )
    return prompt


def build_chat_messages(
    query: str,
    window_docs: List[Tuple[str, str]],
) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": build_ranking_prompt(query, window_docs)},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Permutation parsing  (identical to Sliding_Window.py)
# ─────────────────────────────────────────────────────────────────────────────

_RANK_RE  = re.compile(r"\[(\d+)\]")
_VALID_RE = re.compile(r"^\[\d+\]( > \[\d+\])*$")


def parse_permutation(output: str, window_size: int) -> List[int]:
    if not _VALID_RE.match(output.strip()):
        log.debug(f"Non-standard model output: {output[:80]!r}")
    found = [int(x) for x in _RANK_RE.findall(output)]
    seen: set = set()
    perm: List[int] = []
    for idx in found:
        if 1 <= idx <= window_size and idx not in seen:
            perm.append(idx)
            seen.add(idx)
    for i in range(1, window_size + 1):
        if i not in seen:
            perm.append(i)
    return perm


def apply_permutation(window_docids: List[str], perm: List[int]) -> List[str]:
    return [window_docids[i - 1] for i in perm]


# ─────────────────────────────────────────────────────────────────────────────
# TDPart helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_prompt(
    tokenizer,
    query: str,
    window_docids: List[str],
    corpus: Dict[str, str],
    max_model_len: Optional[int] = None,
    output_reserve: int = 200,
) -> str:
    """Render a ranked window into a vLLM-ready text prompt.

    If max_model_len is provided, progressively reduce the per-document word
    budget until the prompt fits within the reserved input budget.
    """
    window_texts = [corpus.get(d, "") for d in window_docids]
    word_budget = MAX_DOC_WORDS

    while True:
        window_docs = [
            (docid, truncate_text(text, word_budget))
            for docid, text in zip(window_docids, window_texts)
        ]
        messages = build_chat_messages(query, window_docs)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if max_model_len is None:
            return prompt

        n_tok = len(tokenizer.encode(prompt))
        if n_tok <= max_model_len - output_reserve or word_budget <= 1:
            if n_tok > max_model_len - output_reserve:
                log.warning(
                    f"  Prompt still exceeds budget after shrinking to {word_budget} "
                    f"word/doc: {n_tok} tokens > {max_model_len - output_reserve}."
                )
            return prompt

        # Shrink aggressively enough to converge in a few iterations.
        shrink_ratio = (max_model_len - output_reserve) / max(n_tok, 1)
        next_budget = max(1, min(word_budget - 1, int(word_budget * shrink_ratio * 0.95)))
        word_budget = next_budget


def partition_on_pivot(
    reordered : List[str],
    pivot     : str,
) -> Tuple[List[str], List[str]]:
    """
    Given a reranked window that contains the pivot, split into:
      above_pivot : docs ranked strictly above (better than) the pivot
      below_pivot : docs ranked strictly below (worse than) the pivot

    If the pivot is missing from the output (model error), all docs go below.
    """
    try:
        p_pos = reordered.index(pivot)
    except ValueError:
        log.debug(f"Pivot {pivot!r} missing from reranked window — treating all as below.")
        return [], [d for d in reordered if d != pivot]
    return reordered[:p_pos], reordered[p_pos + 1:]


def sliding_window_steps(list_len: int, window_size: int, stride: int) -> List[int]:
    """
    Return 0-indexed start positions for a bottom-up sliding window over
    a list of length list_len.  Same logic as Sliding_Window.py.
    """
    start  = list_len - window_size
    starts : List[int] = []
    while start > 0:
        starts.append(start)
        start -= stride
    starts.append(0)
    return starts


# ─────────────────────────────────────────────────────────────────────────────
# TDPart main algorithm
# ─────────────────────────────────────────────────────────────────────────────

def tdpart_rerank(
    llm,
    tokenizer,
    queries      : Dict[str, str],
    run          : Dict[str, List[str]],
    corpus       : Dict[str, str],
    sampling_params,
    window_size  : int           = WINDOW_SIZE,
    pivot_k      : int           = PIVOT_K,
    budget       : Optional[int] = BUDGET,
    top_k        : int           = TOP_K,
    stride           : int           = STRIDE,
    max_model_len    : int           = 4096,
    ckpt_file        : Optional[Path] = None,
) -> Tuple[Dict[str, List[str]], dict]:
    """
    TDPart (Top-Down Partitioning) reranking — Algorithm 1.

    Strategy
    --------
    Phase 1 — First window (1 vLLM call):
      Rank the top-W docs for every query.  Extract the pivot at 1-indexed
      rank pivot_k.  Docs ranked above pivot → A_i; below → B.

    Phase 2 — Parallel comparison batches (1 vLLM call):
      Split remaining docs into batches of W-1.  Prepend pivot as the first doc
      in each batch (making it W docs total).  Rank ALL batches for ALL queries
      in one call.  Route above-pivot → A_i, below-pivot → B.  Budget cap applies here.

    Phase 3 — Final sort of A_i (≥1 vLLM call):
      If |A_i| ≤ W: one call to sort all queries' A_i together.
      If |A_i| > W: a bottom-up sliding window pass over A_i (batched across
      queries per step; same approach as Sliding_Window.py).

    Final list: A_i_sorted  +  [pivot]  +  B
      (any docs in R that were skipped due to budget are appended to B)

    Returns
    -------
    (ranked, timing)
      ranked : {qid: [docid_rank1, …, docid_rank_top_k]}
      timing : structured dict with per-phase and aggregate timing info
    """
    # ── Initialise per-query state ───────────────────────────────────────────
    initial_docs: Dict[str, List[str]] = {
        qid: list(docs[:top_k])
        for qid, docs in run.items()
        if qid in queries
    }
    qids_ordered = list(initial_docs.keys())
    n_queries    = len(qids_ordered)

    # Working state — populated during phases
    A      : Dict[str, List[str]] = {qid: [] for qid in qids_ordered}
    B      : Dict[str, List[str]] = {qid: [] for qid in qids_ordered}
    pivots : Dict[str, str]       = {}
    # R[qid] = docs not yet compared; starts as initial_docs[qid][window_size:]
    R      : Dict[str, List[str]] = {qid: [] for qid in qids_ordered}

    # ── Checkpoint: resume from a previous run ───────────────────────────────
    resume_phase   = 1
    phase_timings: List[dict] = []

    if ckpt_file and ckpt_file.exists():
        try:
            ckpt         = json.loads(ckpt_file.read_text())
            resume_phase = int(ckpt.get("next_phase", 1))
            phase_timings = ckpt.get("phase_timings", [])
            A.update(ckpt.get("A", {}))
            B.update(ckpt.get("B", {}))
            pivots.update(ckpt.get("pivots", {}))
            R.update(ckpt.get("R", {}))
            log.info(
                f"  Checkpoint loaded: resuming from Phase {resume_phase}."
            )
        except Exception as e:
            log.warning(f"  Could not load checkpoint ({e}); starting from Phase 1.")
            resume_phase  = 1
            phase_timings = []

    def save_ckpt(next_phase: int) -> None:
        if not ckpt_file:
            return
        try:
            ckpt_file.write_text(json.dumps({
                "next_phase"   : next_phase,
                "A"            : A,
                "B"            : B,
                "pivots"       : pivots,
                "R"            : R,
                "phase_timings": phase_timings,
            }))
        except Exception as e:
            log.warning(f"  Checkpoint write failed: {e}")

    wall_start     = time.time()
    wall_start_iso = datetime.datetime.now().isoformat(timespec="seconds")

    # ── Prompt length safety check (3 random queries, first window) ─────────
    if resume_phase <= 1:
        sample_qids = random.sample(list(initial_docs.keys()), min(3, len(initial_docs)))
        max_seen = 0
        for qid in sample_qids:
            fw   = initial_docs[qid][:window_size]
            text = make_prompt(
                tokenizer, queries[qid], fw, corpus, max_model_len=max_model_len
            )
            n_tok = len(tokenizer.encode(text))
            max_seen = max(max_seen, n_tok)
            if n_tok > max_model_len - 200:
                log.warning(
                    f"  LONG PROMPT: qid={qid} → {n_tok} tokens "
                    f"(budget={max_model_len - 200}). "
                    f"Consider reducing --max-doc-words."
                )
        log.info(
            f"  Prompt length check: max sample = {max_seen} tokens "
            f"(limit {max_model_len - 200} input + 200 output = {max_model_len})."
        )

    # ── Phase 1: First window ─────────────────────────────────────────────────
    if resume_phase <= 1:
        t0 = time.time()
        log.info(
            f"  Phase 1 — first window: batching {n_queries:,} prompts "
            f"(W={window_size}, pivot_k={pivot_k}) …"
        )

        prompts       : List[str]        = []
        first_windows : Dict[str, List[str]] = {}

        for qid in qids_ordered:
            docs = initial_docs[qid]
            fw   = docs[:window_size]
            R[qid]            = docs[window_size:]
            first_windows[qid] = fw
            prompts.append(
                make_prompt(
                    tokenizer, queries[qid], fw, corpus, max_model_len=max_model_len
                )
            )

        t_build = time.time() - t0
        t_infer_start = time.time()
        ph1_outputs = llm.generate(prompts, sampling_params)
        t_infer = time.time() - t_infer_start

        t_parse_start = time.time()
        for qid, output in zip(qids_ordered, ph1_outputs):
            fw        = first_windows[qid]
            eff_w     = len(fw)
            generated = output.outputs[0].text.strip()
            perm      = parse_permutation(generated, eff_w)
            reordered = apply_permutation(fw, perm)

            # Pivot index is 0-based: pivot_k - 1 (clamped for short lists)
            p_idx         = min(pivot_k - 1, eff_w - 1)
            pivots[qid]   = reordered[p_idx]
            A[qid]        = list(reordered[:p_idx])       # ranks 1 … k-1
            B[qid]        = list(reordered[p_idx + 1:])   # ranks k+1 … W
        t_parse = time.time() - t_parse_start

        elapsed = time.time() - t0
        log.info(
            f"  Phase 1 done in {elapsed:.2f}s  "
            f"[build={t_build:.2f}s  infer={t_infer:.2f}s  parse={t_parse:.2f}s]"
        )
        phase_timings.append({
            "phase"           : 1,
            "label"           : "first_window",
            "n_prompts"       : n_queries,
            "prompt_build_s"  : round(t_build,  4),
            "vllm_infer_s"    : round(t_infer,  4),
            "permute_parse_s" : round(t_parse,  4),
            "phase_total_s"   : round(elapsed,  4),
        })
        save_ckpt(next_phase=2)

    # ── Phase 2: Parallel comparison batches ─────────────────────────────────
    if resume_phase <= 2:
        t0 = time.time()

        # Pre-build every comparison batch for every query in one pass.
        # Each batch = [pivot] + (up to W-1 docs from R).
        batch_prompts : List[str]              = []
        batch_meta    : List[Tuple[str, List[str]]] = []  # (qid, window_with_pivot_first)

        for qid in qids_ordered:
            pivot   = pivots[qid]
            r_docs  = list(R[qid])
            batch_size = window_size - 1
            for i in range(0, len(r_docs), batch_size):
                chunk  = r_docs[i : i + batch_size]
                window = [pivot] + chunk          # pivot prepended → always in position [1]
                batch_prompts.append(
                    make_prompt(
                        tokenizer,
                        queries[qid],
                        window,
                        corpus,
                        max_model_len=max_model_len,
                    )
                )
                batch_meta.append((qid, window))

        n_batches = len(batch_prompts)
        log.info(
            f"  Phase 2 — parallel comparison: {n_batches:,} batches "
            f"across {n_queries:,} queries (1 vLLM call) …"
        )

        t_build = time.time() - t0
        t_infer_start = time.time()
        if batch_prompts:
            ph2_outputs = llm.generate(batch_prompts, sampling_params)
        else:
            ph2_outputs = []
        t_infer = time.time() - t_infer_start

        t_parse_start = time.time()
        for (qid, window), output in zip(batch_meta, ph2_outputs):
            pivot     = pivots[qid]
            eff_w     = len(window)
            generated = output.outputs[0].text.strip()
            perm      = parse_permutation(generated, eff_w)
            reordered = apply_permutation(window, perm)

            above, below = partition_on_pivot(reordered, pivot)

            # Budget cap: if A_i is already full, route above-pivot docs to B
            if budget is not None:
                space = max(0, budget - len(A[qid]))
                A[qid].extend(above[:space])
                B[qid].extend(above[space:])   # over-budget above → backfill
            else:
                A[qid].extend(above)
            B[qid].extend(below)
        t_parse = time.time() - t_parse_start

        elapsed = time.time() - t0
        batches_per_query = n_batches / n_queries if n_queries else 0
        log.info(
            f"  Phase 2 done in {elapsed:.2f}s  "
            f"[build={t_build:.2f}s  infer={t_infer:.2f}s  parse={t_parse:.2f}s]  "
            f"avg {batches_per_query:.1f} batches/query"
        )
        phase_timings.append({
            "phase"            : 2,
            "label"            : "parallel_comparison",
            "n_batches_total"  : n_batches,
            "avg_batches_query": round(batches_per_query, 2),
            "prompt_build_s"   : round(t_build,  4),
            "vllm_infer_s"     : round(t_infer,  4),
            "permute_parse_s"  : round(t_parse,  4),
            "phase_total_s"    : round(elapsed,  4),
        })
        save_ckpt(next_phase=3)

    # ── Phase 3: Final sort of A_i ────────────────────────────────────────────
    if resume_phase <= 3:
        t0 = time.time()

        ai_sizes      = {qid: len(A[qid]) for qid in qids_ordered}
        sort_needed   = [qid for qid in qids_ordered if ai_sizes[qid] > 1]
        trivial       = [qid for qid in qids_ordered if ai_sizes[qid] <= 1]
        max_ai        = max(ai_sizes.values()) if ai_sizes else 0

        log.info(
            f"  Phase 3 — final sort: {len(sort_needed):,} queries need sorting "
            f"({len(trivial):,} trivial), max |A_i|={max_ai}"
        )

        ph3_calls   = 0
        t_build_p3  = 0.0
        t_infer_p3  = 0.0
        t_parse_p3  = 0.0

        if max_ai <= window_size:
            # All A_i fit in one window → 1 vLLM call for all queries
            _tb = time.time()
            prompts_p3: List[str] = []
            sort_qids_p3: List[str] = []
            for qid in sort_needed:
                prompts_p3.append(
                    make_prompt(
                        tokenizer,
                        queries[qid],
                        A[qid],
                        corpus,
                        max_model_len=max_model_len,
                    )
                )
                sort_qids_p3.append(qid)
            t_build_p3 += time.time() - _tb

            if prompts_p3:
                ph3_calls += 1
                _ti = time.time()
                ph3_outputs = llm.generate(prompts_p3, sampling_params)
                t_infer_p3 += time.time() - _ti

                _tp = time.time()
                for qid, output in zip(sort_qids_p3, ph3_outputs):
                    generated = output.outputs[0].text.strip()
                    perm      = parse_permutation(generated, len(A[qid]))
                    A[qid]    = apply_permutation(A[qid], perm)
                t_parse_p3 += time.time() - _tp

        else:
            # Some A_i exceed window_size → sliding window pass over each A_i.
            # Steps are sequential within a query (each depends on the previous),
            # but all queries are batched together per step.
            step_starts = sliding_window_steps(max_ai, window_size, stride)
            log.info(
                f"  Phase 3 sliding window over A_i: {len(step_starts)} steps "
                f"(max |A_i|={max_ai}, W={window_size}, stride={stride})"
            )

            for step_idx, start in enumerate(step_starts, 1):
                end       = start + window_size
                _tb = time.time()
                prompts_s : List[str] = []
                step_qids : List[str] = []

                for qid in sort_needed:
                    ai = A[qid]
                    a_start = min(start, len(ai))
                    a_end   = min(end,   len(ai))
                    chunk   = ai[a_start:a_end]
                    if not chunk:
                        continue
                    prompts_s.append(
                        make_prompt(
                            tokenizer,
                            queries[qid],
                            chunk,
                            corpus,
                            max_model_len=max_model_len,
                        )
                    )
                    step_qids.append(qid)
                t_build_p3 += time.time() - _tb

                if not prompts_s:
                    continue

                ph3_calls += 1
                _ti = time.time()
                step_outputs = llm.generate(prompts_s, sampling_params)
                t_infer_p3 += time.time() - _ti

                _tp = time.time()
                for qid, output in zip(step_qids, step_outputs):
                    ai      = A[qid]
                    a_start = min(start, len(ai))
                    a_end   = min(end,   len(ai))
                    chunk   = ai[a_start:a_end]
                    if not chunk:
                        continue
                    generated = output.outputs[0].text.strip()
                    perm      = parse_permutation(generated, len(chunk))
                    reordered = apply_permutation(chunk, perm)
                    A[qid]    = ai[:a_start] + reordered + ai[a_end:]
                t_parse_p3 += time.time() - _tp

                log.info(
                    f"  Phase 3 step {step_idx}/{len(step_starts)}: "
                    f"A_i[{start}:{end}] — {len(prompts_s)} prompts"
                )

        elapsed = time.time() - t0
        log.info(
            f"  Phase 3 done in {elapsed:.2f}s  "
            f"[build={t_build_p3:.2f}s  infer={t_infer_p3:.2f}s  parse={t_parse_p3:.2f}s]  "
            f"({ph3_calls} vLLM call(s))"
        )
        phase_timings.append({
            "phase"           : 3,
            "label"           : "final_sort",
            "n_vllm_calls"    : ph3_calls,
            "max_ai_size"     : max_ai,
            "prompt_build_s"  : round(t_build_p3, 4),
            "vllm_infer_s"    : round(t_infer_p3, 4),
            "permute_parse_s" : round(t_parse_p3, 4),
            "phase_total_s"   : round(elapsed,    4),
        })
        save_ckpt(next_phase=4)   # 4 = done

    # ── Assemble final ranked list ────────────────────────────────────────────
    # Final order: A_i (sorted candidates) + pivot + B (backfill)
    ranked: Dict[str, List[str]] = {}
    for qid in qids_ordered:
        ranked[qid] = A[qid] + [pivots[qid]] + B[qid]

    # ── Aggregate timing ──────────────────────────────────────────────────────
    total_s          = time.time() - wall_start
    wall_end_iso     = datetime.datetime.now().isoformat(timespec="seconds")
    total_infer_s    = sum(p.get("vllm_infer_s", 0) for p in phase_timings)
    avg_s_per_query  = total_s / n_queries if n_queries else 0

    log.info(
        f"\n  ── TDPart summary ──────────────────────────────────────\n"
        f"  Queries          : {n_queries:,}\n"
        f"  Total wall time  : {total_s:.2f}s\n"
        f"  Avg per query    : {avg_s_per_query * 1000:.1f}ms\n"
        f"  vLLM infer total : {total_infer_s:.2f}s  "
        f"({total_infer_s / total_s * 100:.1f}% of wall time)\n"
        f"  ─────────────────────────────────────────────────────────"
    )

    total_build_s = sum(p.get("prompt_build_s", 0) for p in phase_timings)

    timing = {
        "wall_clock_start"    : wall_start_iso,
        "wall_clock_end"      : wall_end_iso,
        "total_rerank_s"      : round(total_s,                  4),
        "avg_s_per_query"     : round(avg_s_per_query,          4),
        "avg_ms_per_query"    : round(avg_s_per_query * 1000,   2),
        "total_vllm_infer_s"  : round(total_infer_s,            4),
        "total_prompt_build_s": round(total_build_s,            4),
        "pct_time_in_vllm"    : round(total_infer_s / total_s * 100, 2) if total_s else 0,
        "n_queries"           : n_queries,
        "n_phases"            : len(phase_timings),
        "phases"              : phase_timings,
    }

    if ckpt_file and ckpt_file.exists():
        ckpt_file.unlink()
        log.info("  Checkpoint deleted (run complete).")

    return ranked, timing


# ─────────────────────────────────────────────────────────────────────────────
# TREC output  (identical to Sliding_Window.py)
# ─────────────────────────────────────────────────────────────────────────────

def write_trec_run(
    ranked      : Dict[str, List[str]],
    output_file : Path,
    tag         : str = "TDPart",
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def sort_key(qid: str):
        return int(qid) if qid.isdigit() else qid

    with open(output_file, "w") as f:
        for qid in sorted(ranked.keys(), key=sort_key):
            docs = ranked[qid]
            for rank, docid in enumerate(docs, 1):
                score = len(docs) - rank + 1
                f.write(f"{qid} Q0 {docid} {rank} {score} {tag}\n")

    log.info(f"  Written → {output_file.name}")


def write_timing(timing: dict, output_file: Path) -> None:
    with open(output_file, "w") as f:
        json.dump(timing, f, indent=2)
    log.info(f"  Timing  → {output_file.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TDPart Reranking with RankZephyr or RankVicuna (vLLM)",
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
    p.add_argument("--top-k",       type=int,   default=TOP_K,        help="Docs to rerank per query.")
    p.add_argument("--window-size", type=int,   default=WINDOW_SIZE,  help="Window size W.")
    p.add_argument("--pivot-k",     type=int,   default=PIVOT_K,      help="1-indexed pivot rank within first window (default W/2).")
    p.add_argument("--budget",      type=int,   default=BUDGET,       help="Max |A_i| candidate cap. Default=window-size. Set 0 for no budget.")
    p.add_argument("--stride",      type=int,   default=10,           help="Stride for Phase 3 sliding window over A_i (only used when |A_i|>W).")
    p.add_argument(
        "--gpus", type=int, default=1,
        help="Tensor parallel size for vLLM.",
    )
    p.add_argument("--gpu-mem-util",  type=float, default=0.90)
    p.add_argument("--max-model-len", type=int,   default=4096)
    p.add_argument("--max-num-seqs",  type=int,   default=512)
    p.add_argument("--max-doc-words", type=int,   default=MAX_DOC_WORDS)
    p.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    p.add_argument(
        "--model", default="rankzephyr", choices=list(MODELS.keys()),
        help="Reranker model to use. Default: rankzephyr.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    global MAX_DOC_WORDS
    MAX_DOC_WORDS = args.max_doc_words

    # Budget: 0 means no cap
    budget = None if args.budget == 0 else args.budget

    raw = args.dataset
    if raw == ["all"] or raw == "all":
        datasets_to_run = list(DATASETS.keys())
    else:
        datasets_to_run = raw if isinstance(raw, list) else [raw]
        unknown = [d for d in datasets_to_run if d not in DATASETS]
        if unknown:
            log.error(f"Unknown dataset(s): {unknown}. Valid: {list(DATASETS.keys())}")
            sys.exit(1)

    retriever  = args.retriever
    W, K, B    = args.window_size, args.top_k, budget
    model_name = MODELS[args.model]
    model_tag  = args.model + "_7b"
    model_disp = "RankZephyr_7B" if args.model == "rankzephyr" else "RankVicuna_7B"

    log.info("=" * 65)
    log.info(f"Model   : {model_name}")
    log.info(f"GPUs    : {args.gpus}  |  mem_util={args.gpu_mem_util}  "
             f"|  max_len={args.max_model_len}  |  max_seqs={args.max_num_seqs}")
    log.info(f"TDPart  : W={W}  pivot_k={args.pivot_k}  budget={B}  top_k={K}")
    log.info("=" * 65)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        log.error("vLLM is not installed. Install with: pip install vllm")
        sys.exit(1)

    from transformers import AutoTokenizer

    log.info("Loading tokenizer …")
    t_model_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.model == "rankvicuna" and not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = VICUNA_CHAT_TEMPLATE

    log.info("Initialising vLLM engine …")
    llm = LLM(
        model                  = model_name,
        tensor_parallel_size   = args.gpus,
        dtype                  = "bfloat16",
        max_model_len          = args.max_model_len,
        gpu_memory_utilization = args.gpu_mem_util,
        enforce_eager          = True,
        trust_remote_code      = False,
        max_num_seqs           = args.max_num_seqs,
        enable_chunked_prefill = True,
    )
    t_model_load = time.time() - t_model_load
    log.info(f"Model + tokenizer loaded in {t_model_load:.1f}s.")

    sampling_params = SamplingParams(
        temperature         = 0.0,
        max_tokens          = 200,
        skip_special_tokens = True,
    )

    run_tag = f"{model_disp}_{retriever}_tdpart_w{W}k{args.pivot_k}"

    for dataset_name in datasets_to_run:
        cfg = DATASETS[dataset_name]

        log.info("")
        log.info("─" * 65)
        log.info(f"Dataset  : {dataset_name}   Retriever : {retriever}")
        log.info("─" * 65)

        run_file = RETRIEVER_DIR / retriever / f"{dataset_name}.top{K}.txt"
        if not run_file.exists():
            log.warning(f"Input run not found: {run_file} — skipping.")
            continue

        out_stem        = f"{dataset_name}.{retriever}.{model_tag}.tdpart_w{W}k{args.pivot_k}.top{K}"
        out_file        = OUT_DIR / f"{out_stem}.txt"
        out_timing_file = OUT_DIR / f"{out_stem}.timing.json"

        if out_file.exists() and not args.force:
            log.info(f"Output already exists (use --force to overwrite): {out_file.name}")
            continue

        valid_qids = load_qrel_qids(cfg["qrels"], cfg["qrels_format"])
        queries    = load_queries(cfg["queries"], cfg["query_format"], valid_qids)
        log.info(f"  Queries with qrels: {len(queries):,}")

        run = load_run(run_file)
        run = {qid: docs for qid, docs in run.items() if qid in queries}
        log.info(f"  Run queries (overlap with qrels): {len(run):,}")

        if not run:
            log.warning("  No queries in run — skipping.")
            continue

        t_corpus = time.time()
        needed   = get_needed_docids(run)
        corpus   = load_corpus_selective(cfg["corpus"], cfg["corpus_format"], needed)
        t_corpus = time.time() - t_corpus

        ckpt_file = OUT_DIR / f"{out_stem}.ckpt.json"

        ranked, timing = tdpart_rerank(
            llm             = llm,
            tokenizer       = tokenizer,
            queries         = queries,
            run             = run,
            corpus          = corpus,
            sampling_params = sampling_params,
            window_size     = W,
            pivot_k         = args.pivot_k,
            budget          = B,
            top_k           = K,
            stride          = args.stride,
            max_model_len   = args.max_model_len,
            ckpt_file       = ckpt_file,
        )

        timing["dataset"]       = dataset_name
        timing["retriever"]     = retriever
        timing["model"]         = model_name
        timing["corpus_load_s"] = round(t_corpus,     4)
        timing["model_load_s"]  = round(t_model_load, 4)
        timing["config"] = {
            "window_size"          : W,
            "pivot_k"              : args.pivot_k,
            "budget"               : B,
            "top_k"                : K,
            "stride"               : args.stride,
            "tensor_parallel_size" : args.gpus,
            "gpu_memory_util"      : args.gpu_mem_util,
            "max_model_len"        : args.max_model_len,
            "max_num_seqs"         : args.max_num_seqs,
        }

        write_trec_run(ranked, out_file, tag=run_tag)
        write_timing(timing, out_timing_file)

    log.info("")
    log.info("All done.")


if __name__ == "__main__":
    main()
