#!/usr/bin/env python3
"""
Sliding Window Listwise Reranking with RankZephyr or RankVicuna.
  Models: castorini/rank_zephyr_7b_v1_full  (default)
          castorini/rank_vicuna_7b_v1        (--model rankvicuna)

Algorithm : Bottom-up sliding window (window=20, stride=10).
Inference  : vLLM offline mode, TP=1 per process (run 3 parallel processes, one per GPU).
Input      : TREC top-100 run files from BM25 or SPLADE.
Output     : Reranked TREC run files in Sliding_Window/runs/.
Resumeable : Checkpoint saved after every window step; safe to kill and restart.

File naming: {dataset}.{retriever}.{model_tag}.w{W}s{S}.top{K}.txt
   e.g.    : msmarco-dl19.bm25.rankzephyr_7b.w20s10.top100.txt
             msmarco-dl19.bm25.rankvicuna_7b.w20s10.top100.txt

Token budget (verified safe for window=20, max_doc_words=100):
   20 docs × ~130 tok + overhead ~200 tok ≈ 2800 tok < max_model_len 4096.
   Output permutation is ≤200 tokens.  No truncation occurs.

Usage (prefer launch_sliding_window.sh for multi-GPU parallel runs):
    CUDA_VISIBLE_DEVICES=0 python3 Sliding_Window.py \\
        --dataset fever --retriever bm25 --gpus 1

    CUDA_VISIBLE_DEVICES=1 python3 Sliding_Window.py \\
        --dataset dbpedia-entity scifact msmarco-dl20 --retriever bm25 --gpus 1
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

os.environ.setdefault("VLLM_USE_V1", "0")

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Paths & global constants
# ──────────────────────────────────────────────────────────────────────────────

BASE          = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parents[1]))
DATA_DIR      = BASE / "1_Download_dataset/data"
RETRIEVER_DIR = BASE / "3_Initial_Retriever"
OUT_DIR       = BASE / "4_Sliding_Window/runs"

MODELS = {
    "rankzephyr": "castorini/rank_zephyr_7b_v1_full",
    "rankvicuna":  "castorini/rank_vicuna_7b_v1",
}
WINDOW_SIZE  = 20
STRIDE       = 10
TOP_K        = 100
MAX_DOC_WORDS = 100   # per-document word budget inside the prompt


# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ──────────────────────────────────────────────────────────────────────────────

DATASETS: Dict[str, dict] = {
    "msmarco-dl19": {
        "corpus":        DATA_DIR / "msmarco/collection.tsv",
        "corpus_format": "msmarco_tsv",       # pid<TAB>text  (no header)
        "queries":       DATA_DIR / "msmarco/queries.dl19.tsv",
        "qrels":         DATA_DIR / "msmarco/qrels.dl19-passage.txt",
        "query_format":  "tsv",               # qid<TAB>text
        "qrels_format":  "trec",              # qid 0 docid rel (space-sep, no header)
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
        "corpus_format": "beir_jsonl",        # {"_id","title","text"}
        "queries":       DATA_DIR / "beir/scifact/queries.jsonl",
        "qrels":         DATA_DIR / "beir/scifact/qrels/test.tsv",
        "query_format":  "jsonl",
        "qrels_format":  "beir_tsv",          # header + qid<TAB>docid<TAB>rel
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


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_qrel_qids(qrels_file: Path, fmt: str) -> set:
    """Return set of query IDs that have at least one relevance judgment."""
    qids: set = set()
    with open(qrels_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if fmt == "beir_tsv" and i == 0:
                continue                        # skip TSV header
            parts = line.split("\t") if fmt == "beir_tsv" else line.split()
            qids.add(parts[0])
    return qids


def load_queries(
    query_file: Path,
    fmt: str,
    valid_qids: Optional[set] = None,
) -> Dict[str, str]:
    """Load queries as {qid: text}, optionally filtered to valid_qids."""
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
    """
    Parse a TREC run file into {qid: [docid_rank1, docid_rank2, ...]}.
    Documents are returned in ascending rank order (rank 1 first).
    """
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
    """
    Scan the corpus file and load only the documents whose IDs are in
    needed_ids.  Returns {docid: text_for_prompt}.
    """
    corpus: Dict[str, str] = {}
    total_needed = len(needed_ids)
    log.info(f"  Loading {total_needed:,} docs from {corpus_file.name} …")
    t0 = time.time()

    if fmt == "msmarco_tsv":
        # Format: docid<TAB>text  (no header, ~8.8 M lines)
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
        # Format: {"_id": ..., "title": ..., "text": ...}
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


# ──────────────────────────────────────────────────────────────────────────────
# RankZephyr prompt construction  (exact format from rank_llm)
# ──────────────────────────────────────────────────────────────────────────────

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
    """Truncate to max_words words to keep prompt within model context."""
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def build_ranking_prompt(
    query: str,
    window_docs: List[Tuple[str, str]],   # [(docid, text), ...]
) -> str:
    """
    Build the RankLLM / RankZephyr listwise ranking prompt.

    Format (singleturn_listwise):
      prefix  : I will provide you with {num} passages …
      body    : [{rank}] {candidate}   (one per document)
      suffix  : Search Query: {query}.  Rank … Only respond …
    """
    num = len(window_docs)

    # Prefix
    prompt = (
        f"I will provide you with {num} passages, each indicated by a numerical "
        f"identifier []. Rank the passages based on their relevance to the search "
        f"query: {query}.\n\n"
    )

    # Body – one line per document
    for rank_i, (docid, text) in enumerate(window_docs, 1):
        truncated = truncate_text(text, MAX_DOC_WORDS)
        prompt += f"[{rank_i}] {truncated}\n"

    # Suffix
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
    """Wrap the ranking prompt in a Zephyr-style chat message list."""
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": build_ranking_prompt(query, window_docs)},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Output parsing
# ──────────────────────────────────────────────────────────────────────────────

# Matches the extraction regex from the rank_llm config
_RANK_RE = re.compile(r"\[(\d+)\]")

# Validates the full output is well-formed (used only for logging)
_VALID_RE = re.compile(r"^\[\d+\]( > \[\d+\])*$")


def parse_permutation(output: str, window_size: int) -> List[int]:
    """
    Parse model output '[3] > [1] > [7] > …' into a 1-indexed permutation list.

    Contract:
     - Returns a list of length == window_size.
     - Duplicates are removed (first occurrence wins).
     - Missing indices are appended at the end in original order.
    """
    if not _VALID_RE.match(output.strip()):
        log.debug(f"Non-standard model output: {output[:80]!r}")

    found  = [int(x) for x in _RANK_RE.findall(output)]
    seen   : set = set()
    perm   : List[int] = []

    for idx in found:
        if 1 <= idx <= window_size and idx not in seen:
            perm.append(idx)
            seen.add(idx)

    # Append any identifiers the model forgot (preserves original relative order)
    for i in range(1, window_size + 1):
        if i not in seen:
            perm.append(i)

    return perm


def apply_permutation(window_docids: List[str], perm: List[int]) -> List[str]:
    """Reorder a docid list by a 1-indexed permutation."""
    return [window_docids[i - 1] for i in perm]


# ──────────────────────────────────────────────────────────────────────────────
# Sliding window algorithm
# ──────────────────────────────────────────────────────────────────────────────

def get_window_starts(top_k: int, window_size: int, stride: int) -> List[int]:
    """
    Return the 0-indexed start positions for the bottom-up sliding window.

    Example (top_k=100, window=20, stride=10):
        [80, 70, 60, 50, 40, 30, 20, 10, 0]   ← 9 steps

    The last entry is always 0 so the top window is always visited even if
    (top_k - window_size) is not a perfect multiple of stride.
    """
    start  = top_k - window_size
    starts : List[int] = []
    while start > 0:
        starts.append(start)
        start -= stride
    starts.append(0)
    return starts


def check_prompt_length(tokenizer, queries, run, corpus, window_size, max_model_len):
    """
    Spot-check 3 random queries: tokenize their first-window prompt and warn
    if any exceeds max_model_len.  Catches truncation before the full run.
    """
    sample_qids = random.sample(list(run.keys()), min(3, len(run)))
    max_seen = 0
    for qid in sample_qids:
        docs       = run[qid][:window_size]
        window_doc = [(d, corpus.get(d, "")) for d in docs]
        messages   = build_chat_messages(queries[qid], window_doc)
        text       = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        n_tok = len(tokenizer.encode(text))
        max_seen = max(max_seen, n_tok)
        if n_tok > max_model_len - 200:   # 200 tok reserved for output
            log.warning(
                f"  LONG PROMPT: qid={qid} → {n_tok} tokens "
                f"(budget={max_model_len - 200}).  "
                f"Consider reducing --max-doc-words."
            )
    log.info(f"  Prompt length check: max sample = {max_seen} tokens "
             f"(limit {max_model_len - 200} input + 200 output = {max_model_len}).")


def sliding_window_rerank(
    llm,
    tokenizer,
    queries    : Dict[str, str],
    run        : Dict[str, List[str]],
    corpus     : Dict[str, str],
    sampling_params,
    window_size: int = WINDOW_SIZE,
    stride     : int = STRIDE,
    top_k      : int = TOP_K,
    ckpt_file  : Optional[Path] = None,
) -> Tuple[Dict[str, List[str]], dict]:
    """
    Sliding-window listwise reranking for all queries in one dataset.

    Strategy
    --------
    Within a single query the n_steps window steps are SEQUENTIAL — step k+1
    depends on step k's reordered list.  Across queries at the *same* step
    the work is independent, so all N queries are batched into ONE vLLM
    generate() call per step (n_steps calls total regardless of dataset size).

    Returns
    -------
    (ranked, timing)
      ranked : {qid: [docid_rank1, …, docid_rank_top_k]}
      timing : structured dict with per-step and aggregate timing info
    """
    # Work on per-query copies (don't mutate the caller's dict)
    ranked: Dict[str, List[str]] = {
        qid: list(docs[:top_k])
        for qid, docs in run.items()
        if qid in queries
    }

    window_starts  = get_window_starts(top_k, window_size, stride)
    n_steps        = len(window_starts)
    qids_ordered   = list(ranked.keys())
    n_queries      = len(qids_ordered)

    # ── Checkpoint: resume from previous run if available ────────────────────
    resume_from_step = 0          # 0 = start fresh; N = first N steps already done
    step_timings: List[dict] = []

    if ckpt_file and ckpt_file.exists():
        try:
            ckpt           = json.loads(ckpt_file.read_text())
            resume_from_step = int(ckpt.get("completed_steps", 0))
            step_timings   = ckpt.get("step_timings", [])
            ckpt_ranked    = ckpt.get("ranked", {})
            ranked.update(ckpt_ranked)   # override with saved state
            log.info(
                f"  Checkpoint loaded: {resume_from_step}/{n_steps} steps already done. "
                f"Resuming from step {resume_from_step + 1}."
            )
        except Exception as e:
            log.warning(f"  Could not load checkpoint ({e}); starting from scratch.")
            resume_from_step = 0
            step_timings     = []
    else:
        log.info(f"  No checkpoint found; starting from step 1.")

    log.info(f"  {n_queries:,} queries × {n_steps} window steps "
             f"= {n_queries * n_steps:,} total LLM calls (batched into {n_steps}).")
    log.info(f"  Window positions (0-indexed start): {window_starts}")

    # ── Timing bookkeeping ───────────────────────────────────────────────────
    rerank_wall_start = time.time()
    rerank_start_iso  = datetime.datetime.now().isoformat(timespec="seconds")

    for step_idx, start in enumerate(window_starts, 1):
        # Skip steps already completed in a previous run
        if step_idx <= resume_from_step:
            log.info(f"  Skipping step {step_idx}/{n_steps} (checkpoint).")
            continue
        end    = start + window_size
        t_step = time.time()
        log.info(f"  Step {step_idx}/{n_steps}  docs[{start}:{end}]  "
                 f"(ranks {start + 1}–{min(end, top_k)}) — "
                 f"batching {n_queries:,} prompts …")

        # ── Build one prompt per query ──────────────────────────────────────
        raw_prompts: List[str] = []
        effective_window_sizes: List[int] = []

        t_build = time.time()
        for qid in qids_ordered:
            docs        = ranked[qid]
            a_start     = min(start, len(docs))
            a_end       = min(end,   len(docs))
            window_ids  = docs[a_start:a_end]
            window_docs = [(d, corpus.get(d, "")) for d in window_ids]

            messages = build_chat_messages(queries[qid], window_docs)
            text     = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            raw_prompts.append(text)
            effective_window_sizes.append(len(window_docs))
        t_build = time.time() - t_build

        # ── Batch inference ──────────────────────────────────────────────────
        t_infer = time.time()
        outputs = llm.generate(raw_prompts, sampling_params)
        t_infer = time.time() - t_infer

        # ── Apply permutations ───────────────────────────────────────────────
        t_parse = time.time()
        for qid, output, eff_w in zip(qids_ordered, outputs, effective_window_sizes):
            docs    = ranked[qid]
            a_start = min(start, len(docs))
            a_end   = min(end,   len(docs))
            w_ids   = docs[a_start:a_end]

            if not w_ids:
                continue

            generated = output.outputs[0].text.strip()
            perm      = parse_permutation(generated, eff_w)
            reordered = apply_permutation(w_ids, perm)

            ranked[qid] = docs[:a_start] + reordered + docs[a_end:]
        t_parse = time.time() - t_parse

        step_elapsed = time.time() - t_step
        ms_per_query = step_elapsed / n_queries * 1000

        log.info(
            f"  Step {step_idx}/{n_steps} done in {step_elapsed:.2f}s  "
            f"[build={t_build:.2f}s  infer={t_infer:.2f}s  parse={t_parse:.2f}s]  "
            f"→ {ms_per_query:.1f}ms/query"
        )

        step_timings.append({    # must stay before checkpoint save
            "step"              : step_idx,
            "window_start_0idx" : start,
            "window_end_0idx"   : end,
            "ranks"             : f"{start + 1}-{min(end, top_k)}",
            "n_prompts_in_batch": n_queries,
            "prompt_build_s"    : round(t_build,        4),
            "vllm_infer_s"      : round(t_infer,        4),
            "permute_parse_s"   : round(t_parse,        4),
            "step_total_s"      : round(step_elapsed,   4),
            "ms_per_query"      : round(ms_per_query,   2),
        })

        # ── Save checkpoint after every step ────────────────────────────────
        if ckpt_file:
            try:
                ckpt_file.write_text(json.dumps({
                    "completed_steps": step_idx,
                    "ranked"         : ranked,
                    "step_timings"   : step_timings,
                }))
            except Exception as e:
                log.warning(f"  Checkpoint write failed: {e}")

    # ── Aggregate timing ────────────────────────────────────────────────────
    total_rerank_s   = time.time() - rerank_wall_start
    rerank_end_iso   = datetime.datetime.now().isoformat(timespec="seconds")
    total_infer_s    = sum(s["vllm_infer_s"]   for s in step_timings)
    total_build_s    = sum(s["prompt_build_s"] for s in step_timings)
    avg_s_per_query  = total_rerank_s / n_queries

    log.info(
        f"\n  ── Reranking summary ────────────────────────────────────\n"
        f"  Queries          : {n_queries:,}\n"
        f"  Window steps     : {n_steps}\n"
        f"  Total wall time  : {total_rerank_s:.2f}s\n"
        f"  Avg per query    : {avg_s_per_query:.3f}s  "
        f"({avg_s_per_query * 1000:.1f}ms)\n"
        f"  vLLM infer total : {total_infer_s:.2f}s  "
        f"({total_infer_s / total_rerank_s * 100:.1f}% of wall time)\n"
        f"  Prompt build tot : {total_build_s:.2f}s\n"
        f"  ────────────────────────────────────────────────────────"
    )

    timing = {
        "wall_clock_start"    : rerank_start_iso,
        "wall_clock_end"      : rerank_end_iso,
        "total_rerank_s"      : round(total_rerank_s,  4),
        "avg_s_per_query"     : round(avg_s_per_query, 4),
        "avg_ms_per_query"    : round(avg_s_per_query * 1000, 2),
        "total_vllm_infer_s"  : round(total_infer_s,   4),
        "total_prompt_build_s": round(total_build_s,   4),
        "pct_time_in_vllm"    : round(total_infer_s / total_rerank_s * 100, 2),
        "n_queries"           : n_queries,
        "n_steps"             : n_steps,
        "steps"               : step_timings,
    }

    # ── Delete checkpoint on clean completion ────────────────────────────────
    if ckpt_file and ckpt_file.exists():
        ckpt_file.unlink()
        log.info("  Checkpoint deleted (run complete).")

    return ranked, timing


# ──────────────────────────────────────────────────────────────────────────────
# TREC output
# ──────────────────────────────────────────────────────────────────────────────

def write_trec_run(
    ranked     : Dict[str, List[str]],
    output_file: Path,
    tag        : str = "RankZephyr",
) -> None:
    """Write reranked results in standard TREC run format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def sort_key(qid):
        return int(qid) if qid.isdigit() else qid

    with open(output_file, "w") as f:
        for qid in sorted(ranked.keys(), key=sort_key):
            docs = ranked[qid]
            for rank, docid in enumerate(docs, 1):
                score = len(docs) - rank + 1      # descending integer scores
                f.write(f"{qid} Q0 {docid} {rank} {score} {tag}\n")

    log.info(f"  Written → {output_file.name}")


def write_timing(timing: dict, output_file: Path) -> None:
    """Write timing dict as a JSON file alongside the TREC run."""
    with open(output_file, "w") as f:
        json.dump(timing, f, indent=2)
    log.info(f"  Timing  → {output_file.name}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sliding Window Reranking with RankZephyr (vLLM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", nargs="+", default=["all"],
        help=(
            "Dataset(s) to process.  Pass one name, several names, or 'all'. "
            f"Choices: {list(DATASETS.keys()) + ['all']}"
        ),
    )
    p.add_argument(
        "--retriever", default="bm25",
        choices=["bm25", "splade"],
        help="First-stage retriever whose TREC files are used as input.",
    )
    p.add_argument("--top-k",       type=int,   default=TOP_K,       help="Docs to rerank per query.")
    p.add_argument("--window-size", type=int,   default=WINDOW_SIZE, help="Sliding window size.")
    p.add_argument("--stride",      type=int,   default=STRIDE,      help="Sliding window stride.")
    p.add_argument(
        "--gpus", type=int, default=1,
        help="Tensor parallel size for vLLM. Default 1 (one process per GPU via CUDA_VISIBLE_DEVICES).",
    )
    p.add_argument(
        "--gpu-mem-util", type=float, default=0.90,
        help="vLLM GPU memory utilisation fraction.",
    )
    p.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Maximum token sequence length passed to vLLM.",
    )
    p.add_argument(
        "--max-num-seqs", type=int, default=512,
        help="Maximum concurrent sequences in one vLLM forward pass.",
    )
    p.add_argument(
        "--max-doc-words", type=int, default=MAX_DOC_WORDS,
        help=(
            "Max words per document in the prompt. "
            "100 words ≈ 130 tokens; 20 docs → ~2800 tok total, safely under 4096."
        ),
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "--model", default="rankzephyr", choices=list(MODELS.keys()),
        help="Reranker model to use. Default: rankzephyr.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Override module-level word budget if user passed --max-doc-words
    global MAX_DOC_WORDS
    MAX_DOC_WORDS = args.max_doc_words

    raw = args.dataset
    if raw == ["all"] or raw == "all":
        datasets_to_run = list(DATASETS.keys())
    else:
        datasets_to_run = raw if isinstance(raw, list) else [raw]
        unknown = [d for d in datasets_to_run if d not in DATASETS]
        if unknown:
            log.error(f"Unknown dataset(s): {unknown}.  Valid: {list(DATASETS.keys())}")
            sys.exit(1)
    retriever  = args.retriever
    W, S, K    = args.window_size, args.stride, args.top_k
    model_name = MODELS[args.model]
    model_tag  = args.model + "_7b"
    model_disp = "RankZephyr_7B" if args.model == "rankzephyr" else "RankVicuna_7B"

    # ── Load model once, reuse for all datasets ──────────────────────────────
    log.info("=" * 65)
    log.info(f"Model : {model_name}")
    log.info(f"GPUs  : {args.gpus}  |  mem_util={args.gpu_mem_util}  "
             f"|  max_len={args.max_model_len}  |  max_seqs={args.max_num_seqs}")
    log.info(f"Window: size={W}  stride={S}  top_k={K}")
    log.info("=" * 65)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        log.error(
            "vLLM is not installed.  Install with:\n"
            "  pip install vllm\n"
            "or the appropriate pip for your environment."
        )
        sys.exit(1)

    from transformers import AutoTokenizer

    log.info("Loading tokenizer …")
    t_model_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.model == "rankvicuna" and not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = VICUNA_CHAT_TEMPLATE

    log.info("Initialising vLLM engine (this may take ~30-60s) …")
    # vLLM 0.8.x: direct LLM params + EngineArgs forwarded via **kwargs
    llm = LLM(
        model                  = model_name,
        tensor_parallel_size   = args.gpus,
        dtype                  = "bfloat16",
        max_model_len          = args.max_model_len,
        gpu_memory_utilization = args.gpu_mem_util,
        enforce_eager          = True,           # skip cuda-graph compilation (Python.h missing)
        trust_remote_code      = False,
        # EngineArgs forwarded via **kwargs:
        max_num_seqs           = args.max_num_seqs,
        enable_chunked_prefill = True,           # safe OOM-free large batches
    )
    t_model_load = time.time() - t_model_load
    log.info(f"Model + tokenizer loaded in {t_model_load:.1f}s.")

    sampling_params = SamplingParams(
        temperature         = 0.0,   # greedy — deterministic, highest accuracy
        max_tokens          = 200,   # permutation is ≤100 tok; 200 is safe headroom
        skip_special_tokens = True,
    )

    # Run tag embedded in the TREC output (identifies method fully)
    run_tag = f"{model_disp}_{retriever}_w{W}s{S}"

    # ── Process each dataset ─────────────────────────────────────────────────
    for dataset_name in datasets_to_run:
        cfg = DATASETS[dataset_name]

        log.info("")
        log.info("─" * 65)
        log.info(f"Dataset  : {dataset_name}   Retriever : {retriever}")
        log.info("─" * 65)

        # Input run file
        run_file = RETRIEVER_DIR / retriever / f"{dataset_name}.top{K}.txt"
        if not run_file.exists():
            log.warning(f"Input run not found: {run_file} — skipping.")
            continue

        # Output file stem (shared by .txt and .timing.json)
        out_stem = (
            f"{dataset_name}.{retriever}.{model_tag}.w{W}s{S}.top{K}"
        )
        out_file        = OUT_DIR / f"{out_stem}.txt"
        out_timing_file = OUT_DIR / f"{out_stem}.timing.json"

        if out_file.exists() and not args.force:
            log.info(f"Output already exists (use --force to overwrite): {out_file.name}")
            continue

        # ── Load queries ─────────────────────────────────────────────────────
        valid_qids = load_qrel_qids(cfg["qrels"], cfg["qrels_format"])
        queries    = load_queries(cfg["queries"], cfg["query_format"], valid_qids)
        log.info(f"  Queries with qrels: {len(queries):,}")

        # ── Load run ─────────────────────────────────────────────────────────
        run = load_run(run_file)
        run = {qid: docs for qid, docs in run.items() if qid in queries}
        log.info(f"  Run queries (overlap with qrels): {len(run):,}")

        if not run:
            log.warning("  No queries in run — skipping.")
            continue

        # ── Load corpus (only required documents) ────────────────────────────
        t_corpus = time.time()
        needed   = get_needed_docids(run)
        corpus   = load_corpus_selective(cfg["corpus"], cfg["corpus_format"], needed)
        t_corpus = time.time() - t_corpus

        # ── Prompt length safety check (3 random queries) ────────────────────
        check_prompt_length(tokenizer, queries, run, corpus, W, args.max_model_len)

        # ── Checkpoint file lives next to the output ─────────────────────────
        ckpt_file = OUT_DIR / f"{out_stem}.ckpt.json"

        # ── Sliding window reranking ─────────────────────────────────────────
        ranked, timing = sliding_window_rerank(
            llm             = llm,
            tokenizer       = tokenizer,
            queries         = queries,
            run             = run,
            corpus          = corpus,
            sampling_params = sampling_params,
            window_size     = W,
            stride          = S,
            top_k           = K,
            ckpt_file       = ckpt_file,
        )

        # ── Augment timing with run-level metadata ───────────────────────────
        timing["dataset"]              = dataset_name
        timing["retriever"]            = retriever
        timing["model"]                = model_name
        timing["corpus_load_s"]        = round(t_corpus,      4)
        timing["model_load_s"]         = round(t_model_load,  4)
        timing["config"] = {
            "window_size"          : W,
            "stride"               : S,
            "top_k"                : K,
            "tensor_parallel_size" : args.gpus,
            "gpu_memory_util"      : args.gpu_mem_util,
            "max_model_len"        : args.max_model_len,
            "max_num_seqs"         : args.max_num_seqs,
        }

        # ── Write outputs ─────────────────────────────────────────────────────
        write_trec_run(ranked, out_file, tag=run_tag)
        write_timing(timing, out_timing_file)

    log.info("")
    log.info("All done.")


if __name__ == "__main__":
    main()
