#!/usr/bin/env python3
"""
GenSliding — Generative Variable-Stride Sliding Window Reranking with
RankZephyr or RankVicuna.
  Models: castorini/rank_zephyr_7b_v1_full  (default)
          castorini/rank_vicuna_7b_v1        (--model rankvicuna)

GenSliding replaces the fixed pivot extracted from the first window (TDPart)
with a pre-generated LLM pivot document, and replaces the fixed stride of
Sliding Window with an adaptive stride driven by the pivot comparison outcome.

Algorithm
---------
  The pre-generated pivot is injected into the corpus as a synthetic doc.

  Main loop — bottom-up adaptive rounds (sequential across rounds,
  batched across ALL queries per round):
    Pointer p starts at top_k (bottom of the initial ranked list).
    Each round t:
      1. Take window  docs[p-W : p]  (W real docs).
      2. Append pivot as the (W+1)-th document.
      3. Rerank window+pivot in one vLLM call (all active queries batched).
      4. Find pivot's position in the reranked output.
         Docs ranked above pivot → A_t;  below → B_t.
      5. Adaptive stride = min(S_max, max(1, |A_t|)).
         Intuition: we have classified |A_t| docs as above-pivot, so we
         can safely advance the pointer by that many positions without
         missing any unclassified doc.
      6. p ← p − stride.
      7. A_global = A_t + A_global;  B_global = B_t + B_global.
         (Prepend: docs from higher windows come first in the final list.)
    Repeat until p ≤ W for that query (exits the active set).

  Final window (1 vLLM call, all queries batched):
    Each query with 0 < p ≤ W has up to W remaining docs at the top of the
    list.  Rank docs[0 : p] + pivot and update A_global, B_global.

  Output: A_global + B_global  (synthetic pivot excluded — it is not a
  real retrieved document).

vs Fixed Sliding Window (stride always S_max, 9 sequential steps for top-100):
  GenSliding uses fewer LLM calls when |A_t| is large (good docs are
  clustered near the top of the window) and overlaps more when |A_t| is
  small, dynamically adapting to per-query difficulty.

vs SNOW (one parallel pass over 5 non-overlapping groups):
  SNOW makes 2 vLLM calls total (1 group pass + 1 final sort).
  GenSliding makes 1 + n_rounds calls where n_rounds ≈ 2–9 depending on
  how quickly the adaptive stride converges.  GenSliding reranks overlapping
  windows so each doc is compared against the pivot multiple times, trading
  a few extra LLM calls for finer partitioning.

vs GenTDPart (one parallel comparison pass over all docs, 3 phases):
  GenTDPart commits every doc to A or B in a single Phase 2 pass.
  GenSliding re-examines the boundary region at each round via overlapping
  windows, potentially improving partition quality near the pivot boundary.

File naming (varies by --pivot-type):
  separate    : {dataset}.{retriever}.{model_tag}.gensliding_tau{tau}_w{W}s{S}.top{K}.txt
  tautogether : {dataset}.{retriever}.{model_tag}.gensliding_tautogether_tau{tau}_w{W}s{S}.top{K}.txt
  single      : {dataset}.{retriever}.{model_tag}.gensliding_single_w{W}s{S}.top{K}.txt

   e.g.: msmarco-dl19.bm25.rankzephyr_7b.gensliding_tau2_w20s10.top100.txt
         msmarco-dl19.bm25.rankzephyr_7b.gensliding_tautogether_tau2_w20s10.top100.txt
         msmarco-dl19.bm25.rankzephyr_7b.gensliding_single_w20s10.top100.txt

Resumable: checkpoint saved after every round and after the final window;
safe to kill and restart.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 gensliding.py \\
        --dataset msmarco-dl19 --retriever bm25 --gpus 1

    CUDA_VISIBLE_DEVICES=1 python3 gensliding.py \\
        --dataset scifact trec-covid --retriever splade --gpus 1 \\
        --pivot-type tautogether --pivot-tau 3
"""

import os
import sys
import json
import time
import random
import datetime
import argparse
import logging
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("VLLM_USE_V1", "0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

BASE          = Path("/DATA/cs26int00020/Cultural_ablation")
TDPART_PATH   = BASE / "5_TDPart/tdpart.py"
PIVOT_DIR     = BASE / "2_Pivot_generation/pivot_docs"
RETRIEVER_DIR = BASE / "3_Initial_Retriever"
OUT_DIR       = BASE / "8_GenSliding/runs"


# ─────────────────────────────────────────────────────────────────────────────
# Load TDPart module (shared helpers — avoids code duplication)
# ─────────────────────────────────────────────────────────────────────────────

def _load_tdpart():
    spec = importlib.util.spec_from_file_location("tdpart_base", TDPART_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load TDPart module from {TDPART_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


td = _load_tdpart()

MODELS        = td.MODELS
WINDOW_SIZE   = td.WINDOW_SIZE    # 20   — W in the algorithm
MAX_STRIDE    = td.STRIDE         # 10   — S_max (upper bound on adaptive stride)
TOP_K         = td.TOP_K          # 100
MAX_DOC_WORDS = td.MAX_DOC_WORDS  # 100
DATASETS      = td.DATASETS
PIVOT_TAU     = 2                 # default tau level for separate/tautogether modes


# ─────────────────────────────────────────────────────────────────────────────
# Pivot helpers (identical to SNOW and GenTDPart)
# ─────────────────────────────────────────────────────────────────────────────

def load_generated_pivots(pivot_file: Path, tau: int = PIVOT_TAU) -> Dict[str, str]:
    """Return {qid: pivot_text} for the requested tau level from a multi-tau pivot file."""
    data    = json.loads(pivot_file.read_text())
    tau_key = str(tau)
    pivots: Dict[str, str] = {}
    for entry in data.get("queries", []):
        qid  = str(entry["qid"])
        text = entry.get("pivots", {}).get(tau_key, "").strip()
        if text:
            pivots[qid] = text
    return pivots


def load_single_pivot(pivot_file: Path) -> Dict[str, str]:
    """Return {qid: pivot_text} for single-doc pivots (no tau levels)."""
    data   = json.loads(pivot_file.read_text())
    pivots: Dict[str, str] = {}
    for entry in data.get("queries", []):
        qid  = str(entry["qid"])
        text = entry.get("pivot", "").strip()
        if text:
            pivots[qid] = text
    return pivots


def gen_pivot_docid(qid: str, tau: int) -> str:
    """Return the reserved synthetic docid for a query's generated pivot."""
    return f"__genpivot__{qid}__tau{tau}"


# ─────────────────────────────────────────────────────────────────────────────
# GenSliding algorithm
# ─────────────────────────────────────────────────────────────────────────────

def gensliding_rerank(
    llm,
    tokenizer,
    queries          : Dict[str, str],
    run              : Dict[str, List[str]],
    corpus           : Dict[str, str],
    generated_pivots : Dict[str, str],
    sampling_params,
    pivot_tau    : int           = PIVOT_TAU,
    window_size  : int           = WINDOW_SIZE,
    max_stride   : int           = MAX_STRIDE,
    top_k        : int           = TOP_K,
    max_model_len: int           = 4096,
    ckpt_file    : Optional[Path] = None,
) -> Tuple[Dict[str, List[str]], dict]:
    """
    GenSliding reranking — variable-stride bottom-up sliding window driven by
    a pre-generated LLM pivot document.

    Main loop:
      All active queries (p > W) are batched into ONE vLLM call per round.
      After each round, each query's pointer advances by its own adaptive
      stride = min(S_max, max(1, |A_t|)), so different queries may exit the
      main loop at different rounds.

    Final window:
      Queries with 0 < p ≤ W are processed together in one vLLM call.

    No Phase 3 sort: A_global is already ordered by window origin
    (docs from higher windows prepended first), giving a natural top-down
    ordering without an extra reranking pass.

    Parameters
    ----------
    max_stride  : Upper bound S_max on the adaptive stride.  The actual stride
                  at each step equals min(max_stride, max(1, |A_t|)).
    """
    # ── Initialise per-query state ────────────────────────────────────────────
    doc_lists: Dict[str, List[str]] = {
        qid: list(docs[:top_k])
        for qid, docs in run.items()
        if qid in queries
    }
    qids_ordered = list(doc_lists.keys())
    n_queries    = len(qids_ordered)

    missing = [qid for qid in qids_ordered if qid not in generated_pivots]
    if missing:
        raise ValueError(
            f"Missing generated pivots for {len(missing)} queries. "
            f"First few: {missing[:5]}"
        )

    # Inject pivot text into corpus under reserved synthetic docids
    pivot_ids: Dict[str, str] = {}
    for qid in qids_ordered:
        pid           = gen_pivot_docid(qid, pivot_tau)
        corpus[pid]   = generated_pivots[qid]
        pivot_ids[qid] = pid

    # Per-query sliding state
    # p[qid]  : current bottom pointer (starts at top_k, decrements by stride)
    # A[qid]  : docs ranked above pivot, accumulated across rounds (prepended)
    # B[qid]  : docs ranked below pivot, accumulated across rounds (prepended)
    p : Dict[str, int]       = {qid: len(doc_lists[qid]) for qid in qids_ordered}
    A : Dict[str, List[str]] = {qid: [] for qid in qids_ordered}
    B : Dict[str, List[str]] = {qid: [] for qid in qids_ordered}

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    resume_round   = 0      # round index to resume from (0 = fresh start)
    main_loop_done = False  # True once all queries have exited the main loop
    round_timings : List[dict] = []

    if ckpt_file and ckpt_file.exists():
        try:
            ckpt           = json.loads(ckpt_file.read_text())
            resume_round   = int(ckpt.get("next_round",    0))
            main_loop_done = bool(ckpt.get("main_loop_done", False))
            round_timings  = ckpt.get("round_timings",   [])
            p.update({qid: int(v) for qid, v in ckpt.get("p", {}).items()})
            A.update(ckpt.get("A", {}))
            B.update(ckpt.get("B", {}))
            log.info(
                f"  Checkpoint loaded: next_round={resume_round}, "
                f"main_loop_done={main_loop_done}."
            )
        except Exception as e:
            log.warning(f"  Could not load checkpoint ({e}); starting from scratch.")
            resume_round   = 0
            main_loop_done = False
            round_timings  = []

    def save_ckpt(next_round: int, main_done: bool) -> None:
        if not ckpt_file:
            return
        try:
            ckpt_file.write_text(json.dumps({
                "next_round"    : next_round,
                "main_loop_done": main_done,
                "p"             : p,
                "A"             : A,
                "B"             : B,
                "round_timings" : round_timings,
            }))
        except Exception as e:
            log.warning(f"  Checkpoint write failed: {e}")

    wall_start     = time.time()
    wall_start_iso = datetime.datetime.now().isoformat(timespec="seconds")

    # ── Prompt length safety check (probe first bottom window) ────────────────
    if resume_round == 0 and not main_loop_done:
        sample_qids = random.sample(qids_ordered, min(3, n_queries))
        max_seen = 0
        for qid in sample_qids:
            # First window: docs[-W:] + pivot = W+1 docs
            probe_ids = doc_lists[qid][-window_size:] + [pivot_ids[qid]]
            text      = td.make_prompt(
                tokenizer, queries[qid], probe_ids, corpus, max_model_len=max_model_len
            )
            n_tok    = len(tokenizer.encode(text))
            max_seen = max(max_seen, n_tok)
            if n_tok > max_model_len - 200:
                log.warning(
                    f"  LONG PROMPT: qid={qid} → {n_tok} tokens "
                    f"(budget={max_model_len - 200}). "
                    f"Consider reducing --max-doc-words."
                )
        log.info(
            f"  Prompt length check (W+1={window_size + 1} docs): "
            f"max sample = {max_seen} tokens "
            f"(limit {max_model_len - 200} input + 200 output = {max_model_len})."
        )

    # ── Main loop: variable-stride bottom-up rounds ───────────────────────────
    if not main_loop_done:
        active    = [qid for qid in qids_ordered if p[qid] > window_size]
        round_idx = resume_round   # incremented at the top of each iteration

        log.info(
            f"  Main loop: {n_queries:,} queries, W={window_size}, S_max={max_stride}. "
            f"Starting from round {round_idx + 1} "
            f"with {len(active):,} active queries."
        )

        while active:
            round_idx += 1
            t_round    = time.time()

            # ── Build one prompt per active query ─────────────────────────────
            _tb           = time.time()
            batch_prompts : List[str]                         = []
            batch_meta    : List[Tuple[str, int, int]]        = []  # (qid, start, end)

            for qid in active:
                start      = max(0, p[qid] - window_size)
                end        = p[qid]
                window_ids = doc_lists[qid][start:end] + [pivot_ids[qid]]
                batch_prompts.append(
                    td.make_prompt(
                        tokenizer, queries[qid], window_ids, corpus,
                        max_model_len=max_model_len,
                    )
                )
                batch_meta.append((qid, start, end))
            t_build = time.time() - _tb

            log.info(
                f"  Round {round_idx}: {len(active):,} active queries  "
                f"docs[p-{window_size}:p]+pivot — batching {len(batch_prompts)} prompts …"
            )

            # ── Batch inference ───────────────────────────────────────────────
            _ti     = time.time()
            outputs = llm.generate(batch_prompts, sampling_params)
            t_infer = time.time() - _ti

            # ── Parse, partition, advance pointers ───────────────────────────
            _tp          = time.time()
            stride_vals  : List[int] = []

            for (qid, start, end), output in zip(batch_meta, outputs):
                window_ids   = doc_lists[qid][start:end] + [pivot_ids[qid]]
                pivot        = pivot_ids[qid]
                eff_w        = len(window_ids)
                generated    = output.outputs[0].text.strip()
                perm         = td.parse_permutation(generated, eff_w)
                reordered    = td.apply_permutation(window_ids, perm)
                above, below = td.partition_on_pivot(reordered, pivot)

                # Prepend to globals: docs from higher windows lead the final list
                A[qid] = above + A[qid]
                B[qid] = below + B[qid]

                # Adaptive stride: advance by how many docs we classified above pivot
                stride      = min(max_stride, max(1, len(above)))
                p[qid]     -= stride
                stride_vals.append(stride)
            t_parse = time.time() - _tp

            # Update active set: queries still needing main-loop rounds
            active      = [qid for qid in qids_ordered if p[qid] > window_size]
            r_elapsed   = time.time() - t_round
            avg_stride  = sum(stride_vals) / len(stride_vals) if stride_vals else 0

            log.info(
                f"  Round {round_idx} done in {r_elapsed:.2f}s  "
                f"[build={t_build:.2f}s  infer={t_infer:.2f}s  "
                f"parse={t_parse:.2f}s]  "
                f"avg_stride={avg_stride:.1f}  active_remaining={len(active)}"
            )
            round_timings.append({
                "round"          : round_idx,
                "n_prompts"      : len(batch_prompts),
                "n_active_after" : len(active),
                "avg_stride"     : round(avg_stride,  2),
                "prompt_build_s" : round(t_build,     4),
                "vllm_infer_s"   : round(t_infer,     4),
                "permute_parse_s": round(t_parse,     4),
                "round_total_s"  : round(r_elapsed,   4),
            })
            save_ckpt(next_round=round_idx, main_done=False)

        main_loop_done = True
        log.info(
            f"  Main loop complete after {round_idx} rounds "
            f"(resumed from {resume_round})."
        )
        save_ckpt(next_round=round_idx, main_done=True)

    # ── Final window pass ─────────────────────────────────────────────────────
    # Queries with 0 < p ≤ window_size still have unclassified docs at the top.
    # Rank docs[0:p] + pivot for all such queries in one batched vLLM call.
    final_qids = [qid for qid in qids_ordered if 0 < p[qid]]

    if final_qids:
        t0 = time.time()
        log.info(
            f"  Final window: {len(final_qids):,} queries with p ≤ {window_size} "
            f"unclassified docs — 1 vLLM call …"
        )

        _tb           = time.time()
        batch_prompts : List[str]              = []
        batch_meta_f  : List[Tuple[str, int]]  = []   # (qid, p_val)

        for qid in final_qids:
            remaining_ids = doc_lists[qid][:p[qid]] + [pivot_ids[qid]]
            batch_prompts.append(
                td.make_prompt(
                    tokenizer, queries[qid], remaining_ids, corpus,
                    max_model_len=max_model_len,
                )
            )
            batch_meta_f.append((qid, p[qid]))
        t_build_f = time.time() - _tb

        _ti       = time.time()
        outputs   = llm.generate(batch_prompts, sampling_params)
        t_infer_f = time.time() - _ti

        _tp = time.time()
        for (qid, p_val), output in zip(batch_meta_f, outputs):
            remaining_ids = doc_lists[qid][:p_val] + [pivot_ids[qid]]
            pivot         = pivot_ids[qid]
            eff_w         = len(remaining_ids)
            generated     = output.outputs[0].text.strip()
            perm          = td.parse_permutation(generated, eff_w)
            reordered     = td.apply_permutation(remaining_ids, perm)
            above, below  = td.partition_on_pivot(reordered, pivot)
            # Prepend: top-of-list docs from the final window lead A_global
            A[qid]  = above + A[qid]
            B[qid]  = below + B[qid]
            p[qid]  = 0   # mark fully processed
        t_parse_f = time.time() - _tp

        elapsed_f = time.time() - t0
        log.info(
            f"  Final window done in {elapsed_f:.2f}s  "
            f"[build={t_build_f:.2f}s  infer={t_infer_f:.2f}s  "
            f"parse={t_parse_f:.2f}s]"
        )
        round_timings.append({
            "round"          : "final_window",
            "n_prompts"      : len(batch_prompts),
            "n_active_after" : 0,
            "avg_stride"     : None,
            "prompt_build_s" : round(t_build_f,  4),
            "vllm_infer_s"   : round(t_infer_f,  4),
            "permute_parse_s": round(t_parse_f,  4),
            "round_total_s"  : round(elapsed_f,  4),
        })

    # ── Assemble final ranking ────────────────────────────────────────────────
    # A_global (above-pivot, ordered top→bottom by window origin) + B_global.
    # Synthetic pivot excluded.  Truncate to top_k as a safety guard.
    ranked: Dict[str, List[str]] = {}
    for qid in qids_ordered:
        ranked[qid] = (A[qid] + B[qid])[:top_k]

    # ── Aggregate timing ──────────────────────────────────────────────────────
    total_s          = time.time() - wall_start
    wall_end_iso     = datetime.datetime.now().isoformat(timespec="seconds")
    total_infer_s    = sum(r.get("vllm_infer_s",   0) for r in round_timings)
    total_build_s    = sum(r.get("prompt_build_s",  0) for r in round_timings)
    avg_s_per_query  = total_s / n_queries if n_queries else 0
    n_main_rounds    = sum(1 for r in round_timings if r["round"] != "final_window")
    total_vllm_calls = len(round_timings)

    log.info(
        f"\n  ── GenSliding summary ──────────────────────────────────\n"
        f"  Queries          : {n_queries:,}\n"
        f"  Main rounds      : {n_main_rounds}  (+1 final window)\n"
        f"  Total vLLM calls : {total_vllm_calls}\n"
        f"  Pivot tau        : {pivot_tau}\n"
        f"  Total wall time  : {total_s:.2f}s\n"
        f"  Avg per query    : {avg_s_per_query * 1000:.1f}ms\n"
        f"  vLLM infer total : {total_infer_s:.2f}s  "
        f"({total_infer_s / total_s * 100:.1f}% of wall time)\n"
        f"  ─────────────────────────────────────────────────────────"
    )

    timing = {
        "wall_clock_start"    : wall_start_iso,
        "wall_clock_end"      : wall_end_iso,
        "total_rerank_s"      : round(total_s,                4),
        "avg_s_per_query"     : round(avg_s_per_query,        4),
        "avg_ms_per_query"    : round(avg_s_per_query * 1000, 2),
        "total_vllm_infer_s"  : round(total_infer_s,          4),
        "total_prompt_build_s": round(total_build_s,          4),
        "pct_time_in_vllm"    : round(total_infer_s / total_s * 100, 2) if total_s else 0,
        "n_queries"           : n_queries,
        "n_main_rounds"       : n_main_rounds,
        "total_vllm_calls"    : total_vllm_calls,
        "pivot_tau"           : pivot_tau,
        "pivot_source"        : "Pivot_generation",
        "rounds"              : round_timings,
    }

    if ckpt_file and ckpt_file.exists():
        ckpt_file.unlink()
        log.info("  Checkpoint deleted (run complete).")

    return ranked, timing


# ─────────────────────────────────────────────────────────────────────────────
# TREC output & timing (delegates to tdpart helpers)
# ─────────────────────────────────────────────────────────────────────────────

def write_trec_run(
    ranked      : Dict[str, List[str]],
    output_file : Path,
    tag         : str = "GenSliding",
) -> None:
    td.write_trec_run(ranked, output_file, tag=tag)


def write_timing(timing: dict, output_file: Path) -> None:
    td.write_timing(timing, output_file)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GenSliding Reranking with RankZephyr or RankVicuna (vLLM)",
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
    p.add_argument("--top-k",        type=int,   default=TOP_K,       help="Docs to rerank per query.")
    p.add_argument("--window-size",  type=int,   default=WINDOW_SIZE, help="Window size W (real docs per window; pivot appended as W+1-th).")
    p.add_argument("--max-stride",   type=int,   default=MAX_STRIDE,  help="Upper bound S_max on the adaptive stride.")
    p.add_argument("--gpus",         type=int,   default=1,           help="Tensor parallel size for vLLM.")
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-model-len",type=int,   default=4096)
    p.add_argument("--max-num-seqs", type=int,   default=512)
    p.add_argument("--max-doc-words",type=int,   default=MAX_DOC_WORDS)
    p.add_argument(
        "--pivot-tau", type=int, default=PIVOT_TAU,
        help="Tau level for generated pivot document (0–3, default 2).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    p.add_argument(
        "--pivot-type", default="separate",
        choices=["separate", "tautogether", "single"],
        help=(
            "'separate': one file per tau level, pivot_docs/{dataset}.json. "
            "'tautogether': joint tau generation, pivot_docs/tautogether/{dataset}.json. "
            "'single': one document per query, no tau, pivot_docs/single/{dataset}.json."
        ),
    )
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

    td.MAX_DOC_WORDS = args.max_doc_words

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
    W, K, S    = args.window_size, args.top_k, args.max_stride
    model_name = MODELS[args.model]
    model_tag  = args.model + "_7b"
    model_disp = "RankZephyr_7B" if args.model == "rankzephyr" else "RankVicuna_7B"

    log.info("=" * 65)
    log.info(f"Model      : {model_name}")
    log.info(f"GPUs       : {args.gpus}  |  mem_util={args.gpu_mem_util}  "
             f"|  max_len={args.max_model_len}  |  max_seqs={args.max_num_seqs}")
    tau_info = f"pivot_tau={args.pivot_tau}" if args.pivot_type != "single" else "pivot_type=single"
    log.info(f"GenSliding : W={W}  S_max={S}  {tau_info}  top_k={K}")
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
        tokenizer.chat_template = td.VICUNA_CHAT_TEMPLATE

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

    # Pivot routing — derive directory and run/file tags per pivot type
    if args.pivot_type == "separate":
        pivot_dir = PIVOT_DIR
        run_tag   = f"{model_disp}_{retriever}_gensliding_tau{args.pivot_tau}"
    elif args.pivot_type == "tautogether":
        pivot_dir = PIVOT_DIR / "tautogether"
        run_tag   = f"{model_disp}_{retriever}_gensliding_tautogether_tau{args.pivot_tau}"
    else:  # single
        pivot_dir = PIVOT_DIR / "single"
        run_tag   = f"{model_disp}_{retriever}_gensliding_single"

    for dataset_name in datasets_to_run:
        cfg = DATASETS[dataset_name]

        log.info("")
        log.info("─" * 65)
        log.info(f"Dataset  : {dataset_name}   Retriever : {retriever}")
        log.info("─" * 65)

        # Locate initial retrieval run
        run_file = RETRIEVER_DIR / retriever / f"{dataset_name}.top{K}.txt"
        if not run_file.exists():
            log.warning(f"Input run not found: {run_file} — skipping.")
            continue

        # Locate pivot file
        pivot_file = pivot_dir / f"{dataset_name}.json"
        if not pivot_file.exists():
            log.warning(f"Pivot file not found: {pivot_file} — skipping.")
            continue

        # Derive output file stem
        if args.pivot_type == "single":
            out_stem = f"{dataset_name}.{retriever}.{model_tag}.gensliding_single_w{W}s{S}.top{K}"
        elif args.pivot_type == "tautogether":
            out_stem = f"{dataset_name}.{retriever}.{model_tag}.gensliding_tautogether_tau{args.pivot_tau}_w{W}s{S}.top{K}"
        else:
            out_stem = f"{dataset_name}.{retriever}.{model_tag}.gensliding_tau{args.pivot_tau}_w{W}s{S}.top{K}"
        out_file        = OUT_DIR / f"{out_stem}.txt"
        out_timing_file = OUT_DIR / f"{out_stem}.timing.json"

        if out_file.exists() and not args.force:
            log.info(f"Output already exists (use --force to overwrite): {out_file.name}")
            continue

        # Load queries (filtered to those with qrel judgements)
        valid_qids = td.load_qrel_qids(cfg["qrels"], cfg["qrels_format"])
        queries    = td.load_queries(cfg["queries"], cfg["query_format"], valid_qids)
        log.info(f"  Queries with qrels: {len(queries):,}")

        # Load retrieval run (filtered to judged queries)
        run = td.load_run(run_file)
        run = {qid: docs for qid, docs in run.items() if qid in queries}
        log.info(f"  Run queries (overlap with qrels): {len(run):,}")

        if not run:
            log.warning("  No queries in run — skipping.")
            continue

        # Load generated pivots and filter run to queries with both a run entry and a pivot
        if args.pivot_type == "single":
            generated_pivots = load_single_pivot(pivot_file)
            log.info(f"  Generated pivots loaded: {len(generated_pivots):,} (single)")
        else:
            generated_pivots = load_generated_pivots(pivot_file, args.pivot_tau)
            log.info(f"  Generated pivots loaded: {len(generated_pivots):,} (tau={args.pivot_tau})")

        run = {qid: docs for qid, docs in run.items() if qid in generated_pivots}
        if not run:
            log.warning("  No queries have both run entries and generated pivots — skipping.")
            continue
        log.info(f"  Queries with run + pivot: {len(run):,}")

        # Load corpus (only documents actually needed)
        t_corpus = time.time()
        needed   = td.get_needed_docids(run)
        corpus   = td.load_corpus_selective(cfg["corpus"], cfg["corpus_format"], needed)
        t_corpus = time.time() - t_corpus

        ckpt_file = OUT_DIR / f"{out_stem}.ckpt.json"

        ranked, timing = gensliding_rerank(
            llm              = llm,
            tokenizer        = tokenizer,
            queries          = queries,
            run              = run,
            corpus           = corpus,
            generated_pivots = generated_pivots,
            sampling_params  = sampling_params,
            pivot_tau        = args.pivot_tau,
            window_size      = W,
            max_stride       = S,
            top_k            = K,
            max_model_len    = args.max_model_len,
            ckpt_file        = ckpt_file,
        )

        timing["dataset"]       = dataset_name
        timing["retriever"]     = retriever
        timing["model"]         = model_name
        timing["corpus_load_s"] = round(t_corpus,     4)
        timing["model_load_s"]  = round(t_model_load, 4)
        timing["config"] = {
            "window_size"          : W,
            "max_stride"           : S,
            "pivot_type"           : args.pivot_type,
            "pivot_tau"            : args.pivot_tau if args.pivot_type != "single" else None,
            "top_k"                : K,
            "tensor_parallel_size" : args.gpus,
            "gpu_memory_util"      : args.gpu_mem_util,
            "max_model_len"        : args.max_model_len,
            "max_num_seqs"         : args.max_num_seqs,
        }

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_trec_run(ranked, out_file, tag=run_tag)
        write_timing(timing, out_timing_file)

    log.info("")
    log.info("All done.")


if __name__ == "__main__":
    main()
