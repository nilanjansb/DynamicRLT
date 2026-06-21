#!/usr/bin/env python3
"""
GenTDPart reranking with RankZephyr or RankVicuna.
  Models: castorini/rank_zephyr_7b_v1_full  (default)
          castorini/rank_vicuna_7b_v1        (--model rankvicuna)

This variant keeps the TDPart phase structure and comparison logic, but
replaces Phase 1 (initial listwise ranking + pivot extraction) with a
pre-generated LLM pivot document.  The synthetic pivot is injected into
the corpus under a reserved docid and excluded from the final output.

Algorithm
---------
  Phase 1 — Load pivot (no LLM call):
    Inject the generated pivot text into the corpus.
    Set R = all top_k initial docs (nothing is ranked yet).

  Phase 2 — Parallel comparison batches (1 vLLM call):
    Split R into batches of W-1 docs.  Prepend pivot to each batch.
    Rank ALL batches for ALL queries in one call.
    Route above-pivot → A, below-pivot → B.  Budget cap applies.

  Phase 3 — Final sort of A (≥1 vLLM call):
    If |A| ≤ W: one batched call across all queries.
    If |A| > W: bottom-up sliding window over A, batched per step.

  Final list: A_sorted + B  (synthetic pivot excluded from output)

Pivot source: 2_Pivot_generation/pivot_docs/<dataset>.json

Output naming (varies by --pivot-type):
  separate    : {dataset}.{retriever}.{model_tag}.gentdpart_tau{tau}_w{W}.top{N}.txt
  tautogether : {dataset}.{retriever}.{model_tag}.gentdpart_tautogether_tau{tau}_w{W}.top{N}.txt
  single      : {dataset}.{retriever}.{model_tag}.gentdpart_single_w{W}.top{N}.txt

   e.g.: dbpedia-entity.bm25.rankzephyr_7b.gentdpart_tau2_w20.top100.txt
         dbpedia-entity.bm25.rankzephyr_7b.gentdpart_tautogether_tau2_w20.top100.txt
         dbpedia-entity.bm25.rankzephyr_7b.gentdpart_single_w20.top100.txt
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

BASE = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parents[1]))
TDPART_PATH = BASE / "5_TDPart/tdpart.py"
PIVOT_DIR = BASE / "2_Pivot_generation/pivot_docs"
RETRIEVER_DIR = BASE / "3_Initial_Retriever"
OUT_DIR = BASE / "7_GenTDPart/runs"


def load_tdpart_module():
    spec = importlib.util.spec_from_file_location("tdpart_base", TDPART_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load TDPart module from {TDPART_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


td = load_tdpart_module()

MODELS     = td.MODELS
WINDOW_SIZE = td.WINDOW_SIZE
PIVOT_K = td.PIVOT_K
BUDGET = td.BUDGET
TOP_K = td.TOP_K
STRIDE = td.STRIDE
MAX_DOC_WORDS = td.MAX_DOC_WORDS
PIVOT_TAU = 2
DATASETS = td.DATASETS


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
    return f"__genpivot__{qid}__tau{tau}"


def gentdpart_rerank(
    llm,
    tokenizer,
    queries         : Dict[str, str],
    run             : Dict[str, List[str]],
    corpus          : Dict[str, str],
    generated_pivots: Dict[str, str],
    sampling_params,
    pivot_tau    : int           = PIVOT_TAU,
    window_size  : int           = WINDOW_SIZE,
    budget       : Optional[int] = BUDGET,
    top_k        : int           = TOP_K,
    stride       : int           = STRIDE,
    max_model_len: int           = 4096,
    ckpt_file    : Optional[Path] = None,
) -> Tuple[Dict[str, List[str]], dict]:
    """
    GenTDPart reranking — TDPart with a pre-generated pivot replacing Phase 1.

    Phase 1 (no LLM): pivot is injected into the corpus; R = all top_k docs.
    Phase 2 (1 vLLM call): batches of W-1 docs each prepended with pivot are
      ranked in parallel.  Above-pivot → A, below-pivot → B (budget capped).
    Phase 3 (≥1 vLLM call): A is sorted — single call if |A| ≤ W, else a
      bottom-up sliding window pass, both batched across queries per step.
    Final: A_sorted + B  (synthetic pivot excluded from output).
    """
    initial_docs: Dict[str, List[str]] = {
        qid: list(docs[:top_k])
        for qid, docs in run.items()
        if qid in queries
    }
    qids_ordered = list(initial_docs.keys())
    n_queries = len(qids_ordered)

    missing_pivots = [qid for qid in qids_ordered if qid not in generated_pivots]
    if missing_pivots:
        raise ValueError(
            f"Missing generated pivots for {len(missing_pivots)} query ids. "
            f"First few: {missing_pivots[:5]}"
        )

    A: Dict[str, List[str]] = {qid: [] for qid in qids_ordered}
    B: Dict[str, List[str]] = {qid: [] for qid in qids_ordered}
    pivots: Dict[str, str] = {}
    R: Dict[str, List[str]] = {qid: [] for qid in qids_ordered}

    for qid in qids_ordered:
        pivot_id = gen_pivot_docid(qid, pivot_tau)
        corpus[pivot_id] = generated_pivots[qid]
        pivots[qid] = pivot_id

    resume_phase = 1
    phase_timings: List[dict] = []

    if ckpt_file and ckpt_file.exists():
        try:
            ckpt = json.loads(ckpt_file.read_text())
            resume_phase = int(ckpt.get("next_phase", 1))
            phase_timings = ckpt.get("phase_timings", [])
            A.update(ckpt.get("A", {}))
            B.update(ckpt.get("B", {}))
            pivots.update(ckpt.get("pivots", {}))
            R.update(ckpt.get("R", {}))
            log.info(f"  Checkpoint loaded: resuming from Phase {resume_phase}.")
        except Exception as e:
            log.warning(f"  Could not load checkpoint ({e}); starting from Phase 1.")
            resume_phase = 1
            phase_timings = []

    def save_ckpt(next_phase: int) -> None:
        if not ckpt_file:
            return
        try:
            ckpt_file.write_text(json.dumps({
                "next_phase": next_phase,
                "A": A,
                "B": B,
                "pivots": pivots,
                "R": R,
                "phase_timings": phase_timings,
            }))
        except Exception as e:
            log.warning(f"  Checkpoint write failed: {e}")

    wall_start = time.time()
    wall_start_iso = datetime.datetime.now().isoformat(timespec="seconds")

    # Prompt length safety check before any LLM inference (probe Phase 2 window)
    if resume_phase <= 1:
        sample_qids = random.sample(list(initial_docs.keys()), min(3, len(initial_docs)))
        max_seen = 0
        for qid in sample_qids:
            probe_window = [pivots[qid]] + initial_docs[qid][:max(0, window_size - 1)]
            text = td.make_prompt(
                tokenizer, queries[qid], probe_window, corpus, max_model_len=max_model_len
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

    if resume_phase <= 1:
        t0 = time.time()
        log.info(
            f"  Phase 1 — load generated pivots: {n_queries:,} queries "
            f"(tau={pivot_tau}) …"
        )
        for qid in qids_ordered:
            A[qid] = []
            B[qid] = []
            R[qid] = list(initial_docs[qid])
        elapsed = time.time() - t0
        log.info(f"  Phase 1 done in {elapsed:.2f}s [load-only]")
        phase_timings.append({
            "phase": 1,
            "label": "load_generated_pivot",
            "pivot_tau": pivot_tau,
            "n_queries": n_queries,
            "prompt_build_s": 0.0,
            "vllm_infer_s": 0.0,
            "permute_parse_s": 0.0,
            "phase_total_s": round(elapsed, 4),
        })
        save_ckpt(next_phase=2)

    if resume_phase <= 2:
        t0 = time.time()
        batch_prompts: List[str] = []
        batch_meta: List[Tuple[str, List[str]]] = []

        for qid in qids_ordered:
            pivot = pivots[qid]
            r_docs = list(R[qid])
            batch_size = window_size - 1
            for i in range(0, len(r_docs), batch_size):
                chunk = r_docs[i:i + batch_size]
                window = [pivot] + chunk
                batch_prompts.append(
                    td.make_prompt(
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
        ph2_outputs = llm.generate(batch_prompts, sampling_params) if batch_prompts else []
        t_infer = time.time() - t_infer_start

        t_parse_start = time.time()
        for (qid, window), output in zip(batch_meta, ph2_outputs):
            pivot = pivots[qid]
            eff_w = len(window)
            generated = output.outputs[0].text.strip()
            perm = td.parse_permutation(generated, eff_w)
            reordered = td.apply_permutation(window, perm)
            above, below = td.partition_on_pivot(reordered, pivot)

            if budget is not None:
                space = max(0, budget - len(A[qid]))
                A[qid].extend(above[:space])
                B[qid].extend(above[space:])
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
            "phase": 2,
            "label": "parallel_comparison",
            "n_batches_total": n_batches,
            "avg_batches_query": round(batches_per_query, 2),
            "prompt_build_s": round(t_build, 4),
            "vllm_infer_s": round(t_infer, 4),
            "permute_parse_s": round(t_parse, 4),
            "phase_total_s": round(elapsed, 4),
        })
        save_ckpt(next_phase=3)

    if resume_phase <= 3:
        t0 = time.time()
        ai_sizes = {qid: len(A[qid]) for qid in qids_ordered}
        sort_needed = [qid for qid in qids_ordered if ai_sizes[qid] > 1]
        trivial = [qid for qid in qids_ordered if ai_sizes[qid] <= 1]
        max_ai = max(ai_sizes.values()) if ai_sizes else 0

        log.info(
            f"  Phase 3 — final sort: {len(sort_needed):,} queries need sorting "
            f"({len(trivial):,} trivial), max |A_i|={max_ai}"
        )

        ph3_calls = 0
        t_build_p3 = 0.0
        t_infer_p3 = 0.0
        t_parse_p3 = 0.0

        if max_ai <= window_size:
            _tb = time.time()
            prompts_p3: List[str] = []
            sort_qids_p3: List[str] = []
            for qid in sort_needed:
                prompts_p3.append(
                    td.make_prompt(
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
                    perm = td.parse_permutation(generated, len(A[qid]))
                    A[qid] = td.apply_permutation(A[qid], perm)
                t_parse_p3 += time.time() - _tp
        else:
            step_starts = td.sliding_window_steps(max_ai, window_size, stride)
            log.info(
                f"  Phase 3 sliding window over A_i: {len(step_starts)} steps "
                f"(max |A_i|={max_ai}, W={window_size}, stride={stride})"
            )

            for step_idx, start in enumerate(step_starts, 1):
                end = start + window_size
                _tb = time.time()
                prompts_s: List[str] = []
                step_qids: List[str] = []

                for qid in sort_needed:
                    ai = A[qid]
                    a_start = min(start, len(ai))
                    a_end = min(end, len(ai))
                    chunk = ai[a_start:a_end]
                    if not chunk:
                        continue
                    prompts_s.append(
                        td.make_prompt(
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
                    ai = A[qid]
                    a_start = min(start, len(ai))
                    a_end = min(end, len(ai))
                    chunk = ai[a_start:a_end]
                    if not chunk:
                        continue
                    generated = output.outputs[0].text.strip()
                    perm = td.parse_permutation(generated, len(chunk))
                    reordered = td.apply_permutation(chunk, perm)
                    A[qid] = ai[:a_start] + reordered + ai[a_end:]
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
            "phase": 3,
            "label": "final_sort",
            "n_vllm_calls": ph3_calls,
            "max_ai_size": max_ai,
            "prompt_build_s": round(t_build_p3, 4),
            "vllm_infer_s": round(t_infer_p3, 4),
            "permute_parse_s": round(t_parse_p3, 4),
            "phase_total_s": round(elapsed, 4),
        })
        save_ckpt(next_phase=4)

    ranked: Dict[str, List[str]] = {}
    for qid in qids_ordered:
        ranked[qid] = (A[qid] + B[qid])[:top_k]

    total_s = time.time() - wall_start
    wall_end_iso = datetime.datetime.now().isoformat(timespec="seconds")
    total_infer_s = sum(p.get("vllm_infer_s", 0) for p in phase_timings)
    avg_s_per_query = total_s / n_queries if n_queries else 0
    total_build_s = sum(p.get("prompt_build_s", 0) for p in phase_timings)

    log.info(
        f"\n  ── GenTDPart summary ───────────────────────────────────\n"
        f"  Queries          : {n_queries:,}\n"
        f"  Pivot tau        : {pivot_tau}\n"
        f"  Total wall time  : {total_s:.2f}s\n"
        f"  Avg per query    : {avg_s_per_query * 1000:.1f}ms\n"
        f"  vLLM infer total : {total_infer_s:.2f}s  "
        f"({total_infer_s / total_s * 100:.1f}% of wall time)\n"
        f"  ─────────────────────────────────────────────────────────"
    )

    timing = {
        "wall_clock_start": wall_start_iso,
        "wall_clock_end": wall_end_iso,
        "total_rerank_s": round(total_s, 4),
        "avg_s_per_query": round(avg_s_per_query, 4),
        "avg_ms_per_query": round(avg_s_per_query * 1000, 2),
        "total_vllm_infer_s": round(total_infer_s, 4),
        "total_prompt_build_s": round(total_build_s, 4),
        "pct_time_in_vllm": round(total_infer_s / total_s * 100, 2) if total_s else 0,
        "n_queries": n_queries,
        "n_phases": len(phase_timings),
        "pivot_tau": pivot_tau,
        "pivot_source": "Pivot_generation",
        "phases": phase_timings,
    }

    if ckpt_file and ckpt_file.exists():
        ckpt_file.unlink()
        log.info("  Checkpoint deleted (run complete).")

    return ranked, timing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GenTDPart Reranking with RankZephyr or RankVicuna (vLLM)",
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
    p.add_argument("--top-k", type=int, default=TOP_K, help="Docs to rerank per query.")
    p.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Window size W.")
    p.add_argument("--pivot-k", type=int, default=PIVOT_K, help="Ignored in GenTDPart (no Phase 1 ranking); accepted for launch-script compatibility.")
    p.add_argument("--budget", type=int, default=BUDGET, help="Max |A_i| candidate cap. Set 0 for no budget.")
    p.add_argument("--stride", type=int, default=STRIDE, help="Stride for Phase 3 sliding window over A_i.")
    p.add_argument("--gpus", type=int, default=1, help="Tensor parallel size for vLLM.")
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-doc-words", type=int, default=MAX_DOC_WORDS)
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


def main() -> None:
    args = parse_args()

    td.MAX_DOC_WORDS = args.max_doc_words

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
    log.info(f"Model      : {model_name}")
    log.info(f"GPUs       : {args.gpus}  |  mem_util={args.gpu_mem_util}  "
             f"|  max_len={args.max_model_len}  |  max_seqs={args.max_num_seqs}")
    tau_info = f"pivot_tau={args.pivot_tau}" if args.pivot_type != "single" else "pivot_type=single"
    log.info(f"GenTDPart  : W={W}  {tau_info}  budget={B}  top_k={K}")
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
        model=model_name,
        tensor_parallel_size=args.gpus,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True,
        trust_remote_code=False,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=True,
    )
    t_model_load = time.time() - t_model_load
    log.info(f"Model + tokenizer loaded in {t_model_load:.1f}s.")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=200,
        skip_special_tokens=True,
    )

    # Pivot routing — derive directory, loader, filename tag, and run tag per type
    if args.pivot_type == "separate":
        pivot_dir = PIVOT_DIR
        run_tag   = f"{model_disp}_{retriever}_gentdpart_tau{args.pivot_tau}"
    elif args.pivot_type == "tautogether":
        pivot_dir = PIVOT_DIR / "tautogether"
        run_tag   = f"{model_disp}_{retriever}_gentdpart_tautogether_tau{args.pivot_tau}"
    else:  # single
        pivot_dir = PIVOT_DIR / "single"
        run_tag   = f"{model_disp}_{retriever}_gentdpart_single"

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

        pivot_file = pivot_dir / f"{dataset_name}.json"
        if not pivot_file.exists():
            log.warning(f"Pivot file not found: {pivot_file} — skipping.")
            continue

        if args.pivot_type == "single":
            out_stem = f"{dataset_name}.{retriever}.{model_tag}.gentdpart_single_w{W}.top{K}"
        elif args.pivot_type == "tautogether":
            out_stem = f"{dataset_name}.{retriever}.{model_tag}.gentdpart_tautogether_tau{args.pivot_tau}_w{W}.top{K}"
        else:
            out_stem = f"{dataset_name}.{retriever}.{model_tag}.gentdpart_tau{args.pivot_tau}_w{W}.top{K}"
        out_file = OUT_DIR / f"{out_stem}.txt"
        out_timing_file = OUT_DIR / f"{out_stem}.timing.json"

        if out_file.exists() and not args.force:
            log.info(f"Output already exists (use --force to overwrite): {out_file.name}")
            continue

        valid_qids = td.load_qrel_qids(cfg["qrels"], cfg["qrels_format"])
        queries = td.load_queries(cfg["queries"], cfg["query_format"], valid_qids)
        log.info(f"  Queries with qrels: {len(queries):,}")

        run = td.load_run(run_file)
        run = {qid: docs for qid, docs in run.items() if qid in queries}
        log.info(f"  Run queries (overlap with qrels): {len(run):,}")

        if not run:
            log.warning("  No queries in run — skipping.")
            continue

        if args.pivot_type == "single":
            generated_pivots = load_single_pivot(pivot_file)
            log.info(f"  Generated pivots loaded: {len(generated_pivots):,} (single)")
        else:
            generated_pivots = load_generated_pivots(pivot_file, args.pivot_tau)
            log.info(f"  Generated pivots loaded: {len(generated_pivots):,} (tau={args.pivot_tau})")

        # Filter to queries that have both a run entry and a generated pivot
        run = {qid: docs for qid, docs in run.items() if qid in generated_pivots}
        if not run:
            log.warning("  No queries have both run entries and generated pivots — skipping.")
            continue
        log.info(f"  Queries with run + pivot: {len(run):,}")

        t_corpus = time.time()
        needed = td.get_needed_docids(run)
        corpus = td.load_corpus_selective(cfg["corpus"], cfg["corpus_format"], needed)
        t_corpus = time.time() - t_corpus

        ckpt_file = OUT_DIR / f"{out_stem}.ckpt.json"

        ranked, timing = gentdpart_rerank(
            llm             = llm,
            tokenizer       = tokenizer,
            queries         = queries,
            run             = run,
            corpus          = corpus,
            generated_pivots = generated_pivots,
            sampling_params = sampling_params,
            pivot_tau       = args.pivot_tau,
            window_size     = W,
            budget          = B,
            top_k           = K,
            stride          = args.stride,
            max_model_len   = args.max_model_len,
            ckpt_file       = ckpt_file,
        )

        timing["dataset"] = dataset_name
        timing["retriever"] = retriever
        timing["model"] = model_name
        timing["corpus_load_s"] = round(t_corpus, 4)
        timing["model_load_s"] = round(t_model_load, 4)
        timing["config"] = {
            "window_size": W,
            "pivot_k": args.pivot_k,
            "pivot_type": args.pivot_type,
            "pivot_tau": args.pivot_tau if args.pivot_type != "single" else None,
            "budget": B,
            "top_k": K,
            "stride": args.stride,
            "tensor_parallel_size": args.gpus,
            "gpu_memory_util": args.gpu_mem_util,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
        }

        td.write_trec_run(ranked, out_file, tag=run_tag)
        td.write_timing(timing, out_timing_file)

    log.info("")
    log.info("All done.")


if __name__ == "__main__":
    main()
