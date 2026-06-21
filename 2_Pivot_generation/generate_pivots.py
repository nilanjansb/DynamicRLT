"""
Pivot document generator using Llama-3.1-8B-Instruct.

Three modes
-----------
  default        — 4 prompts per query (one per tau ∈ {0,1,2,3}), batched across
                   queries in chunks of batch_size.  Generates one document per
                   relevance grade separately.
                   Output written to <output-dir>/<dataset>.json

  --tau-together — 1 prompt per query; all 4 tau documents generated in a single
                   forward pass using [TAU=N] section markers.
                   Output written to <output-dir>/tautogether/<dataset>.json

  --single       — 1 prompt per query; generates one plain relevant document
                   with no mention of relevance grades or tau.
                   Output written to <output-dir>/single/<dataset>.json

Usage:
  python3.12 generate_pivots.py \
      --datasets msmarco-dl19 msmarco-dl20 trec-covid scifact \
      --device cuda:1 \
      [--tau-together | --single] [--batch-size 16] [--max-new-tokens N] [--output-dir pivot_docs]

Output: <output-dir>/<dataset>.json  per dataset
Resume: safe to re-run — skips any query already written.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


HF_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN   = os.environ.get("HF_TOKEN")
DATA_ROOT  = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parents[1] / "1_Download_dataset/data"))
TAU_LEVELS = [0, 1, 2, 3]

# ── Prompt templates ──────────────────────────────────────────────────────────

# Mode: individual  (one call per query per tau level)
PROMPT_TEMPLATE = (
    'You are an expert information retrieval specialist tasked with document generation. '
    'Your task is to generate a single document that matches a specific relevance grade '
    'and is relevant to the given query.\n'
    'Given the query: "{Q}"\n'
    'Generate a document that would receive a relevance score of "{t}"\n\n'
    'Relevance grade score definitions:\n'
    '0: Non-relevant; the document or passage provides no useful information for the query.\n'
    '1: Partially relevant; offers some related information but is of limited utility.\n'
    '2: Somewhat relevant; partially addresses the query but leaves significant information needs unmet.\n'
    '3: Highly relevant; provides an ideal response to the query, often comprehensive and precise.\n\n'
    'Output the document text only. '
    'Do not include any title, label, relevance score, explanation, commentary, or self-assessment. '
    'Do not use markdown formatting, bullet points, headers, bold, or any special symbols. '
    'Write plain continuous prose only.'
)

# Mode: tau-together  (one call per query, all 4 tau documents at once)
PROMPT_TEMPLATE_TOGETHER = (
    'You are an expert information retrieval specialist tasked with document generation. '
    'For the given query, generate four documents — one at each relevance grade (0, 1, 2, 3) — in a single response.\n\n'
    'Given the query: "{Q}"\n\n'
    'Relevance grade definitions:\n'
    '0: Non-relevant; the document or passage provides no useful information for the query.\n'
    '1: Partially relevant; offers some related information but is of limited utility.\n'
    '2: Somewhat relevant; partially addresses the query but leaves significant information needs unmet.\n'
    '3: Highly relevant; provides an ideal response to the query, often comprehensive and precise.\n\n'
    'Output all four documents using exactly this format, with each section beginning on a new line:\n'
    '[TAU=0]\n'
    '<document for grade 0>\n'
    '[TAU=1]\n'
    '<document for grade 1>\n'
    '[TAU=2]\n'
    '<document for grade 2>\n'
    '[TAU=3]\n'
    '<document for grade 3>\n\n'
    'Use the [TAU=N] markers exactly as shown. '
    'Do not include any titles, explanations, commentary, or self-assessment beyond the markers. '
    'Do not use markdown formatting, bullet points, headers, bold, or any special symbols. '
    'Write plain continuous prose for each document.'
)

# Mode: single  (one call per query, one plain relevant document, no tau)
PROMPT_TEMPLATE_SINGLE = (
    'You are an expert information retrieval specialist tasked with document generation. '
    'Your task is to generate a single document that is relevant to the given query.\n'
    'Given the query: "{Q}"\n\n'
    'Output the document text only. '
    'Do not include any title, label, explanation, commentary, or self-assessment. '
    'Do not use markdown formatting, bullet points, headers, bold, or any special symbols. '
    'Write plain continuous prose only.'
)


# ---------------------------------------------------------------------------
# Query loaders
# ---------------------------------------------------------------------------

def load_msmarco_queries(split):
    """Load judged MS MARCO test queries for DL19 or DL20."""
    qrel_file  = f"{DATA_ROOT}/msmarco/qrels.dl{split[-2:]}-passage.txt"
    query_file = f"{DATA_ROOT}/msmarco/queries.dl{split[-2:]}.tsv"

    judged_ids = set()
    with open(qrel_file) as f:
        for line in f:
            judged_ids.add(line.strip().split()[0])

    queries = {}
    with open(query_file) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2 and parts[0] in judged_ids:
                queries[parts[0]] = parts[1]

    return [(qid, queries[qid]) for qid in sorted(judged_ids) if qid in queries]


def load_beir_queries(dataset_name):
    """Load BEIR test-set queries (test qrels → query IDs → text)."""
    qrel_file  = f"{DATA_ROOT}/beir/{dataset_name}/qrels/test.tsv"
    query_file = f"{DATA_ROOT}/beir/{dataset_name}/queries.jsonl"

    judged_ids = set()
    with open(qrel_file) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # header
            judged_ids.add(line.strip().split("\t")[0])

    queries = {}
    with open(query_file) as f:
        for line in f:
            doc = json.loads(line)
            _id = doc.get("_id") or doc.get("id")
            if _id in judged_ids:
                queries[_id] = doc.get("text", "")

    return [(qid, queries[qid]) for qid in sorted(judged_ids) if qid in queries]


DATASET_LOADERS = {
    "msmarco-dl19":       lambda: load_msmarco_queries("dl19"),
    "msmarco-dl20":       lambda: load_msmarco_queries("dl20"),
    "trec-covid":         lambda: load_beir_queries("trec-covid"),
    "scifact":            lambda: load_beir_queries("scifact"),
    "webis-touche2020":   lambda: load_beir_queries("webis-touche2020"),
    "dbpedia-entity":     lambda: load_beir_queries("dbpedia-entity"),
    "fever":              lambda: load_beir_queries("fever"),
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_prompt(query, tau):
    return PROMPT_TEMPLATE.replace("{Q}", query).replace("{t}", str(tau))

def build_prompt_together(query):
    return PROMPT_TEMPLATE_TOGETHER.replace("{Q}", query)

def build_prompt_single(query):
    return PROMPT_TEMPLATE_SINGLE.replace("{Q}", query)


def parse_tau_together(text):
    """Extract per-tau documents from a [TAU=N] delimited generation."""
    pivots = {}
    for tau in TAU_LEVELS:
        marker      = f"[TAU={tau}]"
        next_marker = f"[TAU={tau + 1}]" if (tau + 1) in TAU_LEVELS else None
        start = text.find(marker)
        if start == -1:
            pivots[str(tau)] = ""
            continue
        start += len(marker)
        end = text.find(next_marker) if next_marker else len(text)
        if end == -1:
            end = len(text)
        pivots[str(tau)] = text[start:end].strip()
    return pivots


def generate_batch(model, tokenizer, prompts, device, max_new_tokens):
    """
    Batch-generate completions for a list of prompts.
    Uses left-padding (required for causal LM batch decoding).
    Greedy decoding (do_sample=False) makes output fully deterministic.
    """
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    enc = tokenizer(
        formatted,
        padding=True,
        truncation=True,
        max_length=768,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — fully deterministic
            temperature=1.0,          # ignored with do_sample=False; explicit for clarity
            top_p=1.0,                # override model's generation_config default (0.9) to silence warning
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    new_tokens = out[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def process_dataset(dataset_name, model, tokenizer, device,
                    batch_size, max_new_tokens, out_dir, mode="individual"):
    """
    Generate pivot documents for all judged queries in dataset_name.

    mode="individual"  : 4 prompts per query (one per tau), batched as
                         batch_size // 4 queries per GPU call.
                         Saves {"qid", "query", "pivots": {"0":…,"1":…,"2":…,"3":…}}.
    mode="tau_together": 1 prompt per query, all 4 taus in one generation.
                         Saves same schema as individual.
    mode="single"      : 1 prompt per query, no tau structure.
                         Saves {"qid", "query", "pivot": "…"}.

    Resumable: queries already written to out_path are skipped on re-run.
    """
    out_path = Path(out_dir) / f"{dataset_name}.json"

    # Resume: load already-completed queries
    done_qids = set()
    existing  = []
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
        existing  = data.get("queries", [])
        done_qids = {e["qid"] for e in existing}
        print(f"  [{dataset_name}] Resuming: {len(done_qids)} queries already done.")

    queries = DATASET_LOADERS[dataset_name]()
    pending = [(qid, qt) for qid, qt in queries if qid not in done_qids]
    print(f"  [{dataset_name}] {len(pending)} queries to generate ({len(done_qids)} skipped).")

    if not pending:
        print(f"  [{dataset_name}] Already complete.")
        return

    results = list(existing)

    if mode == "individual":
        # 4 prompts per query; divide batch_size to keep GPU tensor size constant
        queries_per_chunk = max(1, batch_size // len(TAU_LEVELS))
    else:
        # tau-together and single: 1 prompt per query
        queries_per_chunk = max(1, batch_size)

    with tqdm(total=len(pending), desc=f"{dataset_name} [{device}] [{mode}]", unit="query") as pbar:
        for i in range(0, len(pending), queries_per_chunk):
            query_chunk = pending[i : i + queries_per_chunk]

            if mode == "tau_together":
                batch_prompts = [build_prompt_together(qt) for _, qt in query_chunk]
                generations   = generate_batch(model, tokenizer, batch_prompts, device, max_new_tokens)
                for (qid, qt), text in zip(query_chunk, generations):
                    results.append({
                        "qid":    qid,
                        "query":  qt,
                        "pivots": parse_tau_together(text),
                    })

            elif mode == "single":
                batch_prompts = [build_prompt_single(qt) for _, qt in query_chunk]
                generations   = generate_batch(model, tokenizer, batch_prompts, device, max_new_tokens)
                for (qid, qt), text in zip(query_chunk, generations):
                    results.append({
                        "qid":   qid,
                        "query": qt,
                        "pivot": text.strip(),
                    })

            else:  # individual
                batch_prompts = []
                batch_meta    = []
                for qid, qt in query_chunk:
                    for tau in TAU_LEVELS:
                        batch_prompts.append(build_prompt(qt, tau))
                        batch_meta.append((qid, qt, tau))

                generations = generate_batch(model, tokenizer, batch_prompts, device, max_new_tokens)

                buf = {}
                for (qid, qt, tau), text in zip(batch_meta, generations):
                    if qid not in buf:
                        buf[qid] = {"qid": qid, "query": qt, "pivots": {}}
                    buf[qid]["pivots"][str(tau)] = text.strip()
                for entry in buf.values():
                    results.append(entry)

            pbar.update(len(query_chunk))
            _save(out_path, dataset_name, queries, results, mode)

    print(f"  [{dataset_name}] Done. {len(results)} queries saved → {out_path}")


def _save(out_path, dataset_name, all_queries, results, mode):
    """Atomically write results to out_path via a temp file + os.replace."""
    data = {
        "dataset":       dataset_name,
        "model":         HF_MODEL,
        "mode":          mode,
        "total_queries": len(all_queries),
    }
    if mode != "single":
        data["tau_levels"] = TAU_LEVELS
    data["queries"] = results

    tmp = str(out_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, out_path)   # atomic write


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets",       nargs="+", required=True,
                   choices=list(DATASET_LOADERS.keys()))
    p.add_argument("--device",         default="cuda:1")
    p.add_argument("--batch-size",     type=int, default=16,
                   help="prompts per GPU call (individual mode: num_queries × 4 tau = batch tensor size)")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="tokens to generate per call; defaults to 300 (individual/single) or 1300 (--tau-together)")
    p.add_argument("--output-dir",     default="pivot_docs")

    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--tau-together", action="store_true",
                            help="generate all 4 tau documents in one GPU call; output → <output-dir>/tautogether/")
    mode_group.add_argument("--single",       action="store_true",
                            help="generate one plain relevant document per query (no tau); output → <output-dir>/single/")
    return p.parse_args()


def main():
    args = parse_args()

    if args.tau_together:
        mode    = "tau_together"
        out_dir = Path(args.output_dir) / "tautogether"
    elif args.single:
        mode    = "single"
        out_dir = Path(args.output_dir) / "single"
    else:
        mode    = "individual"
        out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    max_new_tokens = args.max_new_tokens or (1300 if mode == "tau_together" else 300)

    print(f"Loading {HF_MODEL} on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, token=HF_TOKEN)
    tokenizer.padding_side = "left"   # required for batch generation with causal LM
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.float16,
        device_map=args.device,
        token=HF_TOKEN,
    )
    model.eval()

    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated(args.device)/1024**3:.1f} GB used")
    print(f"Mode: {mode}  |  batch-size: {args.batch_size}  |  max_new_tokens: {max_new_tokens}")
    print()

    for ds in args.datasets:
        print(f"=== {ds} ===")
        process_dataset(
            ds, model, tokenizer, args.device,
            args.batch_size, max_new_tokens, str(out_dir),
            mode=mode,
        )
        print()

    print("All datasets complete.")


if __name__ == "__main__":
    main()
