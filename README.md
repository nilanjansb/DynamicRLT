# DynamicRLT
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Implementation for **DynamicRLT**, a generative IR framework that replaces extraction-based pivots in listwise LLM reranking with generative pivots — synthetic LLM generated documents produced at controlled relevance grades — enabling parallel comparison passes and adaptive window strides. We evaluate on seven BEIR/TREC benchmarks against BM25 and SPLADE as the first-stage retrievers.

---

## Pipeline Overview

1. **Initial Retrieval**: BM25 / SPLADE → top-100 per query (TREC run file)
2. **Pivot Generation**: Llama-3.1-8B-Instruct → pivot docs per query, per τ
3. **Reranking**: RankZephyr / RankVicuna via vLLM (Sliding Window, TDPart, SNOW, GenTDPart, GenSliding)
4. **Evaluation**: pytrec\_eval → nDCG@10

---

## Datasets

| Dataset Key | Collection | Topics Source | # Queries | Relevance Metric |
|------------|------------|---------------|-----------|-----------------|
| `dl19` | MS MARCO Passage (8.8M) | TREC DL-2019 | 43 | NDCG@10, AP@10 |
| `dl20` | MS MARCO Passage (8.8M) | TREC DL-2020 | 54 | NDCG@10, AP@10 |
| `covid` | BEIR COVID-19 (171K) | TREC-COVID | 50 | NDCG@10, MAP |
| `scifact` | BEIR SciFact (5.2K) | SciFact | 300 | NDCG@10, MAP |
| `touchev2` | BEIR Touché-2020 (383K) | Touché | 49 | NDCG@10, MAP |
| `dbpedia` | BEIR DBPedia (4.6M) | DBPedia-Entity | 400 | NDCG@10, MAP |
| `fever` | BEIR FEVER (5.4M) | FEVER | 6,666 | NDCG@10, MAP |

---

## 1. Initial Retrieval

**BM25** via Pyserini. **SPLADE** via `naver/splade-cocondenser-selfdistil`. Both retrieve top-100 documents per query and produce TREC-format run files. BM25 is single-threaded. SPLADE is GPU-accelerated. Output files are stored in `3_Initial_Retriever/runs/` and named `{dataset}.{retriever}.txt` (e.g., `msmarco-dl19.bm25.txt`).

```bash
python3 3_Initial_Retriever/bm25_retrieve.py
python3 3_Initial_Retriever/splade_retrieve.py
```

---

## 2. Pivot Document Generation

**Model:** `meta-llama/Llama-3.1-8B-Instruct`  
**Script:** `2_Pivot_generation/generate_pivots.py`

A *pivot document* p̂ is a synthetic passage generated to represent a target relevance grade τ for a given query. It serves as a semantic boundary: documents ranked above p̂ are considered relevant candidates; those below are not.

Relevance grades τ ∈ {0, 1, 2, 3}:
- **τ=0** — non-relevant
- **τ=1** — partially relevant
- **τ=2** — borderline relevant (default for reranking)
- **τ=3** — highly relevant

Three generation modes are supported:

| Mode | Calls per query | Description |
|---|---|---|
| `individual` (default) | 4 | Separate prompt per τ level |
| `--tau-together` | 1 | All four τ documents in one pass using `[TAU=N]` section markers |
| `--single` | 1 | One relevant document with no explicit grade conditioning |

```bash
python3 2_Pivot_generation/generate_pivots.py \
    --datasets msmarco-dl19 msmarco-dl20 trec-covid scifact \
              webis-touche2020 dbpedia-entity fever \
    --device cuda:0
```

---

## 3. Reranking

All rerankers use **vLLM** offline batch inference (bfloat16, greedy decoding, `temperature=0.0`). The same listwise ranking prompt is shared between both rerankers.

**Models:**
- `castorini/rank_zephyr_7b_v1_full` — Zephyr chat template
- `castorini/rank_vicuna_7b_v1` — FastChat Vicuna v1.1 template

**Prompt format:** Given query Q and candidate documents d₁…dₙ, the model outputs a permutation `[N] > [N-1] > … > [1]` sorted by estimated relevance.

Default parameters: window W=20, top-K=100, max\_doc\_words=100.

---

### 3.1 Sliding Window (Baseline)

Standard bottom-up listwise reranking with fixed stride S=10.

```
docs[0..99] (initial BM25/SPLADE order)

Round 1:  rank docs[80:100]  →  reordered in-place
Round 2:  rank docs[70:90]   →  reordered in-place
...
Round 9:  rank docs[0:20]    →  reordered in-place
```

Each round is one batched vLLM call across all queries. **9 sequential calls total.** Complexity: ⌈(K − W) / S⌉ + 1 passes.

```bash
CUDA_VISIBLE_DEVICES=0 python3 4_Sliding_Window/Sliding_Window.py \
    --dataset fever --retriever bm25 --gpus 1
```

---

### 3.2 TDPart (Baseline)

Top-down partitioning. Extracts a pivot from the initial ranked list via a listwise call, then partitions the remainder in parallel.

**Phase 1: First window** (1 vLLM call):  
Rank docs[0:W]. Extract pivot p at rank k (k = W/2 = 10). Let A₀ = docs ranked above p in this window, B₀ = docs ranked below.

**Phase 2: Parallel comparison** (1 vLLM call):  
Split remaining docs R = docs[W:K] into batches of W−1. Prepend p to each batch. Rank all batches in one call. Route above-pivot → A, below-pivot → B.

Budget cap b (default = W): if |A| > b, excess docs are routed to B to keep Phase 3 tractable.

**Phase 3: Final sort of A** (≥1 vLLM call):  
If |A| ≤ W: one batched call. If |A| > W: bottom-up sliding window over A.

**Output:** A_sorted + B. Total: **3 phases, typically 3 vLLM calls.**

```bash
CUDA_VISIBLE_DEVICES=0 python3 5_TDPart/tdpart.py \
    --dataset dbpedia-entity --retriever bm25 --gpus 1
```

---

### 3.3 SNOW

**S**hared **N**on-**O**verlapping **W**indow. Replaces TDPart's extracted pivot with a pre-generated pivot p̂_τ, enabling Phase 1 to be skipped entirely and Phase 2 to run over the full top-K in one shot.

```
top-K docs split into G = K/W = 5 non-overlapping groups of W−1 = 19 docs each.
docs[95:100] → appended verbatim to B (5 remaining)
```

**Phase 1: Parallel group comparison** (1 vLLM call, G × Q prompts):  
For each group g: window = docs[g·19 : (g+1)·19] + [p̂]. Rank and partition at p̂.  
Merge: A = ∪ above_g across all groups, B = ∪ below_g + remaining.

**Phase 2: Final sort of A** (≥1 vLLM call):  
Same as TDPart Phase 3.

**Output:** A_sorted + B. Total: **2–5 vLLM calls** (vs 9 for Sliding Window).

SNOW achieves >3× call reduction because all group comparisons are independent and can be batched as a single vLLM request. The quality of the partition depends on the quality of p̂, controlled by τ.

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 6_SNOW/snow.py \
    --dataset fever --retriever bm25 --gpus 2
```

---

### 3.4 GenTDPart

TDPart with a generated pivot. Identical phase structure to TDPart but Phase 1 (the first-window LLM call + pivot extraction) is replaced by direct pivot injection.

**Phase 1: Pivot injection** (0 vLLM calls):  
Load p̂_τ from `2_Pivot_generation/pivot_docs/`. Inject as synthetic doc in corpus. Set R = all top-K docs.

**Phase 2: Parallel comparison** (1 vLLM call):  
Identical to TDPart Phase 2, except p̂ is the generated pivot rather than an extracted one.

**Phase 3: Final sort of A** (≥1 vLLM call):  
Identical to TDPart Phase 3.

**Output:** A_sorted + B (synthetic docid excluded). **Saves 1 vLLM call** relative to TDPart. Removes dependence on the quality of the first-window ranking for pivot selection.

```bash
CUDA_VISIBLE_DEVICES=0 python3 7_GenTDPart/gentdpart.py \
    --dataset msmarco-dl19 --retriever bm25 --gpus 1
```

---

### 3.5 GenSliding

Adaptive-stride bottom-up sliding window. Uses a pre-generated pivot p̂_τ in place of a fixed stride; each window contains W−1 real docs + p̂ = **W docs total** (same per-window budget as SNOW).

**Phase 1: Adaptive sliding rounds** (sequential across rounds, batched across all queries per round):

```
docs[0..99] (initial BM25/SPLADE order, p starts at K=100)

Round 1:  rank docs[81:100] + p̂  →  A₁ (above p̂), B₁ (below p̂)
          stride₁ = min(S_max, max(1, |B₁|))  →  p = 100 − stride₁

Round 2:  rank docs[p−19:p] + p̂  →  A₂, B₂
          stride₂ = min(S_max, max(1, |B₂|))  →  p = p − stride₂

...until p ≤ 19 for each query (queries exit the active set independently)
```

A_global = A₁ + A₂ + … prepended each round (top→bottom order preserved). When few docs land above p̂ (low-quality region, |B_t| large), stride is large and the pointer fast-forwards; when many land above p̂ (high-quality region, |B_t| small), stride shrinks to 1 and windows overlap heavily to carefully rank the good docs.

**Phase 2: Final window** (1 vLLM call):  
Rank docs[0:p] + p̂ for all queries with 0 < p ≤ 19. Update A_global, B_global.

**Output:** A_global + B_global (p̂ excluded). Total: **n_rounds + 1 vLLM calls**, n_rounds ≈ 2–9 depending on stride convergence.

```bash
CUDA_VISIBLE_DEVICES=0 python3 8_GenSliding/gensliding.py \
    --dataset msmarco-dl19 --retriever bm25 --gpus 1
```

---

## 4. Evaluation

**Script:** `9_Evaluate/evaluateTrec.py`  
**Metric:** nDCG@10 via `pytrec_eval`  
**Input:** TREC-format run files from any reranker  

```bash
python3 9_Evaluate/evaluateTrec.py \
    --metrics ndcg_cut_10 \
    --json results.json \
    runs/*.txt
```

---

## Run File Naming Convention

```
{dataset}.{retriever}.{model}.{method}[_{pivot_type}[_tau{τ}]]_w{W}[s{S}].top{K}.txt
```

Examples:
```
msmarco-dl19.bm25.rankzephyr_7b.w20s10.top100.txt             # Sliding Window
msmarco-dl19.bm25.rankzephyr_7b.tdpart_w20k10.top100.txt      # TDPart
msmarco-dl19.bm25.rankzephyr_7b.snow_tau2_w20.top100.txt      # SNOW τ=2 individual
msmarco-dl19.bm25.rankzephyr_7b.snow_tautogether_tau2_w20.top100.txt  # SNOW τ=2 tautogether
msmarco-dl19.bm25.rankzephyr_7b.snow_single_w20.top100.txt    # SNOW single
msmarco-dl19.bm25.rankzephyr_7b.gentdpart_tau2_w20.top100.txt # GenTDPart τ=2
msmarco-dl19.bm25.rankzephyr_7b.gensliding_tau2_w20s10.top100.txt    # GenSliding τ=2
```

---

## Project Structure

```
.
├── 1_Download_dataset/     datasets (BEIR via HF, MSMARCO)
├── 2_Pivot_generation/     generate_pivots.py + pivot_docs/
├── 3_Initial_Retriever/    bm25_retrieve.py, splade_retrieve.py + run files
├── 4_Sliding_Window/       Sliding_Window.py + runs/
├── 5_TDPart/               tdpart.py + runs/
├── 6_SNOW/                 snow.py + runs/
├── 7_GenTDPart/            gentdpart.py + runs/
├── 8_GenSliding/           gensliding.py + runs/
└── 9_Evaluate/             evaluateTrec.py
```

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- vLLM (V0 engine; `VLLM_USE_V1=0`)
- Transformers, pytrec\_eval, pyserini, beir
- GPU: ≥24 GB VRAM recommended for 7B rerankers. Llama-3.1-8B pivot generation on any CUDA device.

**Hardware used (for reproducibility):**

| Component | Spec |
|-----------|------|
| GPU | 3x NVIDIA RTX 6000 Ada Generation, 48 GB VRAM each |
| CPU | Intel Xeon Gold 6542Y, 2 sockets × 24 cores (96 threads total) |
| RAM | 251 GB |
