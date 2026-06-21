#!/usr/bin/env bash
# ============================================================
# RankVicuna full experiment run
#   GPU 0  →  BM25   (all methods, sequential)
#   GPU 1  →  SPLADE (all methods, sequential)
# Datasets: all except fever
# Methods:  Sliding Window, TDPart,
#           SNOW  (separate tau 0-3 | tautogether tau 0-3 | single)
#           GenTDPart (separate tau 0-3 | tautogether tau 0-3 | single)
# ============================================================

set -uo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
SW="$PROJECT/4_Sliding_Window/Sliding_Window.py"
TD="$PROJECT/5_TDPart/tdpart.py"
SN="$PROJECT/6_SNOW/snow.py"
GT="$PROJECT/7_GenTDPart/gentdpart.py"
LOGS="$PROJECT/rankvicuna_logs"
DATASETS="msmarco-dl19 msmarco-dl20 scifact trec-covid webis-touche2020 dbpedia-entity"
TAUS="0 1 2 3"
MEM=0.90

mkdir -p "$LOGS"

# ── Per-retriever block (run entirely on one GPU) ────────────────────────────
run_retriever() {
    local R=$1   # bm25 or splade
    local L="$LOGS/$R"
    mkdir -p "$L"

    echo "[$(date '+%H:%M:%S')] ===== START $R ====="

    # ── 1. Sliding Window ────────────────────────────────────────────────────
    echo "[$(date '+%H:%M:%S')] $R: Sliding Window"
    python3 "$SW" --model rankvicuna --retriever "$R" \
        --dataset $DATASETS --gpu-mem-util $MEM --max-doc-words 75 \
        > "$L/01_slidingwindow.log" 2>&1
    echo "[$(date '+%H:%M:%S')] $R: Sliding Window DONE"

    # ── 2. TDPart ────────────────────────────────────────────────────────────
    echo "[$(date '+%H:%M:%S')] $R: TDPart"
    python3 "$TD" --model rankvicuna --retriever "$R" \
        --dataset $DATASETS --gpu-mem-util $MEM \
        > "$L/02_tdpart.log" 2>&1
    echo "[$(date '+%H:%M:%S')] $R: TDPart DONE"

    # ── 3. SNOW separate tau 0–3 ─────────────────────────────────────────────
    for TAU in $TAUS; do
        echo "[$(date '+%H:%M:%S')] $R: SNOW separate tau=$TAU"
        python3 "$SN" --model rankvicuna --retriever "$R" \
            --dataset $DATASETS --pivot-type separate --pivot-tau "$TAU" \
            --gpu-mem-util $MEM \
            > "$L/03_snow_separate_tau${TAU}.log" 2>&1
        echo "[$(date '+%H:%M:%S')] $R: SNOW separate tau=$TAU DONE"
    done

    # ── 4. SNOW tautogether tau 0–3 ──────────────────────────────────────────
    for TAU in $TAUS; do
        echo "[$(date '+%H:%M:%S')] $R: SNOW tautogether tau=$TAU"
        python3 "$SN" --model rankvicuna --retriever "$R" \
            --dataset $DATASETS --pivot-type tautogether --pivot-tau "$TAU" \
            --gpu-mem-util $MEM \
            > "$L/04_snow_tautogether_tau${TAU}.log" 2>&1
        echo "[$(date '+%H:%M:%S')] $R: SNOW tautogether tau=$TAU DONE"
    done

    # ── 5. SNOW single ───────────────────────────────────────────────────────
    echo "[$(date '+%H:%M:%S')] $R: SNOW single"
    python3 "$SN" --model rankvicuna --retriever "$R" \
        --dataset $DATASETS --pivot-type single \
        --gpu-mem-util $MEM \
        > "$L/05_snow_single.log" 2>&1
    echo "[$(date '+%H:%M:%S')] $R: SNOW single DONE"

    # ── 6. GenTDPart separate tau 0–3 ────────────────────────────────────────
    for TAU in $TAUS; do
        echo "[$(date '+%H:%M:%S')] $R: GenTDPart separate tau=$TAU"
        python3 "$GT" --model rankvicuna --retriever "$R" \
            --dataset $DATASETS --pivot-type separate --pivot-tau "$TAU" \
            --gpu-mem-util $MEM \
            > "$L/06_gentdpart_separate_tau${TAU}.log" 2>&1
        echo "[$(date '+%H:%M:%S')] $R: GenTDPart separate tau=$TAU DONE"
    done

    # ── 7. GenTDPart tautogether tau 0–3 ─────────────────────────────────────
    for TAU in $TAUS; do
        echo "[$(date '+%H:%M:%S')] $R: GenTDPart tautogether tau=$TAU"
        python3 "$GT" --model rankvicuna --retriever "$R" \
            --dataset $DATASETS --pivot-type tautogether --pivot-tau "$TAU" \
            --gpu-mem-util $MEM \
            > "$L/07_gentdpart_tautogether_tau${TAU}.log" 2>&1
        echo "[$(date '+%H:%M:%S')] $R: GenTDPart tautogether tau=$TAU DONE"
    done

    # ── 8. GenTDPart single ──────────────────────────────────────────────────
    echo "[$(date '+%H:%M:%S')] $R: GenTDPart single"
    python3 "$GT" --model rankvicuna --retriever "$R" \
        --dataset $DATASETS --pivot-type single \
        --gpu-mem-util $MEM \
        > "$L/08_gentdpart_single.log" 2>&1
    echo "[$(date '+%H:%M:%S')] $R: GenTDPart single DONE"

    echo "[$(date '+%H:%M:%S')] ===== ALL DONE $R ====="
}

export -f run_retriever
export SW TD SN GT LOGS DATASETS TAUS MEM

# ── Launch both retrievers in parallel ──────────────────────────────────────
(
    export CUDA_VISIBLE_DEVICES=0
    run_retriever bm25
) > "$LOGS/bm25_master.log" 2>&1 &
PID_BM25=$!

(
    export CUDA_VISIBLE_DEVICES=1
    run_retriever splade
) > "$LOGS/splade_master.log" 2>&1 &
PID_SPLADE=$!

echo "============================================================"
echo "  BM25   running on GPU 0  (PID $PID_BM25)"
echo "  SPLADE running on GPU 1  (PID $PID_SPLADE)"
echo ""
echo "  Monitor:"
echo "    tail -f $LOGS/bm25_master.log"
echo "    tail -f $LOGS/splade_master.log"
echo "============================================================"

wait $PID_BM25
BM25_EXIT=$?
wait $PID_SPLADE
SPLADE_EXIT=$?

echo ""
echo "BM25   exit: $BM25_EXIT"
echo "SPLADE exit: $SPLADE_EXIT"
if [ $BM25_EXIT -eq 0 ] && [ $SPLADE_EXIT -eq 0 ]; then
    echo "All runs completed successfully."
else
    echo "One or more runs failed — check logs in $LOGS"
    exit 1
fi
