#!/usr/bin/env bash
# GenSliding: all runs — bm25 + splade, rankzephyr, 6 datasets, all 9 modes
# GPU: 1  (runs sequentially — vLLM reinitialises between calls)

cd "$(dirname "$0")"
mkdir -p logs

DATASETS="msmarco-dl19 msmarco-dl20 scifact trec-covid webis-touche2020 dbpedia-entity"
LOG=logs/gensliding_rankzephyr_all.log

for RETRIEVER in bm25 splade; do
    echo "[$(date '+%H:%M:%S')] === Starting retriever: $RETRIEVER ===" | tee -a "$LOG"

    # separate tau 0-3
    for TAU in 0 1 2 3; do
        echo "[$(date '+%H:%M:%S')] $RETRIEVER | separate | tau$TAU" | tee -a "$LOG"
        CUDA_VISIBLE_DEVICES=1 python3 gensliding.py \
            --dataset $DATASETS \
            --retriever $RETRIEVER \
            --model rankzephyr \
            --pivot-type separate \
            --pivot-tau $TAU \
            --gpus 1 \
            >> "$LOG" 2>&1
    done

    # tautogether tau 0-3
    for TAU in 0 1 2 3; do
        echo "[$(date '+%H:%M:%S')] $RETRIEVER | tautogether | tau$TAU" | tee -a "$LOG"
        CUDA_VISIBLE_DEVICES=1 python3 gensliding.py \
            --dataset $DATASETS \
            --retriever $RETRIEVER \
            --model rankzephyr \
            --pivot-type tautogether \
            --pivot-tau $TAU \
            --gpus 1 \
            >> "$LOG" 2>&1
    done

    # single
    echo "[$(date '+%H:%M:%S')] $RETRIEVER | single" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=1 python3 gensliding.py \
        --dataset $DATASETS \
        --retriever $RETRIEVER \
        --model rankzephyr \
        --pivot-type single \
        --gpus 1 \
        >> "$LOG" 2>&1

    echo "[$(date '+%H:%M:%S')] === Done: $RETRIEVER ===" | tee -a "$LOG"
done

echo "[$(date '+%H:%M:%S')] All GenSliding runs complete." | tee -a "$LOG"
