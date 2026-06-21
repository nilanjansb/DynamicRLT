#!/usr/bin/env python3
"""SPLADE baseline retrieval for all datasets using pyserini LuceneImpactSearcher.
Outputs TREC run files for top-100 and top-1000 documents.
"""

import json
import os
from pathlib import Path

BASE       = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parents[1]))
INDEX_DIR  = BASE / "SPLADE_indexes"
DATA_DIR   = BASE / "1_Download_dataset/data"
OUT_DIR    = BASE / "3_Initial_Retriever/splade"

from pyserini.search.lucene import LuceneImpactSearcher

MODEL_NAME = "naver/splade-cocondenser-ensembledistil"

DATASETS = {
    "msmarco-dl19": {
        "index":        INDEX_DIR / "msmarco",
        "queries":      DATA_DIR / "msmarco/queries.dl19.tsv",
        "qrels":        DATA_DIR / "msmarco/qrels.dl19-passage.txt",
        "query_format": "tsv",
        "qrels_format": "trec",
    },
    "msmarco-dl20": {
        "index":        INDEX_DIR / "msmarco",
        "queries":      DATA_DIR / "msmarco/queries.dl20.tsv",
        "qrels":        DATA_DIR / "msmarco/qrels.dl20-passage.txt",
        "query_format": "tsv",
        "qrels_format": "trec",
    },
    "scifact": {
        "index":        INDEX_DIR / "scifact",
        "queries":      DATA_DIR / "beir/scifact/queries.jsonl",
        "qrels":        DATA_DIR / "beir/scifact/qrels/test.tsv",
        "query_format": "jsonl",
        "qrels_format": "beir_tsv",
    },
    "trec-covid": {
        "index":        INDEX_DIR / "trec-covid",
        "queries":      DATA_DIR / "beir/trec-covid/queries.jsonl",
        "qrels":        DATA_DIR / "beir/trec-covid/qrels/test.tsv",
        "query_format": "jsonl",
        "qrels_format": "beir_tsv",
    },
    "webis-touche2020": {
        "index":        INDEX_DIR / "webis-touche2020",
        "queries":      DATA_DIR / "beir/webis-touche2020/queries.jsonl",
        "qrels":        DATA_DIR / "beir/webis-touche2020/qrels/test.tsv",
        "query_format": "jsonl",
        "qrels_format": "beir_tsv",
    },
    "fever": {
        "index":        INDEX_DIR / "fever",
        "queries":      DATA_DIR / "beir/fever/queries.jsonl",
        "qrels":        DATA_DIR / "beir/fever/qrels/test.tsv",
        "query_format": "jsonl",
        "qrels_format": "beir_tsv",
    },
    "dbpedia-entity": {
        "index":        INDEX_DIR / "dbpedia-entity",
        "queries":      DATA_DIR / "beir/dbpedia-entity/queries.jsonl",
        "qrels":        DATA_DIR / "beir/dbpedia-entity/qrels/test.tsv",
        "query_format": "jsonl",
        "qrels_format": "beir_tsv",
    },
}


def load_queries(query_file, fmt, valid_qids=None):
    queries = {}
    if fmt == "tsv":
        with open(query_file) as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    qid, text = parts
                    queries[qid] = text
    elif fmt == "jsonl":
        with open(query_file) as f:
            for line in f:
                obj = json.loads(line.strip())
                qid  = str(obj["_id"])
                text = obj["text"]
                queries[qid] = text
    if valid_qids is not None:
        queries = {q: t for q, t in queries.items() if q in valid_qids}
    return queries


def load_qrel_qids(qrels_file, fmt):
    qids = set()
    with open(qrels_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if fmt == "beir_tsv" and i == 0:
                continue  # skip header
            parts = line.split("\t") if fmt == "beir_tsv" else line.split()
            qids.add(parts[0])
    return qids


def retrieve(dataset_name, cfg, searcher, top_k):
    valid_qids = load_qrel_qids(cfg["qrels"], cfg["qrels_format"])
    queries    = load_queries(cfg["queries"], cfg["query_format"], valid_qids)
    print(f"  {len(queries)} queries with qrels")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{dataset_name}.top{top_k}.txt"
    written  = 0
    with open(out_file, "w") as f:
        for qid in sorted(queries.keys(), key=lambda x: int(x) if x.isdigit() else x):
            text = queries[qid]
            hits = searcher.search(text, k=top_k)
            for rank, hit in enumerate(hits, 1):
                f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} SPLADE\n")
                written += 1

    print(f"  Wrote {written} lines -> {out_file.name}")
    return out_file


def main():
    for name, cfg in DATASETS.items():
        print(f"\n=== {name} ===")
        index_path = str(cfg["index"])
        print(f"  Loading index: {index_path}")
        searcher = LuceneImpactSearcher(index_path, MODEL_NAME)
        retrieve(name, cfg, searcher, top_k=100)
        retrieve(name, cfg, searcher, top_k=1000)


if __name__ == "__main__":
    main()
