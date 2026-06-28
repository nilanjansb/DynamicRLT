"""
Download BEIR datasets from HuggingFace (corpus, queries, qrels).

Output layout: data/beir/<name>/{corpus.jsonl, queries.jsonl, qrels/test.tsv}
"""

import json
from pathlib import Path
from datasets import load_dataset

DATASETS = [
    ("trec-covid",       "BeIR/trec-covid",       "BeIR/trec-covid-qrels"),
    ("scifact",          "BeIR/scifact",           "BeIR/scifact-qrels"),
    ("webis-touche2020", "BeIR/webis-touche2020",  "BeIR/webis-touche2020-qrels"),
    ("dbpedia-entity",   "BeIR/dbpedia-entity",    "BeIR/dbpedia-entity-qrels"),
    ("fever",            "BeIR/fever",             "BeIR/fever-qrels"),
]

OUT_DIR = Path(__file__).parent / "data" / "beir"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(path, rows):
    """Write a list of dicts to a JSONL file, one JSON object per line."""
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def download(name, hf_id, qrels_hf_id):
    """Download corpus, queries, and qrels for one BEIR dataset from HuggingFace.

    Skips any file that already exists so interrupted runs can be resumed.
    Outputs are written to OUT_DIR/<name>/{corpus.jsonl, queries.jsonl, qrels/test.tsv}.
    """
    dataset_dir = OUT_DIR / name
    done = (
        (dataset_dir / "corpus.jsonl").exists()
        and (dataset_dir / "queries.jsonl").exists()
        and (dataset_dir / "qrels" / "test.tsv").exists()
    )
    if done:
        print(f"  [{name}] already downloaded — skipping.")
        return

    print(f"  [{name}] Downloading ...")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "qrels").mkdir(exist_ok=True)

    # corpus (large — HuggingFace)
    if not (dataset_dir / "corpus.jsonl").exists():
        corpus_ds = load_dataset(hf_id, "corpus", split="corpus")
        rows = [{"_id": r["_id"], "title": r.get("title", ""), "text": r["text"]}
                for r in corpus_ds]
        save_jsonl(dataset_dir / "corpus.jsonl", rows)
        print(f"    corpus  : {len(rows):>9,} docs")
    else:
        print(f"    corpus  : already exists")

    # queries (small — HuggingFace)
    if not (dataset_dir / "queries.jsonl").exists():
        queries_ds = load_dataset(hf_id, "queries", split="queries")
        rows = [{"_id": r["_id"], "text": r["text"]} for r in queries_ds]
        save_jsonl(dataset_dir / "queries.jsonl", rows)
        print(f"    queries : {len(rows):>9,} queries")
    else:
        print(f"    queries : already exists")

    # qrels (tiny — HuggingFace BeIR/<name>-qrels)
    if not (dataset_dir / "qrels" / "test.tsv").exists():
        qrels_ds = load_dataset(qrels_hf_id, split="test")
        qrels_path = dataset_dir / "qrels" / "test.tsv"
        count = 0
        with open(qrels_path, "w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for r in qrels_ds:
                f.write(f"{r['query-id']}\t{r['corpus-id']}\t{r['score']}\n")
                count += 1
        print(f"    qrels   : {count:>9,} judgements")
    else:
        print(f"    qrels   : already exists")

    print(f"    → {dataset_dir}")


if __name__ == "__main__":
    print("Downloading BEIR datasets ...\n")
    for name, hf_id, qrels_id in DATASETS:
        download(name, hf_id, qrels_id)
        print()
    print("BEIR download complete.")
    print(f"Data saved under: {OUT_DIR.resolve()}")
