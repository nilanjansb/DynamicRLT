"""
Download MS MARCO Passage V1 (~8.8M passages) and
TREC DL-19 / DL-20 topics + qrels.

Sources
-------
  Corpus          : official MS MARCO TSV (msmarco.z22.web.core.windows.net)
  DL-19/20 topics : same host (gzipped TSV)
  DL-19/20 qrels  : TREC NIST (trec.nist.gov)
  Fallback        : ir_datasets  ('msmarco-passage/trec-dl-2019/judged' etc.)

Outputs
-------
data/msmarco/
    collection.tsv          – passage corpus    (pid TAB text)
    queries.dl19.tsv        – DL-19 topics      (qid TAB text)
    queries.dl20.tsv        – DL-20 topics      (qid TAB text)
    qrels.dl19-passage.txt  – DL-19 qrels       (TREC format)
    qrels.dl20-passage.txt  – DL-20 qrels       (TREC format)
"""

import gzip
import shutil
import tarfile
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data" / "msmarco"

# ── file manifest ─────────────────────────────────────────────────────────────
FILES = {
    # MS MARCO Passage V1 corpus (TSV, gzipped ~3 GB)
    "collection.tsv": (
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
        True,
        "collection.tsv",   # name inside the archive (tar, not gz-only)
    ),
    # DL-19 topics (43 judged queries)
    "queries.dl19.tsv": (
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
        True,
        None,               # None → plain gzip, no tar
    ),
    # DL-20 topics (54 judged queries)
    "queries.dl20.tsv": (
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
        True,
        None,
    ),
    # DL-19 qrels (TREC NIST)
    "qrels.dl19-passage.txt": (
        "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
        False,
        None,
    ),
    # DL-20 qrels (TREC NIST)
    "qrels.dl20-passage.txt": (
        "https://trec.nist.gov/data/deep/2020qrels-pass.txt",
        False,
        None,
    ),
}


def download_file(dest: Path, url: str, is_compressed: bool, inner_name):
    """Download a single file from *url* and decompress it into *dest*.

    Args:
        dest:          Final destination path for the extracted file.
        url:           Remote URL to fetch.
        is_compressed: True if the payload is gzip-compressed (.gz or .tar.gz).
        inner_name:    Member name to extract from a .tar.gz archive, or None
                       for a plain .gz file.
    """
    print(f"  Downloading {dest.name} …")
    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)

        if not is_compressed:
            tmp.rename(dest)
        elif inner_name is not None:
            # tar.gz archive → extract the named member
            with tarfile.open(tmp) as tar:
                member = tar.getmember(inner_name)
                with tar.extractfile(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            tmp.unlink()
        else:
            # plain .gz file
            with gzip.open(tmp, "rb") as gz_in, open(dest, "wb") as f_out:
                shutil.copyfileobj(gz_in, f_out)
            tmp.unlink()

        print(f"    → {dest}")
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"    WARNING: {e}")
        print(f"    Manual URL: {url}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname, (url, compressed, inner) in FILES.items():
        dest = OUT_DIR / fname
        if dest.exists():
            print(f"  Already exists: {dest}")
        else:
            download_file(dest, url, compressed, inner)

    print()
    try:
        import ir_datasets  # noqa: F401
        print("ir_datasets available. Alternative loader:")
        print("  ds19 = ir_datasets.load('msmarco-passage/trec-dl-2019/judged')")
        print("  ds20 = ir_datasets.load('msmarco-passage/trec-dl-2020/judged')")
    except ImportError:
        print("Tip: pip install ir_datasets  for an alternative DL-19/DL-20 loader.")

    print("\nMS MARCO download complete.")
