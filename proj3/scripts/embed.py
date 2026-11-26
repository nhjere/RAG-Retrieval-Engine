"""Turn a BEIR dataset into the .fbin vector files the Go search engine reads.

Data preparation only: no part of the parallel system lives here.

Usage:
    python3 scripts/embed.py                        # quora, default sizes
    python3 scripts/embed.py --dataset scifact --sizes dev=0

Output, one folder per size under --out:
    passages.fbin   [int32 N][int32 Dim][N*Dim float32]
    queries.fbin    same layout
    passages.jsonl  {"id": i, "doc_id": ..., "text": ...}, line i = vector i
    queries.jsonl   {"id": i, "query_id": ..., "text": ...}

Requires: pip install sentence-transformers numpy
"""

import argparse
import gzip
import json
import os
import shutil
import urllib.request
import zipfile

import numpy as np

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# BeIR removed the raw files from main; these commits still have them
HF_PINNED = {"quora": "6205996560df11e3a3da9ab4f926788fc30a7db4"}


def parse_sizes(spec):
    """'small=50000,large=0' -> [('small', 50000), ('large', 0)]; 0 means all."""
    sizes = []
    for part in spec.split(","):
        name, count = part.split("=")
        sizes.append((name.strip(), int(count)))
    return sizes


def hf_fetch(repo, name, revision=None):
    """Try the repo's main branch, then the pinned commit if one is known."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=repo, filename=name, repo_type="dataset")
    except Exception:
        if revision is None:
            raise
        return hf_hub_download(repo_id=repo, filename=name, repo_type="dataset",
                               revision=revision)


def download_hf(dataset, folder):
    """Fetch the same BEIR files from Hugging Face and lay them out like the zip."""
    pinned = HF_PINNED.get(dataset)
    files = [
        (f"BeIR/{dataset}", "corpus.jsonl.gz", "corpus.jsonl", pinned),
        (f"BeIR/{dataset}", "queries.jsonl.gz", "queries.jsonl", pinned),
        (f"BeIR/{dataset}-qrels", "test.tsv", os.path.join("qrels", "test.tsv"), None),
    ]
    os.makedirs(os.path.join(folder, "qrels"), exist_ok=True)
    for repo, name, dest, revision in files:
        src = hf_fetch(repo, name, revision)
        opener = gzip.open if name.endswith(".gz") else open
        with opener(src, "rb") as fin, open(os.path.join(folder, dest), "wb") as fout:
            shutil.copyfileobj(fin, fout)


def download_ukp(dataset, raw_dir):
    """Original BEIR host; often slow, so it is only the fallback."""
    zip_path = os.path.join(raw_dir, dataset + ".zip")
    with urllib.request.urlopen(BEIR_URL.format(dataset), timeout=60) as r, \
            open(zip_path, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_dir)


def download(dataset, raw_dir):
    """Download the dataset once; later runs reuse the folder."""
    folder = os.path.join(raw_dir, dataset)
    if os.path.exists(os.path.join(folder, "qrels", "test.tsv")):
        return folder
    os.makedirs(raw_dir, exist_ok=True)
    print(f"downloading {dataset} from Hugging Face...")
    try:
        download_hf(dataset, folder)
    except Exception as err:
        print(f"Hugging Face failed ({err}); trying the BEIR server...")
        download_ukp(dataset, raw_dir)
    return folder


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(folder):
    docs = read_jsonl(os.path.join(folder, "corpus.jsonl"))
    ids = [d["_id"] for d in docs]
    texts = [(d.get("title", "") + " " + d.get("text", "")).strip() for d in docs]
    return ids, texts


def load_test_queries(folder):
    """Only queries listed in the test split, in queries.jsonl order."""
    test_ids = set()
    with open(os.path.join(folder, "qrels", "test.tsv"), encoding="utf-8") as f:
        next(f)  # header: query-id corpus-id score
        for line in f:
            if line.strip():
                test_ids.add(line.split("\t")[0])
    queries = read_jsonl(os.path.join(folder, "queries.jsonl"))
    kept = [q for q in queries if q["_id"] in test_ids]
    return [q["_id"] for q in kept], [q["text"] for q in kept]


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed(model, texts, batch_size):
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # length 1, so dot product = cosine similarity
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.ascontiguousarray(vecs, dtype=np.float32)  # Go reads float32


def embed_cached(model, texts, batch_size, cache_path):
    """Embedding the full corpus is the slow step, so keep a copy for reruns."""
    if os.path.exists(cache_path):
        vecs = np.load(cache_path)
        if vecs.shape[0] == len(texts):
            return vecs
    vecs = embed(model, texts, batch_size)
    np.save(cache_path, vecs)
    return vecs


def write_fbin(path, vecs):
    n, dim = vecs.shape
    with open(path, "wb") as f:
        np.array([n, dim], dtype="<i4").tofile(f)
        vecs.astype("<f4").tofile(f)


def write_jsonl(path, key, ids, texts):
    with open(path, "w", encoding="utf-8") as f:
        for i, (orig_id, text) in enumerate(zip(ids, texts)):
            f.write(json.dumps({"id": i, key: orig_id, "text": text}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="quora", help="BEIR dataset name")
    parser.add_argument("--out", default="data", help="output folder")
    parser.add_argument(
        "--sizes",
        default="small=50000,medium=200000,large=0",
        help="name=passage_count pairs; 0 = whole corpus",
    )
    parser.add_argument("--batch", type=int, default=256, help="embedding batch size")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed")
    args = parser.parse_args()

    raw_dir = os.path.join(args.out, "raw")
    folder = download(args.dataset, raw_dir)

    doc_ids, doc_texts = load_corpus(folder)
    query_ids, query_texts = load_test_queries(folder)
    print(f"{len(doc_texts)} passages, {len(query_texts)} test queries")

    # shuffle once so every size is a random sample, and smaller sizes are
    # prefixes of larger ones (embed the corpus once, slice it per size)
    order = np.random.default_rng(args.seed).permutation(len(doc_texts))
    doc_ids = [doc_ids[i] for i in order]
    doc_texts = [doc_texts[i] for i in order]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device=pick_device())
    cache = os.path.join(raw_dir, f"{args.dataset}_seed{args.seed}_passages.npy")
    doc_vecs = embed_cached(model, doc_texts, args.batch, cache)
    query_vecs = embed(model, query_texts, args.batch)

    for name, count in parse_sizes(args.sizes):
        n = len(doc_texts) if count <= 0 else min(count, len(doc_texts))
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        write_fbin(os.path.join(out_dir, "passages.fbin"), doc_vecs[:n])
        write_fbin(os.path.join(out_dir, "queries.fbin"), query_vecs)
        write_jsonl(os.path.join(out_dir, "passages.jsonl"), "doc_id",
                    doc_ids[:n], doc_texts[:n])
        write_jsonl(os.path.join(out_dir, "queries.jsonl"), "query_id",
                    query_ids, query_texts)
        print(f"{name}: {n} passages x {doc_vecs.shape[1]} dims -> {out_dir}")


if __name__ == "__main__":
    main()