#!/usr/bin/env python3
"""Fetch/build the labeled datasets used by the quality benchmark, cached as .npz.

Writes X (n, d) float32 and y (n,) int to /tmp. Each can then be fed to
benchmark_quality.py via --npz. Sources are chosen to avoid OpenML (which was returning
HTTP 504); everything comes from HuggingFace or scikit-learn.

    Fashion-MNIST   60k x 784, 10 classes   (zalando-datasets/fashion_mnist)
    DBpedia-14      ~49k x 384, 14 classes  (fancyzhx/dbpedia_14, embedded with BGE-small)
    Olivetti faces  400 x 4096, 40 classes  (sklearn; a small-n stress case)

Run: conda run --no-capture-output -n rapids-26.04 python examples/prepare_datasets.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np


def fashion_mnist(out="/tmp/fashion_mnist.npz"):
    from datasets import load_dataset
    fm = load_dataset("zalando-datasets/fashion_mnist", split="train")
    X = np.stack([np.asarray(im, dtype=np.float32).reshape(-1) for im in fm["image"]])
    y = np.asarray(fm["label"], dtype=np.int64)
    np.savez(out, X=X, y=y)
    print(f"{out}: {X.shape} classes={len(np.unique(y))}")


def dbpedia14(out="/tmp/dbpedia_emb.npz", per_class=3500, model="BAAI/bge-small-en-v1.5"):
    """A labeled vector store: DBpedia ontology entities embedded with BGE-small."""
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    db = load_dataset("fancyzhx/dbpedia_14", split="train")
    labels = np.asarray(db["label"], dtype=np.int64)
    rng = np.random.default_rng(0)
    keep = np.concatenate([rng.choice(np.where(labels == c)[0], per_class, replace=False)
                           for c in np.unique(labels)])
    rng.shuffle(keep)
    texts = [db["content"][i] for i in keep.tolist()]
    emb = SentenceTransformer(model, device="cuda").encode(
        texts, batch_size=512, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=True).astype(np.float32)
    np.savez(out, X=emb, y=labels[keep])
    print(f"{out}: {emb.shape} classes={len(np.unique(labels[keep]))}")


def olivetti(out="/tmp/olivetti.npz"):
    from sklearn.datasets import fetch_olivetti_faces
    f = fetch_olivetti_faces()
    np.savez(out, X=f.data.astype("float32"), y=f.target)
    print(f"{out}: {f.data.shape} classes={len(np.unique(f.target))}")


if __name__ == "__main__":
    fashion_mnist()
    dbpedia14()
    olivetti()
